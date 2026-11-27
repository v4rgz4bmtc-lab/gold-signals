"""
Bot XAUUSD M15 - SIGNAL ONLY (execution manuelle)
- Signal BUY/SELL (avec SL et TP) quand une entree algorithmique apparait sur M15 :
  tendance H1 + tendance M15 + repli sur l'EMA20 + bougie de reprise + RSI.
- Message d'analyse toutes les 30 minutes (prix, tendance, niveaux).
- Alerte si les prix Yahoo ont trop de retard (aucun signal dans ce cas).

    python run_once.py          -> une verification (GitHub Actions)
    python run_once.py test     -> message de test
"""
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
import yfinance as yf

# ------------------------- PARAMETRES -------------------------
TICKER = "GC=F"
SYMBOL = "XAUUSD"
RR = 1.5                    # ratio TP / SL
ATR_PERIOD = 14
RSI_PERIOD = 14
SL_BUFFER_ATR = 0.3         # marge sous le plus bas / au-dessus du plus haut
MIN_RISK_ATR = 0.8          # SL minimum (x ATR M15)
MAX_RISK_ATR = 2.5          # SL maximum (x ATR M15), sinon pas de signal
MAX_SIGNALS_PER_DAY = 10
MIN_MINUTES_BETWEEN = 30
STALE_MINUTES = 60          # prix trop anciens -> pas de signal
START_HOUR_UTC, END_HOUR_UTC = 6, 21     # plage active (UTC). Cameroun = UTC+1
ANALYSIS_EVERY_MIN = 30
PIP = 0.1
# Position du jour (heure fixe, sans attendre de confirmation)
DAILY_HOURS_UTC = [7, 12]   # 7h et 12h UTC = 8h et 13h au Cameroun
DAILY_SL_ATR = 1.5          # SL = 1,5 x ATR M15
DAILY_RR = 2.0              # TP = 2 x SL
DAILY_WINDOW_H = 3          # envoyee si le bot passe dans les 3h suivant l'heure
STATE_FILE = "state.json"
# --------------------------------------------------------------

_m = re.search(r"\d{6,}:[A-Za-z0-9_-]{30,}", os.environ.get("TELEGRAM_TOKEN", ""))
TOKEN = _m.group(0) if _m else os.environ.get("TELEGRAM_TOKEN", "").strip()
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()


def send_telegram(text: str) -> None:
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        r = requests.post(url, data={"chat_id": CHAT_ID, "text": text}, timeout=15)
        if r.status_code != 200:
            print("Erreur Telegram:", r.text)
    except Exception as e:
        print("Erreur reseau Telegram:", e)


def _yahoo(interval: str, period: str, minutes: int) -> pd.DataFrame:
    """Secours : Yahoo (or en futures). Bougies cloturees uniquement."""
    df = yf.download(TICKER, interval=interval, period=period,
                     progress=False, auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close"]].dropna()
    df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
    if len(df) and df.index[-1] + timedelta(minutes=minutes) > datetime.now(timezone.utc):
        df = df.iloc[:-1]
    return df


DERIV_URL = "wss://ws.derivws.com/websockets/v3?app_id=1089"
DERIV_SYMBOL = "frxXAUUSD"
GRANULARITY = {"15m": 900, "1h": 3600}
SOURCE = "Deriv"


def _deriv(granularity: int, count: int) -> pd.DataFrame:
    """Bougies de l'or directement depuis Deriv (memes prix que ton courtier)."""
    try:
        import websocket
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "websocket-client"])
        import websocket
    ws = websocket.create_connection(DERIV_URL, timeout=15)
    try:
        ws.send(json.dumps({"ticks_history": DERIV_SYMBOL, "adjust_start_time": 1, "count": count,
                            "end": "latest", "granularity": granularity, "style": "candles"}))
        data = json.loads(ws.recv())
    finally:
        ws.close()
    if "error" in data:
        raise RuntimeError(data["error"].get("message", "erreur Deriv"))
    df = pd.DataFrame(data.get("candles", []))
    if df.empty:
        return df
    df.index = pd.to_datetime(df["epoch"], unit="s", utc=True)
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
    df = df[["Open", "High", "Low", "Close"]].astype(float)
    if df.index[-1] + timedelta(seconds=granularity) > datetime.now(timezone.utc):
        df = df.iloc[:-1]
    return df


def get_data(interval: str, period: str, minutes: int) -> pd.DataFrame:
    """Deriv d'abord ; Yahoo en secours si Deriv ne repond pas."""
    global SOURCE
    try:
        df = _deriv(GRANULARITY[interval], 500 if interval == "15m" else 300)
        if len(df) >= 60:
            SOURCE = "Deriv"
            return df
        print(f"DIAG: Deriv a renvoye trop peu de bougies ({len(df)})")
    except Exception as e:
        print("DIAG: Deriv indisponible :", e)
    SOURCE = "Yahoo"
    return _yahoo(interval, period, minutes)


# ----------------------------- INDICATEURS -----------------------------
def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def rsi(s: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = up / dn.replace(0, 1e-9)
    return 100 - 100 / (1 + rs)


def atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    pc = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def context(m15: pd.DataFrame, h1: pd.DataFrame) -> dict:
    """Calcule tous les indicateurs sur la derniere bougie M15 cloturee."""
    e20, e50 = ema(m15["Close"], 20), ema(m15["Close"], 50)
    h1_ema50 = ema(h1["Close"], 50)
    return {
        "price": float(m15["Close"].iloc[-1]),
        "e20": float(e20.iloc[-1]), "e50": float(e50.iloc[-1]),
        "rsi": float(rsi(m15["Close"]).iloc[-1]),
        "atr": float(atr(m15).iloc[-1]),
        "h1_up": bool(h1["Close"].iloc[-1] > h1_ema50.iloc[-1]),
        "m15_up": bool(e20.iloc[-1] > e50.iloc[-1]),
        "support": float(m15["Low"].iloc[-20:].min()),
        "resistance": float(m15["High"].iloc[-20:].max()),
    }


# ----------------------------- STRATEGIE -----------------------------
def compute_signal(m15: pd.DataFrame, h1: pd.DataFrame):
    """Retourne (side, entry, sl, tp) ou (None, raison)."""
    if len(m15) < 60 or len(h1) < 60:
        return None, "donnees insuffisantes"
    c = context(m15, h1)
    e20 = ema(m15["Close"], 20)
    O, H, L, C = (m15[k].values for k in ("Open", "High", "Low", "Close"))
    a = c["atr"]
    entry = float(C[-1])

    up = c["h1_up"] and c["m15_up"]
    dn = (not c["h1_up"]) and (not c["m15_up"])
    if not (up or dn):
        return None, "H1 et M15 ne vont pas dans le meme sens"

    if up:
        touched = min(L[-1], L[-2]) <= e20.iloc[-1] + 0.2 * a
        reclaim = C[-1] > e20.iloc[-1] and C[-1] > O[-1]
        if not touched:
            return None, "tendance haussiere, mais pas de repli sur l'EMA20"
        if not reclaim:
            return None, "repli present, mais pas de bougie de reprise"
        if not (40 <= c["rsi"] <= 70):
            return None, f"RSI {c['rsi']:.0f} hors zone 40-70"
        sl = min(L[-1], L[-2]) - SL_BUFFER_ATR * a
        risk = entry - sl
        side = "BUY"
    else:
        touched = max(H[-1], H[-2]) >= e20.iloc[-1] - 0.2 * a
        reclaim = C[-1] < e20.iloc[-1] and C[-1] < O[-1]
        if not touched:
            return None, "tendance baissiere, mais pas de repli sur l'EMA20"
        if not reclaim:
            return None, "repli present, mais pas de bougie de reprise"
        if not (30 <= c["rsi"] <= 60):
            return None, f"RSI {c['rsi']:.0f} hors zone 30-60"
        sl = max(H[-1], H[-2]) + SL_BUFFER_ATR * a
        risk = sl - entry
        side = "SELL"

    if risk < MIN_RISK_ATR * a:
        risk = MIN_RISK_ATR * a
        sl = entry - risk if side == "BUY" else entry + risk
    if risk > MAX_RISK_ATR * a:
        return None, f"SL trop large ({risk:.1f} $)"
    tp = entry + RR * risk if side == "BUY" else entry - RR * risk
    return (side, entry, sl, tp), "ok"


def format_signal(side, entry, sl, tp) -> str:
    sl_d, tp_d = abs(sl - entry), abs(tp - entry)
    sl_pips, tp_pips = sl_d / PIP, tp_d / PIP
    tol_pips = max(10, round(0.2 * sl_pips))
    sl_txt = "sous" if side == "BUY" else "au-dessus de"
    tp_txt = "au-dessus de" if side == "BUY" else "sous"
    return (
        "MALIK-SIGNAL-BOT - M15\n"
        "SIGNAL ONLY - execute manually\n"
        f"Symbol: {SYMBOL}\n"
        f"Action: {side} au marche\n"
        f"SL: {sl_pips:.0f} pips {sl_txt} ton entree ({sl_d:.2f} $)\n"
        f"TP: {tp_pips:.0f} pips {tp_txt} ton entree ({tp_d:.2f} $)\n"
        f"Prix repere: {entry:.2f} ({SOURCE}). Entre seulement si ton prix Deriv est a "
        f"moins de {tol_pips} pips ({tol_pips * PIP:.1f} $) de ce repere, sinon passe.\n"
        "Rappel: 1% de risque max."
    )


def format_analysis(c: dict, candle_time: datetime, reason: str) -> str:
    h1 = "haussiere" if c["h1_up"] else "baissiere"
    m15 = "haussiere" if c["m15_up"] else "baissiere"
    if c["h1_up"] and c["m15_up"]:
        plan = f"Biais BUY : j'attends un repli vers l'EMA20 ({c['e20']:.2f})."
    elif (not c["h1_up"]) and (not c["m15_up"]):
        plan = f"Biais SELL : j'attends un rebond vers l'EMA20 ({c['e20']:.2f})."
    else:
        plan = "Pas de biais clair (H1 et M15 en desaccord) : j'attends."
    return (
        f"ANALYSE {SYMBOL} M15 ({candle_time:%H:%M} UTC)\n"
        f"Prix: {c['price']:.2f}\n"
        f"Tendance H1: {h1} | M15: {m15}\n"
        f"RSI: {c['rsi']:.0f} | ATR: {c['atr']:.1f}\n"
        f"Support: {c['support']:.2f} | Resistance: {c['resistance']:.2f}\n"
        f"{plan}\n"
        f"Signal: aucun ({reason})"
    )


# ----------------------------- ETAT -----------------------------
def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_state(st: dict) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(st, f)


# ----------------------------- PRINCIPAL -----------------------------
def main(now: datetime = None) -> None:
    if not TOKEN or not CHAT_ID:
        print("TELEGRAM_TOKEN / TELEGRAM_CHAT_ID manquants.")
        sys.exit(1)

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        send_telegram("TEST - MALIK-SIGNAL-BOT M15\nLe bot est bien connecte a Telegram.")
        print("Message de test envoye.")
        return

    now = now or datetime.now(timezone.utc)
    today = now.date().isoformat()
    st = load_state()
    in_hours = START_HOUR_UTC <= now.hour < END_HOUR_UTC

    m15 = get_data("15m", "10d", 15)
    h1 = get_data("1h", "30d", 60)
    if len(m15) < 60 or len(h1) < 60:
        print(f"DIAG: donnees insuffisantes m15={len(m15)} h1={len(h1)}")
        return

    candle_time = m15.index[-1].to_pydatetime()
    age = (now - (candle_time + timedelta(minutes=15))).total_seconds() / 60
    print(f"DIAG: derniere bougie M15 = {candle_time:%Y-%m-%d %H:%M} UTC | age = {age:.0f} min | prix = {m15['Close'].iloc[-1]:.2f} | source = {SOURCE}")

    # Prix trop anciens : on previent et on ne trade pas
    if age > STALE_MINUTES:
        print("DIAG: prix trop anciens, aucun signal")
        if in_hours and st.get("stale_day") != today:
            send_telegram(f"ATTENTION : les prix sont en retard de {age/60:.1f} h (source {SOURCE}). "
                          "Aucun signal tant que ce n'est pas revenu a la normale.")
            st["stale_day"] = today
            save_state(st)
        return

    result, reason = compute_signal(m15, h1)
    print(f"DIAG: {'SIGNAL ' + result[0] if result else 'pas de signal : ' + reason}")

    # --- Signal
    candle_iso = candle_time.isoformat()
    if st.get("last_candle") != candle_iso:
        st["last_candle"] = candle_iso
        if result and in_hours:
            if st.get("day") != today:
                st["day"], st["count"] = today, 0
            last = st.get("last_signal_time")
            recent = last and now - datetime.fromisoformat(last) < timedelta(minutes=MIN_MINUTES_BETWEEN)
            if st["count"] < MAX_SIGNALS_PER_DAY and not recent:
                send_telegram(format_signal(*result))
                st["count"] += 1
                st["last_signal_time"] = now.isoformat()
                print("DIAG: signal envoye")
        save_state(st)

    # --- Position du jour (heure fixe, sans confirmation)
    for h in DAILY_HOURS_UTC:
        key = f"{today}-{h}"
        start = now.replace(hour=h, minute=0, second=0, microsecond=0)
        if start <= now < start + timedelta(hours=DAILY_WINDOW_H) and key not in st.get("daily_sent", []):
            c = context(m15, h1)
            side = "BUY" if c["h1_up"] else "SELL"
            risk = DAILY_SL_ATR * c["atr"]
            entry = c["price"]
            sl = entry - risk if side == "BUY" else entry + risk
            tp = entry + DAILY_RR * risk if side == "BUY" else entry - DAILY_RR * risk
            send_telegram(format_signal(side, entry, sl, tp).replace(
                "MALIK-SIGNAL-BOT - M15", "POSITION DU JOUR - M15 (heure fixe)"))
            st["daily_sent"] = [k for k in st.get("daily_sent", []) if k.startswith(today)] + [key]
            save_state(st)
            print("DIAG: position du jour envoyee")

    # --- Analyse toutes les 30 minutes
    slot = now.replace(minute=(now.minute // ANALYSIS_EVERY_MIN) * ANALYSIS_EVERY_MIN, second=0, microsecond=0).isoformat()
    if in_hours and st.get("last_analysis") != slot:
        send_telegram(format_analysis(context(m15, h1), candle_time, reason if not result else "signal envoye"))
        st["last_analysis"] = slot
        save_state(st)


if __name__ == "__main__":
    main()
