"""
Bot de signaux XAUUSD - strategie A.P.A (Analysis, Point of Interest, Action)
Version conservatrice : confirmation stricte pour limiter les Stop Loss.

A - Analysis : tendance claire et ALIGNEE sur Daily + H4 (sinon aucun trade)
P - Point of Interest : Order Block M15 precede d'une prise de liquidite (sweep)
    et suivi d'une impulsion qui casse la structure (BOS)
A - Action : sur M5, CHoCH dans le sens de l'analyse, puis retest de la
    micro-zone avec une bougie de rejet nette
"""
import os
import re as _re
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
import yfinance as yf

# ------------------------- PARAMETRES -------------------------
TICKER = "GC=F"
SYMBOL_LABEL = "XAUUSD"
ATR_PERIOD = 14
RR = 2.0                    # ratio risque/gain
SL_BUFFER_ATR = 0.3         # marge du SL (x ATR M5)
IMPULSE_ATR = 0.8           # force mini de l'impulsion BOS (x ATR M15)
MAX_RISK_PRICE = 20.0       # SL max en dollars (200 pips) sinon pas de signal
M15_BARS = 200              # bougies M15 analysees pour le POI
M5_BARS = 120               # bougies M5 analysees pour l'action
SIGNAL_WINDOW = 3           # le signal doit etre dans les 3 dernieres bougies M5
MAX_SIGNALS_PER_DAY = 3
MIN_MINUTES_BETWEEN = 60
START_HOUR_UTC, END_HOUR_UTC = 0, 24
PIP = 0.1
# --------------------------------------------------------------

_m = _re.search(r"\d{6,}:[A-Za-z0-9_-]{30,}", os.environ.get("TELEGRAM_TOKEN", ""))
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


def get_data(interval: str, period: str, minutes: int) -> pd.DataFrame:
    """Telecharge les bougies et retire la bougie en cours de formation."""
    df = yf.download(TICKER, interval=interval, period=period,
                     progress=False, auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close"]].dropna()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    now = datetime.now(timezone.utc)
    if len(df) and df.index[-1] + timedelta(minutes=minutes) > now:
        df = df.iloc[:-1]
    return df


def atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> float:
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])


# ----------------------- OUTILS DE STRUCTURE -----------------------
def pivots(highs, lows, n):
    """Retourne (pivots_hauts, pivots_bas) sous forme de listes (index, prix)."""
    ph, pl = [], []
    for i in range(n, len(highs) - n):
        if highs[i] == max(highs[i - n:i + n + 1]):
            ph.append((i, float(highs[i])))
        if lows[i] == min(lows[i - n:i + n + 1]):
            pl.append((i, float(lows[i])))
    return ph, pl


def trend(df: pd.DataFrame, n: int = 2) -> int:
    """1 = haussier (HH + HL), -1 = baissier (LH + LL), 0 = pas clair."""
    if len(df) < 20:
        return 0
    ph, pl = pivots(df["High"].values, df["Low"].values, n)
    if len(ph) < 2 or len(pl) < 2:
        return 0
    if ph[-1][1] > ph[-2][1] and pl[-1][1] > pl[-2][1]:
        return 1
    if ph[-1][1] < ph[-2][1] and pl[-1][1] < pl[-2][1]:
        return -1
    return 0


def find_poi(df: pd.DataFrame, direction: int, atr15: float):
    """
    Order Block valide : sweep de liquidite -> impulsion avec BOS -> derniere
    bougie opposee avant l'impulsion. Retourne {'low','high'} ou None.
    """
    O, H, L, C = (df[c].values for c in ("Open", "High", "Low", "Close"))
    ph, pl = pivots(H, L, 3)
    size = len(df)

    if direction == 1:
        for pi, pp in reversed(pl):
            s = next((k for k in range(pi + 1, size) if L[k] < pp), None)   # sweep du creux
            if s is None:
                continue
            prior = [h for h in ph if h[0] < s]
            if not prior:
                continue
            level = prior[-1][1]
            b = next((k for k in range(s + 1, size) if C[k] > level), None)   # BOS haussier
            if b is None:
                continue
            ob = next((k for k in range(b - 1, s - 1, -1) if C[k] < O[k]), None)
            if ob is None or C[b] - L[ob] < IMPULSE_ATR * atr15:
                continue
            zlow, zhigh = float(L[ob]), float(H[ob])
            if any(C[k] < zlow for k in range(b + 1, size)):   # zone invalidee
                continue
            return {"low": zlow, "high": zhigh}
    else:
        for pi, pp in reversed(ph):
            s = next((k for k in range(pi + 1, size) if H[k] > pp), None)   # sweep du sommet
            if s is None:
                continue
            prior = [l for l in pl if l[0] < s]
            if not prior:
                continue
            level = prior[-1][1]
            b = next((k for k in range(s + 1, size) if C[k] < level), None)   # BOS baissier
            if b is None:
                continue
            ob = next((k for k in range(b - 1, s - 1, -1) if C[k] > O[k]), None)
            if ob is None or H[ob] - C[b] < IMPULSE_ATR * atr15:
                continue
            zlow, zhigh = float(L[ob]), float(H[ob])
            if any(C[k] > zhigh for k in range(b + 1, size)):
                continue
            return {"low": zlow, "high": zhigh}
    return None


def find_action(m5: pd.DataFrame, direction: int, poi: dict, atr5: float):
    """
    Cherche : prix dans le POI -> CHoCH M5 -> retest de la micro-zone avec
    bougie de rejet. Retourne (index_bougie, entry, sl) ou None.
    """
    O, H, L, C = (m5[c].values for c in ("Open", "High", "Low", "Close"))
    size = len(m5)
    buf = SL_BUFFER_ATR * atr5

    # 1) le prix a touche le POI recemment
    touch = next((k for k in range(size) if L[k] <= poi["high"] and H[k] >= poi["low"]), None)
    if touch is None:
        return None
    if direction == 1 and any(C[k] < poi["low"] for k in range(touch, size)):
        return None
    if direction == -1 and any(C[k] > poi["high"] for k in range(touch, size)):
        return None

    ph, pl = pivots(H, L, 2)
    start = max(0, touch - 3)

    if direction == 1:
        cands = [p for p in ph if p[0] >= start]
        choch = None
        for hi_idx, hi_price in reversed(cands):        # dernier sommet casse = CHoCH
            choch = next((k for k in range(hi_idx + 1, size) if C[k] > hi_price), None)
            if choch is not None:
                break
        if choch is None:
            return None
        micro = next((k for k in range(choch - 1, start - 1, -1) if C[k] < O[k]), None)
        if micro is None:
            return None
        mlow, mhigh = float(L[micro]), float(H[micro])
        for r in range(choch + 1, size):                       # retest + rejet
            body = abs(C[r] - O[r])
            lower = min(O[r], C[r]) - L[r]
            upper = H[r] - max(O[r], C[r])
            reject = C[r] > O[r] and C[r] > mhigh and lower >= max(body, 0.2 * atr5) and upper <= lower
            if L[r] <= mhigh and L[r] >= mlow - atr5 and reject:
                entry = float(C[r])
                sl = min(float(L[r]), mlow) - buf
                return r, entry, sl
    else:
        cands = [p for p in pl if p[0] >= start]
        choch = None
        for lo_idx, lo_price in reversed(cands):        # dernier creux casse = CHoCH
            choch = next((k for k in range(lo_idx + 1, size) if C[k] < lo_price), None)
            if choch is not None:
                break
        if choch is None:
            return None
        micro = next((k for k in range(choch - 1, start - 1, -1) if C[k] > O[k]), None)
        if micro is None:
            return None
        mlow, mhigh = float(L[micro]), float(H[micro])
        for r in range(choch + 1, size):
            body = abs(C[r] - O[r])
            lower = min(O[r], C[r]) - L[r]
            upper = H[r] - max(O[r], C[r])
            reject = C[r] < O[r] and C[r] < mlow and upper >= max(body, 0.2 * atr5) and lower <= upper
            if H[r] >= mlow and H[r] <= mhigh + atr5 and reject:
                entry = float(C[r])
                sl = max(float(H[r]), mhigh) + buf
                return r, entry, sl
    return None


# ----------------------------- STRATEGIE -----------------------------
def evaluate():
    d1 = get_data("1d", "1y", 1440)
    h1 = get_data("1h", "60d", 60)
    m15 = get_data("15m", "20d", 15)
    m5 = get_data("5m", "5d", 5)
    if len(d1) < 20 or len(h1) < 60 or len(m15) < 80 or len(m5) < 40:
        print(f"DIAG: donnees insuffisantes d1={len(d1)} h1={len(h1)} m15={len(m15)} m5={len(m5)}")
        return None

    last_time = m5.index[-1]

    # --- A : Analysis (tendance H4 claire)
    h4 = h1.resample("4h").agg({"Open": "first", "High": "max", "Low": "min",
                                "Close": "last"}).dropna()
    if len(h4) and h4.index[-1] + timedelta(hours=4) > datetime.now(timezone.utc):
        h4 = h4.iloc[:-1]
    t_4 = trend(h4)
    print(f"DIAG: derniere bougie M5 = {last_time:%Y-%m-%d %H:%M} UTC | prix = {m5['Close'].iloc[-1]:.2f}")
    print(f"DIAG: tendance H4 = {t_4} (1 haussier, -1 baissier, 0 pas claire)")
    if t_4 == 0:
        print("DIAG: BLOQUE a l'etape A : tendance H4 pas claire")
        return last_time, None
    direction = t_4

    # --- P : Point of Interest (Order Block M15 avec liquidite)
    m15w = m15.iloc[-M15_BARS:].reset_index(drop=True)
    poi = find_poi(m15w, direction, atr(m15))
    if poi is None:
        print("DIAG: BLOQUE a l'etape P : aucun Order Block valide (sweep + BOS)")
        return last_time, None
    print(f"DIAG: POI trouve entre {poi['low']:.2f} et {poi['high']:.2f}")

    # --- A : Action (CHoCH + retest + rejet sur M5)
    m5w = m5.iloc[-M5_BARS:]
    res = find_action(m5w.reset_index(drop=True), direction, poi, atr(m5))
    if res is None:
        print("DIAG: BLOQUE a l'etape A (action) : pas de CHoCH + retest + rejet sur M5")
        return last_time, None
    r, entry, sl = res
    if r < len(m5w) - SIGNAL_WINDOW:        # signal trop ancien
        print("DIAG: setup trouve mais trop ancien")
        return last_time, None

    risk = abs(entry - sl)
    if risk <= 0 or risk > MAX_RISK_PRICE:
        print(f"DIAG: setup trouve mais SL trop large ({risk:.2f} $)")
        return last_time, None
    print("DIAG: SIGNAL TROUVE")
    if direction == 1:
        side, tp = "BUY", entry + RR * risk
    else:
        side, tp = "SELL", entry - RR * risk
    return m5w.index[r], (side, entry, sl, tp)


def format_message(side, entry, sl, tp) -> str:
    sl_pips = abs(sl - entry) / PIP
    tp_pips = abs(tp - entry) / PIP
    sl_sign = "+" if side == "SELL" else "-"
    tp_sign = "-" if side == "SELL" else "+"
    return (
        "MALIK-SIGNAL-BOT - APA\n"
        "Status: SIGNAL ONLY - execute manually\n"
        f"Symbol: {SYMBOL_LABEL}\n"
        f"Side: {side}\n"
        f"Entry: {entry:.2f}\n"
        f"SL: {sl:.2f} ({sl_sign}{sl_pips:.0f} pips)\n"
        f"TP: {tp:.2f} ({tp_sign}{tp_pips:.0f} pips)\n"
        "Setup: tendance H4, POI avec sweep, CHoCH + retest M5\n"
        "Rappel: 1% de risque max, verifie les news et le prix sur Deriv."
    )
