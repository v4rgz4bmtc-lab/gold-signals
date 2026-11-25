import os
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
import yfinance as yf

TICKER = "GC=F"
SYMBOL_LABEL = "XAUUSD"
EMA_FAST, EMA_SLOW = 50, 200
ZONE_LOOKBACK = 24
ZONE_TOL_ATR = 1.0
ATR_PERIOD = 14
SL_BUFFER_ATR = 0.3
RR = 2.0
MAX_SIGNALS_PER_DAY = 2
MIN_MINUTES_BETWEEN = 60
START_HOUR_UTC, END_HOUR_UTC = 0, 24
PIP = 0.1

import re as _re
_m = _re.search(r"\d{6,}:[A-Za-z0-9_-]{30,}", os.environ.get("TELEGRAM_TOKEN", ""))
TOKEN = _m.group(0) if _m else os.environ.get("TELEGRAM_TOKEN", "").strip()
print("Token:", "format OK, longueur", len(TOKEN)) if _m else print("Token: format INVALIDE")
("TELEGRAM_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


def send_telegram(text: str) -> None:
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        r = requests.post(url, data={"chat_id": CHAT_ID, "text": text}, timeout=15)
        if r.status_code != 200:
            print("Erreur Telegram:", r.text)
    except Exception as e:
        print("Erreur reseau Telegram:", e)


def get_data(interval: str, period: str, minutes: int) -> pd.DataFrame:
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


def evaluate():
        h1 = get_data("1h", "60d", 60)
    m15 = h1
    m5 = get_data("30m", "10d", 30)
    h4 = h1.resample("4h").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
    h1 = h4

    if len(h1) < EMA_SLOW + 2 or len(m15) < ZONE_LOOKBACK + ATR_PERIOD or len(m5) < ATR_PERIOD + 3:
        return None

    candle_time = m5.index[-1]
    c, p = m5.iloc[-1], m5.iloc[-2]

    ema_f = h1["Close"].ewm(span=EMA_FAST, adjust=False).mean().iloc[-1]
    ema_s = h1["Close"].ewm(span=EMA_SLOW, adjust=False).mean().iloc[-1]
    hc = h1["Close"].iloc[-1]
    up = ema_f > ema_s and hc > ema_f
    dn = ema_f < ema_s and hc < ema_f
    if not (up or dn):
        return candle_time, None

    zone = m15.iloc[-ZONE_LOOKBACK:]
    support = float(zone["Low"].min())
    resistance = float(zone["High"].max())
    tol = ZONE_TOL_ATR * atr(m15)
    buf = SL_BUFFER_ATR * atr(m5)

    body = abs(c["Close"] - c["Open"])
    upper = c["High"] - max(c["Open"], c["Close"])
    lower = min(c["Open"], c["Close"]) - c["Low"]

    if up:
        in_zone = support - tol <= c["Low"] <= support + tol
        bull = c["Close"] > c["Open"]
        engulf = bull and p["Open"] > p["Close"] and c["Close"] > p["Open"] and c["Open"] <= p["Close"]
        pin = bull and body > 0 and lower >= 2 * body and upper <= body
        if in_zone and (engulf or pin):
            entry = float(c["Close"])
            sl = min(float(c["Low"]), support) - buf
            risk = entry - sl
            if risk > 0:
                return candle_time, ("BUY", entry, sl, entry + RR * risk)

    if dn:
        in_zone = resistance - tol <= c["High"] <= resistance + tol
        bear = c["Close"] < c["Open"]
        engulf = bear and p["Open"] < p["Close"] and c["Close"] < p["Open"] and c["Open"] >= p["Close"]
        pin = bear and body > 0 and upper >= 2 * body and lower <= body
        if in_zone and (engulf or pin):
            entry = float(c["Close"])
            sl = max(float(c["High"]), resistance) + buf
            risk = sl - entry
            if risk > 0:
                return candle_time, ("SELL", entry, sl, entry - RR * risk)

    return candle_time, None


def format_message(side, entry, sl, tp) -> str:
    sl_pips = abs(sl - entry) / PIP
    tp_pips = abs(tp - entry) / PIP
    sl_sign = "+" if side == "SELL" else "-"
    tp_sign = "-" if side == "SELL" else "+"
    return (
        "DMKJ_TRADING_AI_M30\n"
        "Status: SIGNAL ONLY - execute manually\n"
        f"Symbol: {SYMBOL_LABEL}\n"
        f"Side: {side}\n"
        f"Entry: {entry:.2f}\n"
        f"SL: {sl:.2f} ({sl_sign}{sl_pips:.0f} pips)\n"
        f"TP: {tp:.2f} ({tp_sign}{tp_pips:.0f} pips)\n"
        "Rappel: 1% de risque max, verifie les news avant d'entrer."
    )
