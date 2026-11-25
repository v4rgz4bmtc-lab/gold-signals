import json
import os
import sys
from datetime import datetime, timedelta, timezone

import bot_signaux_xauusd as b

STATE_FILE = "state.json"


def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"last_candle": None, "last_signal_time": None, "day": None, "count": 0}


def save_state(st: dict) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(st, f)


def main() -> None:
    if not b.TOKEN or not b.CHAT_ID:
        print("TELEGRAM_TOKEN / TELEGRAM_CHAT_ID manquants.")
        sys.exit(1)

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        b.send_telegram(b.format_message("SELL", 3565.30, 3580.30, 3535.30)
                        .replace("MALIK-SIGNAL-BOT", "TEST - MALIK-SIGNAL-BOT"))
        print("Message de test envoye.")
        return

    st = load_state()
    result = b.evaluate()
    if not result:
        return
    candle_time, signal = result
    candle_iso = candle_time.isoformat()
    if st.get("last_candle") == candle_iso:
        return
    st["last_candle"] = candle_iso
    try:
        if signal is None:
            return
        now = datetime.now(timezone.utc)
        if now - candle_time > timedelta(minutes=45):
            return
        if not (b.START_HOUR_UTC <= now.hour < b.END_HOUR_UTC):
            return
        today = now.date().isoformat()
        if st.get("day") != today:
            st["day"], st["count"] = today, 0
        if st["count"] >= b.MAX_SIGNALS_PER_DAY:
            return
        last = st.get("last_signal_time")
        if last and now - datetime.fromisoformat(last) < timedelta(minutes=b.MIN_MINUTES_BETWEEN):
            return
        side, entry, sl, tp = signal
        b.send_telegram(b.format_message(side, entry, sl, tp))
        st["count"] += 1
        st["last_signal_time"] = now.isoformat()
        print(f"Signal {side} envoye")
    finally:
        save_state(st)


if __name__ == "__main__":
    main()
