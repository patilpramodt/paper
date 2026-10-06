"""
tools/build_candles.py — turn a day of recorded ticks into 1-minute candles.

    python tools/build_candles.py              # today
    python tools/build_candles.py 2026-10-07   # a given day
    python tools/build_candles.py --all        # every recorded day missing candles

Writes data/candles/1m/<day>/<group>.csv.gz
(ts, symbol, token, OHLC, volume, vwap, oi_open/close/chg, bid, ask, spread,
 buy_qty, sell_qty, imbalance, ticks). Any other size: tools.tickdata.candles().
Run from cron after the close; the trader keeps running fine without it.
"""

import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.tickdata import CANDLES, TICKS, days, resample, ticks  # noqa: E402

GROUPS = ("index", "fut", "nifty_opt", "banknifty_opt", "stock", "stock_opt", "other")


def build(day):
    out_dir = os.path.join(CANDLES, "1m", day)
    os.makedirs(out_dir, exist_ok=True)
    for g in GROUPS:
        if not os.path.isfile(os.path.join(TICKS, day, f"ticks_{g}.csv.gz")):
            continue
        cols = ["recv_ts", "exch_ts", "token", "symbol", "ltp", "volume", "oi",
                "bid1", "ask1", "buy_qty", "sell_qty"]
        c = resample(ticks(day, g, columns=cols), "1min")
        c.to_csv(os.path.join(out_dir, f"{g}.csv.gz"), index=False)
        print(f"{day} {g}: {len(c):,} candles")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg == "--all":
        todo = [d for d in days() if not os.path.isdir(os.path.join(CANDLES, "1m", d))]
    else:
        todo = [arg or datetime.now(timezone(timedelta(hours=5, minutes=30))).date().isoformat()]
    for d in todo:
        if os.path.isdir(os.path.join(TICKS, d)):
            build(d)
        else:
            print(f"{d}: no tick data")
