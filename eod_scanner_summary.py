"""
eod_scanner_summary.py — read-only EOD summary for the stock-option scanners.

Prints, per strategy (RT, FLOW, V1, MORNING_BO):
  - today: trades, net P&L, win rate, avg win / avg loss, exit-reason mix,
    option expiry month traded
  - since GO_LIVE (2026-09-29, the entry-gate + spot-exit config): day-by-day
    net, cumulative net, positive days — compared against the backtest
    expectations in EXPECT
  - RT / FLOW entry-gate activity from the signals CSV and the core log

Safe to run any number of times: it only reads files.

    python3 eod_scanner_summary.py [YYYY-MM-DD]
"""

import csv
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import date

ROOT    = os.path.dirname(os.path.abspath(__file__))
GO_LIVE = "2026-09-29"

STRATS = {
    "RT":         "stock_opt_scanner_rt_trades.csv",
    "FLOW":       "stock_opt_scanner_flow_trades.csv",
    "V1":         "stock_opt_scanner_trades.csv",
    "MORNING_BO": "stock_opt_morning_bo_trades.csv",
}

# From the 2026-09-28 re-pricing of the live paper trades on real option
# candles (see core/entry_gate.py). Rough guide-posts, not promises.
EXPECT = {
    "RT":   "~37 trades/day, win ~47%, net positive on ~60% of days, avg ~+Rs180/trade",
    "FLOW": "~9 trades/day, win ~55%, net positive on ~65% of days, avg ~+Rs750/trade (fragile: 6-day sample)",
}


def load_exits(fname):
    path = os.path.join(ROOT, fname)
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("action") == "EXIT"]
    for r in rows:
        r["date"] = r["timestamp"][:10]
        try:
            r["pnl"] = float(r["pnl"])
        except (TypeError, ValueError):
            r["pnl"] = 0.0
    return rows


def expiry_month(symbol):
    m = re.search(r"\d{2}([A-Z]{3})\d", symbol or "")
    return m.group(1) if m else "?"


def today_block(rows, day):
    t = [r for r in rows if r["date"] == day]
    if not t:
        return "  no completed trades today", t
    pnl  = [r["pnl"] for r in t]
    wins = [p for p in pnl if p > 0]
    loss = [p for p in pnl if p <= 0]
    mix  = Counter(r["reason"] for r in t)
    mon  = Counter(expiry_month(r["symbol"]) for r in t)
    lines = [
        f"  trades={len(t)} net=Rs{sum(pnl):,.0f} win={len(wins) / len(t):.0%} "
        f"avg_win=Rs{(sum(wins) / len(wins)) if wins else 0:,.0f} "
        f"avg_loss=Rs{(sum(loss) / len(loss)) if loss else 0:,.0f}",
        f"  exits: {dict(mix.most_common())}",
        f"  contract month: {dict(mon)}",
    ]
    return "\n".join(lines), t


def since_live_block(rows):
    by_day = defaultdict(float)
    n_day  = Counter()
    for r in rows:
        if r["date"] >= GO_LIVE:
            by_day[r["date"]] += r["pnl"]
            n_day[r["date"]]  += 1
    if not by_day:
        return "  since go-live: no trades yet"
    days  = sorted(by_day)
    cum   = 0.0
    parts = []
    for d in days:
        cum += by_day[d]
        parts.append(f"{d[5:]}:{by_day[d]:+,.0f}({n_day[d]})")
    pos = sum(1 for d in days if by_day[d] > 0)
    n   = sum(n_day.values())
    return (f"  since {GO_LIVE}: {len(days)} days, {n} trades, cum=Rs{cum:,.0f}, "
            f"avg/trade=Rs{cum / n:,.0f}, positive days {pos}/{len(days)}\n"
            f"  by day: {' '.join(parts)}")


def gate_activity(day):
    out = []
    sig = os.path.join(ROOT, "stock_opt_scanner_rt_signals.csv")
    if os.path.isfile(sig):
        with open(sig, newline="") as f:
            ev = Counter(r["event"] for r in csv.DictReader(f) if r["timestamp"].startswith(day))
        blocks = Counter()
        with open(sig, newline="") as f:
            for r in csv.DictReader(f):
                if r["timestamp"].startswith(day) and r["event"] == "ENTRY_GATE_BLOCK":
                    blocks[(r.get("block_reason") or "").split("=")[0].split(" ")[0]] += 1
        out.append(f"  RT signals: triggers={ev.get('SURGE_TRIGGER', 0)} "
                   f"gate_blocked={ev.get('ENTRY_GATE_BLOCK', 0)} {dict(blocks)} "
                   f"confirm_timeout={ev.get('CONFIRM_TIMEOUT', 0)} entry_block={ev.get('ENTRY_BLOCK', 0)}")
    core = os.path.join(ROOT, "logs", day, f"core_{day}.log")
    if os.path.isfile(core):
        passed, noatr, ready = Counter(), Counter(), []
        with open(core, errors="replace") as f:
            for line in f:
                m = re.search(r"\[(STOCK_OPT_SCANNER_(?:RT|FLOW))\]", line)
                if not m:
                    continue
                s = m.group(1).replace("STOCK_OPT_SCANNER_", "")
                if "entry gate PASSED" in line:
                    passed[s] += 1
                if "no_atr5" in line:
                    noatr[s] += 1
                if "] ready |" in line:
                    ready.append(line.strip()[-90:])
        out.append(f"  core log: gate passed {dict(passed)}  blocked for missing ATR {dict(noatr)}")
        for r in ready:
            out.append(f"  startup: ...{r}")
    else:
        out.append(f"  core log not found: {core}")
    return "\n".join(out)


def main():
    day = sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()
    print(f"=== Stock-option scanners — {day} ===")
    any_today = False
    for name, fname in STRATS.items():
        rows = load_exits(fname)
        block, t = today_block(rows, day)
        any_today |= bool(t)
        print(f"\n[{name}]")
        print(block)
        if name in EXPECT:
            print(since_live_block(rows))
            print(f"  backtest expectation: {EXPECT[name]}")
    print("\n[entry gate activity]")
    print(gate_activity(day))
    if not any_today:
        print("\nNO_TRADES_TODAY")


if __name__ == "__main__":
    main()
