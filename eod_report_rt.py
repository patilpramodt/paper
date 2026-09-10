#!/usr/bin/env python3
"""
eod_report_rt.py — STOCK_OPT_SCANNER_RT end-of-day report.

Outputs:
  - stdout: per-trade detail (full diagnostics for SL_HIT only) +
    SL_HIT premature-vs-genuine summary + indicator effectiveness.
  - logs/<date>/eod_diagnostics_<date>.csv: one row per trade, every field.
  - logs/indicator_effectiveness_log.csv: APPENDS one row per indicator
    per day, in the stable logs/ folder (NOT the dated subfolder) —
    this is the cross-day record of whether an indicator's win-rate
    split is real or one-day noise, so it must persist across runs.

Indicator effectiveness methodology: for each indicator, split today's
trades into "cut" (would be rejected by the rule) vs "kept", then
compare win rates. RSI and Bollinger %B use fixed, conventional
overbought/oversold thresholds. EMA gap has no such convention, so
it's calibrated to today's own quartile spread instead — see the
"rule" text per indicator in the output.

Kite's historical API caps at 1-minute bars — this reflects 1-min
underlying price action, not the tick-level view the live strategy
actually used. Screening signal, not a live replay.

Run from repo root (~/paper). Reuses token.json — does not log in.
"""
import csv
import json
import os
import time as time_mod
from collections import defaultdict
from datetime import datetime, date, time as dtime

import pandas as pd

from core.costs import fixed_costs_rs
from core.fast_indicators import compute_fast_indicators
from strategies.bb_stoch_strategy import compute_stochastic, compute_bb
from strategies.stock_options_scanner_realtime_strategy import CFG as RT_CFG

CSV_FILE = "stock_opt_scanner_rt_trades.csv"
SIGNALS_CSV = "stock_opt_scanner_rt_signals.csv"
TOKEN_FILE = "token.json"
LOGS_DIR = "logs"
STUDY_LOG_CSV = os.path.join(LOGS_DIR, "indicator_effectiveness_log.csv")
STOP_REASONS = {"SL_HIT"}
WIN_REASONS = {"TARGET", "TRAIL_HIT", "MAX_TARGET"}
LOSS_REASONS = {"SL_HIT"}
MARKET_OPEN = dtime(9, 15)
BB_PERIOD, BB_STD = 20, 2.0


def load_today_trades(csv_file=CSV_FILE, target_date=None):
    target_date = target_date or date.today()
    by_order = defaultdict(dict)
    try:
        f = open(csv_file, newline="")
    except FileNotFoundError:
        return []
    with f:
        for row in csv.DictReader(f):
            ts = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
            if ts.date() != target_date:
                continue
            by_order[row["order_id"]][row["action"].lower()] = row

    trades = []
    for oid, legs in by_order.items():
        entry, exit_ = legs.get("entry"), legs.get("exit")
        if not entry or not exit_:
            continue
        trades.append({
            "symbol": entry["symbol"], "stock": entry["stock"],
            "entry_time": datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S"),
            "entry_price": float(entry["price"]),
            "exit_time": datetime.strptime(exit_["timestamp"], "%Y-%m-%d %H:%M:%S"),
            "exit_price": float(exit_["price"]), "reason": exit_["reason"],
            "pnl": float(exit_["pnl"]), "sl": float(exit_["sl"]),
            "qty": int(entry["qty"]), "order_id": oid,
        })
    trades.sort(key=lambda t: t["entry_time"])
    return trades


def load_kite():
    with open(TOKEN_FILE) as f:
        data = json.load(f)
    from kiteconnect import KiteConnect
    kite = KiteConnect(api_key=data["api_key"])
    kite.set_access_token(data["access_token"])
    return kite


def build_token_map(kite, exchange):
    return {r["tradingsymbol"]: r["instrument_token"] for r in kite.instruments(exchange)}


def check_counterfactual(kite, nfo_map, trade):
    token = nfo_map.get(trade["symbol"])
    if token is None:
        return {"checked": False, "note": "instrument_token not found"}
    close_time = RT_CFG["close_time"]
    to_dt = datetime.combine(trade["exit_time"].date(), close_time)
    if to_dt <= trade["exit_time"]:
        return {"checked": False, "note": "exit was at/after session close"}
    try:
        candles = kite.historical_data(token, trade["exit_time"], to_dt, "minute")
    except Exception as e:
        return {"checked": False, "note": f"historical_data failed: {e}"}
    if not candles:
        return {"checked": False, "note": "no post-exit candles"}

    qty, entry = trade["qty"], trade["entry_price"]
    target_rs = RT_CFG["target_rs_min"]
    lowest_low = candles[0]["low"]
    lowest_low_time = candles[0]["date"]
    tp_hit_time = tp_hit_price = None

    for c in candles:
        if c["low"] < lowest_low:
            lowest_low = c["low"]
            lowest_low_time = c["date"]
        unreal_at_high = (c["high"] - entry) * qty - fixed_costs_rs(qty, entry, c["high"])
        if unreal_at_high >= target_rs:
            tp_hit_time, tp_hit_price = c["date"], c["high"]
            break

    worst_unreal = (lowest_low - entry) * qty - fixed_costs_rs(qty, entry, lowest_low)
    result = {
        "checked": True, "would_have_hit_target": tp_hit_time is not None,
        "target_rs": target_rs, "lowest_point": round(lowest_low, 2),
        "lowest_point_time": lowest_low_time, "worst_unreal_rs": round(worst_unreal, 0),
    }
    if tp_hit_time is not None:
        result["tp_hit_time"] = tp_hit_time
        result["tp_hit_price"] = round(tp_hit_price, 2)
    return result


def load_entry_signal(stock, entry_time, signals_csv=SIGNALS_CSV):
    try:
        f = open(signals_csv, newline="")
    except FileNotFoundError:
        return None
    best = None
    with f:
        for row in csv.DictReader(f):
            if row["stock"] != stock or row["event"] != "SURGE_TRIGGER":
                continue
            ts = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
            if ts > entry_time:
                continue
            if best is None or ts > best["_ts"]:
                row["_ts"] = ts
                best = row
    return best


def fetch_underlying_candles(kite, nse_map, stock, day, upto_dt):
    token = nse_map.get(stock)
    if token is None:
        return None
    from_dt = datetime.combine(day, MARKET_OPEN)
    if upto_dt <= from_dt:
        return None
    try:
        candles = kite.historical_data(token, from_dt, upto_dt, "minute")
    except Exception:
        return None
    return candles or None


def entry_diagnostics(candles):
    if not candles or len(candles) < 15:
        return None
    fast = compute_fast_indicators(candles)
    df = pd.DataFrame(candles)
    stoch = compute_stochastic(df, k_period=5, k_smooth=3, d_smooth=3)
    bb = compute_bb(df, BB_PERIOD, BB_STD)
    entry_px = float(df["close"].iloc[-1])
    band_width = bb["upper"] - bb["lower"]
    pct_b = (entry_px - bb["lower"]) / band_width if band_width > 0 else 0.5

    ema_gap = fast.get("ema_gap", "")
    ema_gap_pct = round((ema_gap / entry_px) * 100, 4) if isinstance(ema_gap, (int, float)) and entry_px else ""

    lookback = min(30, len(candles))
    recent = df.iloc[-lookback:]
    lo, hi = float(recent["low"].min()), float(recent["high"].max())
    pct_of_range = (entry_px - lo) / (hi - lo) * 100 if hi > lo else 50.0

    return {
        "rsi": fast.get("rsi", ""), "ema_gap": ema_gap, "ema_gap_pct": ema_gap_pct,
        "atr_pct": fast.get("atr_pct", ""), "supertrend_dir": fast.get("supertrend_dir", ""),
        "stoch_k": stoch["k"], "stoch_d": stoch["d"],
        "stoch_overbought": stoch["k"] >= 80, "stoch_oversold": stoch["k"] <= 20,
        "pct_b": round(pct_b, 3), "bb_extreme": pct_b > 1.0 or pct_b < 0.0,
        "pct_of_recent_range": round(pct_of_range, 1), "lookback_bars": lookback,
    }


def print_report(trades, counterfactuals, diagnostics, entry_signals):
    if not trades:
        print("No completed STOCK_OPT_SCANNER_RT trades today.")
        return
    print(f"{'Symbol':<22}{'Entry':<9}{'Exit':<9}{'Reason':<12}{'PnL':>7}")
    print("-" * 62)
    for t in trades:
        print(f"{t['symbol']:<22}{t['entry_time'].strftime('%H:%M:%S'):<9}"
              f"{t['exit_time'].strftime('%H:%M:%S'):<9}{t['reason']:<12}{t['pnl']:>7.0f}")
        if t["reason"] not in STOP_REASONS:
            continue
        cf = counterfactuals.get(t["order_id"])
        if cf:
            if not cf["checked"]:
                print(f"    TP-check   : skipped ({cf['note']})")
            elif cf["would_have_hit_target"]:
                print(f"    TP-check   : WOULD have hit target Rs{cf['target_rs']:.0f} at "
                      f"{cf['tp_hit_time'].strftime('%H:%M')} ({cf['tp_hit_price']}) — "
                      f"but first dipped to {cf['lowest_point']} at "
                      f"{cf['lowest_point_time'].strftime('%H:%M')} (worst unreal Rs{cf['worst_unreal_rs']:.0f})")
            else:
                print(f"    TP-check   : would NOT have hit target — low of remaining session "
                      f"{cf['lowest_point']} at {cf['lowest_point_time'].strftime('%H:%M')} "
                      f"(worst unreal Rs{cf['worst_unreal_rs']:.0f})")
        sig = entry_signals.get(t["order_id"])
        if sig:
            print(f"    Live signal: roc={sig.get('roc','')}  vol_ratio={sig.get('vol_ratio','')}  "
                  f"vwap={sig.get('vwap','')}  close={sig.get('close','')}")
        diag = diagnostics.get(t["order_id"])
        if diag:
            print(f"    Indicators : RSI={diag['rsi']}  Stoch K/D={diag['stoch_k']}/{diag['stoch_d']}"
                  f"{'  [OVERBOUGHT]' if diag['stoch_overbought'] else ''}"
                  f"{'  [OVERSOLD]' if diag['stoch_oversold'] else ''}"
                  f"  BB%B={diag['pct_b']}{'  [OUTSIDE BAND]' if diag['bb_extreme'] else ''}"
                  f"  EMA_gap%={diag['ema_gap_pct']}  ATR%={diag['atr_pct']}  Supertrend={diag['supertrend_dir']}")
            print(f"    Late-entry : entry sat at {diag['pct_of_recent_range']:.0f}% of the "
                  f"last {diag['lookback_bars']} bars' range (near 100% = chasing a high, near 0% = chasing a low)")
        print()
    total = sum(t["pnl"] for t in trades)
    print("-" * 62)
    print(f"{'TOTAL':<51}{total:>7.0f}")


def sl_hit_breakdown(stop_trades, counterfactuals, diagnostics):
    premature, genuine = [], []
    for t in stop_trades:
        cf = counterfactuals.get(t["order_id"])
        if not cf or not cf["checked"]:
            continue
        d = diagnostics.get(t["order_id"])
        (premature if cf["would_have_hit_target"] else genuine).append((t, cf, d))

    def stats(group, label):
        n = len(group)
        row = {"group": label, "n": n}
        if n == 0:
            return row
        row["avg_pnl"] = round(sum(t["pnl"] for t, cf, d in group) / n, 0)
        row["avg_worst_drawdown_rs"] = round(sum(cf["worst_unreal_rs"] for t, cf, d in group) / n, 0)
        with_diag = [(t, cf, d) for t, cf, d in group if d]
        if with_diag:
            m = len(with_diag)
            row["avg_pct_of_range"] = round(sum(d["pct_of_recent_range"] for _, _, d in with_diag) / m, 1)
            row["overbought_pct"] = round(sum(1 for _, _, d in with_diag if d["stoch_overbought"]) / m * 100, 1)
            row["oversold_pct"] = round(sum(1 for _, _, d in with_diag if d["stoch_oversold"]) / m * 100, 1)
        return row

    return [stats(premature, "Premature (stop too tight)"), stats(genuine, "Genuine loss (target unreachable)")]


def print_sl_breakdown(rows):
    print("\n=== SL_HIT breakdown: premature stop vs genuine loss ===")
    for r in rows:
        if r["n"] == 0:
            print(f"  {r['group']}: no checked trades")
            continue
        print(f"  {r['group']}: n={r['n']}  avg_pnl={r.get('avg_pnl')}  "
              f"avg_worst_drawdown_rs={r.get('avg_worst_drawdown_rs')}  "
              f"avg_pct_of_range={r.get('avg_pct_of_range','n/a')}%  "
              f"overbought%={r.get('overbought_pct','n/a')}%  oversold%={r.get('oversold_pct','n/a')}%")


def compute_indicator_tests(trades, diagnostics):
    ema_vals = sorted(
        d["ema_gap_pct"] for t in trades
        if (d := diagnostics.get(t["order_id"])) and isinstance(d.get("ema_gap_pct"), (int, float))
    )
    if ema_vals:
        lo_cut = ema_vals[int(0.25 * len(ema_vals))]
        hi_idx = min(int(0.75 * len(ema_vals)), len(ema_vals) - 1)
        hi_cut = ema_vals[hi_idx]
    else:
        lo_cut = hi_cut = 0.0

    return [
        {
            "key": "pct_b", "name": "Bollinger %B(20,2.0)",
            "rule": "extreme if %B>1 or %B<0 (price outside the bands) — fixed convention",
            "is_extreme": lambda v: v > 1.0 or v < 0.0,
        },
        {
            "key": "rsi", "name": "RSI",
            "rule": "extreme if RSI>70 or RSI<30 (standard overbought/oversold) — fixed convention",
            "is_extreme": lambda v: v > 70 or v < 30,
        },
        {
            "key": "ema_gap_pct", "name": "EMA gap % (fast vs slow, % of price)",
            "rule": f"extreme if outside today's middle 50% (below {lo_cut:.3f}% or above {hi_cut:.3f}%) "
                    f"— no fixed industry threshold for this one, calibrated to TODAY's own spread",
            "is_extreme": (lambda v, lo=lo_cut, hi=hi_cut: v < lo or v > hi),
        },
    ]


def indicator_effectiveness(trades, diagnostics):
    tests = compute_indicator_tests(trades, diagnostics)
    results = []
    for test in tests:
        key, is_extreme = test["key"], test["is_extreme"]
        rows = [(t, diagnostics[t["order_id"]][key]) for t in trades
                if diagnostics.get(t["order_id"]) and isinstance(diagnostics[t["order_id"]].get(key), (int, float))]
        if not rows:
            continue
        cut = [(t, v) for t, v in rows if is_extreme(v)]
        kept = [(t, v) for t, v in rows if not is_extreme(v)]

        def stats(group):
            n = len(group)
            wins = sum(1 for t, v in group if t["reason"] in WIN_REASONS)
            losses = sum(1 for t, v in group if t["reason"] in LOSS_REASONS)
            wr = wins / (wins + losses) * 100 if (wins + losses) else None
            return n, wins, losses, wr, sum(t["pnl"] for t, v in group)

        n_cut, cut_w, cut_l, cut_wr, cut_pnl = stats(cut)
        n_kept, kept_w, kept_l, kept_wr, kept_pnl = stats(kept)
        win_vals = [v for t, v in rows if t["reason"] in WIN_REASONS]
        loss_vals = [v for t, v in rows if t["reason"] in LOSS_REASONS]

        diff_pp = None
        verdict = "not enough win/loss trades in one group to judge"
        if cut_wr is not None and kept_wr is not None:
            diff_pp = kept_wr - cut_wr
            if diff_pp >= 5:
                verdict = "Informative — cutting these removes more losers than winners"
            elif diff_pp <= -5:
                verdict = "Backwards — cutting these removes more WINNERS than losers; don't use as-is"
            else:
                verdict = "No real discrimination — win rates are about equal"

        results.append({
            "indicator": test["name"], "rule": test["rule"],
            "n_cut": n_cut, "cut_wins": cut_w, "cut_losses": cut_l,
            "cut_win_rate_pct": round(cut_wr, 1) if cut_wr is not None else "",
            "n_kept": n_kept, "kept_wins": kept_w, "kept_losses": kept_l,
            "kept_win_rate_pct": round(kept_wr, 1) if kept_wr is not None else "",
            "win_rate_diff_pp": round(diff_pp, 1) if diff_pp is not None else "",
            "pnl_actual_total": round(sum(t["pnl"] for t in trades), 0),
            "pnl_if_cut_removed": round(kept_pnl, 0),
            "avg_value_winners": round(sum(win_vals) / len(win_vals), 3) if win_vals else "",
            "avg_value_losers": round(sum(loss_vals) / len(loss_vals), 3) if loss_vals else "",
            "verdict": verdict,
        })
    return results


def print_indicator_effectiveness(results):
    print("\n=== Indicator effectiveness: cut vs kept, whole day ===")
    for r in results:
        print(f"\n  {r['indicator']} — {r['rule']}")
        print(f"    Cut : n={r['n_cut']}  wins={r['cut_wins']}  losses={r['cut_losses']}  win_rate={r['cut_win_rate_pct']}%")
        print(f"    Kept: n={r['n_kept']}  wins={r['kept_wins']}  losses={r['kept_losses']}  win_rate={r['kept_win_rate_pct']}%")
        print(f"    Win-rate diff (kept - cut): {r['win_rate_diff_pp']} pp   Verdict: {r['verdict']}")
        print(f"    Avg value — winners: {r['avg_value_winners']}  losers: {r['avg_value_losers']}")


def write_report_csv(trades, counterfactuals, diagnostics, entry_signals, out_path):
    fields = [
        "symbol", "stock", "entry_time", "exit_time", "entry_price", "exit_price",
        "reason", "pnl", "qty",
        "tp_checked", "would_have_hit_target", "tp_hit_time", "tp_hit_price",
        "lowest_point", "lowest_point_time", "worst_unreal_rs", "target_rs",
        "live_roc", "live_vol_ratio", "live_vwap", "live_close",
        "rsi", "stoch_k", "stoch_d", "stoch_overbought", "stoch_oversold",
        "pct_b", "bb_extreme", "ema_gap_pct",
        "atr_pct", "supertrend_dir", "pct_of_recent_range", "lookback_bars",
    ]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for t in trades:
            row = {
                "symbol": t["symbol"], "stock": t["stock"],
                "entry_time": t["entry_time"].strftime("%H:%M:%S"),
                "exit_time": t["exit_time"].strftime("%H:%M:%S"),
                "entry_price": t["entry_price"], "exit_price": t["exit_price"],
                "reason": t["reason"], "pnl": t["pnl"], "qty": t["qty"],
            }
            cf = counterfactuals.get(t["order_id"])
            if cf:
                row["tp_checked"] = cf["checked"]
                if cf["checked"]:
                    row["would_have_hit_target"] = cf["would_have_hit_target"]
                    row["lowest_point"] = cf["lowest_point"]
                    row["lowest_point_time"] = cf["lowest_point_time"].strftime("%H:%M:%S")
                    row["worst_unreal_rs"] = cf["worst_unreal_rs"]
                    row["target_rs"] = cf["target_rs"]
                    if cf["would_have_hit_target"]:
                        row["tp_hit_time"] = cf["tp_hit_time"].strftime("%H:%M:%S")
                        row["tp_hit_price"] = cf["tp_hit_price"]
            sig = entry_signals.get(t["order_id"])
            if sig:
                row["live_roc"] = sig.get("roc", "")
                row["live_vol_ratio"] = sig.get("vol_ratio", "")
                row["live_vwap"] = sig.get("vwap", "")
                row["live_close"] = sig.get("close", "")
            diag = diagnostics.get(t["order_id"])
            if diag:
                row.update({k: diag[k] for k in (
                    "rsi", "stoch_k", "stoch_d", "stoch_overbought", "stoch_oversold",
                    "pct_b", "bb_extreme", "ema_gap_pct", "atr_pct", "supertrend_dir",
                    "pct_of_recent_range", "lookback_bars")})
            w.writerow({k: row.get(k, "") for k in fields})
    print(f"Wrote {len(trades)} rows to {out_path}")


def append_study_log(effectiveness, run_date, log_path=STUDY_LOG_CSV):
    if not effectiveness:
        return
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    file_exists = os.path.exists(log_path)
    fields = ["date"] + list(effectiveness[0].keys())
    with open(log_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            w.writeheader()
        for row in effectiveness:
            w.writerow({"date": run_date, **row})
    print(f"Appended {len(effectiveness)} rows to {log_path} (cross-day tracking, stable location)")


if __name__ == "__main__":
    trades = load_today_trades()
    stop_trades = [t for t in trades if t["reason"] in STOP_REASONS]

    counterfactuals, diagnostics, entry_signals = {}, {}, {}
    if trades:
        kite = load_kite()
        nfo_map = build_token_map(kite, "NFO")
        nse_map = build_token_map(kite, "NSE")

        for t in stop_trades:
            counterfactuals[t["order_id"]] = check_counterfactual(kite, nfo_map, t)
            time_mod.sleep(0.35)

        for t in trades:
            entry_signals[t["order_id"]] = load_entry_signal(t["stock"], t["entry_time"])
            candles = fetch_underlying_candles(kite, nse_map, t["stock"], t["entry_time"].date(), t["entry_time"])
            time_mod.sleep(0.35)
            diagnostics[t["order_id"]] = entry_diagnostics(candles)

    print_report(trades, counterfactuals, diagnostics, entry_signals)
    sl_rows = sl_hit_breakdown(stop_trades, counterfactuals, diagnostics)
    print_sl_breakdown(sl_rows)
    effectiveness = indicator_effectiveness(trades, diagnostics)
    print_indicator_effectiveness(effectiveness)

    today_str = str(date.today())
    dated_dir = os.path.join(LOGS_DIR, today_str)
    write_report_csv(trades, counterfactuals, diagnostics, entry_signals,
                      os.path.join(dated_dir, f"eod_diagnostics_{today_str}.csv"))
    append_study_log(effectiveness, today_str)
