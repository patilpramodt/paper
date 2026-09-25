"""
strategies/stock_opt_morning_breakout_strategy.py

STOCK_OPT_MORNING_BO — morning high/low-of-day breakout on single stocks,
traded with the ATM option. PAPER ONLY.

═══════════════════════════════════════════════════════════════════════════
  WHY THIS EXISTS
═══════════════════════════════════════════════════════════════════════════
The STOCK_OPT_SCANNER family (V1 / RT / FLOW) loses money even BEFORE costs:
Rs 300 targets against Rs 1,000-2,000 stops, ~100 trades a day, no edge in
the entry. This strategy came out of a 2026-09-25 backtest search across
~4,100 configurations (VWAP burst, ORB, pullback, VWAP reversion, PDH/PDL,
HOD break x NIFTY / relative-strength / time filters x ATR exits), on 66
sessions of 5-min spot data for the same 14 stocks, ranked on the first 44
days and checked on the last 22:

    first 44 days :  94 trades  50% win  +Rs206/trade
    last  22 days :  71 trades  65% win  +Rs494/trade
    real ATM option replay (Aug 26 - Sep 25, 62 trades):
        exits on option premium   +Rs176/trade
        exits on SPOT levels      ~+Rs705/trade  (intrabar-interpolated)
    random morning entries with the same filters: -Rs438/trade

It is the ONLY configuration family that was positive in both halves. It is
still one month of real option data — treat the paper record as the test.

═══════════════════════════════════════════════════════════════════════════
  RULES (5-minute stock bars; mirror for PE)
═══════════════════════════════════════════════════════════════════════════
  ENTRY — evaluated on each closed bar whose START is 09:45 .. 11:30
   1. close > highest high of today's earlier bars   (new high of day)
   2. close > session VWAP
   3. bar volume >= 1.2 x average of the prior 20 bars (history spans days)
   4. NIFTY bias UP: NIFTY above its day open AND above EMA21 of 5-min closes
   5. relative strength: stock %chg from open - NIFTY %chg from open >= 0.3%
   6. target must be worth >= Rs 300 NET for one lot (checked on the leg's
      first tick with the live spread)
   -> buy ATM CE (nearest expiry). Afternoon breakouts lost money in every
      test — the 11:30 cut-off is part of the edge, not a convenience.

  EXIT — on the STOCK price, not the option premium
   SL      spot moves 2.0 x ATR(14) against the signal-bar close
   TARGET  spot moves 3.0 x ATR(14) in favour
   EOD     15:10
   SAFETY  option unrealised loss >= max_loss_rs (a leg that diverges from
           the stock, e.g. IV crush) — not part of the tested logic

  Spot-level exits are deliberate: stopping on the option premium lost ~75%
  of the edge in the real-option replay, because 5-min option noise hits a
  premium stop that the stock itself never reaches.

  One open position per stock. max_trades_per_stock / max_losses_per_stock
  never bound in the backtest (a stock rarely breaks out twice by 11:30).
"""

import csv
import logging
import os
import statistics
import time
from collections import deque
from datetime import datetime, time as dtime, timedelta, timezone
from typing import Optional

from core.base_strategy import BaseStrategy
from core.costs import effective_spread, fixed_costs_rs, net_pnl_rs

log = logging.getLogger("strategy.stock_opt_morning_bo")

_IST = timezone(timedelta(hours=5, minutes=30))
NIFTY_TOKEN = 256265


CFG = {
    "enabled": True,

    # ── session (IST) ────────────────────────────────────────────────────────
    "first_bar_start": dtime(9, 45),   # 7th bar of the day — needs an opening range
    "last_bar_start":  dtime(11, 30),
    "close_time":      dtime(15, 10),

    # ── signal ───────────────────────────────────────────────────────────────
    "bar_minutes":   5,
    "atr_bars":      14,
    "vol_avg_bars":  20,
    "vol_mult":      1.2,
    "nifty_ema":     21,
    "rs_min":        0.003,            # 0.3% out/under-performance vs NIFTY
    "seed_days":     5,                # calendar days of history fetched at start

    # ── exits (multiples of stock ATR, applied to the SPOT) ──────────────────
    "sl_atr":        2.0,
    "tp_atr":        3.0,
    "max_loss_rs":   4000.0,           # safety net on the option leg only

    # ── option selection / gates ─────────────────────────────────────────────
    "atm_delta":       0.47,           # measured on real Sep-2026 ATM stock options
    "min_net_tp_rs":   300.0,
    "opt_tick_wait_s": 20,
    "prem_min":        3.0,
    "max_spread_pct":  0.05,           # of premium
    "skip_expiry_day": True,           # nearest-expiry ATM on expiry day is a lottery ticket
    "lots":            1,

    # ── limits ───────────────────────────────────────────────────────────────
    "max_trades_per_stock": 3,
    "max_losses_per_stock": 2,
    "stale_price_sec":      45,

    "csv_file": "stock_opt_morning_bo_trades.csv",
}

LIVE_MODE = False   # same single-slot OrderRouter limitation as STOCK_OPT_SCANNER


def _now_ist() -> datetime:
    return datetime.now(tz=_IST).replace(tzinfo=None)


class _Bars:
    """Rolling 5-min bars for one instrument + today's session stats."""

    def __init__(self):
        self.bars     = deque(maxlen=60)    # closed bars, may span previous days
        self.cur      = None
        self.last_cum = None
        self.day      = None
        self.day_open = None
        self.vwap_pv  = 0.0
        self.vwap_v   = 0.0

    def vwap(self) -> Optional[float]:
        return self.vwap_pv / self.vwap_v if self.vwap_v > 0 else None

    def atr(self, n: int) -> Optional[float]:
        bl = list(self.bars)
        if len(bl) < n + 1:
            return None
        trs = []
        for prev, b in zip(bl[-n - 1:-1], bl[-n:]):
            trs.append(max(b["h"] - b["l"], abs(b["h"] - prev["c"]), abs(b["l"] - prev["c"])))
        return sum(trs) / n

    def avg_volume(self, n: int) -> Optional[float]:
        """Average of the n bars BEFORE the most recent closed bar."""
        bl = list(self.bars)[-(n + 1):-1]
        return sum(b["v"] for b in bl) / len(bl) if len(bl) >= n else None

    def new_day(self, day, open_px):
        self.day, self.day_open = day, open_px
        self.vwap_pv = self.vwap_v = 0.0

    def close_bar(self):
        """Move `cur` into history. Returns the closed bar."""
        b = self.cur
        self.bars.append(b)
        self.cur = None
        return b


class StockOptMorningBreakoutStrategy(BaseStrategy):

    INDEX_TOKEN = NIFTY_TOKEN     # NIFTY ticks: bias filter + housekeeping clock
    LIVE_MODE   = LIVE_MODE

    def __init__(self, market_hub):
        super().__init__(market_hub)
        self._store      = None
        self._stocks     = {}      # spot token -> dict(sym, bars)
        self._nifty      = _Bars()
        self._nifty_ema  = None
        self._positions  = {}      # sym -> trade dict (one per stock)
        self._pending    = {}      # opt token -> pending entry
        self._per_stock  = {}      # sym -> {"n": trades, "losses": n}
        self._exit_bar   = {}      # sym -> start of the bar the last trade exited in
        self._completed  = []
        self._today_pnl  = 0.0
        self._ready      = False
        self._eod_done   = False
        self._last_hb    = None
        self._expiry_day = False

    @property
    def name(self) -> str:
        return "STOCK_OPT_MORNING_BO"

    # ══════════════════════════════════════════════════════════════════════════
    # PRE-MARKET
    # ══════════════════════════════════════════════════════════════════════════

    def pre_market(self, premarket_data, instruments) -> bool:
        if not CFG["enabled"]:
            log.info(f"[{self.name}] disabled via CFG")
            return False

        from core.instruments import StockOptionStore
        if not isinstance(instruments, StockOptionStore) or not instruments.universe:
            log.error(f"[{self.name}] needs a loaded StockOptionStore — check t.py wiring")
            return False
        self._store = instruments

        for sym in self._store.universe:
            tok = self._store.spot_token(sym)
            if not tok:
                continue
            self._stocks[tok] = {"sym": sym, "bars": _Bars()}
            self._per_stock[sym] = {"n": 0, "losses": 0}
            self.subscribe_option(tok)
            self._hub.set_token_owner(tok, self.name)

        self._seed_history()

        dte = min(self._store.days_to_expiry(s) for s in self._store.universe)
        self._expiry_day = CFG["skip_expiry_day"] and dte <= 0
        if self._expiry_day:
            log.warning(f"[{self.name}] expiry day for the nearest series — no new entries today")

        self._ready = True
        log.info(
            f"[{self.name}] ready | PAPER | {len(self._stocks)} stocks | entries "
            f"{CFG['first_bar_start']:%H:%M}-{CFG['last_bar_start']:%H:%M} bar start | "
            f"SL {CFG['sl_atr']}xATR TP {CFG['tp_atr']}xATR on spot | min net TP Rs{CFG['min_net_tp_rs']:.0f}"
        )
        return True

    def _seed_history(self):
        """
        ATR(14) and the 20-bar volume average span previous sessions in the
        backtest, so they must be valid at 09:45. Also rebuilds today's VWAP /
        HOD / LOD / day open when the process restarts mid-session.
        """
        kite = getattr(self._hub, "kite", None)
        if kite is None:
            log.warning(f"[{self.name}] no kite handle — indicators warm up live (first ~1h dead)")
            return
        now   = _now_ist()
        start = now - timedelta(days=CFG["seed_days"])
        cur_bar = self._bar_start(now)
        targets = [(tok, s["bars"], s["sym"]) for tok, s in self._stocks.items()]
        targets.append((NIFTY_TOKEN, self._nifty, "NIFTY"))
        for tok, bars, sym in targets:
            try:
                raw = kite.historical_data(tok, start.strftime("%Y-%m-%d %H:%M:%S"),
                                           now.strftime("%Y-%m-%d %H:%M:%S"), "5minute")
            except Exception as e:
                log.warning(f"[{self.name}] history seed failed for {sym}: {e}")
                continue
            finally:
                time.sleep(0.35)   # Kite historical API: 3 requests/second
            for r in raw:
                ts = r["date"].replace(tzinfo=None)
                if ts >= cur_bar:
                    break          # the in-progress bar is rebuilt from live ticks
                if ts.date() == now.date() and bars.day != ts.date():
                    bars.new_day(ts.date(), r["open"])
                bars.cur = {"ts": ts, "o": r["open"], "h": r["high"], "l": r["low"],
                            "c": r["close"], "v": r.get("volume", 0) or 0}
                if ts.date() == now.date():
                    tp = (r["high"] + r["low"] + r["close"]) / 3
                    bars.vwap_pv += tp * bars.cur["v"]
                    bars.vwap_v  += bars.cur["v"]
                bars.close_bar()
                if tok == NIFTY_TOKEN:
                    self._update_nifty_ema(r["close"])
        log.info(f"[{self.name}] seeded history for {len(targets)} instruments")

    # ══════════════════════════════════════════════════════════════════════════
    # NIFTY ticks — bias + heartbeat
    # ══════════════════════════════════════════════════════════════════════════

    def on_tick(self, price: float, ts: datetime, tick_ts: datetime):
        if not self._ready or not price:
            return
        self._feed(self._nifty, price, 0, tick_ts or ts)
        if self._last_hb and (ts - self._last_hb).total_seconds() < 1.0:
            return
        self._last_hb = ts
        self._heartbeat(ts)

    def on_candle(self, candle: dict, ts: datetime):
        return   # not delivered to INDEX_TOKEN strategies; bars are built internally

    def _update_nifty_ema(self, close: float):
        k = 2.0 / (CFG["nifty_ema"] + 1)
        self._nifty_ema = close if self._nifty_ema is None else self._nifty_ema + k * (close - self._nifty_ema)

    def _nifty_state(self):
        """(bias, pct_from_open) using NIFTY's live price. bias: +1 / -1 / 0."""
        n  = self._nifty
        px = self.get_price(NIFTY_TOKEN) or (n.cur["c"] if n.cur else None)
        if not px or not n.day_open or self._nifty_ema is None:
            return 0, None
        k   = 2.0 / (CFG["nifty_ema"] + 1)
        ema = self._nifty_ema + k * (px - self._nifty_ema)   # EMA including the live bar
        pct = px / n.day_open - 1
        if px > n.day_open and px > ema:
            return 1, pct
        if px < n.day_open and px < ema:
            return -1, pct
        return 0, pct

    def _heartbeat(self, ts: datetime):
        if ts.time() >= CFG["close_time"]:
            if self._positions and not self._eod_done:
                for sym in list(self._positions):
                    self._exit(sym, "EOD", ts)
            self._eod_done = True
            for tok in list(self._pending):
                self._drop_pending(tok)
            return

        for tok in list(self._pending):
            if (ts - self._pending[tok]["ts"]).total_seconds() > CFG["opt_tick_wait_s"]:
                p = self._pending[tok]
                self._log_signal(ts, p["sym"], "PENDING_TIMEOUT", p["side"], block="no_option_tick")
                self._drop_pending(tok)

        # Spot-level exits for stocks that have gone quiet, and the option safety net.
        for sym in list(self._positions):
            tr = self._positions[sym]
            spot = self.get_price(tr["spot_token"])
            if spot:
                self._check_spot_exit(sym, spot, ts)
            if sym in self._positions:
                px = self.get_price(tr["token"])
                if px:
                    self._check_option_safety(sym, px, ts)

    # ══════════════════════════════════════════════════════════════════════════
    # TICK ROUTING
    # ══════════════════════════════════════════════════════════════════════════

    def on_option_tick(self, token: int, price: float, ts: datetime, tick_ts: datetime = None):
        if not self._ready or not price:
            return
        st = self._stocks.get(token)
        if st is not None:
            cum = self._hub.last_volume(token)
            closed = self._feed(st["bars"], price, cum, tick_ts or ts)
            if st["sym"] in self._positions:
                self._check_spot_exit(st["sym"], price, ts)
            if closed is not None:
                self._evaluate(st, closed, ts)
            return
        if token in self._pending:
            self._try_fill_pending(token, price, ts)
            return
        for sym, tr in list(self._positions.items()):
            if tr["token"] == token:
                self._check_option_safety(sym, price, ts)
                return

    @staticmethod
    def _bar_start(ts: datetime) -> datetime:
        m = CFG["bar_minutes"]
        return ts.replace(minute=(ts.minute // m) * m, second=0, microsecond=0)

    def _feed(self, b: _Bars, price: float, cum_vol: int, tick_ts: datetime):
        """Update bars from one tick. Returns the bar that just closed, if any."""
        dv = 0
        if cum_vol:
            if b.last_cum is not None and cum_vol >= b.last_cum:
                dv = cum_vol - b.last_cum
            b.last_cum = cum_vol
        if b.day != tick_ts.date():
            b.new_day(tick_ts.date(), price)
            b.last_cum = cum_vol or None
            dv = 0
        if dv > 0:
            b.vwap_pv += price * dv
            b.vwap_v  += dv

        bs, closed = self._bar_start(tick_ts), None
        if b.cur is not None and bs > b.cur["ts"]:
            closed = b.close_bar()
            if b is self._nifty:
                self._update_nifty_ema(closed["c"])
        if b.cur is None:
            b.cur = {"ts": bs, "o": price, "h": price, "l": price, "c": price, "v": dv}
        else:
            b.cur["h"] = max(b.cur["h"], price)
            b.cur["l"] = min(b.cur["l"], price)
            b.cur["c"] = price
            b.cur["v"] += dv
        return closed

    # ══════════════════════════════════════════════════════════════════════════
    # SIGNAL
    # ══════════════════════════════════════════════════════════════════════════

    def _evaluate(self, st: dict, bar: dict, ts: datetime):
        sym, b = st["sym"], st["bars"]
        bt = bar["ts"].time()
        if not (CFG["first_bar_start"] <= bt <= CFG["last_bar_start"]):
            return
        if bar["ts"].date() != b.day or self._expiry_day or self._eod_done:
            return
        if sym in self._positions or any(p["sym"] == sym for p in self._pending.values()):
            return
        ps = self._per_stock[sym]
        if ps["n"] >= CFG["max_trades_per_stock"] or ps["losses"] >= CFG["max_losses_per_stock"]:
            return
        # As tested: no re-entry on the bar the previous trade exited in.
        if sym in self._exit_bar and bar["ts"] <= self._exit_bar[sym]:
            return

        # HOD/LOD of bars BEFORE this one: close_bar() already folded this bar in.
        prior = [x for x in list(b.bars)[:-1] if x["ts"].date() == b.day]
        if not prior:
            return
        hod = max(x["h"] for x in prior)
        lod = min(x["l"] for x in prior)
        c, vwap = bar["c"], b.vwap()
        if vwap is None:
            return
        if c > hod and c > vwap:
            side = "UP"
        elif c < lod and c < vwap:
            side = "DOWN"
        else:
            return

        atr     = b.atr(CFG["atr_bars"])
        avg_vol = b.avg_volume(CFG["vol_avg_bars"])
        nb, npct = self._nifty_state()
        sgn = 1 if side == "UP" else -1
        rs  = (c / b.day_open - 1) - npct if (npct is not None and b.day_open) else None
        vol_ratio = bar["v"] / avg_vol if avg_vol else 0.0
        meta = dict(close=c, vwap=vwap, atr=atr or 0.0, vol_ratio=vol_ratio,
                    nifty_bias=nb, rs=rs if rs is not None else 0.0, hod=hod, lod=lod)

        block = None
        if atr is None or avg_vol is None:
            block = "warmup"
        elif vol_ratio < CFG["vol_mult"]:
            block = f"vol_ratio={vol_ratio:.2f}"
        elif nb != sgn:
            block = f"nifty_bias={nb}"
        elif rs is None or sgn * rs < CFG["rs_min"]:
            block = f"rs={rs if rs is None else round(rs, 4)}"
        if block:
            self._log_signal(ts, sym, "GATE_BLOCK", side, block=block, **meta)
            return

        self._log_signal(ts, sym, "TRIGGER", side, **meta)
        self._arm_entry(st, side, c, atr, ts, meta)

    # ══════════════════════════════════════════════════════════════════════════
    # ENTRY
    # ══════════════════════════════════════════════════════════════════════════

    def _arm_entry(self, st: dict, side: str, spot: float, atr: float, ts: datetime, meta: dict):
        sym = st["sym"]
        opt_type = "CE" if side == "UP" else "PE"
        strike = self._store.atm_strike(sym, spot)
        tok, opt_symbol, lot = self._store.get_option(sym, strike, opt_type)
        if not tok:
            self._log_signal(ts, sym, "NO_CONTRACT", side, block=f"strike={strike}")
            return
        if tok in self._pending:
            return
        spot_tok = self._store.spot_token(sym)
        self._pending[tok] = {
            "sym": sym, "side": side, "opt_type": opt_type, "strike": strike,
            "opt_symbol": opt_symbol, "lot": lot, "qty": lot * CFG["lots"],
            "spot": spot, "atr": atr, "spot_token": spot_tok, "ts": ts, "meta": meta,
        }
        self.subscribe_option(tok)
        self._hub.set_token_owner(tok, self.name)
        log.info(f"[{self.name}] {sym} {side} HOD/LOD break @ {spot:.2f} "
                 f"(vol x{meta['vol_ratio']:.1f}, rs {meta['rs']:+.2%}) → arming {opt_symbol}")

    def _try_fill_pending(self, tok: int, ltp: float, ts: datetime):
        p = self._pending[tok]
        sym, qty = p["sym"], p["qty"]
        bid, ask, _, _ = self._hub.best_bid_ask(tok)
        spread = effective_spread(ltp, bid, ask)

        entry  = ltp + spread / 2.0
        tp_pts = CFG["atm_delta"] * CFG["tp_atr"] * p["atr"]
        net_tp = (tp_pts - spread) * qty - fixed_costs_rs(qty, entry, entry + tp_pts)

        block = None
        if ltp < CFG["prem_min"]:
            block = f"premium={ltp:.2f}"
        elif spread > CFG["max_spread_pct"] * ltp:
            block = f"spread={spread:.2f} ({spread / ltp:.1%})"
        elif net_tp < CFG["min_net_tp_rs"]:
            block = f"net_tp=Rs{net_tp:.0f}<{CFG['min_net_tp_rs']:.0f}"
        if block:
            log.info(f"[{self.name}] {sym} {p['opt_symbol']} entry BLOCKED — {block}")
            self._log_signal(ts, sym, "ENTRY_BLOCK", p["side"], block=block, ltp=ltp, spread=spread)
            self._drop_pending(tok)
            return

        res = self._place_buy(p["opt_symbol"], tok, qty, ltp)
        if res is None:
            log.error(f"[{self.name}] {p['opt_symbol']} BUY failed")
            self._drop_pending(tok)
            return
        order_id, raw_fill = res
        fill = round(raw_fill + spread / 2.0, 2) if not LIVE_MODE else raw_fill

        sgn = 1 if p["side"] == "UP" else -1
        tr = {
            "token": tok, "spot_token": p["spot_token"], "sym": sym,
            "opt_symbol": p["opt_symbol"], "side": p["side"], "opt_type": p["opt_type"],
            "strike": p["strike"], "qty": qty, "lot": p["lot"], "entry": fill,
            "entry_ts": ts, "order_id": order_id, "spread": spread,
            "entry_spot": p["spot"], "atr": p["atr"],
            "sl_spot": round(p["spot"] - sgn * CFG["sl_atr"] * p["atr"], 2),
            "tp_spot": round(p["spot"] + sgn * CFG["tp_atr"] * p["atr"], 2),
            "net_tp_est": net_tp, "meta": p["meta"],
        }
        self._pending.pop(tok, None)
        self._positions[sym] = tr
        self._per_stock[sym]["n"] += 1
        log.info(
            f"[{self.name}] ENTRY {p['opt_symbol']} @ {fill:.2f} qty={qty} | spot {p['spot']:.2f} "
            f"SL {tr['sl_spot']:.2f} TP {tr['tp_spot']:.2f} (ATR {p['atr']:.2f}) | "
            f"est net at TP Rs{net_tp:.0f} | spread {spread:.2f}"
        )
        self._log_trade(ts, tr, "ENTRY", fill, "OPEN", 0.0, 0.0, "")

    def _drop_pending(self, tok: int):
        self._pending.pop(tok, None)
        self._hub.clear_token_owner(tok, self.name)
        self.unsubscribe_option(tok)

    # ══════════════════════════════════════════════════════════════════════════
    # EXITS
    # ══════════════════════════════════════════════════════════════════════════

    def _check_spot_exit(self, sym: str, spot: float, ts: datetime):
        tr = self._positions.get(sym)
        if tr is None:
            return
        up = tr["side"] == "UP"
        if (spot <= tr["sl_spot"]) if up else (spot >= tr["sl_spot"]):
            self._exit(sym, "SL_SPOT", ts)
        elif (spot >= tr["tp_spot"]) if up else (spot <= tr["tp_spot"]):
            self._exit(sym, "TP_SPOT", ts)

    def _check_option_safety(self, sym: str, ltp: float, ts: datetime):
        tr = self._positions.get(sym)
        if tr is None:
            return
        exit_px = max(0.05, ltp - tr["spread"] / 2.0)
        if (exit_px - tr["entry"]) * tr["qty"] <= -CFG["max_loss_rs"]:
            self._exit(sym, "MAX_LOSS", ts, ltp)

    def _exit(self, sym: str, reason: str, ts: datetime, ltp: float = None):
        tr = self._positions.get(sym)
        if tr is None:
            return
        tok, qty = tr["token"], tr["qty"]
        stale_exit = 0
        if ltp is None:
            ltp = self.get_price(tok)
            pts = self.get_price_ts(tok)
            if pts and (ts - pts).total_seconds() > CFG["stale_price_sec"]:
                stale_exit = 1
                log.warning(f"[{self.name}] {tr['opt_symbol']} exit on a {(ts - pts).total_seconds():.0f}s-old option price")
            if ltp is None:
                ltp, stale_exit = tr["entry"], 1
                log.error(f"[{self.name}] {tr['opt_symbol']} NO option price at exit — booked at entry, not real P&L")

        res = self._place_sell_with_retry(tr["opt_symbol"], tok, qty, ltp)
        if res is None:
            log.error(f"[{self.name}] EXIT FAILED {tr['opt_symbol']} — MANUAL CHECK REQUIRED")
            return
        _, raw_exit = res
        exit_px = round(max(0.05, raw_exit - tr["spread"] / 2.0), 2) if not LIVE_MODE else raw_exit
        gross = (exit_px - tr["entry"]) * qty
        net   = net_pnl_rs(tr["entry"], exit_px, qty)
        self._today_pnl += net
        if net <= 0:
            self._per_stock[sym]["losses"] += 1

        tr.update({"exit_price": exit_px, "exit_reason": reason, "pnl": net, "gross_pnl": gross,
                   "time_in_trade_s": (ts - tr["entry_ts"]).total_seconds(), "stale_exit": stale_exit,
                   "exit_spot": self.get_price(tr["spot_token"])})
        self._completed.append(tr)
        self._positions.pop(sym, None)
        self._exit_bar[sym] = self._bar_start(ts)
        self._hub.clear_token_owner(tok, self.name)
        self.unsubscribe_option(tok)
        log.info(f"[{self.name}] EXIT [{reason}] {tr['opt_symbol']} {tr['entry']:.2f} → {exit_px:.2f} | "
                 f"gross={gross:.0f} net={net:.0f} | {tr['time_in_trade_s'] / 60:.1f}min | day={self._today_pnl:.0f}")
        self._log_trade(ts, tr, "EXIT", exit_px, "CLOSED", net, gross, reason)

    # ══════════════════════════════════════════════════════════════════════════
    # LOGGING
    # ══════════════════════════════════════════════════════════════════════════

    _TRADE_FIELDS = [
        "timestamp", "stock", "symbol", "action", "side", "opt_type", "strike",
        "price", "qty", "lot", "status", "pnl", "gross_pnl", "cost_drag", "spread",
        "reason", "mode", "order_id", "time_in_trade_s", "entry_spot", "sl_spot",
        "tp_spot", "exit_spot", "atr", "net_tp_est", "vol_ratio", "rs", "stale_exit",
    ]

    _SIGNAL_FIELDS = [
        "timestamp", "stock", "event", "side", "block_reason", "close", "vwap",
        "hod", "lod", "atr", "vol_ratio", "nifty_bias", "rs", "ltp", "spread",
    ]

    def _log_trade(self, ts, tr, action, price, status, pnl, gross, reason):
        meta = tr.get("meta", {})
        row = {
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"), "stock": tr["sym"],
            "symbol": tr["opt_symbol"], "action": action, "side": tr["side"],
            "opt_type": tr["opt_type"], "strike": tr["strike"], "price": round(price, 2),
            "qty": tr["qty"], "lot": tr["lot"], "status": status, "pnl": round(pnl, 2),
            "gross_pnl": round(gross, 2), "cost_drag": round(gross - pnl, 2),
            "spread": tr["spread"], "reason": reason, "mode": "LIVE" if LIVE_MODE else "PAPER",
            "order_id": tr["order_id"], "time_in_trade_s": round(tr.get("time_in_trade_s", 0)),
            "entry_spot": tr["entry_spot"], "sl_spot": tr["sl_spot"], "tp_spot": tr["tp_spot"],
            "exit_spot": tr.get("exit_spot", ""), "atr": round(tr["atr"], 2),
            "net_tp_est": round(tr["net_tp_est"]), "vol_ratio": round(meta.get("vol_ratio", 0), 2),
            "rs": round(meta.get("rs", 0), 4), "stale_exit": tr.get("stale_exit", 0),
        }
        self._write_csv(CFG["csv_file"], self._TRADE_FIELDS, row)

    def _log_signal(self, ts, sym, event, side, block="", **kw):
        row = {"timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"), "stock": sym,
               "event": event, "side": side, "block_reason": block}
        for k, v in kw.items():
            row[k] = round(v, 4) if isinstance(v, float) else v
        self._write_csv(CFG["csv_file"].replace("_trades.csv", "_signals.csv"), self._SIGNAL_FIELDS, row)

    @staticmethod
    def _write_csv(fname, fields, row):
        try:
            exists = os.path.isfile(fname)
            with open(fname, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                if not exists:
                    w.writeheader()
                w.writerow({k: row.get(k, "") for k in fields})
        except Exception as e:
            log.error(f"[STOCK_OPT_MORNING_BO] CSV write failed ({fname}): {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # EOD
    # ══════════════════════════════════════════════════════════════════════════

    def eod_summary(self):
        log.info(f"\n[{self.name}] {'=' * 50}")
        log.info(f"[{self.name}] END OF DAY | mode={'LIVE' if LIVE_MODE else 'PAPER'}")
        if self._positions:
            log.error(f"[{self.name}] {len(self._positions)} POSITION(S) STILL OPEN: {list(self._positions)}")
        for t in self._completed:
            log.info(f"[{self.name}]   {t['sym']:<12} {t['opt_symbol']} [{t['exit_reason']}] "
                     f"{t['entry']:.2f} → {t['exit_price']:.2f} net={t['pnl']:.0f} "
                     f"({t['time_in_trade_s'] / 60:.1f}min)")
        if self._completed:
            wins = sum(1 for t in self._completed if t["pnl"] > 0)
            gross = sum(t["gross_pnl"] for t in self._completed)
            log.info(f"[{self.name}] Trades {len(self._completed)} | W/L {wins}/{len(self._completed) - wins} "
                     f"| gross {gross:.0f} | NET {self._today_pnl:.0f} "
                     f"| avg {statistics.mean(t['pnl'] for t in self._completed):.0f}")
        else:
            log.info(f"[{self.name}] no trades")
        log.info(f"[{self.name}] {'=' * 50}\n")
