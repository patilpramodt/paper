"""
strategies/stock_options_scanner_strategy.py

STOCK_OPT_SCANNER — multi-stock, multi-position option buyer.

═══════════════════════════════════════════════════════════════════════════
  WHAT THIS IS
═══════════════════════════════════════════════════════════════════════════

Every other strategy in this repo trades ONE underlying and holds ONE
position. This one scans ~15 liquid F&O stocks simultaneously and can hold
several positions at once, buying an ATM CE or PE on whichever stock shows
a volume-confirmed directional thrust, and exiting for a rupee-denominated
profit between CFG["target_rs_min"] and CFG["target_rs_max"].


═══════════════════════════════════════════════════════════════════════════
  DESIGN DECISIONS — read before changing anything
═══════════════════════════════════════════════════════════════════════════

1. WHY IT SCANS STOCKS, NOT OPTIONS
   The obvious design is "watch every option and buy the one that moves".
   It is not implementable here:
     • Kite's WebSocket caps around 3000 tokens per connection. 15 stocks
       x ~40 listed strikes x 2 types = 1200 tokens for the universe alone,
       and that is before WsPCR's ~42 index strikes and every other
       strategy's legs. A 30-stock universe blows the cap outright.
     • The VPS is single-core. Every one of those tokens is a tick storm
       through _on_ticks() and a Python callback per strategy.
     • Option premium is a derived, laggy, noisy view of the underlying.
       Volume "in a direction" is a property OF THE STOCK.

   So: subscribe the 15 NSE equity tokens (15 tokens), decide direction on
   the stock, and only then subscribe the ONE option leg we intend to buy.
   Peak token usage is 15 + max_open_positions. The option is checked for
   liquidity at entry — which is where a liquidity check actually matters.

2. WHY IT IS PAPER-ONLY (and cannot simply be flipped live)
   OrderRouter holds a SINGLE global live slot (`_slot_owner`, one string).
   Two live positions at once are structurally impossible with that router,
   and SPIKE currently owns the slot. This strategy therefore never calls
   acquire_slot() — it manages its own position book. Setting
   LIVE_MODE = True without first making OrderRouter multi-slot would let
   this strategy fight SPIKE for the slot and lose, silently.

3. WHY THE TARGET IS IN RUPEES, NOT POINTS
   Every other strategy uses point targets because index lot sizes are
   fixed. Stock lot sizes are not: RELIANCE 500, SBIN 1500, TATAMOTORS 1425.
   One premium point is Rs 500 on one and Rs 1500 on another. A Rs 500
   target therefore means 1.0 point on RELIANCE and 0.33 points on SBIN —
   and 0.33 points is INSIDE the bid-ask spread, i.e. unreachable.
   _feasibility_check() rejects exactly those trades before entry. This is
   the single most important gate in the file.

4. WHY THE UPPER TARGET IS A CAP, NOT A LIMIT ORDER
   Rs 5000 on one lot is a large premium move (10 pts on RELIANCE, 3.3 on
   SBIN) and a hard take-profit that size would mostly end in time-stops.
   Instead: exit the base target at Rs target_rs_min unless the trade is
   already running, in which case trail toward the cap and hard-exit at
   target_rs_max.

5. TIMEFRAME: 3-MINUTE
   1-minute bars on single stocks are mostly spread noise and produce
   volume spikes on single block trades. 5-minute gives ~75 bars a day
   and is too slow to reach a Rs 500 target before the move is over.
   3-minute is the compromise: ~125 bars/session, enough volume per bar
   for the ratio test to mean something, fast enough that a thrust is
   still live when the option leg gets subscribed.

6. COSTS
   core/costs.py's index model (SPREAD_CAP_PTS=8, 1.2% of premium) badly
   understates stock options. This file uses core.costs.effective_spread()
   / stock_round_trip_cost_pts(), which prefer the REAL top-of-book spread
   from MODE_FULL depth and fall back to a 3.5% model.

7. WHAT IS NOT VALIDATED
   There is zero stock-option history in Trading_data — every backtest in
   this repo is index-only. Nothing here has been tested against real
   fills. It is a blind paper strategy until it has collected its own data.


═══════════════════════════════════════════════════════════════════════════
  ENTRY LOGIC (per stock, evaluated on each closed 3-min bar)
═══════════════════════════════════════════════════════════════════════════
   1. THRUST      close breaks the prior N-bar high (UP) or low (DOWN)
   2. BODY        |close-open| >= body_ratio x (high-low)   — real conviction
   3. VOLUME      bar volume >= vol_mult x 20-bar average   — "good volume"
   4. VWAP        close on the correct side of session VWAP — no counter-trend
   5. RANGE       bar range >= min_range_atr x ATR(14)      — not a dead bar
   -> UP buys ATM CE, DOWN buys ATM PE. Never both legs on one stock.

  Then, once the option leg's first tick arrives (option gates):
   6. PREMIUM     ltp within [prem_min, prem_max]
   7. SPREAD      (ask-bid) <= max(max_spread_pct x ltp, max_spread_abs)
   8. DEPTH       ask_qty >= depth_mult x our quantity
   9. VOLUME/OI   day volume >= min_opt_volume_lots x lot, oi >= min_oi
  10. FEASIBILITY points needed for target_rs_min <= feas_mult x expected
                  option ATR (stock ATR x ~0.5 ATM delta)
  11. COST        sl_points >= min_sl_to_cost_ratio x round-trip cost

═══════════════════════════════════════════════════════════════════════════
  EXIT LADDER
═══════════════════════════════════════════════════════════════════════════
  Evaluated HIGHEST FIRST on every option tick AND on the heartbeat sweep:

   MAX_TARGET   unrealised >= target_rs_max (5000) — hard exit
   SL_HIT       price <= stop, trail not yet armed
   TRAIL_HIT    price <= stop, trail armed
   (ratchet)    trail armed -> stop follows peak by trail_atr_mult x opt ATR
   (arm)        unrealised >= trail_arm_rs (900) -> arm trail, lock 400
   (protect)    unrealised >= target_rs_min (500) -> raise stop, KEEP RUNNING
   EOD          force square-off at close_time

  Rs 500 does NOT close the trade. Set book_at_base_target=True for the old
  hard-exit-at-500 behaviour.

  There is no time stop and no trade-count limit: max_open_positions,
  max_trades_per_stock, max_trades_day and time_stop_min are all None, so
  nothing kills or blocks a trade on age or count. This is a paper strategy
  and a filtered record cannot tell you what the raw signal is worth. The
  stop loss, the Rs 5,000 cap and the EOD square-off are the only exits.

  NOTE ON THE EOD BUG: the square-off runs BEFORE the trading-window guard
  in _heartbeat(). In spike.py and all four candle-breakout files the
  force-exit sat after `if t < start or t > close: return`, so it was dead
  code and positions were never closed (2026-08-26: SPIKE's 4th trade).
"""

import csv
import logging
import os
import statistics
from collections import deque
from datetime import datetime, time as dtime, timedelta, timezone
from typing import Optional

from core.base_strategy import BaseStrategy
from core.costs import (
    effective_spread,
    fixed_costs_rs,
    net_pnl_rs,
    stock_round_trip_cost_pts,
)

log = logging.getLogger("strategy.stock_opt_scanner")

_IST = timezone(timedelta(hours=5, minutes=30))


def _now_ist() -> datetime:
    return datetime.now(tz=_IST).replace(tzinfo=None)


# ── Universe ─────────────────────────────────────────────────────────────────
# 15 most consistently liquid single-stock option chains on NSE. Chosen for
# option-chain depth, not for the stock's cash turnover — a heavily traded
# stock can still have an untradeable option chain. Any name missing from
# the NFO dump on a given day is dropped by StockOptionStore with a warning.
UNIVERSE = [
    "RELIANCE", "HDFCBANK", "ICICIBANK", "SBIN", "INFY",
    "TCS", "AXISBANK", "TATAMOTORS", "TATASTEEL", "BAJFINANCE",
    "KOTAKBANK", "HINDALCO", "MARUTI", "LT", "ADANIENT",
]


CFG = {
    # ── master switch ────────────────────────────────────────────────────────
    "enabled": True,

    # ── session windows (IST) ────────────────────────────────────────────────
    "start_time":      dtime(9, 30),   # skip the first 15 min of open chaos
    "last_entry_time": dtime(14, 45),
    "close_time":      dtime(15, 15),  # force square-off
    # Hour 10 was -20,919 across 132 entries roster-wide in the 56-session
    # study. Kept as a switch rather than a hard rule since that finding is
    # from index strategies, not stocks.
    "avoid_hours":     (),             # e.g. (10,) to re-enable the block

    # ── signal (3-minute stock bars) ─────────────────────────────────────────
    "bar_minutes":      3,
    "min_bars":         8,      # need history before any signal
    "breakout_lookback": 5,     # bars whose high/low must be broken
    "vol_avg_bars":     20,
    "vol_mult":         2.0,    # bar volume vs 20-bar average
    "body_ratio":       0.60,   # |c-o| / (h-l)
    "atr_bars":         14,
    "min_range_atr":    0.80,   # bar range vs ATR — reject dead bars
    "require_vwap_align": True,

    # ── option selection ─────────────────────────────────────────────────────
    "atm_offset_steps": 0,      # 0 = ATM. 1 = one step OTM (cheaper, lower delta)
    "atm_delta":        0.50,   # assumed ATM delta, for the feasibility check
    "opt_tick_wait_s":  20,     # how long to wait for the leg's first tick

    # ── option liquidity gates ───────────────────────────────────────────────
    "prem_min":            15.0,
    "prem_max":            400.0,
    "max_spread_pct":      0.015,   # 1.5% of premium
    "max_spread_abs":      1.00,    # ...or this many points, whichever larger
    "depth_mult":          3.0,     # ask_qty must be this x our qty
    "min_opt_volume_lots": 200,     # day volume, in lots
    "min_oi":              50000,   # contracts

    # ── sizing / targets (rupees) ────────────────────────────────────────────
    "lots":              1,         # always 1 lot
    "target_rs_min":     500.0,     # PROTECT level — stop moves up, trade continues
    "trail_arm_rs":      900.0,     # trail arms here
    "target_rs_max":     5000.0,    # hard cap — never hold past this
    "protect_lock_rs":   150.0,     # profit locked when target_rs_min is reached
    "trail_lock_rs":     400.0,     # minimum profit locked once the trail arms
    "trail_atr_mult":    0.50,      # trail distance = this x expected option ATR
    "min_trail_pts":     0.30,      # ...never tighter than this, or 2x spread
    "book_at_base_target": False,   # True = old behaviour (hard exit at Rs 500)
    "max_loss_rs":       1200.0,    # HARD ceiling on risk, in rupees
    "sl_atr_mult":       0.90,      # SL as a fraction of expected option ATR
    "min_sl_pts":        0.0,       # absolute floor; 0 = let the cost ratio govern
    "feas_mult":         1.20,      # target_pts <= this x expected option ATR
    "min_sl_to_cost_ratio": 3.0,    # SL must be >= 3x the round-trip cost

    # ── position book ────────────────────────────────────────────────────────
    # PAPER DATA COLLECTION: every count-based trade blocker is off. Same
    # decision made for the candle-breakout strategies on 2026-08-12 — a
    # filtered record cannot tell you what the raw signal is worth.
    # None = unlimited. Set a number to re-enable any of these.
    "max_open_positions":   None,
    "max_trades_per_stock": None,
    "max_trades_day":       None,
    "time_stop_min":        None,   # time stop REMOVED — no trade is killed on age
    "stale_price_sec":      45,     # profit decisions ignore prints older than this

    # ── output ───────────────────────────────────────────────────────────────
    "csv_file": "stock_opt_scanner_trades.csv",
}

LIVE_MODE = False   # see design note 2 — do NOT flip without a multi-slot router


class _StockState:
    """Per-underlying rolling state built from NSE equity ticks."""

    __slots__ = (
        "sym", "token", "bars", "cur", "last_cum_vol",
        "vwap_pv", "vwap_v", "trades_today", "last_signal_bar",
    )

    def __init__(self, sym: str, token: int):
        self.sym             = sym
        self.token           = token
        self.bars            = deque(maxlen=60)   # closed bars
        self.cur             = None               # bar under construction
        self.last_cum_vol    = None               # last volume_traded seen
        self.vwap_pv         = 0.0
        self.vwap_v          = 0.0
        self.trades_today    = 0
        self.last_signal_bar = None

    # ── VWAP ─────────────────────────────────────────────────────────────────
    def vwap(self) -> Optional[float]:
        return (self.vwap_pv / self.vwap_v) if self.vwap_v > 0 else None

    # ── indicators over closed bars ──────────────────────────────────────────
    def atr(self, n: int) -> Optional[float]:
        if len(self.bars) < n:
            return None
        rs = [b["h"] - b["l"] for b in list(self.bars)[-n:]]
        return sum(rs) / len(rs) if rs else None

    def avg_volume(self, n: int) -> Optional[float]:
        if len(self.bars) < 3:
            return None
        vols = [b["v"] for b in list(self.bars)[-n:] if b["v"] > 0]
        return (sum(vols) / len(vols)) if vols else None

    def prior_high(self, n: int) -> Optional[float]:
        bl = list(self.bars)[-(n + 1):-1]
        return max(b["h"] for b in bl) if bl else None

    def prior_low(self, n: int) -> Optional[float]:
        bl = list(self.bars)[-(n + 1):-1]
        return min(b["l"] for b in bl) if bl else None


class StockOptionsScannerStrategy(BaseStrategy):
    """
    Multi-stock option buyer. Registered with INDEX_TOKEN = None so it
    receives BankNifty index ticks — used ONLY as a housekeeping heartbeat
    (see _heartbeat). All real work happens in on_option_tick(), which
    receives this strategy's privately-owned tokens.
    """

    INDEX_TOKEN = None
    LIVE_MODE   = LIVE_MODE

    def __init__(self, market_hub):
        super().__init__(market_hub)
        self._store        = None                 # StockOptionStore
        self._stocks       = {}                   # token -> _StockState
        self._by_sym       = {}                   # sym   -> _StockState
        self._positions    = {}                   # opt_token -> trade dict
        self._pending      = {}                   # opt_token -> pending entry dict
        self._completed    = []
        self._today_pnl    = 0.0
        self._trades_today = 0
        self._ready        = False
        self._eod_done     = False
        self._last_hb      = None

    @property
    def name(self) -> str:
        return "STOCK_OPT_SCANNER"

    # ══════════════════════════════════════════════════════════════════════════
    # PRE-MARKET
    # ══════════════════════════════════════════════════════════════════════════

    def pre_market(self, premarket_data, instruments) -> bool:
        """
        `instruments` here must be a StockOptionStore (see t.py wiring), NOT
        the BankNifty InstrumentStore every other strategy receives.
        """
        if not CFG["enabled"]:
            log.info(f"[{self.name}] disabled via CFG — not running today")
            return False

        from core.instruments import StockOptionStore
        if not isinstance(instruments, StockOptionStore):
            log.error(
                f"[{self.name}] pre_market got {type(instruments).__name__}, "
                f"expected StockOptionStore. Check t.py wiring — this strategy "
                f"needs the stock chain, not the BankNifty chain."
            )
            return False

        self._store = instruments
        universe    = self._store.universe
        if not universe:
            log.error(f"[{self.name}] empty universe after instrument load — skipping day")
            return False

        # Subscribe each underlying's NSE equity token and claim exclusive
        # routing, so 14 other strategies do not get a callback per stock tick.
        for sym in universe:
            tok = self._store.spot_token(sym)
            if not tok:
                continue
            st = _StockState(sym, tok)
            self._stocks[tok] = st
            self._by_sym[sym] = st
            self.subscribe_option(tok)
            self._hub.set_token_owner(tok, self.name)

        self._ready = True
        log.info(
            f"[{self.name}] ready | mode={'LIVE' if LIVE_MODE else 'PAPER'} | "
            f"{len(self._stocks)} underlyings | {CFG['bar_minutes']}-min bars | "
            f"target Rs {CFG['target_rs_min']:.0f}-{CFG['target_rs_max']:.0f} | "
            f"max {CFG['max_open_positions']} concurrent"
        )
        for sym in universe:
            log.info(
                f"[{self.name}]   {sym:<12} lot={self._store.lot_size(sym):<5} "
                f"step={self._store.strike_step(sym)} "
                f"exp={self._store.expiry(sym)} dte={self._store.days_to_expiry(sym)}"
            )
        return True

    # ══════════════════════════════════════════════════════════════════════════
    # HEARTBEAT — driven by BankNifty index ticks
    # ══════════════════════════════════════════════════════════════════════════

    def on_tick(self, price: float, ts: datetime, tick_ts: datetime):
        """
        BankNifty index tick. The index price is irrelevant here; this is used
        purely as a reliable clock for time-stops, stale pending entries and
        the EOD square-off. Index ticks arrive several times a second all
        session, including when a given stock has gone quiet.
        """
        if not self._ready:
            return
        # Throttle: housekeeping once a second is plenty, and this callback
        # fires on every BankNifty tick.
        if self._last_hb and (ts - self._last_hb).total_seconds() < 1.0:
            return
        self._last_hb = ts
        self._heartbeat(ts)

    def on_candle(self, candle: dict, ts: datetime):
        """BankNifty 5-min candle — not used. Stock bars are built internally."""
        return

    def _heartbeat(self, ts: datetime):
        t = ts.time()

        # ── EOD square-off FIRST ─────────────────────────────────────────────
        # Deliberately before any window guard. In spike.py and all four
        # candle-breakout files this block sat after `if t > close_time: return`
        # and was therefore unreachable, leaving positions open overnight.
        if t >= CFG["close_time"]:
            if self._positions and not self._eod_done:
                log.info(f"[{self.name}] EOD square-off: {len(self._positions)} open")
                for tok in list(self._positions.keys()):
                    self._exit(tok, "EOD", ts)
                self._eod_done = True
            self._pending.clear()
            return

        # ── expire stale pending entries ─────────────────────────────────────
        for tok in list(self._pending.keys()):
            p = self._pending[tok]
            if (ts - p["ts"]).total_seconds() > CFG["opt_tick_wait_s"]:
                log.info(
                    f"[{self.name}] {p['sym']} {p['opt_symbol']} — no option tick "
                    f"in {CFG['opt_tick_wait_s']}s, abandoning entry (illiquid leg)"
                )
                self._log_signal(ts, p["sym"], "PENDING_TIMEOUT", p["side"],
                                 block="no_option_tick")
                self._drop_pending(tok)

        # ── STOP-LOSS SWEEP (FIX) ────────────────────────────────────────────
        # _manage_position() only runs when the OPTION ticks. Stock options
        # can go minutes between prints, so a leg that stops trading was
        # previously left completely unstopped while the underlying ran
        # against it. The heartbeat rides BankNifty index ticks, which arrive
        # continuously all session, so the stop is now evaluated regardless of
        # whether our own contract has printed.
        for tok in list(self._positions.keys()):
            px = self.get_price(tok)
            if px:
                self._manage_position(tok, px, ts)

        # ── time stop ────────────────────────────────────────────────────────
        # Disabled by default (time_stop_min = None). Age alone says nothing
        # about whether a breakout is going to work; killing a flat trade at
        # 25 minutes was removing exactly the delayed continuations the
        # strategy exists to catch. Set an integer to re-enable.
        if CFG["time_stop_min"]:
            for tok in list(self._positions.keys()):
                tr = self._positions[tok]
                age_min = (ts - tr["entry_ts"]).total_seconds() / 60.0
                if age_min >= CFG["time_stop_min"] and not tr["trail_armed"]:
                    self._exit(tok, "TIME_STOP", ts)

    # ══════════════════════════════════════════════════════════════════════════
    # TICK ROUTING — stock spot ticks and option leg ticks both land here
    # ══════════════════════════════════════════════════════════════════════════

    def on_option_tick(self, token: int, price: float, ts: datetime, tick_ts: datetime = None):
        if not self._ready or not price:
            return

        st = self._stocks.get(token)
        if st is not None:
            self._on_spot_tick(st, price, ts, tick_ts or ts)
            return

        if token in self._positions:
            self._manage_position(token, price, ts)
            return

        if token in self._pending:
            self._try_fill_pending(token, price, ts)

    # ── stock spot ticks → 3-minute bars ─────────────────────────────────────

    def _on_spot_tick(self, st: _StockState, price: float, ts: datetime, tick_ts: datetime):
        # Volume for the bar comes from the DELTA of the cumulative
        # volume_traded field, not last_traded_quantity. last_traded_quantity
        # is the size of one print and misses everything between two ticks —
        # on a stock receiving several trades per tick interval that
        # undercounts badly, and the whole signal rests on the volume ratio.
        cum = self._hub.last_volume(st.token)
        dv  = 0
        if cum:
            if st.last_cum_vol is not None and cum >= st.last_cum_vol:
                dv = cum - st.last_cum_vol
            st.last_cum_vol = cum

        # session VWAP (real volume weighting — unlike the index proxy)
        if dv > 0:
            st.vwap_pv += price * dv
            st.vwap_v  += dv

        bstart = self._bar_start(tick_ts)
        cur    = st.cur

        if cur is None:
            st.cur = {"ts": bstart, "o": price, "h": price, "l": price, "c": price, "v": dv}
            return

        if bstart > cur["ts"]:
            st.bars.append(cur)
            st.cur = {"ts": bstart, "o": price, "h": price, "l": price, "c": price, "v": dv}
            self._evaluate(st, cur, ts)
            return

        cur["h"] = max(cur["h"], price)
        cur["l"] = min(cur["l"], price)
        cur["c"] = price
        cur["v"] += dv

    @staticmethod
    def _bar_start(ts: datetime) -> datetime:
        m = CFG["bar_minutes"]
        return ts.replace(minute=(ts.minute // m) * m, second=0, microsecond=0)

    # ══════════════════════════════════════════════════════════════════════════
    # SIGNAL
    # ══════════════════════════════════════════════════════════════════════════

    def _evaluate(self, st: _StockState, bar: dict, ts: datetime):
        t = ts.time()

        if not (CFG["start_time"] <= t <= CFG["last_entry_time"]):
            return
        if ts.hour in CFG["avoid_hours"]:
            return
        # Count-based blockers are opt-in (all None by default — see CFG).
        # A stock may hold several concurrent positions on DIFFERENT strikes;
        # _arm_entry() still refuses a second position on the same token,
        # because _positions is keyed by token and a duplicate would silently
        # overwrite the first trade's record.
        if CFG["max_trades_day"] and self._trades_today >= CFG["max_trades_day"]:
            return
        if CFG["max_open_positions"] and \
                len(self._positions) + len(self._pending) >= CFG["max_open_positions"]:
            return
        if CFG["max_trades_per_stock"] and st.trades_today >= CFG["max_trades_per_stock"]:
            return
        if len(st.bars) < CFG["min_bars"]:
            return
        if st.last_signal_bar == bar["ts"]:
            return

        rng  = bar["h"] - bar["l"]
        body = abs(bar["c"] - bar["o"])
        if rng <= 0:
            return

        atr     = st.atr(CFG["atr_bars"])
        avg_vol = st.avg_volume(CFG["vol_avg_bars"])
        vwap    = st.vwap()
        if atr is None or not avg_vol:
            return

        vol_ratio  = bar["v"] / avg_vol
        body_ratio = body / rng
        ph = st.prior_high(CFG["breakout_lookback"])
        pl = st.prior_low(CFG["breakout_lookback"])

        side  = None
        block = None

        if bar["c"] > bar["o"] and ph is not None and bar["c"] >= ph:
            side = "UP"
        elif bar["c"] < bar["o"] and pl is not None and bar["c"] <= pl:
            side = "DOWN"
        else:
            return   # no thrust — not even worth logging, this is most bars

        if body_ratio < CFG["body_ratio"]:
            block = f"body_ratio={body_ratio:.2f}"
        elif vol_ratio < CFG["vol_mult"]:
            block = f"vol_ratio={vol_ratio:.2f}"
        elif rng < CFG["min_range_atr"] * atr:
            block = f"range={rng:.2f}<atr_gate={CFG['min_range_atr'] * atr:.2f}"
        elif CFG["require_vwap_align"] and vwap:
            if side == "UP" and bar["c"] < vwap:
                block = "below_vwap"
            elif side == "DOWN" and bar["c"] > vwap:
                block = "above_vwap"

        meta = {
            "close": bar["c"], "body_ratio": body_ratio, "vol_ratio": vol_ratio,
            "atr": atr, "vwap": vwap, "range": rng,
        }

        if block:
            self._log_signal(ts, st.sym, "GATE_BLOCK", side, block=block, **meta)
            return

        st.last_signal_bar = bar["ts"]
        self._log_signal(ts, st.sym, "TRIGGER", side, **meta)
        self._arm_entry(st, side, bar["c"], atr, ts, meta)

    # ══════════════════════════════════════════════════════════════════════════
    # ENTRY — stage 1: resolve and subscribe the leg
    # ══════════════════════════════════════════════════════════════════════════

    def _arm_entry(self, st: _StockState, side: str, spot: float,
                   stock_atr: float, ts: datetime, meta: dict):
        sym      = st.sym
        opt_type = "CE" if side == "UP" else "PE"
        step     = self._store.strike_step(sym)
        atm      = self._store.atm_strike(sym, spot)
        if atm is None:
            return

        # Offset is applied in the OTM direction for whichever leg we buy.
        off    = CFG["atm_offset_steps"] * step
        strike = atm + off if opt_type == "CE" else atm - off

        tok, opt_symbol, lot = self._store.get_option(sym, strike, opt_type)
        if not tok:
            self._log_signal(ts, sym, "NO_CONTRACT", side, block=f"strike={strike}")
            return
        if tok in self._positions or tok in self._pending:
            return

        qty = lot * CFG["lots"]

        # Expected option ATR from the stock ATR via ATM delta. Used by the
        # feasibility check — an option's own ATR is unknown until we have
        # watched it, and we need this decision BEFORE subscribing.
        opt_atr = stock_atr * CFG["atm_delta"]

        self._pending[tok] = {
            "sym": sym, "side": side, "opt_type": opt_type, "strike": strike,
            "opt_symbol": opt_symbol, "lot": lot, "qty": qty,
            "opt_atr": opt_atr, "stock_atr": stock_atr, "spot": spot,
            "ts": ts, "meta": meta,
        }
        self.subscribe_option(tok)
        self._hub.set_token_owner(tok, self.name)

        log.info(
            f"[{self.name}] {sym} {side} thrust @ {spot:.2f} "
            f"(vol x{meta['vol_ratio']:.1f}, body {meta['body_ratio']:.0%}) "
            f"→ arming {opt_symbol} qty={qty}, awaiting first tick"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # ENTRY — stage 2: liquidity + feasibility on the leg's first real tick
    # ══════════════════════════════════════════════════════════════════════════

    def _try_fill_pending(self, tok: int, ltp: float, ts: datetime):
        p   = self._pending[tok]
        sym = p["sym"]
        qty = p["qty"]
        lot = p["lot"]

        bid, ask, bid_qty, ask_qty = self._hub.best_bid_ask(tok)
        volume = self._hub.last_volume(tok)
        oi     = self._hub.last_oi(tok)
        spread = effective_spread(ltp, bid, ask)
        rt_pts = stock_round_trip_cost_pts(ltp, qty, bid, ask)

        # ── liquidity gates ──────────────────────────────────────────────────
        block = None
        if not (CFG["prem_min"] <= ltp <= CFG["prem_max"]):
            block = f"premium={ltp:.2f}"
        elif spread > max(CFG["max_spread_pct"] * ltp, CFG["max_spread_abs"]):
            block = f"spread={spread:.2f}"
        elif ask_qty and ask_qty < CFG["depth_mult"] * qty:
            block = f"ask_qty={ask_qty}<{CFG['depth_mult'] * qty:.0f}"
        elif volume and volume < CFG["min_opt_volume_lots"] * lot:
            block = f"opt_volume={volume}"
        elif oi and oi < CFG["min_oi"]:
            block = f"oi={oi}"

        # ── feasibility: can this contract reach Rs target_rs_min at all? ────
        # The whole point of the strategy. target_pts is what the premium must
        # move for the minimum rupee target; if that exceeds what the option
        # typically moves in a bar or two, the trade is dead on arrival
        # regardless of how good the stock signal was.
        target_pts = CFG["target_rs_min"] / qty + rt_pts
        if block is None and target_pts > CFG["feas_mult"] * p["opt_atr"]:
            block = (
                f"infeasible: need {target_pts:.2f}pts for Rs{CFG['target_rs_min']:.0f} "
                f"(qty={qty}), option ATR≈{p['opt_atr']:.2f}"
            )

        # ── SL sizing and the cost-ratio guard ───────────────────────────────
        # FIX: the old code did `sl_pts = max(sl_pts, min_sl_pts)` with a 2.0
        # floor, which SILENTLY BROKE the rupee risk cap on every large-lot
        # stock. SBIN at qty=1500: max_loss_rs/qty = 0.80pts, the floor forced
        # 2.00pts, and actual risk became Rs 3,000 against a stated cap of
        # Rs 1,200 — 2.5x the intended size, on the biggest lots in the book.
        #
        # The money cap is now absolute. If the room it allows is too tight to
        # survive the round-trip cost, the TRADE IS REJECTED rather than the
        # stop being widened past the cap.
        sl_pts   = min(CFG["max_loss_rs"] / qty, CFG["sl_atr_mult"] * p["opt_atr"])
        required = max(CFG["min_sl_pts"], CFG["min_sl_to_cost_ratio"] * rt_pts)
        if block is None and sl_pts < required:
            block = (
                f"sl={sl_pts:.2f}pts (Rs{sl_pts * qty:.0f}) < required {required:.2f}pts "
                f"[{CFG['min_sl_to_cost_ratio']}x cost {rt_pts:.2f}] — rejected, not widened"
            )

        if block:
            log.info(f"[{self.name}] {sym} {p['opt_symbol']} entry BLOCKED — {block}")
            self._log_signal(ts, sym, "ENTRY_BLOCK", p["side"], block=block,
                             ltp=ltp, spread=spread, oi=oi, opt_volume=volume,
                             rt_cost=rt_pts)
            self._drop_pending(tok)
            return

        # ── place ────────────────────────────────────────────────────────────
        res = self._place_buy(p["opt_symbol"], tok, qty, ltp)
        if res is None:
            log.error(f"[{self.name}] {p['opt_symbol']} BUY failed")
            self._drop_pending(tok)
            return
        order_id, raw_fill = res

        # OrderRouter._paper_fill() returns the raw LTP for both legs — no
        # spread at all. Model the crossing here so paper P&L is comparable
        # to a live curve. (Same fix as FIX B in the candle-breakout files,
        # but with the stock spread model.)
        fill = round(raw_fill + spread / 2.0, 2) if not LIVE_MODE else raw_fill

        tr = {
            "token": tok, "sym": sym, "opt_symbol": p["opt_symbol"],
            "side": p["side"], "opt_type": p["opt_type"], "strike": p["strike"],
            "qty": qty, "lot": lot, "entry": fill, "entry_ts": ts,
            "order_id": order_id, "sl": round(fill - sl_pts, 2),
            "sl_pts": sl_pts, "rt_pts": rt_pts, "spread": spread,
            "peak": fill, "trail_armed": False, "protected": False,
            "opt_atr": p["opt_atr"],
            # ATR-based trail distance, fixed at entry. FIX: the old trail was
            # a flat Rs 700 converted to points at runtime, which made the
            # distance a function of LOT SIZE rather than volatility —
            # 1.40pts on RELIANCE (lot 500) but 0.47pts on SBIN (lot 1500),
            # the latter being inside a normal spread. Now it scales with the
            # contract's own expected movement, with a floor of 2x spread so
            # it can never sit inside the quote.
            "trail_pts": max(
                CFG["trail_atr_mult"] * p["opt_atr"],
                CFG["min_trail_pts"],
                2.0 * spread,
            ),
            "entry_oi": oi, "entry_volume": volume,
        }
        self._positions[tok] = tr
        self._pending.pop(tok, None)
        self._trades_today += 1
        self._by_sym[sym].trades_today += 1

        log.info(
            f"[{self.name}] ENTRY {p['opt_symbol']} @ {fill:.2f} qty={qty} "
            f"SL={tr['sl']:.2f} ({sl_pts:.2f}pts / Rs{sl_pts * qty:.0f}) "
            f"target Rs{CFG['target_rs_min']:.0f} = {target_pts:.2f}pts | "
            f"cost={rt_pts:.2f}pts spread={spread:.2f} | "
            f"open={len(self._positions)}/{CFG['max_open_positions']}"
        )
        self._log_trade(ts, tr, "ENTRY", fill, "OPEN", 0.0, 0.0, "")

    def _drop_pending(self, tok: int):
        self._pending.pop(tok, None)
        self._hub.clear_token_owner(tok)
        self.unsubscribe_option(tok)

    # ══════════════════════════════════════════════════════════════════════════
    # POSITION MANAGEMENT
    # ══════════════════════════════════════════════════════════════════════════

    def _manage_position(self, tok: int, ltp: float, ts: datetime):
        """
        Exit ladder. Called from option ticks AND from the heartbeat sweep.

        ORDER MATTERS — this is the bug that was fixed here. The old version
        checked the Rs 500 base target LAST but the Rs 900 trail-arm BEFORE
        it, in a chain of early returns. A trade climbing normally
        (400 → 500 → 600 → 900) hit the Rs 500 exit at step 5 and was closed
        long before it could ever reach step 3. The trail only armed if a
        SINGLE tick jumped from under 500 to over 900. So the runner
        mechanism the design describes was effectively dead: every winner was
        a Rs 500 winner, and the Rs 5,000 cap was unreachable.

        Now the ladder is evaluated HIGHEST FIRST, and Rs 500 no longer closes
        the trade — it moves the stop up and lets the trade continue:

            >= target_rs_max (5000)  -> hard exit
            price <= stop            -> exit (SL_HIT or TRAIL_HIT)
            trail armed              -> ratchet the ATR trail
            >= trail_arm_rs (900)    -> arm trail, lock trail_lock_rs
            >= target_rs_min (500)   -> lock protect_lock_rs, keep running
                                        (or hard-exit if book_at_base_target)
        """
        tr  = self._positions[tok]
        qty = tr["qty"]

        pts   = self.get_price_ts(tok)
        stale = bool(pts and (ts - pts).total_seconds() > CFG["stale_price_sec"])

        exit_px = round(max(0.05, ltp - tr["spread"] / 2.0), 2)
        unreal  = (exit_px - tr["entry"]) * qty - fixed_costs_rs(qty, tr["entry"], exit_px)

        # ── 1. hard cap ──────────────────────────────────────────────────────
        if unreal >= CFG["target_rs_max"]:
            self._exit(tok, "MAX_TARGET", ts, ltp)
            return

        # ── 2. stop loss — evaluated even on a stale print ───────────────────
        # A leg that has stopped trading is exactly when the stop matters
        # most. Skipping this on staleness is what left positions unstopped.
        if ltp <= tr["sl"]:
            self._exit(tok, "TRAIL_HIT" if tr["trail_armed"] else "SL_HIT", ts, ltp)
            return

        # Everything below is a PROFIT decision, and those should not be made
        # on an old print — a stale high would ratchet the trail to a level
        # the market has already left.
        if stale:
            return

        if ltp > tr["peak"]:
            tr["peak"] = ltp

        # ── 3. ratchet the ATR trail ─────────────────────────────────────────
        if tr["trail_armed"]:
            new_sl = round(tr["peak"] - tr["trail_pts"], 2)
            if new_sl > tr["sl"]:
                tr["sl"] = new_sl
            return

        # ── 4. arm the trail ─────────────────────────────────────────────────
        if unreal >= CFG["trail_arm_rs"]:
            tr["trail_armed"] = True
            lock   = round(tr["entry"] + CFG["trail_lock_rs"] / qty, 2)
            trail  = round(tr["peak"] - tr["trail_pts"], 2)
            tr["sl"] = max(lock, trail, tr["sl"])
            log.info(
                f"[{self.name}] {tr['opt_symbol']} TRAIL ARMED at Rs{unreal:.0f} — "
                f"stop {tr['sl']:.2f}, trailing {tr['trail_pts']:.2f}pts behind peak"
            )
            return

        # ── 5. protect level — move the stop up, do NOT close the trade ──────
        if unreal >= CFG["target_rs_min"]:
            if CFG["book_at_base_target"]:
                self._exit(tok, "TARGET", ts, ltp)
                return
            if not tr["protected"]:
                tr["protected"] = True
                # The lock must clear the round-trip cost, otherwise
                # "locked profit" is a loss once the exit is paid for.
                lock_pts = max(CFG["protect_lock_rs"] / qty, tr["rt_pts"] * 1.10)
                new_sl   = round(tr["entry"] + lock_pts, 2)
                if new_sl > tr["sl"]:
                    tr["sl"] = new_sl
                log.info(
                    f"[{self.name}] {tr['opt_symbol']} PROTECT at Rs{unreal:.0f} — "
                    f"stop raised to {tr['sl']:.2f} (locks ~Rs{lock_pts * qty:.0f}), "
                    f"trade continues toward Rs{CFG['trail_arm_rs']:.0f}"
                )

    def _exit(self, tok: int, reason: str, ts: datetime, ltp: float = None):
        tr = self._positions.get(tok)
        if tr is None:
            return
        qty = tr["qty"]

        # FIX: the old fallback was `self.get_price(tok) or tr["entry"]`.
        # When a leg had gone silent the exit was booked AT THE ENTRY PRICE
        # and logged as a Rs 0 breakeven — corrupting the P&L record on
        # precisely the trades most likely to be losers. The last real print
        # is now used, and the row is flagged so these can be excluded from
        # any expectancy calculation.
        stale_exit = 0
        if ltp is None:
            ltp  = self.get_price(tok)
            pts  = self.get_price_ts(tok)
            if pts and (ts - pts).total_seconds() > CFG["stale_price_sec"]:
                stale_exit = 1
                log.warning(
                    f"[{self.name}] {tr['opt_symbol']} exiting on a STALE price "
                    f"({(ts - pts).total_seconds():.0f}s old) — P&L unreliable"
                )
            if ltp is None:
                ltp = tr["entry"]
                stale_exit = 1
                log.error(
                    f"[{self.name}] {tr['opt_symbol']} NO price available at exit — "
                    f"booking at entry. This row is not real P&L."
                )

        res = self._place_sell_with_retry(tr["opt_symbol"], tok, qty, ltp)
        if res is None:
            log.error(
                f"[{self.name}] EXIT FAILED {tr['opt_symbol']} — "
                f"position may still be open. MANUAL CHECK REQUIRED."
            )
            return
        _, raw_exit = res

        exit_px = round(max(0.05, raw_exit - tr["spread"] / 2.0), 2) if not LIVE_MODE else raw_exit
        gross   = (exit_px - tr["entry"]) * qty
        net     = net_pnl_rs(tr["entry"], exit_px, qty)

        self._today_pnl += net
        tis = (ts - tr["entry_ts"]).total_seconds()

        tr.update({
            "exit_price": exit_px, "exit_reason": reason,
            "pnl": net, "gross_pnl": gross, "time_in_trade_s": tis,
            "stale_exit": stale_exit,
        })
        self._completed.append(tr)
        self._positions.pop(tok, None)

        # Release the token — the refcount leak that FIX D fixed elsewhere.
        self._hub.clear_token_owner(tok)
        self.unsubscribe_option(tok)

        log.info(
            f"[{self.name}] EXIT [{reason}] {tr['opt_symbol']} "
            f"{tr['entry']:.2f} → {exit_px:.2f} | gross={gross:.0f} net={net:.0f} "
            f"| {tis / 60:.1f}min | day={self._today_pnl:.0f} "
            f"| open={len(self._positions)}"
        )
        self._log_trade(ts, tr, "EXIT", exit_px, "CLOSED", net, gross, reason)

    # ══════════════════════════════════════════════════════════════════════════
    # LOGGING
    # ══════════════════════════════════════════════════════════════════════════

    _TRADE_FIELDS = [
        "timestamp", "stock", "symbol", "action", "side", "opt_type", "strike",
        "price", "qty", "lot", "sl", "sl_pts", "status", "pnl", "gross_pnl",
        "cost_drag", "rt_cost_pts", "spread", "reason", "mode", "order_id",
        "time_in_trade_s", "entry_oi", "entry_volume", "opt_atr", "peak",
        "trail_pts", "protected", "trail_armed", "stale_exit",
    ]

    _SIGNAL_FIELDS = [
        "timestamp", "stock", "event", "side", "block_reason",
        "close", "body_ratio", "vol_ratio", "atr", "vwap", "range",
        "ltp", "spread", "oi", "opt_volume", "rt_cost",
    ]

    def _log_trade(self, ts, tr, action, price, status, pnl, gross, reason):
        row = {
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "stock": tr["sym"], "symbol": tr["opt_symbol"], "action": action,
            "side": tr["side"], "opt_type": tr["opt_type"], "strike": tr["strike"],
            "price": round(price, 2), "qty": tr["qty"], "lot": tr["lot"],
            "sl": tr["sl"], "sl_pts": round(tr["sl_pts"], 2),
            "status": status, "pnl": round(pnl, 2), "gross_pnl": round(gross, 2),
            "cost_drag": round(gross - pnl, 2), "rt_cost_pts": tr["rt_pts"],
            "spread": tr["spread"], "reason": reason,
            "mode": "LIVE" if LIVE_MODE else "PAPER", "order_id": tr["order_id"],
            "time_in_trade_s": round(tr.get("time_in_trade_s", 0)),
            "entry_oi": tr.get("entry_oi", ""), "entry_volume": tr.get("entry_volume", ""),
            "opt_atr": round(tr.get("opt_atr", 0), 2), "peak": round(tr.get("peak", 0), 2),
            "trail_pts": round(tr.get("trail_pts", 0), 2),
            "protected": int(tr.get("protected", False)),
            "trail_armed": int(tr.get("trail_armed", False)),
            "stale_exit": tr.get("stale_exit", 0),
        }
        self._write_csv(CFG["csv_file"], self._TRADE_FIELDS, row)

    def _log_signal(self, ts, sym, event, side, block="", **kw):
        row = {
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "stock": sym, "event": event, "side": side, "block_reason": block,
        }
        for k, v in kw.items():
            row[k] = round(v, 4) if isinstance(v, float) else v
        self._write_csv(
            CFG["csv_file"].replace("_trades.csv", "_signals.csv"),
            self._SIGNAL_FIELDS, row,
        )

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
            log.error(f"[STOCK_OPT_SCANNER] CSV write failed ({fname}): {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # EOD
    # ══════════════════════════════════════════════════════════════════════════

    def eod_summary(self):
        log.info(f"\n[{self.name}] {'=' * 50}")
        log.info(f"[{self.name}] END OF DAY | mode={'LIVE' if LIVE_MODE else 'PAPER'}")
        log.info(f"[{self.name}] Trades taken   : {self._trades_today}")

        if self._positions:
            log.error(
                f"[{self.name}] {len(self._positions)} POSITION(S) STILL OPEN "
                f"at EOD summary: {[t['opt_symbol'] for t in self._positions.values()]}"
            )

        wins   = [t for t in self._completed if t["pnl"] > 0]
        losses = [t for t in self._completed if t["pnl"] <= 0]
        gross  = sum(t.get("gross_pnl", 0) for t in self._completed)

        for t in self._completed:
            log.info(
                f"[{self.name}]   {t['sym']:<12} {t['opt_symbol']} [{t['exit_reason']}] "
                f"entry={t['entry']:.2f} exit={t['exit_price']:.2f} "
                f"gross={t.get('gross_pnl', 0):.0f} net={t['pnl']:.0f} "
                f"({t['time_in_trade_s'] / 60:.1f}min)"
            )

        by_reason = {}
        for t in self._completed:
            by_reason.setdefault(t["exit_reason"], []).append(t["pnl"])
        for r, v in sorted(by_reason.items()):
            log.info(f"[{self.name}]   {r:<12} n={len(v):<3} sum={sum(v):.0f} avg={statistics.mean(v):.0f}")

        if self._completed:
            wr = len(wins) / len(self._completed) * 100
            log.info(f"[{self.name}] W/L            : {len(wins)}/{len(losses)}  (win rate {wr:.1f}%)")
        log.info(f"[{self.name}] Gross PnL      : {gross:.0f}")
        log.info(f"[{self.name}] Cost drag      : {gross - self._today_pnl:.0f}")
        log.info(f"[{self.name}] NET PnL        : {self._today_pnl:.0f}")
        log.info(f"[{self.name}] {'=' * 50}\n")
