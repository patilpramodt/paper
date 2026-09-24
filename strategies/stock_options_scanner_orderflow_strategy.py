"""
strategies/stock_options_scanner_orderflow_strategy.py

STOCK_OPT_SCANNER_FLOW — multi-stock option buyer, ORDER-FLOW ENTRY ONLY.

This is a third, separate scanner. V1 (stock_options_scanner_strategy.py,
bar-close breakout) and RT (stock_options_scanner_realtime_strategy.py,
tick-level volume-surge+ROC) are both left unmodified. This file shares
their universe/StockOptionStore plumbing and their proven session/EOD
mechanics, but the ENTRY and EXIT are deliberately stripped to the bare
minimum requested — no other version of this strategy family is this bare.


═══════════════════════════════════════════════════════════════════════════
  ENTRY LOGIC — order-book imbalance + trade-flow imbalance. NOTHING ELSE.
═══════════════════════════════════════════════════════════════════════════
Two measurements, both leading rather than lagging price (see the RT file's
own docstring for that distinction) — neither one waits for price to have
already moved:

  1. BOOK IMBALANCE   Resting bid vs ask quantity, summed across the top
                       depth_levels of the STOCK's own MODE_FULL tick depth
                       (core.market_hub.last_depth()) — recomputed fresh on
                       every tick, no smoothing:
                           book = (bid_qty - ask_qty) / (bid_qty + ask_qty)
                       A skew here is resting orders, not executed trades —
                       it can appear before price actually breaks.

  2. FLOW IMBALANCE    Tick-rule classification of the stock's own executed
                       volume over a rolling flow_window_sec: each print is
                       buyer-initiated if price rose from the last print,
                       seller-initiated if it fell, uncounted if unchanged.
                           flow = (buy_vol - sell_vol) / (buy_vol + sell_vol)
                       This can pick up directional pressure while a stock
                       is still absorbing supply and price hasn't moved.

  ENTRY: both must agree in direction and each cross its own threshold,
  checked fresh on every tick —
      book >=  book_imbalance_threshold  AND  flow >=  flow_imbalance_threshold  -> CE
      book <= -book_imbalance_threshold  AND  flow <= -flow_imbalance_threshold  -> PE

  THAT IS THE ENTIRE SIGNAL. There is no breakout requirement, no VWAP
  alignment, no body/range filter, no premium band, no spread/depth/OI
  liquidity gate, no feasibility check, no cost-ratio SL floor, and no
  confirmation wait — the option leg is bought on its own first tick after
  arming. book_imbalance_threshold and flow_imbalance_threshold are the
  ONLY two numbers governing whether a trade happens.

  Two purely mechanical (not market-condition) exceptions, kept because
  removing them crashes or corrupts the run rather than changing what
  counts as a good trade:
    - a stock already holding an open position or a pending entry is
      skipped (_stock_busy) — a second entry on the same token would
      silently overwrite the first trade's record, same reason V1/RT guard
      on token identity;
    - opt_tick_wait_s abandons a pending entry if the option leg simply
      never prints, so a dead subscription doesn't sit open all day.

  KNOWN CONSEQUENCE OF "NO GATES": there is also no cooldown. If book/flow
  are still skewed the instant a trade exits, this will re-enter the same
  stock immediately. That is a deliberate result of leaving out every gate
  that would normally throttle re-entry, not an oversight.


═══════════════════════════════════════════════════════════════════════════
  EXIT — flat rupees, no ladder, no trailing
═══════════════════════════════════════════════════════════════════════════
   SL_HIT   unrealised <= -max_loss_rs (2000)  -> exit, evaluated even on a
                                                  stale print (same reasoning
                                                  as V1/RT: a leg that has
                                                  stopped trading is exactly
                                                  when the stop matters most)
   TP_HIT   unrealised >=  target_rs   (300)   -> exit
   EOD      force square-off at close_time

  No protect level, no rung, no trail-arm — the position is flat until one
  of the two rupee thresholds is crossed, or the session ends.


═══════════════════════════════════════════════════════════════════════════
  WHAT IS CARRIED OVER FROM V1/RT UNCHANGED (plumbing, not signal logic)
═══════════════════════════════════════════════════════════════════════════
  - Same 15-stock UNIVERSE and StockOptionStore (t.py passes the same
    stock_instruments to all three).
  - Same session window (9:30-14:45 entries, 15:15 EOD) and the same EOD
    square-off-before-window-guard ordering that fixed the dead-code bug in
    spike.py / the candle-breakout files.
  - Same heartbeat stop-loss sweep on the BankNifty clock tick, so a leg
    that goes quiet is still stopped.
  - Same paper-fill spread-crossing model (core.costs.effective_spread) so
    PAPER P&L isn't free of spread cost.
  - PAPER-only for the same reason as V1/RT: OrderRouter's single live slot
    is SPIKE's; this strategy never calls acquire_slot().
"""

import csv
import logging
import os
import statistics
from collections import deque
from datetime import datetime, time as dtime, timedelta, timezone
from typing import Optional

from core.base_strategy import BaseStrategy
from core.costs import effective_spread, fixed_costs_rs, net_pnl_rs

log = logging.getLogger("strategy.stock_opt_scanner_flow")

_IST = timezone(timedelta(hours=5, minutes=30))


def _now_ist() -> datetime:
    return datetime.now(tz=_IST).replace(tzinfo=None)


# ── Universe ─────────────────────────────────────────────────────────────────
# Same 15 names as V1/RT — t.py loads one shared StockOptionStore for all
# three scanners.
UNIVERSE = [
    "RELIANCE", "HDFCBANK", "ICICIBANK", "SBIN", "INFY",
    "TCS", "AXISBANK", "TATAMOTORS", "TATASTEEL", "BAJFINANCE",
    "KOTAKBANK", "HINDALCO", "MARUTI", "LT", "ADANIENT",
]


CFG = {
    # ── master switch ────────────────────────────────────────────────────────
    "enabled": True,

    # ── session windows (IST) ────────────────────────────────────────────────
    "start_time":      dtime(9, 30),
    "last_entry_time": dtime(14, 45),
    "close_time":      dtime(15, 15),   # force square-off

    # ── SIGNAL — the only two knobs that decide whether a trade happens ──────
    "depth_levels":              5,     # levels of last_depth() summed for book imbalance
    "book_imbalance_threshold": 0.65,   # (bid_qty-ask_qty)/(bid_qty+ask_qty), both directions
    "flow_window_sec":          15,     # rolling window for the tick-rule flow ratio
    "flow_imbalance_threshold": 0.65,   # (buy_vol-sell_vol)/(buy_vol+sell_vol)
    "min_flow_ticks":            5,     # min classified ticks in-window before trusting flow

    # ── option selection ─────────────────────────────────────────────────────
    "atm_offset_steps": 0,      # 0 = ATM
    "opt_tick_wait_s":  20,     # abandon a pending entry if the leg never prints (hygiene only)

    # ── sizing / exit (rupees, FLAT) ──────────────────────────────────────────
    "lots":        1,
    "target_rs":   300.0,       # hard TP
    "max_loss_rs": 1900.0,      # hard SL

    # ── position book — no trade-count blockers ──────────────────────────────
    "max_open_positions":   None,
    "max_trades_per_stock": None,
    "max_trades_day":       None,
    "stale_price_sec":      45,

    # ── output ───────────────────────────────────────────────────────────────
    "csv_file": "stock_opt_scanner_flow_trades.csv",
}

LIVE_MODE = False   # PAPER-only — see design note above

# ── entry-confirmation gate ──────────────────────────────────────────────────
# A signal no longer arms an entry immediately. Confirmation is now done on
# freshly-built _CONFIRM_CANDLE_SEC-second candles (own clock, started the
# instant the signal fires — not aligned to the wall-clock second grid)
# rather than on the raw tick stream. Each candle that closes is checked
# against two conditions together:
#   1. COLOR   the candle closed the same color as the signal — green
#              (close > open) for UP, red (close < open) for DOWN
#   2. LEVEL   the candle's close also cleared the reference level set when
#              the signal fired — the price of the very tick at which the
#              decision was taken ("that tick's high/low")
# The first candle to satisfy both confirms the entry. If none does within
# _CONFIRM_TIMEOUT_S seconds (_CONFIRM_TIMEOUT_S / _CONFIRM_CANDLE_SEC
# candles), the signal is dropped and no trade is taken.
_CONFIRM_TIMEOUT_S  = 60
_CONFIRM_CANDLE_SEC = 5


class _StockState:
    """Per-underlying rolling state — just enough for the two imbalance measures."""

    __slots__ = ("sym", "token", "last_price", "last_cum_vol", "flow_ticks")

    def __init__(self, sym: str, token: int):
        self.sym          = sym
        self.token         = token
        self.last_price    = None
        self.last_cum_vol  = None
        # (tick_ts, signed_volume): +dv on an uptick (buyer-initiated),
        # -dv on a downtick (seller-initiated). Unchanged-price ticks are
        # not classified (standard tick-rule convention) and dropped.
        self.flow_ticks    = deque(maxlen=4000)


class StockOptionsScannerOrderflowStrategy(BaseStrategy):
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
        self._stocks        = {}                  # token -> _StockState
        self._by_sym        = {}                  # sym   -> _StockState
        self._positions      = {}                 # opt_token -> trade dict
        self._pending        = {}                 # opt_token -> pending entry dict
        self._confirming     = {}                 # stock token -> confirmation-gate dict
        self._completed     = []
        self._today_pnl     = 0.0
        self._trades_today  = 0
        self._ready          = False
        self._eod_done       = False
        self._last_hb         = None

    @property
    def name(self) -> str:
        return "STOCK_OPT_SCANNER_FLOW"

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
                f"expected StockOptionStore. Check t.py wiring."
            )
            return False

        self._store = instruments
        universe    = self._store.universe
        if not universe:
            log.error(f"[{self.name}] empty universe after instrument load — skipping day")
            return False

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
            f"{len(self._stocks)} underlyings | book>={CFG['book_imbalance_threshold']:.2f} "
            f"AND flow>={CFG['flow_imbalance_threshold']:.2f} | "
            f"TP Rs{CFG['target_rs']:.0f} / SL Rs{CFG['max_loss_rs']:.0f} flat"
        )
        return True

    # ══════════════════════════════════════════════════════════════════════════
    # HEARTBEAT — driven by BankNifty index ticks
    # ══════════════════════════════════════════════════════════════════════════

    def on_tick(self, price: float, ts: datetime, tick_ts: datetime):
        if not self._ready:
            return
        if self._last_hb and (ts - self._last_hb).total_seconds() < 1.0:
            return
        self._last_hb = ts
        self._heartbeat(ts)

    def on_candle(self, candle: dict, ts: datetime):
        return

    def _heartbeat(self, ts: datetime):
        t = ts.time()

        # ── EOD square-off FIRST — same fix as V1/RT (must run before the
        # window guard below, or it's dead code, as it was in spike.py etc.)
        if t >= CFG["close_time"]:
            if self._positions and not self._eod_done:
                log.info(f"[{self.name}] EOD square-off: {len(self._positions)} open")
                for tok in list(self._positions.keys()):
                    self._exit(tok, "EOD", ts)
                self._eod_done = True
            self._pending.clear()
            self._confirming.clear()
            return

        # ── expire stale pending entries ─────────────────────────────────────
        for tok in list(self._pending.keys()):
            p = self._pending[tok]
            if (ts - p["ts"]).total_seconds() > CFG["opt_tick_wait_s"]:
                log.info(
                    f"[{self.name}] {p['sym']} {p['opt_symbol']} — no option tick "
                    f"in {CFG['opt_tick_wait_s']}s, abandoning entry (illiquid leg)"
                )
                self._drop_pending(tok)

        # ── expire stale entry-confirmation waits ────────────────────────────
        for tok in list(self._confirming.keys()):
            if ts >= self._confirming[tok]["deadline_ts"]:
                self._confirm_expire(tok, ts)

        # ── stop-loss sweep — same reasoning as V1/RT: an option leg can go
        # minutes between prints, so ride the BankNifty clock to keep
        # checking the stop even when our own contract hasn't ticked.
        for tok in list(self._positions.keys()):
            px = self.get_price(tok)
            if px:
                self._manage_position(tok, px, ts)

    # ══════════════════════════════════════════════════════════════════════════
    # TICK ROUTING
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

    # ── stock spot ticks → tick-rule flow buffer + signal check ──────────────

    def _on_spot_tick(self, st: _StockState, price: float, ts: datetime, tick_ts: datetime):
        self._check_confirm(st, price, ts)
        cum = self._hub.last_volume(st.token)
        dv  = 0
        if cum:
            if st.last_cum_vol is not None and cum >= st.last_cum_vol:
                dv = cum - st.last_cum_vol
            st.last_cum_vol = cum

        if dv > 0 and st.last_price is not None:
            if price > st.last_price:
                st.flow_ticks.append((tick_ts, dv))     # buyer-initiated
            elif price < st.last_price:
                st.flow_ticks.append((tick_ts, -dv))    # seller-initiated
            # price unchanged: not classified under the tick rule — dropped
        st.last_price = price

        self._evaluate(st, price, ts, tick_ts)

    # ══════════════════════════════════════════════════════════════════════════
    # SIGNAL — book imbalance + flow imbalance, nothing else
    # ══════════════════════════════════════════════════════════════════════════

    def _book_imbalance(self, token: int) -> Optional[float]:
        d = self._hub.last_depth(token)
        if not d:
            return None
        levels  = CFG["depth_levels"]
        bid_qty = sum(int(lvl.get("quantity") or 0) for lvl in (d.get("buy") or [])[:levels])
        ask_qty = sum(int(lvl.get("quantity") or 0) for lvl in (d.get("sell") or [])[:levels])
        total   = bid_qty + ask_qty
        if total <= 0:
            return None
        return (bid_qty - ask_qty) / total

    def _flow_imbalance(self, st: _StockState, tick_ts: datetime) -> Optional[float]:
        cutoff = tick_ts - timedelta(seconds=CFG["flow_window_sec"])
        while st.flow_ticks and st.flow_ticks[0][0] < cutoff:
            st.flow_ticks.popleft()
        if len(st.flow_ticks) < CFG["min_flow_ticks"]:
            return None
        buy  = sum(v for _, v in st.flow_ticks if v > 0)
        sell = -sum(v for _, v in st.flow_ticks if v < 0)
        total = buy + sell
        if total <= 0:
            return None
        return (buy - sell) / total

    def _evaluate(self, st: _StockState, price: float, ts: datetime, tick_ts: datetime):
        t = ts.time()
        if not (CFG["start_time"] <= t <= CFG["last_entry_time"]):
            return
        if CFG["max_trades_day"] and self._trades_today >= CFG["max_trades_day"]:
            return
        if CFG["max_open_positions"] and \
                len(self._positions) + len(self._pending) >= CFG["max_open_positions"]:
            return
        if self._stock_busy(st.sym):
            return

        book = self._book_imbalance(st.token)
        if book is None:
            return
        flow = self._flow_imbalance(st, tick_ts)
        if flow is None:
            return

        side = None
        if book >= CFG["book_imbalance_threshold"] and flow >= CFG["flow_imbalance_threshold"]:
            side = "UP"
        elif book <= -CFG["book_imbalance_threshold"] and flow <= -CFG["flow_imbalance_threshold"]:
            side = "DOWN"
        if side is None:
            return

        meta = {"book": book, "flow": flow}
        self._log_signal(ts, st.sym, "IMBALANCE_TRIGGER", side, **meta)
        log.info(
            f"[{self.name}] {st.sym} {side} imbalance @ {price:.2f} "
            f"book={book:+.2f} flow={flow:+.2f}"
        )
        self._start_confirm(st, side, price, ts, meta)

    def _stock_busy(self, sym: str) -> bool:
        """One live attempt per stock at a time — bookkeeping so a second
        entry on the same token can't silently overwrite the first trade's
        record, not a market-condition filter."""
        for tr in self._positions.values():
            if tr["sym"] == sym:
                return True
        for p in self._pending.values():
            if p["sym"] == sym:
                return True
        for c in self._confirming.values():
            if c["st"].sym == sym:
                return True
        return False

    # ══════════════════════════════════════════════════════════════════════════
    # ENTRY — stage 0: confirmation gate (do not enter immediately)
    # ══════════════════════════════════════════════════════════════════════════

    def _start_confirm(self, st: _StockState, side: str, spot: float, ts: datetime, meta: dict):
        """
        Signal fired, but the entry is not armed yet. `spot` — the price of
        the very tick at which the decision (signal) was taken — is kept as
        the reference ("that tick's high/low"). From this instant,
        _CONFIRM_CANDLE_SEC-second candles are built off the live tick
        stream (own clock — bucket 0 starts exactly at `ts`, not the wall-
        clock second grid). The first such candle that both (a) closes the
        signal's own color and (b) closes beyond `spot` calls _arm_entry().
        If none does within _CONFIRM_TIMEOUT_S seconds, the signal is
        dropped and no trade is taken.
        """
        if st.token in self._confirming:
            return
        self._confirming[st.token] = {
            "st": st, "side": side, "spot": spot,
            "meta": meta, "ref": spot, "start_ts": ts, "cur5": None,
            "deadline_ts": ts + timedelta(seconds=_CONFIRM_TIMEOUT_S),
        }
        log.info(
            f"[{self.name}] {st.sym} {side} signal @ {spot:.2f} — awaiting "
            f"confirmation via {_CONFIRM_CANDLE_SEC}s candles beyond last tick "
            f"{spot:.2f} (up to {_CONFIRM_TIMEOUT_S}s)"
        )

    def _check_confirm(self, st: _StockState, price: float, ts: datetime):
        c = self._confirming.get(st.token)
        if c is None:
            return
        if ts >= c["deadline_ts"]:
            self._confirm_expire(st.token, ts)
            return

        elapsed = (ts - c["start_ts"]).total_seconds()
        bucket  = c["start_ts"] + timedelta(
            seconds=_CONFIRM_CANDLE_SEC * int(elapsed // _CONFIRM_CANDLE_SEC)
        )
        cur5 = c["cur5"]

        if cur5 is None:
            # first tick of this signal's confirmation window — open candle 1
            c["cur5"] = {"bucket": bucket, "o": price, "h": price, "l": price, "c": price}
            return

        if bucket == cur5["bucket"]:
            # still inside the current 5s candle — just update it
            cur5["h"] = max(cur5["h"], price)
            cur5["l"] = min(cur5["l"], price)
            cur5["c"] = price
            return

        # this tick belongs to a new 5s bucket -> the previous candle just closed
        if self._confirm_candle_matches(cur5, c["side"], c["ref"]):
            self._confirming.pop(st.token, None)
            color = "GREEN" if c["side"] == "UP" else "RED"
            log.info(
                f"[{self.name}] {st.sym} {c['side']} confirmed — {color} "
                f"{_CONFIRM_CANDLE_SEC}s candle o={cur5['o']:.2f} h={cur5['h']:.2f} "
                f"l={cur5['l']:.2f} c={cur5['c']:.2f} cleared ref {c['ref']:.2f}"
            )
            self._arm_entry(c["st"], c["side"], cur5["c"], ts, c["meta"])
            return

        # candle closed the wrong color / didn't clear ref — open the next one
        c["cur5"] = {"bucket": bucket, "o": price, "h": price, "l": price, "c": price}

    @staticmethod
    def _confirm_candle_matches(c5: dict, side: str, ref: float) -> bool:
        """
        A _CONFIRM_CANDLE_SEC-second candle confirms the signal only when it
        is BOTH the signal's own color (green close>open for UP, red
        close<open for DOWN — a flat close matches neither) AND its close
        has cleared `ref` in the signal's direction.
        """
        if side == "UP":
            return c5["c"] > c5["o"] and c5["c"] >= ref
        return c5["c"] < c5["o"] and c5["c"] <= ref

    def _confirm_expire(self, tok: int, ts: datetime):
        c = self._confirming.pop(tok, None)
        if c is None:
            return
        log.info(
            f"[{self.name}] {c['st'].sym} {c['side']} confirmation timed out "
            f"({_CONFIRM_TIMEOUT_S}s) — signal dropped, no entry"
        )
        self._log_signal(ts, c["st"].sym, "CONFIRM_TIMEOUT", c["side"], **c["meta"])

    # ══════════════════════════════════════════════════════════════════════════
    # ENTRY — stage 1: resolve and subscribe the leg
    # ══════════════════════════════════════════════════════════════════════════

    def _arm_entry(self, st: _StockState, side: str, spot: float, ts: datetime, meta: dict):
        sym      = st.sym
        opt_type = "CE" if side == "UP" else "PE"
        step     = self._store.strike_step(sym)
        atm      = self._store.atm_strike(sym, spot)
        if atm is None:
            return

        off    = CFG["atm_offset_steps"] * step
        strike = atm + off if opt_type == "CE" else atm - off

        tok, opt_symbol, lot = self._store.get_option(sym, strike, opt_type)
        if not tok:
            return
        if tok in self._positions or tok in self._pending:
            return

        qty = lot * CFG["lots"]
        self._pending[tok] = {
            "sym": sym, "side": side, "opt_type": opt_type, "strike": strike,
            "opt_symbol": opt_symbol, "lot": lot, "qty": qty,
            "ts": ts, "meta": meta,
        }
        self.subscribe_option(tok)
        self._hub.set_token_owner(tok, self.name)

        log.info(
            f"[{self.name}] {sym} {side} imbalance @ {spot:.2f} "
            f"(book={meta['book']:+.2f} flow={meta['flow']:+.2f}) "
            f"→ arming {opt_symbol} qty={qty}, awaiting first tick"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # ENTRY — stage 2: buy on the leg's first tick. NO liquidity/feasibility
    # gate — this is the file's whole point.
    # ══════════════════════════════════════════════════════════════════════════

    def _try_fill_pending(self, tok: int, ltp: float, ts: datetime):
        p   = self._pending[tok]
        sym = p["sym"]
        qty = p["qty"]
        lot = p["lot"]

        bid, ask, _, _ = self._hub.best_bid_ask(tok)
        spread = effective_spread(ltp, bid, ask)

        res = self._place_buy(p["opt_symbol"], tok, qty, ltp)
        if res is None:
            log.error(f"[{self.name}] {p['opt_symbol']} BUY failed")
            self._drop_pending(tok)
            return
        order_id, raw_fill = res

        # Model the spread crossing so PAPER P&L isn't free of it (same
        # accounting fix as V1/RT — this doesn't reject anything, it just
        # makes the simulated fill honest).
        fill = round(raw_fill + spread / 2.0, 2) if not LIVE_MODE else raw_fill

        sl_pts = CFG["max_loss_rs"] / qty

        tr = {
            "token": tok, "sym": sym, "opt_symbol": p["opt_symbol"],
            "side": p["side"], "opt_type": p["opt_type"], "strike": p["strike"],
            "qty": qty, "lot": lot, "entry": fill, "entry_ts": ts,
            "order_id": order_id, "sl": round(fill - sl_pts, 2),
            "sl_pts": sl_pts, "spread": spread,
            "entry_book": p["meta"]["book"], "entry_flow": p["meta"]["flow"],
        }
        self._positions[tok] = tr
        self._pending.pop(tok, None)
        self._trades_today += 1

        log.info(
            f"[{self.name}] ENTRY {p['opt_symbol']} @ {fill:.2f} qty={qty} "
            f"SL={tr['sl']:.2f} ({sl_pts:.2f}pts / Rs{CFG['max_loss_rs']:.0f}) "
            f"TP=+Rs{CFG['target_rs']:.0f} | entry_book={tr['entry_book']:+.2f} "
            f"entry_flow={tr['entry_flow']:+.2f} | open={len(self._positions)}"
        )
        self._log_trade(ts, tr, "ENTRY", fill, "OPEN", 0.0, 0.0, "")

    def _drop_pending(self, tok: int):
        self._pending.pop(tok, None)
        self._hub.clear_token_owner(tok, self.name)
        self.unsubscribe_option(tok)

    # ══════════════════════════════════════════════════════════════════════════
    # POSITION MANAGEMENT — flat TP/SL, no ladder, no trail
    # ══════════════════════════════════════════════════════════════════════════

    def _manage_position(self, tok: int, ltp: float, ts: datetime):
        tr  = self._positions[tok]
        qty = tr["qty"]

        pts   = self.get_price_ts(tok)
        stale = bool(pts and (ts - pts).total_seconds() > CFG["stale_price_sec"])

        exit_px = round(max(0.05, ltp - tr["spread"] / 2.0), 2)
        unreal  = (exit_px - tr["entry"]) * qty - fixed_costs_rs(qty, tr["entry"], exit_px)

        # SL — checked even on a stale print (same reasoning as V1/RT: a
        # leg that has stopped trading is exactly when the stop matters most)
        if ltp <= tr["sl"]:
            self._exit(tok, "SL_HIT", ts, ltp)
            return
        if stale:
            return

        # TP — flat, immediate, no ladder
        if unreal >= CFG["target_rs"]:
            self._exit(tok, "TP_HIT", ts, ltp)
            return

    def _exit(self, tok: int, reason: str, ts: datetime, ltp: float = None):
        tr = self._positions.get(tok)
        if tr is None:
            return
        qty = tr["qty"]

        stale_exit = 0
        if ltp is None:
            ltp = self.get_price(tok)
            pts = self.get_price_ts(tok)
            if pts and (ts - pts).total_seconds() > CFG["stale_price_sec"]:
                stale_exit = 1
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

        self._hub.clear_token_owner(tok, self.name)
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
        "cost_drag", "spread", "reason", "mode", "order_id",
        "time_in_trade_s", "entry_book", "entry_flow", "stale_exit",
    ]

    _SIGNAL_FIELDS = [
        "timestamp", "stock", "event", "side", "book", "flow",
    ]

    def _log_trade(self, ts, tr, action, price, status, pnl, gross, reason):
        row = {
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "stock": tr["sym"], "symbol": tr["opt_symbol"], "action": action,
            "side": tr["side"], "opt_type": tr["opt_type"], "strike": tr["strike"],
            "price": round(price, 2), "qty": tr["qty"], "lot": tr["lot"],
            "sl": tr["sl"], "sl_pts": round(tr["sl_pts"], 2),
            "status": status, "pnl": round(pnl, 2), "gross_pnl": round(gross, 2),
            "cost_drag": round(gross - pnl, 2),
            "spread": tr["spread"], "reason": reason,
            "mode": "LIVE" if LIVE_MODE else "PAPER", "order_id": tr["order_id"],
            "time_in_trade_s": round(tr.get("time_in_trade_s", 0)),
            "entry_book": round(tr.get("entry_book", 0.0), 4),
            "entry_flow": round(tr.get("entry_flow", 0.0), 4),
            "stale_exit": tr.get("stale_exit", 0),
        }
        self._write_csv(CFG["csv_file"], self._TRADE_FIELDS, row)

    def _log_signal(self, ts, sym, event, side, **kw):
        row = {
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "stock": sym, "event": event, "side": side,
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
            log.error(f"[STOCK_OPT_SCANNER_FLOW] CSV write failed ({fname}): {e}")

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

