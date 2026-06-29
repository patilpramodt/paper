"""
strategies/spike_nifty.py

SPIKE_NIFTY Strategy — 9:15 spike trade on Nifty 50, exits by 9:30.
Mirrors SPIKE (BankNifty) but with Nifty-specific parameters.

ENTRY LOGIC (updated):
  No direct gap entry. Regardless of gap direction, wait for the first
  TWO CONSECUTIVE closed 10-second (2-tick) candles after 9:15 that agree
  in direction:
    - Both candles RED   (close < open) → take PE
    - Both candles GREEN (close > open) → take CE
    - Either candle is a doji, or the two disagree → no signal, wait
      for the next pair
  Gap direction is still logged for reference but does NOT trigger entry.

INDEX
──────
  Index          : NSE:NIFTY 50
  INDEX_TOKEN    : 256265  (fixed Zerodha instrument token)
  Expiry         : Weekly  (Nifty 50 retained weekly expiry after SEBI Oct-2024
                            circular; get_nearest_expiry() picks it automatically)
  Strike step    : 50 pts  (BankNifty uses 100)
  Lot size       : 65      (verify before going live — SEBI revises periodically)

SL / TRAILING
──────────────
  Initial SL     : entry − 25 pts
  Trail trigger  : profit >= +25 pts  → trailing activates
  Trail SL       : highest_seen − 12 pts  (i.e. CMP − 12)

MARKET PROTECTION
──────────────────
  20% band — set globally in core/order_router.py as MARKET_PROTECTION = 20.
  Zerodha converts MARKET orders to LIMIT at LTP ± 20% internally.
  No per-strategy override needed.

LIVE / PAPER MODE
─────────────────
  Set LIVE_MODE = True below to enable real orders for this strategy.
  All other strategies remain in paper mode until their own flag is changed.

ORDER EXECUTION (live mode)
────────────────────────────
  Entry : REGULAR + MARKET + MIS via OrderRouter.place_buy()
  Exit  : REGULAR + MARKET + MIS via OrderRouter.place_sell_with_retry()

  NO exchange SL orders are placed after entry.
  Reason: SL-M is not available for options on NSE/Zerodha.
          SL Limit can gap through the trigger entirely.
          We monitor the option price on every WebSocket tick (on_option_tick)
          and fire a MARKET sell the moment the software SL is breached.

TRAILING SL (software, tick-by-tick)
──────────────────────────────────────
  Managed entirely in on_option_tick() on every WebSocket tick.
  trail_trigger_pts : profit needed before trailing starts (25 pts)
  trail_distance    : SL trails this many pts below highest_seen (12 pts)
  Both live and paper mode use identical trail logic.

INDEX ROUTING
──────────────
  INDEX_TOKEN = 256265 tells MarketHub to route Nifty ticks exclusively to
  this strategy's on_tick(). BankNifty ticks (260105) are never delivered here.
  All other strategies (BankNifty) are unaffected.

INSTRUMENTS
────────────
  Nifty 50 options use option_root="NIFTY" in InstrumentStore.
  t.py loads a separate InstrumentStore and PreMarketData for this strategy
  using index_token=256265, and calls pre_market(nifty_pm, nifty_instruments).

BUG FIXES IN THIS VERSION
──────────────────────────
  BUG FIX 1 — NameError: color (CRITICAL — strategy was never trading):
    _check_2candle_signal() used an undefined variable `color` in the
    _build_entry() reason string. This crashed on every valid signal, so
    SPIKE_NIFTY has never executed a trade.
    Fix: replaced with the literal string "2x10s_signal".

  BUG FIX 2 — _do_exit() state guard with _exit_in_progress flag:
    Previously: t["state"] = "CLOSED" was set BEFORE the SELL executed.
    If the SELL failed, state was permanently CLOSED and _do_exit() could
    never be re-entered. Position stayed open in Zerodha with no software
    awareness. Slot was also released unconditionally (BUG FIX 3).
    Fix: state is only set CLOSED after SELL confirm. _exit_in_progress
    prevents duplicate concurrent exit attempts.

  BUG FIX 3 — Slot not released when SELL fails with position still open:
    Slot is now released ONLY when position is confirmed closed. If SELL
    fails with position still open, slot stays LOCKED and emergency exit
    thread takes over.

  BUG FIX 4 — No _exit_in_progress guard in on_option_tick():
    Previously: on_option_tick() SL check had no guard against _exit_in_progress.
    Rapid tick bursts could enter _do_exit() concurrently, placing two sell orders.
    Fix: added _exit_in_progress check before the SL branch (matches spike.py).

  BUG FIX 5 — Emergency exit background thread (ported from spike.py):
    After all 3 SELL retries fail with position confirmed still open,
    a daemon thread retries the SELL every 30 seconds for up to 15 min.
    The slot stays locked throughout.

  BUG FIX 6 — Spike window gate: entries blocked after 9:30:
    If no signal fired in 9:15–9:30 window, _trade_done remained False
    all day, allowing entries at e.g. 11:00 AM.
    Fix: on_tick() sets _trade_done=True when spike_exit_time is reached
    with no open trade.

  BUG FIX 7 — Pending entry overwritten by subsequent candle signal:
    If _pending_entry was set and the next 10s candle fired a new signal,
    _pending_entry was silently overwritten.
    Fix: on_tick() checks _pending_entry is None before calling
    _check_2candle_signal().

  BUG FIX 8 — Pending entry resolved after spike window closes:
    If _pending_entry was set at 9:29:58 and the option tick arrived after
    9:30 (after _trade_done was set True by BUG FIX 6), on_option_tick()
    would still call _build_entry() and enter a post-window position.
    Fix: on_option_tick() pending entry resolution checks not self._trade_done.

  BUG FIX 9 — Missing import time as _time_mod:
    Emergency exit thread uses _time_mod.sleep(). The import was present in
    spike.py but missing here.
    Fix: added "import time as _time_mod".

  BUG FIX 10 — Missing emergency retry config keys:
    CFG was missing "emergency_retry_sec" and "emergency_max_attempts" needed
    by the emergency exit thread.
    Fix: both keys added to CFG.

  BUG FIX 11 — _check_signal was @staticmethod calling SpikeNiftyStrategy
    directly (self-reference on _is_doji). Already correct — retained as-is.

  OTHER (pre-existing fixes retained):
  - pre_market() pre-subscribes ATM CE+PE even when prev_close is None.
  - on_tick() subscribes ATM options on very first market tick if
    pre-subscription failed.
  - Pending entry mechanism for stale/missing pre-9:15 option prices.
  - SL grace period: SL check suppressed for sl_grace_seconds after entry.
  - Unused option leg unsubscribed after entry.
  - SL calculated from actual exchange fill price, not pre-order LTP.
"""

import csv
import logging
import os
import threading
import time as _time_mod          # BUG FIX 9: needed by emergency exit thread
from datetime import datetime, time as dtime, timedelta, timezone
from typing import Optional

from core.base_strategy import BaseStrategy
from core.candle import SecondCandleBuilder

log = logging.getLogger("strategy.spike_nifty")

_IST = timezone(timedelta(hours=5, minutes=30))

def _now_ist() -> datetime:
    return datetime.now(tz=_IST).replace(tzinfo=None)

# ─────────────────────────────────────────────────────────────────────────────
#  LIVE MODE FLAG
#  Change to True when you are ready to trade SPIKE_NIFTY with real money.
#  All other strategies remain in paper mode until their own flag is changed.
# ─────────────────────────────────────────────────────────────────────────────
LIVE_MODE = False

# ─────────────────────────────────────────────────────────────────────────────
#  NIFTY STRIKE STEP
#  Nifty 50 options are quoted at 50-pt intervals. BankNifty uses 100 pts.
# ─────────────────────────────────────────────────────────────────────────────
NIFTY_STRIKE_STEP = 50

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CFG = {
    # ── Lot size ──────────────────────────────────────────────────────────────
    # Nifty 50 lot size. SEBI revises periodically — verify before going live.
    "quantity"               : 65,

    # ── Session windows ───────────────────────────────────────────────────────
    "start_time"             : dtime(9, 15),
    "spike_exit_time"        : dtime(9, 30),
    "close_time"             : dtime(15, 15),

    # ── SL / trailing parameters ──────────────────────────────────────────────
    # initial_sl_buffer : initial SL = fill_price − 25 pts
    # trail_trigger_pts : trail activates when profit >= +25 pts
    # trail_distance    : SL = highest_seen − 12 pts  ("CMP − 12")
    "initial_sl_buffer"      : 25,
    "trail_trigger_pts"      : 25,
    "trail_distance"         : 12,

    # ── Signal filters ────────────────────────────────────────────────────────
    "doji_threshold"         : 0.10,
    "bucket_sec"             : 10,
    "min_candles_before_mom" : 2,

    # ── Output ────────────────────────────────────────────────────────────────
    "csv_file"               : "spike_nifty_trades.csv",

    # ── SL grace period ───────────────────────────────────────────────────────
    # After a live BUY confirm, WebSocket may deliver stale/buffered ticks
    # from the ~15s _confirm_order() polling window. At 9:15 AM these can
    # easily be below the SL. Suppress SL checks for this many seconds.
    "sl_grace_seconds"       : 10,

    # ── Emergency exit (BUG FIX 10: keys were missing) ───────────────────────
    # emergency_retry_sec    : seconds between emergency SELL retry attempts
    # emergency_max_attempts : max retries before giving up (30×30s = 15 min)
    "emergency_retry_sec"    : 30,
    "emergency_max_attempts" : 30,
}


class SpikeNiftyStrategy(BaseStrategy):

    # ── Index routing — MarketHub uses this to deliver Nifty ticks only ───────
    # Zerodha instrument token for NSE:NIFTY 50 index.
    # Strategies without this attribute (or INDEX_TOKEN = None) receive
    # the main BankNifty ticks — they are unaffected by this strategy.
    INDEX_TOKEN = 256265

    # Expose the module-level flag as a class attribute so BaseStrategy works
    LIVE_MODE = LIVE_MODE

    @property
    def name(self) -> str:
        return "SPIKE_NIFTY"

    def __init__(self, market_hub):
        super().__init__(market_hub)

        self._index_8s       = SecondCandleBuilder(seconds=CFG["bucket_sec"])
        self._opt_8s         = None

        self._gap_direction  : Optional[str]  = None
        self._gap_filter_done: bool           = False
        self._market_opened  : bool           = False

        self._pre_ce_token   : Optional[int]  = None
        self._pre_pe_token   : Optional[int]  = None
        self._pre_ce_sym     : Optional[str]  = None
        self._pre_pe_sym     : Optional[str]  = None

        self._trade          = None
        self._trade_done     : bool           = False
        self._today_pnl      : float          = 0.0
        self._completed      : list           = []
        self._pending_entry  = None

        self._prev_body_high    : Optional[float] = None
        self._prev_body_low     : Optional[float] = None
        self._prev_last5m_high  : Optional[float] = None
        self._prev_last5m_low   : Optional[float] = None
        self._prev_last5m_close : Optional[float] = None
        self._expiry_date       = None
        self._instruments       = None

        self._lock = threading.Lock()

        mode_tag = "[LIVE]" if LIVE_MODE else "[PAPER]"
        log.info(
            f"[{self.name}] Initialized {mode_tag} | "
            f"qty={CFG['quantity']} SL=−{CFG['initial_sl_buffer']} "
            f"trail_trigger=+{CFG['trail_trigger_pts']} trail_dist=−{CFG['trail_distance']}"
        )

    # ── Pre-market ────────────────────────────────────────────────────────────

    def pre_market(self, pm, instruments) -> bool:
        """
        Called from t.py with Nifty-specific PreMarketData and InstrumentStore.
          pm          : PreMarketData fetched with index_token=256265 (Nifty 50)
          instruments : InstrumentStore loaded with option_root="NIFTY"
        Nifty has weekly expiry — get_nearest_expiry() picks the nearest weekly.
        """
        from core.instruments import get_atm_strike

        now = _now_ist().time()
        if now >= CFG["spike_exit_time"]:
            log.warning(f"[{self.name}] Started after spike window — skipping today.")
            self._trade_done = True
            return True

        self._instruments       = instruments
        self._prev_body_high    = pm.prev_body_high
        self._prev_body_low     = pm.prev_body_low
        self._prev_last5m_high  = pm.prev_last5m_high
        self._prev_last5m_low   = pm.prev_last5m_low
        self._prev_last5m_close = pm.prev_last5m_close
        self._expiry_date       = pm.expiry_date

        log.info(
            f"[{self.name}] Pre-market | "
            f"body=[{self._prev_body_low} – {self._prev_body_high}] "
            f"prev_close={pm.prev_close} expiry={pm.expiry_date} "
            f"mode={'LIVE' if LIVE_MODE else 'PAPER'}"
        )

        ref_price = pm.prev_close or pm.prev_last5m_close

        if ref_price is None:
            log.warning(
                f"[{self.name}] No prev_close and no prev_last5m_close — "
                f"cannot pre-subscribe options. Will attempt on first tick."
            )
            return True

        strike = get_atm_strike(ref_price, step=NIFTY_STRIKE_STEP)
        log.info(
            f"[{self.name}] Token lookup | strike={strike} "
            f"expiry={pm.expiry_date} ref_price={ref_price:.2f} "
            f"(source={'prev_close' if pm.prev_close else 'prev_last5m_close'})"
        )

        ce_tok, ce_sym = instruments.get_option_token(strike, "CE", pm.expiry_date)
        pe_tok, pe_sym = instruments.get_option_token(strike, "PE", pm.expiry_date)

        self._pre_ce_token = ce_tok
        self._pre_ce_sym   = ce_sym
        self._pre_pe_token = pe_tok
        self._pre_pe_sym   = pe_sym

        if ce_tok:
            self.subscribe_option(ce_tok)
            log.info(f"[{self.name}] Pre-subscribed CE: {ce_sym} ({ce_tok})")
        else:
            log.error(f"[{self.name}] CE token not found | strike={strike} expiry={pm.expiry_date}")

        if pe_tok:
            self.subscribe_option(pe_tok)
            log.info(f"[{self.name}] Pre-subscribed PE: {pe_sym} ({pe_tok})")
        else:
            log.error(f"[{self.name}] PE token not found | strike={strike} expiry={pm.expiry_date}")

        return True

    # ── Tick handlers ─────────────────────────────────────────────────────────

    def on_tick(self, price: float, ts: datetime, tick_ts: datetime):
        """
        Receives Nifty 50 index ticks only.
        MarketHub routes ticks for INDEX_TOKEN=256265 exclusively here.
        BankNifty ticks (260105) are never delivered to this method.

        Entry logic:
          Gap direction is detected on the first tick and logged, but does NOT
          trigger an immediate entry. All entries — gap or no gap — wait for
          the first two consecutive closed 10-second (2-tick) candles after
          9:15 that agree in direction.
            Both candles GREEN → CE
            Both candles RED   → PE
            Doji, or disagreement → skip, wait for next pair
        """
        t = ts.time()

        if t < CFG["start_time"] or t > CFG["close_time"]:
            return

        if not self._market_opened and t >= CFG["start_time"]:
            self._market_opened = True
            log.info(f"[{self.name}] Market open tick received: {price:.2f}")

            if self._pre_ce_token is None or self._pre_pe_token is None:
                self._subscribe_atm_on_open(price)

        closed_8s = self._index_8s.feed_tick(price, tick_ts)

        # Detect gap direction on first tick — for reference / logging only.
        # No direct gap entry is taken.
        if not self._gap_filter_done and self._market_opened:
            self._determine_gap_direction(price)
            self._gap_filter_done = True

        # BUG FIX 6: Close spike window when time passes 9:30 with no trade.
        # Without this guard, _check_2candle_signal is called all day until
        # 15:15, and any signal at e.g. 11:00 AM would trigger a real entry.
        if not self._trade_done and self._trade is None and t >= CFG["spike_exit_time"]:
            log.info(f"[{self.name}] Spike window closed (9:30) with no trade — done for today.")
            self._trade_done = True

        # BUG FIX 7: Guard _pending_entry so a second candle signal cannot
        # overwrite a pending entry that has not been resolved yet.
        # All entries (gap or no gap) go through the first 2-tick candle signal.
        if (not self._trade_done and
                self._trade is None and
                self._pending_entry is None and   # BUG FIX 7
                closed_8s is not None):
            self._check_2candle_signal(closed_8s, price, ts)

        # BUG FIX 2 + BUG FIX 4: only attempt time-exit if no exit already in progress
        if (self._trade and
                self._trade["state"] == "OPEN" and
                not self._trade.get("_exit_in_progress")):   # BUG FIX 4
            if t >= CFG["spike_exit_time"]:
                opt_price = self.get_price(self._trade["token"]) or self._trade["entry"]
                self._do_exit(opt_price, "SPIKE_WINDOW_END", ts)

    def on_candle(self, candle: dict, ts: datetime):
        pass

    def on_option_tick(self, token: int, price: float, ts: datetime, tick_ts: datetime = None):
        """
        Option price arrives on every WebSocket tick.

        Job 1 — Pending entry resolution:
          If option had no valid post-9:15 price at signal time, a pending
          entry was stored. The first live tick here resolves it.

        Job 2 — Software trailing SL (runs on every tick while trade is open):
          Trail activates when profit >= trail_trigger_pts (25 pts).
          New SL = highest_seen − trail_distance (12 pts).
          SL only moves up, never down.

        Job 3 — SL grace period:
          SL checks are suppressed for sl_grace_seconds after entry fill to
          prevent stale buffered ticks from causing an immediate false exit.

        BUG FIX 4 — _exit_in_progress guard:
          When _do_exit() is executing (or emergency exit thread is running),
          _exit_in_progress is True. on_option_tick() skips SL checks entirely
          to prevent concurrent duplicate exit attempts from rapid tick delivery.

        BUG FIX 8 — Pending entry not resolved after spike window closes:
          Also checks not self._trade_done so that a pending entry set at
          9:29:58 is not entered after the window has already closed.
        """
        # ── Job 1: Resolve pending entry ──────────────────────────────────────
        # BUG FIX 8: added "not self._trade_done" to block post-window resolution
        if (self._pending_entry and
                token == self._pending_entry["token"] and
                not self._trade and
                not self._trade_done):             # BUG FIX 8
            p = self._pending_entry
            self._pending_entry = None
            log.info(
                f"[{self.name}] Pending entry resolved — first live tick for {p['sym']} "
                f"@ {price:.2f} (was stale at entry signal time)"
            )
            self._build_entry(p["sym"], p["token"], p["signal"], ts, p["reason"])
            return

        # ── Job 2 & 3: Trailing SL management ────────────────────────────────
        if not (self._trade and token == self._trade.get("token")):
            return

        if self._opt_8s is None:
            self._opt_8s = SecondCandleBuilder(seconds=CFG["bucket_sec"])

        use_ts = tick_ts if tick_ts else ts
        self._opt_8s.feed_tick(price, use_ts)

        # BUG FIX 4: skip all SL logic when exit is already being handled
        if self._trade["state"] != "OPEN" or self._trade.get("_exit_in_progress"):
            return

        # Update highest seen
        if price > self._trade["highest_seen"]:
            self._trade["highest_seen"] = price

        # Compute new trailing SL
        new_sl = self._compute_trailing_sl(
            self._trade["entry"], self._trade["highest_seen"], self._trade["sl"]
        )
        if new_sl > self._trade["sl"]:
            log.info(
                f"[{self.name}] TSL updated: {self._trade['sl']:.0f} → {new_sl:.0f} "
                f"(highest={self._trade['highest_seen']:.0f}, dist=−{CFG['trail_distance']})"
            )
            self._trade["sl"] = new_sl

        # SL grace period check
        sl_active_from = self._trade.get("sl_active_from")
        if sl_active_from is not None and ts < sl_active_from:
            log.debug(
                f"[{self.name}] SL grace active — skipping check "
                f"(ts={ts.strftime('%H:%M:%S')} < until={sl_active_from.strftime('%H:%M:%S')}) "
                f"price={price:.0f} sl={self._trade['sl']:.0f}"
            )
            return

        # SL breached — MARKET exit immediately
        if price <= self._trade["sl"]:
            self._do_exit(price, "SL_HIT", ts)

    # ── Subscribe ATM on open (fallback) ──────────────────────────────────────

    def _subscribe_atm_on_open(self, open_price: float):
        """Late-subscribe ATM options if pre-market subscription failed."""
        from core.instruments import get_atm_strike
        strike = get_atm_strike(open_price, step=NIFTY_STRIKE_STEP)
        expiry = self._expiry_date
        log.info(
            f"[{self.name}] Late-subscribing ATM options on open tick | "
            f"strike={strike} expiry={expiry} open={open_price:.2f}"
        )

        ce_tok, ce_sym = self._instruments.get_option_token(strike, "CE", expiry)
        pe_tok, pe_sym = self._instruments.get_option_token(strike, "PE", expiry)

        if ce_tok and self._pre_ce_token is None:
            self.subscribe_option(ce_tok)
            self._pre_ce_token = ce_tok
            self._pre_ce_sym   = ce_sym
            log.info(f"[{self.name}] Late-subscribed CE: {ce_sym} ({ce_tok})")

        if pe_tok and self._pre_pe_token is None:
            self.subscribe_option(pe_tok)
            self._pre_pe_token = pe_tok
            self._pre_pe_sym   = pe_sym
            log.info(f"[{self.name}] Late-subscribed PE: {pe_sym} ({pe_tok})")

    # ── Gap direction (reference only — does not trigger entry) ───────────────

    def _determine_gap_direction(self, open_price: float):
        """
        Classify today's open vs prev day's last 5-min range (or body).
        Result is stored in self._gap_direction for logging/CSV reference only.
        It no longer controls which option is entered — the 2-tick candle does.
        """
        h5 = self._prev_last5m_high
        l5 = self._prev_last5m_low

        if h5 is not None and l5 is not None:
            log.info(
                f"[{self.name}] Gap ref → prev last 5-min: "
                f"H={h5:.0f}  L={l5:.0f}  Today open={open_price:.0f}"
            )
            if open_price > h5:
                self._gap_direction = "CE"
                log.info(f"[{self.name}]  GAP UP: open={open_price:.0f} > last5m_high={h5:.0f}  (ref only)")
            elif open_price < l5:
                self._gap_direction = "PE"
                log.info(f"[{self.name}]  GAP DOWN: open={open_price:.0f} < last5m_low={l5:.0f}  (ref only)")
            else:
                self._gap_direction = "BOTH"
                log.info(f"[{self.name}]  NO GAP: open inside [{l5:.0f}–{h5:.0f}] (ref only)")
            return

        log.warning(f"[{self.name}] Last 5-min candle data unavailable — falling back to daily body")
        bh, bl = self._prev_body_high, self._prev_body_low

        if bh is None or bl is None:
            self._gap_direction = "BOTH"
            log.warning(f"[{self.name}] No gap reference at all — defaulting to BOTH")
            return

        if open_price > bh:
            self._gap_direction = "CE"
            log.info(f"[{self.name}]  GAP UP (fallback): open={open_price:.0f} > body_high={bh:.0f}  (ref only)")
        elif open_price < bl:
            self._gap_direction = "PE"
            log.info(f"[{self.name}]  GAP DOWN (fallback): open={open_price:.0f} < body_low={bl:.0f}  (ref only)")
        else:
            self._gap_direction = "BOTH"
            log.info(f"[{self.name}]  NO GAP (fallback): open inside [{bl:.0f}–{bh:.0f}]")

    # ── 2x10s candle signal ─────────────────────────────────────────────────────

    def _check_2candle_signal(self, latest_closed: dict, index_price: float, ts: datetime):
        """
        Entry signal based on the last TWO closed 10-second (2-tick) candles
        after 9:15 — they must agree in direction:
          - Both candles GREEN (close > open) → take CE
          - Both candles RED   (close < open) → take PE
          - Either candle is a doji, or they disagree → no signal, wait
            for the next pair

        This logic applies regardless of gap direction.
        Gap direction is ignored for the entry decision.
        """
        last_two = self._index_8s.last_n_closed(2)
        if len(last_two) < 2:
            return

        c1, c2 = last_two[-2], last_two[-1]
        signal = self._check_signal(c1, c2)
        if not signal:
            return

        log.info(
            f"[{self.name}] 2x10s candle signal: {signal} "
            f"at {ts.strftime('%H:%M:%S')} "
            f"(c1 o={c1['open']:.1f} c={c1['close']:.1f} | "
            f"c2 o={c2['open']:.1f} c={c2['close']:.1f})"
        )

        if signal == "CE" and self._pre_ce_token:
            sym, token = self._pre_ce_sym, self._pre_ce_token
        elif signal == "PE" and self._pre_pe_token:
            sym, token = self._pre_pe_sym, self._pre_pe_token
        else:
            from core.instruments import get_atm_strike
            strike = get_atm_strike(index_price, step=NIFTY_STRIKE_STEP)
            expiry = self._expiry_date
            log.info(
                f"[{self.name}] Fallback token lookup | signal={signal} "
                f"strike={strike} expiry={expiry}"
            )
            token, sym = self._instruments.get_option_token(strike, signal, expiry)

        if not token or not sym:
            log.error(f"[{self.name}] No token for {signal} — trade SKIPPED")
            return

        self.subscribe_option(token)
        # BUG FIX 1: was `f"2tick_candle_{color.lower()}"` — `color` was never defined.
        self._build_entry(sym, token, signal, ts, reason="2x10s_signal")

    # ── Entry ─────────────────────────────────────────────────────────────────

    def _build_entry(self, sym: str, token: int, signal: str, ts: datetime, reason: str):
        """
        Attempt to enter a position.

        Price guard: if the option has no valid post-9:15 price yet, store a
        pending entry and return — on_option_tick() resolves it on first tick.

        SL is set from the actual exchange fill price (not pre-order LTP), so
        it's exactly initial_sl_buffer (25 pts) below the confirmed fill.

        Market protection (20%) is handled globally by order_router.py.

        SL grace period: sl_active_from = fill_time + sl_grace_seconds.
        on_option_tick() will not check SL until this time has passed.
        """
        opt_price = self.get_price(token)
        price_ts  = self.get_price_ts(token)
        market_open_today = ts.replace(hour=9, minute=15, second=0, microsecond=0)

        # Price guard — no valid post-9:15 tick yet
        if (not opt_price or opt_price <= 0) or \
           (price_ts is None or price_ts < market_open_today):
            log.warning(
                f"[{self.name}] No valid post-9:15 price for {sym} "
                f"(price={opt_price} priced_at={price_ts}) — storing pending entry"
            )
            self._pending_entry = {
                "sym": sym, "token": token, "signal": signal, "ts": ts, "reason": reason
            }
            return

        # Acquire live trade slot (paper mode always succeeds)
        if not self._acquire_slot():
            log.warning(f"[{self.name}] Trade slot blocked — another live strategy has a position")
            return

        # Place BUY — returns (order_id, fill_price) or None on failure
        result = self._place_buy(sym, token, CFG["quantity"], opt_price)
        if result is None:
            self._release_slot()
            log.error(f"[{self.name}] BUY order FAILED for {sym} — entry aborted")
            return

        order_id, fill_price = result

        log.info(
            f"[{self.name}] BUY confirmed | pre_ltp={opt_price:.2f} "
            f"fill_price={fill_price:.2f} | slippage={fill_price - opt_price:+.2f}"
        )

        # SL from actual fill price — not pre-order LTP
        sl             = fill_price - CFG["initial_sl_buffer"]
        sl_active_from = ts + timedelta(seconds=CFG["sl_grace_seconds"])

        log.info(
            f"[{self.name}] SL grace until {sl_active_from.strftime('%H:%M:%S')} "
            f"({CFG['sl_grace_seconds']}s after fill)"
        )

        self._opt_8s = SecondCandleBuilder(seconds=CFG["bucket_sec"])
        self._trade = {
            "state"            : "OPEN",
            "symbol"           : sym,
            "token"            : token,
            "signal"           : signal,
            "entry"            : fill_price,
            "sl"               : sl,
            "highest_seen"     : fill_price,
            "entry_time"       : ts,
            "sl_active_from"   : sl_active_from,
            "gap_direction"    : self._gap_direction,
            "order_id"         : order_id,
            "qty"              : CFG["quantity"],
            # BUG FIX 2: exit guard flag — was absent in original spike_nifty.py
            "_exit_in_progress": False,
        }

        # Unsubscribe the unused option leg to prevent stale ticks
        if signal == "CE" and self._pre_pe_token:
            self.unsubscribe_option(self._pre_pe_token)
            log.info(f"[{self.name}] Unsubscribed unused PE: {self._pre_pe_sym} ({self._pre_pe_token})")
        elif signal == "PE" and self._pre_ce_token:
            self.unsubscribe_option(self._pre_ce_token)
            log.info(f"[{self.name}] Unsubscribed unused CE: {self._pre_ce_sym} ({self._pre_ce_token})")

        mode_tag = "LIVE" if LIVE_MODE else "PAPER"
        log.info(
            f"[{self.name}] [{mode_tag}] ENTRY {sym} @ {fill_price:.0f} | "
            f"SL={sl:.0f} (−{CFG['initial_sl_buffer']}) | "
            f"Trail starts @ {fill_price + CFG['trail_trigger_pts']:.0f} "
            f"(+{CFG['trail_trigger_pts']}) → SL = CMP−{CFG['trail_distance']} | "
            f"Reason={reason} | order_id={order_id}"
        )

        self._log_csv({
            "timestamp"    : ts.strftime("%Y-%m-%d %H:%M:%S"),
            "symbol"       : sym,
            "action"       : "ENTRY",
            "price"        : fill_price,
            "sl"           : sl,
            "status"       : "OPEN",
            "pnl"          : 0,
            "reason"       : reason,
            "gap_direction": self._gap_direction,
            "mode"         : mode_tag,
            "order_id"     : order_id,
        })

    # ── Exit ──────────────────────────────────────────────────────────────────

    def _do_exit(self, exit_price: float, reason: str, ts: datetime):
        """
        Close the open position.

        BUG FIX 2 — _exit_in_progress guard:
          Set _exit_in_progress = True at the top so concurrent on_option_tick
          calls (WebSocket tick bursts) cannot enter here simultaneously.
          state is only set to "CLOSED" AFTER the SELL is confirmed —
          not before (the original bug). This preserves re-entry ability if
          SELL fails transiently.

        BUG FIX 3 — Slot management:
          Slot is released ONLY when the position is confirmed closed.
          If SELL fails with position still open, slot stays LOCKED and
          an emergency exit thread takes over. This prevents another live
          strategy from entering while the position is unresolved.

        Flow:
          success  → state=CLOSED, release_slot, _finalize_exit
          fail + position closed by exchange → state=CLOSED,
                  release_slot, _finalize_exit (with AUTO_CLOSED reason)
          fail + position still open → _exit_in_progress stays True,
                  slot LOCKED, _start_emergency_exit() thread launched
        """
        t = self._trade
        if not t or t["state"] != "OPEN":
            return
        # BUG FIX 2: prevent duplicate concurrent exit attempts
        if t.get("_exit_in_progress"):
            return
        t["_exit_in_progress"] = True

        result = self._place_sell_with_retry(
            t["symbol"], t["token"], t["qty"], exit_price, max_retries=3
        )

        if result is not None:
            # ── SUCCESS: SELL confirmed ────────────────────────────────────────
            order_id, sell_price = result
            t["state"] = "CLOSED"       # BUG FIX 2: only set CLOSED after confirm
            self._release_slot()        # BUG FIX 3: released only on success
            self._finalize_exit(t, sell_price, order_id, reason, ts)
            return

        # ── ALL 3 RETRIES FAILED ───────────────────────────────────────────────
        if LIVE_MODE:
            still_open = self._hub.order_router._is_position_open(t["symbol"])

            if not still_open:
                # Position closed by exchange (auto square-off or phantom fill)
                log.warning(
                    f"[{self.name}] SELL failed after 3 retries but position confirmed "
                    f"CLOSED by exchange (auto square-off or phantom fill). "
                    f"Treating as closed."
                )
                t["state"] = "CLOSED"
                self._release_slot()
                self._finalize_exit(t, exit_price, None, f"{reason}_EXCHANGE_CLOSED", ts)
                return

            # Position confirmed still OPEN — keep slot locked, start emergency thread
            log.error(
                f"\n{'!'*60}\n"
                f"[{self.name}] CRITICAL: SELL failed after 3 retries — "
                f"position STILL OPEN for {t['symbol']}!\n"
                f"  Slot is LOCKED — no other strategy can enter.\n"
                f"  Emergency exit thread starting (retry every "
                f"{CFG['emergency_retry_sec']}s for up to "
                f"{CFG['emergency_max_attempts'] * CFG['emergency_retry_sec'] // 60} min).\n"
                f"  *** CHECK ZERODHA CONSOLE IF EMERGENCY EXIT ALSO FAILS! ***\n"
                f"{'!'*60}"
            )
            # _exit_in_progress stays True — prevents on_option_tick re-entering.
            # Emergency thread is now the sole manager of this position.
            self._start_emergency_exit(t, exit_price, reason, ts)
            return

        # Paper mode: paper sell never fails, but handle gracefully
        t["state"] = "CLOSED"
        self._release_slot()
        self._finalize_exit(t, exit_price, None, reason, ts)

    # ── Emergency exit (background thread) ───────────────────────────────────

    def _start_emergency_exit(
        self, t: dict, ref_price: float, reason: str, ts: datetime
    ):
        """
        BUG FIX 5: Launch a daemon thread that retries SELL every
        emergency_retry_sec seconds for up to emergency_max_attempts attempts.

        The global trade slot stays LOCKED throughout to prevent other
        live strategies from entering while this position is unresolved.

        Terminates when:
          a) SELL succeeds       → finalize, release slot
          b) Position confirmed closed by exchange → finalize, release slot
          c) Max attempts reached → log CRITICAL, force-release slot

        The main WebSocket loop is unaffected — this runs independently.
        """
        def _loop():
            max_attempts = CFG["emergency_max_attempts"]
            retry_sec    = CFG["emergency_retry_sec"]

            for attempt in range(1, max_attempts + 1):
                _time_mod.sleep(retry_sec)

                log.error(
                    f"[{self.name}] Emergency exit attempt {attempt}/{max_attempts} | "
                    f"{t['symbol']} | slot LOCKED"
                )

                # Check if exchange already closed the position
                still_open = self._hub.order_router._is_position_open(t["symbol"])
                if not still_open:
                    log.info(
                        f"[{self.name}] Emergency exit: {t['symbol']} confirmed CLOSED "
                        f"by exchange on attempt {attempt}"
                    )
                    t["state"] = "CLOSED"
                    self._release_slot()
                    now = _now_ist()
                    self._finalize_exit(t, ref_price, None,
                                        f"{reason}_EXCHANGE_CLOSED", now)
                    return

                # Refresh LTP and try SELL
                ltp = self.get_price(t["token"]) or ref_price
                result = self._hub.order_router.place_sell(
                    self.name, t["symbol"], t["token"], t["qty"], ltp, LIVE_MODE
                )

                if result:
                    order_id, sell_price = result
                    t["state"] = "CLOSED"
                    self._release_slot()
                    log.info(
                        f"[{self.name}] Emergency exit SUCCESS on attempt {attempt} "
                        f"@ {sell_price:.0f}"
                    )
                    now = _now_ist()
                    self._finalize_exit(t, sell_price, order_id,
                                        f"{reason}_EMERGENCY", now)
                    return

                log.error(
                    f"[{self.name}] Emergency exit attempt {attempt}/{max_attempts} FAILED | "
                    f"{t['symbol']} | ltp={ltp:.0f}"
                )

            # All attempts exhausted
            log.error(
                f"\n{'!'*60}\n"
                f"[{self.name}] GAVE UP emergency exit for {t['symbol']} after "
                f"{max_attempts} attempts ({max_attempts * retry_sec // 60} min).\n"
                f"  *** SQUARE OFF MANUALLY IN ZERODHA CONSOLE IMMEDIATELY! ***\n"
                f"  Force-releasing slot to prevent indefinite lock.\n"
                f"{'!'*60}"
            )
            t["state"] = "CLOSED"
            self._release_slot()
            self._trade_done = True

        thread = threading.Thread(
            target=_loop,
            name="spike-nifty-emergency-exit",
            daemon=True,
        )
        thread.start()
        log.info(f"[{self.name}] Emergency exit thread started for {t['symbol']}")

    # ── Exit finalize (shared by normal + emergency paths) ────────────────────

    def _finalize_exit(
        self,
        t:          dict,
        sell_price: float,
        order_id:   Optional[str],
        reason:     str,
        ts:         datetime,
    ):
        """
        Record PnL, log, write CSV. Called after position is confirmed closed
        by either the normal exit path or the emergency exit thread.
        """
        pnl = (sell_price - t["entry"]) * t["qty"]
        self._today_pnl += pnl
        self._trade_done = True

        mode_tag = "LIVE" if LIVE_MODE else "PAPER"
        log.info(
            f"[{self.name}] [{mode_tag}] EXIT [{reason}] {t['symbol']} @ {sell_price:.0f} "
            f"| PnL={pnl:.0f} ({pnl / t['qty']:.1f}/unit) "
            f"| Today={self._today_pnl:.0f} | order_id={order_id}"
        )

        self._log_csv({
            "timestamp"    : ts.strftime("%Y-%m-%d %H:%M:%S"),
            "symbol"       : t["symbol"],
            "action"       : "EXIT",
            "price"        : sell_price,
            "sl"           : t["sl"],
            "status"       : "CLOSED",
            "pnl"          : round(pnl, 2),
            "reason"       : reason,
            "gap_direction": t["gap_direction"],
            "mode"         : mode_tag,
            "order_id"     : order_id,
        })
        self._completed.append({
            **t,
            "exit_price"  : sell_price,
            "exit_reason" : reason,
            "pnl"         : pnl,
        })

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _check_signal(c1: dict, c2: dict) -> Optional[str]:
        """
        Two consecutive 10-second candles must agree in direction:
          both GREEN (close > open) → CE
          both RED   (close < open) → PE
          either is a doji, or they disagree → None (no signal)
        """
        if SpikeNiftyStrategy._is_doji(c1) or SpikeNiftyStrategy._is_doji(c2):
            return None
        if c1["close"] > c1["open"] and c2["close"] > c2["open"]:
            return "CE"
        if c1["close"] < c1["open"] and c2["close"] < c2["open"]:
            return "PE"
        return None

    @staticmethod
    def _is_doji(c: dict) -> bool:
        rng = c["high"] - c["low"]
        if rng == 0:
            return True
        return abs(c["close"] - c["open"]) / rng < CFG["doji_threshold"]

    @staticmethod
    def _compute_trailing_sl(entry: float, highest: float, current_sl: float) -> float:
        """
        Trail SL logic:
          - No movement until profit >= trail_trigger_pts (25 pts)
          - Once active: SL = highest_seen − trail_distance (12 pts)
          - SL only moves up — never reverts if price dips
        """
        profit = highest - entry
        if profit >= CFG["trail_trigger_pts"]:
            new_sl = highest - CFG["trail_distance"]
            return max(new_sl, current_sl)
        return current_sl

    def _log_csv(self, row: dict):
        fname  = CFG["csv_file"]
        exists = os.path.isfile(fname)
        fields = [
            "timestamp", "symbol", "action", "price",
            "sl", "status", "pnl", "reason", "gap_direction", "mode", "order_id",
        ]
        with open(fname, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if not exists:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in fields})

    def eod_summary(self):
        log.info(f"\n[{self.name}] {'='*50}")
        log.info(f"[{self.name}] END OF DAY | mode={'LIVE' if LIVE_MODE else 'PAPER'}")
        log.info(f"[{self.name}] Gap direction  : {self._gap_direction}")
        log.info(f"[{self.name}] Trade executed : {'Yes' if self._trade_done else 'No'}")
        for t in self._completed:
            log.info(
                f"[{self.name}]   {t['symbol']} [{t['exit_reason']}] "
                f"entry={t['entry']:.0f} exit={t['exit_price']:.0f} "
                f"PnL={t['pnl']:.0f} ({t['pnl'] / t['qty']:.1f}/unit)"
            )
        log.info(f"[{self.name}] Today PnL      : {self._today_pnl:.0f}")
        log.info(f"[{self.name}] {'='*50}\n")

