"""
strategies/spike.py

SPIKE Strategy — 9:15 spike trade on BankNifty, exits by 9:30.

ENTRY LOGIC:
  No direct gap entry. Wait for the first two consecutive closed 10-second
  (2-tick) candles after 9:15 that agree in direction:
    - Both candles GREEN (close > open) → take CE
    - Both candles RED   (close < open) → take PE
    - Either candle is a doji, or the two disagree → no signal, wait for
      the next pair
  Gap direction (today's open vs. prev session's last 5-min range) is
  still computed and logged/CSV'd for reference, but does NOT trigger
  entry on its own.

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
  trail_trigger_pts : profit needed before trailing starts
  trail_distance    : SL trails this many pts below highest_seen
  Both live and paper mode use identical trail logic.

BUG FIXES IN THIS VERSION
──────────────────────────
  BUG FIX 1 (order_router.py) — Zero-fill detection:
    _fetch_fill_price() now checks filled_quantity > 0 after COMPLETE.
    If filled_quantity=0 (phantom fill at 9:15 AM when options have no
    market maker), returns None → entry aborted → no phantom position.

  BUG FIX 2 — _do_exit() state guard with _exit_in_progress flag:
    Previously: t["state"] = "CLOSED" was set BEFORE the SELL executed.
    If the SELL failed completely, state was permanently CLOSED and no
    re-entry to _do_exit() was possible. The position would stay open
    in Zerodha with no software awareness.
    Fix: state is only set to CLOSED AFTER a successful SELL confirm.
    An _exit_in_progress flag prevents duplicate exits from concurrent
    WebSocket ticks while the SELL is being attempted.

  BUG FIX 3 — Slot not released when SELL fails with position still open:
    Previously: self._release_slot() was called unconditionally after
    _place_sell_with_retry(), even when the SELL failed and the position
    was confirmed still open. This allowed another strategy to enter a
    live position while the Spike position was unresolved — potentially
    two simultaneous live positions consuming double margin.
    Fix: slot is only released when the position is confirmed closed.
    If SELL fails with position still open, slot stays locked and an
    emergency exit background thread takes over.

  BUG FIX 4 (order_router.py) — position check skipped on SELL attempt 1:
    place_sell_with_retry() no longer calls _is_position_open() after the
    first failed SELL attempt. A single transient blip gets a free retry
    before the position check, preventing premature retry-stop when the
    position was closed by exchange (auto square-off / phantom fill).

  BUG FIX 5 — Emergency exit background thread:
    After all 3 SELL retries fail with position confirmed still open,
    a daemon thread retries the SELL every 30 seconds for up to 15 min.
    The slot stays locked throughout. On success or exchange close
    confirmation, slot is released and PnL is logged normally.

  BUG FIX 6 — Spike window gate: entries blocked after 9:30.
    Previously: if no signal fired in 9:15–9:30 window, _trade_done
    remained False and _trade remained None all day. on_tick() would
    continue calling _check_2candle_signal() until 15:15, allowing a
    signal at e.g. 11:00 AM to enter a real position.
    Fix: on_tick() sets _trade_done=True as soon as spike_exit_time
    is reached with no open trade, permanently blocking further entries.

  BUG FIX 7 — Pending entry overwrite by subsequent candle signal.
    Previously: if _pending_entry was set (option had no valid price yet)
    and the next 10s candle fired a new signal, _pending_entry was
    silently overwritten — potentially switching CE→PE or vice versa.
    Fix: on_tick() checks _pending_entry is None before calling
    _check_2candle_signal().

  BUG FIX 8 — Pending entry resolved after spike window closes.
    Previously: if _pending_entry was set at 9:29:58 and the option tick
    arrived at 9:30:02 (after _trade_done was set True by BUG FIX 6),
    on_option_tick() would still call _build_entry() and enter a position
    outside the spike window.
    Fix: on_option_tick() pending entry resolution also checks
    not self._trade_done before calling _build_entry().

  OTHER (pre-existing fixes retained):
  - pre_market() subscribes ATM CE+PE even when prev_close is None.
  - on_tick() subscribes ATM options on very first market tick if
    pre-subscription failed.
  - Pending entry mechanism for stale/missing pre-9:15 option prices.
  - SL grace period: SL check suppressed for SL_GRACE_SECONDS after
    entry to prevent stale buffered ticks triggering a false SL exit.
  - Unused option leg unsubscribed after entry.
  - SL calculated from actual exchange fill price, not pre-order LTP.
"""

import csv
import logging
import os
import threading
import time as _time_mod
from datetime import datetime, time as dtime, timedelta, timezone
from typing import Optional

from core.base_strategy import BaseStrategy
from core.candle import SecondCandleBuilder

log = logging.getLogger("strategy.spike")

_IST = timezone(timedelta(hours=5, minutes=30))

def _now_ist() -> datetime:
    return datetime.now(tz=_IST).replace(tzinfo=None)

# ─────────────────────────────────────────────────────────────────────────────
#  LIVE MODE FLAG
# ─────────────────────────────────────────────────────────────────────────────
LIVE_MODE = True

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CFG = {
    "quantity"               : 30,
    "start_time"             : dtime(9, 15),
    "spike_exit_time"        : dtime(9, 30),
    "close_time"             : dtime(15, 15),
    "initial_sl_buffer"      : 90,
    "trail_trigger_pts"      : 50,
    "trail_distance"         : 25,
    "doji_threshold"         : 0.10,
    "bucket_sec"             : 10,
    "min_candles_before_mom" : 2,
    "csv_file"               : "spike_trades.csv",
    # SL grace period: seconds after entry fill before SL checks activate.
    # Prevents stale buffered WebSocket ticks (queued during the ~8s
    # _quick_check_order() polling window) from triggering a false SL exit.
    "sl_grace_seconds"       : 10,
    # Emergency exit: seconds between retry attempts when all 3 SELL retries fail.
    "emergency_retry_sec"    : 30,
    # Emergency exit: max attempts before giving up and logging CRITICAL.
    "emergency_max_attempts" : 30,  # 30 × 30s = 15 minutes
}


class SpikeStrategy(BaseStrategy):

    LIVE_MODE = LIVE_MODE

    @property
    def name(self) -> str:
        return "SPIKE"

    def __init__(self, market_hub):
        super().__init__(market_hub)

        self._index_8s          = SecondCandleBuilder(seconds=CFG["bucket_sec"])
        self._opt_8s            = None

        self._gap_direction     : Optional[str]  = None
        self._gap_filter_done   : bool           = False
        self._market_opened     : bool           = False

        self._pre_ce_token      : Optional[int]  = None
        self._pre_pe_token      : Optional[int]  = None
        self._pre_ce_sym        : Optional[str]  = None
        self._pre_pe_sym        : Optional[str]  = None

        self._trade             = None
        self._trade_done        : bool           = False
        self._today_pnl         : float          = 0.0
        self._completed         : list           = []
        self._pending_entry     = None

        self._prev_body_high    : Optional[float] = None
        self._prev_body_low     : Optional[float] = None
        self._prev_last5m_high  : Optional[float] = None
        self._prev_last5m_low   : Optional[float] = None
        self._prev_last5m_close : Optional[float] = None
        self._expiry_date       = None
        self._instruments       = None

        self._lock              = threading.Lock()

        mode_tag = "[LIVE]" if LIVE_MODE else "[PAPER]"
        log.info(f"[SPIKE] Initialized in {mode_tag} mode")

    # ── Pre-market ────────────────────────────────────────────────────────────

    def pre_market(self, pm, instruments) -> bool:
        from core.instruments import get_atm_strike

        now = _now_ist().time()
        if now >= CFG["spike_exit_time"]:
            log.warning(f"[SPIKE] Started after spike window — skipping today.")
            self._trade_done = True
            return True

        self._instruments       = instruments
        self._prev_body_high    = pm.prev_body_high
        self._prev_body_low     = pm.prev_body_low
        self._prev_last5m_high  = pm.prev_last5m_high
        self._prev_last5m_low   = pm.prev_last5m_low
        self._prev_last5m_close = pm.prev_last5m_close
        self._expiry_date       = pm.expiry_date

        log.info(f"[SPIKE] Pre-market | "
                 f"body=[{self._prev_body_low} – {self._prev_body_high}] "
                 f"prev_close={pm.prev_close} | mode={'LIVE' if LIVE_MODE else 'PAPER'}")

        ref_price = pm.prev_close or pm.prev_last5m_close

        if ref_price is None:
            log.warning(f"[SPIKE] No prev_close and no prev_last5m_close — "
                        f"cannot pre-subscribe options. Will attempt on first tick.")
            return True

        strike = get_atm_strike(ref_price)
        log.info(f"[SPIKE] Token lookup | strike={strike} "
                 f"expiry={pm.expiry_date} ref_price={ref_price:.2f} "
                 f"(source={'prev_close' if pm.prev_close else 'prev_last5m_close'})")

        ce_tok, ce_sym = instruments.get_option_token(strike, "CE", pm.expiry_date)
        pe_tok, pe_sym = instruments.get_option_token(strike, "PE", pm.expiry_date)

        self._pre_ce_token = ce_tok
        self._pre_ce_sym   = ce_sym
        self._pre_pe_token = pe_tok
        self._pre_pe_sym   = pe_sym

        if ce_tok:
            self.subscribe_option(ce_tok)
            log.info(f"[SPIKE] Pre-subscribed CE: {ce_sym} ({ce_tok})")
        else:
            log.error(f"[SPIKE] CE token not found | strike={strike} expiry={pm.expiry_date}")

        if pe_tok:
            self.subscribe_option(pe_tok)
            log.info(f"[SPIKE] Pre-subscribed PE: {pe_sym} ({pe_tok})")
        else:
            log.error(f"[SPIKE] PE token not found | strike={strike} expiry={pm.expiry_date}")

        return True

    # ── Tick handlers ─────────────────────────────────────────────────────────

    def on_tick(self, price: float, ts: datetime, tick_ts: datetime):
        t = ts.time()

        if t < CFG["start_time"] or t > CFG["close_time"]:
            return

        if not self._market_opened and t >= CFG["start_time"]:
            self._market_opened = True
            log.info(f"[SPIKE] Market open tick received: {price:.2f}")
            if self._pre_ce_token is None or self._pre_pe_token is None:
                self._subscribe_atm_on_open(price)

        closed_8s = self._index_8s.feed_tick(price, tick_ts)

        # Gap direction is detected on the first tick and logged for
        # reference / CSV only — it no longer triggers an entry.
        if not self._gap_filter_done and self._market_opened:
            self._determine_gap_direction(price)
            self._gap_filter_done = True

        # BUG FIX 6: Close spike window when time passes 9:30 with no trade.
        # Without this guard, _check_2candle_signal is called all day until
        # 15:15, and any signal at e.g. 11:00 AM would trigger a real entry.
        if not self._trade_done and self._trade is None and t >= CFG["spike_exit_time"]:
            log.info("[SPIKE] Spike window closed (9:30) with no trade — done for today.")
            self._trade_done = True

        # BUG FIX 7: Guard _pending_entry so a second candle signal cannot
        # overwrite a pending entry that has not been resolved yet.
        # All entries go through the 2x10s candle signal.
        if (not self._trade_done and
                self._trade is None and
                self._pending_entry is None and   # BUG FIX 7
                closed_8s is not None):
            self._check_2candle_signal(closed_8s, price, ts)

        # BUG FIX 2: only attempt time-exit if no exit already in progress
        if (self._trade and
                self._trade["state"] == "OPEN" and
                not self._trade.get("_exit_in_progress")):
            if t >= CFG["spike_exit_time"]:
                opt_price = self.get_price(self._trade["token"]) or self._trade["entry"]
                self._do_exit(opt_price, "SPIKE_WINDOW_END", ts)

    def on_candle(self, candle: dict, ts: datetime):
        pass

    def on_option_tick(self, token: int, price: float, ts: datetime, tick_ts: datetime = None):
        """
        Option price arrives on every WebSocket tick.

        Job 1 — Pending entry resolution:
          If option had no valid price at signal time, stored as _pending_entry.
          Resolved here on first live tick.

        Job 2 — Software trailing SL management:
          Runs on every tick while trade is OPEN and no exit is in progress.
          SL grace period suppresses SL checks for sl_grace_seconds after fill
          to prevent stale buffered ticks from causing a false immediate exit.

        BUG FIX 2 — _exit_in_progress guard:
          When _do_exit() is executing (or emergency exit thread is running),
          _exit_in_progress is True. on_option_tick() skips SL checks entirely
          to prevent concurrent duplicate exit attempts from rapid tick delivery.

        BUG FIX 8 — Pending entry not resolved after spike window closes:
          _trade_done is also checked before resolving pending entry so that
          a pending entry set at 9:29:58 is not entered after the window closes.
        """
        # Job 1: resolve pending entry
        # BUG FIX 8: added "not self._trade_done" to block post-window resolution
        if (self._pending_entry and
                token == self._pending_entry["token"] and
                not self._trade and
                not self._trade_done):           # BUG FIX 8
            p = self._pending_entry
            self._pending_entry = None
            log.info(f"[SPIKE] Pending entry resolved — first live tick for {p['sym']} "
                     f"@ {price:.2f} (was stale at entry signal time)")
            self._build_entry(p["sym"], p["token"], p["signal"], ts, p["reason"])
            return

        # Job 2: trailing SL management
        if not (self._trade and token == self._trade.get("token")):
            return

        if self._opt_8s is None:
            self._opt_8s = SecondCandleBuilder(seconds=CFG["bucket_sec"])
        use_ts = tick_ts if tick_ts else ts
        self._opt_8s.feed_tick(price, use_ts)

        # BUG FIX 2: skip all SL logic when exit is already being handled
        if self._trade["state"] != "OPEN" or self._trade.get("_exit_in_progress"):
            return

        # Update trailing high
        if price > self._trade["highest_seen"]:
            self._trade["highest_seen"] = price

        # Compute trailing SL
        new_sl = self._compute_trailing_sl(
            self._trade["entry"], self._trade["highest_seen"], self._trade["sl"]
        )
        if new_sl > self._trade["sl"]:
            log.info(f"[SPIKE] TSL: {self._trade['sl']:.0f} → {new_sl:.0f} "
                     f"(highest={self._trade['highest_seen']:.0f})")
            self._trade["sl"] = new_sl

        # SL grace period: ignore SL checks for sl_grace_seconds after fill
        sl_active_from = self._trade.get("sl_active_from")
        if sl_active_from is not None and ts < sl_active_from:
            log.debug(
                f"[SPIKE] SL grace active — skipping SL check "
                f"(ts={ts.strftime('%H:%M:%S')} < active_from={sl_active_from.strftime('%H:%M:%S')}) "
                f"price={price:.0f} sl={self._trade['sl']:.0f}"
            )
            return

        # SL breached — fire exit
        if price <= self._trade["sl"]:
            self._do_exit(price, "SL_HIT", ts)

    # ── Subscribe ATM on open ─────────────────────────────────────────────────

    def _subscribe_atm_on_open(self, open_price: float):
        from core.instruments import get_atm_strike
        strike = get_atm_strike(open_price)
        expiry = self._expiry_date
        log.info(f"[SPIKE] Late-subscribing ATM options on open tick | "
                 f"strike={strike} expiry={expiry} open={open_price:.2f}")

        ce_tok, ce_sym = self._instruments.get_option_token(strike, "CE", expiry)
        pe_tok, pe_sym = self._instruments.get_option_token(strike, "PE", expiry)

        if ce_tok and self._pre_ce_token is None:
            self.subscribe_option(ce_tok)
            self._pre_ce_token = ce_tok
            self._pre_ce_sym   = ce_sym
            log.info(f"[SPIKE] Late-subscribed CE: {ce_sym} ({ce_tok})")

        if pe_tok and self._pre_pe_token is None:
            self.subscribe_option(pe_tok)
            self._pre_pe_token = pe_tok
            self._pre_pe_sym   = pe_sym
            log.info(f"[SPIKE] Late-subscribed PE: {pe_sym} ({pe_tok})")

    # ── Gap direction (reference only — does not trigger entry) ───────────────

    def _determine_gap_direction(self, open_price: float):
        """
        Classify today's open vs prev day's last 5-min range (or body).
        Result is stored in self._gap_direction for logging/CSV reference only.
        It does NOT control entry — the 2x10s candle signal does.
        """
        h5, l5 = self._prev_last5m_high, self._prev_last5m_low

        if h5 is not None and l5 is not None:
            log.info(f"[SPIKE] Gap ref → prev last 5-min: "
                     f"H={h5:.0f}  L={l5:.0f}  Today open={open_price:.0f}")
            if open_price > h5:
                self._gap_direction = "CE"
                log.info(f"[SPIKE]  GAP UP: open={open_price:.0f} > last5m_high={h5:.0f}  (ref only)")
            elif open_price < l5:
                self._gap_direction = "PE"
                log.info(f"[SPIKE]  GAP DOWN: open={open_price:.0f} < last5m_low={l5:.0f}  (ref only)")
            else:
                self._gap_direction = "BOTH"
                log.info(f"[SPIKE]  NO GAP: open inside [{l5:.0f}–{h5:.0f}] (ref only)")
            return

        log.warning("[SPIKE] Last 5-min candle data unavailable — falling back to daily body")
        bh, bl = self._prev_body_high, self._prev_body_low

        if bh is None or bl is None:
            self._gap_direction = "BOTH"
            log.warning("[SPIKE] No gap reference at all — defaulting to BOTH")
            return

        if open_price > bh:
            self._gap_direction = "CE"
            log.info(f"[SPIKE]  GAP UP (fallback): open={open_price:.0f} > body_high={bh:.0f} (ref only)")
        elif open_price < bl:
            self._gap_direction = "PE"
            log.info(f"[SPIKE]  GAP DOWN (fallback): open={open_price:.0f} < body_low={bl:.0f} (ref only)")
        else:
            self._gap_direction = "BOTH"
            log.info(f"[SPIKE]  NO GAP (fallback): open inside [{bl:.0f}–{bh:.0f}]")

    # ── 2-candle signal ───────────────────────────────────────────────────────

    def _check_2candle_signal(self, latest_closed: dict, index_price: float, ts: datetime):
        last_two = self._index_8s.last_n_closed(2)
        if len(last_two) < 2:
            return

        c1, c2 = last_two[-2], last_two[-1]
        signal = self._check_signal(c1, c2)
        if not signal:
            return

        log.info(f"[SPIKE] 2-candle signal: {signal} at {ts.strftime('%H:%M:%S')}")

        if signal == "CE" and self._pre_ce_token:
            sym, token = self._pre_ce_sym, self._pre_ce_token
        elif signal == "PE" and self._pre_pe_token:
            sym, token = self._pre_pe_sym, self._pre_pe_token
        else:
            from core.instruments import get_atm_strike
            strike = get_atm_strike(index_price)
            expiry = self._expiry_date
            log.info(f"[SPIKE] Fallback token lookup | signal={signal} "
                     f"strike={strike} expiry={expiry}")
            token, sym = self._instruments.get_option_token(strike, signal, expiry)

        if not token or not sym:
            log.error(f"[SPIKE] No token for {signal} — trade SKIPPED")
            return

        self.subscribe_option(token)
        self._build_entry(sym, token, signal, ts, reason="2x8s_signal")

    # ── Entry ─────────────────────────────────────────────────────────────────

    def _build_entry(self, sym: str, token: int, signal: str, ts: datetime, reason: str):
        """
        Attempt to enter a position.

        Price guard: if option has no valid post-9:15 price, store pending entry.
        on_option_tick() resolves it on first live tick.

        BUG FIX 1 (order_router): _place_buy() now returns None if the order
        was COMPLETE but filled_quantity=0 (zero-fill / phantom position).
        The entry is aborted cleanly — no phantom position is created.

        SL = actual fill price − initial_sl_buffer (not pre-order LTP).
        """
        opt_price = self.get_price(token)
        price_ts  = self.get_price_ts(token)
        market_open_today = ts.replace(hour=9, minute=15, second=0, microsecond=0)

        if (not opt_price or opt_price <= 0) or \
           (price_ts is None or price_ts < market_open_today):
            log.warning(
                f"[SPIKE] No valid post-9:15 price for {sym} "
                f"(price={opt_price} priced_at={price_ts}) — storing pending entry"
            )
            self._pending_entry = {
                "sym": sym, "token": token, "signal": signal, "ts": ts, "reason": reason
            }
            return

        if not self._acquire_slot():
            log.warning("[SPIKE] Trade slot blocked — another live strategy has a position")
            return

        result = self._place_buy(sym, token, CFG["quantity"], opt_price)
        if result is None:
            self._release_slot()
            log.error(f"[SPIKE] BUY order FAILED for {sym} — entry aborted")
            return

        order_id, fill_price = result

        log.info(
            f"[SPIKE] BUY confirmed | pre_ltp={opt_price:.2f} "
            f"fill_price={fill_price:.2f} | diff={fill_price - opt_price:+.2f}"
        )

        sl             = fill_price - CFG["initial_sl_buffer"]
        sl_active_from = ts + timedelta(seconds=CFG["sl_grace_seconds"])

        log.info(
            f"[SPIKE] SL grace period active until "
            f"{sl_active_from.strftime('%H:%M:%S')} "
            f"({CFG['sl_grace_seconds']}s after entry fill)"
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
            # BUG FIX 2: exit guard flag
            "_exit_in_progress": False,
        }

        if signal == "CE" and self._pre_pe_token:
            self.unsubscribe_option(self._pre_pe_token)
            log.info(f"[SPIKE] Unsubscribed unused PE leg: "
                     f"{self._pre_pe_sym} ({self._pre_pe_token})")
        elif signal == "PE" and self._pre_ce_token:
            self.unsubscribe_option(self._pre_ce_token)
            log.info(f"[SPIKE] Unsubscribed unused CE leg: "
                     f"{self._pre_ce_sym} ({self._pre_ce_token})")

        mode_tag = "LIVE" if LIVE_MODE else "PAPER"
        log.info(
            f"[SPIKE] [{mode_tag}] ENTRY {sym} @ {fill_price:.0f} | "
            f"SL={sl:.0f} | Trail kicks at {fill_price + CFG['trail_trigger_pts']:.0f} "
            f"| Reason={reason} | order_id={order_id}"
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
          not before. This preserves re-entry ability if SELL fails.

        BUG FIX 3 — Slot management:
          Slot is released ONLY when the position is confirmed closed.
          If SELL fails with position still open, slot stays LOCKED and
          an emergency exit thread takes over. This prevents another live
          strategy from entering while the Spike position is unresolved.

        Flow:
          success  → state=CLOSED, release_slot, _finalize_exit
          fail + position closed (phantom/auto-squared) → state=CLOSED,
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
            t["state"] = "CLOSED"           # BUG FIX 2: only CLOSED after confirm
            self._release_slot()            # BUG FIX 3: released only on success
            self._finalize_exit(t, sell_price, order_id, reason, ts)
            return

        # ── ALL 3 RETRIES FAILED ───────────────────────────────────────────────
        if LIVE_MODE:
            still_open = self._hub.order_router._is_position_open(t["symbol"])

            if not still_open:
                # Position closed by exchange (auto square-off or phantom fill)
                log.warning(
                    f"[SPIKE] SELL failed after 3 retries but position confirmed "
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
                f"[SPIKE] CRITICAL: SELL failed after 3 retries — "
                f"position STILL OPEN for {t['symbol']}!\n"
                f"  Slot is LOCKED — no other strategy can enter.\n"
                f"  Emergency exit thread starting (retry every "
                f"{CFG['emergency_retry_sec']}s for up to "
                f"{CFG['emergency_max_attempts'] * CFG['emergency_retry_sec'] // 60} min).\n"
                f"  *** CHECK ZERODHA CONSOLE IF EMERGENCY EXIT ALSO FAILS! ***\n"
                f"{'!'*60}"
            )
            # _exit_in_progress stays True — prevents on_option_tick from re-entering here.
            # Emergency thread is now the sole manager of this position.
            self._start_emergency_exit(t, exit_price, reason, ts)
            return

        # Paper mode: shouldn't reach here (paper sell never fails), handle gracefully
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
                    f"[SPIKE] Emergency exit attempt {attempt}/{max_attempts} | "
                    f"{t['symbol']} | slot LOCKED"
                )

                # Check if exchange already closed the position
                still_open = self._hub.order_router._is_position_open(t["symbol"])
                if not still_open:
                    log.info(
                        f"[SPIKE] Emergency exit: {t['symbol']} confirmed CLOSED "
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
                        f"[SPIKE] Emergency exit SUCCESS on attempt {attempt} "
                        f"@ {sell_price:.0f}"
                    )
                    now = _now_ist()
                    self._finalize_exit(t, sell_price, order_id,
                                        f"{reason}_EMERGENCY", now)
                    return

                log.error(
                    f"[SPIKE] Emergency exit attempt {attempt}/{max_attempts} FAILED | "
                    f"{t['symbol']} | ltp={ltp:.0f}"
                )

            # All attempts exhausted
            log.error(
                f"\n{'!'*60}\n"
                f"[SPIKE] GAVE UP emergency exit for {t['symbol']} after "
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
            name="spike-emergency-exit",
            daemon=True,
        )
        thread.start()
        log.info(f"[SPIKE] Emergency exit thread started for {t['symbol']}")

    # ── Exit finalize (shared by normal + emergency paths) ────────────────────

    def _finalize_exit(
        self,
        t:         dict,
        sell_price: float,
        order_id:  Optional[str],
        reason:    str,
        ts:        datetime,
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
            f"[SPIKE] [{mode_tag}] EXIT [{reason}] {t['symbol']} @ {sell_price:.0f} "
            f"| PnL {pnl:.0f} | Today {self._today_pnl:.0f} | order_id={order_id}"
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
        if SpikeStrategy._is_doji(c1) or SpikeStrategy._is_doji(c2):
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
        profit = highest - entry
        if profit >= CFG["trail_trigger_pts"]:
            new_sl = highest - CFG["trail_distance"]
            return max(new_sl, current_sl)
        return current_sl

    def _log_csv(self, row: dict):
        fname  = CFG["csv_file"]
        exists = os.path.isfile(fname)
        fields = ["timestamp", "symbol", "action", "price",
                  "sl", "status", "pnl", "reason", "gap_direction", "mode", "order_id"]
        with open(fname, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if not exists:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in fields})

    def eod_summary(self):
        log.info(f"\n[SPIKE] {'='*50}")
        log.info(f"[SPIKE] END OF DAY | mode={'LIVE' if LIVE_MODE else 'PAPER'}")
        log.info(f"[SPIKE] Gap direction  : {self._gap_direction}")
        log.info(f"[SPIKE] Trade executed : {'Yes' if self._trade_done else 'No'}")
        for t in self._completed:
            log.info(f"[SPIKE]   {t['symbol']} {t['exit_reason']} "
                     f"entry={t['entry']:.0f} exit={t['exit_price']:.0f} PnL={t['pnl']:.0f}")
        log.info(f"[SPIKE] Today PnL      : {self._today_pnl:.0f}")
        log.info(f"[SPIKE] {'='*50}\n")
