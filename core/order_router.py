"""
core/order_router.py

Central order execution layer — single point for all live order placement.

═══════════════════════════════════════════════════════════════════
  DESIGN DECISIONS (read before modifying)
═══════════════════════════════════════════════════════════════════

1. MARKET orders with market protection — fills at best available price

   Zerodha's API requires market_protection for MARKET orders on options:
     "Market orders without market protection are not allowed via API."

   Fix: use ORDER_TYPE_MARKET with market_protection=20 (20% band).
   Zerodha internally converts these to a LIMIT order at LTP ± 20%.

   How the resulting LIMIT order behaves:
     • If liquidity exists → fills almost instantly (< 1 second)
     • If the limit price is stale (gap-open) → the order sits OPEN, never fills
     • If rejected outright → exchange returns REJECTED immediately

   This is why we need the quick-check logic (design point 4 below).

   market_protection values:
     -1         → Auto protection applied by Zerodha/NSE (recommended)
     1..100     → Custom protection percentage (e.g., 10 = ±10% band)

2. One-live-trade-at-a-time slot

   Only one strategy can hold a live position at a time. This prevents:
     - Double margin consumption across 4 strategies
     - Conflicting CE/PE positions (SPIKE long CE + BB_STOCH long PE = disaster)
     - Capital exhaustion

   The slot is only enforced when LIVE_MODE=True. Paper-mode strategies never
   compete for the slot and trade independently.

3. Per-strategy LIVE_MODE flag

   Each strategy file exposes:
       LIVE_MODE = False   # change to True to enable real orders for that strategy

   When False → paper fill (no Kite API call, simulated fill at LTP).
   When True  → real MARKET order via kite.place_order() with market protection.

   Default: ALL strategies are paper. Change only the one you want to go live.

4. Quick placement check + cancel + ONE retry

   Zerodha converts our MARKET order to a LIMIT order internally. That LIMIT
   order may:
     a) Fill instantly            → COMPLETE in < 1s (happy path)
     b) Be rejected outright      → REJECTED immediately (bad params, margin, circuit)
     c) Sit as OPEN / OPEN PENDING → stuck; will NEVER fill if price has moved away

   Case (c) is the failure mode we must handle. Without this logic, the bot
   would wait 15s for a stuck LIMIT order, then log TIMEOUT and abort —
   wasting the signal and leaving the slot locked.

   Fix: immediately after place_order(), poll for QUICK_CHECK_SEC (3 seconds).
     • COMPLETE / REJECTED / CANCELLED reached → handle normally.
     • Still stuck (OPEN / OPEN PENDING) after QUICK_CHECK_SEC:
         1. Cancel the stuck order.
         2. Place ONE fresh replacement order with the latest LTP from WebSocket.
         3. Run the full 15s _confirm_order() on the replacement.
         4. If replacement also fails → log "signal skipped" and return None.

   Only ONE retry is allowed. This prevents infinite loops if the market is
   genuinely illiquid or the session is in a bad state.

5. Return value: (order_id, fill_price) tuple

   place_buy() and place_sell() both return Optional[Tuple[str, float]]:
     - Live mode:  (order_id, average_price) fetched from order_history after COMPLETE.
                   Falls back to ref_ltp if average_price is missing.
     - Paper mode: ("PAPER-{ms}", ltp) — simulated fill at the ref LTP.
     - Failure:    None

   Callers unpack like:
       result = self._place_buy(sym, token, qty, ltp)
       if result is None:
           return
       order_id, fill_price = result

6. Token / session expiry mid-session

   If the Zerodha access token expires while the bot is running (rare but
   possible after midnight refresh), any Kite REST call raises
   kiteconnect.exceptions.TokenException (HTTP 403).

   Fix: _is_token_exception() inspects the exception type name and message.
   When detected:
     • A loud, actionable error is logged with exact remediation steps.
     • None is returned so the caller aborts the trade / releases the slot.
     • No further Kite calls are attempted for that order.

7. Order status confirmation

   After kite.place_order() returns an order_id, we poll kite.order_history()
   until the order reaches a terminal state (COMPLETE / REJECTED / CANCELLED)
   or a timeout expires.

   Without this, NSE could silently reject an option order (circuit filter,
   no quotes at that ms) and the bot would manage a phantom SL and eventually
   fire a SELL against a position that never existed.

   _confirm_order() returns one of:
     "COMPLETE"      → fill confirmed, caller may treat trade as open
     "REJECTED"      → exchange rejected; caller must abort entry / alert on sell
     "CANCELLED"     → order was cancelled (rare for MARKET orders)
     "TIMEOUT"       → did not reach terminal state in time; caller treats as failed
     "TOKEN_EXPIRED" → access token invalid mid-session; caller must abort + alert

═══════════════════════════════════════════════════════════════════
"""

import logging
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple

log = logging.getLogger("core.router")

_IST = timezone(timedelta(hours=5, minutes=30))

# ── Market protection for MARKET orders ──────────────────────────────────────
# -1  → Auto protection: Zerodha/NSE applies default band automatically
# 1-100 → Custom %: e.g. 20 means order executes within ±20% of last price
# Must be within circuit limits. 20 is a safe default for options.
MARKET_PROTECTION = 20

# ── Quick placement check timeout ────────────────────────────────────────────
# Seconds to poll for COMPLETE before declaring the order "stuck" and cancelling.
# Keep short: MARKET→LIMIT fills in < 1s when liquid. 3s is generous.
QUICK_CHECK_SEC = 3.0

# ── Quick check poll interval ─────────────────────────────────────────────────
QUICK_POLL_SEC = 0.5


def _now_ist() -> datetime:
    return datetime.now(tz=_IST).replace(tzinfo=None)


class OrderRouter:
    """
    Single point for all order placement across all strategies.
    Thread-safe. Enforces one-live-trade-at-a-time when in live mode.

    Usage in a strategy:

        # Entry
        ok = self._hub.order_router.acquire_slot(self.name, LIVE_MODE)
        if not ok:
            return  # another live strategy already has a position
        result = self._hub.order_router.place_buy(self.name, symbol, token, qty, ltp, LIVE_MODE)
        if result is None:
            return  # order failed
        order_id, fill_price = result

        # Exit
        result = self._hub.order_router.place_sell(self.name, symbol, token, qty, ltp, LIVE_MODE)
        self._hub.order_router.release_slot(self.name, LIVE_MODE)
    """

    def __init__(self, hub):
        self._hub  = hub           # MarketHub — for hub.kite access
        self._lock = threading.Lock()

        # Live trade slot — only one LIVE strategy can hold it at a time.
        # Paper strategies bypass this entirely.
        self._slot_owner: Optional[str] = None

        log.info("OrderRouter ready — all strategies default to PAPER mode")

    # ─────────────────────────────────────────────────────────────────────────
    #  Slot management
    # ─────────────────────────────────────────────────────────────────────────

    def acquire_slot(self, strategy_name: str, live_mode: bool) -> bool:
        """
        Try to acquire the global trade slot before entering a position.

        Paper mode  → always returns True (no contention, strategies independent).
        Live mode   → returns True only if no other live strategy holds the slot.

        Returns True = may proceed with entry.
        Returns False = blocked; another live strategy has an open position.
        """
        if not live_mode:
            return True  # paper strategies never blocked

        with self._lock:
            if self._slot_owner is not None:
                log.warning(
                    f"[Router] LIVE slot BLOCKED: {strategy_name} wants to enter "
                    f"but slot already held by {self._slot_owner}"
                )
                return False
            self._slot_owner = strategy_name
            log.info(f"[Router] LIVE slot ACQUIRED by {strategy_name}")
            return True

    def release_slot(self, strategy_name: str, live_mode: bool):
        """
        Release the slot after a live position is fully closed.
        Only the current owner can release. Paper mode is a no-op.
        """
        if not live_mode:
            return

        with self._lock:
            if self._slot_owner != strategy_name:
                log.warning(
                    f"[Router] {strategy_name} tried to release slot held by "
                    f"{self._slot_owner} — ignored"
                )
                return
            self._slot_owner = None
            log.info(f"[Router] LIVE slot RELEASED by {strategy_name}")

    def slot_owner(self) -> Optional[str]:
        """Which live strategy currently holds the slot. None if free."""
        return self._slot_owner

    def is_slot_mine(self, strategy_name: str) -> bool:
        """Quick check used by strategies before placing an exit order."""
        return self._slot_owner == strategy_name

    # ─────────────────────────────────────────────────────────────────────────
    #  Order placement (public API)
    # ─────────────────────────────────────────────────────────────────────────

    def place_buy(
        self,
        strategy_name: str,
        symbol:        str,
        token:         int,
        qty:           int,
        ltp:           float,
        live_mode:     bool,
    ) -> Optional[Tuple[str, float]]:
        """
        Place a BUY (entry) order.

        live_mode=False → paper fill logged; returns ("PAPER-{ms}", ltp)
        live_mode=True  → MARKET MIS order via Kite with market protection;
                          uses quick-check → cancel → one-retry logic;
                          returns (order_id, fill_price) ONLY when COMPLETE.

        Returns (order_id, fill_price) on success, None on failure.
        """
        if live_mode:
            return self._live_order(
                strategy_name, symbol, token, qty, ltp,
                transaction_type="BUY",
            )
        return self._paper_fill("BUY", strategy_name, symbol, ltp)

    def place_sell(
        self,
        strategy_name: str,
        symbol:        str,
        token:         int,
        qty:           int,
        ltp:           float,
        live_mode:     bool,
    ) -> Optional[Tuple[str, float]]:
        """
        Place a SELL (exit) order.

        live_mode=False → paper fill logged; returns ("PAPER-{ms}", ltp)
        live_mode=True  → MARKET MIS order via Kite with market protection;
                          uses quick-check → cancel → one-retry logic;
                          returns (order_id, fill_price) ONLY when COMPLETE.

        Returns (order_id, fill_price) on success, None on failure.
        """
        if live_mode:
            return self._live_order(
                strategy_name, symbol, token, qty, ltp,
                transaction_type="SELL",
            )
        return self._paper_fill("SELL", strategy_name, symbol, ltp)

    # ─────────────────────────────────────────────────────────────────────────
    #  Internal — live execution (quick-check + cancel + one retry)
    # ─────────────────────────────────────────────────────────────────────────

    def _live_order(
        self,
        strategy_name:    str,
        symbol:           str,
        token:            int,
        qty:              int,
        ltp:              float,
        transaction_type: str,   # "BUY" or "SELL"
    ) -> Optional[Tuple[str, float]]:
        """
        Place a MARKET MIS order (Zerodha converts to LIMIT with market_protection)
        and confirm it filled, with a cancel + one-retry if it gets stuck.

        Flow
        ────
        Attempt 1:
          1. _place_raw_order() → order_id (or None on API error).
          2. _quick_check_order() polls for QUICK_CHECK_SEC (3s):
               • COMPLETE  → fetch fill price → return ✓
               • REJECTED / CANCELLED → log and return None (no retry worthwhile)
               • TOKEN_EXPIRED → log and return None
               • Still non-terminal (OPEN / OPEN PENDING / VALIDATION PENDING):
                   → Zerodha's LIMIT conversion is stuck (stale price). Go to step 3.
          3. _cancel_order() — cancels the stuck order.

        Attempt 2 (ONE retry):
          4. Refresh LTP from WebSocket (free, no network) for a fresher limit price.
          5. _place_raw_order() → order_id2 (or None on API error → skip signal).
          6. _confirm_order() full 15s poll:
               • COMPLETE  → fetch fill price → return ✓
               • Anything else → log "signal skipped" → return None.

        Only ONE retry is attempted. If both attempts fail the signal is skipped.
        """
        kite = self._hub.kite
        if kite is None:
            log.error(
                f"[Router][LIVE] Kite not loaded — "
                f"cannot place {transaction_type} for {strategy_name}"
            )
            return None

        # ── Attempt 1 ─────────────────────────────────────────────────────────
        log.info(
            f"[Router][LIVE] {transaction_type} attempt 1/2 | "
            f"{strategy_name} | {symbol} | qty={qty} | ref_ltp={ltp:.2f}"
        )

        order_id = self._place_raw_order(kite, symbol, qty, ltp, transaction_type, strategy_name)
        if order_id is None:
            return None   # API call itself failed; error already logged

        # Quick check: did the order fill (or fail) within QUICK_CHECK_SEC?
        quick_status = self._quick_check_order(order_id)

        if quick_status == "COMPLETE":
            log.info(
                f"[Router][LIVE] {transaction_type} filled on attempt 1 ✓ | "
                f"{strategy_name} | {symbol} | order_id={order_id}"
            )
            return self._fetch_fill_price(order_id, ltp, strategy_name, symbol, transaction_type)

        if quick_status in ("REJECTED", "CANCELLED", "TOKEN_EXPIRED"):
            self._log_order_failure(transaction_type, strategy_name, symbol, qty, ltp,
                                    order_id, quick_status, attempt=1)
            return None

        # ── Order is stuck (OPEN / OPEN PENDING) → cancel it ─────────────────
        log.warning(
            f"\n{'─'*60}\n"
            f"[Router][LIVE] {transaction_type} order STUCK after {QUICK_CHECK_SEC}s "
            f"(status={quick_status!r}) — cancelling and retrying once\n"
            f"  Strategy : {strategy_name}\n"
            f"  Symbol   : {symbol}  qty={qty}\n"
            f"  order_id : {order_id}\n"
            f"  ref_ltp  : {ltp:.2f}\n"
            f"{'─'*60}"
        )
        self._cancel_order(order_id, strategy_name, symbol)

        # ── Attempt 2 (ONE retry) ──────────────────────────────────────────────
        # Refresh LTP from WebSocket cache — free, no network call.
        fresh_ltp = self._hub.last_price(token) or ltp
        if fresh_ltp != ltp:
            log.info(
                f"[Router][LIVE] LTP refreshed for retry: {ltp:.2f} → {fresh_ltp:.2f}"
            )

        log.info(
            f"[Router][LIVE] {transaction_type} attempt 2/2 (retry) | "
            f"{strategy_name} | {symbol} | qty={qty} | ref_ltp={fresh_ltp:.2f}"
        )

        order_id2 = self._place_raw_order(kite, symbol, qty, fresh_ltp, transaction_type,
                                          strategy_name)
        if order_id2 is None:
            log.error(
                f"[Router][LIVE] Retry placement FAILED — "
                f"signal SKIPPED for {strategy_name} | {symbol}"
            )
            return None

        # Full confirm on the retry order
        status2 = self._confirm_order(str(order_id2))

        if status2 == "COMPLETE":
            log.info(
                f"[Router][LIVE] {transaction_type} filled on attempt 2 ✓ | "
                f"{strategy_name} | {symbol} | order_id={order_id2}"
            )
            return self._fetch_fill_price(order_id2, fresh_ltp, strategy_name, symbol,
                                          transaction_type)

        # Both attempts exhausted → skip signal
        log.error(
            f"\n{'!'*60}\n"
            f"[Router][LIVE] {transaction_type} FAILED after 2 attempts — "
            f"SIGNAL SKIPPED\n"
            f"  Strategy  : {strategy_name}\n"
            f"  Symbol    : {symbol}  qty={qty}\n"
            f"  Attempt 1 : order_id={order_id}  status={quick_status!r}\n"
            f"  Attempt 2 : order_id={order_id2}  status={status2!r}\n"
            f"  ref_ltp   : {ltp:.2f}  retry_ltp={fresh_ltp:.2f}\n"
            + (
                f"  → *** SELL FAILED — CHECK ZERODHA CONSOLE AND "
                f"SQUARE OFF MANUALLY IF POSITION IS OPEN! ***\n"
                if transaction_type == "SELL" else
                f"  → Signal skipped. No position opened.\n"
            )
            + f"{'!'*60}"
        )
        return None

    # ─────────────────────────────────────────────────────────────────────────
    #  Internal — raw order placement (single API call, no polling)
    # ─────────────────────────────────────────────────────────────────────────

    def _place_raw_order(
        self,
        kite,
        symbol:           str,
        qty:              int,
        ltp:              float,
        transaction_type: str,
        strategy_name:    str,
    ) -> Optional[str]:
        """
        Call kite.place_order() once and return the order_id string, or None on failure.
        Does NOT poll for confirmation — that is the caller's job.
        """
        try:
            txn = (
                kite.TRANSACTION_TYPE_BUY
                if transaction_type == "BUY"
                else kite.TRANSACTION_TYPE_SELL
            )
            order_id = kite.place_order(
                variety           = kite.VARIETY_REGULAR,
                exchange          = kite.EXCHANGE_NFO,
                tradingsymbol     = symbol,
                transaction_type  = txn,
                quantity          = qty,
                product           = kite.PRODUCT_MIS,
                order_type        = kite.ORDER_TYPE_MARKET,
                market_protection = MARKET_PROTECTION,
            )
            log.info(
                f"[Router][LIVE] {transaction_type} ORDER PLACED | "
                f"{strategy_name} | {symbol} | qty={qty} | "
                f"ref_ltp={ltp:.2f} | market_protection={MARKET_PROTECTION} | "
                f"order_id={order_id}"
            )
            return str(order_id)

        except Exception as exc:
            if self._is_token_exception(exc):
                log.error(
                    f"\n{'!'*60}\n"
                    f"[Router][TOKEN EXPIRED] {transaction_type} FAILED — access token invalid!\n"
                    f"  Strategy : {strategy_name}\n"
                    f"  Symbol   : {symbol}\n"
                    f"  Error    : {exc}\n"
                    f"  → NO order was placed. Slot will be released by caller.\n"
                    f"  → RESTART the bot with a fresh token (re-run t.py / auto_login).\n"
                    f"  → If you were already in a live position, CHECK ZERODHA CONSOLE NOW.\n"
                    f"{'!'*60}"
                )
            else:
                log.error(
                    f"[Router][LIVE] {transaction_type} PLACE_ORDER FAILED | "
                    f"{strategy_name} | {symbol}: {exc}"
                )
            return None

    # ─────────────────────────────────────────────────────────────────────────
    #  Internal — quick placement check (QUICK_CHECK_SEC seconds)
    # ─────────────────────────────────────────────────────────────────────────

    def _quick_check_order(self, order_id: str) -> str:
        """
        Poll order_history for QUICK_CHECK_SEC seconds to see if the order
        reached a terminal state quickly (expected for a liquid MARKET→LIMIT order).

        Returns:
          Terminal   : "COMPLETE", "REJECTED", "CANCELLED", "TOKEN_EXPIRED"
          Non-terminal (still open after timeout): last observed status string
            e.g. "OPEN", "OPEN PENDING", "VALIDATION PENDING"
            — caller interprets these as "stuck" → cancel + retry.
        """
        TERMINAL = {"COMPLETE", "REJECTED", "CANCELLED"}
        deadline    = time.monotonic() + QUICK_CHECK_SEC
        attempt     = 0
        last_status = "UNKNOWN"

        while time.monotonic() < deadline:
            attempt += 1
            try:
                history = self._hub.kite.order_history(order_id)
                if not history:
                    log.debug(
                        f"[Router] quick_check({order_id}) attempt={attempt} "
                        f"— empty history, retrying"
                    )
                    time.sleep(QUICK_POLL_SEC)
                    continue

                last_status = history[-1].get("status", "UNKNOWN").upper()
                log.debug(
                    f"[Router] quick_check({order_id}) attempt={attempt} "
                    f"status={last_status}"
                )

                if last_status in TERMINAL:
                    log.info(
                        f"[Router] quick_check({order_id}) → {last_status} "
                        f"after {attempt} poll(s)"
                    )
                    return last_status

            except Exception as exc:
                if self._is_token_exception(exc):
                    log.error(
                        f"[Router][TOKEN EXPIRED] quick_check({order_id}): {exc}"
                    )
                    return "TOKEN_EXPIRED"
                log.warning(
                    f"[Router] quick_check({order_id}) poll error "
                    f"(attempt {attempt}): {exc} — retrying"
                )

            time.sleep(QUICK_POLL_SEC)

        log.info(
            f"[Router] quick_check({order_id}) timed out after {QUICK_CHECK_SEC}s "
            f"({attempt} polls) — last status={last_status!r} → treating as STUCK"
        )
        return last_status   # non-terminal → caller cancels + retries

    # ─────────────────────────────────────────────────────────────────────────
    #  Internal — cancel a stuck order
    # ─────────────────────────────────────────────────────────────────────────

    def _cancel_order(
        self,
        order_id:      str,
        strategy_name: str,
        symbol:        str,
    ) -> bool:
        """
        Attempt to cancel an open order via kite.cancel_order().

        Returns True if the API call succeeded.
        Returns False on error — caller proceeds to retry regardless, since
        the cancel may still go through asynchronously.
        """
        kite = self._hub.kite
        if kite is None:
            log.warning(f"[Router] _cancel_order: kite is None — cannot cancel {order_id}")
            return False

        try:
            kite.cancel_order(variety=kite.VARIETY_REGULAR, order_id=order_id)
            log.info(
                f"[Router] CANCEL sent for stuck order | "
                f"{strategy_name} | {symbol} | order_id={order_id}"
            )
            # Brief pause: give the exchange a moment to process the cancel
            # before we place the retry order.
            time.sleep(0.5)
            return True

        except Exception as exc:
            # If order already filled or was already cancelled, the API throws.
            # Log it but don't block the retry.
            log.warning(
                f"[Router] _cancel_order({order_id}) error "
                f"(may already be filled/cancelled): {exc}"
            )
            return False

    # ─────────────────────────────────────────────────────────────────────────
    #  Internal — fetch fill price after COMPLETE
    # ─────────────────────────────────────────────────────────────────────────

    def _fetch_fill_price(
        self,
        order_id:         str,
        ref_ltp:          float,
        strategy_name:    str,
        symbol:           str,
        transaction_type: str,
    ) -> Tuple[str, float]:
        """
        Fetch the actual average_price from order_history for a COMPLETE order.
        Falls back to ref_ltp if history is unavailable.
        """
        fill_price = ref_ltp  # safe fallback
        try:
            history = self._hub.kite.order_history(order_id)
            for h in reversed(history):
                ap = h.get("average_price")
                if h.get("status", "").upper() == "COMPLETE" and ap:
                    fill_price = float(ap)
                    break
        except Exception as exc:
            log.warning(
                f"[Router][LIVE] Could not fetch fill price for {order_id}: {exc} "
                f"— falling back to ref_ltp={ref_ltp:.2f}"
            )

        log.info(
            f"[Router][LIVE] {transaction_type} CONFIRMED ✓ | "
            f"{strategy_name} | {symbol} | order_id={order_id} | "
            f"fill_price={fill_price:.2f} (ref_ltp={ref_ltp:.2f})"
        )
        return order_id, fill_price

    def _log_order_failure(
        self,
        transaction_type: str,
        strategy_name:    str,
        symbol:           str,
        qty:              int,
        ltp:              float,
        order_id:         str,
        status:           str,
        attempt:          int,
    ):
        """Log a structured error for an order in a bad terminal state."""
        log.error(
            f"\n{'!'*60}\n"
            f"[Router][LIVE] {transaction_type} {status} on attempt {attempt}/2\n"
            f"  Strategy  : {strategy_name}\n"
            f"  Symbol    : {symbol}  qty={qty}\n"
            f"  ref_ltp   : {ltp:.2f}  market_protection={MARKET_PROTECTION}\n"
            f"  order_id  : {order_id}\n"
            + (
                f"  → *** SELL FAILED — CHECK ZERODHA CONSOLE AND "
                f"SQUARE OFF MANUALLY IF POSITION IS OPEN! ***\n"
                if transaction_type == "SELL" else
                f"  → Possible cause: insufficient funds, circuit filter, or OI limit.\n"
                f"  → Check Zerodha console for rejection reason.\n"
            )
            + f"{'!'*60}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    #  Internal — full order status confirmation (used for retry attempt)
    # ─────────────────────────────────────────────────────────────────────────

    def _confirm_order(
        self,
        order_id:      str,
        timeout_sec:   float = 15.0,
        poll_interval: float = 0.5,
    ) -> str:
        """
        Poll kite.order_history(order_id) until the order reaches a terminal state
        or the timeout expires.

        Terminal states returned by NSE via Zerodha:
          "COMPLETE"  → exchange filled the order
          "REJECTED"  → exchange rejected
          "CANCELLED" → order was cancelled

        Non-terminal states (keep polling):
          "PUT ORDER REQ RECEIVED", "VALIDATION PENDING", "OPEN PENDING",
          "OPEN", "TRIGGER PENDING", "MODIFY PENDING", "CANCEL PENDING"

        Returns one of:
          "COMPLETE", "REJECTED", "CANCELLED", "TIMEOUT", "TOKEN_EXPIRED"
        """
        TERMINAL = {"COMPLETE", "REJECTED", "CANCELLED"}
        deadline = time.monotonic() + timeout_sec
        attempt  = 0

        while time.monotonic() < deadline:
            attempt += 1
            try:
                history = self._hub.kite.order_history(order_id)
                if not history:
                    log.warning(
                        f"[Router] order_history({order_id}) returned empty list "
                        f"(attempt {attempt}) — retrying"
                    )
                    time.sleep(poll_interval)
                    continue

                status = history[-1].get("status", "").upper()
                log.debug(
                    f"[Router] order_history({order_id}) attempt={attempt} "
                    f"status={status}"
                )

                if status in TERMINAL:
                    log.info(
                        f"[Router] order_history({order_id}) → {status} "
                        f"after {attempt} poll(s)"
                    )
                    return status

                if attempt == 1 or attempt % 5 == 0:
                    log.info(
                        f"[Router] Waiting for order {order_id} | "
                        f"status={status!r} | attempt={attempt}"
                    )

            except Exception as exc:
                if self._is_token_exception(exc):
                    log.error(
                        f"\n{'!'*60}\n"
                        f"[Router][TOKEN EXPIRED] Cannot confirm order {order_id}: {exc}\n"
                        f"  → Access token expired mid-session during order confirmation.\n"
                        f"  → The order may or may not have been filled.\n"
                        f"  → CHECK ZERODHA CONSOLE IMMEDIATELY for open positions.\n"
                        f"  → Restart the bot with a fresh token before next trade.\n"
                        f"{'!'*60}"
                    )
                    return "TOKEN_EXPIRED"

                log.warning(
                    f"[Router] order_history({order_id}) poll error "
                    f"(attempt {attempt}): {exc} — retrying"
                )

            time.sleep(poll_interval)

        log.error(
            f"\n{'!'*60}\n"
            f"[Router][LIVE] Order {order_id} status poll TIMED OUT after {timeout_sec}s\n"
            f"  Polled {attempt} time(s). Order did not reach COMPLETE/REJECTED.\n"
            f"  → The order may be pending, partially filled, or stuck.\n"
            f"  → CHECK ZERODHA CONSOLE for this order_id and act manually.\n"
            f"  → Bot treats this as FAILED to prevent phantom position management.\n"
            f"{'!'*60}"
        )
        return "TIMEOUT"

    # ─────────────────────────────────────────────────────────────────────────
    #  Internal — token / session expiry detection
    # ─────────────────────────────────────────────────────────────────────────

    def _is_token_exception(self, exc: Exception) -> bool:
        """
        Return True if the exception indicates an expired or invalid Zerodha
        access token (HTTP 403 / TokenException).
        """
        exc_class = type(exc).__name__.lower()
        exc_msg   = str(exc).lower()

        return (
            "tokenexception"          in exc_class
            or "403"                  in exc_msg
            or "invalid access token" in exc_msg
            or "token is invalid"     in exc_msg
            or ("access token"        in exc_msg and "invalid" in exc_msg)
        )

    # ─────────────────────────────────────────────────────────────────────────
    #  Sell with retry (SL / trigger exit safety net)
    # ─────────────────────────────────────────────────────────────────────────

    def _is_position_open(self, symbol: str) -> bool:
        """
        Query kite.positions() and return True if a non-zero net position
        exists for `symbol`. On any API error returns True conservatively.
        """
        try:
            kite = self._hub.kite
            if kite is None:
                log.warning("[Router] _is_position_open: kite is None — assuming open")
                return True

            positions = kite.positions()
            net = positions.get("net", [])
            for pos in net:
                if pos.get("tradingsymbol") == symbol:
                    qty = pos.get("quantity") or pos.get("net_quantity", 0)
                    if qty and int(qty) != 0:
                        log.debug(
                            f"[Router] Position check: {symbol} qty={qty} → OPEN"
                        )
                        return True
            log.info(
                f"[Router] Position check: {symbol} not found or qty=0 → CLOSED"
            )
            return False

        except Exception as exc:
            if self._is_token_exception(exc):
                log.error(
                    f"[Router] _is_position_open: TOKEN EXPIRED while checking {symbol}. "
                    f"Assuming open — CHECK ZERODHA CONSOLE."
                )
            else:
                log.warning(
                    f"[Router] _is_position_open API error for {symbol}: {exc} "
                    f"— assuming open (safe default)"
                )
            return True

    def place_sell_with_retry(
        self,
        strategy_name: str,
        symbol:        str,
        token:         int,
        qty:           int,
        ltp:           float,
        live_mode:     bool,
        max_retries:   int = 3,
    ) -> Optional[Tuple[str, float]]:
        """
        Place a SELL (exit) order with automatic outer retry when the position
        is confirmed still open after an inner failure.

        Note: place_sell() itself already does one internal quick-check→cancel→retry
        cycle (design point 4). This outer retry is a separate safety net for
        cases where both internal attempts failed and the position is still open.

        Flow for each attempt (1 … max_retries):
          1. Call place_sell() — returns (order_id, fill_price) on success.
          2. On success → return immediately.
          3. On failure:
             a. Call kite.positions() to check if position is still open.
             b. If CLOSED → stop (closed by a prior attempt or auto square-off).
             c. If OPEN   → refresh LTP from WebSocket and retry.
          4. After max_retries failures with position still open → MANUAL alert.
        """
        for attempt in range(1, max_retries + 1):

            log.info(
                f"[Router] SELL attempt {attempt}/{max_retries} | "
                f"{strategy_name} | {symbol} | qty={qty} | ltp={ltp:.2f}"
            )

            result = self.place_sell(
                strategy_name, symbol, token, qty, ltp, live_mode
            )

            if result is not None:
                if attempt > 1:
                    log.info(
                        f"[Router] SELL succeeded on attempt {attempt}/{max_retries} ✓ | "
                        f"{strategy_name} | {symbol}"
                    )
                return result

            if not live_mode:
                return None

            log.warning(
                f"[Router] SELL attempt {attempt}/{max_retries} FAILED | "
                f"{strategy_name} | {symbol} — checking live position status..."
            )

            still_open = self._is_position_open(symbol)

            if not still_open:
                log.warning(
                    f"\n{'='*60}\n"
                    f"[Router] {symbol} position is ALREADY CLOSED.\n"
                    f"  Closed by a prior attempt, auto square-off, or the exchange.\n"
                    f"  No further SELL retries needed.\n"
                    f"{'='*60}"
                )
                return None

            if attempt < max_retries:
                fresh = self._hub.last_price(token)
                if fresh and fresh != ltp:
                    log.info(
                        f"[Router] LTP refreshed for retry: {ltp:.2f} → {fresh:.2f}"
                    )
                    ltp = fresh
                log.warning(
                    f"[Router] Position STILL OPEN — "
                    f"retrying (attempt {attempt + 1}/{max_retries}) | "
                    f"{strategy_name} | {symbol}"
                )
            else:
                log.error(
                    f"\n{'!'*60}\n"
                    f"[Router][CRITICAL] SELL FAILED after {max_retries} attempts "
                    f"— position STILL OPEN!\n"
                    f"  Strategy  : {strategy_name}\n"
                    f"  Symbol    : {symbol}  qty={qty}\n"
                    f"  Last LTP  : {ltp:.2f}\n"
                    f"  → *** SQUARE OFF MANUALLY IN ZERODHA CONSOLE NOW! ***\n"
                    f"  → Navigate to Positions → {symbol} → Exit\n"
                    f"{'!'*60}"
                )

        return None

    # ─────────────────────────────────────────────────────────────────────────
    #  Internal — paper simulation
    # ─────────────────────────────────────────────────────────────────────────

    def _paper_fill(
        self,
        side:          str,
        strategy_name: str,
        symbol:        str,
        ltp:           float,
    ) -> Tuple[str, float]:
        """
        Simulate an order fill in paper mode.
        Returns (fake_id, ltp) — same tuple shape as live mode.
        """
        fake_id = f"PAPER-{int(time.time() * 1000)}"
        log.info(
            f"[Router][PAPER] {side} simulated | "
            f"{strategy_name} | {symbol} | ltp={ltp:.2f} | id={fake_id}"
        )
        return fake_id, ltp
