"""
core/order_router.py

Central order execution layer — single point for all live order placement.

═══════════════════════════════════════════════════════════════════
  DESIGN DECISIONS (read before modifying)
═══════════════════════════════════════════════════════════════════

1. LIMIT orders with slippage buffer — NOT plain MARKET orders

   Zerodha's API now REJECTS plain MARKET orders for options with:
     "Market orders without market protection are not allowed via API."

   Fix: use ORDER_TYPE_LIMIT with a small slippage buffer above LTP for BUY
   and below LTP for SELL. This guarantees near-instant fill on liquid ATM
   strikes at 9:15 AM while satisfying Zerodha's API requirement.

   Buffer:
     BUY  → price = round_tick(ltp × 1.02)   (+2% above LTP)
     SELL → price = round_tick(ltp × 0.98)   (−2% below LTP)

   Tick size for NFO options is ₹0.05 — all prices rounded to nearest tick.

   Why not SL-M / IOC / BO / CO:
     BO  (Bracket Orders)  — BLOCKED for all FnO since 2021 (SEBI directive)
     CO  (Cover Orders)    — BLOCKED for FnO
     SL-M                  — NOT available for options on NSE/Zerodha
     SL Limit              — can gap through trigger entirely
     IOC LIMIT             — rejection risk at 9:15 AM gap-open (stale price)
     MARKET                — rejected by Zerodha API ("market protection" error)
     LIMIT + buffer        — fills in ms on liquid ATM strikes ✓  ← used here

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
   When True  → real LIMIT order via kite.place_order().

   Default: ALL strategies are paper. Change only the one you want to go live.

4. Order status confirmation + actual fill price

   After kite.place_order() returns an order_id, we poll kite.order_history()
   until the order reaches a terminal state (COMPLETE / REJECTED / CANCELLED)
   or a timeout expires.

   Without this, NSE could silently reject an option order (circuit filter,
   no quotes at that ms) and the bot would manage a phantom SL and eventually
   fire a SELL against a position that never existed.

   _confirm_order() returns a tuple:
     (status, avg_price)

   Where status is one of:
     "COMPLETE"      → fill confirmed, avg_price is the actual exchange fill price
     "REJECTED"      → exchange rejected; avg_price is 0.0
     "CANCELLED"     → order was cancelled; avg_price is 0.0
     "TIMEOUT"       → did not reach terminal state in time; avg_price is 0.0
     "TOKEN_EXPIRED" → access token invalid mid-session; avg_price is 0.0

   place_buy() / place_sell() return a tuple:
     (order_id, fill_price)  on success
     None                    on failure

   Callers must unpack:
     result = self._place_buy(sym, token, qty, ltp)
     if result is None:
         # handle failure
     order_id, fill_price = result

   Using the actual fill_price (from order_history average_price) for SL
   calculation is more accurate than the pre-order LTP, especially at
   volatile 9:15 AM opens where the price can move 20–30 pts during the
   ~15s _confirm_order() polling window.

5. Token / session expiry mid-session

   If the Zerodha access token expires while the bot is running (rare but
   possible after midnight refresh), any Kite REST call raises
   kiteconnect.exceptions.TokenException (HTTP 403).

   Fix: _is_token_exception() inspects the exception type name and message.
   When detected:
     • A loud, actionable error is logged with exact remediation steps.
     • None is returned so the caller aborts the trade / releases the slot.
     • No further Kite calls are attempted for that order.

═══════════════════════════════════════════════════════════════════
"""

import logging
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple

log = logging.getLogger("core.router")

_IST = timezone(timedelta(hours=5, minutes=30))

# ── Slippage buffer for LIMIT orders ─────────────────────────────────────────
# BUY  limit = LTP × (1 + BUY_SLIP)   — pay up to 2% above last price
# SELL limit = LTP × (1 − SELL_SLIP)  — accept down to 2% below last price
# Both rounded to NFO tick size (₹0.05).
# Increase if fills are missed on highly volatile opens; decrease to save cost.
BUY_SLIP   = 0.1   # 10%
SELL_SLIP  = 0.1   # 10%
TICK_SIZE  = 0.05  # NFO options tick


def _now_ist() -> datetime:
    return datetime.now(tz=_IST).replace(tzinfo=None)


def _limit_price(ltp: float, side: str) -> float:
    """
    Round ltp ± slippage buffer to nearest NFO tick (₹0.05).

    BUY  → ltp × 1.10 rounded UP   (ensures we're above the ask)
    SELL → ltp × 0.90 rounded DOWN  (ensures we're below the bid)

    Minimum price floored at TICK_SIZE to avoid zero/negative prices on
    very cheap OTM options.
    """
    if side == "BUY":
        raw = ltp * (1 + BUY_SLIP)
        ticked = round(raw / TICK_SIZE) * TICK_SIZE
    else:
        raw = ltp * (1 - SELL_SLIP)
        ticked = round(raw / TICK_SIZE) * TICK_SIZE

    return max(ticked, TICK_SIZE)


class OrderRouter:
    """
    Single point for all order placement across all strategies.
    Thread-safe. Enforces one-live-trade-at-a-time when in live mode.

    place_buy() and place_sell() return a (order_id, fill_price) tuple on
    success, or None on failure. Callers must check for None before unpacking.

    Usage in a strategy:

        # Entry
        result = self._place_buy(symbol, token, qty, ltp)
        if result is None:
            return  # order failed
        order_id, fill_price = result
        sl = fill_price - INITIAL_SL_BUFFER   # use actual fill price for SL

        # Exit
        result = self._place_sell(symbol, token, qty, ltp)
        # result may be None if sell failed — handle accordingly
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
    #  Order placement
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
        live_mode=True  → LIMIT MIS order via Kite at LTP + buffer;
                          returns (order_id, actual_fill_price) ONLY when
                          exchange confirms COMPLETE.

        Returns (order_id, fill_price) on success, or None on failure.
        fill_price is the actual average price from the exchange (live) or
        the simulated LTP (paper). Use this for SL calculation, not the
        pre-order LTP.
        """
        if live_mode:
            return self._live_order(
                strategy_name, symbol, qty, ltp,
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
        live_mode=True  → LIMIT MIS order via Kite at LTP − buffer;
                          returns (order_id, actual_fill_price) ONLY when
                          exchange confirms COMPLETE.

        Returns (order_id, fill_price) on success, or None on failure.
        """
        if live_mode:
            return self._live_order(
                strategy_name, symbol, qty, ltp,
                transaction_type="SELL",
            )
        return self._paper_fill("SELL", strategy_name, symbol, ltp)

    # ─────────────────────────────────────────────────────────────────────────
    #  Internal — live execution
    # ─────────────────────────────────────────────────────────────────────────

    def _live_order(
        self,
        strategy_name:    str,
        symbol:           str,
        qty:              int,
        ltp:              float,
        transaction_type: str,   # "BUY" or "SELL"
    ) -> Optional[Tuple[str, float]]:
        """
        Place a LIMIT MIS order on NFO and confirm it reached the exchange.

        Returns (order_id, fill_price) on COMPLETE, None on any failure.

        fill_price is taken from order_history[last]["average_price"] — the
        actual exchange-confirmed average fill price. This is more accurate
        than the pre-order LTP for SL calculation, especially at volatile
        9:15 AM opens.

        Why LIMIT with buffer (not MARKET):
          - Zerodha API rejects plain MARKET orders for options:
              "Market orders without market protection are not allowed via API."
          - LIMIT at LTP ± buffer fills in milliseconds on liquid ATM strikes.
          - Buffer absorbs spread and slippage at volatile 9:15 AM opens.
          - Tick-rounded to ₹0.05 as required by NSE for NFO options.

        Flow:
          1. Validate kite session (None check).
          2. Compute limit_price = _limit_price(ltp, side).
          3. place_order() → order_id.
          4. _confirm_order() polls order_history() until COMPLETE / REJECTED /
             CANCELLED / TIMEOUT / TOKEN_EXPIRED.
          5. Return (order_id, fill_price) only on COMPLETE; None otherwise.
        """
        kite = self._hub.kite
        if kite is None:
            log.error(
                f"[Router][LIVE] Kite not loaded — "
                f"cannot place {transaction_type} for {strategy_name}"
            )
            return None

        limit_px = _limit_price(ltp, transaction_type)

        try:
            txn = (
                kite.TRANSACTION_TYPE_BUY
                if transaction_type == "BUY"
                else kite.TRANSACTION_TYPE_SELL
            )
            order_id = kite.place_order(
                variety          = kite.VARIETY_REGULAR,
                exchange         = kite.EXCHANGE_NFO,
                tradingsymbol    = symbol,
                transaction_type = txn,
                quantity         = qty,
                product          = kite.PRODUCT_MIS,
                order_type       = kite.ORDER_TYPE_LIMIT,   # ← LIMIT, not MARKET
                price            = limit_px,                # ← LTP ± buffer
            )
            log.info(
                f"[Router][LIVE] {transaction_type} PLACED | "
                f"{strategy_name} | {symbol} | qty={qty} | "
                f"ref_ltp={ltp:.2f} | limit_px={limit_px:.2f} | order_id={order_id}"
            )

        except Exception as exc:
            # ── Token / session expiry ─────────────────────────────────────────
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
                    f"[Router][LIVE] {transaction_type} FAILED | "
                    f"{strategy_name} | {symbol}: {exc}"
                )
            return None

        # ── Confirm the order actually reached the exchange ────────────────────
        # place_order() returning an order_id does NOT mean the exchange accepted
        # the order. NSE can reject LIMIT option orders if:
        #   • Insufficient funds / margin
        #   • The strike has a circuit filter at that moment
        #   • The order hits OI / position limits
        # Without confirmation, the bot would manage a phantom SL and later
        # fire a SELL against a position that never existed.
        #
        # _confirm_order() also returns the actual average fill price from
        # order_history so callers can use the real fill price for SL calculation
        # instead of the pre-order LTP (which may be stale after ~15s of polling).
        status, fill_price = self._confirm_order(str(order_id), fallback_price=ltp)

        if status == "COMPLETE":
            log.info(
                f"[Router][LIVE] {transaction_type} CONFIRMED ✓ | "
                f"{strategy_name} | {symbol} | "
                f"fill_price={fill_price:.2f} | order_id={order_id}"
            )
            return str(order_id), fill_price

        elif status == "TOKEN_EXPIRED":
            log.error(
                f"[Router][LIVE] {transaction_type} order {order_id} placed but "
                f"confirmation poll lost token. Check Zerodha console immediately."
            )
            return None

        else:
            # REJECTED, CANCELLED, or TIMEOUT
            log.error(
                f"\n{'!'*60}\n"
                f"[Router][LIVE] {transaction_type} NOT COMPLETE — status={status}\n"
                f"  Strategy  : {strategy_name}\n"
                f"  Symbol    : {symbol}  qty={qty}\n"
                f"  limit_px  : {limit_px:.2f}  (ref_ltp={ltp:.2f})\n"
                f"  order_id  : {order_id}\n"
                f"  → Treating as FAILED. Slot released by caller.\n"
                f"  → No phantom SL will be tracked.\n"
                + (
                    f"  → *** SELL FAILED — CHECK ZERODHA CONSOLE AND "
                    f"SQUARE OFF MANUALLY IF POSITION IS OPEN! ***\n"
                    if transaction_type == "SELL" else
                    f"  → Possible cause: insufficient funds, circuit filter, or OI limit.\n"
                    f"  → Check Zerodha console for rejection reason.\n"
                )
                + f"{'!'*60}"
            )
            return None

    # ─────────────────────────────────────────────────────────────────────────
    #  Internal — order status confirmation
    # ─────────────────────────────────────────────────────────────────────────

    def _confirm_order(
        self,
        order_id:       str,
        fallback_price: float = 0.0,
        timeout_sec:    float = 15.0,
        poll_interval:  float = 0.5,
    ) -> Tuple[str, float]:
        """
        Poll kite.order_history(order_id) until the order reaches a terminal
        state or the timeout expires.

        Returns a tuple: (status, avg_price)

          status values:
            "COMPLETE"      → exchange filled the order; avg_price is real fill
            "REJECTED"      → exchange rejected; avg_price is 0.0
            "CANCELLED"     → order was cancelled; avg_price is 0.0
            "TIMEOUT"       → timed out; avg_price is 0.0
            "TOKEN_EXPIRED" → session ended mid-poll; avg_price is 0.0

          avg_price:
            On COMPLETE — taken from order_history[last]["average_price"].
            Falls back to fallback_price (pre-order LTP) if average_price is
            missing or zero in the history response (should not happen on
            Zerodha for a COMPLETE order, but defensive).
            0.0 for all non-COMPLETE statuses.

        Terminal states returned by NSE via Zerodha:
          "COMPLETE"  → exchange filled the order (normal outcome for LIMIT + buffer)
          "REJECTED"  → exchange rejected (insufficient funds, circuit, OI limit)
          "CANCELLED" → order was cancelled (unusual for LIMIT)

        Non-terminal states (keep polling):
          "PUT ORDER REQ RECEIVED"  → order received by Zerodha OMS
          "VALIDATION PENDING"      → validation in progress
          "OPEN PENDING"            → awaiting exchange acknowledgement
          "OPEN"                    → acknowledged, waiting for match
          "TRIGGER PENDING"         → SL trigger not yet hit (not applicable here)
          "MODIFY PENDING"          → modification in progress
          "CANCEL PENDING"          → cancellation in progress
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

                last   = history[-1]
                status = last.get("status", "").upper()
                log.debug(
                    f"[Router] order_history({order_id}) attempt={attempt} "
                    f"status={status}"
                )

                if status in TERMINAL:
                    # Extract actual fill price on COMPLETE
                    avg_price = 0.0
                    if status == "COMPLETE":
                        raw_avg = last.get("average_price") or 0.0
                        avg_price = float(raw_avg) if raw_avg else fallback_price
                        if avg_price <= 0:
                            # Defensive fallback — should not happen for COMPLETE
                            avg_price = fallback_price
                            log.warning(
                                f"[Router] order {order_id} COMPLETE but "
                                f"average_price missing — using fallback {fallback_price:.2f}"
                            )

                    log.info(
                        f"[Router] order_history({order_id}) → {status} "
                        f"after {attempt} poll(s)"
                        + (f" | avg_fill={avg_price:.2f}" if status == "COMPLETE" else "")
                    )
                    return status, avg_price

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
                    return "TOKEN_EXPIRED", 0.0

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
        return "TIMEOUT", 0.0

    # ─────────────────────────────────────────────────────────────────────────
    #  Internal — token / session expiry detection
    # ─────────────────────────────────────────────────────────────────────────

    def _is_token_exception(self, exc: Exception) -> bool:
        """
        Return True if the exception indicates an expired or invalid Zerodha
        access token (HTTP 403 / TokenException).

        kiteconnect raises kiteconnect.exceptions.TokenException for:
          • Invalid access token
          • Expired access token (tokens expire after Zerodha's daily midnight reset)
          • Revoked access token

        We inspect both the exception class name and the message string because:
          • Some network libraries wrap the 403 in a generic requests.HTTPError
          • The message text is consistent across kiteconnect library versions
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
    #  Internal — paper simulation
    # ─────────────────────────────────────────────────────────────────────────

    def _paper_fill(
        self,
        side:          str,   # "BUY" or "SELL"
        strategy_name: str,
        symbol:        str,
        ltp:           float,
    ) -> Tuple[str, float]:
        fake_id = f"PAPER-{int(time.time() * 1000)}"
        log.info(
            f"[Router][PAPER] {side} simulated | "
            f"{strategy_name} | {symbol} | ltp={ltp:.2f} | id={fake_id}"
        )
        # Paper fill: return ltp as fill_price (no slippage simulated)
        return fake_id, ltp

