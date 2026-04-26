"""
core/order_router.py

Central order execution layer — single point for all live order placement.

═══════════════════════════════════════════════════════════════════
  DESIGN DECISIONS (read before modifying)
═══════════════════════════════════════════════════════════════════

1. MARKET orders — NOT IOC, NOT SL orders on exchange

   For BankNifty ATM options, a REGULAR MARKET order fills in milliseconds.
   IOC LIMIT risks outright rejection at 9:15 AM gap-open (spread 5–15 pts,
   limit price is stale before it hits the exchange).

   Zerodha / NSE constraints for options:
     BO  (Bracket Orders)  — BLOCKED for all FnO since 2021 (SEBI directive)
     CO  (Cover Orders)    — BLOCKED for FnO
     SL-M                  — NOT available for options; only equity/futures
     SL Limit              — available but can gap through trigger entirely
     MARKET (REGULAR+MIS)  — available ✓  ← only this is used here

   Exchange SL orders are therefore unreliable for options in every possible form.
   We monitor SL on every WebSocket tick in software and fire a MARKET sell the
   moment price breaches. This is faster and more dependable.

2. One-live-trade-at-a-time slot

   Only one strategy can hold a live position at a time.  This prevents:
     - Double margin consumption across 4 strategies
     - Conflicting CE/PE positions (SPIKE long CE + BB_STOCH long PE = disaster)
     - Capital exhaustion

   The slot is only enforced when LIVE_MODE=True.  Paper-mode strategies never
   compete for the slot and trade independently (so you can keep reviewing paper
   data from all 4 strategies while only SPIKE is live, for example).

3. Per-strategy LIVE_MODE flag

   Each strategy file exposes:
       LIVE_MODE = False   # change to True to enable real orders for that strategy

   When False → paper fill (no Kite API call, simulated fill at LTP).
   When True  → real MARKET order via kite.place_order().

   Default: ALL strategies are paper.  Change only the one you want to go live.

4. Order status confirmation (FIX — was missing)

   After kite.place_order() returns an order_id, we poll kite.order_history()
   until the order reaches a terminal state (COMPLETE / REJECTED / CANCELLED)
   or a timeout expires.

   Without this, NSE could silently reject an option MARKET order (circuit
   filter, no quotes at that ms) and the bot would manage a phantom SL and
   eventually fire a SELL against a position that never existed.

   _confirm_order() returns one of:
     "COMPLETE"      → fill confirmed, caller may treat trade as open
     "REJECTED"      → exchange rejected; caller must abort entry / alert on sell
     "CANCELLED"     → order was cancelled (should not happen for MARKET)
     "TIMEOUT"       → did not reach terminal state in time; caller treats as failed
     "TOKEN_EXPIRED" → access token invalid mid-session; caller must abort + alert

5. Token / session expiry mid-session (FIX — was only checked at startup)

   If the Zerodha access token expires while the bot is running (rare but
   possible after midnight refresh), any Kite REST call raises
   kiteconnect.exceptions.TokenException (HTTP 403).

   Previously this surfaced only as a generic exception in the log, and
   the bot continued thinking kite was fine — it would keep retrying
   orders and managing phantom positions.

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
from typing import Optional

log = logging.getLogger("core.router")

_IST = timezone(timedelta(hours=5, minutes=30))


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
        oid = self._hub.order_router.place_buy(self.name, symbol, token, qty, ltp, LIVE_MODE)

        # Exit
        self._hub.order_router.place_sell(self.name, symbol, token, qty, ltp, LIVE_MODE)
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
    ) -> Optional[str]:
        """
        Place a BUY (entry) order.

        live_mode=False → paper fill logged; returns "PAPER-{ms}"
        live_mode=True  → REGULAR MARKET MIS via Kite; returns order_id string
                          ONLY when exchange confirms COMPLETE.

        Returns order_id / paper_id, or None on failure.
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
    ) -> Optional[str]:
        """
        Place a SELL (exit) order.

        live_mode=False → paper fill logged; returns "PAPER-{ms}"
        live_mode=True  → REGULAR MARKET MIS via Kite; returns order_id string
                          ONLY when exchange confirms COMPLETE.

        Returns order_id / paper_id, or None on failure.
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
    ) -> Optional[str]:
        """
        Place a REGULAR MARKET MIS order on NFO and confirm it reached the exchange.

        Why MARKET (not IOC LIMIT, not SL-M, not BO):
          - BO/CO: blocked for FnO by exchange since 2021.
          - SL-M:  not available for options on NSE/Zerodha.
          - IOC:   useful for futures/equity; for single-lot options it adds
                   rejection risk with zero benefit (MARKET fills in ms).
          - MARKET: fills immediately at best available price. Correct for
                   time-sensitive entries/exits on liquid ATM strikes.

        Flow:
          1. Validate kite session (None check).
          2. place_order() → order_id.
          3. _confirm_order() polls order_history() until COMPLETE / REJECTED /
             CANCELLED / TIMEOUT / TOKEN_EXPIRED.
          4. Return order_id only on COMPLETE; None on anything else.
        """
        kite = self._hub.kite
        if kite is None:
            log.error(
                f"[Router][LIVE] Kite not loaded — "
                f"cannot place {transaction_type} for {strategy_name}"
            )
            return None

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
                order_type       = kite.ORDER_TYPE_MARKET,
            )
            log.info(
                f"[Router][LIVE] {transaction_type} PLACED | "
                f"{strategy_name} | {symbol} | qty={qty} | "
                f"ref_ltp={ltp:.2f} | order_id={order_id}"
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
        # the order. NSE can reject MARKET option orders if:
        #   • The strike has a circuit filter at that moment
        #   • There are no quotes at that millisecond
        #   • The order hits OI / position limits
        # Without confirmation, the bot would manage a phantom SL and later
        # fire a SELL against a position that never existed.
        status = self._confirm_order(str(order_id))

        if status == "COMPLETE":
            log.info(
                f"[Router][LIVE] {transaction_type} CONFIRMED ✓ | "
                f"{strategy_name} | {symbol} | order_id={order_id}"
            )
            return str(order_id)

        elif status == "TOKEN_EXPIRED":
            # Detailed alert already logged inside _confirm_order.
            # The order may have filled — tell caller to check console.
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
                f"  order_id  : {order_id}\n"
                f"  → Treating as FAILED. Slot released by caller.\n"
                f"  → No phantom SL will be tracked.\n"
                + (
                    f"  → *** SELL FAILED — CHECK ZERODHA CONSOLE AND "
                    f"SQUARE OFF MANUALLY IF POSITION IS OPEN! ***\n"
                    if transaction_type == "SELL" else ""
                )
                + f"{'!'*60}"
            )
            return None

    # ─────────────────────────────────────────────────────────────────────────
    #  Internal — order status confirmation
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
          "COMPLETE"  → exchange filled the order (normal outcome for MARKET)
          "REJECTED"  → exchange rejected the order (circuit, no quotes, limit breach)
          "CANCELLED" → order was cancelled (unusual for MARKET)

        Non-terminal states (keep polling):
          "PUT ORDER REQ RECEIVED"  → order received by Zerodha OMS
          "VALIDATION PENDING"      → validation in progress
          "OPEN PENDING"            → awaiting exchange acknowledgement
          "OPEN"                    → acknowledged, waiting for match
          "TRIGGER PENDING"         → SL trigger not yet hit (not applicable here)
          "MODIFY PENDING"          → modification in progress
          "CANCEL PENDING"          → cancellation in progress

        Returns one of:
          "COMPLETE"      — normal success path
          "REJECTED"      — exchange rejected; entry must be aborted
          "CANCELLED"     — order was cancelled
          "TIMEOUT"       — timed out; treat as failed, check console
          "TOKEN_EXPIRED" — session ended mid-poll; check console immediately
        """
        TERMINAL = {"COMPLETE", "REJECTED", "CANCELLED"}
        deadline = time.monotonic() + timeout_sec
        attempt  = 0

        while time.monotonic() < deadline:
            attempt += 1
            try:
                history = self._hub.kite.order_history(order_id)
                # Zerodha returns a list of status entries; the last entry is current
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

                # Non-terminal — log first attempt and every 5th to avoid spam
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

        # Timeout reached — order status unknown
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

        kiteconnect raises kiteconnect.exceptions.TokenException for:
          • Invalid access token
          • Expired access token (tokens expire after Zerodha's daily midnight reset)
          • Revoked access token

        We inspect both the exception class name and the message string because:
          • Some network libraries wrap the 403 in a generic requests.HTTPError
          • The message text is consistent across kiteconnect library versions

        Note: a fresh token check at startup (hub.load_kite) is not enough.
        Tokens are valid until midnight IST. If the bot starts before midnight
        and runs past it (unusual for intraday, but possible for overnight debug
        runs), the token expires mid-session. This handler covers that edge case.
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
    ) -> str:
        fake_id = f"PAPER-{int(time.time() * 1000)}"
        log.info(
            f"[Router][PAPER] {side} simulated | "
            f"{strategy_name} | {symbol} | ltp={ltp:.2f} | id={fake_id}"
        )
        return fake_id
