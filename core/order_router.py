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
        Place a REGULAR MARKET MIS order on NFO.

        Why MARKET (not IOC LIMIT, not SL-M, not BO):
          - BO/CO: blocked for FnO by exchange since 2021.
          - SL-M:  not available for options on NSE/Zerodha.
          - IOC:   useful for futures/equity; for single-lot options it adds
                   rejection risk with zero benefit (MARKET fills in ms).
          - MARKET: fills immediately at best available price. Correct for
                   time-sensitive entries/exits on liquid ATM strikes.
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
            return str(order_id)

        except Exception as exc:
            log.error(
                f"[Router][LIVE] {transaction_type} FAILED | "
                f"{strategy_name} | {symbol}: {exc}"
            )
            return None

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
