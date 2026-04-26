"""
core/base_strategy.py

Abstract base class for every trading strategy.

All strategies MUST implement:
   name              unique string identifier
   on_tick()         called on every index tick
   on_candle()       called when a 5-min index candle closes
   on_option_tick()  called when subscribed option has a tick
   pre_market()      called once with shared PreMarketData
   eod_summary()     called at 3:30 PM for logging

MarketHub calls these methods. Strategy never touches Kite or WebSocket
directly — it uses MarketHub.subscribe() to add option tokens, and
MarketHub.order_router to place buy/sell orders.

LIVE_MODE per strategy
──────────────────────
Each subclass declares a class-level flag:

    LIVE_MODE = False   # change to True to enable real orders

When False: all trades are paper-simulated (no Kite API calls).
When True:  real MARKET orders placed via core/order_router.py.

Only ONE strategy should have LIVE_MODE=True at a time, so the global
trade slot in OrderRouter prevents simultaneous live positions.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional


class BaseStrategy(ABC):

    # Subclasses override this to go live.
    # Keep False in every strategy file until explicitly switching to live.
    LIVE_MODE: bool = False

    def __init__(self, market_hub):
        """
        market_hub: reference to MarketHub instance.
        Use market_hub.subscribe(token)              to subscribe option tokens.
        Use market_hub.last_price(token)             to get live price.
        Use market_hub.order_router.place_buy/sell() to place orders.
        """
        self._hub = market_hub

    # ── Abstract interface ────────────────────────────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique strategy name. Used in log prefixes and CSV filename."""
        ...

    @abstractmethod
    def pre_market(self, premarket_data, instruments) -> bool:
        """
        Called once after PreMarketData is ready (around 9:00–9:14 AM).
        Returns True to run today, False to skip.
        """
        ...

    @abstractmethod
    def on_tick(self, price: float, ts: datetime, tick_ts: datetime):
        """
        Called on every raw index tick.
        ts      = IST wall-clock time of processing
        tick_ts = exchange timestamp from the tick (use for candle alignment)
        """
        ...

    @abstractmethod
    def on_candle(self, candle: dict, ts: datetime):
        """
        Called when a 5-minute index candle closes.
        candle: {ts, open, high, low, close, volume}
        """
        ...

    def on_option_tick(self, token: int, price: float, ts: datetime, tick_ts: datetime = None):
        """
        Called when a subscribed option token has a new tick.
        Override in strategies that need live option prices.
        Default: no-op.
        """
        pass

    @abstractmethod
    def eod_summary(self):
        """Called at 3:30 PM. Print/log end-of-day PnL and trade summary."""
        ...

    # ── Option subscription helpers ───────────────────────────────────────────

    def subscribe_option(self, token: int):
        """Subscribe an option token to the WebSocket via MarketHub."""
        self._hub.subscribe(token)

    def unsubscribe_option(self, token: int):
        """Unsubscribe when position is closed."""
        self._hub.unsubscribe(token)

    def get_price(self, token: int) -> Optional[float]:
        """Get last known price for any token."""
        return self._hub.last_price(token)

    def get_price_ts(self, token: int) -> Optional[datetime]:
        """Get timestamp of when the last price tick arrived (IST)."""
        return self._hub.last_price_ts(token)

    # ── Order helpers (thin wrappers over OrderRouter) ────────────────────────

    def _acquire_slot(self) -> bool:
        """
        Try to acquire the global live-trade slot before entering.
        Paper mode: always returns True (no slot contention).
        Live mode:  True only if no other live strategy has an open position.
        """
        return self._hub.order_router.acquire_slot(self.name, self.LIVE_MODE)

    def _release_slot(self):
        """Release the slot after a position is fully closed."""
        self._hub.order_router.release_slot(self.name, self.LIVE_MODE)

    def _place_buy(self, symbol: str, token: int, qty: int, ltp: float) -> Optional[str]:
        """
        Place a BUY (entry) order.
        Paper mode: logged simulation, no Kite call.
        Live mode:  REGULAR MARKET MIS on NFO via Kite.
        Returns order_id string, or None on failure.
        """
        return self._hub.order_router.place_buy(
            self.name, symbol, token, qty, ltp, self.LIVE_MODE
        )

    def _place_sell(self, symbol: str, token: int, qty: int, ltp: float) -> Optional[str]:
        """
        Place a SELL (exit) order.
        Paper mode: logged simulation, no Kite call.
        Live mode:  REGULAR MARKET MIS on NFO via Kite.
        Returns order_id string, or None on failure.
        """
        return self._hub.order_router.place_sell(
            self.name, symbol, token, qty, ltp, self.LIVE_MODE
        )
