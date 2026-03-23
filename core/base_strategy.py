"""
core/base_strategy.py

Abstract base class for every trading strategy.

All strategies MUST implement:
   name           unique string identifier
   on_tick()      called on every index tick
   on_candle()    called when a 5-min index candle closes
   on_option_tick()  called when subscribed option has a tick
   pre_market()   called once with shared PreMarketData
   eod_summary()  called at 3:30 PM for logging

MarketHub calls these methods. Strategy never touches Kite or WebSocket
directly  it uses MarketHub.subscribe() to add option tokens.

Adding a new strategy:
  1. Create strategies/my_strategy.py
  2. class MyStrategy(BaseStrategy): ... implement methods above
  3. Add it to STRATEGIES list in t.py  that's it.
"""

from abc import ABC, abstractmethod
from datetime import datetime


class BaseStrategy(ABC):

    def __init__(self, market_hub):
        """
        market_hub: reference to MarketHub instance.
        Use market_hub.subscribe(token) to subscribe option tokens.
        Use market_hub.last_price(token) to get live price of any token.
        """
        self._hub = market_hub

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique strategy name. Used in log prefixes and CSV filename."""
        ...

    @abstractmethod
    def pre_market(self, premarket_data, instruments) -> bool:
        """
        Called once after PreMarketData is ready (around 9:009:14 AM).
        premarket_data: PreMarketData instance (vix, pcr, prev_close, ema200, etc.)
        instruments:    InstrumentStore instance (get_option_token, get_atm_strike)
        Returns True to run today, False to skip.
        """
        ...

    @abstractmethod
    def on_tick(self, price: float, ts: datetime, tick_ts: datetime):
        """
        Called on every raw index tick.
        ts      = datetime.now() at time of processing (wall clock)
        tick_ts = exchange timestamp from the tick itself (use THIS for candle buckets)
        Use for opening range tracking, live price monitoring.
        Spike strategy uses tick_ts for its 8-second candle alignment.
        """
        ...

    @abstractmethod
    def on_candle(self, candle: dict, ts: datetime):
        """
        Called when a 5-minute index candle closes.
        candle dict: {ts, open, high, low, close, volume}
        Use for indicator computation and entry signals.
        """
        ...

    def on_option_tick(self, token: int, price: float, ts: datetime):
        """
        Called when a subscribed option token has a new tick.
        Override in strategies that need live option prices (premium SL, etc.)
        Default: no-op.
        """
        pass

    @abstractmethod
    def eod_summary(self):
        """
        Called at 3:30 PM. Print/log end-of-day PnL and trade summary.
        """
        ...

    #  Convenience helpers (strategies use these) 

    def subscribe_option(self, token: int):
        """Subscribe an option token to the WebSocket via MarketHub."""
        self._hub.subscribe(token)

    def unsubscribe_option(self, token: int):
        """Unsubscribe when position is closed."""
        self._hub.unsubscribe(token)

    def get_price(self, token: int) -> float | None:
        """Get last known price for any token."""
        return self._hub.last_price(token)

    def get_price_ts(self, token: int):
        """Get timestamp of when the last price tick arrived (IST)."""
        return self._hub.last_price_ts(token)
