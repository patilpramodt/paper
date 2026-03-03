"""
core/vwap.py

True session VWAP  accumulates from 9:15 AM.
One instance lives in MarketHub, broadcast to all strategies.

Why session VWAP matters:
   Institutional desks reference VWAP from session start (9:15).
   A rolling 21-candle VWAP resets constantly and is meaningless.
   Price above session VWAP = buyers in control all day.
   Price below session VWAP = sellers in control all day.
"""

import threading


class SessionVWAP:
    """
    Accumulates (price  volume) and volume from 9:15 AM.
    Reset once per day (called by MarketHub at session start).
    Thread-safe reads/writes.
    """

    def __init__(self):
        self._cum_vp  = 0.0
        self._cum_vol = 0
        self._value   = None
        self._lock    = threading.Lock()

    def reset(self):
        """Call at session start (9:15 AM) each day."""
        with self._lock:
            self._cum_vp  = 0.0
            self._cum_vol = 0
            self._value   = None

    def update(self, high: float, low: float, close: float, volume: int):
        """
        Update with every index tick.
        Typical price = (H + L + C) / 3.
        """
        tp = (high + low + close) / 3
        with self._lock:
            self._cum_vp  += tp * volume
            self._cum_vol += volume
            if self._cum_vol > 0:
                self._value = self._cum_vp / self._cum_vol

    @property
    def value(self) -> float | None:
        with self._lock:
            return self._value

    @property
    def ready(self) -> bool:
        with self._lock:
            return self._value is not None
