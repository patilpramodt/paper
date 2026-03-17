"""
core/vwap.py

True session VWAP — accumulates from 9:15 AM.
One instance lives in MarketHub, broadcast to all strategies.

Why session VWAP matters:
   Institutional desks reference VWAP from session start (9:15).
   A rolling 21-candle VWAP resets constantly and is meaningless.
   Price above session VWAP = buyers in control all day.
   Price below session VWAP = sellers in control all day.

FIX (Bug 3):
   BankNifty INDEX token (260105) is a computed index — it has NO traded
   volume. last_traded_quantity is always 0 on Kite WebSocket for index tokens.
   Original code accumulated (price × 0) forever → _value remained None.

   Fix: accept a proxy_weight parameter (default=1). MarketHub passes 1 per
   tick for the index token so VWAP accumulates tick-weighted prices.
   When real volume is available (e.g. futures token), pass actual qty.
"""

import threading


class SessionVWAP:
    """
    Accumulates (price × weight) and weight from 9:15 AM.
    Reset once per day (called by MarketHub at session start).
    Thread-safe reads/writes.

    FIX: update() now accepts an optional proxy_weight (default=1).
    For BankNifty INDEX (no traded qty), pass proxy_weight=1 so each tick
    contributes equally. For futures/equities with real volume, pass actual qty.
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

    def update(self, high: float, low: float, close: float, volume: int,
               proxy_weight: int = 1):
        """
        Update with every index tick.
        Typical price = (H + L + C) / 3.

        FIX (Bug 3): use proxy_weight=1 when volume=0 (index token).
        This ensures VWAP is always computed from the first tick onward.
        Pass actual volume when available (e.g. futures token).
        """
        tp = (high + low + close) / 3
        # Use actual volume if available, otherwise fall back to proxy_weight.
        # This lets futures/equities use real volume while index uses tick count.
        weight = volume if volume > 0 else proxy_weight
        with self._lock:
            self._cum_vp  += tp * weight
            self._cum_vol += weight
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
