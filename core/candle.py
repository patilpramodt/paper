"""
core/candle.py

Shared candle builders used by ALL strategies.
Both 5-minute (for ORB) and 8-second (for Spike) candles.
Each strategy gets its own instance per token  no sharing of state.
"""

import threading
from datetime import datetime


class CandleBuilder:
    """
    Converts raw ticks into OHLCV candles of any duration.

    Usage:
        cb = CandleBuilder(minutes=5)
        closed = cb.feed_tick(price, volume, timestamp)
        # Returns closed candle dict when a bar closes, else None

    The candle dict has keys:
        ts, open, high, low, close, volume
    """

    def __init__(self, minutes: int = 5):
        self.minutes        = minutes
        self.current_candle = None
        self.closed_candles = []
        self._lock          = threading.Lock()

    def _bar_start(self, ts: datetime) -> datetime:
        """Round timestamp down to nearest N-minute bar."""
        m = (ts.minute // self.minutes) * self.minutes
        return ts.replace(minute=m, second=0, microsecond=0)

    def feed_tick(self, price: float, volume: int, ts: datetime):
        """
        Feed one tick. Returns closed candle dict when a bar closes, else None.
        Thread-safe.
        """
        bar = self._bar_start(ts)
        with self._lock:
            if self.current_candle is None:
                self.current_candle = {
                    "ts": bar, "open": price, "high": price,
                    "low": price, "close": price, "volume": volume
                }
                return None

            if bar == self.current_candle["ts"]:
                c = self.current_candle
                c["high"]    = max(c["high"], price)
                c["low"]     = min(c["low"],  price)
                c["close"]   = price
                c["volume"] += volume
                return None
            else:
                closed = dict(self.current_candle)
                self.closed_candles.append(closed)
                self.current_candle = {
                    "ts": bar, "open": price, "high": price,
                    "low": price, "close": price, "volume": volume
                }
                return closed

    def get_closed(self) -> list:
        """Return copy of all closed candles."""
        with self._lock:
            return list(self.closed_candles)

    def get_all(self) -> list:
        """Return closed candles + current open candle."""
        with self._lock:
            result = list(self.closed_candles)
            if self.current_candle:
                result.append(dict(self.current_candle))
            return result

    def last_n_closed(self, n: int) -> list:
        with self._lock:
            return list(self.closed_candles[-n:]) if len(self.closed_candles) >= n else []

    def last_closed(self):
        with self._lock:
            return dict(self.closed_candles[-1]) if self.closed_candles else None

    def closed_after(self, entry_time: datetime) -> list:
        """Return closed candles whose bar start >= entry_time (for Spike momentum check)."""
        with self._lock:
            return [c for c in self.closed_candles if c["ts"] >= entry_time]


class SecondCandleBuilder:
    """
    N-second candle builder (for Spike strategy's 8-second candles).
    Uses the same interface as CandleBuilder.
    """

    def __init__(self, seconds: int = 8):
        self.seconds        = seconds
        self.current_candle = None
        self.closed_candles = []
        self._lock          = threading.Lock()

    def _bar_start(self, ts: datetime) -> datetime:
        s = (ts.second // self.seconds) * self.seconds
        return ts.replace(second=s, microsecond=0)

    def feed_tick(self, price: float, ts: datetime):
        bar = self._bar_start(ts)
        with self._lock:
            if self.current_candle is None:
                self.current_candle = {
                    "ts": bar, "open": price, "high": price, "low": price, "close": price
                }
                return None

            if bar == self.current_candle["ts"]:
                c = self.current_candle
                c["high"]  = max(c["high"], price)
                c["low"]   = min(c["low"],  price)
                c["close"] = price
                return None
            else:
                closed = dict(self.current_candle)
                self.closed_candles.append(closed)
                self.current_candle = {
                    "ts": bar, "open": price, "high": price, "low": price, "close": price
                }
                return closed

    def last_n_closed(self, n: int) -> list:
        with self._lock:
            return list(self.closed_candles[-n:]) if len(self.closed_candles) >= n else []

    def closed_after(self, entry_time: datetime) -> list:
        with self._lock:
            return [c for c in self.closed_candles if c["ts"] >= entry_time]
