"""
core/candle.py

Shared candle builders used by ALL strategies.
Both N-minute (for ORB / BB_STOCH / ScalperV7) and N-second (for SPIKE and
the candle-breakout strategies).
Each strategy gets its own instance per token — no sharing of state.

FIXES IN THIS VERSION
─────────────────────
FIX 1 — Unbounded memory growth:
    `closed_candles` was a plain list that grew for the whole session and
    was never trimmed. With ~15 SecondCandleBuilder instances at 5s/10s
    resolution that is ~4,500-9,000 dicts per builder per day, plus a full
    list copy on every `get_closed()` / `get_all()` call (which some
    strategies invoke on every tick). Now backed by a bounded deque —
    `max_history` defaults to enough bars for any indicator in the repo.

FIX 2 — Sub-minute buckets that don't divide 60:
    `_bar_start()` computed `(ts.second // seconds) * seconds`, which
    restarts the bucketing at every minute boundary. For seconds=8 that
    produced buckets [0,8,16,24,32,40,48,56] and then a runt 4-second
    bucket at :56-:00 — every minute. SPIKE uses bucket_sec=10 today so
    this is currently latent, but any future 8s/7s/9s bucket would emit
    short candles that silently understate body size and break the
    body-threshold logic in the candle-breakout strategies.
    Now: seconds that do not divide 60 are rejected at construction with a
    clear error, so the bug cannot be reintroduced silently.

FIX 3 — Out-of-order ticks opened a bar in the past:
    On a WebSocket reconnect Zerodha replays a snapshot whose exchange
    timestamp can be older than the currently-open bar. The old code
    treated `bar != current["ts"]` as "bar closed", emitted a bogus closed
    candle, and reset the live bar backwards in time. Now an older bucket
    is folded into the live bar instead.

FIX 4 — `get_current()` on both builders:
    Previously only SecondCandleBuilder exposed the in-flight bar.
"""

import threading
from collections import deque
from datetime import datetime

# Default cap on retained closed candles. The deepest lookback anywhere in
# the repo is CFG["buf_size"]=100 (nifty_directional) plus indicator warm-up,
# so 500 is generous headroom while still bounding memory.
DEFAULT_MAX_HISTORY = 500


class CandleBuilder:
    """
    Converts raw ticks into OHLCV candles of any duration in minutes.

    Usage:
        cb = CandleBuilder(minutes=5)
        closed = cb.feed_tick(price, volume, timestamp)
        # Returns closed candle dict when a bar closes, else None

    The candle dict has keys:
        ts, open, high, low, close, volume
    """

    def __init__(self, minutes: int = 5, max_history: int = DEFAULT_MAX_HISTORY):
        if minutes <= 0 or 60 % minutes != 0:
            raise ValueError(
                f"CandleBuilder(minutes={minutes}) — minutes must divide 60 "
                f"evenly, otherwise bars restart at every hour boundary and "
                f"produce runt candles."
            )
        self.minutes        = minutes
        self.current_candle = None
        # FIX 1: bounded history instead of an ever-growing list.
        self.closed_candles = deque(maxlen=max_history)
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

            # FIX 3: out-of-order tick (reconnect snapshot) — fold into the
            # live bar instead of emitting a bogus close and rewinding time.
            if bar < self.current_candle["ts"]:
                c = self.current_candle
                c["high"]    = max(c["high"], price)
                c["low"]     = min(c["low"],  price)
                c["volume"] += volume
                return None

            closed = dict(self.current_candle)
            self.closed_candles.append(closed)
            self.current_candle = {
                "ts": bar, "open": price, "high": price,
                "low": price, "close": price, "volume": volume
            }
            return closed

    def get_current(self):
        """Return a copy of the still-forming (not yet closed) candle, or None."""
        with self._lock:
            return dict(self.current_candle) if self.current_candle else None

    def get_closed(self) -> list:
        """Return copy of all retained closed candles."""
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
            return list(self.closed_candles)[-n:] if len(self.closed_candles) >= n else []

    def last_closed(self):
        with self._lock:
            return dict(self.closed_candles[-1]) if self.closed_candles else None

    def closed_after(self, entry_time: datetime) -> list:
        """Return closed candles whose bar start >= entry_time."""
        with self._lock:
            return [c for c in self.closed_candles if c["ts"] >= entry_time]


class SecondCandleBuilder:
    """
    N-second candle builder (SPIKE's 10s bars, candle-breakout 10s/5s bars).
    Same interface as CandleBuilder, minus volume (index ticks carry none).
    """

    def __init__(self, seconds: int = 10, max_history: int = DEFAULT_MAX_HISTORY):
        # FIX 2: reject bucket sizes that don't tile a minute evenly.
        if seconds <= 0 or 60 % seconds != 0:
            raise ValueError(
                f"SecondCandleBuilder(seconds={seconds}) — seconds must divide "
                f"60 evenly (1,2,3,4,5,6,10,12,15,20,30). Otherwise the last "
                f"bucket of every minute is truncated, producing short candles "
                f"that understate body size and corrupt breakout thresholds."
            )
        self.seconds        = seconds
        self.current_candle = None
        self.closed_candles = deque(maxlen=max_history)
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

            # FIX 3: out-of-order tick — fold into the live bar.
            if bar < self.current_candle["ts"]:
                c = self.current_candle
                c["high"] = max(c["high"], price)
                c["low"]  = min(c["low"],  price)
                return None

            closed = dict(self.current_candle)
            self.closed_candles.append(closed)
            self.current_candle = {
                "ts": bar, "open": price, "high": price, "low": price, "close": price
            }
            return closed

    def get_current(self):
        """Return a copy of the still-forming (not yet closed) candle, or None."""
        with self._lock:
            return dict(self.current_candle) if self.current_candle else None

    def last_n_closed(self, n: int) -> list:
        with self._lock:
            return list(self.closed_candles)[-n:] if len(self.closed_candles) >= n else []

    def last_closed(self):
        with self._lock:
            return dict(self.closed_candles[-1]) if self.closed_candles else None

    def closed_after(self, entry_time: datetime) -> list:
        with self._lock:
            return [c for c in self.closed_candles if c["ts"] >= entry_time]