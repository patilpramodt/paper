"""
core/market_hub.py

MarketHub — the single connection layer.

Owns:
   ONE KiteConnect session
   ONE KiteTicker WebSocket
   ONE CandleBuilder for the index (5-min)
   ONE SessionVWAP
   ONE dict of last prices for all subscribed tokens

Responsibilities:
   Receives raw ticks from WebSocket
   Builds 5-min candles from index ticks
   Updates SessionVWAP
   Broadcasts tick + candle events to ALL registered strategies
   Manages token subscriptions (adds/removes on behalf of strategies)
   Reconnects WebSocket on disconnect

Strategies NEVER touch Kite or WebSocket directly.
They call market_hub.subscribe(token) and receive events via callbacks.

IST FIX (applied throughout):
   All datetime.now() calls replaced with _now_ist() which returns
   current wall-clock time in IST (UTC+5:30).
   GitHub Actions runners are UTC — bare datetime.now() returned UTC,
   causing the EOD check (MARKET_CLOSE=15:31) to never trigger until
   10:01 UTC, which is after GitHub's timeout kills the job at ~15:00 IST.
   With _now_ist(), the bot self-exits cleanly at 15:31 IST every day.

FIX (Bug 3 + Bug 7):
   BankNifty INDEX token (260105) has last_traded_quantity=0 on every tick
   because it is a computed index — there are no actual trades.
   Original code: session_vwap.update(price, price, price, qty=0)
                  index_candles.feed_tick(price, qty=0, now)
   → VWAP accumulated nothing → _value stayed None all session.
   → All 5-min candles had volume=0 → BB_STOCH/ORB/ScalperV7 volume
     filters permanently bypassed after backfill bars rolled off.

   Fix: pass proxy_weight=1 to SessionVWAP when qty==0 (index token).
        pass volume=1 to CandleBuilder when qty==0 so candles have
        non-zero volume and the rolling volume average is meaningful.
"""

import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, time as dtime, timezone, timedelta
from typing import TYPE_CHECKING

from core.candle import CandleBuilder
from core.vwap import SessionVWAP

if TYPE_CHECKING:
    from core.base_strategy import BaseStrategy

log = logging.getLogger("core.hub")

# ── IST timezone ──────────────────────────────────────────────────────────────
_IST = timezone(timedelta(hours=5, minutes=30))

def _now_ist() -> datetime:
    """Always returns current datetime in IST — works on GitHub Actions (UTC) and local."""
    return datetime.now(tz=_IST).replace(tzinfo=None)

# ── Market hours (all times are IST) ─────────────────────────────────────────
MARKET_OPEN  = dtime(9, 14)
MARKET_CLOSE = dtime(15, 31)


class MarketHub:
    """
    Central nervous system.
    Created once in t.py, passed to every strategy.
    """

    def __init__(self, token_file: str = "token.json"):
        self._token_file   = token_file
        self.kite          = None
        self.api_key       = None
        self.access_token  = None
        self._ws           = None

        # Shared market state
        self._index_token  = 260105      # BankNifty — fixed Zerodha token
        self._last_price   : dict[int, float] = {}
        self._subscribed   : set[int]   = set()
        self._lock         = threading.Lock()

        # Shared infrastructure
        self.index_candles = CandleBuilder(minutes=5)
        self.session_vwap  = SessionVWAP()

        # Strategy registry
        self._strategies   : list["BaseStrategy"] = []

        # Control
        self._done         = threading.Event()
        self._vwap_started = False

        log.info("MarketHub created")

    # ── Kite login ────────────────────────────────────────────────────────────

    def load_kite(self):
        tf = self._token_file
        if not os.path.isfile(tf):
            log.error(f" '{tf}' not found. Format: {{\"api_key\":\"...\",\"access_token\":\"...\"}}")
            sys.exit(1)
        with open(tf) as f:
            data = json.load(f)
        from kiteconnect import KiteConnect
        self.kite         = KiteConnect(api_key=data["api_key"])
        self.kite.set_access_token(data["access_token"])
        self.api_key      = data["api_key"]
        self.access_token = data["access_token"]
        log.info(" Kite session loaded (shared by all strategies)")
        return self.kite

    # ── Strategy registration ─────────────────────────────────────────────────

    def register(self, strategy: "BaseStrategy"):
        """Register a strategy to receive tick and candle events."""
        self._strategies.append(strategy)
        log.info(f" Strategy registered: {strategy.name}")

    # ── Token subscription ────────────────────────────────────────────────────

    def subscribe(self, token: int):
        """
        Subscribe token to WebSocket (called by strategies for their options).
        Thread-safe. Deduplicates — no double subscriptions.
        """
        with self._lock:
            if token in self._subscribed:
                return
            self._subscribed.add(token)
        if self._ws:
            try:
                self._ws.subscribe([token])
                self._ws.set_mode(self._ws.MODE_FULL, [token])
                log.info(f"  Subscribed token {token}")
            except Exception as e:
                log.warning(f"  Subscribe error for {token}: {e}")

    def unsubscribe(self, token: int):
        """Unsubscribe token when a strategy no longer needs it."""
        with self._lock:
            self._subscribed.discard(token)
        if self._ws:
            try:
                self._ws.unsubscribe([token])
            except Exception as e:
                log.warning(f"  Unsubscribe error for {token}: {e}")

    def last_price(self, token: int) -> float | None:
        """Get last known price for any subscribed token."""
        return self._last_price.get(token)

    # ── WebSocket callbacks ───────────────────────────────────────────────────

    def _on_ticks(self, ws, ticks):
        # FIX: was datetime.now() — returned UTC on GitHub Actions
        now = _now_ist()
        t   = now.time()

        # Ignore outside market hours
        if t < MARKET_OPEN or t > MARKET_CLOSE:
            return

        # Start VWAP tracking at 9:15
        if not self._vwap_started and t >= dtime(9, 15):
            self.session_vwap.reset()
            self._vwap_started = True

        for tick in ticks:
            token   = tick.get("instrument_token")
            price   = tick.get("last_price", 0)
            qty     = tick.get("last_traded_quantity", 0)
            # Exchange timestamp — use this for candle time alignment
            # Falls back to IST wall-clock if not present
            raw_ts  = tick.get("timestamp")
            tick_ts = raw_ts.replace(tzinfo=None) if hasattr(raw_ts, "tzinfo") and raw_ts else now
            if not price:
                continue

            # Store last price (used by strategies via last_price())
            with self._lock:
                self._last_price[token] = price

            if token == self._index_token:
                self._handle_index_tick(price, qty, now, tick_ts)
            else:
                # Option tick — broadcast to all strategies
                for strat in self._strategies:
                    try:
                        strat.on_option_tick(token, price, now, tick_ts)
                    except Exception as e:
                        log.error(f"[{strat.name}] on_option_tick error: {e}")

    def _handle_index_tick(self, price: float, qty: int, now: datetime, tick_ts: datetime):
        """Process index tick: update VWAP, build candles, broadcast.

        FIX (Bug 3 + Bug 7):
        BankNifty INDEX token always has qty=0 (it's a computed index, not a
        traded instrument). We pass qty to vwap.update() with proxy_weight=1
        so VWAP accumulates tick-weighted prices rather than staying None.
        Same fix for candles: use max(qty, 1) so candle volume is never zero.
        Volume filters in BB_STOCH, ORB, and ScalperV7 all depend on non-zero
        candle volumes — without this fix they permanently bypass volume checks.
        """
        # FIX: use tick count (1) as proxy when qty==0 (index token)
        # This ensures VWAP is computed and candle volumes are non-zero.
        proxy_vol = qty if qty > 0 else 1

        # Update session VWAP (tick-weighted with proxy_weight)
        self.session_vwap.update(price, price, price, volume=qty, proxy_weight=1)

        # Broadcast raw tick to all strategies (both wall-clock ts and exchange tick_ts)
        for strat in self._strategies:
            try:
                strat.on_tick(price, now, tick_ts)
            except Exception as e:
                log.error(f"[{strat.name}] on_tick error: {e}")

        # Build 5-min candle using IST wall-clock time and proxy volume
        closed = self.index_candles.feed_tick(price, proxy_vol, now)
        if closed is not None:
            # Broadcast closed candle to all strategies
            for strat in self._strategies:
                try:
                    strat.on_candle(closed, now)
                except Exception as e:
                    log.error(f"[{strat.name}] on_candle error: {e}")

    def _on_connect(self, ws, response):
        log.info(" WebSocket connected")
        # Subscribe all pending tokens
        with self._lock:
            tokens = list(self._subscribed)
        if tokens:
            ws.subscribe(tokens)
            ws.set_mode(ws.MODE_FULL, tokens)
        log.info(f" Subscribed {len(tokens)} tokens: {tokens}")

    def _on_close(self, ws, code, reason):
        log.warning(f"WebSocket closed [{code}]: {reason}")

    def _on_error(self, ws, code, reason):
        log.error(f"WebSocket error [{code}]: {reason}")

    def _on_reconnect(self, ws, attempts):
        log.info(f"  WebSocket reconnecting (attempt {attempts})...")

    def _on_noreconnect(self, ws):
        log.error(" WebSocket could not reconnect. Stopping.")
        self._done.set()

    # ── Historical backfill ───────────────────────────────────────────────────

    def backfill(self, kite, index_token: int = 260105):
        """
        Fetch today's completed 5-min candles from 9:15 AM to now and replay
        them into all strategies BEFORE the WebSocket starts.

        Why this matters:
          Without backfill, strategies start with empty indicator buffers.
          BB_STOCH needs 20 candles (100 min) before it can trade.
          ScalperV7's ATR/MACD/RSI need 5+ candles to stabilise.
          If the bot starts at 1:30 PM those strategies would never trade.

        With backfill:
          All indicator buffers are pre-filled with real market data from 9:15.
          Strategies are fully warmed up from the first live tick onward.

        Special handling per strategy:
          SPIKE      -- on_candle is a no-op in SPIKE, harmless to call.
                        on_tick is NOT called to avoid touching gap detection.
          ORB        -- on_tick IS called for 9:15-9:30 candles only, using
                        candle H and L, so the opening range is reconstructed.
          ScalperV7  -- on_candle fills _buf_5m (5-min indicators). The 1-min
                        candle buffer warms up from the first live tick (fast).
          BB_STOCH   -- on_candle fills _buf_5m (5-min BB/Stoch buffer).
        """
        from datetime import date, timedelta, time as dtime

        # FIX: was datetime.now() — returned UTC on GitHub Actions
        now = _now_ist()
        market_open = dtime(9, 15)

        if now.time() <= market_open:
            log.info(" Backfill: market not yet open -- nothing to replay")
            return

        today   = now.date()
        from_dt = datetime.combine(today, market_open)
        # Only fetch completed bars. The current bar is still forming,
        # so exclude it by going back 5 minutes from now.
        to_dt   = now - timedelta(minutes=5)

        if to_dt < from_dt:
            log.info(" Backfill: less than one 5-min bar completed -- nothing to replay")
            return

        log.info("=" * 60)
        log.info("  HISTORICAL BACKFILL — warming up all strategies")
        log.info("=" * 60)
        log.info(f"  Fetching 5-min candles: 09:15 → {to_dt.strftime('%H:%M')}")

        try:
            raw = kite.historical_data(
                instrument_token=index_token,
                from_date=from_dt,
                to_date=to_dt,
                interval="5minute"
            )
        except Exception as e:
            log.error(f"  Backfill fetch FAILED: {e}")
            log.warning("  Strategies will warm up from live ticks (slower startup)")
            return

        if not raw:
            log.info("  Backfill: Kite returned 0 candles -- nothing to replay")
            return

        log.info(f"  {len(raw)} candles received — replaying now...")

        # OR window: 9:15 to 9:30 -- candles in this range feed ORB's opening range
        or_start = dtime(9, 15)
        or_end   = dtime(9, 30)

        # Pre-seed the hub's shared CandleBuilder so any code reading
        # index_candles.get_closed() sees the full history from 9:15
        for bar in raw:
            raw_ts = bar["date"]
            candle_ts = raw_ts.replace(tzinfo=None) if hasattr(raw_ts, "tzinfo") else raw_ts
            candle = {
                "ts"    : candle_ts,
                "open"  : float(bar["open"]),
                "high"  : float(bar["high"]),
                "low"   : float(bar["low"]),
                "close" : float(bar["close"]),
                "volume": int(bar["volume"]),
            }
            with self._lock:
                self.index_candles.closed_candles.append(candle)

        # Replay each candle into strategies
        n_replayed = 0
        candle = None  # track last candle for OR-lock tick below
        for bar in raw:
            raw_ts    = bar["date"]
            candle_ts = raw_ts.replace(tzinfo=None) if hasattr(raw_ts, "tzinfo") else raw_ts
            candle = {
                "ts"    : candle_ts,
                "open"  : float(bar["open"]),
                "high"  : float(bar["high"]),
                "low"   : float(bar["low"]),
                "close" : float(bar["close"]),
                "volume": int(bar["volume"]),
            }
            bar_time = candle_ts.time() if hasattr(candle_ts, "time") else candle_ts

            # For the opening range window: feed candle H and L into ORB's on_tick
            # so it reconstructs its opening range exactly as if it had seen every tick.
            # Only ORB gets these synthetic ticks -- other strategies don't use on_tick
            # for indicator warmup, and SPIKE's gap detection must not be triggered.
            if or_start <= bar_time < or_end:
                for strat in self._strategies:
                    if strat.name != "ORB_v2":
                        continue
                    try:
                        strat.on_tick(candle["high"], candle_ts, candle_ts)
                        strat.on_tick(candle["low"],  candle_ts, candle_ts)
                    except Exception as e:
                        log.error(f"[{strat.name}] backfill on_tick error: {e}")

            # Replay candle into ALL strategies.
            # SPIKE.on_candle is a no-op so it's harmless.
            for strat in self._strategies:
                try:
                    strat.on_candle(candle, candle_ts)
                except Exception as e:
                    log.error(f"[{strat.name}] backfill on_candle error: {e}")

            n_replayed += 1

        # Check if OR window is completely in the past -- if so, tell ORB to lock now
        # (normally locked by on_tick when t >= 9:30, but backfill uses historical ts)
        if now.time() >= or_end and candle is not None:
            for strat in self._strategies:
                if strat.name == "ORB_v2":
                    try:
                        # Trigger OR lock by sending a synthetic tick at 9:30 exactly
                        lock_ts = datetime.combine(today, or_end)
                        strat.on_tick(candle["close"], lock_ts, lock_ts)
                    except Exception as e:
                        log.error(f"[ORB_v2] backfill OR-lock tick error: {e}")

        log.info(f"  Backfill complete: {n_replayed} candles replayed into all strategies")
        log.info(f"  Strategies are now warmed up from 09:15 data")
        log.info("=" * 60 + "\n")

    # ── Main run loop ─────────────────────────────────────────────────────────

    def run(self):
        """
        Start WebSocket and run until 3:31 PM IST.
        Called from t.py after all strategies are registered.
        """
        from kiteconnect import KiteTicker

        # Pre-subscribe index token
        self._subscribed.add(self._index_token)

        ticker = KiteTicker(self.api_key, self.access_token)
        ticker.on_ticks       = self._on_ticks
        ticker.on_connect     = self._on_connect
        ticker.on_close       = self._on_close
        ticker.on_error       = self._on_error
        ticker.on_reconnect   = self._on_reconnect
        ticker.on_noreconnect = self._on_noreconnect
        self._ws = ticker

        log.info(" Starting WebSocket (shared for all strategies)...")
        ticker.connect(threaded=True)

        log.info(" Market running... (9:15 AM – 3:30 PM)")
        try:
            while not self._done.is_set():
                # FIX: was datetime.now() — returned UTC on GitHub Actions.
                # The EOD check never fired on UTC time, causing the job to
                # run until GitHub's timeout killed it (~15:00 IST).
                # With _now_ist(), the bot self-exits cleanly at 15:31 IST.
                now = _now_ist()
                if now.time() > MARKET_CLOSE:
                    log.info(" Market closed (3:30 PM) — running EOD summaries")
                    for strat in self._strategies:
                        try:
                            strat.eod_summary()
                        except Exception as e:
                            log.error(f"[{strat.name}] eod_summary error: {e}")
                    break
                time.sleep(1)
        except KeyboardInterrupt:
            log.info("\n Interrupted by user")
            for strat in self._strategies:
                try:
                    strat.eod_summary()
                except Exception:
                    pass

        ticker.close()
        log.info("MarketHub shutdown complete")
