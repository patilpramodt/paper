"""
core/market_hub.py

MarketHub — the single connection layer.

Owns:
   ONE KiteConnect session
   ONE KiteTicker WebSocket
   ONE CandleBuilder for the index (5-min)
   ONE SessionVWAP
   ONE dict of last prices for all subscribed tokens
   ONE dict of last OI for all subscribed tokens  ← NEW (for WsPCR)

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

OI FIX (WsPCR support):
   Added _last_oi dict and last_oi(token) method.
   Zerodha sends oi field in every MODE_FULL tick for options/futures.
   _on_ticks() now stores tick["oi"] for each token so WsPCR can read
   it without any extra API calls or subscriptions.
   3 lines of change: dict init in __init__, store in _on_ticks, getter method.

MULTI-INDEX FIX (Nifty strategies added):
   Bug A — backfill() strategy filter:
     Old filter: `if strat_idx is not None and strat_idx != index_token: continue`
     Problem:    strat_idx=None (BankNifty strategies) always passed the filter,
                 so ALL BankNifty strategies received Nifty 50 backfill candles
                 when hub.backfill(index_token=256265) was called.  Their
                 _buf_5m buffers were corrupted with mixed BN+Nifty price data,
                 and ORB's opening range was overwritten with Nifty ~24,500 levels.
     Fix:        Treat INDEX_TOKEN=None as "tracks main BankNifty index (260105)".
                 A strategy only receives backfill candles for its own index.
                 The canonical index for a strategy is:
                   strat_idx if strat_idx is not None else 260105 (BankNifty)

   Bug B — _handle_index_tick() on_candle broadcast:
     Old code:   Broadcast closed BankNifty candle to ALL strategies with no filter.
     Problem:    BBStochNiftyStrategy (INDEX_TOKEN=256265) received every closed
                 BankNifty candle, appended it to _buf_5m, and ran _evaluate_entry()
                 on BankNifty data.  Live candles via on_tick() also fed Nifty candles
                 into _buf_5m → buffer had two candles per period (one BN, one Nifty).
     Fix:        Skip strategies whose INDEX_TOKEN is not None — they track a
                 different index and build their own candles internally.
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

# ── Main BankNifty index token (fixed Zerodha instrument token) ───────────────
_BANKNIFTY_TOKEN = 260105


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
        self._index_token  = _BANKNIFTY_TOKEN   # BankNifty — fixed Zerodha token
        # Extra index tokens registered by strategies with INDEX_TOKEN != None.
        # Each token here is subscribed to WebSocket and routed exclusively to
        # strategies whose INDEX_TOKEN matches it (e.g. 256265 for Nifty 50).
        self._extra_index_tokens: set[int] = set()
        self._last_price   : dict[int, float]    = {}
        self._last_price_ts: dict[int, datetime] = {}   # FIX: track when price arrived
        self._subscribed   : set[int]   = set()
        self._lock         = threading.Lock()

        # OI cache — populated from MODE_FULL ticks (for WsPCR)
        # Zerodha sends oi field for every option/futures tick in MODE_FULL.
        # WsPCR reads this to compute PCR without any external HTTP calls.
        self._last_oi      : dict[int, int]      = {}

        # Shared infrastructure
        self.index_candles = CandleBuilder(minutes=5)
        self.session_vwap  = SessionVWAP()

        # Strategy registry
        self._strategies   : list["BaseStrategy"] = []

        # Control
        self._done         = threading.Event()
        self._vwap_started = False

        # Set by t.py after hub creation: hub.order_router = OrderRouter(hub)
        self.order_router  = None

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

    def add_index_token(self, token: int):
        """
        Register an additional index token (e.g. 256265 for Nifty 50).

        Ticks for this token are routed exclusively to strategies whose class
        attribute INDEX_TOKEN equals this token — not to all strategies.
        The token is added to _subscribed so it is included when the WebSocket
        connects. Call this before hub.run() so _on_connect picks it up.

        Strategies signal their index by declaring:
            INDEX_TOKEN = 256265  # class-level attribute
        Strategies without INDEX_TOKEN (or INDEX_TOKEN = None) receive the
        main BankNifty ticks as before.
        """
        self._extra_index_tokens.add(token)
        with self._lock:
            self._subscribed.add(token)
        if self._ws:
            try:
                self._ws.subscribe([token])
                self._ws.set_mode(self._ws.MODE_FULL, [token])
            except Exception as e:
                log.warning(f"  add_index_token subscribe error for {token}: {e}")
        log.info(f" Extra index token registered: {token}")

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

    def last_price_ts(self, token: int) -> "datetime | None":
        """Get timestamp of when the last price tick arrived (IST)."""
        return self._last_price_ts.get(token)

    def last_oi(self, token: int) -> int:
        """
        Get last known Open Interest for any subscribed option/futures token.

        Zerodha sends oi in every MODE_FULL tick. This is populated by
        _on_ticks() for all non-index tokens. Returns 0 if no tick received yet.

        Used by WsPCR to compute PCR from WebSocket OI data without HTTP calls.
        """
        return self._last_oi.get(token, 0)

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

            # Store last price + timestamp (used by strategies via last_price())
            with self._lock:
                self._last_price[token]    = price
                self._last_price_ts[token] = now   # FIX: record when this tick arrived

                # OI FIX: store open interest from MODE_FULL tick.
                # Zerodha sends oi for options/futures in every full-mode tick.
                # Index tokens (main + extra) have no OI — skip them.
                # WsPCR reads this to compute PCR without any HTTP calls.
                if token != self._index_token and token not in self._extra_index_tokens:
                    oi = tick.get("oi", 0)
                    if oi:
                        self._last_oi[token] = oi

            if token == self._index_token:
                self._handle_index_tick(price, qty, now, tick_ts)
            elif token in self._extra_index_tokens:
                # Extra index (e.g. Nifty 50) — route only to matching strategies
                self._handle_extra_index_tick(token, price, now, tick_ts)
            else:
                # Option tick — broadcast to all strategies
                for strat in self._strategies:
                    try:
                        strat.on_option_tick(token, price, now, tick_ts)
                    except Exception as e:
                        log.error(f"[{strat.name}] on_option_tick error: {e}")

    def _handle_index_tick(self, price: float, qty: int, now: datetime, tick_ts: datetime):
        """Process BankNifty index tick: update VWAP, build candles, broadcast.

        FIX (Bug 3 + Bug 7):
        BankNifty INDEX token always has qty=0 (it's a computed index, not a
        traded instrument). We pass qty to vwap.update() with proxy_weight=1
        so VWAP accumulates tick-weighted prices rather than staying None.
        Same fix for candles: use max(qty, 1) so candle volume is never zero.
        Volume filters in BB_STOCH, ORB, and ScalperV7 all depend on non-zero
        candle volumes — without this fix they permanently bypass volume checks.

        MULTI-INDEX FIX (Bug B):
        on_candle broadcast now skips strategies with INDEX_TOKEN set.
        Those strategies (e.g. BBStochNiftyStrategy) track a different index
        and build their own candles internally from on_tick(). Broadcasting
        BankNifty candles to them would corrupt their _buf_5m indicator buffers
        with BankNifty price data interleaved with their own Nifty candles.
        """
        # FIX: use tick count (1) as proxy when qty==0 (index token)
        # This ensures VWAP is computed and candle volumes are non-zero.
        proxy_vol = qty if qty > 0 else 1

        # Update session VWAP (tick-weighted with proxy_weight)
        self.session_vwap.update(price, price, price, volume=qty, proxy_weight=1)

        # Broadcast raw tick to all strategies that use the main BankNifty index.
        # Strategies with a class-level INDEX_TOKEN attribute (e.g. SpikeNiftyStrategy)
        # receive ticks from _handle_extra_index_tick() instead — not here.
        for strat in self._strategies:
            strat_index = getattr(strat, "INDEX_TOKEN", None)
            if strat_index is not None:
                continue   # this strategy tracks a different index
            try:
                strat.on_tick(price, now, tick_ts)
            except Exception as e:
                log.error(f"[{strat.name}] on_tick error: {e}")

        # Build 5-min BankNifty candle using IST wall-clock time and proxy volume
        closed = self.index_candles.feed_tick(price, proxy_vol, now)
        if closed is not None:
            # MULTI-INDEX FIX (Bug B): only broadcast to strategies that track the
            # main BankNifty index (INDEX_TOKEN = None).  Extra-index strategies
            # (e.g. BBStochNiftyStrategy with INDEX_TOKEN=256265) build their own
            # 5-min candles internally inside on_tick() — sending them BankNifty
            # candles here would inject wrong price data into their _buf_5m buffers.
            for strat in self._strategies:
                if getattr(strat, "INDEX_TOKEN", None) is not None:
                    continue   # tracks a different index — skip BankNifty candles
                try:
                    strat.on_candle(closed, now)
                except Exception as e:
                    log.error(f"[{strat.name}] on_candle error: {e}")

    def _handle_extra_index_tick(self, token: int, price: float, now: datetime, tick_ts: datetime):
        """
        Route an extra index tick (e.g. Nifty 50, token=256265) exclusively to
        strategies whose INDEX_TOKEN class attribute matches this token.

        No candle building is done here — extra-index strategies build their own
        candles internally (e.g. BBStochNiftyStrategy uses an internal CandleBuilder
        in on_tick(); SpikeNiftyStrategy builds 8-second candles).
        """
        for strat in self._strategies:
            strat_index = getattr(strat, "INDEX_TOKEN", None)
            if strat_index != token:
                continue
            try:
                strat.on_tick(price, now, tick_ts)
            except Exception as e:
                log.error(f"[{strat.name}] on_tick error (extra index {token}): {e}")

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
          SPIKE/SPIKE_NIFTY  -- on_candle is a no-op, harmless to call.
                                on_tick is NOT called to avoid touching gap
                                detection.
          ORB                -- on_tick IS called for 9:15-9:30 BankNifty candles
                                only, so the opening range is reconstructed.
          ScalperV7          -- on_candle fills _buf_5m (5-min indicators).
          BB_STOCH           -- on_candle fills _buf_5m (5-min BB/Stoch buffer).
          BB_STOCH_NIFTY     -- on_candle fills _buf_5m with Nifty candles
                                (only when index_token=256265 is passed).

        MULTI-INDEX FIX (Bug A):
          Strategy filter now uses the strategy's canonical index:
            - strat_idx = None   → strategy tracks BankNifty (260105)
            - strat_idx = 256265 → strategy tracks Nifty 50
          A strategy only receives backfill candles for its own index.

          Old filter: `if strat_idx is not None and strat_idx != index_token`
            Problem: INDEX_TOKEN=None strategies always passed the filter,
            so ALL BankNifty strategies received Nifty backfill candles,
            corrupting their _buf_5m with Nifty price levels.

          New filter: canonical_idx = strat_idx or _BANKNIFTY_TOKEN
                      skip if canonical_idx != index_token

          ORB on_tick feeding and the shared index_candles pre-seed are
          both guarded to run only on the BankNifty (260105) backfill pass.
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

        index_label = "BankNifty" if index_token == _BANKNIFTY_TOKEN else f"Nifty({index_token})"
        log.info("=" * 60)
        log.info(f"  HISTORICAL BACKFILL [{index_label}] — warming up strategies")
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

        # OR window: 9:15 to 9:30 — BankNifty-only; used to reconstruct ORB opening range
        or_start = dtime(9, 15)
        or_end   = dtime(9, 30)

        # MULTI-INDEX FIX (Bug A): Only pre-seed the shared BankNifty CandleBuilder
        # when this is the BankNifty backfill pass.  The shared index_candles store
        # is keyed to BankNifty prices — appending Nifty candles here would pollute
        # any code that reads index_candles.get_closed() expecting BankNifty levels.
        if index_token == _BANKNIFTY_TOKEN:
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

            # MULTI-INDEX FIX (Bug A): Only feed ORB's opening range from BankNifty
            # candles.  If this guard were absent, the Nifty backfill pass would
            # overwrite ORB's high/low with Nifty ~24,500 levels, making ORB
            # trade on a completely wrong opening range.
            if index_token == _BANKNIFTY_TOKEN and or_start <= bar_time < or_end:
                for strat in self._strategies:
                    if strat.name != "ORB_v2":
                        continue
                    try:
                        strat.on_tick(candle["high"], candle_ts, candle_ts)
                        strat.on_tick(candle["low"],  candle_ts, candle_ts)
                    except Exception as e:
                        log.error(f"[{strat.name}] backfill on_tick error: {e}")

            # MULTI-INDEX FIX (Bug A): Strategy routing — each strategy only
            # receives candles from its own index.
            #
            # Old filter: `if strat_idx is not None and strat_idx != index_token`
            #   Bug: INDEX_TOKEN=None (BankNifty strategies) always passed the
            #   filter and received Nifty backfill candles too.
            #
            # New filter: derive each strategy's canonical index token.
            #   INDEX_TOKEN=None  → tracks BankNifty (260105)
            #   INDEX_TOKEN=X     → tracks index X (e.g. Nifty 50 = 256265)
            # Only send the candle if the strategy's canonical index matches
            # the index_token this backfill run is fetching.
            for strat in self._strategies:
                strat_idx    = getattr(strat, "INDEX_TOKEN", None)
                canonical    = strat_idx if strat_idx is not None else _BANKNIFTY_TOKEN
                if canonical != index_token:
                    continue   # wrong index — skip
                try:
                    strat.on_candle(candle, candle_ts)
                except Exception as e:
                    log.error(f"[{strat.name}] backfill on_candle error: {e}")

            n_replayed += 1

        # MULTI-INDEX FIX (Bug A): OR-lock tick is BankNifty-specific.
        # Only run this for the BankNifty backfill pass.
        if index_token == _BANKNIFTY_TOKEN and now.time() >= or_end and candle is not None:
            for strat in self._strategies:
                if strat.name == "ORB_v2":
                    try:
                        lock_ts = datetime.combine(today, or_end)
                        strat.on_tick(candle["close"], lock_ts, lock_ts)
                    except Exception as e:
                        log.error(f"[ORB_v2] backfill OR-lock tick error: {e}")

        log.info(f"  Backfill complete: {n_replayed} candles replayed [{index_label}]")
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
