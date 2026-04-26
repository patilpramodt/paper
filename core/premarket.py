"""
core/premarket.py

PCR SOURCE CHANGE (this version):
  NSE HTTP _fetch_pcr() has been REMOVED entirely.
  PCR is now sourced ONLY from Zerodha WebSocket OI data via WsPCR
  (core/pcr_kite.py). This eliminates:
    • NSE bot-detection / soft-block risk
    • Intermittent empty-OI responses at market open
    • Session management overhead
    • Outbound HTTP dependency on nseindia.com (works on GitHub Actions)

  If ws_pcr is not passed to start_live_refresh(), PCR updates are
  completely skipped for the session. All strategies handle PCR=None
  gracefully (PCR filter is bypassed when value is None).

  VIX is still fetched from NSE (nseindia.com/api/allIndices) because
  Zerodha does not provide India VIX via WebSocket or REST API.

Other fixes retained from previous version:
  1. prev_close fetch: retry up to 3 times with 15s timeout
  2. PCR fetch: SKIPPED in fetch_all() — OI data not ready at 9:08 AM.
               First PCR fetch now delayed to 9:16 AM in live refresh thread.
  3. Live refresh: PCR skipped until 9:16 AM on first run.
  4. WsPCR integration: start_live_refresh() accepts ws_pcr= kwarg.
     When ws_pcr is supplied, PCR is read from WebSocket OI data via
     ws_pcr.compute_pcr() — no NSE HTTP at all.
     When ws_pcr=None, PCR updates are skipped for the entire session.
"""

import logging
import threading
from datetime import datetime, timedelta, date, time as dtime, timezone

# IST FIX: GitHub Actions runners are UTC — use IST-aware now() everywhere
_IST = timezone(timedelta(hours=5, minutes=30))

def _now_ist() -> datetime:
    """Always returns current datetime in IST — works on GitHub Actions (UTC) and local."""
    return datetime.now(tz=_IST).replace(tzinfo=None)

import numpy as np
import pandas as pd
import requests

log = logging.getLogger("core.premarket")

INDEX_TOKEN = 260105   # BankNifty fixed Zerodha token


class PreMarketData:

    def __init__(self):
        self.vix             : float | None = None
        self.pcr             : float | None = None
        self.prev_close      : float | None = None
        self.prev_body_high  : float | None = None
        self.prev_body_low   : float | None = None
        self.prev_last5m_high  : float | None = None
        self.prev_last5m_low   : float | None = None
        self.prev_last5m_close : float | None = None
        self.ema200_daily    : float | None = None
        self.expiry_date     : date  | None = None
        self.dte_days        : int   | None = None
        self.fetch_ok        : bool         = False
        self._lock           = threading.Lock()

    # ── Live refresh ──────────────────────────────────────────────────────────

    def start_live_refresh(
        self,
        stop_event: threading.Event,
        vix_interval: int = 300,
        pcr_interval: int = 600,
        ws_pcr=None,
    ):
        """
        Start background thread that refreshes VIX and PCR during market hours.

        Parameters
        ──────────
        stop_event    : threading.Event — set by MarketHub at EOD to stop the thread
        vix_interval  : seconds between VIX refreshes (default 5 min)
        pcr_interval  : seconds between PCR refreshes (default 10 min)
        ws_pcr        : WsPCR instance (required for PCR updates).
                        PCR is read from Zerodha WebSocket OI data via
                        ws_pcr.compute_pcr() — fast, reliable, no NSE HTTP.

                        When ws_pcr=None:
                          PCR updates are skipped entirely for this session.
                          All strategies handle PCR=None (bypass PCR filter).
                          This is logged clearly at startup.

        WsPCR path (ws_pcr is not None):
          - No NSE session management
          - No bot-detection risk
          - PCR is always current (reflects every MODE_FULL OI tick)
          - Works on GitHub Actions (no outbound NSE HTTP needed)
          - compute_pcr() returns None if < min_active tokens have OI > 0,
            which happens during the first ~60s after WebSocket connects.
            In that case the last known PCR value is kept unchanged.

        VIX:
          - Still fetched from NSE HTTP (nseindia.com/api/allIndices).
          - Zerodha does not provide India VIX via WebSocket or REST API.
          - VIX changes slowly (minutes) so occasional fetch failures are fine.
        """
        import time as time_mod

        _ws_pcr = ws_pcr
        _pcr_available = _ws_pcr is not None

        def _refresh_loop():
            MARKET_OPEN  = dtime(9, 15)
            MARKET_CLOSE = dtime(15, 30)
            # Delay first PCR fetch until 9:16 AM — NSE OI data / WS OI ticks
            # need ~30-60s to arrive and populate after WebSocket connects.
            PCR_START    = dtime(9, 16)

            last_vix_refresh = 0.0
            last_pcr_refresh = 0.0

            if _pcr_available:
                log.info(
                    f"[LiveRefresh] Started — VIX every {vix_interval//60}min | "
                    f"PCR every {pcr_interval//60}min via WsPCR (Zerodha WebSocket OI) "
                    f"(first PCR after 9:16 AM)"
                )
            else:
                log.warning(
                    f"[LiveRefresh] Started — VIX every {vix_interval//60}min | "
                    f"PCR DISABLED (ws_pcr=None — no PCR updates this session). "
                    f"All strategies will run without PCR filter."
                )

            while not stop_event.is_set():
                try:
                    now_t = _now_ist().time()

                    if not (MARKET_OPEN <= now_t <= MARKET_CLOSE):
                        stop_event.wait(timeout=30)
                        continue

                    mono = time_mod.monotonic()

                    # ── VIX refresh (NSE HTTP) ────────────────────────────────
                    if mono - last_vix_refresh >= vix_interval:
                        new_vix = self._fetch_vix()
                        if new_vix is not None:
                            with self._lock:
                                old = self.vix
                                self.vix = new_vix
                            if old is None or abs(new_vix - old) >= 0.05:
                                log.info(f"[LiveRefresh] VIX updated: {old} → {new_vix:.2f}")
                            else:
                                log.debug(f"[LiveRefresh] VIX unchanged: {new_vix:.2f}")
                        else:
                            log.warning(f"[LiveRefresh] VIX fetch failed — keeping last: {self.vix}")
                        last_vix_refresh = mono

                    # ── PCR refresh via WsPCR (Zerodha WebSocket OI) ──────────
                    # Only runs after 9:16 AM and only when ws_pcr is wired.
                    if _pcr_available and now_t >= PCR_START and mono - last_pcr_refresh >= pcr_interval:
                        new_pcr = _ws_pcr.compute_pcr()
                        if new_pcr is not None:
                            with self._lock:
                                old = self.pcr
                                self.pcr = new_pcr
                            if old is None or abs(new_pcr - old) >= 0.01:
                                log.info(
                                    f"[LiveRefresh] PCR updated (WS-OI): "
                                    f"{old} → {new_pcr:.3f}"
                                )
                            else:
                                log.debug(
                                    f"[LiveRefresh] PCR unchanged (WS-OI): {new_pcr:.3f}"
                                )
                            _ws_pcr.log_summary()
                        else:
                            log.warning(
                                f"[LiveRefresh] WsPCR returned None — "
                                f"keeping last PCR: {self.pcr} "
                                f"(OI ticks still warming up)"
                            )
                        last_pcr_refresh = mono

                except Exception as e:
                    log.error(
                        f"[LiveRefresh] Unexpected error in refresh loop: {e}",
                        exc_info=True
                    )

                stop_event.wait(timeout=60)

            log.info("[LiveRefresh] Thread stopped.")

        t = threading.Thread(target=_refresh_loop, name="PreMarket-LiveRefresh", daemon=True)
        t.start()
        return t

    # ── fetch_all ─────────────────────────────────────────────────────────────

    def fetch_all(self, kite, index_token: int = INDEX_TOKEN, instruments=None):
        log.info("=" * 56)
        log.info("  PRE-MARKET DATA FETCH (shared for all strategies)")
        log.info("=" * 56)

        # ── VIX ───────────────────────────────────────────────────────────────
        self.vix = self._fetch_vix()
        log.info(f"  India VIX   : {self.vix}")

        # ── PCR: always None at fetch time ────────────────────────────────────
        # PCR comes from WsPCR (Zerodha WebSocket OI) starting at 9:16 AM.
        # NSE option chain HTTP is not used — removed entirely.
        # Running without PCR at open is safe — strategies handle PCR=None.
        self.pcr = None
        log.info("  PCR         : None (will be fetched at 9:16 AM via WsPCR/Zerodha OI)")

        # ── Previous day OHLC — with retry and longer timeout ─────────────────
        for attempt in range(1, 4):
            try:
                import time as _t
                today = _now_ist().date()
                raw = kite.historical_data(
                    instrument_token=index_token,
                    from_date=datetime.combine(today - timedelta(days=7), dtime(9, 15)),
                    to_date=datetime.combine(today - timedelta(days=1), dtime(15, 30)),
                    interval="day",
                )
                if raw:
                    prev = raw[-1]
                    self.prev_close     = float(prev["close"])
                    self.prev_body_high = float(max(prev["open"], prev["close"]))
                    self.prev_body_low  = float(min(prev["open"], prev["close"]))
                    log.info(f"  Prev Close  : {self.prev_close:.2f}")
                    log.info(f"  Prev Body   : [{self.prev_body_low:.2f} – {self.prev_body_high:.2f}]")
                    break
            except Exception as e:
                log.warning(f"  Prev close fetch attempt {attempt}/3 failed: {e}")
                if attempt < 3:
                    _t.sleep(3)

        # ── Previous day last 5-min candle — with retry ───────────────────────
        for attempt in range(1, 4):
            try:
                import time as _t
                today = _now_ist().date()
                raw5m = kite.historical_data(
                    instrument_token=index_token,
                    from_date=datetime.combine(today - timedelta(days=7), dtime(15, 20)),
                    to_date=datetime.combine(today - timedelta(days=1), dtime(15, 30)),
                    interval="5minute",
                )
                if raw5m:
                    last5m = raw5m[-1]
                    self.prev_last5m_high  = float(last5m["high"])
                    self.prev_last5m_low   = float(last5m["low"])
                    self.prev_last5m_close = float(last5m["close"])
                    log.info(
                        f"  Prev Last5m : H={self.prev_last5m_high:.2f} "
                        f"L={self.prev_last5m_low:.2f} "
                        f"C={self.prev_last5m_close:.2f} "
                        f"(ts={last5m.get('date', '?')})"
                    )
                    break
            except Exception as e:
                log.warning(f"  Prev last5m fetch attempt {attempt}/3 failed: {e}")
                if attempt < 3:
                    _t.sleep(3)

        # ── 200 EMA ───────────────────────────────────────────────────────────
        try:
            raw = kite.historical_data(
                instrument_token=index_token,
                from_date=_now_ist() - timedelta(days=300),
                to_date=_now_ist(),
                interval="day",
            )
            if len(raw) >= 50:
                closes = pd.Series([r["close"] for r in raw])
                self.ema200_daily = float(closes.ewm(span=200, adjust=False).mean().iloc[-1])
                current = float(raw[-1]["close"])
                bias = "ABOVE (CE bias)" if current > self.ema200_daily else "BELOW (PE bias)"
                log.info(f"  200EMA      : {self.ema200_daily:.2f}  price {bias}")
        except Exception as e:
            log.warning(f"  200EMA fetch failed: {e}")

        # ── Expiry + DTE ──────────────────────────────────────────────────────
        from core.instruments import get_nearest_expiry
        if instruments is not None:
            try:
                self.expiry_date = get_nearest_expiry(instruments._df, _now_ist().date())
                self._expiry_date_hint = self.expiry_date.strftime("%d-%m-%Y")
                log.info(f"  Expiry      : {self.expiry_date}  (from live NFO instruments)")
            except Exception as e:
                log.error(f"  Expiry resolution FAILED: {e}")
                self.fetch_ok = False
                return False
        else:
            log.error("  fetch_all() called without instruments= parameter.")
            self.fetch_ok = False
            return False

        self.dte_days = (self.expiry_date - _now_ist().date()).days
        log.info(f"  DTE         : {self.dte_days}")
        if self.dte_days == 0:
            log.warning("    0DTE — strategies will apply stricter filters")

        self.fetch_ok = True
        log.info("=" * 56)
        return True

    # ── NSE helpers (VIX only — PCR removed) ─────────────────────────────────

    @staticmethod
    def _nse_session():
        """Create a requests session with NSE browser-like headers for VIX fetch."""
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept":     "application/json",
            "Referer":    "https://www.nseindia.com/",
        }
        s = requests.Session()
        s.get("https://www.nseindia.com", headers=headers, timeout=5)
        return s, headers

    def _fetch_vix(self) -> float | None:
        """
        Fetch India VIX from NSE allIndices API.

        VIX is NOT available via Zerodha WebSocket or REST API, so NSE HTTP
        is the only source. VIX changes slowly (minutes) so occasional failures
        are acceptable — the last known value is kept until next refresh.
        """
        try:
            s, h = self._nse_session()
            r = s.get("https://www.nseindia.com/api/allIndices", headers=h, timeout=5)
            for idx in r.json().get("data", []):
                if "INDIA VIX" in idx.get("indexSymbol", ""):
                    return float(idx["last"])
        except Exception as e:
            log.warning(f"  VIX fetch error: {e}")
        return None

    # NOTE: _fetch_pcr() has been removed.
    # PCR is now sourced exclusively from WsPCR (core/pcr_kite.py) which reads
    # Zerodha WebSocket OI data — no NSE HTTP, no bot-detection risk.
    # If PCR is needed and WsPCR is unavailable, the value stays None and
    # strategies skip the PCR filter (they all check: if self._pcr is not None).
