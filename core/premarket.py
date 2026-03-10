"""
core/premarket.py  (fixed)

Fixes applied:
  1. prev_close fetch: retry up to 3 times with 15s timeout (was single 7s call)
  2. PCR fetch: SKIPPED in fetch_all() — NSE OI data is never ready at 9:08 AM.
               First PCR fetch now delayed to 9:16 AM in live refresh thread.
  3. Live refresh: PCR skipped until 9:16 AM on first run to avoid empty OI data.
"""

import logging
import threading
from datetime import datetime, timedelta, date, time as dtime

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
    ):
        import time as time_mod

        def _refresh_loop():
            MARKET_OPEN  = dtime(9, 15)
            MARKET_CLOSE = dtime(15, 30)
            # FIX: delay first PCR fetch until 9:16 AM — NSE OI data not
            # populated before that on most days.
            PCR_START    = dtime(9, 16)

            last_vix_refresh = 0.0
            last_pcr_refresh = 0.0

            log.info(
                f"[LiveRefresh] Started — VIX every {vix_interval//60}min, "
                f"PCR every {pcr_interval//60}min (first PCR after 9:16 AM)"
            )

            while not stop_event.is_set():
                try:
                    now_t = datetime.now().time()

                    if not (MARKET_OPEN <= now_t <= MARKET_CLOSE):
                        stop_event.wait(timeout=30)
                        continue

                    mono = time_mod.monotonic()

                    # ── VIX refresh ──────────────────────────────────────
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

                    # ── PCR refresh (only after 9:16 AM) ─────────────────
                    # FIX: skip PCR entirely before 9:16 to avoid always-failing
                    # empty OI data at market open
                    if now_t >= PCR_START and mono - last_pcr_refresh >= pcr_interval:
                        new_pcr = self._fetch_pcr()
                        if new_pcr is not None:
                            with self._lock:
                                old = self.pcr
                                self.pcr = new_pcr
                            if old is None or abs(new_pcr - old) >= 0.01:
                                log.info(f"[LiveRefresh] PCR updated: {old} → {new_pcr:.3f}")
                            else:
                                log.debug(f"[LiveRefresh] PCR unchanged: {new_pcr:.3f}")
                        else:
                            log.warning(f"[LiveRefresh] PCR fetch failed — keeping last: {self.pcr}")
                        last_pcr_refresh = mono

                except Exception as e:
                    # FIX: catch ALL exceptions so live refresh thread never dies silently
                    log.error(f"[LiveRefresh] Unexpected error in refresh loop: {e}", exc_info=True)

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

        # ── PCR: SKIPPED at 9:08 AM — NSE OI data is empty at open ───────────
        # FIX: PCR is now fetched by live refresh thread starting at 9:16 AM.
        #      Running without PCR at open is safe — strategies handle PCR=None.
        self.pcr = None
        log.info("  PCR         : None (will be fetched at 9:16 AM by live refresh)")

        # ── Previous day OHLC — with retry and longer timeout ─────────────────
        # FIX: Kite API is overloaded at 9:08 AM. Retry 3 times with 15s timeout.
        for attempt in range(1, 4):
            try:
                import time as _t
                today = datetime.now().date()
                raw = kite.historical_data(
                    instrument_token=index_token,
                    from_date=datetime.combine(today - timedelta(days=7), dtime(9, 15)),
                    to_date=datetime.combine(today - timedelta(days=1), dtime(15, 30)),
                    interval="day",
                    timeout=15,   # FIX: was 7s, now 15s
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
                today = datetime.now().date()
                raw5m = kite.historical_data(
                    instrument_token=index_token,
                    from_date=datetime.combine(today - timedelta(days=7), dtime(15, 20)),
                    to_date=datetime.combine(today - timedelta(days=1), dtime(15, 30)),
                    interval="5minute",
                    timeout=15,
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
                from_date=datetime.now() - timedelta(days=300),
                to_date=datetime.now(),
                interval="day",
                timeout=15,
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
                self.expiry_date = get_nearest_expiry(instruments._df, date.today())
                log.info(f"  Expiry      : {self.expiry_date}  (from live NFO instruments)")
            except Exception as e:
                log.error(f"  Expiry resolution FAILED: {e}")
                self.fetch_ok = False
                return False
        else:
            log.error("  fetch_all() called without instruments= parameter.")
            self.fetch_ok = False
            return False

        self.dte_days = (self.expiry_date - date.today()).days
        log.info(f"  DTE         : {self.dte_days}")
        if self.dte_days == 0:
            log.warning("    0DTE — strategies will apply stricter filters")

        self.fetch_ok = True
        log.info("=" * 56)
        return True

    # ── NSE helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _nse_session():
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept":     "application/json",
            "Referer":    "https://www.nseindia.com/",
        }
        s = requests.Session()
        s.get("https://www.nseindia.com", headers=headers, timeout=5)
        return s, headers

    def _fetch_vix(self) -> float | None:
        try:
            s, h = self._nse_session()
            r = s.get("https://www.nseindia.com/api/allIndices", headers=h, timeout=5)
            for idx in r.json().get("data", []):
                if "INDIA VIX" in idx.get("indexSymbol", ""):
                    return float(idx["last"])
        except Exception as e:
            log.warning(f"  VIX fetch error: {e}")
        return None

    def _fetch_pcr(self) -> float | None:
        import time as _time
        url = "https://www.nseindia.com/api/option-chain-indices?symbol=BANKNIFTY"
        for attempt in range(1, 4):
            try:
                s, h = self._nse_session()
                _time.sleep(1.5)
                r = s.get(url, headers=h, timeout=8)
                if r.status_code != 200:
                    log.warning(f"  PCR fetch attempt {attempt}: HTTP {r.status_code}")
                    _time.sleep(2)
                    continue
                data = r.json()
                records = data.get("records", {}).get("data", [])
                totce, totpe = 0, 0
                for row in records:
                    ce = row.get("CE", {})
                    pe = row.get("PE", {})
                    totce += ce.get("openInterest", 0) if ce else 0
                    totpe += pe.get("openInterest", 0) if pe else 0
                if totce == 0:
                    d = data.get("filtered", {})
                    totce = d.get("CE", {}).get("totOI", 0)
                    totpe = d.get("PE", {}).get("totOI", 0)
                if totce > 0:
                    pcr = round(totpe / totce, 3)
                    log.info(f"  PCR fetch OK (attempt {attempt}): PE_OI={totpe:,} CE_OI={totce:,} PCR={pcr}")
                    return pcr
                else:
                    top_keys = list(data.get("records", {}).keys())
                    log.warning(f"  PCR attempt {attempt}: CE totOI=0 (records keys: {top_keys})")
            except Exception as e:
                log.warning(f"  PCR fetch attempt {attempt} error: {e}")
            _time.sleep(2)
        log.error("  PCR fetch failed after 3 attempts -- running without PCR filter")
        return None
