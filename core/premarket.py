"""
core/premarket.py

All pre-market data fetched ONCE and stored in PreMarketData.
Strategies read from this shared object  zero duplicate API calls.

Data fetched:
   India VIX     from NSE API (no auth needed)
   PCR           from NSE option chain API
   Previous close & body high/low  from Kite historical
   200 EMA daily  from Kite historical (300 days)
   Today's expiry + DTE
"""

import logging
import threading
from datetime import datetime, timedelta, date, time as dtime

import numpy as np
import pandas as pd
import requests

log = logging.getLogger("core.premarket")

INDEX_TOKEN = 260105   # BankNifty  fixed Zerodha token


class PreMarketData:
    """
    Single shared container. Populated by fetch_all() at startup.
    All strategies read from .vix, .pcr, .prev_close, etc.
    Thread-safe (read-only after fetch).
    """

    def __init__(self):
        self.vix          : float | None = None
        self.pcr          : float | None = None
        self.prev_close      : float | None = None
        self.prev_body_high  : float | None = None   # max(open, close) of prev day daily candle
        self.prev_body_low   : float | None = None   # min(open, close) of prev day daily candle
        # Last 5-min candle of previous day (15:25 candle) — used by Spike for gap detection
        self.prev_last5m_high  : float | None = None   # high  of prev day 15:25 candle
        self.prev_last5m_low   : float | None = None   # low   of prev day 15:25 candle
        self.prev_last5m_close : float | None = None   # close of prev day 15:25 candle
        self.ema200_daily    : float | None = None
        self.expiry_date  : date  | None = None
        self.dte_days     : int   | None = None
        self.fetch_ok     : bool         = False
        self._lock        = threading.Lock()

    # ── Live refresh ──────────────────────────────────────────────────────────

    def start_live_refresh(
        self,
        stop_event: threading.Event,
        vix_interval: int = 300,   # seconds — refresh VIX every 5 min
        pcr_interval: int = 600,   # seconds — refresh PCR every 10 min
    ):
        """
        Spawn a background daemon thread that refreshes VIX and PCR
        during market hours (9:15–15:30).

        Rules:
          • If a fetch SUCCEEDS  → update the value and log if it changed.
          • If a fetch FAILS     → keep the last known good value (never
                                   set to None mid-session).
          • Thread stops cleanly when stop_event is set (market close /
            KeyboardInterrupt in hub.run()).
        """
        def _refresh_loop():
            MARKET_OPEN  = dtime(9, 15)
            MARKET_CLOSE = dtime(15, 30)

            last_vix_refresh = 0.0
            last_pcr_refresh = 0.0

            log.info(
                f"[LiveRefresh] Started — VIX every {vix_interval//60}min, "
                f"PCR every {pcr_interval//60}min"
            )

            while not stop_event.is_set():
                now_t = datetime.now().time()

                # Only refresh during market hours
                if not (MARKET_OPEN <= now_t <= MARKET_CLOSE):
                    stop_event.wait(timeout=30)
                    continue

                mono = time_mod.monotonic()

                # ── VIX refresh ──────────────────────────────────────────
                if mono - last_vix_refresh >= vix_interval:
                    new_vix = self._fetch_vix()
                    if new_vix is not None:
                        with self._lock:
                            old = self.vix
                            self.vix = new_vix
                        if old is None or abs(new_vix - old) >= 0.05:
                            log.info(
                                f"[LiveRefresh] VIX updated: "
                                f"{old} → {new_vix:.2f}"
                            )
                        else:
                            log.debug(
                                f"[LiveRefresh] VIX unchanged: {new_vix:.2f}"
                            )
                    else:
                        log.warning(
                            f"[LiveRefresh] VIX fetch failed — "
                            f"keeping last value: {self.vix}"
                        )
                    last_vix_refresh = mono

                # ── PCR refresh ──────────────────────────────────────────
                if mono - last_pcr_refresh >= pcr_interval:
                    new_pcr = self._fetch_pcr()
                    if new_pcr is not None:
                        with self._lock:
                            old = self.pcr
                            self.pcr = new_pcr
                        if old is None or abs(new_pcr - old) >= 0.01:
                            log.info(
                                f"[LiveRefresh] PCR updated: "
                                f"{old} → {new_pcr:.3f}"
                            )
                        else:
                            log.debug(
                                f"[LiveRefresh] PCR unchanged: {new_pcr:.3f}"
                            )
                    else:
                        log.warning(
                            f"[LiveRefresh] PCR fetch failed — "
                            f"keeping last value: {self.pcr}"
                        )
                    last_pcr_refresh = mono

                # Sleep 60 s between loop iterations (fine-grained enough
                # to hit the 5-min / 10-min windows accurately)
                stop_event.wait(timeout=60)

            log.info("[LiveRefresh] Thread stopped.")

        import time as time_mod   # avoid shadowing the outer `time` import

        t = threading.Thread(target=_refresh_loop, name="PreMarket-LiveRefresh",
                             daemon=True)
        t.start()
        return t

    # ─────────────────────────────────────────────────────────────────────────

    def fetch_all(self, kite, index_token: int = INDEX_TOKEN, instruments=None):
        """
        Call once at 9:009:14 AM before market opens.
        Populates all fields. Returns True if critical data available.

        instruments (optional): InstrumentStore instance. If provided, expiry
        is derived from actual available NFO contracts instead of hardcoded
        weekday arithmetic (safer -- handles exchange-changed expiry days).
        """
        log.info("" * 56)
        log.info("  PRE-MARKET DATA FETCH (shared for all strategies)")
        log.info("" * 56)

        #  VIX 
        self.vix = self._fetch_vix()
        log.info(f"  India VIX   : {self.vix}")

        #  PCR 
        self.pcr = self._fetch_pcr()
        log.info(f"  PCR         : {self.pcr}")

        #  Previous day OHLC (daily candle — for prev_close + body high/low) 
        try:
            today  = datetime.now().date()
            raw    = kite.historical_data(
                instrument_token=index_token,
                from_date=datetime.combine(today - timedelta(days=7), dtime(9, 15)),
                to_date=datetime.combine(today - timedelta(days=1), dtime(15, 30)),
                interval="day"
            )
            if raw:
                prev                = raw[-1]
                self.prev_close     = float(prev["close"])
                self.prev_body_high = float(max(prev["open"], prev["close"]))
                self.prev_body_low  = float(min(prev["open"], prev["close"]))
                log.info(f"  Prev Close  : {self.prev_close:.2f}")
                log.info(f"  Prev Body   : [{self.prev_body_low:.2f} – {self.prev_body_high:.2f}]")
        except Exception as e:
            log.warning(f"  Prev close fetch failed: {e}")

        #  Previous day LAST 5-MIN candle (15:25 candle) — used by Spike for gap detection 
        #  Gap up   : today open > prev_last5m_high  (opened above last candle's high)
        #  Gap down : today open < prev_last5m_low   (opened below last candle's low)
        #  No gap   : today open inside last candle range → wait for 2×8s signal
        try:
            today = datetime.now().date()
            raw5m = kite.historical_data(
                instrument_token=index_token,
                from_date=datetime.combine(today - timedelta(days=7), dtime(15, 20)),
                to_date=datetime.combine(today - timedelta(days=1), dtime(15, 30)),
                interval="5minute"
            )
            if raw5m:
                last5m = raw5m[-1]   # the 15:25 candle (last 5-min of prev session)
                self.prev_last5m_high  = float(last5m["high"])
                self.prev_last5m_low   = float(last5m["low"])
                self.prev_last5m_close = float(last5m["close"])
                log.info(
                    f"  Prev Last5m : H={self.prev_last5m_high:.2f} "
                    f"L={self.prev_last5m_low:.2f} "
                    f"C={self.prev_last5m_close:.2f} "
                    f"(ts={last5m.get('date', '?')})"
                )
            else:
                log.warning("  Prev Last5m : No data returned — gap filter will fall back to daily body")
        except Exception as e:
            log.warning(f"  Prev last 5-min fetch failed: {e}")

        #  200 EMA from daily data 
        try:
            raw = kite.historical_data(
                instrument_token=index_token,
                from_date=datetime.now() - timedelta(days=300),
                to_date=datetime.now(),
                interval="day"
            )
            if len(raw) >= 50:
                closes = pd.Series([r["close"] for r in raw])
                self.ema200_daily = float(
                    closes.ewm(span=200, adjust=False).mean().iloc[-1]
                )
                current = float(raw[-1]["close"])
                bias    = "ABOVE (CE bias)" if current > self.ema200_daily else "BELOW (PE bias)"
                log.info(f"  200EMA      : {self.ema200_daily:.2f}  price {bias}")
        except Exception as e:
            log.warning(f"  200EMA fetch failed: {e}")

        #  Expiry + DTE 
        # BankNifty has NO weekly expiry (removed by SEBI in 2024) -- monthly only.
        # Always resolve from actual NFO instrument data, never from weekday arithmetic.
        from core.instruments import get_nearest_expiry
        if instruments is not None:
            try:
                self.expiry_date = get_nearest_expiry(instruments._df, date.today())
                log.info(f"  Expiry      : {self.expiry_date}  "
                         f"(from live NFO instruments)")
            except Exception as e:
                log.error(f"  Expiry resolution FAILED: {e}  "
                          f"Cannot trade safely without a confirmed expiry date.")
                self.fetch_ok = False
                return False
        else:
            log.error("  fetch_all() called without instruments= parameter. "
                      "Cannot resolve expiry safely for BankNifty monthly contracts. "
                      "Pass instruments=instruments_store to fetch_all().")
            self.fetch_ok = False
            return False

        self.dte_days = (self.expiry_date - date.today()).days
        log.info(f"  DTE         : {self.dte_days}")
        if self.dte_days == 0:
            log.warning("    0DTE  strategies will apply stricter filters")

        self.fetch_ok = True
        log.info("" * 56)
        return True

    #  NSE API helpers 

    @staticmethod
    def _nse_session():
        headers = {
            "User-Agent" : "Mozilla/5.0",
            "Accept"     : "application/json",
            "Referer"    : "https://www.nseindia.com/",
        }
        s = requests.Session()
        s.get("https://www.nseindia.com", headers=headers, timeout=5)
        return s, headers

    def _fetch_vix(self) -> float | None:
        """
        India VIX from NSE allIndices endpoint.
        VIX > 20 = fearful market, ORB breakouts fail more often.
        VIX 12-18 = normal, optimal for ORB.
        """
        try:
            s, h = self._nse_session()
            r    = s.get("https://www.nseindia.com/api/allIndices", headers=h, timeout=5)
            for idx in r.json().get("data", []):
                if "INDIA VIX" in idx.get("indexSymbol", ""):
                    return float(idx["last"])
        except Exception as e:
            log.warning(f"  VIX fetch error: {e}")
        return None

    def _fetch_pcr(self) -> float | None:
        """
        Put-Call Ratio from NSE option chain.
        PCR = Total PE OI / Total CE OI (near-month filtered)
        PCR > 1.3 = extreme bearish sentiment (PE crowded  avoid PE)
        PCR < 0.7 = extreme bullish sentiment (CE crowded  avoid CE)
        0.81.2   = balanced  trade either direction

        Retries up to 3 times -- NSE requires a session cookie from the
        homepage before the API will respond with valid JSON.
        """
        import time as _time
        url = "https://www.nseindia.com/api/option-chain-indices?symbol=BANKNIFTY"
        for attempt in range(1, 4):
            try:
                s, h = self._nse_session()
                _time.sleep(1.5)   # give NSE time to set cookies properly
                r = s.get(url, headers=h, timeout=8)
                if r.status_code != 200:
                    log.warning(f"  PCR fetch attempt {attempt}: HTTP {r.status_code}")
                    _time.sleep(2)
                    continue
                data = r.json()

                # NSE changed response structure -- real OI is now in
                # data["records"]["data"] (list of strike dicts), NOT in
                # data["filtered"] which is often empty or missing.
                records = data.get("records", {}).get("data", [])
                totce, totpe = 0, 0
                for row in records:
                    ce = row.get("CE", {})
                    pe = row.get("PE", {})
                    totce += ce.get("openInterest", 0) if ce else 0
                    totpe += pe.get("openInterest", 0) if pe else 0

                # Fallback: try the old filtered path in case NSE reverts
                if totce == 0:
                    d     = data.get("filtered", {})
                    totce = d.get("CE", {}).get("totOI", 0)
                    totpe = d.get("PE", {}).get("totOI", 0)

                if totce > 0:
                    pcr = round(totpe / totce, 3)
                    log.info(f"  PCR fetch OK (attempt {attempt}): "
                             f"PE_OI={totpe:,} CE_OI={totce:,} PCR={pcr}")
                    return pcr
                else:
                    top_keys = list(data.get("records", {}).keys())
                    log.warning(f"  PCR attempt {attempt}: CE totOI=0 "
                                f"(records keys: {top_keys})")
            except Exception as e:
                log.warning(f"  PCR fetch attempt {attempt} error: {e}")
            _time.sleep(2)
        log.error("  PCR fetch failed after 3 attempts -- running without PCR filter")
        return None
