"""

                     MULTI-STRATEGY TRADER
                         python t.py


  WHAT HAPPENS WHEN YOU RUN THIS:

  1. AUTO LOGIN  checks token.json (today's token reused if valid)
                 otherwise logs in via TOTP — no manual step needed
  2. ONE Kite login   shared by all strategies
  3. ONE NFO instrument load   shared by all strategies
  4. ONE pre-market data fetch (VIX, PCR, prev close, 200EMA)
  5. WsPCR setup — subscribe ATM±1000pt CE+PE for WebSocket OI-based PCR
  6. Each strategy's pre_market() called  they read shared data
  7. ONE WebSocket  all strategies get ticks via callbacks
  8. Strategies run fully independently after that
  9. Each strategy writes its own CSV log
  10. At 3:30 PM  all strategies get eod_summary() called, WS closes

  ACTIVE STRATEGIES:

    SPIKE       9:15 gap/spike trade, exits by 9:30
    ORB_v2      Opening range breakout, entry 9:40-10:15
    SCALPER_V7  Intraday scalper (11 filters), all-day 9:45-14:30
    BB_STOCH    Bollinger+Stochastic+Volume, all-day

  PCR SOURCE:
    WsPCR — computed from Zerodha WebSocket OI data (no NSE HTTP).
    Zerodha sends oi in every MODE_FULL tick. MarketHub stores it per
    token. WsPCR reads it and returns PCR = PE_OI / CE_OI.
    Zero external dependencies. Works on GitHub Actions.
    Fallback: if WsPCR returns None, NSE HTTP _fetch_pcr() is kept in
    premarket.py but is not called when ws_pcr= is wired.

  LOGIN MODES:
    Local run   set env vars in config_secrets.env or export in terminal
    GitHub CI   set all 5 vars as GitHub Secrets

  REQUIRED ENVIRONMENT VARIABLES:
    ZERODHA_API_KEY      Kite Connect API key
    ZERODHA_API_SECRET   Kite Connect API secret
    ZERODHA_USER_ID      Zerodha client ID  e.g. AB1234
    ZERODHA_PASSWORD     Zerodha login password
    ZERODHA_TOTP_SECRET  Base32 TOTP secret from Zerodha 2FA setup page

  RESOURCE SAVINGS (all 4 strategies share):
    1 WebSocket instead of 4 (Zerodha limit: 3 per account)
    1 Kite session (avoids double login issues)
    1 NFO instruments call (saves ~2sec + API quota)
    1 VIX fetch (NSE rate-limits aggressive scrapers)
    0 PCR HTTP calls — PCR comes from WebSocket OI ticks (WsPCR)
    1 historical data call for prev close + 200EMA
    ScalperV7 NO LONGER makes REST polling calls for candles
     candles are built live from the shared WebSocket ticks
    Same ATM option token  MarketHub deduplicates subscriptions

"""

import atexit
import logging
import os
import sys
import time
from datetime import datetime, time as dtime, timezone, timedelta

# ── IST timezone (UTC+5:30) ───────────────────────────────────────────────────
IST = timezone(timedelta(hours=5, minutes=30))

def now_ist():
    """Always returns current datetime in IST — works on GitHub Actions (UTC) and local."""
    return datetime.now(tz=IST).replace(tzinfo=None)

# ── Load .env file for local runs (silently ignored if file doesn't exist) ──
try:
    from dotenv import load_dotenv
    load_dotenv("config_secrets.env")
except ImportError:
    pass   # python-dotenv not installed — env vars must already be exported

# ── Core framework ───────────────────────────────────────────────────────────
from core.auto_login  import auto_login
from core.market_hub  import MarketHub
from core.instruments import InstrumentStore
from core.premarket   import PreMarketData
from core.pcr_kite    import WsPCR

# ── Strategies — ADD/REMOVE here to enable/disable ──────────────────────────
from strategies.spike               import SpikeStrategy
from strategies.orb_v2              import ORBStrategy
from strategies.scalper_v7_strategy import ScalperV7Strategy
from strategies.bb_stoch_strategy   import BBStochStrategy

# ─────────────────────────────────────────────────────────────────────────────
#  STRATEGY REGISTRY — Add new strategy CLASS here (not instance)
#  Order matters: first in list = first to receive ticks
# ─────────────────────────────────────────────────────────────────────────────
ACTIVE_STRATEGIES = [
    SpikeStrategy,       # Spike:      9:15-9:30   gap/spike trade
    ORBStrategy,         # ORB v2:     9:40-10:15  breakout trade
    ScalperV7Strategy,   # Scalper V7: all-day,    11-filter momentum scalper
    BBStochStrategy,     # BB+Stoch:   all-day,    Bollinger+Stochastic+Volume
]

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
TOKEN_FILE     = "token.json"
LOG_FILE       = "trader.log"
PID_FILE       = "trader.pid"
PREMARKET_TIME = dtime(9,  8)    # Start pre-market setup at 9:08 AM IST
MARKET_START   = dtime(9, 14)    # Start WebSocket at 9:14 AM IST
MARKET_END     = dtime(15, 0)    # MarketHub forces WS close at 3:31 PM IST

# WsPCR config — subscribe ATM ± PCR_RANGE pt CE+PE tokens for OI-based PCR
# BankNifty strikes are 100pt apart. range=1000 → 21 strikes each side = 42 tokens.
# Most overlap with BB_STOCH ATM±400 tokens (deduped by hub.subscribe()).
PCR_SPOT_RANGE = 1000   # points either side of ATM
PCR_STRIKE_STEP = 100   # BankNifty strike interval

# ─────────────────────────────────────────────────────────────────────────────
#  LOGGING — Console + file, all strategies use same logger
# ─────────────────────────────────────────────────────────────────────────────
def setup_logging():
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_FILE, mode="a"),
        ],
    )

setup_logging()
log = logging.getLogger("t")


# ─────────────────────────────────────────────────────────────────────────────
#  PID LOCK — Prevent two instances running simultaneously
# ─────────────────────────────────────────────────────────────────────────────
def acquire_pid_lock():
    if os.path.isfile(PID_FILE):
        with open(PID_FILE) as f:
            old_pid = f.read().strip()
        try:
            os.kill(int(old_pid), 0)
            log.error(
                f"Trader already running (PID {old_pid}). "
                f"If stale, delete '{PID_FILE}' and retry."
            )
            sys.exit(1)
        except (OSError, ValueError):
            log.warning(f"Stale PID file (PID {old_pid}) — overwriting")

    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))

    atexit.register(lambda: os.remove(PID_FILE) if os.path.isfile(PID_FILE) else None)
    log.info(f"PID lock acquired (PID {os.getpid()})")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("""

        ══ MULTI-STRATEGY TRADER ══ python t.py ══
     SPIKE  +  ORB v2  +  SCALPER V7  +  BB STOCH  (paper mode)

""")

    # ── PID lock ──────────────────────────────────────────────────────────────
    acquire_pid_lock()

    # ── AUTO LOGIN ────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("  AUTO LOGIN")
    log.info("=" * 60)
    auto_login()   # writes / refreshes token.json

    # ── Create MarketHub (single connection layer) ────────────────────────────
    hub = MarketHub(token_file=TOKEN_FILE)
    hub.load_kite()

    # ── Load NFO instruments ONCE (shared by all strategies) ─────────────────
    instruments = InstrumentStore()
    instruments.load(hub.kite, option_root="BANKNIFTY")

    # ── Instantiate strategies and register with hub ──────────────────────────
    strategies = []
    for StratClass in ACTIVE_STRATEGIES:
        strat = StratClass(hub)
        hub.register(strat)
        strategies.append(strat)

    log.info(f"{len(strategies)} strategies loaded: {[s.name for s in strategies]}")

    # ── Sleep until pre-market setup time (9:08 AM IST) ──────────────────────
    now = now_ist()   # ← IST fix: was datetime.now() which gave UTC on GitHub
    if now.time() < PREMARKET_TIME:
        wait = (datetime.combine(now.date(), PREMARKET_TIME) - now).seconds
        log.info(f"Waiting {wait}s until 9:08 AM IST pre-market setup...")
        time.sleep(wait)

    # ── Fetch shared pre-market data ONCE ────────────────────────────────────
    log.info("=" * 60)
    log.info("  SHARED PRE-MARKET DATA FETCH")
    log.info("=" * 60)
    pm = PreMarketData()
    pm.fetch_all(hub.kite, index_token=260105, instruments=instruments)

    # ── WsPCR setup ───────────────────────────────────────────────────────────
    # Create WsPCR AFTER fetch_all() so expiry_date is available.
    # setup() subscribes ATM±PCR_SPOT_RANGE CE+PE tokens.
    # Since the WebSocket isn't connected yet, hub.last_price(index_token)
    # returns None — WsPCR.setup() detects this and defers subscriptions
    # to the first compute_pcr() call (which happens at 9:16 AM after the
    # WebSocket has connected and the first index tick has arrived).
    #
    # Token count: range=1000, step=100 → 21 strikes × 2 types = 42 tokens.
    # ~20 of these overlap with BB_STOCH's ATM±400 (deduped by hub.subscribe()).
    # Net new subscriptions: ~22 tokens. Negligible bandwidth overhead.
    log.info("=" * 60)
    log.info("  WsPCR SETUP — WebSocket OI-based PCR")
    log.info("=" * 60)
    ws_pcr = None
    if pm.expiry_date is not None:
        ws_pcr = WsPCR(
            hub         = hub,
            instruments = instruments,
            expiry_date = pm.expiry_date,
            spot_range  = PCR_SPOT_RANGE,
            step        = PCR_STRIKE_STEP,
        )
        ws_pcr.setup()   # deferred if WS not yet connected — OK
        log.info(
            f"  WsPCR ready | expiry={pm.expiry_date} | "
            f"range=±{PCR_SPOT_RANGE}pt | step={PCR_STRIKE_STEP}pt"
        )
    else:
        log.warning("  WsPCR skipped — expiry_date not available (fetch_all failed?)")

    # ── Live refresh: VIX every 5 min, PCR every 10 min ──────────────────────
    # When ws_pcr is not None, PCR is read from WebSocket OI (no NSE HTTP).
    # When ws_pcr is None (expiry fetch failed), NSE HTTP fallback is used.
    pm.start_live_refresh(
        hub._done,
        vix_interval=300,
        pcr_interval=600,
        ws_pcr=ws_pcr,        # ← NEW: wire WsPCR into the refresh loop
    )

    # ── Call each strategy's pre_market() with shared data ───────────────────
    log.info("=" * 60)
    log.info("  STRATEGY PRE-MARKET SETUP")
    log.info("=" * 60)
    active_strategies = []
    for strat in strategies:
        try:
            ok = strat.pre_market(pm, instruments)
            if ok:
                active_strategies.append(strat)
                log.info(f"  [{strat.name}] ready for today")
            else:
                log.info(f"  [{strat.name}] skipping today (pre_market returned False)")
        except Exception as e:
            log.error(f"  [{strat.name}] pre_market error: {e}")

    if not active_strategies:
        log.info("No strategies active today. Exiting.")
        return

    # ── Backfill historical candles (warm up indicators for late starts) ──────
    hub.backfill(hub.kite, index_token=260105)

    # ── Sleep until WebSocket start time (9:14 AM IST) ───────────────────────
    now = now_ist()   # ← IST fix: was datetime.now() which gave UTC on GitHub
    if now.time() < MARKET_START:
        wait = (datetime.combine(now.date(), MARKET_START) - now).seconds
        log.info(f"Waiting {wait}s until 9:14 AM IST to start WebSocket...")
        time.sleep(wait)

    # ── Print active strategy summary ─────────────────────────────────────────
    log.info("=" * 60)
    log.info("  MARKET START — ALL STRATEGIES ACTIVE")
    log.info("=" * 60)
    log.info(f"  Login     : AUTO (TOTP) -> token.json")
    log.info(f"  VIX       : {pm.vix}")
    log.info(f"  PCR       : {pm.pcr}  (live via {'WsPCR' if ws_pcr else 'NSE-HTTP'})")
    log.info(f"  Expiry    : {pm.expiry_date}  (DTE={pm.dte_days})")
    log.info(f"  200EMA    : {pm.ema200_daily}")
    log.info(f"  Prev Close: {pm.prev_close}")
    log.info(f"  WsPCR     : {'enabled' if ws_pcr else 'disabled (expiry missing)'}")
    if ws_pcr:
        log.info(f"  WsPCR cfg : range=±{PCR_SPOT_RANGE}pt step={PCR_STRIKE_STEP}pt")
    log.info(f"  Strategies: {[s.name for s in active_strategies]}")
    log.info("=" * 60)

    # ── Start MarketHub (WebSocket) — runs until 3:31 PM IST ─────────────────
    hub.run()

    # ── Log WsPCR final summary ───────────────────────────────────────────────
    if ws_pcr:
        ws_pcr.log_summary()

    log.info("Trader session complete.")


if __name__ == "__main__":
    main()

