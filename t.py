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

    SPIKE        9:15 gap/spike trade, exits by 9:30
    ORB_v2       Opening range breakout, entry 9:40-10:15
    SCALPER_V7   Intraday scalper (11 filters), all-day 9:45-14:30
    BB_STOCH     Bollinger+Stochastic+Volume, all-day
    HEDGED_SELL  Iron Condor (sell OTM CE+PE, buy hedge), entry 9:30-10:15
    SMART_HEDGE  Directional spread — scores PCR+EMA+gap+candle, auto-picks
                 Bull Put Spread / Bear Call Spread / Iron Condor, entry 9:35-10:15

  PCR SOURCE:
    WsPCR — computed from Zerodha WebSocket OI data (no NSE HTTP).
    Zerodha sends oi in every MODE_FULL tick. MarketHub stores it per
    token. WsPCR reads it and returns PCR = PE_OI / CE_OI.
    Zero external dependencies. Works on GitHub Actions.
    If WsPCR returns None (OI not yet warm), PCR stays None for that cycle.
    NSE HTTP _fetch_pcr() has been removed — WsPCR is the only PCR source.

  LOGIN MODES:
    Local run   set env vars in config_secrets.env or export in terminal
    GitHub CI   set all 5 vars as GitHub Secrets

  REQUIRED ENVIRONMENT VARIABLES:
    ZERODHA_API_KEY      Kite Connect API key
    ZERODHA_API_SECRET   Kite Connect API secret
    ZERODHA_USER_ID      Zerodha client ID  e.g. AB1234
    ZERODHA_PASSWORD     Zerodha login password
    ZERODHA_TOTP_SECRET  Base32 TOTP secret from Zerodha 2FA setup page

  RESOURCE SAVINGS (all 6 strategies share):
    1 WebSocket instead of 6 (Zerodha limit: 3 per account)
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
from core.order_router import OrderRouter

# ── Strategies — ADD/REMOVE here to enable/disable ──────────────────────────
from strategies.spike                import SpikeStrategy
from strategies.spike_nifty          import SpikeNiftyStrategy
from strategies.orb_v2               import ORBStrategy
from strategies.scalper_v7_strategy  import ScalperV7Strategy
from strategies.bb_stoch_strategy         import BBStochStrategy
from strategies.bb_stoch_nifty_strategy  import BBStochNiftyStrategy
from strategies.hedged_sell_strategy import HedgedSellStrategy
from strategies.smart_hedge_strategy import SmartHedgeStrategy
from strategies.nifty_expiry_straddle_strategy     import NiftyExpiryStraddleStrategy
from strategies.banknifty_expiry_momentum_strategy  import BankNiftyExpiryMomentumStrategy
from strategies.nifty_directional_strategy          import NiftyDirectionalStrategy
from strategies.nifty_fut_directional_strategy      import NiftyFutDirectionalStrategy
from strategies.nifty_candle_breakout_strategy       import NiftyCandleBreakoutStrategy
from strategies.banknifty_candle_breakout_strategy    import BankNiftyCandleBreakoutStrategy

# ─────────────────────────────────────────────────────────────────────────────
#  STRATEGY REGISTRY — Add new strategy CLASS here (not instance)
#  Order matters: first in list = first to receive ticks
# ─────────────────────────────────────────────────────────────────────────────
ACTIVE_STRATEGIES = [
    SpikeStrategy,                    # Spike:                    9:15-9:30   BankNifty gap/spike
    SpikeNiftyStrategy,               # Spike Nifty:              9:15-9:30   Nifty 50 gap/spike
    ORBStrategy,                      # ORB v2:                   9:40-10:15  breakout trade
    ScalperV7Strategy,                # Scalper V7:               all-day     11-filter momentum scalper
    BBStochStrategy,                  # BB+Stoch BankNifty:       all-day     Bollinger+Stoch+Volume
    BBStochNiftyStrategy,             # BB+Stoch Nifty:           all-day     Bollinger+Stoch+Volume (Nifty)
    HedgedSellStrategy,               # Hedged Sell:              9:30-10:15  Iron Condor classic
    SmartHedgeStrategy,               # Smart Hedge:              9:35-10:15  directional spread auto-pick
    NiftyExpiryStraddleStrategy,      # Nifty Expiry Straddle:    9:20-11:30  SHORT straddle, Nifty WEEKLY expiry only
    BankNiftyExpiryMomentumStrategy,  # BankNifty Expiry Momentum: 14:00-15:20 directional buy, BankNifty MONTHLY expiry only
    NiftyDirectionalStrategy,         # Nifty Directional:         9:30-14:30  CE/PE buy on directional days, Mode A+B
    NiftyFutDirectionalStrategy,      # Nifty Fut Directional:     9:30-14:30  Nifty FUTURES LONG on directional days, Mode A+B
    NiftyCandleBreakoutStrategy,      # Nifty Candle Breakout:     9:15-15:15  10s marubozu + 5s confirm + tick breakout
    BankNiftyCandleBreakoutStrategy,  # BankNifty Candle Breakout: 9:15-15:15  20pt marubozu + 5s confirm + tick breakout
]

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
TOKEN_FILE     = "token.json"
LOG_DIR        = "logs"
PID_FILE       = "trader.pid"
PREMARKET_TIME = dtime(9,  8)    # Start pre-market setup at 9:08 AM IST
MARKET_START   = dtime(9, 14)    # Start WebSocket at 9:14 AM IST
MARKET_END     = dtime(15, 0)    # MarketHub forces WS close at 3:31 PM IST

PCR_SPOT_RANGE  = 1000   # points either side of ATM
PCR_STRIKE_STEP       = 100    # strike interval for BankNifty WsPCR subscriptions
NIFTY_PCR_STRIKE_STEP = 50     # strike interval for Nifty 50 WsPCR subscriptions


# ─────────────────────────────────────────────────────────────────────────────
#  LOGGING SETUP
# ─────────────────────────────────────────────────────────────────────────────
def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)

    # Date stamp in IST so log filenames always reflect the trading day,
    # even when the process is running in UTC (e.g. GitHub Actions).
    _date = now_ist().strftime("%Y-%m-%d")

    # Each call creates a dated sub-folder  logs/2026-05-13/
    dated_dir = os.path.join(LOG_DIR, _date)
    os.makedirs(dated_dir, exist_ok=True)

    def _file(basename):
        """Return a FileHandler writing to  logs/<YYYY-MM-DD>/<basename>."""
        name, ext = os.path.splitext(basename)
        filename = f"{name}_{_date}{ext}"          # e.g. core_2026-05-13.log
        h = logging.FileHandler(
            os.path.join(dated_dir, filename),
            encoding="utf-8",
        )
        h.setFormatter(logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
            datefmt="%H:%M:%S",
        ))
        return h

    def _console():
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
            datefmt="%H:%M:%S",
        ))
        return h

    # Root logger → core_<date>.log + console
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(_file("core.log"))
    root.addHandler(_console())

    # ── Per-strategy loggers → own dated file, propagate=False ───────────────
    _STRAT = {
        "strategy.spike":                       "spike.log",
        "strategy.spike_nifty":                 "spike_nifty.log",
        "strategy.orb":                         "orb_v2.log",
        "strategy.scalper_v7":                  "scalper_v7.log",
        "strategy.bb_stoch":                    "bb_stoch.log",
        "strategy.bb_stoch_nifty":              "bb_stoch_nifty.log",
        "strategy.hedged_sell":                 "hedged_sell.log",
        "strategy.smart_hedge":                 "smart_hedge.log",
        "strategy.nifty_expiry_straddle":       "nifty_expiry_straddle.log",
        "strategy.banknifty_expiry_momentum":   "banknifty_expiry_momentum.log",
        "strategy.nifty_directional":            "nifty_directional.log",
        "strategy.nifty_fut_directional":        "nifty_fut_directional.log",
        "strategy.nifty_candle_breakout":        "nifty_candle_breakout.log",
        "strategy.banknifty_candle_breakout":     "banknifty_candle_breakout.log",
    }
    for name, fname in _STRAT.items():
        lg = logging.getLogger(name)
        lg.setLevel(logging.DEBUG)
        lg.propagate = False          # ← stays OUT of core.log
        lg.addHandler(_file(fname))
        lg.addHandler(_console())

    # Emit the dated log directory so it is easy to find in output
    logging.getLogger("t").info("Logs for today → %s/", dated_dir)


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
  SPIKE + SPIKE_NIFTY + ORB v2 + SCALPER V7 + BB STOCH + BB STOCH NIFTY
  + HEDGED SELL + SMART HEDGE
  + NIFTY EXPIRY STRADDLE  (Nifty weekly expiry only, 9:20–11:30)
  + BANKNIFTY EXPIRY MOMENTUM  (BankNifty monthly expiry only, 14:00–15:20)
  + NIFTY DIRECTIONAL  (Nifty CE/PE buy, Mode A+B, 9:30–14:30)
  + NIFTY FUT DIRECTIONAL  (Nifty Futures LONG, Mode A+B, 9:30–14:30)
  + NIFTY CANDLE BREAKOUT  (10s marubozu + 5s confirm + tick breakout, 9:15–15:15)
  (paper mode)

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

    # ── Create OrderRouter (single point for all order placement) ─────────────
    hub.order_router = OrderRouter(hub)
    log.info("OrderRouter attached to hub (all strategies in PAPER mode by default)")

    # ── Load NFO instruments ONCE (shared by all strategies) ─────────────────
    instruments = InstrumentStore()
    instruments.load(hub.kite, option_root="BANKNIFTY")

    # ── Load Nifty instruments for SpikeNiftyStrategy ─────────────────────────
    # SpikeNiftyStrategy uses option_root="NIFTY" (50-pt strike intervals).
    # A separate InstrumentStore is needed because BANKNIFTY and NIFTY have
    # different strike names, steps, and expiry calendars in NFO.
    nifty_instruments = InstrumentStore()
    nifty_instruments.load(hub.kite, option_root="NIFTY")

    # ── Register Nifty 50 index token with MarketHub ───────────────────────────
    # MarketHub routes ticks for this token exclusively to strategies whose
    # INDEX_TOKEN class attribute == 256265 (i.e. SpikeNiftyStrategy).
    # Must be called before hub.run() so _on_connect subscribes it.
    hub.add_index_token(256265)   # NSE:NIFTY 50 — fixed Zerodha instrument token

    # ── Instantiate strategies and register with hub ──────────────────────────
    strategies = []
    for StratClass in ACTIVE_STRATEGIES:
        strat = StratClass(hub)
        hub.register(strat)
        strategies.append(strat)

    log.info(f"{len(strategies)} strategies loaded: {[s.name for s in strategies]}")

    # ── Sleep until pre-market setup time (9:08 AM IST) ──────────────────────
    now = now_ist()
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

    # ── Fetch Nifty pre-market data for SpikeNiftyStrategy ───────────────────
    # Separate PreMarketData instance: Nifty 50 index token (256265),
    # Nifty instruments. SpikeNiftyStrategy.pre_market() receives this instead
    # of the shared BankNifty pm.
    log.info("=== Nifty Pre-Market Fetch ===")
    nifty_pm = PreMarketData()
    nifty_pm.fetch_all(hub.kite, index_token=256265, instruments=nifty_instruments)

    # ── WsPCR setup ───────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("  WsPCR SETUP — WebSocket OI-based PCR")
    log.info("=" * 60)

    # ── BankNifty WsPCR ───────────────────────────────────────────────────────
    ws_pcr = None
    if pm.expiry_date is not None:
        ws_pcr = WsPCR(
            hub         = hub,
            instruments = instruments,
            expiry_date = pm.expiry_date,
            spot_range  = PCR_SPOT_RANGE,
            step        = PCR_STRIKE_STEP,
            # index_token omitted → defaults to hub._index_token (BankNifty 260105)
        )
        ws_pcr.setup()
        log.info(
            f"  BankNifty WsPCR ready | expiry={pm.expiry_date} | "
            f"range=±{PCR_SPOT_RANGE}pt | step={PCR_STRIKE_STEP}pt"
        )
    else:
        log.warning("  BankNifty WsPCR skipped — expiry_date not available (fetch_all failed?)")

    # ── Nifty 50 WsPCR (Bug C fix) ────────────────────────────────────────────
    # BB_STOCH_NIFTY and SPIKE_NIFTY both use PCR as a directional filter.
    # Before this fix they received BankNifty PCR, which diverges from Nifty
    # PCR on days with sector-specific option flows.  A dedicated Nifty WsPCR
    # instance uses nifty_instruments (Nifty 50 chain), 50pt strike step, and
    # reads spot from Nifty index token 256265 — fully independent of the
    # BankNifty ws_pcr above.
    nifty_ws_pcr = None
    if nifty_pm.expiry_date is not None:
        nifty_ws_pcr = WsPCR(
            hub         = hub,
            instruments = nifty_instruments,
            expiry_date = nifty_pm.expiry_date,
            spot_range  = PCR_SPOT_RANGE,
            step        = NIFTY_PCR_STRIKE_STEP,
            index_token = 256265,          # Bug C fix: Nifty 50 spot, not BankNifty
        )
        nifty_ws_pcr.setup()
        log.info(
            f"  Nifty 50 WsPCR ready | expiry={nifty_pm.expiry_date} | "
            f"range=±{PCR_SPOT_RANGE}pt | step={NIFTY_PCR_STRIKE_STEP}pt"
        )
    else:
        log.warning("  Nifty 50 WsPCR skipped — nifty expiry_date not available")

    # ── Live refresh: VIX every 5 min, PCR every 10 min ──────────────────────
    # BankNifty PreMarketData: uses BankNifty ws_pcr
    pm.start_live_refresh(
        hub._done,
        vix_interval=300,
        pcr_interval=600,
        ws_pcr=ws_pcr,
    )
    # Nifty 50 PreMarketData: uses dedicated Nifty ws_pcr (Bug C fix)
    # BB_STOCH_NIFTY and SPIKE_NIFTY read pm.pcr via nifty_pm — they now
    # get true Nifty 50 OI-based PCR instead of BankNifty PCR.
    # vix_interval=86400: VIX is India-wide (same for both indices) and is
    # already being refreshed every 5 min by the BankNifty pm thread above.
    # Setting a 24-hour interval here ensures the Nifty thread never fires
    # an extra VIX HTTP call, while still allowing the refresh loop to run
    # for PCR updates.  vix_interval=0 would call _fetch_vix() every 60 s
    # (every loop iteration) — wasteful and potentially rate-limited by NSE.
    nifty_pm.start_live_refresh(
        hub._done,
        vix_interval=86400,      # effectively disabled — BankNifty pm handles VIX
        pcr_interval=600,
        ws_pcr=nifty_ws_pcr,
    )

    # ── Call each strategy's pre_market() with shared data ───────────────────
    log.info("=" * 60)
    log.info("  STRATEGY PRE-MARKET SETUP")
    log.info("=" * 60)
    active_strategies = []
    for strat in strategies:
        try:
            # SpikeNiftyStrategy needs Nifty-specific PreMarketData and instruments.
            # All other strategies use the shared BankNifty pm and instruments.
            strat_index = getattr(strat, "INDEX_TOKEN", None)
            if strat_index == 256265:
                ok = strat.pre_market(nifty_pm, nifty_instruments)
            else:
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

    # ── Backfill Nifty 5-min candles for BBStochNiftyStrategy ─────────────────
    # MarketHub's backfill filtering (INDEX_TOKEN routing) ensures only
    # strategies with INDEX_TOKEN==256265 receive these candles.
    # SpikeNiftyStrategy.on_candle is a no-op, so this only warms up
    # BBStochNiftyStrategy's internal _buf_5m indicator buffer.
    hub.backfill(hub.kite, index_token=256265)

    # ── Sleep until WebSocket start time (9:14 AM IST) ───────────────────────
    now = now_ist()
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
    log.info(f"  PCR (BNF) : {pm.pcr}  (live via {'BankNifty WsPCR' if ws_pcr else 'disabled'})")
    log.info(f"  PCR (NF)  : {nifty_pm.pcr}  (live via {'Nifty WsPCR' if nifty_ws_pcr else 'disabled'})")
    log.info(f"  Expiry    : {pm.expiry_date}  (DTE={pm.dte_days})")
    log.info(f"  200EMA    : {pm.ema200_daily}")
    log.info(f"  Prev Close: {pm.prev_close}")
    log.info(f"  BNF WsPCR : {'enabled' if ws_pcr else 'disabled (expiry missing)'}")
    if ws_pcr:
        log.info(f"  BNF WsPCR cfg : range=±{PCR_SPOT_RANGE}pt step={PCR_STRIKE_STEP}pt")
    log.info(f"  NF  WsPCR : {'enabled' if nifty_ws_pcr else 'disabled (expiry missing)'}")
    if nifty_ws_pcr:
        log.info(f"  NF  WsPCR cfg : range=±{PCR_SPOT_RANGE}pt step={NIFTY_PCR_STRIKE_STEP}pt")
    log.info(f"  Strategies: {[s.name for s in active_strategies]}")
    log.info(f"  --- Nifty 50 (SPIKE_NIFTY) ---")
    log.info(f"  Nifty Expiry  : {nifty_pm.expiry_date}  (DTE={nifty_pm.dte_days})")
    log.info(f"  Nifty Prev Cls: {nifty_pm.prev_close}")
    log.info("=" * 60)

    # ── Start MarketHub (WebSocket) — runs until 3:31 PM IST ─────────────────
    hub.run()

    # ── WsPCR EOD teardown (Bug B fix) ────────────────────────────────────────
    # Properly release all hub refcounts held by WsPCR instances.
    # hub.unsubscribe() only removes a token from the WebSocket when the
    # refcount hits 0 — any other strategy still holding the token is unaffected.
    if ws_pcr:
        ws_pcr.teardown()
        ws_pcr.log_summary()
    if nifty_ws_pcr:
        nifty_ws_pcr.teardown()
        nifty_ws_pcr.log_summary()

    log.info("Trader session complete.")


if __name__ == "__main__":
    main()
