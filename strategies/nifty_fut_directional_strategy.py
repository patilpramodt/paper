"""
strategies/nifty_fut_directional_strategy.py

NIFTY_FUT_DIRECTIONAL — Nifty 50 Futures Intraday Directional Strategy
════════════════════════════════════════════════════════════════════════

INSTRUMENT
  Index  : NSE:NIFTY 50     (INDEX_TOKEN = 256265, Zerodha fixed)
  Trade  : Nifty Futures     (NFO:NIFTY-FUT, nearest month contract)
  Lot    : 25 units per lot  (verify before live — SEBI revises periodically)

WHY FUTURES OVER OPTIONS FOR DIRECTIONAL
  Options need 3 things: right direction + right timing + low IV.
  Futures only need right direction. No theta, no IV crush, no premium
  decay. A slow 30-pt grind in futures gives full delta gain. Same grind
  in options = theta eats you alive.

  One scenario where options win: large + fast (80+ pts in < 60 min).
  That is handled by NIFTY_DIRECTIONAL (options strategy). This strategy
  handles everything else: normal trending days and momentum days alike.

MODE DETECTION  (finalised at first 5-min candle close at/after 9:20 AM)
  Mode B triggers if ANY of:
    • Gap at open > 0.4% vs prev_close
    • First-15-min OR range > 80 pts
    • Expiry day (Thursday) AND gap > 0.2%
  Otherwise: Mode A.

MODE A — NORMAL TRENDING DAY
  Entry windows: 9:30–11:00 AM  and  1:15–2:15 PM
  Hard cutoff  : no new entries after 2:30 PM

  All 4 must pass simultaneously:
    1. EMA stack  : EMA9 > EMA20 > EMA50 (LONG) or EMA9 < EMA20 < EMA50 (SHORT)
    2. VWAP side  : price above VWAP (LONG) or below VWAP (SHORT)
    3. RSI(14)    : 55–68 (LONG) or 32–45 (SHORT)  — not overbought/oversold
    4. Pullback   : last 2 candle lows ≤ EMA9 (LONG), resume close > EMA9
                    last 2 candle highs ≥ EMA9 (SHORT), resume close < EMA9

  Optional PCR gate when live PCR is available:
    LONG  needs PCR ≥ 0.80  (not extreme bearish OI skew)
    SHORT needs PCR ≤ 1.20  (not extreme bullish OI skew)

MODE B — MOMENTUM / EXPIRY / GAP-AND-GO DAY
  Entry window : 9:15–11:00 AM only

  After the initial spike, wait for a micro-consolidation (2–4 bars,
  bar range < 80 pts). Entry = breakout candle close above consolidation
  high (LONG) or below consolidation low (SHORT).

  No IV concern here — futures have no IV. This makes Mode B entries
  cleaner than options: we can enter the consolidation breakout without
  worrying about premium expansion or collapse.

FUTURES-SPECIFIC TRADE MANAGEMENT (index points based, NOT premium %)
  SL      : entry_price − SL_PTS (LONG) or + SL_PTS (SHORT)  [30 pts]
  Trail 1 : when profit ≥ TRAIL1_TRIGGER_PTS [40 pts] → move SL to breakeven
  Trail 2 : when profit ≥ TRAIL2_START_PTS [60 pts] → trail SL at
             entry + (current_profit − TRAIL2_LOCK_PTS) [lock 40 pts]
  Target  : SOFT target at TGT_PTS [90 pts] → close half if HALF_EXIT=True,
             else just trail and let it run.
  Hard exit: 3:00 PM flat, no exceptions.

FUTURES TOKEN LOOKUP
  InstrumentStore is loaded with option_root="NIFTY" in t.py.
  The same _df DataFrame contains Nifty futures (instrument_type="FUT").
  get_futures_token() added to this strategy fetches the nearest-expiry
  Nifty futures contract directly from _df — no API call needed.

  Futures tradingsymbol format: NIFTY25JUNFUT  (NIFTY + YY + MMM + FUT)
  Instrument type in NFO  : "FUT"
  Exchange                : NFO
  Product                 : MIS (intraday, auto square-off by broker)

ORDER FLOW
  BUY  (LONG  entry): _place_buy()              → (order_id, fill_price)
  SELL (exit  LONG) : _place_sell_with_retry()  → (order_id, fill_price)
  SELL (SHORT entry): _place_buy() on SELL side — NOT YET IMPLEMENTED.
                      Short selling futures in MIS is allowed on NSE, but
                      requires careful margin handling. Current version:
                      LONG-ONLY. Short signals are logged but not executed.
                      Short selling support added in next version.

ISOLATION
  INDEX_TOKEN = 256265 ensures only Nifty 50 ticks arrive here.
  No BankNifty state touched. No shared state with NIFTY_DIRECTIONAL.
  LIVE_MODE = False by default. Other strategies completely unaffected.

CSV LOG
  nifty_fut_directional_trades.csv
  Columns: timestamp, direction, action, futures_sym, futures_token,
           index_price, entry_price, sl, pnl, reason, mode, vix, pcr, order_id
"""

import csv
import logging
import os
import threading
from collections import deque
from datetime import datetime, date, time as dtime, timedelta, timezone
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from core.base_strategy import BaseStrategy
from core.candle import CandleBuilder
from core.vwap import SessionVWAP

log = logging.getLogger("strategy.nifty_fut_directional")

_IST = timezone(timedelta(hours=5, minutes=30))


def _now_ist() -> datetime:
    return datetime.now(tz=_IST).replace(tzinfo=None)


# ─────────────────────────────────────────────────────────────────────────────
#  LIVE MODE — change to True only when ready for real orders
# ─────────────────────────────────────────────────────────────────────────────
LIVE_MODE = False

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CFG = {
    # ── Lot ──────────────────────────────────────────────────────────────────
    "quantity"                : 25,      # Nifty futures lot size (1 lot = 25 units)

    # ── Session windows ───────────────────────────────────────────────────────
    "market_open"             : dtime(9, 15),
    "mode_a_start"            : dtime(9, 30),
    "mode_a_window1_end"      : dtime(11, 0),
    "mode_a_window2_start"    : dtime(13, 15),
    "mode_a_window2_end"      : dtime(14, 15),
    "mode_b_start"            : dtime(9, 15),
    "mode_b_cutoff"           : dtime(11, 0),
    "entry_cutoff"            : dtime(14, 30),
    "hard_exit_time"          : dtime(15, 0),

    # ── Mode detection ────────────────────────────────────────────────────────
    "mode_b_gap_pct"          : 0.004,   # > 0.4% gap → Mode B
    "mode_b_gap_pct_expiry"   : 0.002,   # > 0.2% gap on expiry day → Mode B
    "mode_b_range_trigger"    : 80,      # first-15-min range > 80 pts → Mode B
    "mode_b_consol_range"     : 80,      # consolidation bar range must be < this
    "mode_b_consol_min_bars"  : 2,       # min bars needed before breakout check
    "mode_b_consol_max_bars"  : 4,       # consolidation expires after this many bars

    # ── Indicators ────────────────────────────────────────────────────────────
    "ema_fast"                : 9,
    "ema_mid"                 : 20,
    "ema_slow"                : 50,
    "rsi_period"              : 14,
    "buf_size"                : 120,

    # ── Mode A filters ────────────────────────────────────────────────────────
    "rsi_long_min"            : 55,
    "rsi_long_max"            : 68,
    "rsi_short_min"           : 32,
    "rsi_short_max"           : 45,
    "pcr_min_long"            : 0.80,
    "pcr_max_short"           : 1.20,

    # ── SL / Trail (index points) ─────────────────────────────────────────────
    "sl_pts"                  : 30,      # SL distance from entry in index points
    "trail1_trigger_pts"      : 40,      # profit pts to trigger Trail-1 (move to BE)
    "trail2_start_pts"        : 60,      # profit pts to start Trail-2
    "trail2_lock_pts"         : 40,      # pts of profit locked in Trail-2

    # ── Target (soft, optional half-exit) ─────────────────────────────────────
    "tgt_pts"                 : 90,      # soft target in index points
    "half_exit_at_target"     : False,   # set True to close half lot at target

    # ── Trade limit ───────────────────────────────────────────────────────────
    "max_trades_per_day"      : 2,

    # ── SL grace period after BUY confirm ────────────────────────────────────
    "sl_grace_seconds"        : 10,

    # ── CSV ───────────────────────────────────────────────────────────────────
    "csv_file"                : "nifty_fut_directional_trades.csv",
}

_CSV_FIELDS = [
    "timestamp", "direction", "action", "futures_sym", "futures_token",
    "index_price", "entry_price", "exit_price", "sl", "pnl",
    "reason", "mode", "vix", "pcr", "order_id",
]


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _csv_append(row: dict):
    fname  = CFG["csv_file"]
    exists = os.path.isfile(fname)
    with open(fname, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in _CSV_FIELDS})


# ─────────────────────────────────────────────────────────────────────────────
#  FUTURES TOKEN LOOKUP
#  InstrumentStore._df already contains Nifty FUT rows (instrument_type="FUT",
#  name="NIFTY") alongside the CE/PE rows loaded by nifty_instruments.load().
#  We pick the nearest-expiry contract (today or later).
# ─────────────────────────────────────────────────────────────────────────────

def _get_nifty_futures_token(
    instruments,
    for_date: date,
) -> Tuple[Optional[int], Optional[str]]:
    """
    Return (instrument_token, tradingsymbol) for the nearest-expiry
    Nifty futures contract on or after for_date.

    Returns (None, None) if not found (e.g. instruments not loaded,
    or no future contracts in NFO list).
    """
    df = getattr(instruments, "_df", None)
    if df is None:
        log.error("[FUT] InstrumentStore._df is None — instruments not loaded")
        return None, None

    # Filter: name="NIFTY" and instrument_type="FUT"
    fut = df[
        (df["name"] == "NIFTY") &
        (df["instrument_type"] == "FUT") &
        (df["expiry"].dt.date >= for_date)
    ].copy()

    if fut.empty:
        log.error(
            f"[FUT] No Nifty FUT contracts found in NFO instruments "
            f"for date >= {for_date}. Check instruments.load() root."
        )
        return None, None

    # Nearest expiry first
    fut = fut.sort_values("expiry")
    row = fut.iloc[0]
    token = int(row["instrument_token"])
    sym   = str(row["tradingsymbol"])
    expiry = row["expiry"].date() if hasattr(row["expiry"], "date") else row["expiry"]
    log.info(f"[FUT] Resolved Nifty futures: {sym} (token={token} expiry={expiry})")
    return token, sym


# ─────────────────────────────────────────────────────────────────────────────
#  STRATEGY CLASS
# ─────────────────────────────────────────────────────────────────────────────

class NiftyFutDirectionalStrategy(BaseStrategy):
    """
    Nifty 50 Futures intraday directional strategy.
    Trades LONG only (short support planned, not yet active).
    Mode A: trend-following with EMA + VWAP + RSI + pullback filters.
    Mode B: momentum/gap-and-go consolidation breakout.
    All trade management in index points — no IV/theta/premium concerns.
    """

    INDEX_TOKEN = 256265    # Nifty 50 — routes Nifty ticks here, not BankNifty
    LIVE_MODE   = LIVE_MODE

    @property
    def name(self) -> str:
        return "NIFTY_FUT_DIRECTIONAL"

    # ── Init ──────────────────────────────────────────────────────────────────

    def __init__(self, market_hub):
        super().__init__(market_hub)

        # ── Internal 5-min candle builder ─────────────────────────────────────
        # Nifty INDEX_TOKEN strategies build their own candles from on_tick().
        # MarketHub routes raw index ticks here; candle building is internal.
        self._candle_builder = CandleBuilder(minutes=5)

        # ── Candle + indicator buffer ─────────────────────────────────────────
        self._buf: deque = deque(maxlen=CFG["buf_size"])

        # ── Internal Nifty session VWAP ───────────────────────────────────────
        # MarketHub's session_vwap tracks BankNifty — useless for Nifty.
        # We maintain our own, fed on every on_tick() call.
        self._vwap = SessionVWAP()

        # ── Pre-market state ──────────────────────────────────────────────────
        self._pm              = None        # live reference (for live PCR reads)
        self._vix             = None
        self._prev_close      = None
        self._instruments     = None
        self._fut_token       = None        # Nifty futures instrument_token
        self._fut_sym         = None        # tradingsymbol e.g. NIFTY25JUNFUT

        # ── Mode detection ────────────────────────────────────────────────────
        self._mode            : Optional[str]   = None    # "A" or "B"
        self._mode_finalised  : bool            = False
        self._open_price      : Optional[float] = None    # true 9:15 open
        self._or_high         : Optional[float] = None    # OR range tracking
        self._or_low          : Optional[float] = None
        self._spike_direction : Optional[str]   = None    # "LONG" or "SHORT"

        # ── Mode B consolidation state machine ────────────────────────────────
        self._mb_state        : str             = "WATCHING"
        self._mb_consol_high  : Optional[float] = None
        self._mb_consol_low   : Optional[float] = None
        self._mb_consol_bars  : int             = 0

        # ── Trade state ───────────────────────────────────────────────────────
        self._trade           = None
        self._today_pnl       : float           = 0.0
        self._trades_taken    : int             = 0
        self._day_paused      : bool            = False
        self._completed       : list            = []

        self._lock = threading.Lock()

        mode_tag = "[LIVE]" if LIVE_MODE else "[PAPER]"
        log.info(
            f"[{self.name}] Initialized {mode_tag} | "
            f"qty={CFG['quantity']} SL={CFG['sl_pts']}pts "
            f"Trail1@+{CFG['trail1_trigger_pts']}pts "
            f"Trail2@+{CFG['trail2_start_pts']}pts "
            f"(lock {CFG['trail2_lock_pts']}pts) "
            f"Target={CFG['tgt_pts']}pts"
        )

    # ── Pre-market ────────────────────────────────────────────────────────────

    def pre_market(self, pm, instruments) -> bool:
        """
        Called from t.py nifty branch (INDEX_TOKEN=256265).
        Resolves Nifty futures token from the same nifty_instruments store.
        Stores live pm reference — never snapshot — so PCR reads are always fresh.
        """
        self._pm          = pm            # live reference — NOT a snapshot
        self._instruments = instruments
        self._vix         = pm.vix
        self._prev_close  = pm.prev_close

        log.info(
            f"[{self.name}] Pre-market | "
            f"VIX={self._vix} PCR={pm.pcr} "
            f"prev_close={self._prev_close} "
            f"expiry={pm.expiry_date} DTE={pm.dte_days}"
        )

        # Resolve Nifty futures token
        today = _now_ist().date()
        self._fut_token, self._fut_sym = _get_nifty_futures_token(instruments, today)

        if self._fut_token is None:
            log.error(
                f"[{self.name}] Could not resolve Nifty futures token — "
                f"strategy will not trade today."
            )
            return False

        # Subscribe futures token to WebSocket so on_option_tick() receives
        # futures price ticks for live SL/trail management.
        # (MarketHub treats all non-index tokens uniformly via subscribe().)
        self.subscribe_option(self._fut_token)
        log.info(
            f"[{self.name}] Subscribed futures: "
            f"{self._fut_sym} (token={self._fut_token})"
        )

        return True

    # ── on_tick  (every Nifty index tick — used for signals + VWAP) ──────────

    def on_tick(self, price: float, ts: datetime, tick_ts: datetime):
        t = ts.time()

        if t < CFG["market_open"] or t >= CFG["hard_exit_time"]:
            return

        # Update internal Nifty VWAP (index has no volume, use proxy_weight=1)
        self._vwap.update(price, price, price, volume=0, proxy_weight=1)

        # Capture true open at first 9:15 tick
        if self._open_price is None and t >= CFG["market_open"]:
            self._open_price = price
            log.info(f"[{self.name}] Open price: {price:.2f}")

        # Track OR range 9:15–9:30
        if CFG["market_open"] <= t < dtime(9, 30):
            if self._or_high is None or price > self._or_high:
                self._or_high = price
            if self._or_low is None or price < self._or_low:
                self._or_low = price

        # Hard exit at 3:00 PM
        if self._trade and self._trade["state"] == "OPEN" and t >= CFG["hard_exit_time"]:
            current_fut_price = self.get_price(self._fut_token) or price
            self._do_exit(current_fut_price, price, "HARD_EXIT_EOD", ts)
            return

        # Build 5-min candles from index ticks
        closed = self._candle_builder.feed_tick(price, 1, ts)
        if closed is not None:
            self._process_candle(closed, ts)

    # ── on_candle  (backfill replay only) ────────────────────────────────────

    def on_candle(self, candle: dict, ts: datetime):
        """
        Receives historical candles during hub.backfill(index_token=256265).
        Warms up the indicator buffer before live trading begins.
        In live trading, on_tick() builds candles internally.
        """
        self._process_candle(candle, ts)

    # ── on_option_tick  (futures price ticks — SL / trail management) ────────

    def on_option_tick(self, token: int, price: float, ts: datetime, tick_ts: datetime = None):
        """
        Receives live futures price ticks for trade management.
        MarketHub delivers all subscribed non-index token ticks here.
        We subscribed self._fut_token in pre_market() — futures ticks arrive here.
        SL check and trailing happen on every futures tick.
        """
        if not self._trade:
            return
        if token != self._fut_token:
            return
        if self._trade["state"] != "OPEN":
            return
        if price <= 0:
            return

        # SL grace period — skip check for N seconds after entry
        if ts < self._trade.get("sl_active_from", ts):
            return

        entry     = self._trade["entry"]
        sl        = self._trade["sl"]
        direction = self._trade["direction"]
        peak_pnl  = self._trade.get("peak_pnl", 0.0)

        # Current profit in index points
        if direction == "LONG":
            current_pnl_pts = price - entry
        else:
            current_pnl_pts = entry - price

        # Track peak profit
        if current_pnl_pts > peak_pnl:
            self._trade["peak_pnl"] = current_pnl_pts
            peak_pnl = current_pnl_pts

        # Trail 1: move SL to breakeven when profit ≥ TRAIL1_TRIGGER_PTS
        if (not self._trade.get("trail1_active") and
                current_pnl_pts >= CFG["trail1_trigger_pts"]):
            new_sl = entry  # breakeven
            if direction == "LONG" and new_sl > sl:
                self._trade["sl"] = new_sl
                self._trade["trail1_active"] = True
                sl = new_sl
                log.info(
                    f"[{self.name}] Trail-1 → BE: SL={new_sl:.2f} "
                    f"(profit={current_pnl_pts:.1f}pts)"
                )
            elif direction == "SHORT" and new_sl < sl:
                self._trade["sl"] = new_sl
                self._trade["trail1_active"] = True
                sl = new_sl
                log.info(
                    f"[{self.name}] Trail-1 → BE: SL={new_sl:.2f} "
                    f"(profit={current_pnl_pts:.1f}pts)"
                )

        # Trail 2: lock profit once peak_pnl ≥ TRAIL2_START_PTS
        if (self._trade.get("trail1_active") and
                peak_pnl >= CFG["trail2_start_pts"]):
            locked = peak_pnl - CFG["trail2_lock_pts"]
            if direction == "LONG":
                trail2_sl = entry + locked
                if trail2_sl > sl:
                    self._trade["sl"] = trail2_sl
                    sl = trail2_sl
                    log.debug(
                        f"[{self.name}] Trail-2 LONG: SL={trail2_sl:.2f} "
                        f"(peak={peak_pnl:.1f}pts lock={locked:.1f}pts)"
                    )
            else:
                trail2_sl = entry - locked
                if trail2_sl < sl:
                    self._trade["sl"] = trail2_sl
                    sl = trail2_sl
                    log.debug(
                        f"[{self.name}] Trail-2 SHORT: SL={trail2_sl:.2f} "
                        f"(peak={peak_pnl:.1f}pts lock={locked:.1f}pts)"
                    )

        # SL check
        if direction == "LONG" and price <= sl:
            self._do_exit(price, price, "SL_HIT", ts)
            return
        if direction == "SHORT" and price >= sl:
            self._do_exit(price, price, "SL_HIT", ts)
            return

        # Soft target (optional half-exit — currently full exit at target)
        if current_pnl_pts >= CFG["tgt_pts"] and not self._trade.get("target_hit"):
            self._trade["target_hit"] = True
            log.info(
                f"[{self.name}] Target hit: {current_pnl_pts:.1f}pts ≥ "
                f"{CFG['tgt_pts']}pts | trailing from here"
            )
            # Trail continues — no exit forced unless half_exit_at_target is set
            if CFG["half_exit_at_target"]:
                log.info(f"[{self.name}] half_exit_at_target=True — implement partial close here")

    # ── Core candle processor ─────────────────────────────────────────────────

    def _process_candle(self, candle: dict, ts: datetime):
        """
        Called for every 5-min candle (backfill or live).
        1. Append to buffer.
        2. Finalise mode once after 9:20.
        3. Run mode-specific entry check.
        """
        self._buf.append(candle)

        t = ts.time()

        # Finalise mode on first candle at/after 9:20
        if not self._mode_finalised and t >= dtime(9, 20):
            self._finalise_mode(ts)

        # Skip entry logic for stale historical candles (backfill)
        if (_now_ist() - ts).total_seconds() > 600:
            return

        if self._day_paused:
            return
        if self._trades_taken >= CFG["max_trades_per_day"]:
            return
        if self._trade and self._trade["state"] == "OPEN":
            return
        if t >= CFG["entry_cutoff"]:
            return
        if not self._mode:
            return

        if self._mode == "A":
            self._check_mode_a(candle, ts)
        elif self._mode == "B":
            self._check_mode_b(candle, ts)

    # ── Mode detection ────────────────────────────────────────────────────────

    def _finalise_mode(self, ts: datetime):
        prev   = self._prev_close
        open_p = self._open_price

        gap_pct  = abs(open_p - prev) / prev if (prev and open_p) else 0.0
        or_range = (
            (self._or_high - self._or_low)
            if (self._or_high is not None and self._or_low is not None)
            else 0.0
        )

        today     = _now_ist().date()
        is_expiry = (
            self._pm is not None and
            self._pm.expiry_date is not None and
            self._pm.expiry_date == today
        )

        mode_b = (
            gap_pct > CFG["mode_b_gap_pct"] or
            or_range > CFG["mode_b_range_trigger"] or
            (is_expiry and gap_pct > CFG["mode_b_gap_pct_expiry"])
        )

        self._mode = "B" if mode_b else "A"
        self._mode_finalised = True

        if mode_b and prev and open_p:
            self._spike_direction = "LONG" if open_p > prev else "SHORT"

        log.info(
            f"[{self.name}] Mode={self._mode} | "
            f"gap={gap_pct:.2%} OR_range={or_range:.0f}pts "
            f"is_expiry={is_expiry} spike_dir={self._spike_direction}"
        )

    # ── Mode A: normal trending day entry ─────────────────────────────────────

    def _check_mode_a(self, candle: dict, ts: datetime):
        t = ts.time()
        in_window = (
            (CFG["mode_a_start"] <= t < CFG["mode_a_window1_end"]) or
            (CFG["mode_a_window2_start"] <= t < CFG["mode_a_window2_end"])
        )
        if not in_window:
            return

        if len(self._buf) < CFG["ema_slow"] + 5:
            return

        ind = self._compute_indicators()
        if not ind:
            return

        direction = self._mode_a_signal(ind, candle)
        if not direction:
            return

        # PCR gate (live reference, never cached snapshot)
        live_pcr = self._pm.pcr if self._pm else None
        if live_pcr is not None:
            if direction == "LONG" and live_pcr < CFG["pcr_min_long"]:
                log.info(
                    f"[{self.name}] [ModeA] LONG skipped — PCR={live_pcr:.2f} "
                    f"< {CFG['pcr_min_long']}"
                )
                return
            if direction == "SHORT" and live_pcr > CFG["pcr_max_short"]:
                log.info(
                    f"[{self.name}] [ModeA] SHORT skipped — PCR={live_pcr:.2f} "
                    f"> {CFG['pcr_max_short']}"
                )
                return

        log.info(
            f"[{self.name}] [ModeA] Signal: {direction} @ {ts.strftime('%H:%M')} | "
            f"EMA9={ind['ema9']:.0f} EMA20={ind['ema20']:.0f} "
            f"EMA50={ind['ema50']:.0f} RSI={ind['rsi']:.1f} "
            f"VWAP={ind['vwap']:.0f if ind['vwap'] else 'N/A'} "
            f"close={candle['close']:.2f}"
        )

        self._enter(direction, candle["close"], ts, reason="mode_a_pullback")

    def _mode_a_signal(self, ind: dict, candle: dict) -> Optional[str]:
        """Returns 'LONG', 'SHORT', or None. All 4 filters must pass."""
        e9, e20, e50 = ind["ema9"], ind["ema20"], ind["ema50"]
        rsi   = ind["rsi"]
        vwap  = ind["vwap"]
        close = candle["close"]

        # Previous 2 candles for pullback check
        prev_two = list(self._buf)[-3:-1] if len(self._buf) >= 3 else []

        # ── LONG ─────────────────────────────────────────────────────────────
        if (
            e9 > e20 > e50 and
            CFG["rsi_long_min"] <= rsi <= CFG["rsi_long_max"] and
            (vwap is None or close > vwap) and
            any(c["low"] <= e9 for c in prev_two) and
            close > e9
        ):
            return "LONG"

        # ── SHORT ────────────────────────────────────────────────────────────
        if (
            e9 < e20 < e50 and
            CFG["rsi_short_min"] <= rsi <= CFG["rsi_short_max"] and
            (vwap is None or close < vwap) and
            any(c["high"] >= e9 for c in prev_two) and
            close < e9
        ):
            return "SHORT"

        return None

    # ── Mode B: momentum / gap-and-go consolidation breakout ─────────────────

    def _check_mode_b(self, candle: dict, ts: datetime):
        t = ts.time()
        if not (CFG["mode_b_start"] <= t < CFG["mode_b_cutoff"]):
            return

        direction = self._spike_direction
        if direction is None:
            return

        bar_range = candle["high"] - candle["low"]

        # ── WATCHING: wait for a tight consolidation bar ──────────────────────
        if self._mb_state == "WATCHING":
            if bar_range < CFG["mode_b_consol_range"]:
                self._mb_state       = "CONSOL"
                self._mb_consol_high = candle["high"]
                self._mb_consol_low  = candle["low"]
                self._mb_consol_bars = 1
                log.info(
                    f"[{self.name}] [ModeB] Consolidation start @ "
                    f"{ts.strftime('%H:%M')} | "
                    f"range={bar_range:.0f}pts "
                    f"H={candle['high']:.0f} L={candle['low']:.0f}"
                )
            return

        # ── CONSOLIDATING ─────────────────────────────────────────────────────
        if self._mb_state == "CONSOL":
            if bar_range < CFG["mode_b_consol_range"]:
                # Expand zone
                self._mb_consol_high = max(self._mb_consol_high, candle["high"])
                self._mb_consol_low  = min(self._mb_consol_low,  candle["low"])
                self._mb_consol_bars += 1
                log.info(
                    f"[{self.name}] [ModeB] Consol bar {self._mb_consol_bars} | "
                    f"zone=[{self._mb_consol_low:.0f}–{self._mb_consol_high:.0f}] "
                    f"range={self._mb_consol_high - self._mb_consol_low:.0f}pts"
                )

                if self._mb_consol_bars >= CFG["mode_b_consol_min_bars"]:
                    if direction == "LONG" and candle["close"] > self._mb_consol_high:
                        log.info(
                            f"[{self.name}] [ModeB] Breakout LONG @ "
                            f"{ts.strftime('%H:%M')} | "
                            f"close={candle['close']:.0f} > "
                            f"zone_H={self._mb_consol_high:.0f}"
                        )
                        self._mb_state = "WATCHING"
                        self._enter(direction, candle["close"], ts,
                                    reason="mode_b_consol_breakout_long")
                        return

                    if direction == "SHORT" and candle["close"] < self._mb_consol_low:
                        log.info(
                            f"[{self.name}] [ModeB] Breakout SHORT @ "
                            f"{ts.strftime('%H:%M')} | "
                            f"close={candle['close']:.0f} < "
                            f"zone_L={self._mb_consol_low:.0f}"
                        )
                        self._mb_state = "WATCHING"
                        self._enter(direction, candle["close"], ts,
                                    reason="mode_b_consol_breakout_short")
                        return

            else:
                # Wide bar — check for breakout first before resetting
                if self._mb_consol_bars >= CFG["mode_b_consol_min_bars"]:
                    if direction == "LONG" and candle["close"] > self._mb_consol_high:
                        log.info(
                            f"[{self.name}] [ModeB] Breakout LONG (wide bar) @ "
                            f"{ts.strftime('%H:%M')}"
                        )
                        self._mb_state = "WATCHING"
                        self._enter(direction, candle["close"], ts,
                                    reason="mode_b_wide_bar_breakout_long")
                        return
                    if direction == "SHORT" and candle["close"] < self._mb_consol_low:
                        log.info(
                            f"[{self.name}] [ModeB] Breakout SHORT (wide bar) @ "
                            f"{ts.strftime('%H:%M')}"
                        )
                        self._mb_state = "WATCHING"
                        self._enter(direction, candle["close"], ts,
                                    reason="mode_b_wide_bar_breakout_short")
                        return

                # Consolidation expired if too many bars
                if self._mb_consol_bars >= CFG["mode_b_consol_max_bars"]:
                    log.info(
                        f"[{self.name}] [ModeB] Consolidation expired "
                        f"({self._mb_consol_bars} bars) — reset"
                    )
                self._mb_state = "WATCHING"

    # ── Entry ─────────────────────────────────────────────────────────────────

    def _enter(self, direction: str, index_price: float, ts: datetime, reason: str):
        """
        Place a futures BUY (LONG) order.
        SHORT direction is logged but not executed — futures short MIS support
        is planned for next version (requires margin and order-type verification).
        """
        with self._lock:
            if self._trade and self._trade["state"] == "OPEN":
                return
            if self._trades_taken >= CFG["max_trades_per_day"]:
                return
            if self._day_paused:
                return

        # SHORT: log and skip (not yet implemented)
        if direction == "SHORT":
            log.info(
                f"[{self.name}] SHORT signal at {ts.strftime('%H:%M')} "
                f"index={index_price:.2f} — SHORT EXECUTION NOT YET ACTIVE "
                f"(signal logged, no order placed)"
            )
            _csv_append({
                "timestamp"    : ts.strftime("%Y-%m-%d %H:%M:%S"),
                "direction"    : "SHORT",
                "action"       : "SIGNAL_SKIPPED",
                "futures_sym"  : self._fut_sym or "",
                "futures_token": self._fut_token or "",
                "index_price"  : round(index_price, 2),
                "entry_price"  : "",
                "exit_price"   : "",
                "sl"           : "",
                "pnl"          : "",
                "reason"       : reason + "_short_not_implemented",
                "mode"         : self._mode,
                "vix"          : self._vix,
                "pcr"          : self._pm.pcr if self._pm else "",
                "order_id"     : "",
            })
            return

        # Get current futures LTP
        fut_ltp = self.get_price(self._fut_token)
        if not fut_ltp or fut_ltp <= 0:
            log.warning(
                f"[{self.name}] Futures LTP unavailable for {self._fut_sym} "
                f"— skipping entry (no stale-price risk)"
            )
            return

        # Acquire slot
        if not self._acquire_slot():
            log.warning(f"[{self.name}] Trade slot blocked")
            return

        # Place BUY
        result = self._place_buy(self._fut_sym, self._fut_token, CFG["quantity"], fut_ltp)
        if result is None:
            self._release_slot()
            log.error(f"[{self.name}] BUY FAILED for {self._fut_sym}")
            return

        order_id, fill_price = result

        sl             = fill_price - CFG["sl_pts"]   # LONG SL below entry
        sl_active_from = ts + timedelta(seconds=CFG["sl_grace_seconds"])
        mode_tag       = "LIVE" if LIVE_MODE else "PAPER"

        with self._lock:
            self._trade = {
                "state"          : "OPEN",
                "direction"      : "LONG",
                "entry"          : fill_price,
                "sl"             : sl,
                "peak_pnl"       : 0.0,
                "trail1_active"  : False,
                "target_hit"     : False,
                "entry_time"     : ts,
                "sl_active_from" : sl_active_from,
                "reason"         : reason,
                "mode"           : self._mode,
                "index_price"    : index_price,
                "order_id"       : order_id,
                "qty"            : CFG["quantity"],
            }
            self._trades_taken += 1

        log.info(
            f"[{self.name}] [{mode_tag}] LONG ENTRY #{self._trades_taken} | "
            f"{self._fut_sym} @ {fill_price:.2f} | "
            f"SL={sl:.2f} (−{CFG['sl_pts']}pts) | "
            f"Trail1@+{CFG['trail1_trigger_pts']}pts "
            f"Trail2@+{CFG['trail2_start_pts']}pts | "
            f"mode={self._mode} reason={reason} | "
            f"order_id={order_id}"
        )

        _csv_append({
            "timestamp"    : ts.strftime("%Y-%m-%d %H:%M:%S"),
            "direction"    : "LONG",
            "action"       : "ENTRY",
            "futures_sym"  : self._fut_sym,
            "futures_token": self._fut_token,
            "index_price"  : round(index_price, 2),
            "entry_price"  : round(fill_price, 2),
            "exit_price"   : "",
            "sl"           : round(sl, 2),
            "pnl"          : "",
            "reason"       : reason,
            "mode"         : self._mode,
            "vix"          : self._vix,
            "pcr"          : self._pm.pcr if self._pm else "",
            "order_id"     : order_id,
        })

    # ── Exit ──────────────────────────────────────────────────────────────────

    def _do_exit(self, fut_price: float, index_price: float, reason: str, ts: datetime):
        t = self._trade
        if not t or t["state"] != "OPEN":
            return

        with self._lock:
            t["state"] = "CLOSED"

        result = self._place_sell_with_retry(
            self._fut_sym, self._fut_token, t["qty"], fut_price, max_retries=3
        )

        if result is None:
            if LIVE_MODE:
                log.error(
                    f"[{self.name}] SELL FAILED for {self._fut_sym} — "
                    f"SQUARE OFF MANUALLY IN ZERODHA!"
                )
            sell_price = fut_price
            order_id   = None
        else:
            order_id, sell_price = result

        self._release_slot()

        pnl = (sell_price - t["entry"]) * t["qty"]
        self._today_pnl += pnl
        mode_tag = "LIVE" if LIVE_MODE else "PAPER"

        log.info(
            f"[{self.name}] [{mode_tag}] LONG EXIT #{self._trades_taken} | "
            f"[{reason}] {self._fut_sym} @ {sell_price:.2f} | "
            f"entry={t['entry']:.2f} "
            f"PnL={pnl:+.0f} ({pnl / t['qty']:+.1f}/unit) | "
            f"Today={self._today_pnl:+.0f}"
        )

        _csv_append({
            "timestamp"    : ts.strftime("%Y-%m-%d %H:%M:%S"),
            "direction"    : "LONG",
            "action"       : "EXIT",
            "futures_sym"  : self._fut_sym,
            "futures_token": self._fut_token,
            "index_price"  : round(index_price, 2),
            "entry_price"  : round(t["entry"], 2),
            "exit_price"   : round(sell_price, 2),
            "sl"           : round(t["sl"], 2),
            "pnl"          : round(pnl, 2),
            "reason"       : reason,
            "mode"         : t["mode"],
            "vix"          : self._vix,
            "pcr"          : self._pm.pcr if self._pm else "",
            "order_id"     : order_id,
        })

        self._completed.append({
            **t,
            "exit_price"  : sell_price,
            "exit_reason" : reason,
            "pnl"         : pnl,
        })

    # ── Indicators ────────────────────────────────────────────────────────────

    def _compute_indicators(self) -> Optional[dict]:
        if len(self._buf) < CFG["ema_slow"] + 5:
            return None

        df    = pd.DataFrame(list(self._buf))
        close = pd.to_numeric(df["close"], errors="coerce").fillna(0)

        ema9  = float(close.ewm(span=CFG["ema_fast"], adjust=False).mean().iloc[-1])
        ema20 = float(close.ewm(span=CFG["ema_mid"],  adjust=False).mean().iloc[-1])
        ema50 = float(close.ewm(span=CFG["ema_slow"], adjust=False).mean().iloc[-1])

        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(CFG["rsi_period"], min_periods=1).mean()
        loss  = (-delta.clip(upper=0)).rolling(CFG["rsi_period"], min_periods=1).mean()
        rs    = gain / loss.replace(0, np.nan)
        rsi   = float((100 - 100 / (1 + rs)).fillna(50).iloc[-1])

        return {
            "ema9" : ema9,
            "ema20": ema20,
            "ema50": ema50,
            "rsi"  : rsi,
            "vwap" : self._vwap.value,
        }

    # ── EOD summary ───────────────────────────────────────────────────────────

    def eod_summary(self):
        log.info(f"\n[{self.name}] {'═'*55}")
        log.info(
            f"[{self.name}] END OF DAY | "
            f"{'LIVE' if LIVE_MODE else 'PAPER'}"
        )
        log.info(f"[{self.name}] Market mode   : {self._mode or 'not set'}")
        log.info(f"[{self.name}] Futures sym   : {self._fut_sym} ({self._fut_token})")
        log.info(f"[{self.name}] Trades taken  : {self._trades_taken}")
        log.info(f"[{self.name}] VIX           : {self._vix}")
        live_pcr = self._pm.pcr if self._pm else None
        log.info(f"[{self.name}] PCR (live)    : {live_pcr}")

        for i, t in enumerate(self._completed, 1):
            pnl_per = t["pnl"] / t["qty"]
            log.info(
                f"[{self.name}]   #{i} {t['direction']} {self._fut_sym} "
                f"[{t['exit_reason']}] "
                f"entry={t['entry']:.2f} exit={t['exit_price']:.2f} "
                f"PnL={t['pnl']:+.0f} ({pnl_per:+.1f}/unit)"
            )

        log.info(f"[{self.name}] Today PnL     : {self._today_pnl:+.0f}")
        log.info(f"[{self.name}] {'═'*55}\n")
