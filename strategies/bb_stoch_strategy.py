"""
strategies/bb_stoch_strategy.py
-------------------------------------------------------------
BBStochStrategy -- BankNifty options scalper using:
  * Bollinger Bands  (BB)   -- identifies breakout / volatility expansion
  * Stochastic Oscillator   -- momentum confirmation + overbought/oversold
  * Volume Filter           -- confirms genuine institutional participation

SIGNAL LOGIC (all three must agree before entry):
--------------------------------------------------------------
  CE entry (buy call):
    1. BB Breakout : last close breaks ABOVE upper BB band
       OR BB Bounce : last close crosses back above lower BB band
          from below (mean-reversion mode, Stoch oversold)
    2. Stoch      : %K > %D (positive crossover in last 2 bars)
                    AND %K > 50  (momentum in bull territory)
                    OR  %K < 20  (oversold bounce)
    3. Volume     : current bar volume >= VOL_MULT * rolling avg vol
    4. VWAP       : close above session VWAP (hub provides tick-accurate VWAP)
    5. Session    : only between SESSION_START and ENTRY_CUTOFF

  PE entry (buy put):
    Mirror of the above -- BB break below lower band, Stoch bear cross, etc.

TRADE MANAGEMENT (on every option WebSocket tick):
--------------------------------------------------------------
  * SL    = entry_price - ATR_SL_MULT * atr
  * TP    = entry_price + ATR_TP_MULT * atr
  * Trail = moves SL to breakeven when profit >= TRAIL_ARM pts,
            then trails every TRAIL_STEP pts after that

SHARED RESOURCES (from MarketHub -- no extra API calls):
--------------------------------------------------------------
  * 5-min index candles    -> received via on_candle()
  * Session VWAP           -> hub.session_vwap.value
  * WebSocket / Kite login -> MarketHub owns both
  * NFO instruments        -> InstrumentStore (shared singleton)

PRIVATE (per-strategy, not shared):
--------------------------------------------------------------
  * 5-min candle buffer (deque of dicts -> DataFrame for indicators)
  * RiskManager state (trade count, PnL, cooldowns)
  * PaperEngine (CSV logging: bb_stoch_entry.csv, bb_stoch_exit.csv)
  * Active option token subscription
"""

import csv
import logging
import os
import threading
import time as time_module
from collections import deque
from datetime import datetime, date, time as dtime
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from core.base_strategy import BaseStrategy
from core.instruments import get_atm_strike

log = logging.getLogger("strategy.bb_stoch")

# ============================================================
# CONFIG  (all tunables in one place)
# ============================================================
CFG = {
    # Bollinger Bands
    "bb_period"         : 20,       # rolling window for BB mean/std
    "bb_std"            : 2.0,      # number of std deviations for bands
    "bb_squeeze_pct"    : 0.003,    # band-width / close < this = squeeze (skip)
                                    # Lowered from 0.005 -> 0.003: allows entries on
                                    # slow/compressed days while stoch+band-break still gate quality

    # Stochastic Oscillator
    "stoch_k"           : 14,       # %K lookback period
    "stoch_d"           : 3,        # %D smoothing period (SMA of %K)
    "stoch_signal"      : 3,        # signal line smoothing
    "stoch_ob"          : 75,       # overbought level for PE reversal
    "stoch_os"          : 25,       # oversold level for CE reversal

    # Volume filter
    "vol_avg_period"    : 10,       # bars to compute average volume
    "vol_mult"          : 1.2,      # current vol must be >= mult * avg_vol

    # VWAP
    "vwap_buffer"       : 10.0,     # allow entry within N pts of VWAP line

    # Session windows (HH, MM)
    "session_start"     : (9, 45),  # no entries before this (opening chaos)
    "session_cutoff"    : (14, 30), # no NEW entries after this
    "auto_squareoff"    : (15, 15), # force-close all open positions

    # Trade management
    "atr_period"        : 14,       # ATR lookback for SL/TP scaling
    "atr_sl_mult"       : 0.9,      # SL = entry - atr * mult
    "atr_tp_mult"       : 1.6,      # TP = entry + atr * mult
    "sl_min"            : 5.0,      # hard floor on SL distance (option pts)
    "sl_max"            : 15.0,     # hard ceiling
    "tp_min"            : 8.0,      # hard floor on TP distance
    "tp_max"            : 25.0,     # hard ceiling
    "trail_arm"         : 6.0,      # move SL to BE when profit >= this
    "trail_step"        : 2.0,      # then trail every N pts
    "slippage"          : 1.5,      # simulated fill slippage (per side)
    "exit_cooldown"     : 2.0,      # seconds between exit attempts

    # Risk
    "max_trades_day"    : 4,
    "post_sl_cooldown"  : 120,      # 2 min wait after any SL hit
    "max_daily_loss"    : 6000,     # Rs circuit breaker
    "quantity"          : 35,       # lots

    # Signal persistence (candle bars)
    "persistence"       : 2,        # signal must fire N consecutive 5-min bars

    # Minimum 5-min bars needed before first signal
    # bb_period=20 is the binding constraint. stoch needs k+d+d=20 bars.
    # 20 bars = 100 minutes from first candle. Reduced from 25 to allow
    # trading even after a late bot start (25 bars = 125 min = often past noon).
    "min_bars"          : 20,       # need >= bb_period

    # CSV output
    "entry_csv"         : "logs/bb_stoch_entry.csv",
    "exit_csv"          : "logs/bb_stoch_exit.csv",
    "signal_csv"        : "logs/bb_stoch_signals.csv",
}


# ============================================================
# INDICATOR COMPUTATION
# ============================================================

def _compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Wilder's ATR from a OHLCV DataFrame."""
    if len(df) < period + 1:
        return 0.0
    close = df["close"].astype(float)
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    prev  = close.shift(1)
    tr    = pd.concat([high - low,
                       (high - prev).abs(),
                       (low  - prev).abs()], axis=1).max(axis=1)
    atr   = tr.ewm(alpha=1 / period, adjust=False).mean()
    return float(round(atr.iloc[-1], 2))


def compute_bb(df: pd.DataFrame, period: int, nstd: float) -> dict:
    """
    Bollinger Bands on 'close'.
    Returns: mid, upper, lower, bw_pct (bandwidth as % of mid), squeeze (bool)
    """
    if len(df) < period:
        return {"mid": 0, "upper": 0, "lower": 0, "bw_pct": 0, "squeeze": True}
    close = df["close"].astype(float)
    mid   = close.rolling(period).mean()
    std   = close.rolling(period).std()
    upper = mid + nstd * std
    lower = mid - nstd * std
    m     = float(mid.iloc[-1])
    u     = float(upper.iloc[-1])
    l     = float(lower.iloc[-1])
    bw    = (u - l) / m if m > 0 else 0
    return {
        "mid"     : round(m, 2),
        "upper"   : round(u, 2),
        "lower"   : round(l, 2),
        "bw_pct"  : round(bw, 5),
        "squeeze" : bw < CFG["bb_squeeze_pct"],
        # Previous bar values (for crossover detection)
        "prev_upper": round(float(upper.iloc[-2]), 2) if len(df) >= period + 1 else u,
        "prev_lower": round(float(lower.iloc[-2]), 2) if len(df) >= period + 1 else l,
        "prev_close": round(float(close.iloc[-2]), 2) if len(df) >= 2 else float(close.iloc[-1]),
    }


def compute_stoch(df: pd.DataFrame, k: int, d: int) -> dict:
    """
    Full Stochastic Oscillator.
    %K = (close - lowest_low_k) / (highest_high_k - lowest_low_k) * 100
    %D = SMA(%K, d)
    Returns last and prev values for crossover detection.
    """
    need = k + d + 2
    if len(df) < need:
        return {"k": 50, "d": 50, "prev_k": 50, "prev_d": 50}

    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    close = df["close"].astype(float)

    lowest_low   = low.rolling(k).min()
    highest_high = high.rolling(k).max()
    denom        = highest_high - lowest_low
    denom        = denom.replace(0, np.nan)
    raw_k        = (close - lowest_low) / denom * 100
    # Smooth %K with SMA(d)
    smooth_k     = raw_k.rolling(d).mean()
    # Signal = SMA(smooth_k, d)
    signal_k     = smooth_k.rolling(d).mean()

    return {
        "k"      : round(float(smooth_k.iloc[-1]),  2),
        "d"      : round(float(signal_k.iloc[-1]),  2),
        "prev_k" : round(float(smooth_k.iloc[-2]),  2),
        "prev_d" : round(float(signal_k.iloc[-2]),  2),
    }


def compute_vol_ratio(df: pd.DataFrame, avg_period: int) -> float:
    """
    Ratio of last bar's volume vs rolling N-bar average.
    > 1.0 means above average. > vol_mult = confirmed participation.

    Returns -1.0 (sentinel) when volume data is unavailable so the caller
    can distinguish "data missing" from a genuine low-volume bar and skip
    the volume filter rather than silently blocking every trade.
    """
    if "volume" not in df.columns:
        log.warning("[BB_STOCH] No 'volume' column in 5m candles -- "
                    "volume filter will be bypassed (check MarketHub candle aggregation)")
        return -1.0  # sentinel: data unavailable

    vol = df["volume"].astype(float)

    # All zeros / NaN = another common data gap symptom
    if vol.sum() == 0 or vol.isna().all():
        log.warning("[BB_STOCH] Volume column is all zeros/NaN -- "
                    "volume filter bypassed (check feed)")
        return -1.0  # sentinel

    if len(df) < avg_period + 1:
        return -1.0  # not enough history yet -- bypass rather than block

    avg     = float(vol.iloc[-(avg_period + 1):-1].mean())
    current = float(vol.iloc[-1])
    if avg <= 0:
        return -1.0
    return round(current / avg, 3)


# ============================================================
# SIGNAL BUILDER
# ============================================================

def evaluate_signal(df: pd.DataFrame, vwap: Optional[float]) -> dict:
    """
    Run all three filters. Returns a signal dict:
        action      : "BUY_CE" | "BUY_PE" | "HOLD"
        blocked_by  : reason string if HOLD
        bb, stoch, vol_ratio, atr: indicator snapshots for logging
    """
    empty = {"action": "HOLD", "blocked_by": "no_data",
             "bb": {}, "stoch": {}, "vol_ratio": 0, "atr": 0}

    if df is None or len(df) < CFG["min_bars"]:
        return {**empty, "blocked_by": "insufficient_bars"}

    # ---- Indicators ----
    bb       = compute_bb(df, CFG["bb_period"], CFG["bb_std"])
    stoch    = compute_stoch(df, CFG["stoch_k"], CFG["stoch_d"])
    vol_ratio = compute_vol_ratio(df, CFG["vol_avg_period"])
    atr      = _compute_atr(df, CFG["atr_period"])
    close    = float(df["close"].iloc[-1])

    base = {"bb": bb, "stoch": stoch, "vol_ratio": vol_ratio, "atr": atr,
            "close": close}

    # ---- Filter 1: BB squeeze  skip (no trend) ----
    if bb["squeeze"]:
        return {"action": "HOLD", "blocked_by": "bb_squeeze", **base}

    # ---- Filter 2: Volume ----
    # vol_ratio == -1.0 is the sentinel meaning "no volume data available".
    # In that case we bypass the filter (don't block) and log a warning.
    # When real volume data is flowing, vol_ratio must exceed vol_mult (1.2x avg).
    if vol_ratio == -1.0:
        vol_ok = True   # data unavailable -- bypass, do not block
        log.debug("[BB_STOCH] Volume data unavailable -- vol filter bypassed")
    else:
        vol_ok = vol_ratio >= CFG["vol_mult"]

    if not vol_ok:
        return {"action": "HOLD", "blocked_by": "volume_low", **base}

    # ---- Stochastic crossovers ----
    k_cross_up   = stoch["prev_k"] < stoch["prev_d"] and stoch["k"] >= stoch["d"]
    k_cross_down = stoch["prev_k"] > stoch["prev_d"] and stoch["k"] <= stoch["d"]

    # ---- VWAP bias ----
    if vwap and vwap > 0:
        above_vwap = close >= (vwap - CFG["vwap_buffer"])
        below_vwap = close <= (vwap + CFG["vwap_buffer"])
    else:
        above_vwap = True  # data unavailable -- don't block
        below_vwap = True

    # ================================================================
    # CE CONDITIONS  (breakout above upper band OR oversold bounce)
    # ================================================================
    bb_breakout_up = close > bb["upper"] and bb["prev_close"] <= bb["prev_upper"]
    bb_bounce_up   = (close > bb["lower"] and bb["prev_close"] <= bb["prev_lower"]
                      and stoch["k"] < CFG["stoch_os"])

    stoch_bull     = (stoch["k"] > 50 or stoch["k"] < CFG["stoch_os"]) and (k_cross_up or stoch["k"] > stoch["d"])

    if (bb_breakout_up or bb_bounce_up) and stoch_bull and above_vwap:
        return {"action": "BUY_CE", "blocked_by": "", **base,
                "mode": "breakout" if bb_breakout_up else "bounce"}

    # ================================================================
    # PE CONDITIONS  (breakdown below lower band OR overbought reversal)
    # ================================================================
    bb_breakout_dn = close < bb["lower"] and bb["prev_close"] >= bb["prev_lower"]
    bb_bounce_dn   = (close < bb["upper"] and bb["prev_close"] >= bb["prev_upper"]
                      and stoch["k"] > CFG["stoch_ob"])

    stoch_bear     = (stoch["k"] < 50 or stoch["k"] > CFG["stoch_ob"]) and (k_cross_down or stoch["k"] < stoch["d"])

    if (bb_breakout_dn or bb_bounce_dn) and stoch_bear and below_vwap:
        return {"action": "BUY_PE", "blocked_by": "", **base,
                "mode": "breakout" if bb_breakout_dn else "bounce"}

    # ---- Granular block reasons for analysis ----
    if bb_breakout_up or bb_bounce_up:
        if not stoch_bull:
            return {"action": "HOLD", "blocked_by": "stoch_not_bull", **base}
        if not above_vwap:
            return {"action": "HOLD", "blocked_by": "below_vwap", **base}
    if bb_breakout_dn or bb_bounce_dn:
        if not stoch_bear:
            return {"action": "HOLD", "blocked_by": "stoch_not_bear", **base}
        if not below_vwap:
            return {"action": "HOLD", "blocked_by": "above_vwap", **base}

    return {"action": "HOLD", "blocked_by": "no_setup", **base}


# ============================================================
# PAPER TRADE STATE
# ============================================================

class BBTrade:
    """Minimal trade record (no external dependency on scalper_v7_core)."""
    __slots__ = (
        "symbol", "token", "option_type", "qty",
        "entry", "sl", "target", "sl_pts", "tp_pts",
        "trail_stage", "spot", "atr", "timestamp",
        "exit_pending", "last_exit_ts",
    )

    def __init__(self, symbol, token, opt_type, ltp, qty,
                 spot, sl_price, tp_price, sl_pts, tp_pts, atr):
        slip          = CFG["slippage"]
        self.symbol   = symbol
        self.token    = token
        self.option_type = opt_type
        self.qty      = qty
        self.entry    = round(ltp + slip, 2)        # simulated fill
        # Anchor SL/TP to actual fill (not pre-slippage LTP)
        off           = self.entry - ltp
        self.sl       = round(sl_price - off, 2)
        self.target   = round(tp_price + off, 2)
        self.sl_pts   = sl_pts
        self.tp_pts   = tp_pts
        self.trail_stage = 0
        self.spot     = spot
        self.atr      = atr
        self.timestamp = datetime.now().isoformat()
        self.exit_pending  = False
        self.last_exit_ts  = 0.0


# ============================================================
# CSV HELPERS
# ============================================================

def _csv_append(filepath: str, row: dict):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    exists = os.path.isfile(filepath)
    with open(filepath, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


# ============================================================
# STRATEGY
# ============================================================

class BBStochStrategy(BaseStrategy):
    """
    BankNifty options strategy using Bollinger Bands + Stochastic + Volume.
    Plugs into the shared MarketHub framework -- zero extra API calls.

    Data consumed:
      on_candle()       -- 5-min index candles (broadcast by MarketHub)
      on_option_tick()  -- live option LTP for SL/TP management
      hub.session_vwap  -- tick-accurate intraday VWAP
    """

    @property
    def name(self) -> str:
        return "BB_STOCH"

    def __init__(self, market_hub):
        super().__init__(market_hub)

        # Private 5-min candle buffer (fed from MarketHub's on_candle broadcast)
        # Size: bb_period + stoch_k + stoch_d + headroom
        buf_size = CFG["bb_period"] + CFG["stoch_k"] + CFG["stoch_d"] + 20
        self._buf_5m: deque = deque(maxlen=buf_size)

        # Trade state
        self._trade: Optional[BBTrade] = None
        self._active_token: Optional[int] = None
        self._lock = threading.Lock()

        # Risk state
        self._daily_pnl   : float = 0.0
        self._trades_today: int   = 0
        self._last_sl_time: float = 0.0
        self._is_halted   : bool  = False
        self._results     : list  = []
        self._blocked_log : dict  = {}

        # Signal persistence (must fire on N consecutive 5-min bars)
        self._persist_buf: deque = deque(maxlen=CFG["persistence"])

        # Pre-market data
        self._instruments   = None
        self._expiry_date   = None
        self._dte           = 0
        self._session_date  = None

        log.info("[BB_STOCH] Strategy initialized")

    # ----------------------------------------------------------
    # PRE-MARKET
    # ----------------------------------------------------------

    def pre_market(self, premarket_data, instruments) -> bool:
        self._instruments  = instruments
        self._expiry_date  = premarket_data.expiry_date
        self._dte          = premarket_data.dte_days
        self._session_date = date.today()
        self._reset_day()
        log.info(
            f"[BB_STOCH] Pre-market | VIX={premarket_data.vix} "
            f"PCR={premarket_data.pcr} Expiry={self._expiry_date} DTE={self._dte}"
        )
        if premarket_data.pcr is None:
            log.warning("[BB_STOCH] PCR is None -- VWAP bias will be unrestricted. "
                        "Check if PCR fetch failed in core.premarket")
        return True

    def _reset_day(self):
        self._daily_pnl    = 0.0
        self._trades_today = 0
        self._last_sl_time = 0.0
        self._is_halted    = False
        self._results.clear()
        self._blocked_log.clear()
        self._persist_buf.clear()

    # ----------------------------------------------------------
    # TICK CALLBACKS
    # ----------------------------------------------------------

    def on_tick(self, price: float, ts: datetime, tick_ts: datetime):
        """Index tick -- not used directly (we work on 5-min candles)."""
        # Auto square-off check
        if ts.time() >= dtime(*CFG["auto_squareoff"]):
            self._force_close("AUTO-SQUAREOFF")

    def on_candle(self, candle: dict, ts: datetime):
        """
        Called by MarketHub when a 5-min index candle closes.
        This is the main signal evaluation trigger.
        """
        with self._lock:
            self._buf_5m.append(candle)

        # Don't look for entries if a trade is already open
        if self._trade:
            return

        self._evaluate_entry(ts)

    def on_option_tick(self, token: int, price: float, ts: datetime,
                       tick_ts: datetime = None):
        """Live option price -- drives SL/TP/trail management."""
        if token != self._active_token:
            return
        if not self._trade:
            return
        self._manage_trade(price, ts)

    # ----------------------------------------------------------
    # SIGNAL EVALUATION (on every 5-min candle close)
    # ----------------------------------------------------------

    def _evaluate_entry(self, ts: datetime):
        now_time = ts.time()

        # Session gate
        if now_time < dtime(*CFG["session_start"]):
            return
        if now_time >= dtime(*CFG["session_cutoff"]):
            return

        # Risk gate
        if self._is_halted:
            return
        if self._trades_today >= CFG["max_trades_day"]:
            return
        elapsed_since_sl = time_module.time() - self._last_sl_time
        if self._last_sl_time > 0 and elapsed_since_sl < CFG["post_sl_cooldown"]:
            remaining = CFG["post_sl_cooldown"] - elapsed_since_sl
            log.info(f"[BB_STOCH] Post-SL cooldown: {remaining:.0f}s remaining")
            return

        with self._lock:
            candles = list(self._buf_5m)

        df = self._to_df(candles)
        if df.empty:
            return

        n_bars = len(df)
        if n_bars < CFG["min_bars"]:
            log.debug(f"[BB_STOCH] Warming up: {n_bars}/{CFG['min_bars']} bars "
                      f"(ready in ~{(CFG['min_bars']-n_bars)*5} mins)")
            return

        vwap = self._hub.session_vwap.value

        # Run signal logic
        sig    = evaluate_signal(df, vwap)
        action = sig["action"]

        # Log signal bar
        self._log_signal(ts, sig)

        # Persistence check
        self._persist_buf.append(action)
        confirmed = (
            len(self._persist_buf) >= CFG["persistence"]
            and all(a == action and a != "HOLD" for a in self._persist_buf)
        )

        bb    = sig.get("bb", {})
        stoch = sig.get("stoch", {})
        vwap_str = f"{vwap:.2f}" if vwap is not None else "N/A"
        vol_str = f"{vol_ratio:.2f}x" if vol_ratio >= 0 else "N/A(no-data)"
        log.info(
            f"[BB_STOCH] {action:8s} | "
            f"Close={sig.get('close', 0):.2f} | "
            f"BB=[{bb.get('lower', 0):.1f}~{bb.get('upper', 0):.1f}] "
            f"BW={bb.get('bw_pct', 0)*100:.2f}% | "
            f"Stoch K={stoch.get('k', 0):.1f} D={stoch.get('d', 0):.1f} | "
            f"Vol={vol_str} | "
            f"VWAP={vwap_str} | "
            f"Block={sig.get('blocked_by') or 'none'} | Persist={confirmed} | "
            f"PnL={self._daily_pnl:+.1f}Rs T={self._trades_today}/{CFG['max_trades_day']}"
        )

        if not confirmed or action == "HOLD":
            if action == "HOLD":
                bl = sig.get("blocked_by", "")
                self._blocked_log[bl] = self._blocked_log.get(bl, 0) + 1
            return

        self._persist_buf.clear()
        self._enter_trade(action, sig, ts)

    # ----------------------------------------------------------
    # ENTRY
    # ----------------------------------------------------------

    def _enter_trade(self, action: str, sig: dict, ts: datetime):
        opt    = "CE" if action == "BUY_CE" else "PE"
        spot   = sig.get("close", 0)
        atr    = sig.get("atr", 0)
        expiry = self._expiry_date

        if expiry is None:
            log.warning("[BB_STOCH] No expiry date -- skipping entry")
            return

        atm = get_atm_strike(spot)

        # On expiry day use 1-strike ITM for better liquidity
        if self._dte == 0:
            atm = atm - 100 if opt == "CE" else atm + 100

        token, tsym = self._instruments.get_option_token(atm, opt, expiry)
        if not token:
            log.warning(f"[BB_STOCH] Option not found: ATM={atm} {opt} expiry={expiry}")
            return

        # Subscribe and wait for first tick
        self.subscribe_option(token)
        time_module.sleep(0.4)
        ltp = self.get_price(token)

        if not ltp or ltp < 5:
            log.warning(f"[BB_STOCH] LTP unavailable or too low ({ltp}) for {tsym}")
            self.unsubscribe_option(token)
            return

        # Compute SL/TP
        sl_pts, tp_pts, sl_price, tp_price = self._compute_sl_tp(ltp, atr)

        # Option sanity: reject if spread eats too much of target
        est_spread = ltp * 0.03
        if tp_pts > 0 and (est_spread / tp_pts) > 0.35:
            log.warning(f"[BB_STOCH] Spread {est_spread:.1f} too large vs TP {tp_pts:.1f} -- skipping")
            self.unsubscribe_option(token)
            return

        trade = BBTrade(
            symbol   = tsym,
            token    = token,
            opt_type = opt,
            ltp      = ltp,
            qty      = CFG["quantity"],
            spot     = spot,
            sl_price = sl_price,
            tp_price = tp_price,
            sl_pts   = sl_pts,
            tp_pts   = tp_pts,
            atr      = atr,
        )

        with self._lock:
            self._trade        = trade
            self._active_token = token

        self._trades_today += 1
        rr = round(tp_pts / sl_pts, 2) if sl_pts else 0

        log.info(
            f"[BB_STOCH] ENTRY {opt} | {tsym} | "
            f"Fill={trade.entry:.2f} LTP={ltp:.2f} | "
            f"SL={trade.sl:.2f}(-{sl_pts:.1f}) TP={trade.target:.2f}(+{tp_pts:.1f}) | "
            f"RR=1:{rr} ATR={atr:.1f} | "
            f"Mode={sig.get('mode', '?')} BB-BW={sig['bb'].get('bw_pct', 0)*100:.2f}%"
        )

        bb = sig.get("bb", {})
        st = sig.get("stoch", {})
        _csv_append(CFG["entry_csv"], {
            "timestamp"  : trade.timestamp,
            "symbol"     : tsym,
            "opt_type"   : opt,
            "qty"        : CFG["quantity"],
            "ltp"        : ltp,
            "fill"       : trade.entry,
            "sl"         : trade.sl,
            "target"     : trade.target,
            "sl_pts"     : sl_pts,
            "tp_pts"     : tp_pts,
            "rr"         : rr,
            "spot"       : spot,
            "atr"        : atr,
            "mode"       : sig.get("mode", ""),
            "bb_upper"   : bb.get("upper", ""),
            "bb_lower"   : bb.get("lower", ""),
            "bb_bw_pct"  : bb.get("bw_pct", ""),
            "stoch_k"    : st.get("k", ""),
            "stoch_d"    : st.get("d", ""),
            "vol_ratio"  : sig.get("vol_ratio", ""),
            "vwap"       : self._hub.session_vwap.value if self._hub.session_vwap.value is not None else "N/A",
        })

    # ----------------------------------------------------------
    # TRADE MANAGEMENT (called on every option WebSocket tick)
    # ----------------------------------------------------------

    def _manage_trade(self, ltp: float, ts: datetime):
        trade = self._trade
        if not trade:
            return

        now_ts = time_module.time()
        if trade.exit_pending and (now_ts - trade.last_exit_ts) < CFG["exit_cooldown"]:
            return

        pnl_pts = ltp - trade.entry
        self._update_trail(trade, ltp, pnl_pts)

        if ltp >= trade.target:
            self._close_trade(ltp, "TARGET")
        elif ltp <= trade.sl:
            self._close_trade(ltp, "SL")

    def _update_trail(self, trade: BBTrade, ltp: float, pnl_pts: float):
        if pnl_pts >= CFG["trail_arm"] and trade.trail_stage == 0:
            new_sl = round(trade.entry, 2)
            if new_sl > trade.sl:
                trade.sl          = new_sl
                trade.trail_stage = 1
                log.info(f"[BB_STOCH] Trail -> BE | SL={trade.sl:.2f}")
        elif pnl_pts > CFG["trail_arm"] and trade.trail_stage >= 1:
            new_stage = int((pnl_pts - CFG["trail_arm"]) / CFG["trail_step"]) + 1
            if new_stage > trade.trail_stage:
                inc               = (new_stage - trade.trail_stage) * CFG["trail_step"]
                trade.sl          = round(trade.sl + inc, 2)
                trade.trail_stage = new_stage
                log.info(f"[BB_STOCH] Trail stage {new_stage} | SL={trade.sl:.2f}")

    def _close_trade(self, ltp: float, reason: str):
        trade = self._trade
        if not trade:
            return

        trade.exit_pending = True
        trade.last_exit_ts = time_module.time()

        slip     = CFG["slippage"]
        exit_px  = round(ltp - slip, 2)
        pnl_pts  = round(exit_px - trade.entry, 2)
        pnl_rs   = round(pnl_pts * trade.qty, 2)
        rr_act   = round(pnl_pts / trade.sl_pts, 2) if trade.sl_pts else 0

        self._daily_pnl += pnl_rs

        if reason == "SL":
            self._last_sl_time = time_module.time()

        tag = "[TARGET]" if reason == "TARGET" else ("[SL]" if reason == "SL" else "[EXIT]")
        log.info(
            f"[BB_STOCH] {tag} {trade.option_type} | {trade.symbol} | {reason} | "
            f"Exit={exit_px:.2f} | PnL={pnl_pts:+.2f}pts ({pnl_rs:+.2f}Rs) | "
            f"RR={rr_act:+.2f} | DayPnL={self._daily_pnl:+.2f}Rs"
        )

        result = {
            "timestamp"  : datetime.now().isoformat(),
            "symbol"     : trade.symbol,
            "opt_type"   : trade.option_type,
            "qty"        : trade.qty,
            "entry"      : trade.entry,
            "exit"       : exit_px,
            "pnl_pts"    : pnl_pts,
            "pnl_rs"     : pnl_rs,
            "rr_actual"  : rr_act,
            "sl_pts"     : trade.sl_pts,
            "tp_pts"     : trade.tp_pts,
            "reason"     : reason,
            "trail_stage": trade.trail_stage,
            "day_pnl_rs" : round(self._daily_pnl, 2),
        }
        _csv_append(CFG["exit_csv"], result)
        self._results.append(result)

        if self._daily_pnl <= -abs(CFG["max_daily_loss"]):
            self._is_halted = True
            log.warning(f"[BB_STOCH] HALTED -- max daily loss breached ({self._daily_pnl:.0f}Rs)")

        with self._lock:
            self._unsubscribe_active()
            self._trade = None

    def _force_close(self, reason: str):
        """Force-close any open trade (EOD / auto square-off)."""
        with self._lock:
            trade = self._trade
        if not trade:
            return
        ltp = self.get_price(trade.token) or trade.entry
        self._close_trade(ltp, reason)

    def _unsubscribe_active(self):
        """Must be called with self._lock held or after trade cleared."""
        if self._active_token:
            self.unsubscribe_option(self._active_token)
            log.info(f"[BB_STOCH] Unsubscribed token {self._active_token}")
            self._active_token = None

    # ----------------------------------------------------------
    # HELPERS
    # ----------------------------------------------------------

    def _compute_sl_tp(self, ltp: float, atr: float
                       ) -> Tuple[float, float, float, float]:
        """Returns (sl_pts, tp_pts, sl_price, tp_price)."""
        sl_pts = float(atr * CFG["atr_sl_mult"]) if atr > 0 else CFG["sl_min"]
        tp_pts = float(atr * CFG["atr_tp_mult"]) if atr > 0 else CFG["tp_min"]
        sl_pts = max(CFG["sl_min"], min(sl_pts, CFG["sl_max"]))
        tp_pts = max(CFG["tp_min"], min(tp_pts, CFG["tp_max"]))
        # On expiry day tighten TP (theta erodes options fast)
        if self._dte == 0:
            tp_pts = max(CFG["tp_min"], tp_pts * 0.75)
        sl_price = round(ltp - sl_pts, 2)
        tp_price = round(ltp + tp_pts, 2)
        log.info(
            f"[BB_STOCH] SL/TP | ATR={atr:.1f} | "
            f"SL={sl_price:.2f}(-{sl_pts:.1f}) TP={tp_price:.2f}(+{tp_pts:.1f})"
        )
        return sl_pts, tp_pts, sl_price, tp_price

    @staticmethod
    def _to_df(candles: list) -> pd.DataFrame:
        if not candles:
            return pd.DataFrame()
        df = pd.DataFrame(candles)
        for col in ("open", "high", "low", "close", "volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.dropna(subset=["close"]).reset_index(drop=True)

    def _log_signal(self, ts: datetime, sig: dict):
        bb = sig.get("bb", {})
        st = sig.get("stoch", {})
        _csv_append(CFG["signal_csv"], {
            "timestamp"  : ts.isoformat(),
            "action"     : sig.get("action", "HOLD"),
            "blocked_by" : sig.get("blocked_by", ""),
            "close"      : sig.get("close", ""),
            "bb_upper"   : bb.get("upper", ""),
            "bb_lower"   : bb.get("lower", ""),
            "bb_bw_pct"  : bb.get("bw_pct", ""),
            "bb_squeeze" : bb.get("squeeze", ""),
            "stoch_k"    : st.get("k", ""),
            "stoch_d"    : st.get("d", ""),
            "stoch_pk"   : st.get("prev_k", ""),
            "stoch_pd"   : st.get("prev_d", ""),
            "vol_ratio"  : sig.get("vol_ratio", ""),
            "atr"        : sig.get("atr", ""),
            "vwap"       : self._hub.session_vwap.value if self._hub.session_vwap.value is not None else "N/A",
        })

    # ----------------------------------------------------------
    # EOD
    # ----------------------------------------------------------

    def eod_summary(self):
        """Called by MarketHub at 3:30 PM."""
        self._force_close("EOD-SQUAREOFF")

        results = self._results
        total   = len(results)
        wins    = [r for r in results if r["pnl_pts"] > 0]
        losses  = [r for r in results if r["pnl_pts"] <= 0]
        win_pct = len(wins) / total if total else 0
        avg_win  = sum(r["pnl_pts"] for r in wins)   / len(wins)   if wins   else 0
        avg_loss = sum(r["pnl_pts"] for r in losses) / len(losses) if losses else 0
        exp      = round(win_pct * avg_win + (1 - win_pct) * avg_loss, 3) if total else 0

        log.info(
            f"[BB_STOCH] EOD | Trades={total} W={len(wins)} L={len(losses)} "
            f"WinRate={win_pct*100:.1f}% Expect={exp:+.3f}pts "
            f"DayPnL={self._daily_pnl:+.2f}Rs | "
            f"5m-bars={len(self._buf_5m)}"
        )
        if self._blocked_log:
            top = sorted(self._blocked_log.items(), key=lambda x: -x[1])[:5]
            log.info(f"[BB_STOCH] Top blocks: {top}")