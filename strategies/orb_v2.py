"""
strategies/orb_v2.py

ORB (Opening Range Breakout) Strategy v2
Entry window: 9:40–10:15 AM
Filters: ADX(2), EMA(2), VWAP(1.5), RSI(1), Volume(1), ST(1), BB(0.5)
Exit: Target 1.5×OR, Trailing SL after 1R, Premium SL, Time stop
Multi-trade: up to 2 per day if first hits SL

FIXES APPLIED:
  Bug 5 — CandleBuilder fed only close → H=L=O=C on every indicator bar.
           ATR, Supertrend, ADX all computed on flat data → wrong values.
           Fix: removed _cb (CandleBuilder). Replaced with _candle_buf (deque)
           that stores full OHLCV candle dicts directly from on_candle().
           Indicators now see real H/L/C as published by MarketHub.

  Bug 6 — One-bar indicator lag. on_candle() fed close into _cb BEFORE
           calling _check_entry(), so the breakout bar itself was never
           in _cb.get_closed(). Indicators were computed on data excluding
           the bar that triggered the signal.
           Fix: because _candle_buf appends the full candle in on_candle()
           before _check_entry() runs, the breakout bar is always included.

  Bug 9 — Gap filter used 9:30 price instead of 9:15 open price.
           _lock_or(price) is called when t >= 9:30; that price could be
           far from the actual open if the market moved during 9:15–9:30.
           Fix: store the first tick at 9:15 as _open_price. Use that as
           the gap reference in _lock_or().
"""

import csv
import logging
import os
import threading
from collections import deque
from datetime import datetime, date, time as dtime, timedelta

import numpy as np
import pandas as pd

from core.base_strategy import BaseStrategy
from core.instruments import get_atm_strike
from core.pricer import option_premium, pick_iv

log = logging.getLogger("strategy.orb")

# ── Strategy-specific config ──────────────────────────────────────────────────
CFG = {
    "lot_size"          : 15,
    "or_start"          : dtime(9, 15),
    "or_end"            : dtime(9, 30),
    "entry_start"       : dtime(9, 40),
    "entry_cutoff"      : dtime(10, 30),  # widened: was 10:15 (missed slow breakouts)
    "time_stop"         : dtime(15, 15),
    "min_range"         : 180,
    "max_range"         : 400,
    "gap_max_pct"       : 0.012,  # widened: was 0.008 (blocked valid 0.8-1.2% gap days)
    "target_mult"       : 1.5,
    "trail_after_r"     : 1.0,
    "trail_sl_mult"     : 0.5,
    "premium_sl_0dte"   : 0.50,
    "premium_sl_1_2dte" : 0.40,
    "premium_sl_3plus"  : 0.35,
    "max_trades_per_day": 2,
    "max_consec_losses" : 3,
    "rsi_period"        : 14,
    "ema_fast"          : 9,
    "ema_slow"          : 21,
    "vol_multiplier"    : 1.5,
    "vol_avg_period"    : 10,
    "st_period"         : 7,
    "st_mult"           : 3.0,
    "adx_period"        : 14,
    "adx_min"           : 16,     # lowered: was 20 (ADX<20 common in first hour)
    "bb_period"         : 20,
    "bb_std"            : 2.0,
    "rsi_bull"          : 55,
    "rsi_bear"          : 45,
    "vix_max"           : 24.0,   # raised: was 20.0 (India VIX routinely 20-24)
    "pcr_max"           : 1.3,
    "pcr_min"           : 0.7,
    "min_score_normal"  : 4.0,    # lowered: was 5.0 (score 5 rarely hit in calm markets)
    "min_score_0dte"    : 6.0,    # lowered: was 7.0
    "slippage_pts"      : 2.0,
    "brokerage"         : 50,
    "csv_file"          : "orb_trades.csv",
    # FIX (Bug 5): max candle history for indicator buffer
    "candle_buf_size"   : 50,
}


class ORBStrategy(BaseStrategy):

    @property
    def name(self) -> str:
        return "ORB_v2"

    def __init__(self, market_hub):
        super().__init__(market_hub)

        # FIX (Bug 5+6): replaced CandleBuilder with a plain deque of full candle dicts.
        # CandleBuilder was fed only close values → H=L=O=C → wrong ATR/Supertrend/ADX.
        # Now we store each full OHLCV candle dict from on_candle() directly.
        self._candle_buf: deque = deque(maxlen=CFG["candle_buf_size"])

        # Per-day state
        self._or_high       = None
        self._or_low        = None
        self._or_locked     = False
        self._or_width      = None
        self._day_paused    = False
        self._trades_taken  = 0
        self._consec_losses = 0
        self._today_pnl     = 0.0
        self._completed     = []     # list of closed trade dicts

        # FIX (Bug 9): store the first tick price at 9:15 for gap calculation.
        # Gap should be measured from the actual open price, not from whatever
        # price happens to be at 9:30 when OR locks.
        self._open_price    = None

        # Pre-market data (set in pre_market())
        self._vix       = None
        self._pcr       = None
        self._prev_close= None
        self._ema200    = None
        self._expiry    = None
        self._dte_days  = None
        self._instruments = None

        # Active trade
        self._trade     = None
        self._lock      = threading.Lock()

    # ── Pre-market ────────────────────────────────────────────────────────────

    def pre_market(self, pm, instruments) -> bool:
        self._instruments = instruments
        self._vix         = pm.vix
        self._pcr         = pm.pcr
        self._prev_close  = pm.prev_close
        self._ema200      = pm.ema200_daily
        self._expiry      = pm.expiry_date
        self._dte_days    = pm.dte_days

        log.info(f"[{self.name}] Pre-market loaded | VIX={self._vix} PCR={self._pcr} "
                 f"EMA200={self._ema200} DTE={self._dte_days}")

        if self._vix and self._vix > CFG["vix_max"]:
            log.warning(f"[{self.name}]  VIX={self._vix:.1f} > {CFG['vix_max']}  skip today")
            self._day_paused = True
            return False

        return True

    # ── Tick handler ──────────────────────────────────────────────────────────

    def on_tick(self, price: float, ts: datetime, tick_ts: datetime):
        t = ts.time()

        # FIX (Bug 9): capture the first tick at market open as the true gap reference.
        # Gap must be computed vs. 9:15 open, not 9:30 price.
        if self._open_price is None and t >= CFG["or_start"]:
            self._open_price = price
            log.info(f"[{self.name}] Open price captured: {price:.2f}")

        # Build opening range 9:15–9:30
        if CFG["or_start"] <= t < CFG["or_end"] and not self._or_locked:
            if self._or_high is None or price > self._or_high: self._or_high = price
            if self._or_low  is None or price < self._or_low:  self._or_low  = price

        # Lock OR at 9:30
        if t >= CFG["or_end"] and not self._or_locked:
            self._lock_or(price, ts)

        # Monitor active trade exit
        if self._trade and self._trade["state"] == "OPEN":
            self._check_exit(price, ts)

        # Time stop
        if self._trade and self._trade["state"] == "OPEN" and t >= CFG["time_stop"]:
            self._close("TIME_STOP", price, ts)

    def on_candle(self, candle: dict, ts: datetime):
        """Called when 5-min index candle closes. Check for entry.

        FIX (Bug 5+6): store the full OHLCV candle dict in _candle_buf.
        Old code fed only close into a CandleBuilder — all H=L=O=C.
        Now indicators receive real high/low for ATR, Supertrend, ADX.
        Appending BEFORE _check_entry() also fixes the one-bar lag (Bug 6):
        the breakout bar itself is now included in indicator computation.
        """
        # FIX: append full candle dict (not just close) — preserves real OHLCV
        self._candle_buf.append(candle)

        if (not self._day_paused and
                self._or_locked and self._or_width and
                CFG["min_range"] <= self._or_width <= CFG["max_range"] and
                (self._trade is None or self._trade["state"] == "CLOSED") and
                self._trades_taken < CFG["max_trades_per_day"] and
                CFG["entry_start"] <= ts.time() <= CFG["entry_cutoff"]):
            self._check_entry(candle, candle["close"], ts)

    def on_option_tick(self, token: int, price: float, ts: datetime, tick_ts: datetime = None):
        """Live option tick for premium SL."""
        if not self._trade or self._trade.get("opt_token") != token:
            return
        if self._trade["state"] != "OPEN":
            return

        self._trade["last_opt_price"] = price
        entry_eff = self._trade["entry_prem_eff"]
        psl_pct   = self._trade["prem_sl_pct"]

        if entry_eff > 0 and price <= entry_eff * (1 - psl_pct):
            spot = self._hub.last_price(260105) or self._trade["entry_spot"]
            log.warning(f"[{self.name}]  PREMIUM SL | opt={price:.2f}")
            self._close("PREMIUM_SL", spot, ts, exit_prem=price)

    # ── OR locking ────────────────────────────────────────────────────────────

    def _lock_or(self, price: float, ts: datetime):
        if not self._or_high or not self._or_low:
            return
        self._or_locked = True
        self._or_width  = self._or_high - self._or_low

        log.info(f"[{self.name}]  OR locked: H={self._or_high:.0f} "
                 f"L={self._or_low:.0f} W={self._or_width:.0f}pts")

        # FIX (Bug 9): use _open_price (9:15 first tick) for gap calculation.
        # Original code used 'price' which is whatever the index is at 9:30 —
        # this could be far from the real open if the first 15 min were volatile.
        if self._prev_close:
            gap_ref = self._open_price if self._open_price is not None else price
            gap_pct = abs(gap_ref - self._prev_close) / self._prev_close
            if gap_pct > CFG["gap_max_pct"]:
                log.info(f"[{self.name}]  Gap={gap_pct:.2%} > {CFG['gap_max_pct']:.1%} "
                         f"(ref={gap_ref:.2f} vs prev_close={self._prev_close:.2f})  skip")
                self._day_paused = True
                return

        if self._or_width < CFG["min_range"]:
            log.info(f"[{self.name}]  OR too narrow: {self._or_width:.0f}pts")
            self._day_paused = True
        elif self._or_width > CFG["max_range"]:
            log.info(f"[{self.name}]  OR too wide: {self._or_width:.0f}pts")
            self._day_paused = True

    # ── Entry ─────────────────────────────────────────────────────────────────

    def _check_entry(self, candle: dict, price: float, ts: datetime):
        if self._consec_losses >= CFG["max_consec_losses"]:
            log.warning(f"[{self.name}]  Circuit breaker ({self._consec_losses} losses)")
            self._day_paused = True
            return

        orh = self._or_high
        orl = self._or_low

        # Breakout detection (HIGH/LOW)
        if   candle["high"] > orh: direction = "CE"
        elif candle["low"]  < orl: direction = "PE"
        else: return

        # Confirmation: close must be beyond OR (reject wick-only)
        if direction == "CE" and candle["close"] <= orh: return
        if direction == "PE" and candle["close"] >= orl: return

        # 200 EMA trend filter
        if self._ema200:
            if direction == "CE" and price < self._ema200:
                log.info(f"[{self.name}]  CE below 200EMA={self._ema200:.0f}")
                return
            if direction == "PE" and price > self._ema200:
                log.info(f"[{self.name}]  PE above 200EMA={self._ema200:.0f}")
                return

        # PCR gate
        if self._pcr:
            if direction == "CE" and self._pcr < CFG["pcr_min"]:
                log.info(f"[{self.name}]  PCR={self._pcr:.2f} < min={CFG['pcr_min']}  skip CE")
                return
            if direction == "PE" and self._pcr > CFG["pcr_max"]:
                log.info(f"[{self.name}]  PCR={self._pcr:.2f} > max={CFG['pcr_max']}  skip PE")
                return
        else:
            log.debug(f"[{self.name}]  PCR unavailable  skipping PCR filter")

        # FIX (Bug 5+6): use _candle_buf (full OHLCV dicts) instead of _cb.get_closed().
        # _candle_buf already has the current candle appended (done in on_candle before here),
        # so the breakout bar itself is included in indicator computation (fixes Bug 6).
        candles = list(self._candle_buf)
        if len(candles) < 5:
            return
        ind = self._compute_indicators(candles)
        if not ind:
            return

        vwap  = self._hub.session_vwap.value
        score, details = self._score(ind, direction, vwap)
        min_s = CFG["min_score_0dte"] if self._dte_days == 0 else CFG["min_score_normal"]

        log.info(f"\n[{self.name}] {'─'*50}")
        log.info(f"[{self.name}]  {direction} breakout {ts.strftime('%H:%M')} "
                 f"score={score}/{min_s}")
        for v in details.values():
            log.info(f"[{self.name}]   {v}")

        if score < min_s:
            log.info(f"[{self.name}]  Score too low  skip\n")
            return

        self._enter(direction, price, ts)

    def _enter(self, direction: str, spot: float, ts: datetime):
        orh      = self._or_high
        orl      = self._or_low
        rng      = self._or_width
        entry_s  = (orh + 0.05) if direction == "CE" else (orl - 0.05)
        strike   = get_atm_strike(entry_s)
        expiry   = self._expiry
        expiry_dt = datetime.combine(expiry, dtime(15, 30))
        dte      = self._dte_days
        iv       = pick_iv(dte, rng)
        psl_pct  = (CFG["premium_sl_0dte"]   if dte == 0 else
                    CFG["premium_sl_1_2dte"]  if dte <= 2 else
                    CFG["premium_sl_3plus"])

        target_s = (entry_s + rng * CFG["target_mult"] if direction == "CE"
                    else entry_s - rng * CFG["target_mult"])
        sl_s     = orl if direction == "CE" else orh

        # Option token
        opt_token, opt_sym = self._instruments.get_option_token(strike, direction, expiry)

        # Entry premium
        lp = self._hub.last_price(opt_token) if opt_token else None
        ep = float(lp) if lp else option_premium(entry_s, strike, ts, expiry_dt, direction, iv)
        ep_eff = ep + CFG["slippage_pts"]

        self._trade = {
            "state"          : "OPEN",
            "direction"      : direction,
            "entry_spot"     : entry_s,
            "entry_time"     : ts,
            "entry_prem"     : round(ep, 2),
            "entry_prem_eff" : round(ep_eff, 2),
            "strike"         : strike,
            "expiry"         : expiry,
            "expiry_dt"      : expiry_dt,
            "dte"            : dte,
            "iv"             : iv,
            "prem_sl_pct"    : psl_pct,
            "target_spot"    : round(target_s, 2),
            "sl_spot"        : round(sl_s, 2),
            "trailing_sl"    : round(sl_s, 2),
            "trail_active"   : False,
            "opt_token"      : opt_token,
            "opt_sym"        : opt_sym,
            "last_opt_price" : ep,
            "orh"            : orh,
            "orl"            : orl,
            "range_width"    : rng,
        }
        self._trades_taken += 1

        if opt_token:
            self.subscribe_option(opt_token)

        log.info(f"[{self.name}]  TRADE #{self._trades_taken} ENTERED")
        log.info(f"[{self.name}]   {direction} {strike} {expiry}  DTE={dte}")
        log.info(f"[{self.name}]   Prem {ep:.0f}+slip{ep_eff:.0f} | "
                 f"Tgt={target_s:.0f} | SL={sl_s:.0f} | PremSL={psl_pct*100:.0f}%")

    # ── Exit ──────────────────────────────────────────────────────────────────

    def _check_exit(self, price: float, ts: datetime):
        t = self._trade
        rng = t["range_width"]

        # Trailing SL: activate after 1R profit
        if not t["trail_active"]:
            profit = (price - t["entry_spot"]) if t["direction"] == "CE" else (t["entry_spot"] - price)
            if profit >= rng * CFG["trail_after_r"]:
                t["trail_active"] = True
                t["trailing_sl"]  = t["entry_spot"]  # move to breakeven
                log.info(f"[{self.name}]  Trail active: SL→BE={t['entry_spot']:.0f}")

        if t["trail_active"]:
            gap  = rng * CFG["trail_sl_mult"]
            new  = price - gap if t["direction"] == "CE" else price + gap
            if t["direction"] == "CE" and new > t["trailing_sl"]:
                t["trailing_sl"] = round(new, 2)
            elif t["direction"] == "PE" and new < t["trailing_sl"]:
                t["trailing_sl"] = round(new, 2)
        eff_sl = t["trailing_sl"] if t["trail_active"] else t["sl_spot"]

        # Target
        if t["direction"] == "CE" and price >= t["target_spot"]:
            self._close("TARGET_HIT", price, ts); return
        if t["direction"] == "PE" and price <= t["target_spot"]:
            self._close("TARGET_HIT", price, ts); return

        # SL / Trailing SL
        if t["direction"] == "CE" and price <= eff_sl:
            self._close("TRAIL_SL" if t["trail_active"] else "SL_HIT", price, ts); return
        if t["direction"] == "PE" and price >= eff_sl:
            self._close("TRAIL_SL" if t["trail_active"] else "SL_HIT", price, ts); return

        # BS premium SL fallback (when no live option price)
        if not t["opt_token"]:
            sim = option_premium(price, t["strike"], ts, t["expiry_dt"], t["direction"], t["iv"])
            if t["entry_prem_eff"] > 0 and sim <= t["entry_prem_eff"] * (1 - t["prem_sl_pct"]):
                self._close("PREMIUM_SL", price, ts, exit_prem=sim)

    def _close(self, reason: str, spot: float, ts: datetime, exit_prem: float = None):
        t = self._trade
        if t["state"] != "OPEN": return
        t["state"] = "CLOSED"

        if exit_prem is None:
            lp = t.get("last_opt_price")
            if lp and lp > 0:
                exit_prem = lp
            else:
                exit_prem = option_premium(spot, t["strike"], ts,
                                           t["expiry_dt"], t["direction"], t["iv"])

        ep_eff  = round(max(exit_prem - CFG["slippage_pts"], 0), 2)
        pnl_u   = round(ep_eff - t["entry_prem_eff"], 2)
        pnl_g   = round(pnl_u * CFG["lot_size"], 2)
        pnl_n   = round(pnl_g - CFG["brokerage"], 2)
        self._today_pnl += pnl_n

        if pnl_n < 0: self._consec_losses += 1
        else:          self._consec_losses  = 0

        result = {**t,
                  "exit_time"   : ts,
                  "exit_spot"   : round(spot, 2),
                  "exit_reason" : reason,
                  "exit_prem"   : round(exit_prem, 2),
                  "exit_prem_eff": ep_eff,
                  "pnl_total"   : pnl_n,
                  "trade_num"   : self._trades_taken,
                  "vix"         : self._vix,
                  "pcr"         : self._pcr,
        }
        self._completed.append(result)
        self._log_csv(result)

        log.info(f"[{self.name}]  CLOSED #{self._trades_taken}  {reason} "
                 f"{pnl_n:,.0f}  |  Today: {self._today_pnl:,.0f}")

        if t["opt_token"]:
            self.unsubscribe_option(t["opt_token"])

        # Allow 2nd trade if SL hit and within entry window
        if (reason in ("SL_HIT", "PREMIUM_SL", "TRAIL_SL") and
                self._trades_taken < CFG["max_trades_per_day"] and
                ts.time() < CFG["entry_cutoff"]):
            log.info(f"[{self.name}]  2nd trade opportunity available")
            self._trade = None   # reset — on_candle will check next candle
        else:
            self._day_paused = True

    # ── Indicators (Wilder ADX, session VWAP fed from hub) ────────────────────

    def _compute_indicators(self, candles: list) -> dict:
        """
        FIX (Bug 5): candles are now full OHLCV dicts with real high/low/volume.
        Previously, CandleBuilder was fed only close → H=L=O=C → ATR/ADX/ST wrong.
        """
        df    = pd.DataFrame(candles)

        # Ensure required columns exist and are numeric
        for col in ("open", "high", "low", "close", "volume"):
            if col not in df.columns:
                df[col] = df.get("close", 0)
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        close = df["close"]
        high  = df["high"]
        low   = df["low"]
        vol   = df["volume"]

        # RSI
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14, min_periods=1).mean()
        loss  = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
        rsi   = float((100 - (100 / (1 + gain / loss.replace(0, np.nan)))).fillna(50).iloc[-1])

        ema9  = float(close.ewm(span=9,  adjust=False).mean().iloc[-1])
        ema21 = float(close.ewm(span=21, adjust=False).mean().iloc[-1])

        # ATR + Supertrend (now uses real H/L — not flat candles)
        prev_c = close.shift(1)
        tr     = pd.concat([high-low, (high-prev_c).abs(), (low-prev_c).abs()], axis=1).max(1)
        atr7   = tr.rolling(7, min_periods=1).mean()
        hl2    = (high + low) / 2
        bu, bl = hl2 + 3.0*atr7, hl2 - 3.0*atr7
        fu, fl = bu.copy(), bl.copy()
        st_arr = pd.Series(True, index=df.index)
        for i in range(1, len(df)):
            fu.iloc[i] = bu.iloc[i] if close.iloc[i-1]>fu.iloc[i-1] else min(bu.iloc[i], fu.iloc[i-1])
            fl.iloc[i] = bl.iloc[i] if close.iloc[i-1]<fl.iloc[i-1] else max(bl.iloc[i], fl.iloc[i-1])
            if st_arr.iloc[i-1] and close.iloc[i]<fl.iloc[i]:       st_arr.iloc[i]=False
            elif not st_arr.iloc[i-1] and close.iloc[i]>fu.iloc[i]: st_arr.iloc[i]=True
            else:                                                     st_arr.iloc[i]=st_arr.iloc[i-1]
        st_bull = bool(st_arr.iloc[-1])

        # Wilder ADX (now uses real high.diff() / low.diff() — not flat)
        a  = 1.0/14
        dp = pd.Series(np.where((high.diff()>0)&(high.diff()>-low.diff()), high.diff(), 0.0), index=df.index)
        dm = pd.Series(np.where((-low.diff()>0)&(-low.diff()>high.diff()), -low.diff(), 0.0), index=df.index)
        aw = tr.ewm(alpha=a, adjust=False).mean()
        di_p = 100*(dp.ewm(alpha=a, adjust=False).mean()/aw.replace(0,np.nan))
        di_m = 100*(dm.ewm(alpha=a, adjust=False).mean()/aw.replace(0,np.nan))
        dx   = 100*((di_p-di_m).abs()/(di_p+di_m).replace(0,np.nan))
        adx  = float(dx.ewm(alpha=a, adjust=False).mean().fillna(0).iloc[-1])

        # BB
        sma  = close.rolling(20, min_periods=1).mean()
        std  = close.rolling(20, min_periods=1).std().fillna(0)
        bw   = (sma+2*std - (sma-2*std)) / sma.replace(0,np.nan)
        bb_e = bool(bw.iloc[-1]>bw.iloc[-2]) if len(df)>1 else False

        # Volume (now uses real bar volumes, not zeros — Bug 7 fix in market_hub.py)
        va  = float(vol.rolling(10, min_periods=1).mean().iloc[-1])
        vl  = float(vol.iloc[-1])
        v3  = float(vol.iloc[-4:-1].sum()) if len(vol)>=4 else va*3

        return {
            "rsi":rsi, "ema9":ema9, "ema21":ema21,
            "close":float(close.iloc[-1]),
            "st_bull":st_bull, "adx":adx,
            "di_plus":float(di_p.fillna(0).iloc[-1]),
            "di_minus":float(di_m.fillna(0).iloc[-1]),
            "bb_exp":bb_e, "vol_avg":va, "vol_l":vl, "vol_3":v3,
        }

    def _score(self, ind: dict, direction: str, vwap: float | None) -> tuple:
        score, det = 0.0, {}
        c, rsi, e9, e21 = ind["close"], ind["rsi"], ind["ema9"], ind["ema21"]
        st, adx = ind["st_bull"], ind["adx"]

        # ADX [2pts]
        adx_ok = adx >= CFG["adx_min"]
        di_ok  = (direction=="CE" and ind["di_plus"]>ind["di_minus"]) or \
                 (direction=="PE" and ind["di_minus"]>ind["di_plus"])
        if adx_ok and di_ok: score+=2.0; det["adx"]=f" ADX={adx:.1f} DI aligned [+2]"
        elif adx_ok:          score+=1.0; det["adx"]=f" ADX={adx:.1f} DI mismatch [+1]"
        else:                             det["adx"]=f" ADX={adx:.1f}<{CFG['adx_min']} [+0]"

        # EMA [2pts]
        ef = (direction=="CE" and c>e9 and e9>e21) or (direction=="PE" and c<e9 and e9<e21)
        ep = (direction=="CE" and c>e9) or (direction=="PE" and c<e9)
        if ef: score+=2.0; det["ema"]=f" Full stack C={c:.0f} E9={e9:.0f} E21={e21:.0f} [+2]"
        elif ep: score+=1.0; det["ema"]=" Partial EMA [+1]"
        else: det["ema"]=" EMA against [+0]"

        # VWAP [1.5pts]
        if vwap:
            vok = (direction=="CE" and c>vwap) or (direction=="PE" and c<vwap)
            if vok: score+=1.5; det["vwap"]=f" VWAP={vwap:.0f} [+1.5]"
            else:               det["vwap"]=f" VWAP={vwap:.0f} wrong side [+0]"

        # RSI [1pt]
        rok = (direction=="CE" and rsi>=CFG["rsi_bull"]) or (direction=="PE" and rsi<=CFG["rsi_bear"])
        if rok: score+=1.0; det["rsi"]=f" RSI={rsi:.1f} [+1]"
        else: det["rsi"]=f" RSI={rsi:.1f} [+0]"

        # Volume [1pt]
        vs = ind["vol_l"]>ind["vol_3"] and ind["vol_l"]>CFG["vol_multiplier"]*ind["vol_avg"]
        if vs: score+=1.0; det["vol"]=f" Vol surge [+1]"
        else: det["vol"]=" Vol weak [+0]"

        # Supertrend [1pt]
        stok = (direction=="CE" and st) or (direction=="PE" and not st)
        if stok: score+=1.0; det["st"]=" ST aligned [+1]"
        else: det["st"]=" ST against [+0]"

        # BB [0.5pt]
        if ind["bb_exp"]: score+=0.5; det["bb"]=" BB expanding [+0.5]"
        else: det["bb"]=" BB flat [+0]"

        return round(score,1), det

    # ── CSV logger ────────────────────────────────────────────────────────────

    def _log_csv(self, t: dict):
        fname  = CFG["csv_file"]
        exists = os.path.isfile(fname)
        fields = ["trade_num","direction","entry_time","entry_spot","strike",
                  "expiry","dte","entry_prem_eff","target_spot","sl_spot",
                  "exit_time","exit_spot","exit_reason","exit_prem_eff",
                  "pnl_total","vix","pcr","opt_sym"]
        row = {k: t.get(k, "") for k in fields}
        for tf in ["entry_time","exit_time"]:
            if hasattr(row.get(tf), "strftime"):
                row[tf] = row[tf].strftime("%H:%M:%S")
        with open(fname, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if not exists: w.writeheader()
            w.writerow(row)

    def eod_summary(self):
        log.info(f"\n[{self.name}] {'─'*50}")
        log.info(f"[{self.name}] END OF DAY")
        log.info(f"[{self.name}] Trades: {self._trades_taken}  "
                 f"Today PnL: {self._today_pnl:,.0f}")
        for i, t in enumerate(self._completed, 1):
            log.info(f"[{self.name}]   #{i} {t['direction']} {t['exit_reason']} "
                     f"{t['pnl_total']:,.0f}")
        log.info(f"[{self.name}] {'─'*50}\n")
