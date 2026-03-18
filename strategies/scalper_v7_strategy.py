"""
strategies/scalper_v7_strategy.py

ScalperV7Strategy — Ported ScalerV7 into the shared multi-strategy framework.

HOW DATA FLOWS (vs original standalone scalper_v7):

  ORIGINAL scalper_v7                    THIS STRATEGY
  ──────────────────────────────────────────────────────
  REST poll every 0.5s:                  WebSocket callbacks:
    kite.historical_data(1m)  → on_tick()  → 1-min candle
    kite.historical_data(5m)  → on_candle() → 5-min candle
    bus.get_ltp(option_token) → on_option_tick() → SL/TP
  Own KiteTicker (TickerBus) → Shared MarketHub WS
  Own login (load_clients())  → Shared hub.kite
  Own NFO instruments         → Shared InstrumentStore


WHAT IS NOT SHARED (intentionally per-strategy):
   1-min CandleBuilder (private to this strategy)
   5-min candle buffer (populated from hub's on_candle broadcast)
   RiskManager, PaperEngine, PersistenceFilter (stateful, per-strategy)
   CSV logs (scalper_v7_entry.csv / exit.csv etc.)
   Active option token + subscription

SIGNAL EVALUATION TIMING:
  Original: every 0.5s (polling)
  This port: every 1-min candle close (cleaner, same data, less CPU)
  Trade management (SL/TP/trail): on every option WebSocket tick via on_option_tick()

FIXES APPLIED:
  Bug 2 — time.sleep(0.3) was called inside _evaluate_signal(), which is
           invoked from on_tick(), which runs on the KiteTicker WebSocket
           callback thread. Sleeping here blocked ALL tick processing for all
           strategies for 300ms on every entry attempt.
           Additionally, during that sleep the option we just subscribed
           could not receive any ticks, so get_price() always returned None.
           Fix: removed time.sleep(). If LTP is unavailable immediately after
           subscribe, we log a warning and return — the next 1-min bar will
           re-evaluate. The option remains subscribed so subsequent ticks will
           populate the price cache.

  Bug 8 — option_type passed to risk.on_trade_exit() was always "" because
           active_trade is set to None by manage_trade() / close_trade_forced()
           before we read it. This silently broke the directional consecutive-loss
           blocker (CE/PE direction was never incremented, only "" was).
           Fix: capture opt_type = active_trade.option_type BEFORE the close call
           in ALL THREE exit paths:
             - on_option_tick()    (was already fixed)
             - eod_summary()       (fixed in previous pass)
             - _handle_squareoff() (fixed in this pass — was the last remaining gap)
"""

import logging
import threading
from collections import deque
from datetime import datetime, time as dtime
from typing import Optional

import pandas as pd

from core.base_strategy import BaseStrategy
from core.candle import CandleBuilder

from scalper_v7_core.signal_logic import get_signal
from scalper_v7_core.risk_manager import RiskManager
from scalper_v7_core.paper_engine import PaperEngine, PaperTrade
from scalper_v7_core.state_manager import save_state, load_state, clear_state
from scalper_v7_core.config import (
    CANDLE_1M_USE, CANDLE_5M_USE,
)

log = logging.getLogger("strategy.scalper_v7")


# ── Persistence Filter (same as original scalper_v7/main.py) ─────────────────

class PersistenceFilter:
    """Signal must repeat N consecutive 1-min bars before entry fires."""

    def __init__(self, window: int = 2):
        from collections import deque
        self._q    = deque(maxlen=window)
        self._size = window

    def push(self, action: str) -> bool:
        self._q.append(action)
        if len(self._q) < self._size:
            return False
        return all(a == action and a != "HOLD" for a in self._q)

    def reset(self):
        self._q.clear()


# ── Strategy ──────────────────────────────────────────────────────────────────

class ScalperV7Strategy(BaseStrategy):
    """
    Full ScalerV7 (all 11 filters active) running inside the shared framework.
    Receives ticks/candles via callbacks. Never touches Kite or WebSocket directly.
    """

    @property
    def name(self) -> str:
        return "SCALPER_V7"

    def __init__(self, market_hub):
        super().__init__(market_hub)

        # ── Private 1-min candle builder ─────────────────────────────────────
        # MarketHub broadcasts 5-min candles via on_candle().
        # For 1-min, we build our own from raw index ticks in on_tick().
        self._cb_1m = CandleBuilder(minutes=1)

        # Rolling candle buffers — converted to DataFrame for indicators
        # Capacity matches scalper_v7 config (CANDLE_1M_USE=80, CANDLE_5M_USE=50)
        self._buf_1m: deque = deque(maxlen=CANDLE_1M_USE)   # last N closed 1-min candles
        self._buf_5m: deque = deque(maxlen=CANDLE_5M_USE)   # last N closed 5-min candles

        # ── Scalper v7 components ─────────────────────────────────────────────
        self._engine      = PaperEngine()
        self._risk        = RiskManager()
        self._persistence = PersistenceFilter(window=2)

        # ── Active option tracking ────────────────────────────────────────────
        self._active_token: Optional[int] = None   # currently subscribed option
        self._active_sym:   Optional[str] = None

        # ── Pre-market data (set in pre_market()) ─────────────────────────────
        self._instruments = None   # InstrumentStore from t.py (shared)
        self._expiry_date = None   # nearest expiry date (from PreMarketData)

        # ── Thread safety ─────────────────────────────────────────────────────
        self._lock = threading.Lock()

        log.info("[ScalperV7] Strategy initialized")

    # ── Pre-market: called once by t.py at 9:00–9:14 ─────────────────────────

    def pre_market(self, premarket_data, instruments) -> bool:
        """
        Called once with shared PreMarketData + InstrumentStore.
        Returns True to activate today, False to skip.
        """
        self._instruments = instruments
        self._expiry_date = premarket_data.expiry_date
        self._risk.reset_day()

        # Crash recovery — restore any trade open from previous run
        recovered = load_state()
        if recovered:
            log.warning(f"[ScalperV7] Recovering open trade: {recovered.get('symbol')}")
            rt = object.__new__(PaperTrade)
            rt.__dict__.update({
                "fill_price":   float(recovered["entry"]),
                "symbol":       recovered["symbol"],
                "token":        int(recovered.get("token", 0)),
                "option_type":  recovered["option_type"],
                "qty":          int(recovered["qty"]),
                "spot":         float(recovered.get("spot", 0)),
                "atr14":        float(recovered.get("atr14", 0)),
                "sl":           float(recovered["sl"]),
                "target":       float(recovered["target"]),
                "sl_pts":       float(recovered.get("sl_pts", 5)),
                "tp_pts":       float(recovered.get("tp_pts", 9)),
                "trail_stage":  int(recovered.get("trail_stage", 0)),
                "timestamp":    recovered.get("timestamp", ""),
                "signal_meta":  {},
                "exit_pending": False,
                "last_exit_ts": 0.0,
            })
            self._engine.active_trade = rt
            self._active_token = rt.token
            self.subscribe_option(rt.token)   # re-subscribe via MarketHub
            log.info(f"[ScalperV7] Trade restored: {rt.symbol} token={rt.token}")

        log.info(
            f"[ScalperV7] Pre-market complete. "
            f"VIX={premarket_data.vix} PCR={premarket_data.pcr} "
            f"Expiry={premarket_data.expiry_date} DTE={premarket_data.dte_days}"
        )
        return True

    # ── on_tick: raw index tick → build 1-min candles ─────────────────────────

    def on_tick(self, price: float, ts: datetime, tick_ts: datetime):
        """
        Called on every raw BankNifty index tick.
        We use this to build 1-min candles (strategy-private, not shared).
        Signal evaluation is triggered when a 1-min bar closes.
        """
        # Auto square-off check (fast path, no lock needed)
        if self._risk.should_squareoff():
            self._handle_squareoff()
            return

        # Feed tick into 1-min builder (thread-safe internally)
        closed = self._cb_1m.feed_tick(price, 0, ts)
        if closed is None:
            return  # bar still open, nothing to do

        # A 1-min candle just closed — store and evaluate signal
        with self._lock:
            self._buf_1m.append(closed)

        # Don't evaluate if we have an open trade (managed via on_option_tick)
        if self._engine.active_trade:
            return

        self._evaluate_signal(current_spot=price)

    # ── on_candle: 5-min index candle from MarketHub ──────────────────────────

    def on_candle(self, candle: dict, ts: datetime):
        """
        Called when MarketHub's shared 5-min index candle closes.
        We store it in our private buffer for use in signal evaluation.
        """
        with self._lock:
            self._buf_5m.append(candle)

    # ── on_option_tick: option LTP for trade management ───────────────────────

    def on_option_tick(self, token: int, price: float, ts: datetime, tick_ts: datetime = None):
        """
        Called when a subscribed option has a new WebSocket tick.
        This is the SL/TP management loop — replaces the 0.5s polling in original.

        FIX (Bug 8): capture opt_type BEFORE calling manage_trade().
        manage_trade() sets active_trade=None internally when it closes.
        Reading active_trade.option_type AFTER the call always returned None,
        making the directional loss counter receive "" every time.
        """
        if token != self._active_token:
            return
        if not self._engine.active_trade:
            return

        # FIX (Bug 8): read option_type NOW, before manage_trade() clears active_trade
        opt_type = self._engine.active_trade.option_type

        reason = self._engine.manage_trade(price)
        if reason:
            # _close() already set active_trade = None inside PaperEngine
            if self._engine._results:
                last    = self._engine._results[-1]
                pnl_pts = last["pnl_pts"]
                pnl_rs  = last["pnl_rs"]
                # FIX (Bug 8): pass opt_type captured above (not active_trade.option_type
                # which is now None — that caused the risk counter to receive "" always)
                self._risk.on_trade_exit(pnl_pts, pnl_rs, reason, opt_type)

            clear_state()
            self._persistence.reset()
            self._unsubscribe_active_option()
            log.info(f"[ScalperV7] Trade closed: {reason} | Token={token}")

    # ── Signal evaluation (called on every 1-min close) ──────────────────────

    def _evaluate_signal(self, current_spot: float):
        """
        Build snapshot from buffered candles, run signal logic, handle entry.
        This mirrors scalper_v7/main.py::run_once() but uses local candle data.
        """
        with self._lock:
            candles_1m = list(self._buf_1m)
            candles_5m = list(self._buf_5m)

        if len(candles_1m) < 15:
            return  # need minimum bars for indicators

        # Convert candle dicts to DataFrames (same format scalper_v7 expects)
        df_1m = self._candles_to_df(candles_1m)
        df_5m = self._candles_to_df(candles_5m) if len(candles_5m) >= 5 else pd.DataFrame()

        # Inject session VWAP from MarketHub (more accurate than candle-based approximation)
        # The indicators.py VWAP is cumulative from available candles only;
        # MarketHub.session_vwap is tick-accurate from 9:15 AM (true VWAP).
        hub_vwap = self._hub.session_vwap.value if self._hub.session_vwap.ready else None

        # Build snapshot dict (mirrors get_market_snapshot() output)
        from scalper_v7_core.indicators import compute_indicators
        ind_1m = compute_indicators(df_1m)
        ind_5m = compute_indicators(df_5m) if not df_5m.empty else {}

        # Override VWAP with hub's session VWAP if available
        if hub_vwap and hub_vwap > 0:
            ind_1m["vwap"] = hub_vwap

        atm_strike = self._get_atm(current_spot)

        snapshot = {
            "spot":          current_spot,
            "atm_strike":    atm_strike,
            "indicators_1m": ind_1m,
            "indicators_5m": ind_5m,
            "candles_1m":    df_1m,
            "candles_5m":    df_5m,
            "valid":         True,
        }

        # Run signal logic (all 11 V7 filters)
        signal   = get_signal(snapshot)
        action   = signal.get("action", "HOLD")
        ind      = signal.get("ind1m", {})

        # Persistence check (signal must fire 2 consecutive 1-min bars)
        signal_confirmed = self._persistence.push(action)

        log.info(
            f"[ScalperV7] {action:8s} | "
            f"Spot={current_spot:.2f} ATM={atm_strike} | "
            f"RSI={ind.get('rsi14',0):.1f} Rz={ind.get('rsi_z',0):.2f} | "
            f"MACD={ind.get('macd_hist',0):.3f} | "
            f"ATR={ind.get('atr14',0):.1f} | "
            f"Regime={ind.get('regime','?')} | "
            f"Block={signal.get('blocked_by','-') or 'none'} | "
            f"Persist={signal_confirmed} | "
            f"{self._risk.status_line()}"
        )

        # Log every bar to CSV (signal_log)
        self._engine.log_signal(snapshot, signal, signal_confirmed)

        if not signal_confirmed or action == "HOLD":
            if signal.get("in_lunch") and action == "HOLD":
                self._persistence.reset()
            return

        # ── Entry flow ────────────────────────────────────────────────────────
        opt = "CE" if action == "BUY_CE" else "PE"

        can_enter, block_reason = self._risk.can_enter(option_type=opt)
        if not can_enter:
            log.info(f"[ScalperV7] Risk gate blocked: {block_reason}")
            self._persistence.reset()
            return

        # Find option instrument via shared InstrumentStore
        strike_offset = self._risk.expiry_strike_offset()
        tsym, token   = self._find_option(atm_strike, opt, strike_offset)
        if not tsym:
            log.error(f"[ScalperV7] Option not found: ATM={atm_strike} {opt} offset={strike_offset}")
            return

        # Subscribe the option token and read LTP from MarketHub's cache.
        # FIX (Bug 2): removed time.sleep(0.3) here. Sleeping inside the
        # WebSocket callback thread (on_tick → _evaluate_signal) blocked ALL
        # tick processing for all strategies. The option also couldn't receive
        # its first tick during the sleep, so get_price() always returned None.
        #
        # New behaviour: subscribe and immediately read the cache.
        # - If the option was pre-subscribed (from a prior bar's first signal),
        #   get_price() will return a valid LTP immediately.
        # - If this is the first time we see this token (fresh signal), LTP may
        #   be None. We log a warning and return without entering — the next
        #   1-min bar will re-evaluate with the option now subscribed and priced.
        self.subscribe_option(token)
        ltp = self.get_price(token)

        if not ltp:
            log.warning(
                f"[ScalperV7] LTP unavailable for {tsym} (token={token}) — "
                f"option subscribed, will retry next 1-min bar"
            )
            # Keep the token subscribed so the next bar can read it immediately.
            # Do NOT reset persistence — let the next bar confirm and re-attempt.
            return

        # Compute ATR-scaled SL/TP
        atr14 = float(ind.get("atr14", 0))
        sl_price, tp_price, sl_pts, tp_pts = self._risk.compute_sl_tp(ltp, atr14)

        # Option sanity check
        ltp_ok, ltp_reason = self._risk.validate_option_ltp(ltp, current_spot, tp_pts)
        if not ltp_ok:
            log.warning(f"[ScalperV7] Option sanity FAILED: {ltp_reason}")
            self.unsubscribe_option(token)
            return

        # Build signal metadata for CSV
        signal_meta = {
            "rsi14":      ind.get("rsi14",       ""),
            "rsi_z":      ind.get("rsi_z",        ""),
            "rsi_slope":  ind.get("rsi_slope",    ""),
            "rsi_acc":    ind.get("rsi_acc",       ""),
            "macd_hist":  ind.get("macd_hist",    ""),
            "macd_slope": ind.get("macd_slope",   ""),
            "atr14":      atr14,
            "atr_pct":    ind.get("atr_pct",      ""),
            "regime":     ind.get("regime",       ""),
            "volume":     ind.get("volume_trend", ""),
            "trend_5m":   signal.get("trend_bias",""),
            "sl_pts":     sl_pts,
            "tp_pts":     tp_pts,
        }

        # Open paper trade
        self._engine.open_trade(
            symbol      = tsym,
            token       = token,
            option_type = opt,
            ltp         = ltp,
            spot        = current_spot,
            signal_meta = signal_meta,
            sl_price    = sl_price,
            tp_price    = tp_price,
            sl_pts      = sl_pts,
            tp_pts      = tp_pts,
            atr14       = atr14,
        )

        self._risk.on_trade_entry(opt)
        self._active_token = token
        self._active_sym   = tsym
        save_state(self._engine.active_trade.to_dict())
        self._persistence.reset()

        log.info(f"[ScalperV7]  Entered {opt} | {tsym} | LTP={ltp:.2f} | SL={sl_price:.2f} TP={tp_price:.2f}")

    # ── EOD ───────────────────────────────────────────────────────────────────

    def eod_summary(self):
        """Called at 3:30 PM by MarketHub."""
        if self._engine.active_trade:
            # Force close at whatever price we have.
            token    = self._engine.active_trade.token
            ltp      = self.get_price(token) or self._engine.active_trade.entry
            # FIX (Bug 8 — eod_summary): capture opt_type BEFORE close_trade_forced().
            # close_trade_forced() calls _close() internally which sets active_trade=None.
            # Reading active_trade.option_type AFTER the call always evaluates the ternary
            # to "" — the directional loss counter silently received "" on every EOD close.
            opt_type = self._engine.active_trade.option_type
            reason   = self._engine.close_trade_forced(ltp, "EOD-SQUAREOFF")
            if self._engine._results:
                last = self._engine._results[-1]
                self._risk.on_trade_exit(
                    last["pnl_pts"], last["pnl_rs"], reason,
                    opt_type,   # ← captured before close; active_trade is None here
                )
            clear_state()
            self._unsubscribe_active_option()

        self._engine.write_daily_summary()
        log.info(
            f"[ScalperV7] EOD Summary | "
            f"Day PnL={self._engine.daily_pnl_rupees:+.2f} | "
            f"1m bars={len(self._buf_1m)} 5m bars={len(self._buf_5m)}"
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _handle_squareoff(self):
        """Force close any open trade when risk manager triggers auto square-off.

        FIX (Bug 8 — _handle_squareoff): this was the last remaining place where
        active_trade.option_type was read AFTER close_trade_forced() nulled it.
        close_trade_forced() calls _close() internally which sets active_trade=None,
        so the RiskManager's directional consecutive-loss counter received "" on
        every intraday squareoff — corrupting CE/PE tracking for the rest of the
        session and for next-day crash recovery state.

        Same fix as on_option_tick() and eod_summary(): capture opt_type on its
        own line BEFORE the close call.
        """
        if not self._engine.active_trade:
            return
        token    = self._engine.active_trade.token
        ltp      = self.get_price(token) or self._engine.active_trade.entry
        # FIX (Bug 8): capture opt_type BEFORE close_trade_forced() nulls active_trade
        opt_type = self._engine.active_trade.option_type
        reason   = self._engine.close_trade_forced(ltp, "AUTO-SQUAREOFF")
        if self._engine._results:
            last = self._engine._results[-1]
            self._risk.on_trade_exit(
                last["pnl_pts"], last["pnl_rs"], reason,
                opt_type,   # ← captured before close; active_trade is None here
            )
        clear_state()
        self._persistence.reset()
        self._unsubscribe_active_option()

    def _unsubscribe_active_option(self):
        """Unsubscribe option token from WebSocket to save bandwidth."""
        if self._active_token:
            self.unsubscribe_option(self._active_token)
            log.debug(f"[ScalperV7] Unsubscribed token {self._active_token}")
            self._active_token = None
            self._active_sym   = None

    @staticmethod
    def _candles_to_df(candles: list) -> pd.DataFrame:
        """
        Convert list of candle dicts (ts, open, high, low, close, volume)
        to a pandas DataFrame that compute_indicators() expects.
        """
        if not candles:
            return pd.DataFrame()
        df = pd.DataFrame(candles)
        for col in ("open", "high", "low", "close", "volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.dropna(subset=["close"]).reset_index(drop=True)

    @staticmethod
    def _get_atm(spot: float) -> int:
        """Round spot to nearest 100-point BankNifty strike."""
        from scalper_v7_core.config import STRIKE_STEP
        return int(round(spot / STRIKE_STEP) * STRIKE_STEP)

    def _find_option(
        self,
        strike: int,
        opt_type: str,
        strike_offset: int = 0,
    ):
        """
        Find nearest-expiry option token using shared InstrumentStore.
        Falls back to raw kite.instruments scan if InstrumentStore doesn't
        have the helper (maintains backward compatibility).
        """
        if self._instruments is None:
            log.error("[ScalperV7] InstrumentStore not set — call pre_market first")
            return None, None

        try:
            # Try InstrumentStore's API (matches how ORB strategy uses it)
            # adjust strike for expiry-day ITM
            adj_strike = strike
            if strike_offset > 0:
                adj_strike = strike - strike_offset if opt_type == "CE" else strike + strike_offset
                log.info(f"[ScalperV7] Expiry ITM offset: strike {strike} → {adj_strike}")

            token, sym = self._instruments.get_option_token(
                strike      = adj_strike,
                opt_type    = opt_type,
                expiry_date = self._expiry_date,
            )
            if token:
                return sym or f"BANKNIFTY_{adj_strike}{opt_type}", token

        except Exception as e:
            log.warning(f"[ScalperV7] InstrumentStore lookup failed: {e}  falling back to raw scan")

        # Fallback: direct raw scan via scalper_v7's own find_option_instrument
        # Uses hub.kite (shared session, no extra login)
        try:
            from scalper_v7_core.signal_logic import _in_lunch  # noqa  just to confirm import works
            import re, datetime as dt
            from scalper_v7_core.config import ROOT, EXCHANGE_NFO

            if strike_offset > 0:
                adj_strike = strike - strike_offset if opt_type == "CE" else strike + strike_offset
            else:
                adj_strike = strike

            instruments = self._hub.kite.instruments(EXCHANGE_NFO)
            strike_str  = str(int(adj_strike))
            pattern     = re.compile(
                rf"^{ROOT}\d{{2}}[A-Z0-9]{{3,6}}{strike_str}{opt_type}$",
                re.IGNORECASE,
            )
            candidates = [
                (inst["tradingsymbol"], inst["instrument_token"], inst["expiry"])
                for inst in instruments
                if (
                    pattern.match(inst.get("tradingsymbol", ""))
                    and inst.get("segment") == "NFO-OPT"
                    and inst.get("instrument_type") == opt_type
                )
            ]
            if not candidates:
                # broader fallback
                candidates = [
                    (inst["tradingsymbol"], inst["instrument_token"], inst["expiry"])
                    for inst in instruments
                    if (
                        ROOT in inst.get("tradingsymbol", "")
                        and inst.get("tradingsymbol", "").endswith(opt_type)
                        and inst.get("segment") == "NFO-OPT"
                        and f"{strike_str}{opt_type}" in inst.get("tradingsymbol", "")
                    )
                ]
            if not candidates:
                return None, None

            candidates.sort(key=lambda x: x[2])
            tsym, token, expiry = candidates[0]
            log.info(f"[ScalperV7] Raw scan found: {tsym} (exp {expiry})")
            return tsym, token

        except Exception as e:
            log.error(f"[ScalperV7] Raw instrument scan failed: {e}")
            return None, None
