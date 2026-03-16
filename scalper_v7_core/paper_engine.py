# ==============================================================================
# paper_engine.py  V6 Paper Trading Engine
#
# New in V6:
#   [OK] Uses ATR-scaled SL/TP from RiskManager (not fixed values)
#   [OK] Slippage applied at both entry and exit
#   [OK] Richer trade metadata logged (atr, regime, blocked_by history)
#   [OK] Daily stats include: avg_rr, expectancy, best/worst trade
#   [OK] signal_log includes all 7 filter states per tick
# ==============================================================================

import csv
import os
import time
import datetime as dt

# IST FIX: GitHub Actions runners are UTC — timestamps must be IST
_IST = dt.timezone(dt.timedelta(hours=5, minutes=30))

def _now_ist() -> dt.datetime:
    """Always returns current datetime in IST — works on GitHub Actions (UTC) and local."""
    return dt.datetime.now(tz=_IST).replace(tzinfo=None)
from typing import Optional
from scalper_v7_core.config import (
    SLIPPAGE_POINTS, QUANTITY,
    TRAIL_ARM, TRAIL_STEP,
    ENTRY_LOG, EXIT_LOG, SIGNAL_LOG, DAILY_SUMMARY,
    EXIT_COOLDOWN,
)
from scalper_v7_core.logger_setup import log


# ==============================================================================
# CSV Helper
# ==============================================================================

def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _append_csv(filepath: str, data: dict):
    _ensure_dir(filepath)
    exists = os.path.exists(filepath)
    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(data.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(data)


# ==============================================================================
# Paper Trade Object
# ==============================================================================

class PaperTrade:

    def __init__(
        self,
        symbol:      str,
        token:       int,
        option_type: str,
        ltp:         float,
        qty:         int,
        spot:        float,
        sl_price:    float,   # computed by RiskManager
        tp_price:    float,
        sl_pts:      float,
        tp_pts:      float,
        signal_meta: dict,
        atr14:       float,
    ):
        self.fill_price:  float = round(ltp + SLIPPAGE_POINTS, 2)
        self.symbol      = symbol
        self.token       = token
        self.option_type = option_type
        self.qty         = qty
        self.spot        = spot
        self.atr14       = atr14

        # SL/TP anchored to fill price (not pre-slippage LTP)
        fill_offset   = self.fill_price - ltp     # = SLIPPAGE_POINTS
        self.sl       = round(sl_price - fill_offset, 2)
        self.target   = round(tp_price + fill_offset, 2)
        self.sl_pts   = sl_pts
        self.tp_pts   = tp_pts

        self.trail_stage: int   = 0
        self.timestamp:   str   = _now_ist().isoformat()  # FIX: was UTC on GitHub Actions
        self.signal_meta: dict  = signal_meta
        self.exit_pending: bool = False
        self.last_exit_ts: float = 0.0

    @property
    def entry(self) -> float:
        return self.fill_price

    def to_dict(self) -> dict:
        """Serialisable representation for state.json."""
        return {
            "symbol":      self.symbol,
            "token":       self.token,
            "option_type": self.option_type,
            "qty":         self.qty,
            "entry":       self.entry,
            "sl":          self.sl,
            "target":      self.target,
            "sl_pts":      self.sl_pts,
            "tp_pts":      self.tp_pts,
            "spot":        self.spot,
            "atr14":       self.atr14,
            "trail_stage": self.trail_stage,
            "timestamp":   self.timestamp,
        }


# ==============================================================================
# Paper Engine
# ==============================================================================

class PaperEngine:

    def __init__(self):
        self.active_trade:       Optional[PaperTrade] = None
        self._daily_pnl_pts:     float = 0.0
        self._daily_pnl_rs:      float = 0.0
        self._results:           list  = []
        self._blocked_log:       dict  = {}   # reason -> count (filter analytics)

    # 
    # Entry
    # 

    def open_trade(
        self,
        symbol:      str,
        token:       int,
        option_type: str,
        ltp:         float,
        spot:        float,
        signal_meta: dict,
        sl_price:    float,
        tp_price:    float,
        sl_pts:      float,
        tp_pts:      float,
        atr14:       float,
        qty:         int = QUANTITY,
    ) -> PaperTrade:

        trade = PaperTrade(
            symbol, token, option_type, ltp, qty, spot,
            sl_price, tp_price, sl_pts, tp_pts, signal_meta, atr14,
        )
        self.active_trade = trade
        rr = round(tp_pts / sl_pts, 2) if sl_pts > 0 else 0

        log.info(
            f"[EXIT-TGT] PAPER ENTRY {option_type} | {symbol} | "
            f"Fill={trade.fill_price:.2f} (LTP={ltp:.2f}) | "
            f"SL={trade.sl:.2f}(-{sl_pts:.1f}) | TG={trade.target:.2f}(+{tp_pts:.1f}) | "
            f"RR=1:{rr} | ATR={atr14:.1f}"
        )

        row = {
            "timestamp":   trade.timestamp,
            "symbol":      symbol,
            "option_type": option_type,
            "qty":         qty,
            "ltp":         ltp,
            "fill":        trade.fill_price,
            "sl":          trade.sl,
            "target":      trade.target,
            "sl_pts":      sl_pts,
            "tp_pts":      tp_pts,
            "rr":          rr,
            "spot":        spot,
            "atr14":       atr14,
            **{f"sig_{k}": v for k, v in signal_meta.items()},
        }
        _append_csv(ENTRY_LOG, row)
        return trade

    # 
    # Trade Management
    # 

    def manage_trade(self, ltp: float) -> Optional[str]:
        """Check trailing SL and exit conditions. Returns exit reason or None."""
        trade = self.active_trade
        if not trade:
            return None

        now_ts = time.time()
        if trade.exit_pending and (now_ts - trade.last_exit_ts) < EXIT_COOLDOWN:
            return None

        pnl_pts = ltp - trade.entry
        self._update_trail(trade, ltp, pnl_pts)

        log.debug(
            f"[LIVE] {trade.symbol} | LTP={ltp:.2f} | "
            f"PnL={pnl_pts:+.2f} | SL={trade.sl:.2f} | "
            f"TG={trade.target:.2f} | Stage={trade.trail_stage}"
        )

        if ltp >= trade.target: return self._close(ltp, "TARGET")
        if ltp <= trade.sl:     return self._close(ltp, "SL")
        return None

    def close_trade_forced(self, ltp: float, reason: str = "SQUAREOFF") -> str:
        if not self.active_trade:
            return reason
        return self._close(ltp, reason)

    # 
    # Trailing SL
    # 

    def _update_trail(self, trade: PaperTrade, ltp: float, pnl_pts: float):
        if pnl_pts >= TRAIL_ARM and trade.trail_stage == 0:
            new_sl = round(trade.entry, 2)
            if new_sl > trade.sl:
                trade.sl          = new_sl
                trade.trail_stage = 1
                log.info(f"[TRAIL] Trail  BE | SL={trade.sl:.2f}")

        elif pnl_pts > TRAIL_ARM and trade.trail_stage >= 1:
            new_stage = int((pnl_pts - TRAIL_ARM) / TRAIL_STEP) + 1
            if new_stage > trade.trail_stage:
                inc               = (new_stage - trade.trail_stage) * TRAIL_STEP
                trade.sl          = round(trade.sl + inc, 2)
                trade.trail_stage = new_stage
                log.info(f"[TRAIL] Trail stage {new_stage} | SL={trade.sl:.2f}")

    # 
    # Exit
    # 

    def _close(self, ltp: float, reason: str) -> str:
        trade = self.active_trade
        if not trade:
            return reason

        trade.exit_pending  = True
        trade.last_exit_ts  = time.time()

        exit_px  = round(ltp - SLIPPAGE_POINTS, 2)   # simulate sell slippage
        pnl_pts  = round(exit_px - trade.entry, 2)
        pnl_rs   = round(pnl_pts * trade.qty, 2)
        rr_actual = round(pnl_pts / trade.sl_pts, 2) if trade.sl_pts else 0

        self._daily_pnl_pts += pnl_pts
        self._daily_pnl_rs  += pnl_rs

        result = {
            "timestamp":      _now_ist().isoformat(),  # FIX: was UTC on GitHub Actions
            "symbol":         trade.symbol,
            "option_type":    trade.option_type,
            "qty":            trade.qty,
            "entry":          trade.entry,
            "exit":           exit_px,
            "pnl_pts":        pnl_pts,
            "pnl_rs":         pnl_rs,
            "rr_actual":      rr_actual,
            "sl_pts":         trade.sl_pts,
            "tp_pts":         trade.tp_pts,
            "reason":         reason,
            "trail_stage":    trade.trail_stage,
            "spot":           trade.spot,
            "atr14":          trade.atr14,
            "daily_pnl_rs":   round(self._daily_pnl_rs, 2),
        }

        _append_csv(EXIT_LOG, result)
        self._results.append(result)

        emoji = "[EXIT-TGT]" if reason == "TARGET" else ("[EXIT]" if reason == "SQUAREOFF" else "[EXIT-SL]")
        log.info(
            f"{emoji} PAPER EXIT | {trade.symbol} | {reason} | "
            f"Exit={exit_px:.2f} | PnL={pnl_pts:+.2f}pts ({pnl_rs:+.2f}) | "
            f"RR={rr_actual:+.2f} | Day={self._daily_pnl_rs:+.2f}"
        )

        self.active_trade = None
        return reason

    # 
    # Signal Logging (every tick)
    # 

    def log_signal(self, snapshot: dict, signal: dict, persistence_ok: bool):
        ind1m = signal.get("ind1m", {})
        ind5m = signal.get("ind5m", {})

        blocked = signal.get("blocked_by", "")
        if blocked:
            self._blocked_log[blocked] = self._blocked_log.get(blocked, 0) + 1

        row = {
            "timestamp":      _now_ist().isoformat(),  # FIX: was UTC on GitHub Actions
            "spot":           snapshot.get("spot", ""),
            "action":         signal.get("action", "HOLD"),
            "trend_bias":     signal.get("trend_bias", ""),
            "in_lunch":       signal.get("in_lunch", False),
            "blocked_by":     blocked,
            "persistence":    persistence_ok,
            # 1-min indicators
            "ema20_1m":       ind1m.get("ema20",         ""),
            "ema50_1m":       ind1m.get("ema50",         ""),
            "ema_gap_1m":     ind1m.get("ema_gap",       ""),
            "rsi14_1m":       ind1m.get("rsi14",         ""),
            "rsi_z_1m":       ind1m.get("rsi_z",         ""),
            "rsi_slope_1m":   ind1m.get("rsi_slope",     ""),
            "rsi_acc_1m":     ind1m.get("rsi_acc",       ""),
            "macd_hist_1m":   ind1m.get("macd_hist",     ""),
            "macd_slope_1m":  ind1m.get("macd_slope",    ""),
            "atr14_1m":       ind1m.get("atr14",         ""),
            "atr_pct_1m":     ind1m.get("atr_pct",       ""),
            "atr_vol_ratio":  ind1m.get("atr_vol_ratio", ""),
            "regime_1m":      ind1m.get("regime",        ""),
            "volume_1m":      ind1m.get("volume_trend",  ""),
            "struct_hi":      ind1m.get("structure_high",""),
            "struct_lo":      ind1m.get("structure_low", ""),
            "vwap":           ind1m.get("vwap",          ""),
            # 5-min indicators
            "ema_gap_5m":     ind5m.get("ema_gap",       ""),
            "atr14_5m":       ind5m.get("atr14",         ""),
            "atr_pct_5m":     ind5m.get("atr_pct",       ""),
            "regime_5m":      ind5m.get("regime",        ""),
        }
        _append_csv(SIGNAL_LOG, row)

    # 
    # Daily Summary
    # 

    def write_daily_summary(self):
        results = self._results
        total   = len(results)
        wins    = [r for r in results if r["pnl_pts"] > 0]
        losses  = [r for r in results if r["pnl_pts"] <= 0]

        avg_win  = round(sum(r["pnl_pts"] for r in wins)   / len(wins),   2) if wins   else 0
        avg_loss = round(sum(r["pnl_pts"] for r in losses) / len(losses), 2) if losses else 0
        win_pct  = len(wins) / total if total else 0
        expectancy = round(win_pct * avg_win + (1 - win_pct) * avg_loss, 3) if total else 0

        best  = max((r["pnl_pts"] for r in results), default=0)
        worst = min((r["pnl_pts"] for r in results), default=0)

        row = {
            "date":          dt.date.today().isoformat(),
            "trades":        total,
            "wins":          len(wins),
            "losses":        len(losses),
            "win_rate":      f"{win_pct*100:.1f}%",
            "expectancy":    expectancy,
            "pnl_pts":       round(self._daily_pnl_pts, 2),
            "pnl_rs":        round(self._daily_pnl_rs,  2),
            "avg_win_pts":   avg_win,
            "avg_loss_pts":  avg_loss,
            "best_trade":    best,
            "worst_trade":   worst,
            "top_block":     max(self._blocked_log, key=self._blocked_log.get, default=""),
        }
        _append_csv(DAILY_SUMMARY, row)

        log.info(
            f"[STATS] Day Summary | Trades={total} | W/L={len(wins)}/{len(losses)} | "
            f"WinRate={win_pct*100:.1f}% | Expectancy={expectancy:+.3f}pts | "
            f"PnL={self._daily_pnl_rs:+.2f}"
        )
        if self._blocked_log:
            sorted_blocks = sorted(self._blocked_log.items(), key=lambda x: -x[1])
            log.info(f"[DEBUG] Top blocks: {sorted_blocks[:5]}")

    @property
    def daily_pnl_rupees(self) -> float:
        return self._daily_pnl_rs
