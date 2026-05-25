"""
strategies/banknifty_expiry_momentum_strategy.py
──────────────────────────────────────────────────────────────────────────────
BankNiftyExpiryMomentumStrategy  —  BankNifty monthly expiry directional play

RUNS ONLY:  when DTE == 0  (BankNifty monthly expiry day)

PHASE 2  (2:00 PM – 3:20 PM):
  Reference range (built from 5-min candles):
    • 13:55 candle (closes at 14:00) defines:
        range_high = candle["high"]
        range_low  = candle["low"]
    • Fallback: last candle received between 13:45–13:59 if 13:55 not seen.

  Entry (after 14:00, one trade max):
    • BankNifty spot breaks ABOVE range_high → BUY ATM CE
    • BankNifty spot breaks BELOW range_low  → BUY ATM PE
    • Filter 1: EMA9 > EMA21 on 5-min for CE (EMA9 < EMA21 for PE)
    • Filter 2: ATR(14) on 5-min > ATR_MIN (30 pts)
    • One trade per day — first breakout wins, no flip allowed.

  Trade management (on every option tick):
    • SL     = entry_price × (1 − SL_PCT)    [40%]
    • Target = entry_price × (1 + TGT_PCT)   [90%]
    • Trail  : arm when profit >= TRAIL_ARM_PCT × entry [50%] → move SL to BE
               then trail SL up every TRAIL_STEP pts above BE

  Hard exit: 3:20 PM flat, no exceptions.

INDEX / INSTRUMENTS:
    No INDEX_TOKEN → MarketHub routes BankNifty ticks (260105) here.
    Gets pm + instruments (BankNifty) from t.py.
    Strike step 100 pts.  BankNifty lot size 15 (post-SEBI revision).

ORDER FLOW:
    Entry → _place_buy()              (BUY to open long)
    Exit  → _place_sell_with_retry()  (SELL to close long)
    PnL   = (close_price − entry_price) × qty
"""

import csv
import logging
import os
import threading
from collections import deque
from datetime import datetime, time as dtime, timezone, timedelta
from typing import Optional

import numpy as np

from core.base_strategy import BaseStrategy
from core.instruments import get_atm_strike

_IST = timezone(timedelta(hours=5, minutes=30))
def _now_ist() -> datetime:
    return datetime.now(tz=_IST).replace(tzinfo=None)

log = logging.getLogger("strategy.banknifty_expiry_momentum")

# ─────────────────────────────────────────────────────────────────────────────
LIVE_MODE = False

CFG = {
    # Entry filters
    "atr_min"           : 30.0,   # ATR(14) on 5-min must exceed this
    "ema_fast"          : 9,
    "ema_slow"          : 21,
    "min_bars_ema"      : 22,     # bypass EMA filter if fewer bars available
    "min_ltp"           : 20.0,   # reject near-zero premium options

    # Trade management (% of entry premium)
    "sl_pct"            : 0.40,   # SL at 40% loss
    "tgt_pct"           : 0.90,   # target at 90% gain
    "trail_arm_pct"     : 0.50,   # arm trail when profit >= 50% of entry
    "trail_step"        : 15.0,   # trail SL step in pts after BE

    # Timing
    "range_candle_hour" : 13,
    "range_candle_min"  : 55,     # 13:55 candle closes at 14:00
    "entry_start"       : (14, 0),
    "cutoff_time"       : (15, 20),
    "auto_squareoff"    : (15, 25),

    # Position sizing — BankNifty lot (post-SEBI revision)
    "quantity"          : 15,

    "entry_csv"         : "logs/banknifty_expiry_momentum_entry.csv",
    "exit_csv"          : "logs/banknifty_expiry_momentum_exit.csv",
}


# ─────────────────────────────────────────────────────────────────────────────
class _Trade:
    __slots__ = [
        "token", "symbol", "opt_type", "entry_price",
        "sl", "target", "trail_active", "trail_sl",
        "order_id", "qty", "entry_time",
    ]
    def __init__(self, token, symbol, opt_type, entry_price,
                 sl, target, order_id, qty, entry_time):
        self.token        = token
        self.symbol       = symbol
        self.opt_type     = opt_type
        self.entry_price  = entry_price
        self.sl           = sl
        self.target       = target
        self.trail_active = False
        self.trail_sl     = sl
        self.order_id     = order_id
        self.qty          = qty
        self.entry_time   = entry_time


def _csv_append(filepath: str, row: dict):
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    write_header = not os.path.isfile(filepath)
    try:
        with open(filepath, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
            w.writerow(row)
    except Exception as e:
        log.error(f"[BANKNIFTY_EXPIRY_MOMENTUM] CSV write error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
class BankNiftyExpiryMomentumStrategy(BaseStrategy):
    """
    BankNifty monthly expiry day directional momentum — Phase 2 (2–3:20 PM).
    No INDEX_TOKEN → receives BankNifty on_tick() + on_candle() from MarketHub.
    """

    LIVE_MODE = LIVE_MODE

    @property
    def name(self) -> str:
        return "BANKNIFTY_EXPIRY_MOMENTUM"

    def __init__(self, hub):
        super().__init__(hub)
        self._lock = threading.Lock()

        self._active            : bool  = False   # True only when DTE == 0
        self._trade             : Optional[_Trade] = None
        self._entry_attempted   : bool  = False
        self._squareoff_done    : bool  = False
        self._pending_entry     : Optional[dict] = None

        self._range_high        : Optional[float] = None
        self._range_low         : Optional[float] = None
        self._range_set         : bool  = False

        # 5-min candle buffer (BankNifty) — fed by on_candle()
        self._candles           : deque = deque(maxlen=60)

        self._spot              : Optional[float] = None
        self._daily_pnl         : float = 0.0
        self._trade_count       : int   = 0

        self._expiry_date       = None
        self._pm                = None
        self._instruments       = None

        self._active_token      : Optional[int] = None
        self._subscribed        : set   = set()

    # ─────────────────────────────────────────────────────────────────────────
    def pre_market(self, premarket_data, instruments) -> bool:
        """
        No INDEX_TOKEN → receives shared BankNifty pm + instruments.
        Returns True only on BankNifty monthly expiry day (DTE == 0).
        """
        self._pm          = premarket_data
        self._instruments = instruments
        self._expiry_date = premarket_data.expiry_date
        dte               = premarket_data.dte_days

        if dte != 0:
            log.info(
                f"[BANKNIFTY_EXPIRY_MOMENTUM] DTE={dte} — not expiry day, "
                "strategy inactive today"
            )
            return False

        self._active = True
        log.info(
            f"[BANKNIFTY_EXPIRY_MOMENTUM] ✓ EXPIRY DAY | "
            f"expiry={self._expiry_date} VIX={premarket_data.vix} "
            f"PCR={premarket_data.pcr}"
        )
        log.info(
            "[BANKNIFTY_EXPIRY_MOMENTUM] Phase 2: "
            "13:55 candle → range | 14:00 breakout entry | hard exit 15:20"
        )
        return True

    # ─────────────────────────────────────────────────────────────────────────
    def on_tick(self, price: float, ts: datetime, tick_ts: datetime):
        """BankNifty spot tick — drives breakout detection and trade management."""
        if not self._active:
            return
        self._spot = price

        t           = ts.time()
        entry_start = dtime(*CFG["entry_start"])
        cutoff_t    = dtime(*CFG["cutoff_time"])
        squareoff_t = dtime(*CFG["auto_squareoff"])

        # Auto squareoff
        if t >= squareoff_t and not self._squareoff_done:
            if self._trade:
                with self._lock:
                    if self._trade and not self._squareoff_done:
                        log.info("[BANKNIFTY_EXPIRY_MOMENTUM] Auto-squareoff 15:25")
                        self._close_trade(price, "AUTO_SQUAREOFF", ts)
            return

        # Hard cutoff 15:20
        if t >= cutoff_t and not self._squareoff_done:
            if self._trade:
                with self._lock:
                    if self._trade and not self._squareoff_done:
                        log.info("[BANKNIFTY_EXPIRY_MOMENTUM] Cutoff 15:20 — closing")
                        self._close_trade(price, "CUTOFF_EXIT", ts)
            self._squareoff_done = True
            return

        if t < entry_start:
            return

        # One trade only
        if self._entry_attempted or self._trade:
            return

        if not self._range_set:
            return

        # Breakout check
        if price > self._range_high:
            self._try_entry("CE", price, ts)
        elif price < self._range_low:
            self._try_entry("PE", price, ts)

    # ─────────────────────────────────────────────────────────────────────────
    def on_candle(self, candle: dict, ts: datetime):
        """
        BankNifty 5-min candle from MarketHub.
        Builds reference range from the 13:55 candle and feeds indicator buffer.
        """
        if not self._active:
            return
        self._candles.append(candle)

        candle_ts = candle.get("ts", ts)
        h = candle_ts.hour
        m = candle_ts.minute

        # Primary range source: 13:55 candle (closes at 14:00)
        if h == CFG["range_candle_hour"] and m == CFG["range_candle_min"]:
            self._range_high = candle["high"]
            self._range_low  = candle["low"]
            self._range_set  = True
            log.info(
                f"[BANKNIFTY_EXPIRY_MOMENTUM] 2PM range SET | "
                f"high={self._range_high:.1f} low={self._range_low:.1f}"
            )
            return

        # Fallback: keep updating from any candle between 13:45–13:59
        # until the 13:55 candle arrives and sets _range_set = True
        if not self._range_set and h == 13 and 45 <= m <= 59:
            self._range_high = candle["high"]
            self._range_low  = candle["low"]
            log.debug(
                f"[BANKNIFTY_EXPIRY_MOMENTUM] Range fallback update "
                f"high={self._range_high:.1f} low={self._range_low:.1f}"
            )

    # ─────────────────────────────────────────────────────────────────────────
    def on_option_tick(self, token: int, price: float,
                       ts: datetime, tick_ts: datetime = None):
        """Option tick — resolve pending entry or manage open trade."""
        # Pending entry: first tick after breakout signal
        if (self._pending_entry is not None
                and token == self._pending_entry["token"]
                and not self._trade):
            entry = self._pending_entry
            self._pending_entry = None
            if price >= CFG["min_ltp"]:
                self._execute_entry(
                    entry["token"], entry["symbol"],
                    entry["opt_type"], price, entry["ts"]
                )
            return

        # Manage open trade
        if self._active_token is None or token != self._active_token:
            return
        if not self._trade:
            return

        with self._lock:
            if self._trade:
                self._manage_trade(price, ts)

    # ─────────────────────────────────────────────────────────────────────────
    def _try_entry(self, opt_type: str, spot: float, ts: datetime):
        """Validate filters and queue directional entry."""
        self._entry_attempted = True

        if not self._ema_ok(opt_type):
            log.info(
                f"[BANKNIFTY_EXPIRY_MOMENTUM] EMA filter blocked {opt_type} entry"
            )
            return

        atr = self._atr()
        if atr is not None and atr < CFG["atr_min"]:
            log.info(
                f"[BANKNIFTY_EXPIRY_MOMENTUM] ATR={atr:.1f} < {CFG['atr_min']} — skip"
            )
            return

        atm = get_atm_strike(spot, step=100)
        tok, sym = self._instruments.get_option_token(atm, opt_type, self._expiry_date)

        if not tok:
            log.warning(
                f"[BANKNIFTY_EXPIRY_MOMENTUM] Token not found ATM={atm} {opt_type}"
            )
            return

        if tok not in self._subscribed:
            self.subscribe_option(tok)
            self._subscribed.add(tok)

        ltp = self.get_price(tok)

        if ltp is None or ltp < CFG["min_ltp"]:
            # Defer to first option tick
            self._pending_entry = {
                "token": tok, "symbol": sym, "opt_type": opt_type, "ts": ts
            }
            self._active_token = tok
            log.info(
                f"[BANKNIFTY_EXPIRY_MOMENTUM] Pending entry {sym} — waiting for tick"
            )
            return

        self._execute_entry(tok, sym, opt_type, ltp, ts)

    # ─────────────────────────────────────────────────────────────────────────
    def _execute_entry(self, token: int, symbol: str, opt_type: str,
                       ltp: float, ts: datetime):
        if self._trade:
            return

        if not self._acquire_slot():
            log.warning("[BANKNIFTY_EXPIRY_MOMENTUM] Slot busy — entry skipped")
            return

        mode   = "LIVE" if LIVE_MODE else "PAPER"
        result = self._place_buy(symbol, token, CFG["quantity"], ltp)

        if result is None:
            log.error(
                f"[BANKNIFTY_EXPIRY_MOMENTUM] BUY FAILED {symbol} — releasing slot"
            )
            self._release_slot()
            return

        order_id, fill = result
        sl     = fill * (1 - CFG["sl_pct"])
        target = fill * (1 + CFG["tgt_pct"])

        trade = _Trade(
            token=token, symbol=symbol, opt_type=opt_type,
            entry_price=fill, sl=sl, target=target,
            order_id=order_id, qty=CFG["quantity"], entry_time=ts,
        )

        with self._lock:
            self._trade        = trade
            self._active_token = token

        log.info(
            f"[BANKNIFTY_EXPIRY_MOMENTUM] ✓ TRADE OPEN [{mode}] | "
            f"{symbol}@{fill:.1f} SL={sl:.1f} TGT={target:.1f} qty={CFG['quantity']}"
        )

        _csv_append(CFG["entry_csv"], {
            "date"        : ts.strftime("%Y-%m-%d"),
            "time"        : ts.strftime("%H:%M:%S"),
            "mode"        : mode,
            "opt_type"    : opt_type,
            "symbol"      : symbol,
            "entry_price" : round(fill, 2),
            "sl"          : round(sl, 2),
            "target"      : round(target, 2),
            "qty"         : CFG["quantity"],
            "range_high"  : round(self._range_high or 0, 2),
            "range_low"   : round(self._range_low or 0, 2),
            "atr"         : round(self._atr() or 0, 2),
        })

    # ─────────────────────────────────────────────────────────────────────────
    def _manage_trade(self, ltp: float, ts: datetime):
        """SL / target / trail on every option tick. Caller holds lock."""
        trade = self._trade
        if trade is None:
            return

        if ltp >= trade.target:
            self._close_trade(ltp, "TARGET", ts)
            return

        effective_sl = trade.trail_sl if trade.trail_active else trade.sl
        if ltp <= effective_sl:
            reason = "TRAIL_SL" if trade.trail_active else "SL"
            self._close_trade(ltp, reason, ts)
            return

        # Arm trail when profit >= trail_arm_pct × entry
        if not trade.trail_active:
            if (ltp - trade.entry_price) >= trade.entry_price * CFG["trail_arm_pct"]:
                trade.trail_active = True
                trade.trail_sl     = trade.entry_price   # move SL to BE
                log.info(
                    f"[BANKNIFTY_EXPIRY_MOMENTUM] Trail ARMED | "
                    f"ltp={ltp:.1f} SL → BE={trade.entry_price:.1f}"
                )

        # Trail SL upward
        if trade.trail_active:
            new_sl = ltp - CFG["trail_step"]
            if new_sl > trade.trail_sl:
                trade.trail_sl = new_sl

    # ─────────────────────────────────────────────────────────────────────────
    def _close_trade(self, ltp: float, reason: str, ts: datetime):
        """Sell to close the open long position."""
        trade = self._trade
        if trade is None:
            return
        self._trade          = None
        self._squareoff_done = True

        result = self._place_sell_with_retry(
            trade.symbol, trade.token, trade.qty, ltp, max_retries=3
        )

        if result is None:
            log.error(
                f"[BANKNIFTY_EXPIRY_MOMENTUM] SELL FAILED {trade.symbol} | "
                "MANUAL SQUAREOFF NEEDED"
            )
            self._release_slot()
            return

        _, close_price = result
        pnl_pts = close_price - trade.entry_price
        pnl_rs  = pnl_pts * trade.qty

        self._daily_pnl   += pnl_rs
        self._trade_count += 1
        self._release_slot()

        mode = "LIVE" if LIVE_MODE else "PAPER"
        log.info(
            f"[BANKNIFTY_EXPIRY_MOMENTUM] TRADE CLOSED [{mode}] | "
            f"{trade.symbol} reason={reason} "
            f"entry={trade.entry_price:.1f} close={close_price:.1f} "
            f"pts={pnl_pts:+.1f} pnl={pnl_rs:+.0f}Rs daily={self._daily_pnl:+.0f}Rs"
        )

        _csv_append(CFG["exit_csv"], {
            "date"         : ts.strftime("%Y-%m-%d"),
            "time"         : ts.strftime("%H:%M:%S"),
            "mode"         : mode,
            "opt_type"     : trade.opt_type,
            "symbol"       : trade.symbol,
            "entry_price"  : round(trade.entry_price, 2),
            "close_price"  : round(close_price, 2),
            "reason"       : reason,
            "qty"          : trade.qty,
            "pnl_pts"      : round(pnl_pts, 2),
            "pnl_rs"       : round(pnl_rs, 2),
            "daily_pnl"    : round(self._daily_pnl, 2),
            "trail_active" : trade.trail_active,
        })

    # ─────────────────────────────────────────────────────────────────────────
    def _ema_ok(self, opt_type: str) -> bool:
        closes = [c["close"] for c in self._candles]
        if len(closes) < CFG["min_bars_ema"]:
            log.debug(
                f"[BANKNIFTY_EXPIRY_MOMENTUM] EMA bypassed ({len(closes)} bars)"
            )
            return True

        arr = np.array(closes, dtype=float)

        def _ema(period):
            k, e = 2 / (period + 1), arr[0]
            for v in arr[1:]:
                e = v * k + e * (1 - k)
            return e

        fast = _ema(CFG["ema_fast"])
        slow = _ema(CFG["ema_slow"])
        ok   = fast > slow if opt_type == "CE" else fast < slow

        log.debug(
            f"[BANKNIFTY_EXPIRY_MOMENTUM] EMA{CFG['ema_fast']}={fast:.1f} "
            f"EMA{CFG['ema_slow']}={slow:.1f} {opt_type} ok={ok}"
        )
        return ok

    def _atr(self) -> Optional[float]:
        period  = 14
        candles = list(self._candles)
        if len(candles) < period + 1:
            return None
        trs = [
            max(
                candles[i]["high"] - candles[i]["low"],
                abs(candles[i]["high"] - candles[i-1]["close"]),
                abs(candles[i]["low"]  - candles[i-1]["close"]),
            )
            for i in range(1, len(candles))
        ]
        return float(np.mean(trs[-period:]))

    # ─────────────────────────────────────────────────────────────────────────
    def eod_summary(self):
        log.info(
            f"[BANKNIFTY_EXPIRY_MOMENTUM] ══ EOD ══ "
            f"trades={self._trade_count} daily_pnl={self._daily_pnl:+.0f}Rs"
        )
        if self._trade and not self._squareoff_done:
            log.warning(
                "[BANKNIFTY_EXPIRY_MOMENTUM] ⚠ Trade open at EOD — force closing"
            )
            with self._lock:
                if self._trade:
                    ltp = self.get_price(self._trade.token) or self._trade.entry_price
                    self._close_trade(ltp, "EOD_FORCE_CLOSE", _now_ist())
