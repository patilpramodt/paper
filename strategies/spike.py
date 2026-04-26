"""
strategies/spike.py

SPIKE Strategy — 9:15 gap/spike trade, exits by 9:30.

LIVE / PAPER MODE
─────────────────
Set LIVE_MODE = True below to enable real orders for this strategy.
All other strategies remain in paper mode until their own flag is changed.

ORDER EXECUTION (live mode)
────────────────────────────
  Entry : REGULAR + MARKET + MIS via OrderRouter.place_buy()
  Exit  : REGULAR + MARKET + MIS via OrderRouter.place_sell()

  NO exchange SL orders are placed after entry.
  Reason: SL-M is not available for options on NSE/Zerodha.
          SL Limit can gap through the trigger entirely.
          We monitor the option price on every WebSocket tick (on_option_tick)
          and fire a MARKET sell the moment the software SL is breached.
          This is faster and more reliable than any exchange SL for options.

TRAILING SL (software, tick-by-tick)
──────────────────────────────────────
  Managed entirely in on_option_tick() on every WebSocket tick.
  trail_trigger_pts : profit needed before trailing starts
  trail_distance    : SL trails this many pts below highest_seen
  Both live and paper mode use identical trail logic.

FIXES APPLIED (unchanged from previous version)
  - pre_market() subscribes ATM CE+PE even when prev_close is None.
  - on_tick() subscribes ATM options on very first market tick if
    pre-subscription failed.
  - Pending entry mechanism for stale/missing pre-9:15 option prices.
"""

import csv
import logging
import os
import threading
from datetime import datetime, time as dtime, timedelta, timezone
from typing import Optional

from core.base_strategy import BaseStrategy
from core.candle import SecondCandleBuilder

log = logging.getLogger("strategy.spike")

_IST = timezone(timedelta(hours=5, minutes=30))

def _now_ist() -> datetime:
    return datetime.now(tz=_IST).replace(tzinfo=None)

# ─────────────────────────────────────────────────────────────────────────────
#  LIVE MODE FLAG
#  Change to True when you are ready to trade SPIKE with real money.
#  All other strategies remain in paper mode until their own flag is changed.
# ─────────────────────────────────────────────────────────────────────────────
LIVE_MODE = False

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CFG = {
    "quantity"               : 35,   # BankNifty lot size as of Nov 2024
    "start_time"             : dtime(9, 15),
    "spike_exit_time"        : dtime(9, 30),
    "close_time"             : dtime(15, 15),
    "initial_sl_buffer"      : 120,
    "trail_trigger_pts"      : 50,
    "trail_distance"         : 25,
    "doji_threshold"         : 0.10,
    "bucket_sec"             : 8,
    "min_candles_before_mom" : 2,
    "csv_file"               : "spike_trades.csv",
}


class SpikeStrategy(BaseStrategy):

    # expose the module-level flag as a class attribute so BaseStrategy helpers work
    LIVE_MODE = LIVE_MODE

    @property
    def name(self) -> str:
        return "SPIKE"

    def __init__(self, market_hub):
        super().__init__(market_hub)

        self._index_8s     = SecondCandleBuilder(seconds=CFG["bucket_sec"])
        self._opt_8s       = None

        self._gap_direction  : Optional[str] = None
        self._gap_filter_done: bool = False
        self._market_opened  : bool = False

        self._pre_ce_token   : Optional[int] = None
        self._pre_pe_token   : Optional[int] = None
        self._pre_ce_sym     : Optional[str] = None
        self._pre_pe_sym     : Optional[str] = None

        self._trade          = None
        self._trade_done     : bool = False
        self._today_pnl      : float = 0.0
        self._completed      : list  = []
        self._pending_entry  = None

        self._prev_body_high : Optional[float] = None
        self._prev_body_low  : Optional[float] = None
        self._prev_last5m_high  : Optional[float] = None
        self._prev_last5m_low   : Optional[float] = None
        self._prev_last5m_close : Optional[float] = None
        self._expiry_date    = None
        self._instruments    = None

        self._lock           = threading.Lock()

        mode_tag = "[LIVE]" if LIVE_MODE else "[PAPER]"
        log.info(f"[SPIKE] Initialized in {mode_tag} mode")

    # ── Pre-market ────────────────────────────────────────────────────────────

    def pre_market(self, pm, instruments) -> bool:
        from core.instruments import get_atm_strike

        now = _now_ist().time()
        if now >= CFG["spike_exit_time"]:
            log.warning(f"[{self.name}] Started after spike window — skipping today.")
            self._trade_done = True
            return True

        self._instruments       = instruments
        self._prev_body_high    = pm.prev_body_high
        self._prev_body_low     = pm.prev_body_low
        self._prev_last5m_high  = pm.prev_last5m_high
        self._prev_last5m_low   = pm.prev_last5m_low
        self._prev_last5m_close = pm.prev_last5m_close
        self._expiry_date       = pm.expiry_date

        log.info(f"[{self.name}] Pre-market | "
                 f"body=[{self._prev_body_low}  {self._prev_body_high}] "
                 f"prev_close={pm.prev_close} | mode={'LIVE' if LIVE_MODE else 'PAPER'}")

        ref_price = pm.prev_close or pm.prev_last5m_close

        if ref_price is None:
            log.warning(f"[{self.name}] No prev_close and no prev_last5m_close — "
                        f"cannot pre-subscribe options. Will attempt on first tick.")
            return True

        strike = get_atm_strike(ref_price)
        log.info(f"[{self.name}] Token lookup | strike={strike} "
                 f"expiry={pm.expiry_date} ref_price={ref_price:.2f} "
                 f"(source={'prev_close' if pm.prev_close else 'prev_last5m_close'})")

        ce_tok, ce_sym = instruments.get_option_token(strike, "CE", pm.expiry_date)
        pe_tok, pe_sym = instruments.get_option_token(strike, "PE", pm.expiry_date)

        self._pre_ce_token = ce_tok
        self._pre_ce_sym   = ce_sym
        self._pre_pe_token = pe_tok
        self._pre_pe_sym   = pe_sym

        if ce_tok:
            self.subscribe_option(ce_tok)
            log.info(f"[{self.name}] Pre-subscribed CE: {ce_sym} ({ce_tok})")
        else:
            log.error(f"[{self.name}] CE token not found | strike={strike} expiry={pm.expiry_date}")

        if pe_tok:
            self.subscribe_option(pe_tok)
            log.info(f"[{self.name}] Pre-subscribed PE: {pe_sym} ({pe_tok})")
        else:
            log.error(f"[{self.name}] PE token not found | strike={strike} expiry={pm.expiry_date}")

        return True

    # ── Tick handlers ─────────────────────────────────────────────────────────

    def on_tick(self, price: float, ts: datetime, tick_ts: datetime):
        t = ts.time()

        if t < CFG["start_time"] or t > CFG["close_time"]:
            return

        if not self._market_opened and t >= CFG["start_time"]:
            self._market_opened = True
            log.info(f"[{self.name}] Market open tick received: {price:.2f}")

            if self._pre_ce_token is None or self._pre_pe_token is None:
                self._subscribe_atm_on_open(price)

        closed_8s = self._index_8s.feed_tick(price, tick_ts)

        if not self._gap_filter_done and self._market_opened:
            self._determine_gap_direction(price)
            self._gap_filter_done = True

            if self._gap_direction in ("CE", "PE"):
                self._attempt_gap_entry(price, ts)
            return

        if (self._gap_direction == "BOTH" and
                not self._trade_done and
                self._trade is None and
                closed_8s is not None):
            self._check_2candle_signal(closed_8s, price, ts)

        if self._trade and self._trade["state"] == "OPEN":
            if t >= CFG["spike_exit_time"]:
                opt_price = self.get_price(self._trade["token"]) or self._trade["entry"]
                self._do_exit(opt_price, "SPIKE_WINDOW_END", ts)
                return

    def on_candle(self, candle: dict, ts: datetime):
        pass

    def on_option_tick(self, token: int, price: float, ts: datetime, tick_ts: datetime = None):
        """
        Option price arrives on every WebSocket tick.

        Two jobs here:
          1. Resolve pending gap entry (if option had no valid price at signal time).
          2. Software trailing SL management — check on EVERY tick.
             When SL breached: fire MARKET sell immediately (no exchange SL order).
        """
        # 1. Resolve pending gap entry
        if self._pending_entry and token == self._pending_entry["token"] and not self._trade:
            p = self._pending_entry
            self._pending_entry = None
            log.info(f"[{self.name}] Pending entry resolved — first live tick for {p['sym']} "
                     f"@ {price:.2f} (was stale at entry signal time)")
            self._build_entry(p["sym"], p["token"], p["signal"], ts, p["reason"])
            return

        # 2. Software trailing SL management (runs on every tick while trade open)
        if not (self._trade and token == self._trade.get("token")):
            return

        if self._opt_8s is None:
            self._opt_8s = SecondCandleBuilder(seconds=CFG["bucket_sec"])

        use_ts = tick_ts if tick_ts else ts
        self._opt_8s.feed_tick(price, use_ts)

        if self._trade["state"] != "OPEN":
            return

        # Track highest price seen for trailing calculation
        if price > self._trade["highest_seen"]:
            self._trade["highest_seen"] = price

        new_sl = self._compute_trailing_sl(
            self._trade["entry"], self._trade["highest_seen"], self._trade["sl"]
        )
        if new_sl > self._trade["sl"]:
            log.info(f"[{self.name}] TSL: {self._trade['sl']:.0f} → {new_sl:.0f} "
                     f"(highest={self._trade['highest_seen']:.0f})")
            self._trade["sl"] = new_sl

        # SL breached — MARKET sell immediately
        # (No exchange SL order is ever placed: SL-M not available for options,
        #  SL limit can be skipped. Software monitoring on every tick is superior.)
        if price <= self._trade["sl"]:
            self._do_exit(price, "SL_HIT", ts)

    # ── Subscribe ATM on open ─────────────────────────────────────────────────

    def _subscribe_atm_on_open(self, open_price: float):
        from core.instruments import get_atm_strike
        strike = get_atm_strike(open_price)
        expiry = self._expiry_date
        log.info(f"[{self.name}] Late-subscribing ATM options on open tick | "
                 f"strike={strike} expiry={expiry} open={open_price:.2f}")

        ce_tok, ce_sym = self._instruments.get_option_token(strike, "CE", expiry)
        pe_tok, pe_sym = self._instruments.get_option_token(strike, "PE", expiry)

        if ce_tok and self._pre_ce_token is None:
            self.subscribe_option(ce_tok)
            self._pre_ce_token = ce_tok
            self._pre_ce_sym   = ce_sym
            log.info(f"[{self.name}] Late-subscribed CE: {ce_sym} ({ce_tok})")

        if pe_tok and self._pre_pe_token is None:
            self.subscribe_option(pe_tok)
            self._pre_pe_token = pe_tok
            self._pre_pe_sym   = pe_sym
            log.info(f"[{self.name}] Late-subscribed PE: {pe_sym} ({pe_tok})")

    # ── Gap direction ─────────────────────────────────────────────────────────

    def _determine_gap_direction(self, open_price: float):
        h5 = self._prev_last5m_high
        l5 = self._prev_last5m_low

        if h5 is not None and l5 is not None:
            log.info(f"[{self.name}] Gap ref → prev last 5-min: "
                     f"H={h5:.0f}  L={l5:.0f}  Today open={open_price:.0f}")
            if open_price > h5:
                self._gap_direction = "CE"
                log.info(f"[{self.name}]  GAP UP: open={open_price:.0f} > last5m_high={h5:.0f}  → CE")
            elif open_price < l5:
                self._gap_direction = "PE"
                log.info(f"[{self.name}]  GAP DOWN: open={open_price:.0f} < last5m_low={l5:.0f}  → PE")
            else:
                self._gap_direction = "BOTH"
                log.info(f"[{self.name}]  NO GAP: open inside [{l5:.0f}–{h5:.0f}] → 2-candle signal")
            return

        log.warning(f"[{self.name}] Last 5-min candle data unavailable — falling back to daily body")
        bh, bl = self._prev_body_high, self._prev_body_low

        if bh is None or bl is None:
            self._gap_direction = "BOTH"
            log.warning(f"[{self.name}] No gap reference at all — defaulting to BOTH")
            return

        if open_price > bh:
            self._gap_direction = "CE"
            log.info(f"[{self.name}]  GAP UP (fallback): open={open_price:.0f} > body_high={bh:.0f}")
        elif open_price < bl:
            self._gap_direction = "PE"
            log.info(f"[{self.name}]  GAP DOWN (fallback): open={open_price:.0f} < body_low={bl:.0f}")
        else:
            self._gap_direction = "BOTH"
            log.info(f"[{self.name}]  NO GAP (fallback): open inside [{bl:.0f}–{bh:.0f}]")

    def _attempt_gap_entry(self, index_price: float, ts: datetime):
        if self._trade_done or self._trade:
            return

        signal = self._gap_direction

        if signal == "CE" and self._pre_ce_token:
            sym, token = self._pre_ce_sym, self._pre_ce_token
        elif signal == "PE" and self._pre_pe_token:
            sym, token = self._pre_pe_sym, self._pre_pe_token
        else:
            from core.instruments import get_atm_strike
            strike = get_atm_strike(index_price)
            expiry = self._expiry_date
            log.warning(f"[{self.name}] Gap entry fallback lookup | signal={signal} "
                        f"strike={strike} expiry={expiry} spot={index_price:.2f}")
            token, sym = self._instruments.get_option_token(strike, signal, expiry)

        if not token or not sym:
            log.error(f"[{self.name}] No option token for gap {signal} entry | "
                      f"spot={index_price:.2f} expiry={self._expiry_date}")
            return

        self._build_entry(sym, token, signal, ts, reason=f"gap_{signal.lower()}")

    # ── 2-candle signal ───────────────────────────────────────────────────────

    def _check_2candle_signal(self, latest_closed: dict, index_price: float, ts: datetime):
        last_two = self._index_8s.last_n_closed(2)
        if len(last_two) < 2:
            return

        c1, c2 = last_two[-2], last_two[-1]
        signal = self._check_signal(c1, c2)
        if not signal:
            return

        log.info(f"[{self.name}] 2-candle signal: {signal} at {ts.strftime('%H:%M:%S')}")

        if signal == "CE" and self._pre_ce_token:
            sym, token = self._pre_ce_sym, self._pre_ce_token
        elif signal == "PE" and self._pre_pe_token:
            sym, token = self._pre_pe_sym, self._pre_pe_token
        else:
            from core.instruments import get_atm_strike
            strike = get_atm_strike(index_price)
            expiry = self._expiry_date
            log.info(f"[{self.name}] Fallback token lookup | signal={signal} "
                     f"strike={strike} expiry={expiry}")
            token, sym = self._instruments.get_option_token(strike, signal, expiry)

        if not token or not sym:
            log.error(f"[{self.name}] No token for {signal} — trade SKIPPED")
            return

        self.subscribe_option(token)
        self._build_entry(sym, token, signal, ts, reason="2x8s_signal")

    # ── Entry ─────────────────────────────────────────────────────────────────

    def _build_entry(self, sym: str, token: int, signal: str, ts: datetime, reason: str):
        """
        Attempt to enter a position.

        Price guard: if option has no valid post-9:15 price yet, store a
        pending entry and return. on_option_tick() resolves it on first tick.

        Live mode: places MARKET buy via OrderRouter.
        Paper mode: simulates fill at current LTP.

        NO exchange SL order is placed after entry. Trailing SL is managed
        purely in software via on_option_tick() on every WebSocket tick.
        """
        opt_price = self.get_price(token)
        price_ts  = self.get_price_ts(token)
        market_open_today = ts.replace(hour=9, minute=15, second=0, microsecond=0)

        if (not opt_price or opt_price <= 0) or (price_ts is None or price_ts < market_open_today):
            log.warning(
                f"[{self.name}] No valid post-9:15 price for {sym} "
                f"(price={opt_price} priced_at={price_ts}) — storing pending entry"
            )
            self._pending_entry = {
                "sym": sym, "token": token, "signal": signal, "ts": ts, "reason": reason
            }
            return

        # Try to acquire trade slot (live mode only — paper always gets True)
        if not self._acquire_slot():
            log.warning(f"[{self.name}] Trade slot blocked — another live strategy has a position")
            return

        # Place BUY order (paper: simulated fill, live: MARKET order)
        order_id = self._place_buy(sym, token, CFG["quantity"], opt_price)
        if order_id is None:
            # Live order failed — release slot and abort
            self._release_slot()
            log.error(f"[{self.name}] BUY order FAILED for {sym} — entry aborted")
            return

        sl = opt_price - CFG["initial_sl_buffer"]
        self._opt_8s = SecondCandleBuilder(seconds=CFG["bucket_sec"])
        self._trade = {
            "state"       : "OPEN",
            "symbol"      : sym,
            "token"       : token,
            "signal"      : signal,
            "entry"       : opt_price,
            "sl"          : sl,
            "highest_seen": opt_price,
            "entry_time"  : ts,
            "gap_direction": self._gap_direction,
            "order_id"    : order_id,
            "qty"         : CFG["quantity"],
        }

        mode_tag = "LIVE" if LIVE_MODE else "PAPER"
        log.info(f"[{self.name}] [{mode_tag}] ENTRY {sym} @ {opt_price:.0f} | "
                 f"SL={sl:.0f} | Trail kicks at {opt_price + CFG['trail_trigger_pts']:.0f} "
                 f"| Reason={reason} | order_id={order_id}")

        self._log_csv({
            "timestamp"    : ts.strftime("%Y-%m-%d %H:%M:%S"),
            "symbol"       : sym,
            "action"       : "ENTRY",
            "price"        : opt_price,
            "sl"           : sl,
            "status"       : "OPEN",
            "pnl"          : 0,
            "reason"       : reason,
            "gap_direction": self._gap_direction,
            "mode"         : mode_tag,
            "order_id"     : order_id,
        })

    # ── Exit ──────────────────────────────────────────────────────────────────

    def _do_exit(self, exit_price: float, reason: str, ts: datetime):
        """
        Close the open position.

        Live mode: places MARKET sell via OrderRouter, then releases slot.
        Paper mode: simulates fill at exit_price.

        In both modes the SL is managed in software only — no exchange SL
        order was placed at entry, so no cancellation is needed here.
        """
        t = self._trade
        if not t or t["state"] != "OPEN":
            return
        t["state"] = "CLOSED"

        # Place SELL order (paper: simulated, live: MARKET order)
        sell_price = exit_price
        order_id = self._place_sell(t["symbol"], t["token"], t["qty"], exit_price)
        if LIVE_MODE and order_id is None:
            # Live order failed — log loudly, still mark trade closed in software
            log.error(f"[{self.name}] SELL order FAILED for {t['symbol']} — "
                      f"trade marked closed in software but position may still be open in Zerodha! "
                      f"Check and square off manually.")

        # Release global trade slot
        self._release_slot()

        pnl = (sell_price - t["entry"]) * t["qty"]
        self._today_pnl += pnl
        self._trade_done = True

        mode_tag = "LIVE" if LIVE_MODE else "PAPER"
        log.info(f"[{self.name}] [{mode_tag}] EXIT [{reason}] {t['symbol']} @ {sell_price:.0f} "
                 f"| PnL {pnl:.0f} | Today {self._today_pnl:.0f} | order_id={order_id}")

        self._log_csv({
            "timestamp"    : ts.strftime("%Y-%m-%d %H:%M:%S"),
            "symbol"       : t["symbol"],
            "action"       : "EXIT",
            "price"        : sell_price,
            "sl"           : t["sl"],
            "status"       : "CLOSED",
            "pnl"          : round(pnl, 2),
            "reason"       : reason,
            "gap_direction": t["gap_direction"],
            "mode"         : mode_tag,
            "order_id"     : order_id,
        })
        self._completed.append({**t, "exit_price": sell_price, "exit_reason": reason, "pnl": pnl})

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _check_signal(self, c1: dict, c2: dict) -> Optional[str]:
        if self._is_doji(c1) or self._is_doji(c2):
            return None
        if c1["close"] > c1["open"] and c2["close"] > c2["open"]:
            return "CE"
        if c1["close"] < c1["open"] and c2["close"] < c2["open"]:
            return "PE"
        return None

    @staticmethod
    def _is_doji(c: dict) -> bool:
        rng = c["high"] - c["low"]
        if rng == 0:
            return True
        return abs(c["close"] - c["open"]) / rng < CFG["doji_threshold"]

    @staticmethod
    def _compute_trailing_sl(entry: float, highest: float, current_sl: float) -> float:
        profit = highest - entry
        if profit >= CFG["trail_trigger_pts"]:
            new_sl = highest - CFG["trail_distance"]
            return max(new_sl, current_sl)
        return current_sl

    def _log_csv(self, row: dict):
        fname  = CFG["csv_file"]
        exists = os.path.isfile(fname)
        fields = ["timestamp", "symbol", "action", "price",
                  "sl", "status", "pnl", "reason", "gap_direction", "mode", "order_id"]
        with open(fname, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if not exists:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in fields})

    def eod_summary(self):
        log.info(f"\n[{self.name}] {'='*50}")
        log.info(f"[{self.name}] END OF DAY | mode={'LIVE' if LIVE_MODE else 'PAPER'}")
        log.info(f"[{self.name}] Gap direction  : {self._gap_direction}")
        log.info(f"[{self.name}] Trade executed : {'Yes' if self._trade_done else 'No'}")
        for t in self._completed:
            log.info(f"[{self.name}]   {t['symbol']} {t['exit_reason']} "
                     f"entry={t['entry']:.0f} exit={t['exit_price']:.0f} PnL={t['pnl']:.0f}")
        log.info(f"[{self.name}] Today PnL      : {self._today_pnl:.0f}")
        log.info(f"[{self.name}] {'='*50}\n")
