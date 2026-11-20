"""
strategies/heikin_ashi_atm_strategy.py

HA5_ATM — 5-minute Heikin Ashi colour strategy, buys ATM CE / PE.
Two instances: NIFTY_HA5_ATM (Nifty 50) and BANKNIFTY_HA5_ATM (BankNifty).

RULES (as given)
────────────────
  Candles   5-min Heikin Ashi built from index ticks. No indicators/filters.
              HA_close = (O + H + L + C) / 4
              HA_open  = (prev HA_open + prev HA_close) / 2   (first bar: (O + C) / 2)
              GREEN = HA_close > HA_open, RED = HA_close < HA_open
  Entry     Two consecutive completed GREEN HA candles -> BUY ATM CE.
            Two consecutive completed RED   HA candles -> BUY ATM PE.
            Entered when the second candle closes; ATM from the index price
            at that moment.
  Exit      CE: the next completed HA candle that is RED.
            PE: the next completed HA candle that is GREEN.
            Only completed-candle colour is used (no SL / TP / trailing).
  Position  One at a time. No pyramiding, no averaging, no auto-reverse.
            After an exit, a fresh two-candle signal is required.

ASSUMPTIONS (not in the spec — flag if wrong)
─────────────────────────────────────────────
  - "Fresh" signal: the exit candle does NOT count as the first candle of
    the next signal; both candles must close after the exit
    (CFG["exit_candle_counts"] = False). So a CE exit on a red candle
    needs two MORE red candles before a PE.
  - A signal fires only on the candle that makes the streak exactly 2.
    A third same-colour candle does not trigger a late entry.
  - A doji (HA_close == HA_open) has no colour: it breaks a streak and
    does not trigger an exit.
  - Session: HA chain starts fresh each day at the 09:15 candle. No new
    entries from a candle closing after last_entry_time (15:00). Any open
    position is squared off at close_time (15:15).
  - Nearest expiry from pre-market (Nifty weekly, BankNifty monthly),
    including on expiry day. Qty = CFG lots x lot size from the NFO dump.
  - Restart mid-session: today's 5-min candles are replayed from the hub
    backfill to rebuild the HA chain, but the streak is reset at go-live
    (no entering on a signal that closed while the bot was down), and the
    first live candle is partial, so it is treated like a doji.
  - Paper fills are the option LTP at the moment the candle close is seen
    (first tick of the next bar). costs = core round-trip cost model.

INDEX ROUTING
─────────────
  Nifty: INDEX_TOKEN = 256265. BankNifty: no INDEX_TOKEN (main hub index).
  Both build their own 5-min candles from on_tick using the exchange tick
  timestamp; the hub's live on_candle is ignored and on_candle is used only
  for the pre-open backfill replay.
"""

import logging
from datetime import datetime, time as dtime, timedelta, timezone
from typing import Optional

from core.base_strategy import BaseStrategy
from core.candle import CandleBuilder
from core.csv_log import append_row, round_trip_cost_rs
from core.instruments import get_atm_strike

_IST = timezone(timedelta(hours=5, minutes=30))

# ─────────────────────────────────────────────────────────────────────────────
#  LIVE MODE FLAG — paper only. Change to True only when ready for real money.
# ─────────────────────────────────────────────────────────────────────────────
LIVE_MODE = False

BASE_CFG = {
    "enabled"            : True,
    "candle_minutes"     : 5,
    "start_time"         : dtime(9, 15),
    "last_entry_time"    : dtime(15, 0),   # signal candle must CLOSE by this
    "close_time"         : dtime(15, 15),  # square off
    "exit_candle_counts" : False,          # see ASSUMPTIONS
    "lots"               : 1,
    "partial_bar_sec"    : 15,    # first live tick later than this into a bar -> partial
    "max_price_age_sec"  : 5,     # never fill on an option price older than this
    "pending_timeout_sec": 15,    # drop a pending entry with no live tick by then
    "sell_retry_sec"     : 30,    # LIVE only: retry a failed exit this often
}


def _color(ha_open: float, ha_close: float) -> Optional[str]:
    if ha_close > ha_open:
        return "GREEN"
    if ha_close < ha_open:
        return "RED"
    return None


class _HeikinAshiATMBase(BaseStrategy):

    LIVE_MODE    = LIVE_MODE
    STRIKE_STEP  = 50
    LOGGER       = "strategy.ha5_atm"
    CSV_FILE     = "ha5_atm_trades.csv"
    CANDLES_CSV  = "ha5_atm_candles.csv"
    NAME         = "HA5_ATM"

    TRADE_FIELDS = [
        "date", "trade_no", "side", "symbol", "strike", "expiry", "qty",
        "signal_c1", "signal_c2", "entry_time", "entry_spot", "entry_premium",
        "exit_candle", "exit_time", "exit_spot", "exit_premium", "exit_reason",
        "bars_held", "gross_pnl", "costs", "net_pnl", "mode", "order_id", "exit_order_id",
    ]
    CANDLE_FIELDS = [
        "date", "ts", "open", "high", "low", "close",
        "ha_open", "ha_high", "ha_low", "ha_close", "color", "partial",
        "streak", "action",
    ]

    def __init__(self, market_hub):
        super().__init__(market_hub)
        self.cfg  = dict(BASE_CFG)
        self.log  = logging.getLogger(self.LOGGER)
        self._cb  = CandleBuilder(minutes=self.cfg["candle_minutes"])

        self._instruments = None
        self._expiry      = None
        self._enabled     = False

        # HA chain
        self._prev_ha      = None    # (ha_open, ha_close) of last completed bar
        self._last_seed_ts = None    # last backfilled bar start
        self._live         = False
        self._partial_bar  = None    # bar start ts of a partial first live bar

        # Signal state
        self._streak_color = None
        self._streak_n     = 0
        self._streak_bars  = []      # bar start ts of the candles in the streak

        # Position state
        self._trade        = None
        self._pending      = None
        self._next_retry   = None
        self._trades_today = 0
        self._today_gross  = 0.0
        self._today_net    = 0.0
        self._completed    = []
        self._last_spot    = None

        self.log.info(
            f"[{self.name}] Initialized [{'LIVE' if LIVE_MODE else 'PAPER'}] | "
            f"{self.cfg['candle_minutes']}m Heikin Ashi, 2 same-colour -> ATM buy, "
            f"exit on opposite colour | lots={self.cfg['lots']}"
        )

    @property
    def name(self) -> str:
        return self.NAME

    # ── Pre-market ────────────────────────────────────────────────────────────

    def pre_market(self, pm, instruments) -> bool:
        if not self.cfg["enabled"]:
            self.log.info(f"[{self.name}] disabled via CFG")
            return False
        self._instruments = instruments
        self._expiry      = pm.expiry_date
        if self._expiry is None:
            self.log.error(f"[{self.name}] no expiry from pre-market — skipping today")
            return False
        self._enabled = True
        self.log.info(f"[{self.name}] Pre-market | expiry={self._expiry} "
                      f"mode={'LIVE' if LIVE_MODE else 'PAPER'}")
        return True

    # ── Backfill (restart) ────────────────────────────────────────────────────

    def on_candle(self, candle: dict, ts: datetime):
        # Live candles are built in on_tick; the hub only matters for backfill.
        if self._live or not self._enabled:
            return
        if candle["ts"].time() < self.cfg["start_time"]:
            return
        ha_o, ha_h, ha_l, ha_c = self._ha(candle)
        self._prev_ha      = (ha_o, ha_c)
        self._last_seed_ts = candle["ts"]
        self._log_candle(candle, (ha_o, ha_h, ha_l, ha_c), _color(ha_o, ha_c),
                         partial=False, action="BACKFILL")

    # ── Ticks ─────────────────────────────────────────────────────────────────

    def on_tick(self, price: float, ts: datetime, tick_ts: datetime):
        if not self._enabled:
            return
        t_ex = tick_ts or ts
        if t_ex.time() < self.cfg["start_time"]:
            return
        self._last_spot = price

        if not self._live:
            self._go_live(t_ex)

        closed = self._cb.feed_tick(price, 1, t_ex)
        if closed is not None:
            self._on_closed_bar(closed, price, ts)

        # Square-off
        if self._trade and ts.time() >= self.cfg["close_time"]:
            if not self._trade.get("_exit_in_progress") and (
                    not self._trade.get("_exit_failed") or ts >= (self._next_retry or ts)):
                self._exit(price, "EOD_CLOSE", ts, exit_candle=None)
            return

        # LIVE only: a failed exit is retried periodically
        if self._trade and self._trade.get("_exit_failed") and self._next_retry and ts >= self._next_retry:
            self._exit(price, self._trade["_exit_failed"], ts, exit_candle=self._trade.get("_exit_candle"))

        # Pending entry with no live option price
        if self._pending and (ts - self._pending["ts"]).total_seconds() > self.cfg["pending_timeout_sec"]:
            p, self._pending = self._pending, None
            self.unsubscribe_option(p["token"])
            self.log.warning(f"[{self.name}] pending entry {p['sym']} timed out — no fresh option price, skipped")

    def _go_live(self, t_ex: datetime):
        self._live = True
        bar = self._cb._bar_start(t_ex)
        if (t_ex - bar).total_seconds() > self.cfg["partial_bar_sec"]:
            self._partial_bar = bar
        if self._last_seed_ts is not None:
            gap = (bar - self._last_seed_ts) > timedelta(minutes=self.cfg["candle_minutes"])
            self.log.info(
                f"[{self.name}] Restart: HA chain seeded up to {self._last_seed_ts:%H:%M} "
                f"| first live bar {bar:%H:%M}{' (partial)' if self._partial_bar else ''}"
                f"{' — GAP in chain, HA open slightly off for a few bars' if gap else ''} "
                f"| streak reset, waiting for a fresh signal"
            )
        elif self._partial_bar:
            self.log.warning(f"[{self.name}] Started mid-bar with no backfill — first bar {bar:%H:%M} is partial")

    # ── Bar close ─────────────────────────────────────────────────────────────

    def _ha(self, c: dict):
        ha_c = (c["open"] + c["high"] + c["low"] + c["close"]) / 4.0
        if self._prev_ha is None:
            ha_o = (c["open"] + c["close"]) / 2.0
        else:
            ha_o = (self._prev_ha[0] + self._prev_ha[1]) / 2.0
        return ha_o, max(c["high"], ha_o, ha_c), min(c["low"], ha_o, ha_c), ha_c

    def _on_closed_bar(self, c: dict, spot: float, ts: datetime):
        if self._last_seed_ts is not None and c["ts"] <= self._last_seed_ts:
            return   # already seeded from backfill
        ha = self._ha(c)
        self._prev_ha = (ha[0], ha[3])
        partial = (c["ts"] == self._partial_bar)
        color   = None if partial else _color(ha[0], ha[3])
        bar_close_time = (c["ts"] + timedelta(minutes=self.cfg["candle_minutes"])).time()
        action = ""

        # 1) Exit on an opposite-colour completed candle
        if self._trade and not self._trade.get("_exit_in_progress") and not self._trade.get("_exit_failed"):
            want = "RED" if self._trade["side"] == "CE" else "GREEN"
            if color == want:
                action = f"EXIT_{self._trade['side']}"
                self._exit(spot, "HA_COLOR_FLIP", ts, exit_candle=c["ts"])
                if not self.cfg["exit_candle_counts"]:
                    self._reset_streak()
                    self._log_candle(c, ha, color, partial, action)
                    return
            else:
                self._trade["bars_held"] += 1

        # 2) Streak (counted only while flat, and never across a partial bar)
        if self._trade is None and self._pending is None:
            if color is None:
                self._reset_streak()
            elif color == self._streak_color:
                self._streak_n += 1
                self._streak_bars.append(c["ts"])
            else:
                self._streak_color, self._streak_n, self._streak_bars = color, 1, [c["ts"]]

            if self._streak_n == 2:
                side = "CE" if color == "GREEN" else "PE"
                if bar_close_time <= self.cfg["last_entry_time"]:
                    action = (action + " " if action else "") + f"SIGNAL_{side}"
                    self._enter(side, spot, ts, list(self._streak_bars))
                else:
                    action = (action + " " if action else "") + f"SIGNAL_{side}_AFTER_CUTOFF"

        self._log_candle(c, ha, color, partial, action)

    def _reset_streak(self):
        self._streak_color, self._streak_n, self._streak_bars = None, 0, []

    # ── Entry ─────────────────────────────────────────────────────────────────

    def _enter(self, side: str, spot: float, ts: datetime, bars: list):
        strike = get_atm_strike(spot, step=self.STRIKE_STEP)
        token, sym = self._instruments.get_option_token(strike, side, self._expiry)
        if not token:
            self.log.error(f"[{self.name}] no {side} contract strike={strike} expiry={self._expiry} — skipped")
            return
        self.subscribe_option(token)
        sig = {"sym": sym, "token": token, "side": side, "strike": strike,
               "spot": spot, "ts": ts, "bars": bars}
        if not self._try_fill(sig, ts):
            self._pending = sig
            self.log.info(f"[{self.name}] {side} signal — waiting for a live tick on {sym}")

    def _try_fill(self, sig: dict, ts: datetime) -> bool:
        px    = self.get_price(sig["token"])
        px_ts = self.get_price_ts(sig["token"])
        if not px or px <= 0 or px_ts is None or (ts - px_ts).total_seconds() > self.cfg["max_price_age_sec"]:
            return False

        lot = self._instruments.lot_size(sig["token"])
        if lot <= 0:
            self.unsubscribe_option(sig["token"])
            self.log.error(f"[{self.name}] unknown lot size for {sig['sym']} — skipped")
            return True
        qty = lot * self.cfg["lots"]

        if not self._acquire_slot():
            self.unsubscribe_option(sig["token"])
            self.log.warning(f"[{self.name}] trade slot blocked — skipped")
            return True
        res = self._place_buy(sig["sym"], sig["token"], qty, px)
        if res is None:
            self._release_slot()
            self.unsubscribe_option(sig["token"])
            self.log.error(f"[{self.name}] BUY failed for {sig['sym']} — skipped")
            return True
        order_id, fill = res

        self._trades_today += 1
        self._trade = {
            **sig, "qty": qty, "entry": fill, "entry_time": ts, "order_id": order_id,
            "trade_no": self._trades_today, "bars_held": 0,
        }
        b = sig["bars"]
        self.log.info(
            f"[{self.name}] [{'LIVE' if LIVE_MODE else 'PAPER'}] ENTRY #{self._trades_today} "
            f"BUY {sig['sym']} x{qty} @ {fill:.2f} | spot={sig['spot']:.2f} | "
            f"HA {'GREEN' if sig['side'] == 'CE' else 'RED'} x2 ({b[0]:%H:%M}, {b[-1]:%H:%M})"
        )
        return True

    def on_option_tick(self, token: int, price: float, ts: datetime, tick_ts: datetime = None):
        if self._pending and token == self._pending["token"] and self._trade is None:
            p = self._pending
            if self._try_fill(p, ts):
                self._pending = None

    # ── Exit ──────────────────────────────────────────────────────────────────

    def _exit(self, spot: float, reason: str, ts: datetime, exit_candle):
        t = self._trade
        if not t:
            return
        t["_exit_in_progress"] = True
        px    = self.get_price(t["token"]) or t["entry"]
        px_ts = self.get_price_ts(t["token"])
        if px_ts is None or (ts - px_ts).total_seconds() > self.cfg["max_price_age_sec"]:
            self.log.warning(f"[{self.name}] exit price for {t['sym']} is stale "
                             f"({px_ts}) — using last price {px:.2f}")

        res = self._place_sell_with_retry(t["sym"], t["token"], t["qty"], px, max_retries=3)
        if res is None and LIVE_MODE:
            if not self._hub.order_router._is_position_open(t["sym"]):
                res = (None, px)
                reason = f"{reason}_EXCHANGE_CLOSED"
            else:
                t["_exit_in_progress"] = False
                t["_exit_failed"]      = reason
                t["_exit_candle"]      = exit_candle
                self._next_retry = ts + timedelta(seconds=self.cfg["sell_retry_sec"])
                self.log.error(f"[{self.name}] CRITICAL: SELL failed — {t['sym']} STILL OPEN, "
                               f"retrying every {self.cfg['sell_retry_sec']}s")
                return
        exit_order_id, sell = res if res is not None else (None, px)

        self._release_slot()
        self.unsubscribe_option(t["token"])
        gross = (sell - t["entry"]) * t["qty"]
        cost  = round_trip_cost_rs(t["entry"], sell, t["qty"])
        net   = gross - cost
        self._today_gross += gross
        self._today_net   += net
        b = t["bars"]
        append_row(self.CSV_FILE, self.TRADE_FIELDS, {
            "date"         : t["entry_time"].strftime("%Y-%m-%d"),
            "trade_no"     : t["trade_no"],
            "side"         : t["side"],
            "symbol"       : t["sym"],
            "strike"       : t["strike"],
            "expiry"       : self._expiry,
            "qty"          : t["qty"],
            "signal_c1"    : b[0].strftime("%H:%M"),
            "signal_c2"    : b[-1].strftime("%H:%M"),
            "entry_time"   : t["entry_time"].strftime("%H:%M:%S"),
            "entry_spot"   : round(t["spot"], 2),
            "entry_premium": round(t["entry"], 2),
            "exit_candle"  : exit_candle.strftime("%H:%M") if exit_candle else "",
            "exit_time"    : ts.strftime("%H:%M:%S"),
            "exit_spot"    : round(spot, 2),
            "exit_premium" : round(sell, 2),
            "exit_reason"  : reason,
            "bars_held"    : t["bars_held"],
            "gross_pnl"    : round(gross, 2),
            "costs"        : round(cost, 2),
            "net_pnl"      : round(net, 2),
            "mode"         : "LIVE" if LIVE_MODE else "PAPER",
            "order_id"     : t["order_id"],
            "exit_order_id": exit_order_id,
        })
        self.log.info(
            f"[{self.name}] [{'LIVE' if LIVE_MODE else 'PAPER'}] EXIT #{t['trade_no']} [{reason}] "
            f"{t['sym']} @ {sell:.2f} (entry {t['entry']:.2f}, {t['bars_held']} bars) | "
            f"gross={gross:.0f} net={net:.0f} | today net={self._today_net:.0f}"
        )
        self._completed.append({**t, "exit": sell, "reason": reason, "gross": gross, "net": net})
        self._trade = None
        self._next_retry = None

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log_candle(self, c, ha, color, partial, action):
        append_row(self.CANDLES_CSV, self.CANDLE_FIELDS, {
            "date": c["ts"].strftime("%Y-%m-%d"), "ts": c["ts"].strftime("%H:%M"),
            "open": c["open"], "high": c["high"], "low": c["low"], "close": c["close"],
            "ha_open": round(ha[0], 2), "ha_high": round(ha[1], 2),
            "ha_low": round(ha[2], 2), "ha_close": round(ha[3], 2),
            "color": color or "DOJI", "partial": int(partial),
            "streak": f"{self._streak_color or '-'}x{self._streak_n}", "action": action,
        })

    def eod_summary(self):
        self.log.info(f"[{self.name}] {'=' * 50}")
        self.log.info(f"[{self.name}] END OF DAY | trades={self._trades_today} | "
                      f"gross={self._today_gross:.0f} net={self._today_net:.0f}")
        for t in self._completed:
            self.log.info(f"[{self.name}]   #{t['trade_no']} {t['sym']} [{t['reason']}] "
                          f"{t['entry']:.2f} -> {t['exit']:.2f} net={t['net']:.0f}")
        if self._trade:
            self.log.error(f"[{self.name}] position still open at EOD: {self._trade['sym']}")
        self.log.info(f"[{self.name}] {'=' * 50}")


class NiftyHeikinAshiATMStrategy(_HeikinAshiATMBase):
    INDEX_TOKEN = 256265
    STRIKE_STEP = 50
    NAME        = "NIFTY_HA5_ATM"
    LOGGER      = "strategy.nifty_ha5_atm"
    CSV_FILE    = "nifty_ha5_atm_trades.csv"
    CANDLES_CSV = "nifty_ha5_atm_candles.csv"


class BankNiftyHeikinAshiATMStrategy(_HeikinAshiATMBase):
    # No INDEX_TOKEN: BankNifty is the hub's main index.
    STRIKE_STEP = 100
    NAME        = "BANKNIFTY_HA5_ATM"
    LOGGER      = "strategy.banknifty_ha5_atm"
    CSV_FILE    = "banknifty_ha5_atm_trades.csv"
    CANDLES_CSV = "banknifty_ha5_atm_candles.csv"
