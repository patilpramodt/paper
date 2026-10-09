"""
strategies/orb30_strategy.py

ORB30 — first 30-minute candle breakout, traded with the ATM option. PAPER ONLY.

═══════════════════════════════════════════════════════════════════════════
  RULES (30-minute candles on the UNDERLYING; mirror for PE)
═══════════════════════════════════════════════════════════════════════════
  Universe : the 21 scanner stocks (stock_options_scanner_strategy.UNIVERSE)
             + NIFTY + BANKNIFTY

  MARK     first 30-min candle (09:15-09:45):
             upper = high + 2 pts,  lower = low - 2 pts

  ENTRY    any LATER 30-min candle that CLOSES
             above upper -> buy ATM CE
             below lower -> buy ATM PE
           Entry on the first option tick after that candle closes.
           One trade per instrument per day.

  EXIT     SL      spot crosses the breakout candle's low  - 5 pts (CE)
                                       breakout candle's high + 5 pts (PE)
           TARGET  option P&L reaches Rs 500 (gross, at the bid side, all lots)
           EOD     15:15 square-off

  Buffers are absolute points as specified — 2 / 5 pts is large on low-priced
  stocks (TATASTEEL ~Rs150); per-symbol overrides go in CFG["buffers"].

  Every completed trade is written as ONE row to orb30_trades.csv:
    date, entry_time, exit_time, stock, symbol, side, entry, exit, reason, pnl, ...
"""

import csv
import logging
import os
import statistics
import time
from datetime import datetime, time as dtime, timedelta, timezone

from core.base_strategy import BaseStrategy
from core.costs import effective_spread, estimate_spread, net_pnl_rs

log = logging.getLogger("strategy.orb30")

_IST = timezone(timedelta(hours=5, minutes=30))

NIFTY_TOKEN     = 256265
BANKNIFTY_TOKEN = 260105
INDEX_STEP      = {"NIFTY": 50, "BANKNIFTY": 100}


CFG = {
    "enabled": True,

    # ── session (IST) ────────────────────────────────────────────────────────
    "session_open":         dtime(9, 15),
    "bar_minutes":          30,
    "last_entry_bar_start": dtime(14, 15),   # candle closing 14:45 is the last trigger
    "close_time":           dtime(15, 15),

    # ── levels (points on the underlying) ────────────────────────────────────
    "mark_buffer_pts": 2.0,
    "sl_buffer_pts":   5.0,
    "buffers":         {},      # per-symbol override, e.g. {"TATASTEEL": (0.5, 1.0)} = (mark, sl)

    # ── exits ────────────────────────────────────────────────────────────────
    "target_rs": 500.0,         # fixed rupee target on the option position

    # ── option selection ─────────────────────────────────────────────────────
    "index_min_dte":   1,       # roll index options off expiry day
    "opt_tick_wait_s": 20,
    "lots":            1,
    "stale_price_sec": 45,

    "csv_file": "orb30_trades.csv",
}

LIVE_MODE = False   # paper only — the OrderRouter has one live slot for the whole roster


def _now_ist() -> datetime:
    return datetime.now(tz=_IST).replace(tzinfo=None)


class ORB30Strategy(BaseStrategy):

    INDEX_TOKEN = None          # BankNifty ticks via on_tick (also the heartbeat clock)
    LIVE_MODE   = LIVE_MODE

    # Overridden by the variants in orb30_rvol_strategy.py
    _log          = log
    CSV_FILE      = CFG["csv_file"]
    TARGET_RS     = CFG["target_rs"]   # None = no rupee target
    TRADE_INDICES = True

    def __init__(self, market_hub):
        super().__init__(market_hub)
        self._stock_store = None
        self._index_store = {}     # "NIFTY"/"BANKNIFTY" -> InstrumentStore
        self._inst        = {}     # sym -> instrument state (see _add_inst)
        self._by_token    = {}     # spot token -> sym
        self._positions   = {}     # sym -> trade dict
        self._pending     = {}     # opt token -> pending entry
        self._opt_owner   = {}     # opt token -> sym (open positions)
        self._completed   = []
        self._today_pnl   = 0.0
        self._ready       = False
        self._eod_done    = False
        self._last_hb     = None
        self._nifty_ts    = None

    @property
    def name(self) -> str:
        return "ORB30"

    # ══════════════════════════════════════════════════════════════════════════
    # PRE-MARKET
    # ══════════════════════════════════════════════════════════════════════════

    def pre_market(self, premarket_data, instruments, index_stores=None) -> bool:
        if not CFG["enabled"]:
            self._log.info(f"[{self.name}] disabled via CFG")
            return False

        from core.instruments import StockOptionStore
        if isinstance(instruments, StockOptionStore) and instruments.universe:
            self._stock_store = instruments
            for sym in instruments.universe:
                tok = instruments.spot_token(sym)
                if not tok:
                    continue
                self._add_inst(sym, tok, is_index=False)
                self.subscribe_option(tok)
                self._hub.set_token_owner(tok, self.name)
        else:
            self._log.error(f"[{self.name}] no StockOptionStore — trading indices only")

        for sym, tok in (("NIFTY", NIFTY_TOKEN), ("BANKNIFTY", BANKNIFTY_TOKEN)):
            if not self.TRADE_INDICES:
                break
            store = (index_stores or {}).get(sym)
            if store is None or getattr(store, "_df", None) is None:
                self._log.error(f"[{self.name}] no {sym} InstrumentStore — {sym} skipped")
                continue
            self._index_store[sym] = store
            self._add_inst(sym, tok, is_index=True)

        if not self._inst:
            return False

        self._seed_today()
        self._ready = True
        self._log.info(
            f"[{self.name}] ready | PAPER | {len(self._inst)} instruments | "
            f"mark ±{CFG['mark_buffer_pts']} SL buf {CFG['sl_buffer_pts']} | "
            f"{self._exit_desc()} | EOD {CFG['close_time']:%H:%M}"
        )
        return True

    def _add_inst(self, sym: str, tok: int, is_index: bool):
        mark_buf, sl_buf = CFG["buffers"].get(sym, (CFG["mark_buffer_pts"], CFG["sl_buffer_pts"]))
        self._inst[sym] = {
            "token": tok, "is_index": is_index, "cur": None, "last_closed": None,
            "upper": None, "lower": None, "mark_buf": mark_buf, "sl_buf": sl_buf,
            "traded": False,
        }
        self._by_token[tok] = sym

    def _seed_today(self):
        """Rebuild today's closed 30-min candles after a mid-session restart."""
        kite = getattr(self._hub, "kite", None)
        now  = _now_ist()
        if kite is None or now.time() < dtime(9, 45):
            return
        start = datetime.combine(now.date(), CFG["session_open"])
        cur_bar = self._bar_start(now)
        for sym, st in self._inst.items():
            try:
                raw = kite.historical_data(st["token"], start.strftime("%Y-%m-%d %H:%M:%S"),
                                           now.strftime("%Y-%m-%d %H:%M:%S"), "30minute")
            except Exception as e:
                self._log.warning(f"[{self.name}] seed failed for {sym}: {e}")
                continue
            finally:
                time.sleep(0.35)   # Kite historical API: 3 requests/second
            for r in raw:
                ts = r["date"].replace(tzinfo=None)
                if ts >= cur_bar:
                    break
                bar = {"ts": ts, "o": r["open"], "h": r["high"], "l": r["low"], "c": r["close"]}
                self._on_bar_close(sym, bar, now, seeding=True)
        self._log.info(f"[{self.name}] seeded today's candles for {len(self._inst)} instruments")

    # ══════════════════════════════════════════════════════════════════════════
    # TICKS
    # ══════════════════════════════════════════════════════════════════════════

    def on_tick(self, price: float, ts: datetime, tick_ts: datetime):
        """BankNifty index tick. NIFTY is sampled from the hub cache here too."""
        if not self._ready or not price:
            return
        if "BANKNIFTY" in self._inst:
            self._on_spot("BANKNIFTY", price, ts, tick_ts or ts)
        if "NIFTY" in self._inst:
            npx, nts = self.get_price(NIFTY_TOKEN), self.get_price_ts(NIFTY_TOKEN)
            if npx and nts and nts != self._nifty_ts:
                self._nifty_ts = nts
                self._on_spot("NIFTY", npx, ts, nts)
        if self._last_hb and (ts - self._last_hb).total_seconds() < 1.0:
            return
        self._last_hb = ts
        self._heartbeat(ts)

    def on_candle(self, candle: dict, ts: datetime):
        return   # BankNifty 5-min candles — this strategy builds its own 30-min bars

    def on_option_tick(self, token: int, price: float, ts: datetime, tick_ts: datetime = None):
        if not self._ready or not price:
            return
        sym = self._by_token.get(token)
        if sym is not None:
            self._on_spot(sym, price, ts, tick_ts or ts)
            return
        if token in self._pending:
            self._try_fill_pending(token, price, ts)
            return
        sym = self._opt_owner.get(token)
        if sym is not None:
            self._check_target(sym, price, ts)

    def _on_spot(self, sym: str, price: float, ts: datetime, tick_ts: datetime):
        st = self._inst[sym]
        if tick_ts.time() < CFG["session_open"]:
            return
        bs = self._bar_start(tick_ts)
        if st["last_closed"] is not None and bs <= st["last_closed"]:
            return   # late tick for a candle already closed
        if st["cur"] is not None and bs > st["cur"]["ts"]:
            self._close_cur(sym, ts)
        if st["cur"] is None:
            st["cur"] = {"ts": bs, "o": price, "h": price, "l": price, "c": price}
        else:
            c = st["cur"]
            c["h"], c["l"], c["c"] = max(c["h"], price), min(c["l"], price), price
        if sym in self._positions:
            self._check_sl(sym, price, ts)

    @staticmethod
    def _bar_start(ts: datetime) -> datetime:
        o = datetime.combine(ts.date(), CFG["session_open"])
        k = int((ts - o).total_seconds() // (CFG["bar_minutes"] * 60))
        return o + timedelta(minutes=k * CFG["bar_minutes"])

    def _close_cur(self, sym: str, ts: datetime):
        st = self._inst[sym]
        bar, st["cur"] = st["cur"], None
        self._on_bar_close(sym, bar, ts)

    # ══════════════════════════════════════════════════════════════════════════
    # SIGNAL
    # ══════════════════════════════════════════════════════════════════════════

    def _on_bar_close(self, sym: str, bar: dict, ts: datetime, seeding: bool = False):
        st = self._inst[sym]
        st["last_closed"] = bar["ts"]
        bt = bar["ts"].time()

        if bt == CFG["session_open"]:
            st["upper"] = round(bar["h"] + st["mark_buf"], 2)
            st["lower"] = round(bar["l"] - st["mark_buf"], 2)
            self._log.info(f"[{self.name}] {sym} first candle H {bar['h']:.2f} L {bar['l']:.2f} "
                     f"→ marks {st['upper']:.2f} / {st['lower']:.2f}")
            return
        if seeding or st["upper"] is None or st["traded"] or self._eod_done:
            return
        if bt > CFG["last_entry_bar_start"]:
            return
        if sym in self._positions or any(p["sym"] == sym for p in self._pending.values()):
            return

        c = bar["c"]
        if c > st["upper"]:
            side, sl = "UP", round(bar["l"] - st["sl_buf"], 2)
        elif c < st["lower"]:
            side, sl = "DOWN", round(bar["h"] + st["sl_buf"], 2)
        else:
            return
        self._log.info(f"[{self.name}] {sym} {bar['ts']:%H:%M} candle close {c:.2f} "
                 f"{'>' if side == 'UP' else '<'} mark {st['upper'] if side == 'UP' else st['lower']:.2f} "
                 f"→ {side}, SL spot {sl:.2f}")
        self._arm_entry(sym, side, c, sl, ts)

    # ══════════════════════════════════════════════════════════════════════════
    # ENTRY
    # ══════════════════════════════════════════════════════════════════════════

    def _pick_option(self, sym: str, spot: float, opt_type: str):
        """(token, tradingsymbol, lot, strike) for the ATM contract, or Nones."""
        if not self._inst[sym]["is_index"]:
            strike = self._stock_store.atm_strike(sym, spot)
            tok, tsym, lot = self._stock_store.get_option(sym, strike, opt_type)
            return tok, tsym, lot, strike

        df   = self._index_store[sym]._df
        step = INDEX_STEP[sym]
        strike = int(round(spot / step) * step)
        min_exp = _now_ist().date() + timedelta(days=CFG["index_min_dte"])
        chain = df[(df["instrument_type"] == opt_type) & (df["expiry"].dt.date >= min_exp)]
        if chain.empty:
            return None, None, 0, strike
        chain = chain[chain["expiry"] == chain["expiry"].min()]
        for k in (0, 1, -1, 2, -2):
            hit = chain[chain["strike"] == float(strike + k * step)]
            if not hit.empty:
                r = hit.iloc[0]
                return int(r["instrument_token"]), str(r["tradingsymbol"]), int(r["lot_size"]), strike + k * step
        return None, None, 0, strike

    def _arm_entry(self, sym: str, side: str, spot: float, sl_spot: float, ts: datetime):
        opt_type = "CE" if side == "UP" else "PE"
        tok, opt_symbol, lot, strike = self._pick_option(sym, spot, opt_type)
        if not tok or not lot:
            self._log.warning(f"[{self.name}] {sym} no {opt_type} contract near {strike} — signal skipped")
            return
        if tok in self._pending:
            return
        self._pending[tok] = {
            "sym": sym, "side": side, "opt_type": opt_type, "strike": strike,
            "opt_symbol": opt_symbol, "lot": lot, "qty": lot * CFG["lots"],
            "spot": spot, "sl_spot": sl_spot, "ts": ts,
        }
        self.subscribe_option(tok)
        if not self._inst[sym]["is_index"]:
            # Stock-option legs are owned by the stock strategies; join them.
            # Index strikes stay broadcast — owning would cut other strategies off.
            self._hub.set_token_owner(tok, self.name)
        self._log.info(f"[{self.name}] {sym} arming {opt_symbol} (lot {lot})")
        px = self.get_price(tok)
        if px:
            self._try_fill_pending(tok, px, ts)

    def _spread(self, tok: int, ltp: float, is_index: bool) -> float:
        bid, ask, _, _ = self._hub.best_bid_ask(tok)
        if is_index:
            if bid and ask and ask > bid > 0:
                return round(ask - bid, 2)
            return estimate_spread(ltp)
        return effective_spread(ltp, bid, ask)

    def _try_fill_pending(self, tok: int, ltp: float, ts: datetime):
        p = self._pending[tok]
        sym = p["sym"]
        spread = self._spread(tok, ltp, self._inst[sym]["is_index"])
        res = self._place_buy(p["opt_symbol"], tok, p["qty"], ltp)
        if res is None:
            self._log.error(f"[{self.name}] {p['opt_symbol']} BUY failed")
            self._drop_pending(tok)
            return
        order_id, raw_fill = res
        fill = round(raw_fill + spread / 2.0, 2) if not LIVE_MODE else raw_fill
        tr = dict(p, token=tok, entry=fill, entry_ts=ts, order_id=order_id, spread=spread,
                  spot_token=self._inst[sym]["token"], entry_spot=p["spot"])
        self._pending.pop(tok, None)
        self._positions[sym] = tr
        self._opt_owner[tok] = sym
        self._inst[sym]["traded"] = True
        self._log.info(f"[{self.name}] ENTRY {p['opt_symbol']} @ {fill:.2f} qty={p['qty']} | "
                 f"spot {p['spot']:.2f} SL {p['sl_spot']:.2f} | {self._exit_desc()}")

    def _drop_pending(self, tok: int):
        p = self._pending.pop(tok, None)
        if p and not self._inst[p["sym"]]["is_index"]:
            self._hub.clear_token_owner(tok, self.name)
        self.unsubscribe_option(tok)

    # ══════════════════════════════════════════════════════════════════════════
    # EXITS
    # ══════════════════════════════════════════════════════════════════════════

    def _check_sl(self, sym: str, spot: float, ts: datetime):
        tr = self._positions.get(sym)
        if tr is None:
            return
        hit = spot <= tr["sl_spot"] if tr["side"] == "UP" else spot >= tr["sl_spot"]
        if hit:
            self._exit(sym, "SL", ts)

    def _check_target(self, sym: str, ltp: float, ts: datetime):
        tr = self._positions.get(sym)
        if tr is None:
            return
        exit_px = max(0.05, ltp - tr["spread"] / 2.0)
        if self.TARGET_RS is not None and (exit_px - tr["entry"]) * tr["qty"] >= self.TARGET_RS:
            self._exit(sym, "TARGET", ts, ltp)

    def _heartbeat(self, ts: datetime):
        # Close candles whose 30 min are up even if the instrument went quiet.
        for sym, st in self._inst.items():
            cur = st["cur"]
            if cur and ts >= cur["ts"] + timedelta(minutes=CFG["bar_minutes"], seconds=2):
                self._close_cur(sym, ts)

        if ts.time() >= CFG["close_time"]:
            if not self._eod_done:
                for sym in list(self._positions):
                    self._exit(sym, "EOD", ts)
                for tok in list(self._pending):
                    self._drop_pending(tok)
            self._eod_done = True
            return

        for tok in list(self._pending):
            if (ts - self._pending[tok]["ts"]).total_seconds() > CFG["opt_tick_wait_s"]:
                self._log.warning(f"[{self.name}] {self._pending[tok]['opt_symbol']} no option tick — entry dropped")
                self._drop_pending(tok)

        for sym in list(self._positions):
            tr = self._positions[sym]
            spot = self.get_price(tr["spot_token"])
            if spot:
                self._check_sl(sym, spot, ts)
            if sym in self._positions:
                px = self.get_price(tr["token"])
                if px:
                    self._check_target(sym, px, ts)

    def _exit(self, sym: str, reason: str, ts: datetime, ltp: float = None):
        tr = self._positions.get(sym)
        if tr is None:
            return
        tok, qty = tr["token"], tr["qty"]
        if ltp is None:
            ltp = self.get_price(tok)
            pts = self.get_price_ts(tok)
            if pts and (ts - pts).total_seconds() > CFG["stale_price_sec"]:
                self._log.warning(f"[{self.name}] {tr['opt_symbol']} exit on a {(ts - pts).total_seconds():.0f}s-old price")
            if ltp is None:
                ltp = tr["entry"]
                self._log.error(f"[{self.name}] {tr['opt_symbol']} NO option price at exit — booked at entry")

        res = self._place_sell_with_retry(tr["opt_symbol"], tok, qty, ltp)
        if res is None:
            self._log.error(f"[{self.name}] EXIT FAILED {tr['opt_symbol']} — MANUAL CHECK REQUIRED")
            return
        _, raw_exit = res
        exit_px = round(max(0.05, raw_exit - tr["spread"] / 2.0), 2) if not LIVE_MODE else raw_exit
        gross = round((exit_px - tr["entry"]) * qty, 2)
        net   = net_pnl_rs(tr["entry"], exit_px, qty)
        self._today_pnl += net
        tr.update(exit_price=exit_px, exit_ts=ts, exit_reason=reason, pnl=net, gross_pnl=gross,
                  exit_spot=self.get_price(tr["spot_token"]))
        self._completed.append(tr)
        self._positions.pop(sym, None)
        self._opt_owner.pop(tok, None)
        if not self._inst[sym]["is_index"]:
            self._hub.clear_token_owner(tok, self.name)
        self.unsubscribe_option(tok)
        self._log.info(f"[{self.name}] EXIT [{reason}] {tr['opt_symbol']} {tr['entry']:.2f} → {exit_px:.2f} | "
                 f"gross={gross:.0f} net={net:.0f} | day={self._today_pnl:.0f}")
        self._log_trade(tr)

    # ══════════════════════════════════════════════════════════════════════════
    # LOGGING
    # ══════════════════════════════════════════════════════════════════════════

    _FIELDS = [
        "date", "entry_time", "exit_time", "stock", "symbol", "side", "qty",
        "entry", "exit", "reason", "pnl", "gross_pnl",
        "entry_spot", "sl_spot", "exit_spot", "mode",
    ]

    def _exit_desc(self) -> str:
        return f"target Rs{self.TARGET_RS:.0f}" if self.TARGET_RS is not None else "no target"

    def _row(self, tr: dict) -> dict:
        return {
            "date": tr["entry_ts"].strftime("%Y-%m-%d"),
            "entry_time": tr["entry_ts"].strftime("%H:%M:%S"),
            "exit_time": tr["exit_ts"].strftime("%H:%M:%S"),
            "stock": tr["sym"], "symbol": tr["opt_symbol"], "side": tr["opt_type"],
            "qty": tr["qty"], "entry": tr["entry"], "exit": tr["exit_price"],
            "reason": tr["exit_reason"], "pnl": round(tr["pnl"], 2), "gross_pnl": tr["gross_pnl"],
            "entry_spot": tr["entry_spot"], "sl_spot": tr["sl_spot"],
            "exit_spot": tr.get("exit_spot") or "", "mode": "LIVE" if LIVE_MODE else "PAPER",
        }

    def _log_trade(self, tr: dict):
        row = self._row(tr)
        fname = self.CSV_FILE
        try:
            exists = os.path.isfile(fname)
            with open(fname, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self._FIELDS)
                if not exists:
                    w.writeheader()
                w.writerow(row)
        except Exception as e:
            self._log.error(f"[{self.name}] CSV write failed ({fname}): {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # EOD
    # ══════════════════════════════════════════════════════════════════════════

    def eod_summary(self):
        self._log.info(f"[{self.name}] {'=' * 50}")
        self._log.info(f"[{self.name}] END OF DAY | mode={'LIVE' if LIVE_MODE else 'PAPER'}")
        if self._positions:
            self._log.error(f"[{self.name}] {len(self._positions)} POSITION(S) STILL OPEN: {list(self._positions)}")
        for t in self._completed:
            self._log.info(f"[{self.name}]   {t['sym']:<11} {t['opt_symbol']} [{t['exit_reason']}] "
                     f"{t['entry']:.2f} → {t['exit_price']:.2f} net={t['pnl']:.0f}")
        if self._completed:
            wins = sum(1 for t in self._completed if t["pnl"] > 0)
            self._log.info(f"[{self.name}] Trades {len(self._completed)} | W/L {wins}/{len(self._completed) - wins} "
                     f"| NET {self._today_pnl:.0f} | avg {statistics.mean(t['pnl'] for t in self._completed):.0f}")
        else:
            self._log.info(f"[{self.name}] no trades")
        self._log.info(f"[{self.name}] {'=' * 50}")
