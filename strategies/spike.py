"""
strategies/spike.py

Spike Strategy  Bank Nifty 9:15 opening spike trade.

ORIGINAL LOGIC (preserved exactly from spike bot):
   Wake up at 9:14, pre-subscribe ATM CE + PE
   At 9:15 first tick: determine gap direction
      Gap up   (open > prev body high)  enter CE immediately
      Gap down (open < prev body low)   enter PE immediately
      No gap   (inside body)            wait for 2 8-second bullish/bearish signal
   SL: option entry price  120 pts
   Trailing: kicks in at +50 profit, trails 25 pts below highest seen
   Force exit at 9:30 (spike window ends)
   Momentum dead check: option 8s candle after entry is doji/reversed  exit

ADAPTED FOR MODULAR FRAMEWORK:
   Uses shared KiteConnect + WebSocket via MarketHub
   Uses shared PreMarketData (prev_body_high/low, prev_close)
   Uses shared InstrumentStore for token lookup
   Builds its own 8-second candles (index + option) via SecondCandleBuilder
   Receives ticks via on_tick() / on_option_tick() callbacks from MarketHub
   Pre-subscribes ATM CE+PE in pre_market()  hub handles the actual WebSocket call

WHAT IS NOT SHARED (intentionally per-strategy):
   8-second candle builders (index + option)  timing is Spike-specific
   gap_direction state
   Trade state machine
   CSV output (spike_trades.csv separate from orb_trades.csv)
"""

import csv
import logging
import os
import threading
from datetime import datetime, time as dtime, timedelta
from typing import Optional

from core.base_strategy import BaseStrategy
from core.candle import SecondCandleBuilder

log = logging.getLogger("strategy.spike")

#  Spike-specific config 
CFG = {
    "quantity"               : 15,
    "start_time"             : dtime(9, 15),
    "spike_exit_time"        : dtime(9, 30),    # force exit after spike window
    "close_time"             : dtime(15, 15),
    "initial_sl_buffer"      : 120,             # SL = entry_price  120
    "trail_trigger_pts"      : 50,              # trailing activates at +50
    "trail_distance"         : 25,              # trail 25 pts below highest
    "doji_threshold"         : 0.10,            # body/range < 10% = doji
    "bucket_sec"             : 8,               # 8-second candle width
    "min_candles_before_mom" : 2,               # don't check momentum until 2 option candles after entry
    "csv_file"               : "spike_trades.csv",
}


class SpikeStrategy(BaseStrategy):

    @property
    def name(self) -> str:
        return "SPIKE"

    def __init__(self, market_hub):
        super().__init__(market_hub)

        #  8-second candle builders (strategy-private) 
        # Index 8s builder  for 2-candle no-gap signal
        self._index_8s     = SecondCandleBuilder(seconds=CFG["bucket_sec"])
        # Option 8s builder  for momentum dead check post-entry
        self._opt_8s       = None   # created when option token is known

        #  Gap filter state 
        self._gap_direction  : Optional[str] = None  # "CE", "PE", "BOTH"
        self._gap_filter_done: bool = False
        self._market_opened  : bool = False

        #  Pre-subscribed tokens (loaded in pre_market at 9:14) 
        self._pre_ce_token   : Optional[int] = None
        self._pre_pe_token   : Optional[int] = None
        self._pre_ce_sym     : Optional[str] = None
        self._pre_pe_sym     : Optional[str] = None

        #  Trade state 
        self._trade          = None   # dict while position is open
        self._trade_done     : bool = False
        self._today_pnl      : float = 0.0
        self._completed      : list  = []

        #  Pre-market data 
        self._prev_body_high : Optional[float] = None
        self._prev_body_low  : Optional[float] = None
        # Last 5-min candle of prev day — primary gap reference
        self._prev_last5m_high  : Optional[float] = None
        self._prev_last5m_low   : Optional[float] = None
        self._prev_last5m_close : Optional[float] = None
        self._expiry_date    = None   # stored from pm for consistent token lookups
        self._instruments    = None

        self._lock           = threading.Lock()

    #  Pre-market: called once at 9:009:14 

    def pre_market(self, pm, instruments) -> bool:
        """
        Store shared pre-market data.
        Pre-subscribe both ATM CE and PE so options are ready at 9:15 first tick.
        """
        from datetime import datetime
        now = datetime.now().time()

        # If bot restarted after spike window (9:30), skip the day entirely
        if now >= CFG["spike_exit_time"]:
            log.warning(f"[{self.name}] Bot started at {now.strftime('%H:%M:%S')} -- "
                        f"spike window already closed at {CFG['spike_exit_time'].strftime('%H:%M')}. "
                        f"SPIKE skipping today.")
            self._trade_done = True   # prevent any entries
            return True               # still return True so other strategies aren't blocked
        self._instruments    = instruments
        self._prev_body_high = pm.prev_body_high
        self._prev_body_low  = pm.prev_body_low
        # Preferred gap reference: last 5-min candle of prev day (15:25 candle)
        self._prev_last5m_high  = pm.prev_last5m_high
        self._prev_last5m_low   = pm.prev_last5m_low
        self._prev_last5m_close = pm.prev_last5m_close
        prev_close           = pm.prev_close

        log.info(f"[{self.name}] Pre-market | "
                 f"body=[{self._prev_body_low}  {self._prev_body_high}] "
                 f"prev_close={prev_close}")

        # Store expiry for consistent use across all token lookups
        self._expiry_date = pm.expiry_date

        if not prev_close:
            log.warning(f"[{self.name}] No prev_close  gap filter disabled")
            return True

        # Pre-subscribe ATM CE + PE based on previous close
        # Both needed so we can enter without delay at 9:15
        from core.instruments import get_atm_strike
        strike = get_atm_strike(prev_close)

        log.info(f"[{self.name}] Token lookup | strike={strike} expiry={pm.expiry_date} "
                 f"prev_close={prev_close:.2f}")

        ce_tok, ce_sym = instruments.get_option_token(strike, "CE", pm.expiry_date)
        pe_tok, pe_sym = instruments.get_option_token(strike, "PE", pm.expiry_date)

        self._pre_ce_token = ce_tok
        self._pre_ce_sym   = ce_sym
        self._pre_pe_token = pe_tok
        self._pre_pe_sym   = pe_sym

        # Tell MarketHub to subscribe these NOW (so ticks flow in at 9:15)
        if ce_tok:
            self.subscribe_option(ce_tok)
            log.info(f"[{self.name}] Pre-subscribed CE: {ce_sym} ({ce_tok})")
        else:
            log.error(f"[{self.name}] PRE-MARKET FAILED: CE token not found | "
                      f"strike={strike} expiry={pm.expiry_date}  "
                      f"CE trades will be BLOCKED all day")

        if pe_tok:
            self.subscribe_option(pe_tok)
            log.info(f"[{self.name}] Pre-subscribed PE: {pe_sym} ({pe_tok})")
        else:
            log.error(f"[{self.name}] PRE-MARKET FAILED: PE token not found | "
                      f"strike={strike} expiry={pm.expiry_date}  "
                      f"PE trades will be BLOCKED all day")

        return True

    #  Tick handlers 

    def on_tick(self, price: float, ts: datetime, tick_ts: datetime):
        """
        Every index tick.
        tick_ts = exchange timestamp  use for 8-second candle alignment.
        """
        t = ts.time()

        # Outside active window
        if t < CFG["start_time"] or t > CFG["close_time"]:
            return

        # Mark market as opened on first tick at/after 9:15
        if not self._market_opened and t >= CFG["start_time"]:
            self._market_opened = True
            log.info(f"[{self.name}] Market open tick received: {price:.2f}")

        # Feed index 8-second candle builder (uses exchange timestamp)
        closed_8s = self._index_8s.feed_tick(price, tick_ts)

        #  Gap direction: determined on very first index tick at 9:15 
        if not self._gap_filter_done and self._market_opened:
            self._determine_gap_direction(price)
            self._gap_filter_done = True

            # Immediate entry if gap detected
            if self._gap_direction in ("CE", "PE"):
                self._attempt_gap_entry(price, ts)
            return   # gap logic handled, rest runs from next tick

        #  No-gap 2-candle signal: only if no gap and no trade yet 
        if (self._gap_direction == "BOTH" and
                not self._trade_done and
                self._trade is None and
                closed_8s is not None):
            self._check_2candle_signal(closed_8s, price, ts)

        #  Exit monitoring for open trade 
        if self._trade and self._trade["state"] == "OPEN":
            # Force exit at 9:30 (spike window ends)
            if t >= CFG["spike_exit_time"]:
                opt_price = self.get_price(self._trade["token"]) or self._trade["entry"]
                self._do_exit(opt_price, "SPIKE_WINDOW_END", ts)
                return
            # Regular trailing SL check on index tick (spot not option  SL is on option)

    def on_candle(self, candle: dict, ts: datetime):
        """5-min candle  Spike doesn't use 5-min candles, ignore."""
        pass

    def on_option_tick(self, token: int, price: float, ts: datetime, tick_ts: datetime = None):
        """Live option tick."""
        # Feed option 8s candle builder (for momentum dead check)
        if self._trade and token == self._trade.get("token"):
            if self._opt_8s is None:
                self._opt_8s = SecondCandleBuilder(seconds=CFG["bucket_sec"])

            use_ts = tick_ts if tick_ts else ts
            closed_opt_8s = self._opt_8s.feed_tick(price, use_ts)

            if self._trade["state"] == "OPEN":
                # Update highest seen
                if price > self._trade["highest_seen"]:
                    self._trade["highest_seen"] = price

                # Compute trailing SL
                new_sl = self._compute_trailing_sl(
                    self._trade["entry"],
                    self._trade["highest_seen"],
                    self._trade["sl"]
                )
                if new_sl > self._trade["sl"]:
                    log.info(f"[{self.name}] TSL: {self._trade['sl']:.0f}  {new_sl:.0f} "
                             f"(highest={self._trade['highest_seen']:.0f})")
                    self._trade["sl"] = new_sl

                # SL hit check
                if price <= self._trade["sl"]:
                    self._do_exit(self._trade["sl"], "SL", ts)
                    return

                # Momentum dead check (only after enough post-entry candles)
                # DISABLED: commented out to let trailing SL manage the exit
                # instead of cutting early on doji/reversal candles
                # if (closed_opt_8s is not None and
                #         self._is_momentum_dead(self._trade["signal"], self._trade["entry_time"])):
                #     self._do_exit(price, "MOMENTUM_DEAD", ts)

        # Also store price for pre-subscribed tokens before trade entry
        # (no action needed, hub already stores in _last_price)

    #  Gap direction logic 

    def _determine_gap_direction(self, open_price: float):
        """
        Gap detection using previous day's LAST 5-MIN candle (15:25 candle).

        Gap up   : today open > prev_last5m_high  → CE
        Gap down : today open < prev_last5m_low   → PE
        No gap   : today open inside last 5-min range → BOTH (wait for 2x8s signal)

        Falls back to daily body high/low if last 5-min candle data is unavailable.
        """
        # ── Primary: last 5-min candle of prev day (15:25 candle) ──────────
        h5 = self._prev_last5m_high
        l5 = self._prev_last5m_low

        if h5 is not None and l5 is not None:
            log.info(
                f"[{self.name}] Gap ref → prev last 5-min: "
                f"H={h5:.0f}  L={l5:.0f}  Today open={open_price:.0f}"
            )
            if open_price > h5:
                self._gap_direction = "CE"
                log.info(
                    f"[{self.name}]  GAP UP: open={open_price:.0f} "
                    f"> last5m_high={h5:.0f}  CE"
                )
            elif open_price < l5:
                self._gap_direction = "PE"
                log.info(
                    f"[{self.name}]  GAP DOWN: open={open_price:.0f} "
                    f"< last5m_low={l5:.0f}  PE"
                )
            else:
                self._gap_direction = "BOTH"
                log.info(
                    f"[{self.name}]  NO GAP: open={open_price:.0f} "
                    f"inside [{l5:.0f} – {h5:.0f}]  waiting 2-candle signal"
                )
            return

        # ── Fallback: daily body high/low ────────────────────────────────────
        log.warning(
            f"[{self.name}] Last 5-min candle data unavailable — "
            f"falling back to daily body for gap detection"
        )
        bh = self._prev_body_high
        bl = self._prev_body_low

        if bh is None or bl is None:
            self._gap_direction = "BOTH"
            log.warning(f"[{self.name}] No gap reference at all — defaulting to BOTH")
            return

        if open_price > bh:
            self._gap_direction = "CE"
            log.info(
                f"[{self.name}]  GAP UP (fallback): open={open_price:.0f} "
                f"> body_high={bh:.0f}  CE"
            )
        elif open_price < bl:
            self._gap_direction = "PE"
            log.info(
                f"[{self.name}]  GAP DOWN (fallback): open={open_price:.0f} "
                f"< body_low={bl:.0f}  PE"
            )
        else:
            self._gap_direction = "BOTH"
            log.info(
                f"[{self.name}]  NO GAP (fallback): open={open_price:.0f} "
                f"inside [{bl:.0f} – {bh:.0f}]  waiting 2-candle signal"
            )

    def _attempt_gap_entry(self, index_price: float, ts: datetime):
        """Enter immediately on gap  no waiting for candles."""
        if self._trade_done or self._trade:
            return

        signal = self._gap_direction  # "CE" or "PE"

        if signal == "CE" and self._pre_ce_token:
            sym, token = self._pre_ce_sym, self._pre_ce_token
        elif signal == "PE" and self._pre_pe_token:
            sym, token = self._pre_pe_sym, self._pre_pe_token
        else:
            # Fallback: look up current ATM using stored expiry from pre-market
            from core.instruments import get_atm_strike
            strike = get_atm_strike(index_price)
            expiry = self._expiry_date
            log.warning(f"[{self.name}] Gap entry fallback lookup | signal={signal} "
                        f"strike={strike} expiry={expiry} spot={index_price:.2f}")
            token, sym = self._instruments.get_option_token(strike, signal, expiry)

        if not token or not sym:
            log.error(f"[{self.name}] No option token for gap {signal} entry | "
                      f"spot={index_price:.2f} expiry={self._expiry_date} | "
                      f"pre_ce_token={self._pre_ce_token} pre_pe_token={self._pre_pe_token}")
            return

        self._build_entry(sym, token, signal, ts, reason=f"gap_{signal.lower()}")

    #  2-candle no-gap signal 

    def _check_2candle_signal(self, latest_closed: dict, index_price: float, ts: datetime):
        """
        Checks the last 2 closed 8s candles for a 2-candle bullish/bearish signal.
        Ignores doji candles.
        """
        last_two = self._index_8s.last_n_closed(2)
        if len(last_two) < 2:
            return

        c1, c2 = last_two[-2], last_two[-1]
        signal  = self._check_signal(c1, c2)
        if not signal:
            return

        log.info(f"[{self.name}] 2-candle signal: {signal} at {ts.strftime('%H:%M:%S')}")

        # Determine option token (pre-subscribed or live lookup)
        if signal == "CE" and self._pre_ce_token:
            sym, token = self._pre_ce_sym, self._pre_ce_token
        elif signal == "PE" and self._pre_pe_token:
            sym, token = self._pre_pe_sym, self._pre_pe_token
        else:
            # Fallback: live lookup using the same expiry stored from pre-market
            from core.instruments import get_atm_strike
            strike = get_atm_strike(index_price)
            expiry = self._expiry_date  # consistent with pre-market lookup
            log.info(f"[{self.name}] Fallback token lookup | signal={signal} "
                     f"strike={strike} expiry={expiry} spot={index_price:.2f} "
                     f"(pre_ce={'yes' if self._pre_ce_token else 'NONE'} "
                     f"pre_pe={'yes' if self._pre_pe_token else 'NONE'})")
            token, sym = self._instruments.get_option_token(strike, signal, expiry)

        if not token or not sym:
            log.error(f"[{self.name}] No token for {signal} | "
                      f"spot={index_price:.2f} expiry={self._expiry_date} | "
                      f"pre_ce_token={self._pre_ce_token} pre_pe_token={self._pre_pe_token} | "
                      f"This trade is being SKIPPED")
            return

        # Subscribe if not already (may not be pre-subscribed if strike differs)
        self.subscribe_option(token)
        self._build_entry(sym, token, signal, ts, reason="2x8s_signal")

    #  Entry 

    def _build_entry(self, sym: str, token: int, signal: str,
                     ts: datetime, reason: str):
        """Get live option price and create trade."""
        opt_price = self.get_price(token)
        if not opt_price or opt_price <= 0:
            log.warning(f"[{self.name}] No live price for {sym}  cannot enter")
            return

        sl = opt_price - CFG["initial_sl_buffer"]

        # Create fresh 8s option candle builder
        self._opt_8s = SecondCandleBuilder(seconds=CFG["bucket_sec"])

        self._trade = {
            "state"        : "OPEN",
            "symbol"       : sym,
            "token"        : token,
            "signal"       : signal,
            "entry"        : opt_price,
            "sl"           : sl,
            "highest_seen" : opt_price,
            "entry_time"   : ts,
            "gap_direction": self._gap_direction,
        }

        log.info(f"[{self.name}]  ENTRY {sym} @ {opt_price:.0f} | "
                 f"SL={sl:.0f} | Trail kicks at {opt_price + CFG['trail_trigger_pts']:.0f} "
                 f"| Reason={reason}")

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
        })

    #  Exit 

    def _do_exit(self, exit_price: float, reason: str, ts: datetime):
        t = self._trade
        if not t or t["state"] != "OPEN":
            return
        t["state"] = "CLOSED"

        pnl = (exit_price - t["entry"]) * CFG["quantity"]
        self._today_pnl += pnl
        self._trade_done = True

        log.info(f"[{self.name}]  EXIT [{reason}] {t['symbol']} @ {exit_price:.0f} "
                 f"| PnL {pnl:.0f} | Today {self._today_pnl:.0f}")

        self._log_csv({
            "timestamp"    : ts.strftime("%Y-%m-%d %H:%M:%S"),
            "symbol"       : t["symbol"],
            "action"       : "EXIT",
            "price"        : exit_price,
            "sl"           : t["sl"],
            "status"       : "CLOSED",
            "pnl"          : round(pnl, 2),
            "reason"       : reason,
            "gap_direction": t["gap_direction"],
        })

        self._completed.append({**t, "exit_price": exit_price,
                                "exit_reason": reason, "pnl": pnl})

        # Unsubscribe option (unless pre-subscribed ones  leave those)
        # Both strategies could be watching the same ATM, MarketHub deduplicates

    #  Signal helpers 

    def _check_signal(self, c1: dict, c2: dict) -> Optional[str]:
        """2-candle bullish/bearish signal. Returns 'CE', 'PE', or None."""
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
        """
        TSL: only activates once price is 50pts above entry.
        Then SL = highest_seen - 25, but never goes down.
        """
        profit = highest - entry
        if profit >= CFG["trail_trigger_pts"]:
            new_sl = highest - CFG["trail_distance"]
            return max(new_sl, current_sl)
        return current_sl

    def _is_momentum_dead(self, signal: str, entry_time: datetime) -> bool:
        """
        Check option 8s candles AFTER entry for momentum reversal.
        Only valid once MIN_CANDLES_BEFORE_MOM candles have closed post-entry.

        DISABLED: method kept for reference but call site is commented out.
        Exit is now fully managed by trailing SL and force-exit at 9:30.
        """
        # if not self._opt_8s:
        #     return False
        # post = self._opt_8s.closed_after(entry_time)
        # if len(post) < CFG["min_candles_before_mom"]:
        #     return False
        # last = post[-1]
        # if self._is_doji(last):
        #     return True
        # if signal == "CE" and last["close"] < last["open"]:
        #     return True
        # if signal == "PE" and last["close"] > last["open"]:
        #     return True
        return False  # DISABLED

    #  CSV 

    def _log_csv(self, row: dict):
        fname  = CFG["csv_file"]
        exists = os.path.isfile(fname)
        fields = ["timestamp", "symbol", "action", "price",
                  "sl", "status", "pnl", "reason", "gap_direction"]
        with open(fname, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if not exists:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in fields})

    def eod_summary(self):
        log.info(f"\n[{self.name}] {''*50}")
        log.info(f"[{self.name}] END OF DAY")
        log.info(f"[{self.name}] Gap direction  : {self._gap_direction}")
        log.info(f"[{self.name}] Trade executed : {'Yes' if self._trade_done else 'No'}")
        for t in self._completed:
            log.info(f"[{self.name}]   {t['symbol']} {t['exit_reason']} "
                     f"entry={t['entry']:.0f} exit={t['exit_price']:.0f} "
                     f"PnL={t['pnl']:.0f}")
        log.info(f"[{self.name}] Today PnL      : {self._today_pnl:.0f}")
        log.info(f"[{self.name}] {''*50}\n")
