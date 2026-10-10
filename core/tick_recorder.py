"""
core/tick_recorder.py

TickRecorder — our own market-data archive, so backtests never depend on Kite.

WHY
───
Kite's historical API stops at 1-minute bars and does not serve EXPIRED
option contracts at all. Every strategy question that needs sub-minute
behaviour (spike 8s/10s candles, candle-breakout 5s confirms), real option
premiums, bid/ask spreads, or OI/PCR history was unanswerable. From the day
this runs we keep everything ourselves.

WHAT IS RECORDED  (data/ticks/<YYYY-MM-DD>/)
────────────────────────────────────────────
  ticks_index.csv.gz        NIFTY 50, NIFTY BANK, INDIA VIX
  ticks_fut.csv.gz          NIFTY / BANKNIFTY futures (2 expiries) + stock futures
  ticks_nifty_opt.csv.gz    NIFTY options, ATM ± NIFTY_STRIKES, nearest NIFTY_EXPIRIES
  ticks_banknifty_opt.csv.gz BANKNIFTY options, ATM ± BN_STRIKES, nearest BN_EXPIRIES
  ticks_stock.csv.gz        scanner universe NSE equities
  ticks_stock_opt.csv.gz    scanner universe options, ATM ± STOCK_STRIKES
                            (current expiry; next expiry too when DTE <= ROLL_DTE)
  ticks_other.csv.gz        anything a strategy subscribed that isn't above
  snapshot_1m.csv           one row/minute: spots, futures, VIX, basis,
                            PCR (OI + volume, ATM±10 and full recorded chain),
                            ATM straddle, max pain, total CE/PE OI, strategy PCR
  chain_1m.csv.gz           one row/minute/option: ltp, oi, volume, bid/ask
  instruments.csv           token -> symbol/underlying/expiry/strike/type/lot
  premarket.json            prev close, body/last-5m levels, EMA200, VIX, PCR, expiry

Every tick row carries the full MODE_FULL payload: wall-clock receive time,
exchange timestamp, last trade time, LTP, LTQ, ATP, cumulative volume, total
buy/sell qty, day OHLC + prev close, OI (+ day high/low) and 5-level depth.

HOW IT HOOKS IN
───────────────
  • MarketHub._on_ticks() hands every raw tick batch to on_ticks(). That call
    only appends to a deque — formatting and gzip happen on a writer thread,
    so the tick path (and every strategy behind it) is not slowed down.
  • Recorder-only tokens are subscribed via hub.subscribe_passive(). The hub
    receives and caches them but does NOT broadcast them to strategies, so
    ~800 extra tokens cost the strategies nothing.
  • Files are appended as separate gzip members every FLUSH_SEC, so a crash
    or SIGTERM loses at most a few seconds; the files stay readable.

Build 1-minute candles from the ticks after the close with
tools/build_candles.py; load anything with tools/tickdata.py.
"""

import csv
import gzip
import io
import json
import logging
import os
import threading
import time
from collections import deque
from datetime import date, datetime, timedelta, timezone

log = logging.getLogger("core.tick_recorder")

_IST = timezone(timedelta(hours=5, minutes=30))


def _now_ist() -> datetime:
    return datetime.now(tz=_IST).replace(tzinfo=None)


# ── Config ────────────────────────────────────────────────────────────────────
DATA_ROOT      = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "data", "ticks")
NIFTY_STRIKES  = 20      # ATM ± 20 strikes (50 pt)  → ±1000 pts
NIFTY_EXPIRIES = 2       # current + next weekly
BN_STRIKES     = 20      # ATM ± 20 strikes (100 pt) → ±2000 pts
BN_EXPIRIES    = 2       # current + next monthly
STOCK_STRIKES  = 5       # ATM ± 5 strikes per stock
ROLL_DTE       = 7       # also record next-month stock options inside this DTE
INDEX_FUT_EXPIRIES = 2
RECENTER_SEC   = 300     # extend strike ranges if spot drifts
FLUSH_SEC      = 3
GZ_LEVEL       = 5
MIN_FREE_GB    = 3       # skip recording when the disk is nearly full
CTX_MINUTES    = 60      # minutes of PCR / stock OI history kept for strategies
CTX_OI_STEPS   = 3       # stock OI context sums strikes within this many steps of spot

NIFTY_TOKEN = 256265
BANKNIFTY_TOKEN = 260105
VIX_TOKEN   = 264969

GROUPS = ("index", "fut", "nifty_opt", "banknifty_opt", "stock", "stock_opt", "other")

TICK_FIELDS = (
    ["recv_ts", "exch_ts", "last_trade_time", "token", "symbol",
     "ltp", "ltq", "atp", "volume", "buy_qty", "sell_qty",
     "open", "high", "low", "prev_close", "change",
     "oi", "oi_day_high", "oi_day_low"]
    + [f"bid{i}" for i in range(1, 6)] + [f"bidq{i}" for i in range(1, 6)]
    + [f"bido{i}" for i in range(1, 6)]
    + [f"ask{i}" for i in range(1, 6)] + [f"askq{i}" for i in range(1, 6)]
    + [f"asko{i}" for i in range(1, 6)]
)

CHAIN_FIELDS = ["ts", "token", "symbol", "underlying", "expiry", "strike", "opt_type",
                "ltp", "volume", "oi", "bid", "ask", "bid_qty", "ask_qty",
                "underlying_spot"]

SNAP_FIELDS = [
    "ts", "nifty", "banknifty", "vix",
    "nifty_fut", "banknifty_fut", "nifty_basis", "banknifty_basis",
    "nifty_expiry", "nifty_atm", "nifty_atm_ce", "nifty_atm_pe", "nifty_straddle",
    "nifty_pcr_oi_atm10", "nifty_pcr_oi_all", "nifty_pcr_vol_all",
    "nifty_ce_oi", "nifty_pe_oi", "nifty_max_pain",
    "banknifty_expiry", "banknifty_atm", "banknifty_atm_ce", "banknifty_atm_pe",
    "banknifty_straddle",
    "banknifty_pcr_oi_atm10", "banknifty_pcr_oi_all", "banknifty_pcr_vol_all",
    "banknifty_ce_oi", "banknifty_pe_oi", "banknifty_max_pain",
    "strategy_pcr_bn", "strategy_pcr_nifty", "strategy_vix", "n_tokens",
]


def _ts(v):
    if v is None:
        return ""
    if isinstance(v, datetime):
        return v.replace(tzinfo=None).isoformat(sep=" ", timespec="seconds")
    return str(v)


def _num(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.2f}".rstrip("0").rstrip(".") if v else "0"
    return str(v)


class TickRecorder:

    def __init__(self, hub, stock_universe=(), pm_bn=None, pm_nifty=None,
                 root: str = DATA_ROOT):
        self._hub      = hub
        self._universe = list(stock_universe)
        self._pm_bn    = pm_bn
        self._pm_nifty = pm_nifty
        self._day      = _now_ist().date()
        self._dir      = os.path.join(root, self._day.isoformat())
        os.makedirs(self._dir, exist_ok=True)

        self._q: deque = deque()
        self._meta: dict[int, dict] = {}      # token -> instrument row (whole NFO/NSE)
        self._group: dict[int, str] = {}      # token -> file group
        self._recorded: set[int] = set()      # tokens whose metadata was written
        self._opt_by_root: dict[str, dict] = {}  # root -> {(expiry,strike,type): token}
        self._center: dict[tuple, float] = {}    # (root, expiry) -> strike centre used
        self._index_futs: dict[str, int] = {}    # NIFTY/BANKNIFTY -> nearest fut token
        self._stock_spot: dict[str, int] = {}

        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._rows_written = {g: 0 for g in GROUPS}
        self._inst_lock = threading.Lock()

        # Live option context for strategies (core/option_context.py), one
        # entry per minute snapshot. Appends happen on the snapshot thread;
        # readers copy with tuple() and must tolerate a missing minute.
        self.ctx_nifty: deque = deque(maxlen=CTX_MINUTES)         # (ts, pcr_oi_atm10)
        self.ctx_stock: dict[str, deque] = {}                     # sym -> (ts, ce_oi, pe_oi)

    # ── Setup ─────────────────────────────────────────────────────────────────

    def setup(self, kite, nfo_raw=None, nse_raw=None):
        """Resolve the recording universe and subscribe it passively. Call after
        pre-market data is fetched (needs prev closes) and before hub.run()."""
        st = os.statvfs(self._dir)
        free_gb = st.f_bavail * st.f_frsize / 1e9
        if free_gb < MIN_FREE_GB:
            log.error(f"[REC] only {free_gb:.1f} GB free — recorder disabled today")
            return False
        try:
            nfo_raw = nfo_raw if nfo_raw is not None else kite.instruments("NFO")
            nse_raw = nse_raw if nse_raw is not None else kite.instruments("NSE")
        except Exception as e:
            log.error(f"[REC] instrument dump failed — recorder disabled: {e}")
            return False

        for r in nse_raw:
            self._meta[int(r["instrument_token"])] = r
        for r in nfo_raw:
            self._meta[int(r["instrument_token"])] = r

        today = self._day
        tokens: set[int] = {NIFTY_TOKEN, BANKNIFTY_TOKEN, VIX_TOKEN}

        # Index + stock option chains indexed by (expiry, strike, type)
        roots = {"NIFTY", "BANKNIFTY", *self._universe}
        futs: dict[str, list] = {}
        for r in nfo_raw:
            name = r.get("name")
            if name not in roots:
                continue
            exp = r.get("expiry")
            if not exp or exp < today:
                continue
            it = r.get("instrument_type")
            if it in ("CE", "PE"):
                self._opt_by_root.setdefault(name, {})[(exp, float(r["strike"]), it)] = \
                    int(r["instrument_token"])
            elif it == "FUT":
                futs.setdefault(name, []).append((exp, int(r["instrument_token"])))

        for name, lst in futs.items():
            lst.sort()
            keep = lst[:INDEX_FUT_EXPIRIES] if name in ("NIFTY", "BANKNIFTY") else lst[:1]
            tokens.update(t for _, t in keep)
            if name in ("NIFTY", "BANKNIFTY"):
                self._index_futs[name] = keep[0][1]

        # Centres for strike windows
        spot_nifty = self._spot_hint(kite, "NSE:NIFTY 50", self._pm_nifty)
        spot_bn    = self._spot_hint(kite, "NSE:NIFTY BANK", self._pm_bn)
        tokens |= self._chain_tokens("NIFTY", spot_nifty, NIFTY_STRIKES, NIFTY_EXPIRIES)
        tokens |= self._chain_tokens("BANKNIFTY", spot_bn, BN_STRIKES, BN_EXPIRIES)

        eq = {r["tradingsymbol"]: int(r["instrument_token"]) for r in nse_raw
              if r.get("segment") == "NSE" and r.get("instrument_type") == "EQ"}
        stock_spots = {}
        try:
            q = kite.ltp([f"NSE:{s}" for s in self._universe]) if self._universe else {}
            stock_spots = {k.split(":", 1)[1]: v["last_price"] for k, v in q.items()}
        except Exception as e:
            log.warning(f"[REC] stock LTP fetch failed — stock chains centre on first tick: {e}")
        for s in self._universe:
            tok = eq.get(s)
            if tok:
                self._stock_spot[s] = tok
                tokens.add(tok)
            n_exp = 2 if self._stock_dte(s) <= ROLL_DTE else 1
            tokens |= self._chain_tokens(s, stock_spots.get(s), STOCK_STRIKES, n_exp)

        self._write_instruments(tokens)
        self._write_premarket()
        self._hub.recorder = self
        self._hub.subscribe_passive(sorted(tokens))
        log.info(f"[REC] recording {len(tokens)} tokens → {self._dir}")
        return True

    def start(self):
        for target, name in ((self._writer_loop, "rec-writer"),
                             (self._snapshot_loop, "rec-snapshot")):
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self):
        if self._stop.is_set():
            return
        self._stop.set()
        for t in self._threads:
            t.join(timeout=15)
        self._flush()
        total = sum(self._rows_written.values())
        size = sum(os.path.getsize(os.path.join(self._dir, f)) for f in os.listdir(self._dir))
        log.info(f"[REC] stopped | {total:,} ticks | {size / 1e6:.1f} MB | {self._rows_written}")

    # ── Hub callback (WebSocket thread — keep it cheap) ───────────────────────

    def on_ticks(self, now: datetime, ticks):
        self._q.append((now, ticks))

    # ── Universe helpers ──────────────────────────────────────────────────────

    def _spot_hint(self, kite, key, pm):
        try:
            return float(kite.ltp([key])[key]["last_price"])
        except Exception:
            return float(pm.prev_close) if pm is not None and pm.prev_close else None

    def _expiries(self, root):
        return sorted({k[0] for k in self._opt_by_root.get(root, {})})

    def _stock_dte(self, sym):
        exps = self._expiries(sym)
        return (exps[0] - self._day).days if exps else 999

    def _chain_tokens(self, root, spot, n_side, n_exp) -> set[int]:
        chain = self._opt_by_root.get(root, {})
        if not chain or not spot:
            return set()
        out = set()
        for exp in self._expiries(root)[:n_exp]:
            strikes = sorted({k[1] for k in chain if k[0] == exp})
            if not strikes:
                continue
            i = min(range(len(strikes)), key=lambda j: abs(strikes[j] - spot))
            for k in strikes[max(0, i - n_side): i + n_side + 1]:
                for it in ("CE", "PE"):
                    tok = chain.get((exp, k, it))
                    if tok:
                        out.add(tok)
            self._center[(root, exp)] = strikes[i]
        return out

    def _recenter(self):
        """Add strikes when spot has drifted from the window centre."""
        new: set[int] = set()
        targets = [("NIFTY", NIFTY_TOKEN, NIFTY_STRIKES, NIFTY_EXPIRIES),
                   ("BANKNIFTY", BANKNIFTY_TOKEN, BN_STRIKES, BN_EXPIRIES)]
        targets += [(s, tok, STOCK_STRIKES, 2 if self._stock_dte(s) <= ROLL_DTE else 1)
                    for s, tok in self._stock_spot.items()]
        for root, spot_tok, n_side, n_exp in targets:
            spot = self._hub.last_price(spot_tok)
            if not spot:
                continue
            exps = self._expiries(root)[:n_exp]
            strikes = sorted({k[1] for k in self._opt_by_root.get(root, {}) if k[0] in exps})
            if len(strikes) < 2:
                continue
            step = min(b - a for a, b in zip(strikes, strikes[1:]) if b > a)
            c = self._center.get((root, exps[0]))
            if c is None or abs(spot - c) >= step * max(2, n_side // 4):
                new |= self._chain_tokens(root, spot, n_side, n_exp)
        new -= self._recorded
        if new:
            self._write_instruments(new)
            self._hub.subscribe_passive(sorted(new))
            log.info(f"[REC] recentred — added {len(new)} tokens")

    def _classify(self, token) -> str:
        g = self._group.get(token)
        if g:
            return g
        m = self._meta.get(token) or {}
        it, name, seg = m.get("instrument_type"), m.get("name"), m.get("segment", "")
        if token in (NIFTY_TOKEN, BANKNIFTY_TOKEN, VIX_TOKEN) or seg == "INDICES":
            g = "index"
        elif it == "FUT":
            g = "fut"
        elif it in ("CE", "PE") and name == "NIFTY":
            g = "nifty_opt"
        elif it in ("CE", "PE") and name == "BANKNIFTY":
            g = "banknifty_opt"
        elif it in ("CE", "PE") and name in self._universe:
            g = "stock_opt"
        elif it == "EQ" and m.get("tradingsymbol") in self._universe:
            g = "stock"
        else:
            g = "other"
        self._group[token] = g
        return g

    # ── Writers ───────────────────────────────────────────────────────────────

    def _write_instruments(self, tokens):
        path = os.path.join(self._dir, "instruments.csv")
        with self._inst_lock:
            new = [t for t in tokens if t not in self._recorded]
            if not new:
                return
            exists = os.path.isfile(path)
            with open(path, "a", newline="") as f:
                w = csv.writer(f)
                if not exists:
                    w.writerow(["token", "symbol", "underlying", "segment", "instrument_type",
                                "expiry", "strike", "lot_size", "tick_size", "group"])
                for t in new:
                    m = self._meta.get(t) or {}
                    w.writerow([t, m.get("tradingsymbol", ""), m.get("name", ""),
                                m.get("segment", ""), m.get("instrument_type", ""),
                                m.get("expiry") or "", m.get("strike", ""),
                                m.get("lot_size", ""), m.get("tick_size", ""),
                                self._classify(t)])
                    self._recorded.add(t)

    def _write_premarket(self):
        out = {"date": self._day.isoformat(), "recorded_at": _ts(_now_ist())}
        for label, pm in (("banknifty", self._pm_bn), ("nifty", self._pm_nifty)):
            if pm is None:
                continue
            out[label] = {k: (v.isoformat() if isinstance(v, date) else v)
                          for k, v in vars(pm).items()
                          if not k.startswith("_") and isinstance(v, (int, float, str, date, bool, type(None)))}
        with open(os.path.join(self._dir, "premarket.json"), "w") as f:
            json.dump(out, f, indent=2, default=str)

    def _tick_row(self, now, t):
        tok = t.get("instrument_token")
        m = self._meta.get(tok) or {}
        ohlc = t.get("ohlc") or {}
        d = t.get("depth") or {}
        buy = (d.get("buy") or [])[:5]
        sell = (d.get("sell") or [])[:5]
        buy += [{}] * (5 - len(buy))
        sell += [{}] * (5 - len(sell))
        row = [
            now.isoformat(sep=" ", timespec="milliseconds"),
            _ts(t.get("exchange_timestamp") or t.get("timestamp")),
            _ts(t.get("last_trade_time")),
            tok, m.get("tradingsymbol", ""),
            _num(t.get("last_price")), _num(t.get("last_traded_quantity")),
            _num(t.get("average_traded_price")), _num(t.get("volume_traded")),
            _num(t.get("total_buy_quantity")), _num(t.get("total_sell_quantity")),
            _num(ohlc.get("open")), _num(ohlc.get("high")), _num(ohlc.get("low")),
            _num(ohlc.get("close")), _num(t.get("change")),
            _num(t.get("oi")), _num(t.get("oi_day_high")), _num(t.get("oi_day_low")),
        ]
        row += [_num(x.get("price")) for x in buy] + [_num(x.get("quantity")) for x in buy]
        row += [_num(x.get("orders")) for x in buy]
        row += [_num(x.get("price")) for x in sell] + [_num(x.get("quantity")) for x in sell]
        row += [_num(x.get("orders")) for x in sell]
        return row

    def _flush(self):
        if not self._q:
            return
        bufs = {}
        unknown = set()
        while self._q:
            now, ticks = self._q.popleft()
            for t in ticks:
                tok = t.get("instrument_token")
                if tok not in self._recorded:
                    unknown.add(tok)
                g = self._classify(tok)
                if g not in bufs:
                    sio = io.StringIO()
                    bufs[g] = [sio, csv.writer(sio), 0]
                bufs[g][1].writerow(self._tick_row(now, t))
                bufs[g][2] += 1
        if unknown:
            self._write_instruments(unknown)   # strategy-subscribed tokens
        for g, (sio, _, n) in bufs.items():
            path = os.path.join(self._dir, f"ticks_{g}.csv.gz")
            header = not os.path.isfile(path)
            with gzip.open(path, "at", compresslevel=GZ_LEVEL, newline="") as f:
                if header:
                    csv.writer(f).writerow(TICK_FIELDS)
                f.write(sio.getvalue())
            self._rows_written[g] += n

    def _writer_loop(self):
        while not self._stop.wait(FLUSH_SEC):
            try:
                self._flush()
            except Exception as e:
                log.error(f"[REC] flush failed: {e}")

    # ── Minute snapshots ──────────────────────────────────────────────────────

    def _snapshot_loop(self):
        last_recenter = time.time()
        while not self._stop.is_set():
            now = _now_ist()
            # wake just after each minute boundary
            if self._stop.wait(60 - now.second - now.microsecond / 1e6 + 0.5):
                break
            now = _now_ist().replace(second=0, microsecond=0)
            if now.time() < datetime.strptime("09:15", "%H:%M").time():
                continue
            try:
                self._snapshot(now)
            except Exception as e:
                log.error(f"[REC] snapshot failed: {e}")
            if time.time() - last_recenter >= RECENTER_SEC:
                last_recenter = time.time()
                try:
                    self._recenter()
                except Exception as e:
                    log.error(f"[REC] recenter failed: {e}")

    def _chain_stats(self, root, spot):
        """PCR / straddle / max pain from the hub caches for the nearest expiry."""
        h = self._hub
        exps = self._expiries(root)
        if not exps or not spot:
            return {}
        exp = exps[0]
        rows = [(k[1], k[2], tok) for k, tok in self._opt_by_root[root].items()
                if k[0] == exp and tok in self._recorded]
        if not rows:
            return {}
        strikes = sorted({r[0] for r in rows})
        atm = min(strikes, key=lambda k: abs(k - spot))
        ai = strikes.index(atm)
        near = set(strikes[max(0, ai - 10): ai + 11])
        oi = {(k, it): h.last_oi(tok) for k, it, tok in rows}
        vol = {(k, it): h.last_volume(tok) for k, it, tok in rows}
        ce_oi = sum(v for (k, it), v in oi.items() if it == "CE")
        pe_oi = sum(v for (k, it), v in oi.items() if it == "PE")
        ce_n = sum(v for (k, it), v in oi.items() if it == "CE" and k in near)
        pe_n = sum(v for (k, it), v in oi.items() if it == "PE" and k in near)
        ce_v = sum(v for (k, it), v in vol.items() if it == "CE")
        pe_v = sum(v for (k, it), v in vol.items() if it == "PE")
        # max pain: strike minimising total option-writer payout
        pain = None
        if ce_oi and pe_oi:
            pain = min(strikes, key=lambda s: sum(
                oi.get((k, "CE"), 0) * max(0, s - k) + oi.get((k, "PE"), 0) * max(0, k - s)
                for k in strikes))
        tok = self._opt_by_root[root]
        ce = h.last_price(tok.get((exp, atm, "CE")))
        pe = h.last_price(tok.get((exp, atm, "PE")))
        r = lambda a, b: round(a / b, 3) if a and b else ""
        return dict(expiry=exp, atm=atm, atm_ce=_num(ce), atm_pe=_num(pe),
                    straddle=round(ce + pe, 2) if ce and pe else "",
                    pcr_oi_atm10=r(pe_n, ce_n), pcr_oi_all=r(pe_oi, ce_oi),
                    pcr_vol_all=r(pe_v, ce_v), ce_oi=ce_oi, pe_oi=pe_oi, max_pain=pain or "")

    def _near_oi(self, root, spot):
        """(CE OI, PE OI) summed over nearest-expiry strikes within
        CTX_OI_STEPS strike steps of spot — the stock OI the option-context
        gate was tested on (2026-10-10 study, chain_1m)."""
        exps = self._expiries(root)
        if not exps or not spot:
            return None
        rows = [(k[1], k[2], tok) for k, tok in self._opt_by_root[root].items()
                if k[0] == exps[0] and tok in self._recorded]
        strikes = sorted({r[0] for r in rows})
        if len(strikes) < 2:
            return None
        step = min(b - a for a, b in zip(strikes, strikes[1:]) if b > a)
        ce = pe = 0
        for k, it, tok in rows:
            if abs(k - spot) <= CTX_OI_STEPS * step:
                if it == "CE":
                    ce += self._hub.last_oi(tok)
                else:
                    pe += self._hub.last_oi(tok)
        return (ce, pe) if ce and pe else None

    def _update_ctx(self, now, sn):
        pcr = sn.get("pcr_oi_atm10")
        if pcr:
            self.ctx_nifty.append((now, float(pcr)))
        for s, tok in self._stock_spot.items():
            oi = self._near_oi(s, self._hub.last_price(tok))
            if oi:
                self.ctx_stock.setdefault(s, deque(maxlen=CTX_MINUTES)).append((now, *oi))

    def _snapshot(self, now):
        h = self._hub
        n, bn, vix = h.last_price(NIFTY_TOKEN), h.last_price(BANKNIFTY_TOKEN), h.last_price(VIX_TOKEN)
        nf = h.last_price(self._index_futs.get("NIFTY"))
        bf = h.last_price(self._index_futs.get("BANKNIFTY"))
        sn, sb = self._chain_stats("NIFTY", n), self._chain_stats("BANKNIFTY", bn)
        row = {"ts": _ts(now), "nifty": n or "", "banknifty": bn or "", "vix": vix or "",
               "nifty_fut": _num(nf), "banknifty_fut": _num(bf),
               "nifty_basis": round(nf - n, 2) if nf and n else "",
               "banknifty_basis": round(bf - bn, 2) if bf and bn else "",
               "strategy_pcr_bn": getattr(self._pm_bn, "pcr", "") or "",
               "strategy_pcr_nifty": getattr(self._pm_nifty, "pcr", "") or "",
               "strategy_vix": getattr(self._pm_bn, "vix", "") or "",
               "n_tokens": len(self._recorded)}
        for k, v in sn.items():
            row[f"nifty_{k}"] = v
        for k, v in sb.items():
            row[f"banknifty_{k}"] = v
        self._update_ctx(now, sn)
        path = os.path.join(self._dir, "snapshot_1m.csv")
        header = not os.path.isfile(path)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SNAP_FIELDS, extrasaction="ignore")
            if header:
                w.writeheader()
            w.writerow(row)

        # option chain snapshot
        spots = {"NIFTY": n, "BANKNIFTY": bn}
        spots.update({s: h.last_price(t) for s, t in self._stock_spot.items()})
        sio = io.StringIO()
        w = csv.writer(sio)
        with self._inst_lock:
            recorded = list(self._recorded)
        for tok in recorded:
            m = self._meta.get(tok) or {}
            if m.get("instrument_type") not in ("CE", "PE"):
                continue
            ltp = h.last_price(tok)
            if not ltp:
                continue
            bid, ask, bq, aq = h.best_bid_ask(tok)
            w.writerow([_ts(now), tok, m.get("tradingsymbol", ""), m.get("name", ""),
                        m.get("expiry") or "", m.get("strike", ""), m.get("instrument_type"),
                        _num(ltp), h.last_volume(tok), h.last_oi(tok), _num(bid), _num(ask),
                        bq, aq, _num(spots.get(m.get("name")))])
        path = os.path.join(self._dir, "chain_1m.csv.gz")
        header = not os.path.isfile(path)
        with gzip.open(path, "at", compresslevel=GZ_LEVEL, newline="") as f:
            if header:
                csv.writer(f).writerow(CHAIN_FIELDS)
            f.write(sio.getvalue())
