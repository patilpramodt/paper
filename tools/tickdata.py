"""
tools/tickdata.py — load our own recorded market data (core/tick_recorder.py).

    from tools.tickdata import days, ticks, candles, candles_range, snapshot, chain, instruments, premarket

    days()                                   # ['2026-10-07', ...]
    ticks('2026-10-07', 'index')             # all index ticks that day
    ticks('2026-10-07', 'nifty_opt', symbol='NIFTY26O1322600CE')
    candles('2026-10-07', 'index', '1s')     # any size from ticks: 1s 5s 10s 1min 15min 1h 1D
    candles('2026-10-07', 'stock')           # 1-min, prebuilt by tools/build_candles.py
    candles_range('2026-10-07', '2026-12-31', 'index', '1D')   # daily bars across days
    snapshot('2026-10-07')                   # minute PCR / VIX / basis / straddle / max pain
    chain('2026-10-07', underlying='NIFTY')  # minute option chain
    instruments('2026-10-07'), premarket('2026-10-07')

Groups: index, fut, nifty_opt, banknifty_opt, stock, stock_opt, other.

Days pruned from this machine (tools/upload_ticks.py keeps the last KEEP_DAYS)
are fetched on demand from the GitHub Release archive in TICK_REPO and cached
back under data/ticks/<day>/. days(remote=True) lists every archived day.
"""

import json
import os
import shutil
import subprocess
import tempfile

import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
TICKS = os.path.join(ROOT, "ticks")
CANDLES = os.path.join(ROOT, "candles")
TICK_REPO = os.environ.get("TICK_REPO", "patilpramodt/TickData")
SECRETS = os.path.join(os.path.dirname(ROOT), "config_secrets.env")


def gh_env():
    """os.environ plus GH_TOKEN from config_secrets.env (None if no token anywhere)."""
    env = dict(os.environ)
    if env.get("GH_TOKEN"):
        return env
    if os.path.isfile(SECRETS):
        for line in open(SECRETS):
            k, _, v = line.strip().partition("=")
            if k == "GH_TOKEN" and v:
                env["GH_TOKEN"] = v.strip().strip('"').strip("'")
                return env
    return None


def gh(*args, check=True):
    env = gh_env()
    if env is None:
        raise RuntimeError("GH_TOKEN not set (add GH_TOKEN=... to config_secrets.env)")
    return subprocess.run(["gh", *args, "--repo", TICK_REPO], env=env,
                          capture_output=True, text=True, check=check)


def remote_assets(day):
    """{asset_name: size} in the day's release, or None if there is no release."""
    r = gh("release", "view", str(day), "--json", "assets", check=False)
    if r.returncode != 0:
        return None
    return {a["name"]: a["size"] for a in json.loads(r.stdout)["assets"]}


def days(remote=False):
    """Recorded days on this machine; remote=True adds every day archived on GitHub."""
    local = {d for d in os.listdir(TICKS) if os.path.isdir(os.path.join(TICKS, d))} \
        if os.path.isdir(TICKS) else set()
    if remote:
        r = gh("release", "list", "--limit", "5000", "--json", "tagName")
        local |= {x["tagName"] for x in json.loads(r.stdout)}
    return sorted(local)


def _download(day, asset, dest):
    """Fetch one release asset to dest. False if the day or asset isn't archived."""
    with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
        r = gh("release", "download", str(day), "-p", asset, "-D", tmp, check=False)
        src = os.path.join(tmp, asset)
        if r.returncode != 0 or not os.path.isfile(src):
            return False
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.move(src, dest)
        print(f"tickdata: downloaded {day}/{asset} ({os.path.getsize(dest) / 1e6:,.1f} MB)")
        return True


def _p(day, name):
    """Local path of a day's file, downloading it from the archive if it was pruned."""
    path = os.path.join(TICKS, str(day), name)
    if not os.path.isfile(path) and gh_env() is not None:
        _download(day, name, path)
    return path


def has(day, name):
    return os.path.isfile(_p(day, name))


def instruments(day):
    return pd.read_csv(_p(day, "instruments.csv"))


def premarket(day):
    with open(_p(day, "premarket.json")) as f:
        return json.load(f)


def ticks(day, group, symbol=None, token=None, columns=None):
    path = _p(day, f"ticks_{group}.csv.gz")
    df = pd.read_csv(path, usecols=columns, parse_dates=["recv_ts", "exch_ts"],
                     low_memory=False)
    if symbol is not None:
        df = df[df.symbol.isin([symbol] if isinstance(symbol, str) else symbol)]
    if token is not None:
        df = df[df.token.isin([token] if isinstance(token, int) else token)]
    return df.sort_values("recv_ts").reset_index(drop=True)


def resample(df, rule="1min", ts_col="recv_ts", fill=False):
    """
    Build candles of ANY size from ticks: "1s", "5s", "10s", "1min", "3min",
    "15min", "1h", "1D" ... (any pandas offset). Per candle and per symbol:

      open high low close         LTP
      volume                      contracts/shares traded in the candle
      vwap                        volume-weighted price within the candle
      oi_open oi_close oi_chg     open interest (options/futures)
      bid ask spread              last top-of-book in the candle
      buy_qty sell_qty imbalance  last total pending buy/sell qty, (b-s)/(b+s)
      ticks                       number of ticks

    fill=True carries the last close forward into candles with no ticks
    (useful for 1s/5s bars on quiet instruments). Kite streams at most ~1
    tick/sec per instrument, so 1 second is the finest meaningful bar.
    """
    out = []
    for sym, g in df.groupby("symbol"):
        g = g.set_index(ts_col).sort_index()
        rs = lambda c: g[c].resample(rule, label="left", closed="left")
        r = rs("ltp").ohlc()
        cum = rs("volume").last().ffill() if "volume" in g else None
        if cum is not None:
            r["volume"] = cum.diff().fillna(cum - g.volume.iloc[0]).clip(lower=0)
            dv = g.volume.diff().fillna(0).clip(lower=0)
            pv = (g.ltp * dv).resample(rule, label="left", closed="left").sum()
            r["vwap"] = (pv / dv.resample(rule, label="left", closed="left").sum()).where(lambda x: x > 0)
        if "oi" in g:
            r["oi_open"] = rs("oi").first()
            r["oi_close"] = rs("oi").last()
            r["oi_chg"] = r.oi_close - r.oi_open
        if "bid1" in g:
            r["bid"] = rs("bid1").last()
            r["ask"] = rs("ask1").last()
            r["spread"] = r.ask - r.bid
        if "buy_qty" in g:
            r["buy_qty"] = rs("buy_qty").last()
            r["sell_qty"] = rs("sell_qty").last()
            tot = r.buy_qty + r.sell_qty
            r["imbalance"] = ((r.buy_qty - r.sell_qty) / tot).where(tot > 0)
        r["ticks"] = rs("ltp").count()
        if fill:
            r["close"] = r.close.ffill()
            for c in ("open", "high", "low"):
                r[c] = r[c].fillna(r.close)
            r["volume"] = r.get("volume", 0).fillna(0) if cum is not None else 0
        r = r.dropna(subset=["open"])
        r.insert(0, "symbol", sym)
        r.insert(1, "token", g.token.iloc[0])
        out.append(r)
    return pd.concat(out).reset_index().rename(columns={ts_col: "ts"}) if out else pd.DataFrame()


def candles(day, group, rule="1min", symbol=None, fill=False):
    """Candles of any size for one day (1-min read from the prebuilt file if present)."""
    pre = os.path.join(CANDLES, "1m", str(day), f"{group}.csv.gz")
    if rule == "1min" and not fill and not os.path.isfile(pre) and gh_env() is not None:
        _download(day, f"candles_1m_{group}.csv.gz", pre)
    if rule == "1min" and not fill and os.path.isfile(pre):
        df = pd.read_csv(pre, parse_dates=["ts"])
        return df[df.symbol == symbol] if symbol else df
    return resample(ticks(day, group, symbol=symbol), rule, fill=fill)


def candles_range(start, end, group, rule="1D", symbol=None, underlying=None):
    """
    Candles across several days, e.g. daily bars for every recorded day:
        candles_range('2026-10-07', '2026-12-31', 'index', '1D')
    Multi-day bars (e.g. "1W") are built from the concatenated ticks.
    Pass underlying= to select by underlying for options (e.g. all NIFTY strikes).
    """
    frames = []
    for d in days(remote=gh_env() is not None):
        if str(start) <= d <= str(end) and has(d, f"ticks_{group}.csv.gz"):
            t = ticks(d, group, symbol=symbol)
            if underlying is not None:
                inst = instruments(d)
                t = t[t.token.isin(inst[inst.underlying == underlying].token)]
            frames.append(t)
    if not frames:
        return pd.DataFrame()
    allt = pd.concat(frames)
    off = pd.tseries.frequencies.to_offset(rule)
    if isinstance(off, pd.offsets.Tick) and pd.Timedelta(off) < pd.Timedelta("1D"):
        # intraday bars never span overnight: build per day
        return pd.concat([resample(f, rule) for f in frames], ignore_index=True)
    return resample(allt, rule)


def snapshot(day):
    return pd.read_csv(_p(day, "snapshot_1m.csv"), parse_dates=["ts"])


def chain(day, underlying=None):
    df = pd.read_csv(_p(day, "chain_1m.csv.gz"), parse_dates=["ts"])
    return df[df.underlying == underlying] if underlying else df
