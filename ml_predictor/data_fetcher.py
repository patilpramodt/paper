"""
ml_predictor/data_fetcher.py
─────────────────────────────
Fetches 5-min OHLCV for Nifty 50 + BankNifty + India VIX from Kite Connect.
Loops in 90-day chunks (Kite limit: 100 days per request for 5min data).
Saves to ml_predictor/data/*.csv and supports --append mode for daily updates.

Usage:
    # First-time full fetch (2018 → today):
    python ml_predictor/data_fetcher.py

    # Daily append (cron at 15:40):
    python ml_predictor/data_fetcher.py --append

Reuses your existing token.json — no separate login needed.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone, date

import pandas as pd

# ── IST ───────────────────────────────────────────────────────────────────────
IST = timezone(timedelta(hours=5, minutes=30))

def _now_ist():
    return datetime.now(tz=IST).replace(tzinfo=None)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
TOKEN_FILE = os.path.join(os.path.dirname(BASE_DIR), "token.json")

os.makedirs(DATA_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ml.data_fetcher")

# ── Instrument config ─────────────────────────────────────────────────────────
INSTRUMENTS = {
    "NIFTY50":    {"token": 256265,  "csv": "nifty_5min.csv"},
    "BANKNIFTY":  {"token": 260105,  "csv": "banknifty_5min.csv"},
    "INDIAVIX":   {"token": 264969,  "csv": "vix_5min.csv"},
}

CHUNK_DAYS   = 90    # stay under Kite's 100-day limit
SLEEP_BETWEEN_CALLS = 0.4   # 3 req/sec max → stay safe at ~2.5/sec
START_DATE   = datetime(2018, 1, 1)


# ── Kite session ──────────────────────────────────────────────────────────────

def _load_kite():
    """Load KiteConnect using existing token.json (same as t.py does)."""
    try:
        from kiteconnect import KiteConnect
    except ImportError:
        log.error("kiteconnect not installed. Run: pip install kiteconnect")
        sys.exit(1)

    if not os.path.exists(TOKEN_FILE):
        log.error(f"token.json not found at {TOKEN_FILE}. Run t.py first to login.")
        sys.exit(1)

    with open(TOKEN_FILE) as f:
        data = json.load(f)

    if data.get("date") != str(date.today()):
        log.warning("token.json is from a previous day. Running auto_login...")
        # Add parent dir to path so we can import core
        sys.path.insert(0, os.path.dirname(BASE_DIR))
        try:
            from core.auto_login import auto_login
            auto_login()
            with open(TOKEN_FILE) as f:
                data = json.load(f)
        except Exception as e:
            log.error(f"Auto-login failed: {e}. Start t.py first or export env vars.")
            sys.exit(1)

    kite = KiteConnect(api_key=data["api_key"])
    kite.set_access_token(data["access_token"])
    log.info("Kite session loaded from token.json")
    return kite


# ── Core fetch logic ──────────────────────────────────────────────────────────

def fetch_chunks(kite, token: int, from_dt: datetime, to_dt: datetime) -> pd.DataFrame:
    """
    Fetch 5-min candles in 90-day chunks between from_dt and to_dt.
    Returns combined DataFrame sorted by datetime index.
    """
    all_records = []
    chunk_start = from_dt

    while chunk_start < to_dt:
        chunk_end = min(chunk_start + timedelta(days=CHUNK_DAYS), to_dt)

        log.info(f"  Fetching token={token} | {chunk_start.date()} → {chunk_end.date()}")

        try:
            records = kite.historical_data(
                instrument_token=token,
                from_date=chunk_start.strftime("%Y-%m-%d %H:%M:%S"),
                to_date=chunk_end.strftime("%Y-%m-%d %H:%M:%S"),
                interval="5minute",
                continuous=False,
                oi=False,
            )
            all_records.extend(records)
            log.info(f"    → {len(records)} candles fetched")
        except Exception as e:
            log.warning(f"    → Fetch error (skipping chunk): {e}")

        chunk_start = chunk_end
        time.sleep(SLEEP_BETWEEN_CALLS)

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df.rename(columns={"date": "datetime"}, inplace=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)

    # Market hours only 09:15–15:30
    df = df.between_time("09:15", "15:30")

    # Drop duplicates (chunk boundaries can overlap)
    df = df[~df.index.duplicated(keep="last")]

    return df


def fetch_full(kite) -> dict:
    """Fetch full history (2018 → today) for all instruments. Saves CSVs."""
    to_dt = _now_ist()
    results = {}

    for name, cfg in INSTRUMENTS.items():
        log.info(f"\n{'='*55}")
        log.info(f"  {name} — full fetch from {START_DATE.date()} to {to_dt.date()}")
        log.info(f"{'='*55}")

        df = fetch_chunks(kite, cfg["token"], START_DATE, to_dt)

        if df.empty:
            log.warning(f"  No data fetched for {name}")
            continue

        csv_path = os.path.join(DATA_DIR, cfg["csv"])
        df.to_csv(csv_path)
        log.info(f"  Saved {len(df):,} candles → {csv_path}")
        results[name] = df

    return results


def fetch_append(kite) -> dict:
    """
    Append-only mode: only fetch candles newer than what's already in CSV.
    Designed for daily cron at 15:40.
    """
    to_dt = _now_ist()
    results = {}

    for name, cfg in INSTRUMENTS.items():
        csv_path = os.path.join(DATA_DIR, cfg["csv"])

        if not os.path.exists(csv_path):
            log.info(f"  {name}: no CSV found — running full fetch")
            df_new = fetch_chunks(kite, cfg["token"], START_DATE, to_dt)
        else:
            existing = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            last_dt  = existing.index.max()
            from_dt  = last_dt + timedelta(minutes=5)

            if from_dt >= to_dt:
                log.info(f"  {name}: already up to date (last={last_dt})")
                results[name] = existing
                continue

            log.info(f"  {name}: appending from {from_dt.date()} to {to_dt.date()}")
            df_new = fetch_chunks(kite, cfg["token"], from_dt, to_dt)

            if df_new.empty:
                log.info(f"  {name}: no new candles")
                results[name] = existing
                continue

            df_new = pd.concat([existing, df_new])
            df_new = df_new[~df_new.index.duplicated(keep="last")]
            df_new.sort_index(inplace=True)

        df_new.to_csv(csv_path)
        log.info(f"  {name}: {len(df_new):,} total candles → {csv_path}")
        results[name] = df_new

    return results


def load_csv(instrument: str = "BANKNIFTY") -> pd.DataFrame:
    """Load saved CSV for a given instrument. Used by training scripts."""
    name_map = {
        "NIFTY50":   "nifty_5min.csv",
        "BANKNIFTY": "banknifty_5min.csv",
        "INDIAVIX":  "vix_5min.csv",
    }
    fname = name_map.get(instrument.upper())
    if fname is None:
        raise ValueError(f"Unknown instrument: {instrument}")

    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}. Run data_fetcher.py first.")

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.sort_index(inplace=True)
    log.info(f"Loaded {instrument}: {len(df):,} candles ({df.index.min().date()} → {df.index.max().date()})")
    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Nifty/BankNifty 5-min historical data")
    parser.add_argument("--append", action="store_true", help="Append only (daily cron mode)")
    args = parser.parse_args()

    kite = _load_kite()

    if args.append:
        log.info("MODE: Append (fetching new candles only)")
        fetch_append(kite)
    else:
        log.info("MODE: Full fetch (2018 → today)")
        fetch_full(kite)

    log.info("\nDone.")
