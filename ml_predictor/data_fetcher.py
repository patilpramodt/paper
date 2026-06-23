"""
ml_predictor/data_fetcher.py
─────────────────────────────
Fetches 5-min OHLCV for Nifty 50 + BankNifty + India VIX from Kite Connect.
Loops in 90-day chunks (Kite limit: 100 days per request for 5min data).
Saves to ml_predictor/data/*.csv and supports --append mode for daily updates.

Usage:
    # First-time full fetch (2018 → today):
    python3 ml_predictor/data_fetcher.py

    # Daily append (cron at 15:40):
    python3 ml_predictor/data_fetcher.py --append

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
SLEEP_BETWEEN_CALLS = 0.6   # Historical Data API is capped at 2 req/sec
                            # (not the general 10 req/sec "all other endpoints"
                            # limit — confirmed on Zerodha's own developer forum:
                            # /instruments/historical = max 120 req/min = 2/sec).
                            # 0.6s/call -> ~1.67 req/sec, safe margin under that cap.
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

def fetch_chunks(kite, token: int, from_dt: datetime, to_dt: datetime,
                  max_retries: int = 3) -> pd.DataFrame:
    """
    Fetch 5-min candles in 90-day chunks between from_dt and to_dt.
    Returns combined DataFrame sorted by datetime index.

    Retries each chunk up to max_retries times (with backoff) before giving
    up. Previously, ANY exception (a single transient network blip, a
    429 rate-limit response, anything) silently skipped the entire 90-day
    chunk with just a log.warning — no retry, and nothing in the returned
    DataFrame to indicate a gap exists. Confirmed: a downstream training
    run on a CSV with a chunk-shaped hole in it would see fewer rows for
    that period with no error, indistinguishable from "the market was
    closed." This now retries transient failures, and if a chunk still
    fails after all retries, the gap is collected and raised/logged loudly
    via self.failed_chunks (checked by fetch_full/fetch_append) instead of
    disappearing silently.
    """
    all_records = []
    failed_chunks = []
    chunk_start = from_dt

    while chunk_start < to_dt:
        chunk_end = min(chunk_start + timedelta(days=CHUNK_DAYS), to_dt)

        log.info(f"  Fetching token={token} | {chunk_start.date()} → {chunk_end.date()}")

        records = None
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                records = kite.historical_data(
                    instrument_token=token,
                    from_date=chunk_start.strftime("%Y-%m-%d %H:%M:%S"),
                    to_date=chunk_end.strftime("%Y-%m-%d %H:%M:%S"),
                    interval="5minute",
                    continuous=False,
                    oi=False,
                )
                break  # success
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    backoff = SLEEP_BETWEEN_CALLS * (2 ** attempt)  # exponential backoff
                    log.warning(f"    → Fetch error (attempt {attempt}/{max_retries}, "
                                f"retrying in {backoff:.1f}s): {e}")
                    time.sleep(backoff)
                else:
                    log.error(f"    → Fetch FAILED after {max_retries} attempts: {e}")

        if records is not None:
            all_records.extend(records)
            log.info(f"    → {len(records)} candles fetched")
        else:
            failed_chunks.append((chunk_start, chunk_end, str(last_error)))
            log.error(
                f"    → GAP CREATED: {chunk_start.date()} to {chunk_end.date()} "
                f"could not be fetched after {max_retries} attempts. This window "
                f"will be MISSING from the resulting data."
            )

        chunk_start = chunk_end
        time.sleep(SLEEP_BETWEEN_CALLS)

    if failed_chunks:
        log.error(f"  ⚠ {len(failed_chunks)} chunk(s) failed for token={token} — "
                  f"resulting data has gaps. Re-run fetch for these windows:")
        for cs, ce, err in failed_chunks:
            log.error(f"      {cs.date()} to {ce.date()}  (error: {err})")

    if not all_records:
        df = pd.DataFrame()
        df.attrs["failed_chunks"] = failed_chunks
        return df

    df = pd.DataFrame(all_records)
    df.rename(columns={"date": "datetime"}, inplace=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)

    # Strip timezone immediately. Kite's historical_data returns ISO8601
    # timestamps with a +05:30 offset, so pd.to_datetime above produces a
    # tz-AWARE index. Every existing CSV on disk was written by an earlier
    # run as tz-NAIVE (pd.read_csv with parse_dates doesn't preserve offset
    # info the same way, and the very first fetch_full() run already had
    # this problem). fetch_append() later does
    # pd.concat([existing_tz_naive, df_new_tz_aware]) — mixing the two
    # silently degrades the index to dtype=object (confirmed: pandas gives
    # no warning here), and writing THAT to CSV produces a file pandas
    # can no longer even parse as datetimes on the next read (falls back
    # to a plain string Index). Stripping tz here, at the earliest possible
    # point, guarantees every DataFrame this function returns is tz-naive,
    # so it's always safe to concat with existing on-disk data.
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Market hours only 09:15–15:30
    df = df.between_time("09:15", "15:30")

    # Drop duplicates (chunk boundaries can overlap)
    df = df[~df.index.duplicated(keep="last")]

    df.attrs["failed_chunks"] = failed_chunks
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

        if df.attrs.get("failed_chunks"):
            log.error(f"  ⚠ {name}: {len(df.attrs['failed_chunks'])} chunk(s) "
                      f"permanently missing — see errors above. Saved CSV WILL "
                      f"have gaps. Re-run fetch_full for {name} to retry.")

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
            if df_new.attrs.get("failed_chunks"):
                log.error(f"  ⚠ {name}: {len(df_new.attrs['failed_chunks'])} chunk(s) "
                          f"failed during initial full fetch — CSV will have gaps.")
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

            if df_new.attrs.get("failed_chunks"):
                log.error(
                    f"  ⚠ {name}: today's append FAILED after retries — "
                    f"{from_dt.date()} to {to_dt.date()} will be MISSING from "
                    f"the CSV. Tomorrow's append will resume from the last "
                    f"SUCCESSFUL row ({last_dt}), permanently skipping this gap "
                    f"unless you manually re-run fetch_append for this window."
                )

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

    # Strip timezone so all DataFrames are tz-naive (avoids reindex mismatches)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Ensure only market hours (09:15–15:30) — defensive re-filter
    df = df.between_time("09:15", "15:30")

    # Drop any rows with all-NaN OHLCV
    df.dropna(subset=["open", "high", "low", "close"], inplace=True)

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
        results = fetch_append(kite)
    else:
        log.info("MODE: Full fetch (2018 → today)")
        results = fetch_full(kite)

    # Exit non-zero if ANY instrument had a chunk that failed permanently
    # (after retries — see fetch_chunks's max_retries logic). Previously
    # this script always exited 0 regardless of fetch outcome, so cron
    # would report "success" even when a 90-day (full fetch) or single-day
    # (append) window was silently missing from the CSV — the failure was
    # only visible to someone who actively read the log file. A non-zero
    # exit here means cron's own job-failure tracking (mail, monitoring,
    # whatever you have wired to cron) catches this without anyone needing
    # to read logs/ml_data.log proactively.
    any_failures = False
    for name, df in results.items():
        failed = df.attrs.get("failed_chunks") if hasattr(df, "attrs") else None
        if failed:
            any_failures = True
            log.error(f"⚠ {name}: {len(failed)} chunk(s) permanently failed "
                      f"this run — data has gaps.")

    log.info("\nDone." if not any_failures else "\nCompleted WITH ERRORS (see above).")

    if any_failures:
        sys.exit(1)
