"""
core/csv_log.py

Append-only CSV writer that tolerates a changed column list.

csv.DictWriter appends rows under whatever header the file already has, so
adding a column to a strategy's trade CSV silently shifts every new row one
field to the right of its header (pandas then refuses to read the file). When
the on-disk header differs from `fields`, the file is rewritten once with the
new header — old rows keep their values and get blanks in the new columns.
"""

import csv
import os

from core.costs import estimate_spread, fixed_costs_rs

_checked: set[str] = set()


def _migrate_header(fname: str, fields: list[str]):
    with open(fname, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
    if header is None or header == fields:
        return
    with open(fname, newline="") as f:
        rows = list(csv.DictReader(f))
    tmp = fname + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    os.replace(tmp, fname)


def append_row(fname: str, fields: list[str], row: dict):
    exists = os.path.isfile(fname)
    if exists and fname not in _checked:
        _migrate_header(fname, fields)
    _checked.add(fname)
    with open(fname, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})


def round_trip_cost_rs(entry: float, exit_px: float, qty: int) -> float:
    """
    Rupee cost of one index-option round trip under the shared core/costs
    model: full modelled bid-ask spread on the entry premium plus brokerage
    and statutory charges. Paper fills are raw LTPs, so this is what the
    gross `pnl` column leaves out.
    """
    return round(estimate_spread(entry) * qty + fixed_costs_rs(qty, entry, max(exit_px, 0.05)), 2)
