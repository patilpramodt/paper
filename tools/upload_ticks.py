"""
tools/upload_ticks.py — archive a day of recorded ticks to the private GitHub repo.

    python tools/upload_ticks.py              # today
    python tools/upload_ticks.py 2026-10-07   # a given day
    python tools/upload_ticks.py --all        # every recorded day (skips what's already up)
    python tools/upload_ticks.py --prune      # only prune, no upload

Each day becomes a GitHub Release tagged <day> in TICK_REPO, with every file in
data/ticks/<day>/ plus data/candles/1m/<day>/<group>.csv.gz (as candles_1m_<group>.csv.gz)
attached as assets. Releases instead of git commits because single tick files
exceed GitHub's 100 MB git limit (assets allow 2 GB) and ~700 MB/day would bloat a repo.

Needs GH_TOKEN (fine-grained PAT, Contents: read/write on TICK_REPO) in config_secrets.env.
Re-running is safe: assets already uploaded with the same size are skipped.
Run from cron after build_candles.py.

After uploading, local tick days beyond the newest KEEP_DAYS recorded days are
deleted, but only once every one of their files is in the release with the same
size. 1-min candles are small and stay local. tools/tickdata.py downloads pruned
days back on demand.
"""

import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools.tickdata import CANDLES, TICKS, days, gh as _gh, gh_env, remote_assets  # noqa: E402

KEEP_DAYS = int(os.environ.get("KEEP_DAYS", 30))   # recorded (trading) days kept locally


def _files(day):
    """(asset_name, local_path) for everything belonging to the day."""
    out = []
    tdir = os.path.join(TICKS, day)
    for f in sorted(os.listdir(tdir)):
        p = os.path.join(tdir, f)
        if os.path.isfile(p):
            out.append((f, p))
    cdir = os.path.join(CANDLES, "1m", day)
    if os.path.isdir(cdir):
        for f in sorted(os.listdir(cdir)):
            out.append((f"candles_1m_{f}", os.path.join(cdir, f)))
    return out


def upload(day):
    have = remote_assets(day)
    if have is None:
        _gh("release", "create", day, "--title", f"Ticks {day}",
            "--notes", f"Recorded market data for {day} (core/tick_recorder.py).")
        have = {}

    total = 0
    for name, path in _files(day):
        size = os.path.getsize(path)
        if have.get(name) == size:
            continue
        # gh names the asset after the file; "path#label" only sets the label,
        # so stage renamed candle files via a symlink with the asset name.
        src = path
        if os.path.basename(path) != name:
            src = os.path.join("/tmp", f"upload_ticks_{day}_{name}")
            if os.path.lexists(src):
                os.remove(src)
            os.symlink(path, src)
        try:
            _gh("release", "upload", day, src, "--clobber")
        finally:
            if src != path:
                os.remove(src)
        total += size
        print(f"{day} {name}: {size / 1e6:,.1f} MB")
    print(f"{day}: uploaded {total / 1e6:,.1f} MB" if total else f"{day}: already up to date")


def prune(keep=KEEP_DAYS):
    """Delete local tick days older than the newest `keep`, if fully archived."""
    for d in days()[:-keep] if keep > 0 else days():
        have = remote_assets(d) or {}
        missing = [n for n, p in _files(d) if have.get(n) != os.path.getsize(p)]
        if missing:
            print(f"{d}: NOT pruned, not archived yet: {', '.join(missing)}")
            continue
        shutil.rmtree(os.path.join(TICKS, d))
        print(f"{d}: pruned local ticks (archived in release {d})")


if __name__ == "__main__":
    if gh_env() is None:
        sys.exit("GH_TOKEN not set (add GH_TOKEN=... to config_secrets.env)")
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg == "--prune":
        prune()
        sys.exit(0)
    todo = days() if arg == "--all" else \
        [arg or datetime.now(timezone(timedelta(hours=5, minutes=30))).date().isoformat()]
    failed = False
    for d in todo:
        if not os.path.isdir(os.path.join(TICKS, d)):
            print(f"{d}: no tick data")
            continue
        try:
            upload(d)
        except subprocess.CalledProcessError as e:
            failed = True
            print(f"{d}: FAILED {e.stderr.strip()}")
    prune()
    sys.exit(1 if failed else 0)
