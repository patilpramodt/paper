#!/bin/bash
# ml_predictor/run_daily_trackers.sh
# ───────────────────────────────────
# Launches live_tracker.py for BOTH NIFTY50 and BANKNIFTY, each in its own
# detached `screen` session, and exits immediately (does not block).
# Meant to be fired once per trading day by cron — see CRON SETUP below.
#
# Each tracker's own internal loop (live_tracker.py main()) already:
#   - sleeps until MARKET_START (09:15) if launched early
#   - runs its 5-min predict/resolve cycle all day
#   - saves the CSV and exits cleanly at MARKET_END (15:31)
# So this script only needs to run ONCE per day, any time before 09:15.
# A cron misfire (e.g. accidental double-trigger) is handled by the
# is_running() guard below — it will not launch a second copy.
#
# Logs (screen output, not the prediction CSVs) go to:
#   ml_predictor/data/tracker_log_nifty50_<date>.log
#   ml_predictor/data/tracker_log_banknifty_<date>.log
# Prediction CSVs (the actual deliverable) are written by live_tracker.py
# itself to ml_predictor/data/predictions_<instrument>_<date>.csv
#
# CRON SETUP (run `crontab -e` and add):
#   15 9 * * 1-5  /root/paper/ml_predictor/run_daily_trackers.sh >> /root/paper/ml_predictor/data/cron.log 2>&1
#
# 9:15 IST, Mon-Fri only (1-5 = Mon-Fri). If TZ=Asia/Kolkata is set at the
# top of the crontab (as it is here), "15 9" is already IST — no offset
# needed. If your crontab does NOT set TZ, check the server's local time
# with `timedatectl` and adjust "15 9" to the equivalent.
#
# IMPORTANT: this script REPLACES any direct
#   ... python3 ml_predictor/live_tracker.py --instrument NIFTY50 ...
#   ... python3 ml_predictor/live_tracker.py --instrument BANKNIFTY ...
# lines you may have added by hand earlier. Keep only ONE launch path —
# either the two direct lines, OR this single wrapper line. Running both
# launches each instrument's tracker twice, and the second copy will fight
# the first over the same prediction CSV for that day.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$SCRIPT_DIR/data"
TODAY="$(date +%Y-%m-%d)"

mkdir -p "$DATA_DIR"

is_running() {
    # Checks for an existing screen session with this exact name.
    screen -list 2>/dev/null | grep -q "\.$1[[:space:]]"
}

launch_one() {
    local instrument="$1"          # NIFTY50 | BANKNIFTY
    local session_name="ml_tracker_${instrument,,}"   # lowercased
    local log_file="$DATA_DIR/tracker_log_${instrument,,}_${TODAY}.log"

    if is_running "$session_name"; then
        echo "[$(date '+%H:%M:%S')] $session_name already running — skipping."
        return
    fi

    echo "[$(date '+%H:%M:%S')] Launching $session_name → log: $log_file"
    cd "$ROOT_DIR" || exit 1
    screen -dmS "$session_name" bash -c "python3 ml_predictor/live_tracker.py --instrument $instrument >> '$log_file' 2>&1"
}

launch_one "NIFTY50"

# STAGGER: this VM has only 1 CPU core (confirmed via `nproc`). Both trackers
# load TensorFlow/Keras for their LSTM model, and TF's native oneDNN backend
# initializes its own thread pool on load. Launching both processes within
# the same second makes them fight over that single core during this native
# init window, which has been confirmed (June 25 2026) to corrupt the heap —
# both processes crashed with "free(): invalid pointer" at the identical
# point, right after "LSTM loaded", every time they were launched together.
# Logs showed ~10-11s from launch to that crash point, so 30s of separation
# comfortably clears it. Do not remove this sleep unless the VM is upgraded
# to 2+ cores — see the bug writeup, dated June 25 2026, before changing.
sleep 30

launch_one "BANKNIFTY"

echo "[$(date '+%H:%M:%S')] Both trackers dispatched. Check with: screen -list"

