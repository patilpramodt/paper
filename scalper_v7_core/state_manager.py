# ==============================================================================
# state_manager.py  Crash Recovery & State Persistence
#
# Saves active_trade to disk after every state change.
# On startup, reloads any open position so the bot can manage it.
#
# Fixes from V4:
#   [OK] Bot crash no longer orphans open (paper) positions
#   [OK] State file is written atomically (temp file + rename)
#   [OK] Corrupt state file is handled gracefully
# ==============================================================================

import json
import os
import shutil
from typing import Optional
from scalper_v7_core.config import STATE_FILE
from scalper_v7_core.logger_setup import log


def save_state(active_trade: Optional[dict]) -> None:
    """
    Persist current trade state to disk.
    Uses atomic write (temp file  rename) to avoid partial writes.
    """
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    tmp = STATE_FILE + ".tmp"
    payload = active_trade or {}
    try:
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        shutil.move(tmp, STATE_FILE)
        log.debug(f"State saved: {payload.get('symbol', 'no trade')}")
    except Exception as e:
        log.error(f"State save failed: {e}")


def load_state() -> Optional[dict]:
    """
    Load trade state from disk.
    Returns the active_trade dict or None if no open position.
    """
    if not os.path.exists(STATE_FILE):
        return None
    try:
        with open(STATE_FILE) as f:
            data = json.load(f)
        if not data:
            return None
        # Validate minimal required keys
        required = {"symbol", "entry", "sl", "target", "qty", "option_type"}
        if not required.issubset(data.keys()):
            log.warning(f"State file incomplete  ignoring. Keys: {list(data.keys())}")
            return None
        log.warning(
            f"[WARN]  Recovered open trade from state: {data['symbol']} "
            f"entry={data['entry']} sl={data['sl']} target={data['target']}"
        )
        return data
    except Exception as e:
        log.error(f"State load failed: {e}  starting fresh")
        return None


def clear_state() -> None:
    """Remove state file after a trade is closed."""
    try:
        if os.path.exists(STATE_FILE):
            os.remove(STATE_FILE)
        log.debug("State cleared")
    except Exception as e:
        log.error(f"State clear failed: {e}")
