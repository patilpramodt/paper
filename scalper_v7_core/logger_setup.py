# ==============================================================================
# logger_setup.py  Adapted for multi-strategy framework
#
# Instead of creating a standalone logger with its own file handler,
# we plug into the framework's existing logging (set up in t.py).
# All scalper_v7_core modules that do `from scalper_v7_core.logger_setup import log`
# will get the standard Python logger routed through the framework's handlers.
# ==============================================================================

import logging

log = logging.getLogger("strategy.scalper_v7")
