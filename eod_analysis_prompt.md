# Daily EOD Analysis — stock-option scanners (RT, FLOW, V1, MORNING_BO)

You're running unattended after market close. Nobody is here to
answer questions — act decisively, don't hedge or ask for clarification.
This is a REPORT only: do not edit any code, config or CSV.

## Context — what changed on 2026-09-29

RT (STOCK_OPT_SCANNER_RT) and FLOW (STOCK_OPT_SCANNER_FLOW) went live with
a new configuration (commit 2eef8b5, details in core/entry_gate.py):

- Entry gates, a different one per strategy:
  - RT `rs_range`: stock beats NIFTY from the open by >= 0.1% in the trade
    direction AND the day's range so far is >= 12x its 1-min ATR14.
  - FLOW `rs_off_extreme`: stock beats NIFTY by >= 0.5% AND price has pulled
    back >= 3x ATR14 from the day's high (longs) / low (shorts).
- Exit for both (`exit_mode="spot"`): SL 2x / TP 3x a 5-min ATR measured on
  the STOCK price, held to target / stop / EOD. There is no rupee target, no
  profit-lock ladder and no 30-min check any more. Exit reasons are now
  `TP_SPOT`, `SL_SPOT`, `EOD`, and rarely `SL_BACKSTOP` (premium down Rs 5,000).
  A win rate around 45-55% with wins larger than losses is EXPECTED — do
  not flag a sub-60% win rate as a problem by itself.
- Contract roll: RT and FLOW trade NEXT month's contracts on expiry-eve and
  on expiry day (min_dte=2). V1 and MORNING_BO keep the nearest expiry. The
  next stock expiry is 2026-10-27, so on 10-26 and 10-27 RT/FLOW should show
  NOV contracts.

Anything before 2026-09-29 in the RT/FLOW CSVs is the OLD configuration
(exits TARGET / TRAIL_LOCK_HIT / SL_HIT / TP_HIT). Never mix old and new
days when judging whether the new setup works.

## Steps

1. Run: /root/paper/venv/bin/python3 eod_scanner_summary.py
   (read-only, safe to re-run). It prints for each strategy: today's trades,
   net, win rate, avg win/loss, exit mix and contract month; for RT and FLOW,
   day-by-day and cumulative results since 2026-09-29 with the backtest
   expectation; and entry-gate activity plus the startup lines from the core
   log. If it prints NO_TRADES_TODAY (holiday, or the bot didn't run), report
   that briefly to Telegram (step 6) and stop.

2. Sanity checks — any failure is the headline of the message:
   - The RT and FLOW startup lines must show `entry gate=... | exit=spot`.
     If they still show the old text ("target Rs 300-5000" or
     "TP Rs300 / SL Rs1900 flat"), the new code did not load.
   - RT/FLOW exit reasons should only be TP_SPOT / SL_SPOT / EOD /
     SL_BACKSTOP. Any TRAIL_LOCK_HIT, TP_HIT, SL_HIT or T30_EXIT from RT/FLOW
     on or after 2026-09-29 means the old exit is still running.
   - Contract month: RT/FLOW must not trade a contract that expires today or
     tomorrow.
   - Any SL_BACKSTOP, or "blocked for missing ATR" counts above a handful,
     is worth one line.

3. Run the legacy RT diagnostics ONCE, capturing output to a file — it
   appends to logs/indicator_effectiveness_log.csv on every run and takes
   over 2 minutes, so never re-run it:
   /root/paper/venv/bin/python3 eod_report_rt.py > /tmp/eod_rt_out.txt 2>&1
   Then grep /tmp/eod_rt_out.txt for what you need. This script only
   recognises the OLD exit names, so for RT days on or after 2026-09-29 its
   SL_HIT breakdown and indicator win/loss counts will be empty or
   meaningless — mention it only if it shows something clearly useful, and
   skip the per-indicator trend section in that case.

4. Judge the new setup against the backtest expectation printed for RT and
   FLOW, using ONLY days since 2026-09-29:
   - Fewer than 5 live days: report the numbers and say it's too early to
     judge. One bad day is normal — the backtest had losing days on ~40% of
     sessions.
   - 5+ live days: say whether cumulative P&L, trades/day and win rate are
     in line with, better than, or worse than expected. If a strategy is
     cumulatively negative after 10+ live days, say so plainly and recommend
     reviewing or switching it off (require_entry_gate / exit_mode in its
     CFG) — but do not change anything yourself.
   - For anything surprising, read the raw CSVs directly
     (stock_opt_scanner_rt_trades.csv, stock_opt_scanner_flow_trades.csv,
     stock_opt_scanner_rt_signals.csv, logs/<date>/core_<date>.log) rather
     than relying only on the aggregates.

5. Write a short analysis — a few short paragraphs that read well as a
   single chat message, not a formal report:
   - Any sanity-check failure first
   - RT and FLOW: today's trades / net / win rate / exit mix, then running
     total since 2026-09-29 vs expectation
   - V1 and MORNING_BO: one line each (trades and net)
   - One concrete "worth watching" note if anything stands out —
     otherwise say plainly that nothing notable changed

6. Send it to Telegram:
   source /root/paper/config_secrets.env
   curl -s "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
     --data-urlencode "chat_id=${TELEGRAM_CHAT_ID}" \
     --data-urlencode "text=YOUR_ANALYSIS_HERE"
   Use --data-urlencode, not -d — the text has newlines and punctuation
   that need proper encoding. Keep it under ~3500 characters (Telegram's
   hard limit is 4096); tighten the writing rather than truncate
   mid-sentence if it runs long.

7. If anything fails (script error, missing data, Telegram send fails),
   send whatever partial information you have anyway, prefixed
   "⚠️ partial report:" — never send nothing.
