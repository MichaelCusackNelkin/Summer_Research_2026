# Reading the log files

A guide to the four kinds of records the live system writes to `live/logs/`.
This is a *reader's* guide — for operating the system (CLI flags, guard matrix,
timing), see `live/README.md`.

---

## First, the thing that will mislead you

The log directory is **mixed**: real trading records and test-suite output are
written to the same files. Until recently, running `pytest` caused the test
harness to execute the full daily orchestrator against a pretend broker and
append its output right next to production records.

How to tell them apart — the single most useful skill here:

| Signal | Real run | Test run |
|---|---|---|
| Order ID | a number: `151`, `153` | `oid-1`, `oid-7` |
| Fill price | varied, realistic | exactly `100.0`, always |
| In the `.log` file | `argv[0]` ends in `daily_orchestrator.py` | `argv[0]` ends in `pytest/__main__.py` |

The contamination by day (fill records):

| Date | Total lines | Genuinely real |
|---|---|---|
| 2026-07-15 | 2,176 | **0** |
| 2026-07-16 | 43,035 | **15** |
| 2026-07-17 | 22,194 | **98** |
| 2026-07-18 | 544 | **0** |
| 2026-07-20 | 0 | — |

That collapse from 43,035 → 544 → 0 is the fix landing. The autouse
`_isolate_live_output_dirs` fixture in `live/tests/conftest.py` now redirects
`_TARGETS_DIR`, `_FILLS_DIR`, `_TRADES_DIR` and `config.LIVE_LOG_DIR` to a
temporary path. Files from 2026-07-18 onward should be trustworthy;
**anything dated 07-15 through 07-17 must be filtered before you believe it.**

Two more traps in those older files:

- Records can appear **out of chronological order** (in `2026-07-18.jsonl` the
  last timestamp is earlier than the first) because parallel test workers wrote
  to the same file at once. Never assume the file is sorted by time.
- Because the targets file is *overwritten* rather than appended, the last test
  to finish **clobbered** the real record for 07-16 and 07-17. Those two days'
  target books show `oid-N` order IDs; the genuine numbers are gone.

---

## The four kinds of files

A trading day produces four different views of the same afternoon:

1. **`.log`** — the narration. What the program was thinking, minute by minute.
2. **`targets/`** — the plan, and then a short verdict on how it went.
3. **`fills/`** — the receipts. What each order ultimately did.
4. **`trades/`** — the research dataset. Rich per-order detail for studying
   execution quality. *(Does not exist yet — see below.)*

---

## 1. The `.log` files — the narration

**These are not plain text.** Despite the `.log` name, every line is a
self-contained JSON record. That is good news: you can query them properly
instead of grepping blindly.

A line looks like:

```json
{"ts": "2026-07-17T15:36:08.585-04:00", "level": "INFO",
 "logger": "live.daily_orchestrator", "msg": "orchestrator starting",
 "argv": ["..."], "cli_args": {"...": "..."}}
```

Four fields are always present:

- **`ts`** — the time, in **New York time** with millisecond precision, not
  UTC. This matters, because the whole strategy is keyed to 15:50 and 16:00
  New York.
- **`level`** — `INFO` (normal), `WARNING` (odd but survivable), `ERROR`
  (something broke).
- **`logger`** — which program: `live.daily_orchestrator` or `live.retrain`.
- **`msg`** — the human sentence. **This is the stable thing to search on.**

Everything *after* those four varies by line — each message attaches whatever
context is relevant (a symbol, an order ID, a count, a list of problem
tickers). If a line records a crash, an `exc` field holds the full traceback.

Configured in `live/config.py` — `_JsonLineFormatter` (line 228) and
`setup_logging` (line 270). The human-readable
`%(asctime)s [%(levelname)s] ...` format exists only on the *stdout* stream,
never in the files. Note that only `live.daily_orchestrator` and `live.retrain`
call `setup_logging`; other modules (`cache.py`, `trade_recorder.py`,
`live_feed.py`, `replay_feed.py`) use bare `getLogger(__name__)` and — because
the configured loggers set `propagate = False` — do **not** land in these files.

### Why there are three different filename patterns

- `live.daily_orchestrator.log` — **the active file**, today's records.
- `live.daily_orchestrator.log.2026-07-16` — **an archive**. At the first log
  message after midnight New York time, the active file is renamed with the
  date of the content inside it. Thirty are kept; older ones are deleted
  automatically.
- `live.daily_orchestrator-2026-07-15.log` and `daily_orchestrator-2026-07-15.log`
  — **dead files** from two older naming schemes. Nothing writes these anymore
  and the automatic cleanup will never remove them, because it only recognizes
  its own pattern. Safe to delete by hand.

### Why the retrain log is separate

Two different jobs on two different clocks. `retrain` runs off-hours and
rebuilds the statistical models (forecast, risk, impact). The orchestrator runs
during the trading afternoon and places orders. Keeping them apart means a
chatty model rebuild cannot bury the one line that says you have unwanted
overnight exposure.

The retrain log is dominated by cache-coverage decisions: `retrain starting`,
`cache coverage manifest`, one `coverage plan — <schema>: HIT/MISS` per schema,
then `retrain complete`.

### Following a healthy day, start to finish

Search for these messages in this order. This is the actual emitted text:

| # | Message | What it means |
|---|---|---|
| 1 | `orchestrator starting` | carries redacted `argv` + full `cli_args` |
| 2 | `loaded active universe` | the stock list is in |
| 3 | `loaded forecast pipeline` | ↓ the four saved models loading |
| 4 | `loaded covariance artifacts` | |
| 5 | `loaded per-name impact gamma` | |
| 6 | `loaded reference stats` | |
| 7 | `waiting until stream start` | connecting to market data |
| 8 | `waiting for snapshot time` | idling until 15:50 |
| 9 | `snapshotted live feed` | **the moment of decision** |
| 10 | `day_df built` | raw market data → model input features |
| 11 | `alpha computed` | the forecast has an opinion on each stock |
| 12 | `MVO solved` | the optimizer has chosen a dollar portfolio |
| 13 | `share targets computed` | dollars converted to whole share counts |
| 14 | `target book persisted` | **plan saved to disk before any broker contact** |
| 15 | `contracts pre-qualified` | checking each ticker is tradeable |
| 16 | `MKT entry placed` (per stock) | **leg one — getting in** |
| 17 | `MKT entry batch submitted` | |
| 18 | `MOC sweep deadline computed` | when the deadline sweep is scheduled |
| 19 | `MOC exit placed` (per stock) | **leg two — fill-gated MOC as each entry fills** |
| 20 | `MOC exit batch submitted` | |
| 21 | `target book persisted` (2nd) | amended with what actually happened |
| 22 | `post-close position poll: FLAT` | **AUTHORITATIVE — the book is verifiably flat after 16:00** |
| 23 | `target book persisted` (3rd) | post-close status appended |
| 24 | `fills audited` | final tally |

Replay and dry-run terminate early but still return 0:
`DRY RUN — printing target book, no broker connection` and
`REPLAY (no-submit) — book computed and persisted, no broker connection.`

**A quiet ending is the good ending.** The coverage cross-check and the MOC
acceptance audit only speak up when something is wrong. A clean run produces
*no* line from either — the post-close poll ends with the single healthy
`post-close position poll: FLAT` line and rc=0. Absence of every
`OVERNIGHT EXPOSURE` tag is the healthy signal.

### What to search for when worried

Anything containing these means **you may be holding stock overnight, which the
strategy is never supposed to do**:

```
OVERNIGHT EXPOSURE
OVERNIGHT EXPOSURE (pre-close)
OVERNIGHT EXPOSURE (post-close)
POSITIONS MAY BE LIVE INTO THE CLOSE
OPERATOR MUST MANUALLY FLATTEN
MOC REJECTED at exchange
MOC exit: get_positions FAILED
MOC exit: entries filled but zero MOC legs sent
MOC exit: legs skipped past submit cutoff
sweep: MOC legs skipped past submit cutoff
sweep: get_positions FAILED
sweep: dispatcher MOC terminal-rejected
Fill-gated submission errors:
post-close position poll: managed-universe symbols still non-flat
post-close position poll: every attempt FAILED
post-close position poll: reached deadline without a confirmed FLAT
post-close position poll block raised
```

The two `(pre-close)` and `(post-close)` tags identify WHICH safeguard
tripped. `(pre-close)` fires before 16:00 (leaving the operator time to
run `panic`); `(post-close)` fires 16:00-16:05 (definitive verdict after
the auction has printed). Either is rc=11.

Cheapest triage: filter the day's file for `"level":"ERROR"`. On a healthy day
there are almost none.

Investigate but not an emergency:

- `MKT entry fill wait timed out` — entry may still fill later
- `dropping unqualified symbol from target` — universe shrinkage
- `dropping non-NASDAQ-listed symbol from target` — routing filter
- `position drift vs yesterday's target book` — reconcile mismatch
- `raising broker MAX_ORDER_QTY for this session` — risk-limit override
- `coverage cross-check: non-terminal status at cross-check time` — TWS brownout
- `fill-gated dispatch: sweep deadline reached with entry still non-terminal`
  — the fill-gated per-symbol dispatch gave up on this entry; the sweep
  will pick it up. Only an emergency if it appears alongside one of the
  OVERNIGHT EXPOSURE tags above.
- `sweep: cancelled non-terminal entry` — belt-and-braces cancel before
  the sweep's MOC batch; expected on stuck entries.
- `sweep: cancel failed — entry may still be working` — the sweep
  couldn't cancel a still-working entry. This IS a pre-close exposure
  signal (the symbol lands in `cancel_failed_symbols` and rc=11
  short-circuits). Only benign if it appears alongside `MOC exit
  placed` for the same symbol (the sweep's fresh get_positions
  captured the fill and covered it) — but the escalation happens
  regardless because the entry's fate is not verifiable pre-close.
- `sweep: dispatcher MOC terminal-rejected; dropping coverage
  credit so sweep re-submits` — round-3 rejection-recovery. The
  dispatcher's MOC for this symbol was API-accepted but then
  Cancelled/Inactive at the exchange with filled=0; the sweep
  drops the coverage credit and resubmits a fresh MOC sized to
  the actual held qty. Benign iff followed by a `MOC exit placed`
  log line for the same symbol. If the retry also gets rejected
  (past 15:55 cutoff), `_audit_moc_acceptance` OR the coverage
  cross-check catch it and escalate to rc=11.
- `fill-gated dispatch: position snapshot lags fill event
  (expected != current) — deferring MOC to the sweep to avoid
  mis-sizing off a stale book` — round-4 staleness deferral.
  IB emitted the execution report before it updated the
  position book. The dispatcher refuses to size an MOC off the
  stale snapshot; the entry stays in ``pending`` and the sweep
  (which snapshots minutes later) covers it via the two-sided
  residual arithmetic. Benign iff followed by a sweep `MOC exit
  placed` for the same symbol. If it appears every tick for
  many symbols in one run, IB propagation is genuinely stuck —
  worth investigating post-hoc but not an emergency.
- `sleep-to-sweep-deadline raised (non-fatal); proceeding with
  second sweep` — round-4 pump loop hiccup. A raise inside
  `_sleep_until` between the first and second sweep. Non-fatal
  by design — the second sweep still runs, and the sweep's own
  error paths are pessimistic. Only worth a look if it repeats
  daily.
- `pump callback raised during _sleep_until (non-fatal)` —
  round-4. A transient TWS blip during a pumped wall-clock
  wait. The wait continues; a wedged connection would
  eventually be caught by the post-close poll's own
  pessimistic-on-raise branch. Benign in isolation.
- `broker heartbeat during marks wait failed (non-fatal)` —
  round-4. Same idea but during the ~350s pumped
  `wait_for_marks` slot on the rc=0 path. Benign in isolation.
- `post-close position poll: still non-flat, retrying` — IB is settling
  the auction print, poll is backing off. Only an emergency if the loop
  hits the deadline (next line above).
- `post-close position poll: empty read pre-auction; not counting
  toward FLAT confirmation` — round-2 safety bound. The poll got an
  empty response before 16:00:30 and is not treating that as
  authoritative. Not an emergency; the poll keeps going.
- `post-close position poll: flat read; awaiting confirmation` —
  round-2 safety bound. First flat read landed, but the poll needs
  N=2 consecutive to declare FLAT. Not an emergency; the poll keeps
  going.
- `post-close position poll: waiting for not_before wall clock` —
  round-3 wait guard. A live/paper run reached the poll before
  ``DEFAULT_POST_CLOSE_POLL_NOT_BEFORE_TIME`` (default 16:00 + settle).
  The process sleeps until the auction wall clock so the poll can
  give an authoritative post-auction verdict. Benign — expected on
  fast happy-path runs where the audit finished by ~15:52.
- `post-close position poll SKIPPED — replay-submit rehearsal
  reached the poll before today's 16:00 auction wall clock` —
  round-3 REPLAY-only. A ``--replay-submit`` rehearsal cannot
  verify against today's 16:00 print (the fills are queued at the
  paper account for today's close, not the historical date's).
  rc=0 is preserved when pre-close signals were clean, but the
  operator MUST run ``python -m live.panic --paper`` before the
  next real session to flatten the paper positions.

Benign by design: the loud `REPLAY SUBMIT:` banner, and anything suffixed
`(non-fatal)` — tail I/O failures that **cannot** downgrade an rc=11 verdict.

### The exit codes

When the program finishes it returns a number. **The rule that matters:
anything 11 or above means you might be holding stock overnight and a human
needs to look now. Everything from 2 to 10 means it refused to trade —
annoying, but safe.**

| Code | Meaning |
|---|---|
| 0 | Everything worked |
| 2 | Market data feed or feature building failed |
| 3 | The optimizer couldn't find a solution |
| 4 | Too late in the day to submit — **nothing was placed** |
| 5 | The plan was empty; broker never contacted |
| 6 | Asked to replay history *and* trade live — refused |
| 7 | Asked to submit a replay without saying which date |
| 8 | Broker refused the connection (TWS down, wrong port, ID collision) |
| 9 | Bad command-line argument |
| **11** | **Got in, couldn't fully get out. Page someone.** |

Retrain: `0` fine, `9` needed to buy data but wasn't authorized, `10`
authorized but confirmation not given.

---

## 2. The `targets/` folder — the plan and the verdict

One file per day, `YYYY-MM-DD.json`, written **twice** to the same path.
Written by `_write_target_record` (`daily_orchestrator.py:770`), atomically
(staged to a `.tmp` then renamed), so a reader never sees a torn file.

**Write one** happens right after the optimizer solves, **before the broker is
contacted at all**. Deliberate: even if everything downstream explodes, you
have a record of what was intended. This is the only write that happens on
`--dry-run` or a replay without `--replay-submit`.

**Write two** overwrites the same file about six minutes later, adding what
actually happened. Note it fires *before* the fill audit, so this file never
reflects the final 16:00 MOC fills — those are in `fills/`.

### From the first write — the plan

| Key | Meaning |
|---|---|
| `generated_at_utc` / `generated_at_ny` | when the decision was made, same instant twice |
| `mode` | `paper`, `live`, or `replay` |
| `dry_run` | true means it computed a plan and deliberately touched nothing |
| `lambda`, `scale`, `vol_cap` | the optimizer's risk dials |
| `gmv_dollar` | **total size of the book** — longs and shorts added ignoring sign |
| `nmv_dollar` | **net tilt** — longs minus shorts. Near zero means balanced. |
| `universe_file` | which stock list was used |
| `universe_hash` | short fingerprint of that list, so tomorrow's reconciliation can detect it changed underneath |
| **`targets`** | **the heart of the file.** Ticker → share count. Positive long, negative short. |
| `replay_date` | replays only; the historical date being simulated |

Symbols with no usable price are **dropped entirely** from `targets` — the
broker cannot size them.

### From the second write — what happened

- **`entries`** — the getting-in leg, one record per stock: ticker,
  `BUY`/`SELL`, `requested_qty`, `order_id`, `filled_qty`, `status`. An
  `error` field (≤200 chars) appears only on failures.
- **`moc_batch`** — the getting-out leg. `signed_qty` is the *change to your
  holdings*: buying to cover a short is positive, selling to close a long is
  negative. The safety invariant is that each stock's holding plus its exit
  equals exactly zero.
- `moc_order_ids`, `moc_signed_by_sym` — convenience lookups of the same.

Then **eight fields that together are the post-mortem of a bad day.** If any is
non-empty or true, that is what triggered the code-11 page:

| Field | Meaning |
|---|---|
| `moc_skipped_past_cutoff` | exits never sent, too late |
| `moc_positions_unknown` | couldn't read positions, so couldn't compute exits |
| `empty_snapshot_after_fills` | orders filled, yet the broker reported an empty account. Contradiction; suspect stale data. |
| `uncovered_symbols` | independent re-check found stock not covered by an exit order |
| `unfilled_moc_symbols` | exits accepted by the broker but **rejected by the exchange**, usually from submitting past NASDAQ's 15:55 cutoff |
| `fill_gated_submit_errors` | round-4: dispatcher tried to place an MOC for these symbols and the broker `buy`/`sell` call raised (a read timeout AFTER TWS queued the order looks the same to the caller; ignoring this signal risks the sweep double-flattening) |
| `post_close_nonzero_symbols` | AUTHORITATIVE 16:00-16:05 position poll found these managed-universe symbols still non-flat after the auction print |
| `post_close_poll_error` | every post-close poll failed (TWS unreachable); pessimistic rc=11 |

Plus `cancelled_entry_oids` — entries that hadn't resolved and got pulled
by the sweep before the exit was computed.

New round-2 fields also on the record:

| Field | Meaning |
|---|---|
| `sweep_deadline_ny` | when the sweep was scheduled to fire (submit_by − sweep_margin) |
| `sweep_margin_secs` | dynamic margin (scales with book size — 15s floor at 49 names, ~121s at 500; formula budgets for both get_positions and per-symbol status/order costs) |
| `covered_by_fill_gated` | symbols whose MOC came from the fill-gated dispatcher rather than the deadline sweep |
| `cancel_failed_symbols` | entries the sweep tried to cancel but the broker raised on — these entries may still fill mid-window and are a pre-close exposure signal in their own right |

### The `status` values on entries

Some come from the broker (`Filled`, `Cancelled`, `ApiCancelled`, `Inactive`).
Others the program invents:

- `not_submitted` — the order never went out
- `timeout` — waited, gave up
- `poll_failed` — couldn't check on it
- **`wait_budget_exhausted`** — worth understanding. It ran out of time before
  checking this order, so it records `filled_qty: 0` **even though the order
  may genuinely have filled.** A deliberate, known lie, and precisely why the
  independent `uncovered_symbols` cross-check exists. Do not trust a zero here
  on its own.

### `targets/replay/`

Replays of historical dates are **quarantined** into this subfolder. Not
cosmetic: the next morning's reconciliation looks for yesterday's real book
with a non-recursive search of `targets/` only, so a simulated book can never
be mistaken for a real position. Replay filenames use the *historical* date;
the fields inside record the *actual* run time. `--replay-date` with `--live`
is refused outright (rc=6).

### One schema wrinkle

`targets/2026-07-15.json` uses an older format — it has `target_gmv: 1000000.0`
and lacks `lambda`/`scale`. That setting was removed because its name implied
it sized the book, which it did not. Any script reading this folder must
tolerate both shapes.

---

## 3. The `fills/` folder — the receipts

One file per day, `YYYY-MM-DD.jsonl`. The `.jsonl` means one JSON object per
line, appended. Written by `_fill_audit` (`daily_orchestrator.py:1513`).

Written **once**, at the very end of the run, as a single sweep over every
order. Exits are checked first because they cannot settle until the 16:00 close
print; entries get a short per-order cap so one stuck entry cannot eat the
budget. Explicitly best-effort record-keeping — a failure here never changes
the exposure verdict.

**One line per order**, not per fill.

There is **no field naming the record type** — you tell them apart by which
fields are present:

**A completed order** (8 fields):

```json
{"ts": "2026-07-17T00:29:59.525487-04:00", "order_id": "151",
 "symbol": "ABNB", "action": "BUY", "quantity": 5,
 "status": "Filled", "filled": 5, "avg_fill_price": 100.0}
```

`ts` is when the audit *observed* the state — **not** when the exchange filled
it. `avg_fill_price` is null if nothing filled.

**A timeout** (3 fields) — `ts`, `order_id`, `status: "timeout"`. It stopped
waiting; the outcome is genuinely unknown.

**An error** (4 fields) — same plus an `error` message and `status: "error"`.
**Zero of these exist on disk so far.**

On-disk statuses are `Filled`, `Cancelled`, `ApiCancelled` — but remember
nearly all of those are test records.

If an order ID appears in `targets/` but has no line here, the order was
probably never placed — check the `.log` for `leg submission failed` or
`legs skipped`.

---

## 4. The `trades/` folder — does not exist yet

The per-trade recorder (`live/trade_recorder.py`) saves a rich research
dataset, but **`live/logs/trades/` has never been created.** The code has never
run against a real broker session — the only executions so far were tests,
which route to a temporary scratch folder and delete themselves.

When it does run it writes:

```
live/logs/trades/<paper|live|replay>/<trading_date>/<run_id>.parquet
```

A spreadsheet-like table, **one row per order leg per stock** (a stock traded
both in and out gives two rows), with 61 columns: order identity, eight
timestamps, quantities, prices, the spread at decision time, target vs.
achieved dollars, broker commissions, and T+1 through T+5 minute price marks
for studying how the trades move the market. A hidden
`.<run_id>.journal.jsonl` sidecar is written continuously and deleted on
successful finalize — if one survives, the parquet write failed.

Four things to know before analyzing it:

- **It only writes to the real folder if a genuine broker object is present.**
  Otherwise it silently redirects to a temp directory with a warning. If you
  expect a file and don't find one, search the log for
  `redirecting output to scratch dir`.
- **Files are per-*run*, not per-day.** A crash and restart produces two files.
  Combining a day's files is fine, but group by `run_id` before summing
  anything on the target side — targets are per-run intent and would
  double-count. `achieved_dollars` sums cleanly.
- **Use the provided `read_trades()` helper**, not a plain parquet read. The
  schema is at version 4; older version-3 files lack the T+1..T+4 columns, and
  reading a mixed batch naively drops the impact-decay data entirely without
  complaining.
- **The decay curve's x-axis is the observed mark timestamp minus the arrival
  timestamp, not a clean 1/2/3/4/5 minutes.** `arrival_price` is taken at
  snapshot completion (~7-10s after the decision anchor), so realized offsets
  run short of the nominal grid. Fitting on the nominal grid biases the decay
  exponent upward. Filter on `t{n}min_staleness_secs` — a thin name whose only
  print was pre-decision yields the same price at all five offsets, which reads
  as zero decay.

---

## Known problems

**The test-pollution damage is not fully undone.** The conftest fixture stops
*new* contamination, but the 07-15 through 07-17 files are permanently mixed,
and the real order IDs in `targets/2026-07-16.json` and `targets/2026-07-17.json`
were overwritten by mock data and cannot be recovered. There is also a live
risk: the 30-day archive limit counts *files*, so a heavy test day rotating
repeatedly could push genuine trading history off disk. Worth confirming the
fixture covers every path.

**A stale comment.** `live/config.py:280` documents the log filename as
`<name>-YYYY-MM-DD.log`, but line 290 does not include a date. The comment
describes the retired scheme and will mislead the next reader.

**If you build alerting on these files, filter on `argv[0]` first.** The
2026-07-16 log contains 156 deliberately-induced `OVERNIGHT EXPOSURE` errors
from test fixtures. A naive alert would have paged 156 times for one pytest
run.
