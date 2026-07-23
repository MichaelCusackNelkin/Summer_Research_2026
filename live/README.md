# `live/` — daily orchestrator and retrain

Operator's guide. Everything here runs from the repo root through the
project virtualenv (`.venv/bin/python`). Repo conventions (paths,
`DATABENTO_API_KEY` spelling, active universe) live in
[../CLAUDE.md](../CLAUDE.md) — read that once, then this.

## 1. What this is

The research pipeline in this repo estimates three pieces —
a return **forecast** (`forecast/forecasting.ipynb`), a factor
**covariance** matrix (`covariance/covariance.ipynb`), and a square-root
market-**impact** model (`impact/impact_estimation.ipynb`) — and solves
them together as a daily mean-variance optimization
(`optimizer/optimal_portfolio.ipynb`). Those notebooks are the
authoritative research references; this directory is the production port
of the same math.

`live/` contains three scripts. `live/daily_orchestrator.py` runs once
per trading day: it streams Databento imbalance + tbbo from market open,
snapshots at 15:50 America/New_York, engineers the same features the
forecast pipeline was trained on, predicts alpha, solves the MVO,
converts dollar targets to signed share counts, and executes them as
a **two-leg batch — LMT entries at 15:50 priced at the last observed
tbbo mid (rounded PASSIVELY per side: BUY floors down, SELL ceils up)
and sized to `target - current`, then a per-symbol FILL-GATED MOC
counter-flatten (each symbol's MOC fires the moment its entry becomes
terminal Filled) plus a DEADLINE SWEEP before 15:55 that MOC-flattens
every residual managed-universe position and CANCELS every LMT
still working at the sweep deadline** — so the book is flat overnight.
A limit that never crosses is simply not entered (no position, no
MOC, no exposure). A partial fill is MOC-covered at the ACTUAL filled
qty (sized from a fresh `broker.get_positions()`, not from the
requested qty). A post-close position
poll (16:00-16:05) then AUTHORITATIVELY verifies flat before returning
rc=0; a nonzero managed position after 16:05 is rc=11. `live/retrain.py`
runs on a slower cadence (weekly or monthly), refreshes the historical
Databento cache on a rolling window, and rebuilds the three artifact
sets on disk. `live/panic.py` is the account-wide kill switch (see
section 10). The daily reads whatever the retrain last wrote; there
is no other coupling between them.

## 2. Prerequisites

### Install

```bash
.venv/bin/pip install -e ./data_collection
.venv/bin/pip install -e ./ibkr_broker
```

### Environment

- `DATABENTO_API_KEY` in `.env` at the repo root. The legacy hyphenated
  `DATABENTO-API-KEY` spelling is also accepted for compatibility with
  the notebooks — the underscore form is preferred.
- Interactive Brokers TWS (or IB Gateway) running with API access
  enabled. See `ibkr_broker/src/ibkr_broker/broker.py` for the port
  layout (7497 paper, 7496 live, 4002/4001 for Gateway) and TWS API
  setup notes. The daily orchestrator refuses paper-mode connections
  to a live port and refuses live-mode without the confirmation string.

### Artifacts required before the first run

The daily orchestrator fails fast if any of these are missing:

| Path                                              | Producer                                    | Missing behavior                                                                    |
| ------------------------------------------------- | ------------------------------------------- | ----------------------------------------------------------------------------------- |
| `forecast/forecast_pipe.joblib`                   | forecast notebook or `retrain.py --refit forecast`   | `FileNotFoundError`, orchestrator exits before opening the feed |
| `covariance/loadings_return.npy`                  | covariance notebook or `retrain.py --refit covariance` | `FileNotFoundError`                                             |
| `covariance/idiosyncratic_diag_return.npy`        | same                                        | `FileNotFoundError`                                                                 |
| `covariance/factor_cov_matrix.npy`                | same                                        | `FileNotFoundError`                                                                 |
| `covariance/cov_matrix_return.npy`                | same                                        | `FileNotFoundError`                                                                 |
| `top_49_universe.txt` (current active universe)   | `universe_selection.py`                     | `FileNotFoundError`                                                                 |
| `impact/impact_gamma_vec.npy`                     | impact notebook or `retrain.py --refit impact`       | falls back to scalar `--impact-gamma` (0.5), logs a warning     |
| `impact/reference_stats.parquet`                  | `retrain.py --refit impact`                          | falls back to null ADV/vol, live coverage assertion aborts the run (rc=2) |

The forecast pipeline, covariance artifacts, and universe are hard
prerequisites; the impact vector and reference-stats parquet are soft
(the orchestrator will still start without them but will log warnings
and, in the reference-stats case, is likely to abort at snapshot time).

If you have never run `retrain.py`, the safe path is to first run the
three research notebooks end to end to seed the artifact directories,
then let `retrain.py` take over on the recurring schedule.

## 3. Testing without a live Databento subscription

Paper trading with Interactive Brokers and market data from Databento
are **independent** subscriptions. The live imbalance schema in
particular is expensive (order of $1000s/mo); you can validate almost
the entire pipeline before turning it on. In order of increasing cost
and realism:

### Layer 0 — the test suite (cost: 0)

```bash
.venv/bin/python -m pytest data_collection/tests/ live/tests/ -v
```

Proves feature engineering matches the offline notebooks, the MVO
solve produces a valid book against real on-disk covariance, the
symbol-mapping logic handles Databento's live gateway convention, the
atomic-write helpers do not corrupt artifacts, the coverage /
empty-book / unqualified-symbol / oversized-leg guards fire, and the
replay + broker-connect exit paths return coded rcs. Runs in under two
seconds.

### Layer 1 — replay mode against cached historical data (cost: 0 beyond already-cached history)

```bash
.venv/bin/python -m live.daily_orchestrator --replay-date 2026-05-15 --paper
```

Replay sources imbalance + tbbo from the historical parquet cache the
retrain / notebooks already populated, feeds them through the SAME
`_build_live_day_frame -> alpha -> MVO -> sizing -> pre-qualify ->
LMT entries -> MOC` path a real run uses, and (by default) stops one
step short of order submission. The only thing NOT exercised is the
live websocket transport — everything downstream is identical to a
real session, INCLUDING:

- **Replay-day decision clock for the LMT staleness guard**: tick
  age is measured against the replay-day 15:50 anchor, not today's
  wall clock, so historical ticks are correctly evaluated as
  minutes-old (fresh) rather than days-old (silently skipped). If
  this is ever broken, `--replay-submit` will silently place ZERO
  orders while returning rc=0 — exactly the failure mode round-2
  reviewer B caught.
- **Look-ahead cap on the entry price**: `_extract_last_quote_by_sym`
  bounds the tbbo frame at the replay-day decision timestamp
  BEFORE picking the last tick, so the cached whole-day parquet
  cannot leak the 16:00 closing print into the LMT entry price
  (that would price the entry at the exact tick the strategy is
  trying to forecast).

First replay of a date not in the cache would need a one-off Databento
historical materialisation (not the paid live subscription — cheap
historical). By default the cache-aware fetch guard refuses that fetch
and raises `CoverageError` naming the closest cached parquet and the
shortfall, so a cold-cache rehearsal cannot silently rack up a batch
charge. Pass `--allow-fetch` to authorise it (replay only — see the
"Cache-aware fetch guard" section below for the full picture).

`--replay-submit` opts in to placing real paper MOC orders — see
section 4 for the full flag reference, including the guard matrix and
the caveats.

### The cache guardrail and the universe drift trap

Both `daily_orchestrator --replay-date` and `retrain` route every
historical read through the coverage manifest (see
`live/cache.py`). The default is `--allow-fetch` OFF: if the requested
`{schema, [start, end), symbols}` slice is not covered by an existing
parquet the resolver refuses to submit a Databento Batch API job and
raises a self-diagnostic `CoverageError` naming the nearest parquet and
the shortfall (days at head/tail, or count of missing required
symbols). This is what stops a stale `--end-date` or a drifted universe
file from quietly paying for a fresh fetch.

Two gates guard a real fetch:

- `retrain`: `--allow-fetch` AND `--yes` (interactive `yes` on a TTY
  is accepted in place of `--yes`). Both required because batch API
  calls cost real money.
- `daily_orchestrator --replay-date`: `--allow-fetch` alone (single
  gate — replay is a one-day historical pull, cheap).

New retrain exit codes when the gates trip:

- `9` — coverage MISS and `--allow-fetch` not set; nothing fetched.
- `10` — `--allow-fetch` set but confirmation refused (no `--yes`, no
  TTY, `no` at the prompt, or Ctrl-D). Nothing fetched.

**The universe drift trap.** `retrain --refit forecast|impact|all`
reads `imbalance`, `tbbo`, and `statistics` for the **broad**
universe (`--broad-universe-file`, default `universe.txt`), not just
the active set. The parquet cache is keyed on the exact ticker list —
so if the broad universe file has been rebuilt since the cache
parquets were materialised, retrain will MISS even for a small active
universe. Concrete example: `top_50_universe.txt` today includes
`CRWV`; the cached tbbo parquet from an earlier notebook run does
not. A `retrain --refit impact --broad-universe-file top_50_universe.txt`
run against that cache MISSes with:

```
coverage plan — tbbo: MISS (nearest 6436dd54a7ced32262a3242e08d42d20.parquet: parquet missing 1 required symbol(s), e.g. ['CRWV'], shortfall=1)
```

Two ways out:

1. Match the broad file to what the cache was built for — e.g. pass
   `--broad-universe-file universe.txt` (the fully-cached pool) even
   when the active file is smaller. The coverage-HIT filter pares the
   result down to the active set on the fly.
2. Pay the delta: `--allow-fetch --yes` submits a fresh batch job for
   the missing symbols.

The near-miss log names the closest parquet and the shortfall so an
operator can diagnose without hand-inspecting the parquet cache.
Sample outputs:

```
# HIT — schema and range and symbols all present
coverage plan — tbbo: HIT (015b0f098e17be916bbb238312a4d5d4.parquet)

# MISS on missing symbol
coverage plan — tbbo: MISS (nearest 6436dd54a7ced32262a3242e08d42d20.parquet: parquet missing 1 required symbol(s), e.g. ['CRWV'], shortfall=1)

# MISS on short at tail
coverage plan — imbalance: MISS (nearest 015b0f098e17be916bbb238312a4d5d4.parquet: parquet ends 2026-06-10, request needs 2026-06-30 (short 20d at tail), shortfall=20)
```

### Layer 2 — cheap live-transport smoke

The live tbbo / mbp-1 schemas are far cheaper than live imbalance.
Since `LiveFeed`'s connect / reconnect / symbol-mapping code path is
schema-agnostic (imbalance and tbbo take the same route), a
tbbo-only session over `LiveFeed` validates the transport in
production. Do this once before committing to the live imbalance
subscription; it catches auth, network, and symbology issues without
paying for close-imbalance data.

### Layer 3 — the real thing, paper first

Once layers 0-2 pass:

1. Buy the live imbalance subscription.
2. Run `.venv/bin/python -m live.daily_orchestrator --paper --dry-run`
   for a few sessions. Confirm the log shows a well-formed target book
   (non-empty, sensible GMV, coverage above the threshold) and no
   errors.
3. Drop `--dry-run` for `--paper` — real paper MOC orders. Compare the
   next morning's `broker.reconcile_positions` warnings against the
   persisted target JSON.
4. Only after several clean paper sessions, run `--live` with the
   confirmation string.

## 4. The daily orchestrator

```bash
.venv/bin/python -m live.daily_orchestrator --help
```

### CLI reference

Mode:

- `--paper` — paper mode. Default. Connects to TWS port 7497.
- `--live` — live mode. Requires `--live-confirm=YES_I_WANT_LIVE_TRADING`
  and connects to port 7496. Mutually exclusive with `--paper`.
- `--live-confirm LIVE_CONFIRM` — broker confirmation string required
  with `--live`. Must be exactly `YES_I_WANT_LIVE_TRADING`
  (defined at `Broker.LIVE_CONFIRM_MSG`).
- `--dry-run` — compute alpha, solve MVO, log the intended book, do not
  connect to the broker and do not place any orders.

Replay (dress-rehearsal):

- `--replay-date YYYY-MM-DD` — instead of the live gateway, source
  imbalance + tbbo from the cached historical adapter for the given
  trading day (America/New_York). Skips the stream-start and snapshot
  wall-clock waits and bypasses the `--submit-by` cutoff — the wall
  clock is unrelated to the historical date. First replay of an
  un-cached date may trigger a one-off Databento historical
  materialisation (cheap historical, not the paid live subscription).
  Default: unset.
- `--replay-submit` — opt in to placing REAL paper MOC orders during
  a replay run. Off by default; without it, replay is broker-free
  (computes, logs, persists the book, exits before touching TWS).
  Only valid with `--replay-date`. Always paper — combining with
  `--live` is refused regardless.
- `--allow-fetch` — authorise the replay feed to submit a Databento
  Batch API job when the local cache cannot cover the requested day.
  Off by default; a cold-cache replay refuses rather than silently
  paying for a fresh historical materialisation. Only meaningful with
  `--replay-date`; ignored on the live path (no historical cache in
  the loop). Single gate — replay is a one-day historical pull, so no
  `--yes` co-gate.
- `--replay-ignore-cutoff` — in replay mode, IGNORE the wall-clock
  `--submit-by` cutoff. Off by default — replay now HONOURS the
  cutoff so a late-in-the-day rehearsal correctly surfaces the same
  rc=4 / rc=11 codes a real run would. Pass this flag to preserve
  the earlier afternoon-rehearsal ergonomics where you deliberately
  want to exercise the order path even though `now` is past 15:55.
  Only meaningful with `--replay-date`. The SNAPSHOT wall-clock wait
  is unconditionally skipped in replay (the parquet is already on
  disk); this flag only relaxes the submit-by check.

The replay guard matrix, stated plainly:

- `--replay-date` + `--live` → refused, rc=6. Replay can never touch
  a live account.
- `--replay-submit` without `--replay-date` → refused, rc=7. The
  broker path is untouched.
- `--replay-submit` with a live `--broker-port` (7496 / 4001) → the
  broker's own live-port guard trips at connect time, returning rc=8.
  Paper-only, always.

Replay date range: anything inside `2025-06-11..2026-06-10` for the
currently cached parquets. Use a weekday — Wednesdays are the safest
default (no exchange-holiday adjacency); `2026-05-13` is a good
picked-at-random rehearsal date. Weekend dates trigger a MISS on the
manifest since no bars exist for them.

Reference-stats prerequisite: `impact/reference_stats.parquet` MUST
exist before any replay, or the C2 coverage assertion aborts the run
at 0% and returns rc=2. Generate it via `retrain --refit impact`
(see section 6). Explicit ordering:

```bash
# Step 1 — populate impact/reference_stats.parquet.
.venv/bin/python -m live.retrain --refit impact --end-date 2026-06-11
# Step 2 — now replay is safe.
.venv/bin/python -m live.daily_orchestrator --replay-date 2026-05-13 --paper
```

Replay caveats:

1. `--replay-submit` paper fills happen at TODAY's close, not the
   historical date's close, so PnL is not realistic. Replay validates
   **plumbing, not PnL**. The startup log emits a loud WARNING every
   time `--replay-submit` is used.
2. **Baseline pollution.** `--replay-submit` paper fills sit in the
   TWS paper account and pollute the next real run's diff baseline:
   `broker.get_positions()` on the following day will look like drift
   and the strategy will trade to unwind them. Flatten the paper
   account between rehearsals — `python -m live.panic --paper` (see
   section 10). Also do this before switching from `--replay-submit`
   back to a real paper session.
3. Replay ADV / vol come from the retrain-time reference-stats
   snapshot, which is likely as-of a date AFTER the replayed
   historical date. That is mild look-ahead in the impact and cost
   denominators. This is intentional (replay parity with the live
   code path), not accuracy.
4. Replay target books are quarantined in `live/logs/targets/replay/`
   and never enter next-day reconciliation.

Sizing:

- `--vol-cap VOL_CAP` — target daily dollar-volatility cap. Default
  5000. The QP is scale-invariant — this is the effective risk limit.
- `--lambda- LAM` — risk aversion per dollar. Default 5e-05. Must be
  positive; `<= 0` returns rc=9. The optimizer's `scale` is
  auto-computed as `1 / lambda` (see "Lambda auto-scale" below).
- `--impact-gamma IMPACT_GAMMA` — scalar impact gamma used when no
  per-name vector is loaded. Default 0.5.
- `--use-impact-vector` — load `impact/impact_gamma_vec.npy` and use per
  name (default on).
- `--no-impact-vector` — disable per-name impact gamma; use the scalar.

**Lambda auto-scale.** The QP is scale-invariant; `scale` is a numerical
conditioning hint, not a book-sizing knob. The unconstrained optimal
book is approximately `y* ≈ Σ⁻¹α / λ`, so `1/λ` is the natural order
of magnitude — the code auto-sets `scale = 1 / args.lam`. The actual
GMV a run produces comes from `--vol-cap`, `--lambda-`, and the impact
+ cost terms; there is no dedicated GMV knob. (The old `--target-gmv`
flag was misleadingly named — read as a sizing knob but only touched
`scale` — and has been removed.)

Timing (all America/New_York):

- `--stream-start-time STREAM_START_TIME` — time to open the live feed.
  Default 09:30. See section 5 for the timing gotchas.
- `--rebalance-time REBALANCE_TIME` — time the snapshot clock is
  anchored to. Default 15:50.
- `--print-wait-secs PRINT_WAIT_SECS` — seconds after `--rebalance-time`
  to wait before snapshotting the buffers. Default 10. The imbalance
  print publishes at 15:50:00+, so a 0s wait races it.
- `--submit-by SUBMIT_BY` — hard cutoff (HH:MM ET) for MOC EXIT
  submission. Default 15:55 — this is **NASDAQ's** MOC ENTRY cutoff
  of 15:55:00 ET (SEC release 34-84454, 2018 rule change that
  extended it from the old 15:50). MOC orders may not be
  CANCELED OR MODIFIED after 15:50 ET on NASDAQ; the entry cutoff
  is later. (NYSE keeps a 15:50 MOC entry cutoff — different venue,
  different rule. This strategy targets the NASDAQ close cross
  exclusively — see `live/nasdaq_routing.py`.) This flag ONLY bounds
  the MOC exit batch; the LMT entry batch fires unconditionally so
  the operator is never left with a filled entry and no queued
  exit. If the cutoff was already past at orchestrator entry (no
  orders placed), the run exits rc=4 (safe skip). If it trips
  mid-flow AFTER entries filled, the run exits rc=11 (OVERNIGHT
  EXPOSURE — page).

Execution (two-leg — see the "Execution model" section below):

- `--moc-ack-timeout MOC_ACK_TIMEOUT` — bounded wall-clock (seconds)
  the MOC acceptance audit waits per-oid for a terminal state after
  the MOC batch is sent. An MOC that reaches `Cancelled` /
  `ApiCancelled` / `Inactive` with `filled=0` within this window is
  treated as REJECTED at the exchange and escalates the run to rc=11.
  `TimeoutError` (still `Submitted` at deadline) is presumed accepted;
  MOCs sit `Submitted` until 16:00 and fill there. Default 60. Trade-
  off: shorter = faster operator feedback but risks false rc=0 if a
  rejection propagates later than the window (rare; rejections are
  typically terminal within seconds). Longer = more reliable
  rejection detection but adds up to that many seconds to every
  happy run's wall clock. Late rejections beyond the window are
  caught next morning by `broker.reconcile_positions` at startup.

Safety and data quality:

- `--on-feed-failure {abort,proceed}` — action if the live feed is
  stale, never connected, or fails the coverage check. Default `abort`.
  Note: a feed-*start* failure (adapter raises out of
  `LiveFeed.start()` / `ReplayFeed.start()`) always returns rc=2
  regardless of this flag. That is strictly safer — a dead feed
  produces an empty book downstream anyway, and rc=2 makes the exit
  honest.
- `--min-universe-coverage MIN_UNIVERSE_COVERAGE` — minimum fraction
  of the active universe that must appear in the snapshot for the run
  to proceed. Default 0.5.

LMT entry quote guards (not CLI flags — config constants; override in
`live/config.py`):

- `DEFAULT_MAX_QUOTE_STALENESS_SECS` (default `30.0`). Maximum age
  of the tbbo tick used to price each LMT entry, measured against
  the wall clock (live/paper) or the replay-day decision anchor
  (replay). A symbol whose last tick predates this window is
  skipped with `error="quote_stale"`.
- `DEFAULT_MAX_ENTRY_REL_SPREAD` (default `0.005` = 50 bps).
  Maximum `(ask - bid) / mid` at which the mid is a meaningful
  price. Wider quotes skip with `error="quote_wide_spread"`.
- `DEFAULT_MIN_ENTRY_SUBMIT_COVERAGE` (default `0.5`). Minimum
  fraction of priced-in symbols that must actually reach the
  broker (i.e. NOT skipped by any `quote_*` guard). Below this,
  the orchestrator cancels any entries that did land, exits
  `rc=10`, and does NOT proceed to the MOC batch. Catches
  systemic feed wedges that leave you with a truncated fragment
  of a risk-balanced solve. `rc=10` is a pre-order refusal (no
  exposure), so it stays out of the `rc >= 11` pager band.
- `ENTRY_MAX_LIMIT_PRICE` (default `$5,000,000`). Absolute price
  sanity bound on any single limit price. Catches the Databento
  `UNDEF_PRICE` scaled-sentinel variant (~$9.2B); a legitimate
  BRK-A quote around $700k is well inside the bound.

Skip reasons an operator will see in the logs:

- `no_usable_quote_at_entry` — no tbbo record at all for the symbol.
- `quote_missing_fields` / `quote_non_numeric` / `quote_non_finite`
  — one of bid/ask/mid/ts is None, non-numeric, or NaN/inf.
- `quote_non_positive` — bid or ask <= 0.
- `quote_crossed` — ask < bid.
- `quote_absurd_price` — bid or ask > `ENTRY_MAX_LIMIT_PRICE`.
- `quote_stale` — tick older than `DEFAULT_MAX_QUOTE_STALENESS_SECS`.
- `quote_wide_spread` — spread wider than `DEFAULT_MAX_ENTRY_REL_SPREAD`.
- `quote_naive_ts` / `quote_bad_ts` — feed / adapter bug (tz).

Configuration:

- `--universe-file UNIVERSE_FILE` — active-universe filename at the
  repo root. Default `top_49_universe.txt` — chosen for paper trading
  to avoid known universe issues; the plan is to scale toward 500
  later. The on-disk covariance/forecast/impact artifacts must be
  regenerated for whichever universe you select (see § 6).
- `--reference-stats-path REFERENCE_STATS_PATH` — override for the
  reference-stats parquet. Default `impact/reference_stats.parquet`.
- `--no-live-feed` — skip the Databento live feed entirely. Only useful
  with `--dry-run` for smoke tests; alpha will be zero everywhere.

Broker:

- `--broker-host BROKER_HOST` — TWS host. Default 127.0.0.1.
- `--broker-port BROKER_PORT` — TWS port. Default auto (7497 paper,
  7496 live).
- `--broker-client-id BROKER_CLIENT_ID` — TWS client id. Default 1.
  Use distinct ids per running process.

### Golden commands

Dry-run smoke (no broker contact, no live feed — proves the module
imports, artifact files parse, MVO compiles):

```bash
.venv/bin/python -m live.daily_orchestrator \
  --dry-run --no-live-feed \
  --rebalance-time 00:00 --stream-start-time 00:00 \
  --submit-by 23:59 --on-feed-failure proceed
```

Replay dry-run smoke (broker-free AND no target-book write to the
replay directory — the fastest sanity check that alpha + MVO + sizing
still compile against cached historical data):

```bash
.venv/bin/python -m live.daily_orchestrator \
  --replay-date 2026-05-13 --paper --dry-run
```

Replay dress-rehearsal, broker-free (default replay behavior — no
orders, target book persisted to `live/logs/targets/replay/`):

```bash
.venv/bin/python -m live.daily_orchestrator \
  --replay-date 2026-05-13 --paper
```

Replay dress-rehearsal, exercising the real paper order path
(flatten with `python -m live.panic --paper` between runs to avoid
polluting the next real session's baseline):

```bash
.venv/bin/python -m live.daily_orchestrator \
  --replay-date 2026-05-13 --paper --replay-submit
```

Paper trading (real Databento feed, TWS paper port 7497):

```bash
.venv/bin/python -m live.daily_orchestrator \
  --paper --vol-cap 5000 --lambda- 5e-5
```

Live trading (real Databento feed, TWS live port 7496; requires the
exact confirm string):

```bash
.venv/bin/python -m live.daily_orchestrator \
  --live --live-confirm YES_I_WANT_LIVE_TRADING \
  --vol-cap 5000 --lambda- 5e-5
```

### Recommended production sequence

Progressive bring-up — each step validates a strictly larger surface
than the one before it. Do not skip:

```bash
# 1. Confirm cache HITs without touching the network or writing artifacts.
.venv/bin/python -m live.retrain --refit impact --dry-run

# 2. Actually generate impact/reference_stats.parquet (and the impact
#    vector) — needed for any replay to survive the coverage assertion.
.venv/bin/python -m live.retrain --refit impact

# 3. Broker-free rehearsal against cached historical data. Validates
#    feed -> features -> alpha -> MVO -> sizing -> pre-qualify without
#    touching TWS. Pick any weekday inside your cached window; a
#    Wednesday avoids exchange-holiday adjacency.
.venv/bin/python -m live.daily_orchestrator --replay-date 2026-05-13 --paper

# 4. Real paper broker: LMT entries + MOC exits through the paper
#    account. Flatten the paper account (`panic --paper`) before the
#    next real session — replay fills pollute the diff baseline.
.venv/bin/python -m live.daily_orchestrator \
  --replay-date 2026-05-13 --paper --replay-submit

# 5. Live feed, paper broker — the last step before real money.
#    Run for several sessions and eyeball the target-book / fills logs.
.venv/bin/python -m live.daily_orchestrator --paper

# 6. Live feed, live broker. Real money.
.venv/bin/python -m live.daily_orchestrator \
  --live --live-confirm YES_I_WANT_LIVE_TRADING

# If anything goes wrong at any step, the kill switch:
.venv/bin/python -m live.panic --paper --flatten-with MKT
# (or the --live variant with --live-confirm=... --yes)
```

### Timing diagram

```
   wall clock (America/New_York)
   |
   |  --stream-start-time  (default 09:30, launch by ~15:37 in practice — see section 5)
   |          Databento Live opens; imbalance + tbbo begin buffering
   |
   |  15:40  recent_return window opens — buffer MUST be live by now
   |
   |  15:50  --rebalance-time
   |          NYSE/Nasdaq close-imbalance print publishes at 15:50:00+
   |
   |  15:50 + --print-wait-secs (default 10s)
   |          snapshot buffers; features -> alpha -> MVO -> shares
   |          pre-qualify contracts with TWS; drop unqualified
   |          check MAX_ORDER_QTY, raise ceiling if needed
   |
   |  15:50+ LEG 1: LMT entries sized to (target - current), priced
   |          at the mid of the tbbo tick observed in the SAME 15:50
   |          snapshot (bounded at the decision moment — no look-
   |          ahead in replay), rounded PASSIVELY per side (BUY
   |          floors DOWN to the next tick, SELL ceils UP), and
   |          submitted via broker.buy / broker.sell with
   |          order_type="LMT", limit_price=<numeric>,
   |          exchange=ACTIVE_ROUTING, default tif=DAY. A symbol
   |          whose tbbo is missing / stale / crossed / wider than
   |          DEFAULT_MAX_ENTRY_REL_SPREAD / non-positive / above
   |          ENTRY_MAX_LIMIT_PRICE is SKIPPED — no LMT, no MOC,
   |          no exposure (order_id is None). If more than 50%
   |          (1 - DEFAULT_MIN_ENTRY_SUBMIT_COVERAGE) of priced-in
   |          symbols are guard-skipped, any entries that DID
   |          land are cancelled and the run exits rc=10.
   |
   |  15:50+ LEG 2 (fill-gated dispatcher): a single-threaded fast-
   |          polling loop monitors every entry's status concurrently.
   |          As each entry becomes terminal Filled, its opposite-side
   |          MOC is submitted IMMEDIATELY, sized off a fresh
   |          broker.get_positions() for that symbol. Zero-fill entries
   |          (LMT never crossed → Cancelled by the sweep with
   |          filled=0) get NO MOC. Dispatcher WORKS UNTIL THE SWEEP
   |          DEADLINE — DEFAULT_ENTRY_WORKING_SECS defaults to
   |          None, which means the dispatcher does NOT walk away at
   |          the old MKT-era 30s cap; passive limits routinely cross
   |          many minutes past 15:50 in the closing ramp. A
   |          staleness check (baseline+signed_fill == snapshot)
   |          DEFERS the MOC to the sweep whenever the position book
   |          lags the fill event (round-4 fix: IB emits execution
   |          reports before the position update — the naive
   |          dispatcher would have sized an MOC off a stale PRE-fill
   |          snapshot).
   |
   |  ~15:51  FIRST SWEEP (immediately after the dispatcher): cancels
   |          any entry still non-terminal, verifies each dispatcher
   |          MOC is still live (drops coverage credit for any that
   |          flipped terminal-Cancelled/Inactive with filled=0), takes
   |          a fresh get_positions, and submits MOCs for every
   |          managed-universe symbol whose signed residual
   |          `held + Σ covered_MOC` is nonzero. Two-sided: TOPS UP an
   |          under-cover AND TRIMS an over-cover.
   |
   |  <=15:55 SECOND SWEEP at the real sweep_deadline (= submit_by -
   |          dynamic margin; margin scales with book size — see
   |          _compute_sweep_deadline). The SLEEP from ~15:51 to the
   |          deadline is PUMPED with a periodic broker.get_positions
   |          call so the ib_async connection stays alive during the
   |          wait (round-4 fix — otherwise the loop would starve and
   |          risk wedging the post-close poll). On a normal day the
   |          second sweep issues ZERO orders — its value is
   |          confirmation under the ACTUAL "book state we hand to
   |          16:00": a fresh snapshot several minutes later has
   |          caught up on any IB position-propagation lag, verifies
   |          every prior MOC (dispatcher + first sweep) is still
   |          live at the exchange, and re-fires for any residual
   |          (cancel-race fills, late exchange rejections). This is
   |          the GUARANTEE — residual / baseline positions and late
   |          cancel-race fills still get flattened here. Skipped
   |          entirely in --replay-submit mode where the wall clock
   |          is unrelated to the historical replay date; the first
   |          sweep is authoritative in replay.
   |
   |  15:50+ Coverage cross-check: independent per-symbol re-poll
   |          via broker.get_order_status to verify every expected
   |          held position had an MOC counter-order. Uncovered =>
   |          rc=11 (pre-close early warning).
   |
   |  15:50+ MOC acceptance audit (bounded --moc-ack-timeout, default 60s):
   |          per-oid wait_for_fill for a terminal state. Any MOC
   |          reaching Cancelled/ApiCancelled/Inactive with filled=0
   |          => REJECTED at exchange => rc=11 (pre-close early
   |          warning; NASDAQ's 15:55 cutoff was likely blown).
   |
   |  15:55  --submit-by  hard cutoff for the MOC EXIT batch
   |          (NASDAQ MOC ENTRY cutoff — SEC 34-84454). Entries always
   |          fire; MOC batch legs past the cutoff are skipped and
   |          logged (=> rc=11 mid-flow).
   |
   |  16:00  official NASDAQ close cross; MOC prints.
   |
   | 16:00-16:05  POST-CLOSE POSITION POLL — the AUTHORITATIVE
   |          exposure verdict. broker.get_positions() polled with
   |          exponential backoff; a nonzero managed-universe symbol
   |          at 16:05 (OR every poll failing) => rc=11. Replaces
   |          inference with a real position check.
   |
   |  16:05+ Fill audit polls broker.wait_for_fill, writes
   |          live/logs/fills/YYYY-MM-DD.jsonl (best-effort, not
   |          rc-gating).
```

### Safety guarantees

- Default mode is `--paper`. Live mode requires both `--live` and the
  exact confirmation string; the broker refuses paper-mode connections
  against a live port.
- `--dry-run` short-circuits before any broker connection.
- Replay is broker-free by default; `--replay-submit` is the only path
  from a replay run into the paper broker.
- Coverage assertion in `_build_live_day_frame` raises before any broker
  connect if the live snapshot covers less than
  `--min-universe-coverage` of the universe. Combined with
  `--on-feed-failure abort` (default), the orchestrator exits with rc=2.
- `--lambda- <= 0` returns rc=9 without contacting the broker. Kept
  below the `rc >= 11` pager threshold — invalid args are a safe skip,
  not an exposure.
- An empty `share_targets` after alpha/MVO returns rc=5 without
  connecting to the broker. This prevents an all-zero book from
  flattening every existing position.
- Out-of-universe positions are pinned to their current quantity in
  the extended target so they diff-out to zero — the strategy never
  touches a position it did not put on. Both LMT entries and MOC exits
  respect this.
- **LMT entry mass-skip guard (rc=10).** If more than
  `1 - DEFAULT_MIN_ENTRY_SUBMIT_COVERAGE` (default 50%) of the
  priced-in symbols are skipped by any quote guard (systemic feed
  wedge, mass-stale ticks), the orchestrator CANCELS any entries
  that did land and exits `rc=10` BEFORE submitting the MOC batch.
  A truncated fragment of a risk-balanced solve is worse than no
  book at all — it's a non-random slice of names that happened to
  have clean quotes, executed in the closing ramp against a
  covariance fit on the full set.
- Every symbol is pre-qualified through TWS before the per-leg loop.
  Symbols IB refuses to qualify are dropped from the target with a
  warning; the batch continues.
- Legs whose absolute diff would exceed `Broker.MAX_ORDER_QTY` (default
  10,000) cause the ceiling to be raised for this session, with the
  computed new value logged. Nothing crashes on an oversized leg.
- **Structural safety invariant of the two-leg model.** The MOC exit
  quantity is derived from `broker.get_positions()` AFTER the entry
  fills settle, NEVER from `target - current` or the entry's requested
  quantity. A partial LMT fill (or no fill at all) therefore cannot
  cause an over-flatten into an opposite overnight exposure.
  Enforced by both
  `_submit_single_moc` and `_sweep_moc_deadline` not accepting any
  target-side argument in their signatures; a compile-time regression
  test (`test_fill_gated_dispatch_never_regresses_leg_2_sized_from_target`)
  pins these signatures.
- **Fill-gated dispatcher fires each MOC as soon as its entry lands.**
  Concurrent from the operator's perspective (single-threaded fast
  poll loop; ib_async is not thread-safe so we do NOT launch a
  ThreadPoolExecutor). Total MOC-send latency is bounded by fill
  latency, not `n * per_symbol_timeout`. Zero-fill entries get NO
  MOC.
- **Fill-gated staleness check (round-4).** IB emits execution
  reports BEFORE it updates the position book. A reducing entry
  against a nonzero baseline that filled between the dispatcher's
  status read and its position snapshot would otherwise be
  MOC-sized off the STALE PRE-fill position. The dispatcher
  computes the deterministic expected post-fill position
  (`baseline + signed_fill`) and, whenever the tick snapshot
  disagrees, DEFERS the MOC to the sweep — which takes a fresh
  snapshot several minutes later, well past any plausible
  position-propagation lag.
- **Deadline sweep is the GUARANTEE — TWO sweeps.**  Round-4
  runs the sweep at TWO points: once IMMEDIATELY after the
  dispatcher (~15:51) and once at the real deadline
  `submit_by - sweep_margin` (~15:54:42 for 49 names). The first
  gives the operator EARLY exposure detection (pre-close signals
  fire in time to run `panic` before 16:00); the second is
  belt-and-braces after the ib_async position book has fully
  caught up on any propagation lag and after any late exchange
  rejections have flipped the affected MOCs to terminal
  Cancelled. Both sweeps cancel any still-non-terminal entry,
  verify each prior MOC (dispatcher + first sweep) is still live
  at the exchange, take a fresh `get_positions`, and issue MOCs
  for every managed-universe symbol whose signed residual is
  nonzero (TWO-SIDED — TOPS UP an under-cover, TRIMS an
  over-cover). Between the sweeps, `_sleep_until(sweep_deadline)`
  is PUMPED with a periodic `broker.get_positions()` call so the
  ib_async connection stays alive during the wait. Sweep margin
  scales with book size — 15s floor at 49 names, ~121s at 500
  names. On a normal day the second sweep issues zero orders. In
  `--replay-submit` mode the second-sweep wall-clock sleep is
  skipped (the wall clock is unrelated to the historical replay
  date) but the second sweep itself still runs immediately so
  the code path is exercised end-to-end.
- **Ground-truth coverage cross-check.** After the MOC batches, an
  independent per-symbol check re-polls each entry's true fill via
  `broker.get_order_status`, computes `expected_held + moc_signed`,
  and any symbol with non-zero remainder escalates to rc=11 (early
  warning). Closes partial `get_positions()` snapshots, stale
  non-terminal reads, and slow-TWS budget exhaustion.
- **MOC exchange-acceptance audit.** TWS's `placeOrder` returns the
  oid synchronously BEFORE any exchange ack. An MOC submitted past
  NASDAQ's 15:55 exchange entry cutoff is API-accepted then
  Cancelled at the exchange. `_audit_moc_acceptance` waits up to
  `--moc-ack-timeout` seconds per-oid for terminal state; any MOC
  reaching `Cancelled` / `ApiCancelled` / `Inactive` with `filled=0`
  escalates to rc=11 (pre-close early warning).
- **Post-close position poll (16:00-16:05) is the AUTHORITATIVE
  exposure verdict.** After the auction has printed, poll
  `broker.get_positions()` with exponential backoff until the
  managed-universe subset is flat or 16:05 elapses. A nonzero
  position at 16:05 → rc=11. Every poll failing → pessimistic
  rc=11 (cannot verify flat = cannot claim flat). This replaces
  inference as the definitive signal; the pre-close signals above
  are EARLY warnings that short-circuit to rc=11 immediately so
  the operator has time to run panic before 16:00.
  Three round-2 safety bounds keep this from silently declaring
  FLAT early: a **not-before** lower bound (default `16:00 +
  DEFAULT_POST_CLOSE_POLL_SETTLE_SECS`; the poll refuses to return
  FLAT before this), a **min-consecutive-flat-reads** counter
  (default 2; a single transient empty response is not
  authoritative), and a **max-window** clamp
  (`POST_CLOSE_POLL_MAX_WINDOW_MINS = 15`; a misconfigured
  `DEFAULT_POST_CLOSE_POLL_UNTIL_TIME` cannot make the poll spin
  for hours). Also applies to `--replay-submit` runs — any run
  that actually reached the broker gets the same post-auction
  verification. Round-3 adds two more:
  * If the process reaches the poll BEFORE `not_before` on a
    live/paper run, it **sleeps until `not_before`** rather than
    starting a poll that would fail-closed on stale reads. The
    poll deadline is then clamped to leave room for
    `min_consecutive_flat_reads` reads.
  * If the process reaches the poll on a `--replay-submit`
    rehearsal BEFORE `not_before`, the poll is **SKIPPED with a
    WARNING** — a historical replay's fills are queued for
    today's 16:00 close (not the replay date's), so the poll
    cannot verify by definition. The operator must manually run
    `python -m live.panic --paper` to flatten the paper account
    before the next real session; rc=0 is preserved iff
    pre-close signals were clean.
  * The entire post-close block is wrapped in `try/except`.
    Any raise there (a config parse error, an unexpected schema
    issue) is treated as pessimistic EXPOSED (rc=11) — cannot
    verify equals cannot claim flat.
- The per-leg MOC submit re-checks the wall clock against
  `--submit-by` before each order — legs that would land past the
  cutoff are skipped, not silently rejected by TWS. Replay HONOURS
  this cutoff by default; pass `--replay-ignore-cutoff` to
  neutralise it for an afternoon rehearsal (the wall clock is
  unrelated to the historical date, but the paper orders are real).
- Broker-connect failures (live-port guard tripping, TWS down, bad
  client_id) return rc=8 rather than a bare traceback. No orders were
  placed at that point.
- **Tail I/O failures cannot downgrade an rc=11 exposure verdict.**
  Target-book persist and the JSONL fill audit are best-effort with
  `try/except logger.exception(...)`. The exposure decision is
  computed from all in-memory signals BEFORE any disk I/O; a
  disk-full raise on the amended target record does not propagate to
  `main()` and does not drop rc=11 to rc=1.

### Exit codes

**Pager predicate: `rc >= 11` is the ONLY code that implies a live book
at close.** Every other code in [2, 10] is a safe pre-order refusal
or a clean shutdown. Wire the pager to page on `rc >= 11` and to log
(don't page) on any other non-zero.

- `0` — MVO solved and orders (or dry-run / broker-free replay
  summary) completed. For a `--paper` or `--live` run, both legs
  fired, the coverage cross-check found no gaps, and the MOC
  acceptance audit found no exchange rejections.
- `2` — live feed / feature build failed. Includes the coverage
  assertion (a thin snapshot) AND any exception out of
  `LiveFeed.start()` / `ReplayFeed.start()` (e.g. cold-cache replay of
  a weekend, adapter auth error). rc=2 fires regardless of
  `--on-feed-failure` for a feed-*start* exception — strictly safer.
- `3` — MVO solver returned a non-optimal status.
- `4` — clock is past `--submit-by` at the moment the orchestrator was
  about to submit; nothing was placed. Also fires in replay mode with
  `--replay-submit` unless `--replay-ignore-cutoff` is passed. Distinct
  from rc=11 — rc=4 is a SAFE PRE-ORDER SKIP.
- `5` — target book was empty and `--dry-run` was not set. The broker
  was not contacted.
- `6` — `--replay-date` + `--live` was requested. Refused. No broker
  contact.
- `7` — `--replay-submit` was requested without `--replay-date`. No
  broker contact.
- `8` — the broker refused the connection. Most commonly the live-port
  guard on a paper-port mismatch, TWS not running on the expected
  port, or a `--broker-client-id` collision.
- `9` — `--lambda- <= 0` (invalid argument). No broker contact. Below
  the pager threshold — this is a validation failure, not an exposure.
- `10` — LMT entry submit coverage below
  `DEFAULT_MIN_ENTRY_SUBMIT_COVERAGE` (default 0.5). Either a
  systemic feed wedge left too many symbols with unusable quotes,
  or a burst of `quote_*` guard skips left the surviving entry set
  too small to be a coherent risk-balanced book. Any entries that
  DID land are cancelled BEFORE this return, so no exposure
  remains. Distinct from rc=5 (empty share_targets), which fires
  BEFORE entry submission on an empty MVO result. Below the pager
  threshold — this is a pre-order refusal, not an exposure.
- `11` — **OVERNIGHT EXPOSURE POSSIBLE — page the operator.** Fires
  on any of two independent classes of signal:
  * **Pre-close (EARLY warning, before 16:00)** — page immediately
    so the operator has time to run `panic` before the auction.
    Fires on: MOC legs skipped past `--submit-by` mid-flow (from
    either the fill-gated dispatcher's per-tick cutoff refusal OR
    the sweep's own past-cutoff refusal), `get_positions` raised
    during the MOC batch, the snapshot came back stale-empty after
    fills, the coverage cross-check found uncovered symbols
    (partial snapshot / stale non-terminal `get_order_status`),
    the MOC acceptance audit found exchange rejections (MOC placed
    after NASDAQ's 15:55 cutoff), the sweep's belt-and-braces
    cancel step failed for an entry (round-2 addition — an entry
    we could not pull is by definition an uncontrolled position
    source, may still fill mid-window into an unhedged overnight
    position), OR the fill-gated dispatcher's per-symbol MOC
    submission raised at the broker (round-4 addition — a
    `broker.buy`/`sell` exception may leave the request in an
    indeterminate state at TWS; a bare "assume nothing happened"
    would defer to the sweep and risk a double-flatten. The list
    lands in `fill_gated_submit_errors` and appears in the ERROR
    log line as `Fill-gated submission errors: ...`). The ERROR
    log line is tagged `OVERNIGHT EXPOSURE (pre-close)` and lists
    cancel-failed entries in `cancel_failed_symbols`.
  * **Post-close (AUTHORITATIVE, 16:00-16:05)** — polls
    `broker.get_positions()` with backoff; a nonzero managed-
    universe position at 16:05 OR every poll failing pessimistically
    → rc=11. Tag: `OVERNIGHT EXPOSURE (post-close)`. This is the
    definitive verdict — no inference; a real position check.
  Either safeguard alone returns rc=11. Both are load-bearing;
  either running silently on a real exposure would drop the page.

### Execution model — two-leg

**The strategy is intraday, not overnight.** The forecast predicts the
15:50→16:00 close-auction return. To capture that PnL the live path
enters at 15:50 and exits at the 16:00 close, so the book is FLAT
overnight. Each trading day is independent.

Execution is two batches, in strict sequence, per name in the target book:

**LEG 1 — LMT entries at the tbbo mid, PASSIVE per side.** For each
name where `diff = target_shares - current_shares != 0`:

- **Price**: `(bid_px_00 + ask_px_00) / 2` observed in the 15:50
  DECISION-TIME snapshot (the same `feed.snapshot()` that fed the
  MVO, TIME-BOUNDED at the decision moment so replay cannot leak
  the 16:00 closing print into the entry price). Rounded PASSIVELY
  per side: BUY floors DOWN to the next US-equity tick ($0.01 above
  $1, $0.0001 below), SELL ceils UP. This is a correctness
  requirement — a common one-cent spread puts the mid on a
  half-penny, and a symmetric half-up round on both sides would
  send BUYs at the ask (marketable, always fills) and SELLs half a
  cent above the ask (rarely fills), producing a systematically
  net-long realized book the covariance never priced.
- **Submission**: `broker.buy(sym, qty, order_type="LMT",
  limit_price=<numeric>, exchange=ACTIVE_ROUTING)` /
  `broker.sell(...)` at the broker default `tif="DAY"`.
- **Working time**: `DEFAULT_ENTRY_WORKING_SECS` defaults to
  `None`, meaning the dispatcher works EACH ENTRY UNTIL THE SWEEP
  DEADLINE (~15:54:42 on the 49-name book). Passive limits
  routinely cross many minutes past 15:50 as the closing ramp
  gathers momentum — capping the wait at the MKT-era 30s would
  have the sweep cancel almost every entry before it could fill.
- **Sweep at 15:54:42**: any LMT still non-terminal is CANCELLED
  before the MOC batch is sized. An entry that never crossed
  simply leaves no position and no MOC — no exposure risk added.
- **Quote guards** (all skip the symbol; the run-return records
  `error=<reason>`, `order_id=None`, and — where computable —
  the raw mid we would have priced at). A symbol whose tbbo is
  missing / stale (> `DEFAULT_MAX_QUOTE_STALENESS_SECS`,
  default 30s) / crossed / wider than
  `DEFAULT_MAX_ENTRY_REL_SPREAD` (default 50 bps) / non-positive /
  above `ENTRY_MAX_LIMIT_PRICE` (Databento sentinel scale-leak
  guard, default $5M) is SKIPPED. No MKT fallback — the operator's
  policy is that no entry = no exposure.
- **Mass-skip guard**: if MORE than
  `1 - DEFAULT_MIN_ENTRY_SUBMIT_COVERAGE` (default 50%) of
  priced-in symbols are guard-skipped, any entries that DID land
  are CANCELLED and the run exits `rc=10` before the MOC batch.
  This is what stops a systemic feed wedge from committing the
  book to a truncated fragment of a risk-balanced solve.

Fires unconditionally with respect to the `--submit-by` MOC exit
cutoff — LMT entries are the ONLY lateness the strategy is
sensitive to (the signal decays through the 15:50-16:00 window),
so any delay to the sweep or acceptance audit does NOT delay the
entry batch.

*Why passive LMT and not MKT?* Earlier iterations used plain MKT
entries and accepted the spread paid on entry. Under a large-cap
top-49 book at 15:50-15:55, mid-priced passive limits fill often
enough that the spread saved on the entries that DO cross
materially improves realized PnL vs. the alpha the forecast
predicts. The two accepted tradeoffs are (a) LOWER FILL RATE
than MKT (some names simply don't cross, and are not entered —
their contribution to the intended book is 0), and (b) the
staleness / mass-skip guards, which turn a systemic feed wedge
into an rc=10 refusal instead of a silent partial book. Both are
worth it under passive rounding per side; both would be dangerous
without it (see the correctness note above).

**LEG 2 — per-symbol FILL-GATED MOC + DEADLINE SWEEP.**

Two stages, both sized off `broker.get_positions()` (never target-side
data). Together they guarantee every managed-universe symbol with a
nonzero position at 15:55 has an MOC away, AND that no symbol whose
entry never landed picks up a naked MOC.

*Fill-gated dispatcher.* A single-threaded fast-polling loop iterates
every entry once per tick (default 0.25s). Each tick READS ORDER
STATUS FIRST (fast, local ib_async cache), then — if any entry is
newly terminal-Filled — takes a FRESH `broker.get_positions()`
snapshot (whose internal 0.5s sleep gives IB time to propagate the
position update AFTER the fill event; round-3 fix for the
snapshot-before-status race). Each newly-terminal entry's
opposite-side MOC is submitted IMMEDIATELY, sized off the
freshly-taken snapshot. If `filled == 0` (Cancelled / Inactive /
Filled-zero), NO MOC is submitted from this path — this is the
operator's stated goal: "prevent the accidental MOC exit without
ever having entered." The loop is concurrent from the operator's
perspective (each symbol's MOC fires as soon as its own entry
lands, not after all N entries) without spawning threads
(ib_async is not thread-safe).

*Deadline sweep.* At `submit_by - sweep_margin` (margin scales with
book size — see `_compute_sweep_deadline`; 15s floor at 49 names,
~121s at 500 names), any entry still non-terminal is cancelled, a
fresh `get_positions` is taken, and MOCs are submitted for every
symbol whose net "held after entry fills + dispatcher MOC coverage"
is nonzero. The sweep is TWO-SIDED (round-3 fix): it iterates the
UNION of "symbols currently held" AND "symbols the dispatcher
already MOC'd", so a symbol that is FLAT NOW but has a dispatcher
MOC still queued (over-covered because the dispatcher's snapshot
lagged the fill) gets a TRIM MOC in the opposite direction —
restoring the invariant `held + Σ signed_MOC == 0`. Rejected
dispatcher MOCs (terminal Cancelled with `filled=0`) have their
coverage credit dropped so the sweep can resubmit while the 15:55
window is still open.

This is the GUARANTEE for the failure modes the dispatcher alone
cannot cover:

- A **residual/baseline position** carried in from a prior day (its
  symbol may not have an entry today at all — the dispatcher's
  fill-triggered path never fires for it).
- A symbol whose entry submitted successfully but arrived
  non-terminal at the sweep deadline (dispatcher gave up, sweep
  cancels the entry then flattens whatever position landed).
- A **reducing entry against a nonzero baseline** where IB's
  position update lags the fill event. Dispatcher's stale snapshot
  over-sizes the MOC → sweep sees the true post-fill position (0)
  and covered_qty (-X), computes residual (-X), sends BUY MOC X as
  the trim so the net MOC across both legs = 0. (Round-3 fix for
  the SAFETY reviewer's critical finding.)
- A **dispatcher MOC that got rejected at the exchange**
  (short-sale restriction, margin, contract issue). The sweep
  detects the terminal-not-Filled status, drops the coverage
  credit, and resubmits a fresh MOC while the ~4 minutes to
  15:55 remain. (Round-3 fix for the SAFETY reviewer's third
  finding.)

**The core safety invariant.** Both the dispatcher's per-symbol MOC
send (`_submit_single_moc`) and the sweep's MOC batch
(`_sweep_moc_deadline`) size their `qty` argument from a fresh
`broker.get_positions()` snapshot, NEVER from `target - current` or
from the entry's requested quantity. A partial LMT fill (say 50%)
therefore produces a 50% MOC counter and the book returns to zero.
An entry that never crossed leaves no position and no MOC — a
"skipped" MOC for a symbol that never traded is the correct
outcome, not an exposure.
A target-diff MOC would over-flatten a partial into an opposite
overnight exposure — the exact bug this two-leg model exists to
prevent. Structurally enforced: neither `_submit_single_moc` nor
`_sweep_moc_deadline` accepts any target-side argument in its
signature; a regression test locks the signatures so a refactor
that added one would fail at review.

**Coverage cross-check.** After the MOC batch is submitted, an
independent per-symbol check re-polls each entry's TRUE fill via
`broker.get_order_status(oid)`, computes `expected_held =
current_baseline + Σ signed_fills`, and any symbol with
`expected_held + moc_signed_by_sym != 0` is flagged as uncovered →
rc=11. `moc_signed_by_sym` is built by AGGREGATING signed_qty
across ALL MOC legs per symbol (round-3 fix — the dispatcher AND
sweep can each emit a leg for the same symbol on the top-up
path; a last-wins dict comprehension would drop the dispatcher
leg and false-page the fix's own success case). MOC legs that
are already visibly terminal-not-Filled at coverage-check time
are EXCLUDED from the aggregation (round-3 fix — a rejected MOC
covers nothing; counting it would under-count exposure).

This closes:

- **Partial `get_positions()` snapshots** — TWS reconnect returns
  3 of 10 held names; the missing 7 would ride unhedged.
- **Wait-budget-exhausted** — an entry status truncated by the
  dispatcher's per-tick deadline marks `filled_qty=0` for the
  remainder, but the LMTs may truly have filled at the exchange.
  The cross-check re-polls ground-truth via `broker.get_order_status`.
- **TWS event-stream brownout** — `get_order_status` returns
  successfully with a stale non-terminal status. The cross-check
  treats non-terminal-at-cross-check-time as a staleness signal and
  assumes pessimistic full fill (overcounting exposure > silently
  missing it).
- **MOC rejections invisible to the acceptance audit** — the
  round-3 exclusion of terminal-rejected MOCs from
  `moc_signed_by_sym` means the coverage cross-check catches a
  rejection even if the audit was misconfigured or the audit's
  wait budget elapsed before the exchange got around to
  Cancelling the MOC.

**MOC exchange-acceptance audit.** `broker.buy`/`sell` returns the oid
the moment TWS API-accepts `placeOrder` — BEFORE any exchange ack.
An MOC submitted past NASDAQ's 15:55 MOC ENTRY cutoff (SEC release
34-84454) is API-accepted, an oid comes back, but the exchange
REJECTS it and the order becomes terminal `Cancelled` with `filled=0`
within seconds to a minute. `_audit_moc_acceptance` waits up to
`--moc-ack-timeout` (default 60s) per-oid for a terminal state; any
MOC reaching `Cancelled` / `ApiCancelled` / `Inactive` with
`filled=0` is treated as REJECTED → rc=11 (pre-close, early). A
`TimeoutError` (still `Submitted` at the deadline) is presumed
accepted; MOCs sit `Submitted` until 16:00.

**Post-close position poll** (16:00-16:05) is the AUTHORITATIVE
verdict — see the previous section. The audit above is an EARLY
warning that fires before 16:00; the post-close poll is the
definitive "did the book actually flatten" check that runs after
the auction has printed.

**Trade-off.** The pre-close audit + post-close poll combination is
belt-and-braces. The audit catches most rejections in 10-60s so
the operator can run `panic` before 16:00. The post-close poll
catches anything the audit misses (a late rejection, a fill that
did not zero the position, a symbol the audit didn't check).
Widen `--moc-ack-timeout` on chronically slow venues; do NOT
disable the post-close poll — it is the only signal that verifies
the book is FLAT after the auction (rather than just verifying that
MOCs were accepted).

**Tail I/O is best-effort.** The rc=11 verdict is computed from all
in-memory exposure signals (`moc_skipped`, `moc_positions_unknown`,
`empty_snapshot_after_fills`, `uncovered_symbols`, `unfilled_moc_syms`,
`post_close_nonzero_symbols`, `post_close_poll_error`) BEFORE any
disk write. `_write_target_record` and the JSONL fill audit both run
inside `try/except logger.exception(...)`. A disk-full raise on the
amended target record does not propagate to `main()` and does not
drop the exposure verdict from rc=11 to bare-exception rc=1.

## 5. Timing gotchas

The out-of-the-box `--stream-start-time 09:30` default is convenient
if you already have a systemd unit or a shell that fires at market
open, but you do not have to keep the feed alive for six hours.

- `--stream-start-time` only gates the feed when you launch EARLY. If
  you launch after that time, the feed starts IMMEDIATELY on launch.
  So the flag is a floor, not a schedule.
- **The real timing constraint is the 15:40 feature floor.**
  `recent_return` uses the 15:40:00-15:49:59 mid-price window, so the
  buffer must be capturing tbbo by 15:40. Launching "a few minutes
  before 15:50" is TOO LATE — the `recent_return` cross-section will
  silently truncate to only the last minute or two of the window and
  the alpha will skew. There is no error for this; the day_df will
  build with a short window and the coverage assertion will not
  necessarily catch it.
- Practical schedule: launch by **~15:37 ET** so the Live client has
  headroom to connect, subscribe, and receive the first tbbo tick
  before 15:40. Set `--stream-start-time 15:35` (the effective floor)
  and schedule the process to fire a minute before that.
- ADV / vol come from `reference_stats` (retrain-produced), NOT from
  the intraday buffer. The buffer only needs data from 15:40 onward.
  This is why a 15-minute pre-close window is enough — you are not
  reconstructing daily statistics on the fly.

A cron-style schedule that minimises live-imbalance airtime:

```
34 15 * * 1-5   .venv/bin/python -m live.daily_orchestrator --paper --stream-start-time 15:35
```

## 6. The retrain script

```bash
.venv/bin/python -m live.retrain --help
```

### CLI reference

- `--refit {all,covariance,forecast,impact}` — which artifact set to
  rebuild. Default `all`.
- `--window-months WINDOW_MONTHS` — rolling window in months for the
  forecast and impact fits. Default 12.
- `--covariance-window-days COVARIANCE_WINDOW_DAYS` — days of daily
  bars for the covariance fit. Default 730 (about 24 months).
- `--end-date END_DATE` — explicit end date, YYYY-MM-DD, exclusive per
  Databento's convention. Default yesterday.
- `--n-factors N_FACTORS` — number of SVD factors for the covariance.
  Default 10.
- `--broad-universe-file BROAD_UNIVERSE_FILE` — broad-universe file
  used for the Databento historical scan. Default `universe.txt`.
- `--active-universe-file ACTIVE_UNIVERSE_FILE` — active universe used
  to align the saved vectors. Default `top_49_universe.txt`.
- `--forecast-alpha FORECAST_ALPHA` — Lasso regularization strength for
  the forecast fit. Default 1e-4.
- `--dry-run` — print the plan (windows, symbol counts, artifact paths)
  and skip both the Databento materialize and the writes.
- `--allow-fetch` — authorise the coverage resolver to submit a
  Databento Batch API job when the manifest cannot serve a request.
  Off by default; a MISS raises with a helpful message so accidental
  batch charges cannot happen from a stale end-date. Must be combined
  with `--yes` (or an interactive `yes` on a TTY) before any batch job
  is submitted.
- `--yes` — non-interactive confirmation to proceed with fetches
  authorised by `--allow-fetch`. Batch API calls require BOTH
  `--allow-fetch` AND `--yes` (belt and braces). Without `--yes` a
  TTY prompt is offered; on a non-interactive terminal the run refuses
  and returns rc=10.

### Exit codes

- `0` — refit completed and artifacts were atomically written (or the
  `--dry-run` plan was printed).
- `9` — coverage MISS on one or more schemas and `--allow-fetch` was
  not set; nothing was fetched, nothing was written.
- `10` — coverage MISS with `--allow-fetch` set but the confirmation
  was refused (no `--yes` and no TTY, `no` at the prompt, or Ctrl-D /
  Ctrl-C). Nothing was fetched.

See "Cache-aware fetch guard" below for how the resolver decides
HIT vs MISS and how the near-miss log names the closest parquet.

### Cadence recommendation

- Forecast and impact: weekly. The 12-month window ships along, and the
  imbalance signal's regime does not sit still — a shorter refresh
  keeps the coefficients close to the current cross section.
- Covariance: monthly. The factor structure is more stable and the
  24-month window means each refit is a bigger Databento pull.

Run `--refit all` weekly and let the covariance be rebuilt over the
same 24-month window each time if the batch cost is acceptable;
otherwise split the schedule.

### What each refit writes

Every write is staged to a `.tmp` file in the destination directory
and then `os.replace`d into place — either the daily reads the previous
version or the new one, never a torn file.

| `--refit`  | Path                                             | Type    |
| ---------- | ------------------------------------------------ | ------- |
| covariance | `covariance/loadings_return.npy`                 | ndarray |
| covariance | `covariance/idiosyncratic_diag_return.npy`       | ndarray |
| covariance | `covariance/factor_cov_matrix.npy`               | ndarray |
| covariance | `covariance/cov_matrix_return.npy`               | ndarray |
| covariance | `covariance/name_vol.npy`                        | ndarray |
| forecast   | `forecast/forecast_pipe.joblib`                  | sklearn.Pipeline |
| impact     | `impact/impact_gamma_vec.npy`                    | ndarray |
| impact     | `impact/reference_stats.parquet`                 | polars.DataFrame (per-symbol ADV, vol) |

### Cost note

The covariance refit does a materialize call on the `ohlcv-1d` schema
over `ALL_SYMBOLS` for the full window. The first time you run it
against a fresh cache this triggers a Databento batch job and consumes
API budget; subsequent runs hit the parquet cache and are cheap. Use
`--covariance-window-days 30` for the very first smoke test.

### Coupling with the daily

- Feature engineering lives in `live/features.py` and is shared. Both
  scripts must agree on `ALPHA_FEATURES` order. If you change the
  feature set, retrain first, and only then let the daily pick up the
  new pipeline.
- The optimizer indexes vectors positionally by the active universe.
  Changing the active-universe file requires rebuilding all three
  artifact sets in one pass, or the shapes will not line up and the
  daily will fail its shape checks before opening the feed.

## 7. Cache-aware fetch guard

Every historical read from `retrain` and `daily_orchestrator
--replay-date` routes through the coverage manifest in `live/cache.py`
before it can reach the Databento Batch API. This is the guardrail that
turned "stale end-date silently triggers a paid fetch" into a coded
refusal.

### What the resolver does

1. Fast path: try the adapter's exact-hash parquet cache
   (`data/dbn_cache/generic/<md5>.parquet`). Hit → serve verbatim.
2. Fall back to the coverage manifest, which fingerprints every
   parquet in `data/dbn_cache/generic/` on schema, symbol set, and
   per-symbol date range. If an existing parquet's contents cover
   the requested `{schema, [start, end), symbols}` slice, the
   resolver builds a filtered `pl.LazyFrame` off it — no new cache
   slot, no fetch. This is the "coverage HIT".
3. Otherwise MISS. The resolver refuses to fetch unless
   `--allow-fetch` was passed (and, for retrain, `--yes` too — see
   section 6). A refused MISS raises `CoverageError` with the nearest
   parquet, the shortfall, and the reason so the operator can diagnose
   without hand-inspecting the cache.

### HIT criteria

- Schema fingerprint matches (imbalance / tbbo / statistics / ohlcv-1d
  / ohlcv-1m / definition; ohlcv-1d vs 1m disambiguated on
  `ts_event.time()` cadence).
- Parquet's date range brackets the request window, floored to the
  nearest weekday (a parquet ending Friday covers a request whose
  exclusive end lands on Saturday or Sunday).
- Every required symbol appears in the parquet AND its own per-symbol
  range brackets the request (guards against silent IPO / delisting
  false HITs).

### Documented residual gaps

- **Exchange holidays** are not modelled — a request whose closest
  weekday falls on a holiday not in the parquet will still false-MISS.
  A trading calendar is out of scope.
- **Interior-day gaps** inside a parquet are not checked; only
  endpoints. Batch pulls are contiguous in practice, so this rarely
  bites — accept it as a known limitation.

### Sidecar manifest

At `data/dbn_cache/generic/.coverage_manifest.json`, keyed on
`(mtime_ns, size_bytes)` per parquet so a rewrite that preserves
mtime still invalidates. Purely an optimisation — safe to delete;
next run rebuilds it cheaply (one streaming group-by per parquet).

### Flag summary

| Command                               | Flag             | Effect                                                                 |
| ------------------------------------- | ---------------- | ---------------------------------------------------------------------- |
| `retrain`                             | `--allow-fetch`  | Authorise batch fetch on MISS. Requires `--yes` co-gate.               |
| `retrain`                             | `--yes`          | Second gate for `--allow-fetch`. Non-interactive confirmation.         |
| `daily_orchestrator --replay-date`    | `--allow-fetch`  | Authorise historical materialise on MISS. Single gate (cheap pull).    |
| `daily_orchestrator` (live path)      | `--allow-fetch`  | Ignored — the live path uses the websocket feed, no historical cache. |


## 8. Files and directories

| File                         | Responsibility                                                                     |
| ---------------------------- | ---------------------------------------------------------------------------------- |
| `config.py`                  | Repo paths, env resolution, universe loader, time helpers, JSON-lines logging.     |
| `features.py`                | Polars feature engineering shared by daily and retrain; covariance + impact builders. |
| `live_feed.py`               | Threaded wrapper around `databento.Live`; symbology map + record buffering.        |
| `replay_feed.py`             | Adapter-backed historical stand-in for `LiveFeed`; API-compatible for `--replay-date`. |
| `optimizer.py`               | `ForecastModel`, `CostModel`, `ImpactModel`, `Optimizer` (cvxpy QP), share sizing. |
| `daily_orchestrator.py`      | Daily entry point.                                                                 |
| `retrain.py`                 | Retrain entry point.                                                               |
| `tests/`                     | Unit + integration tests. Run with `.venv/bin/python -m pytest live/tests/`.       |

### Log locations

**See `live/LOGS.md` for a field-by-field reader's guide to every log
format** — what each key means, the milestone sequence of a healthy run,
and how to tell real records from test-suite output (the 07-15..07-17
files are heavily contaminated; see that doc before trusting them).

- `live/logs/live.daily_orchestrator.log` — human-readable stream +
  JSON-lines file, nightly rotation with 30-day retention.
- `live/logs/live.retrain.log` — same shape for retrain runs.
- `live/logs/targets/YYYY-MM-DD.json` — the intended book for a real
  (paper/live) day. Written before any broker contact, atomically
  staged.
- `live/logs/targets/replay/YYYY-MM-DD.json` — the intended book for a
  replayed historical date. Quarantined so the next-day
  reconciliation cannot pick it up.
- `live/logs/fills/YYYY-MM-DD.jsonl` — one line per order returned by
  the fill audit. Terminal state (`Filled`, `Cancelled`, timeout, etc.).

`--live-confirm` is redacted from any argv or vars(args) log entry.

## 9. Operational notes and troubleshooting

### "Coverage assertion failed" (rc=2)

The snapshot covered less than `--min-universe-coverage` (default 0.5)
of the active universe. Check, in this order:

1. Did the live feed connect? Grep `live feed started` in the log.
   If missing, `DATABENTO_API_KEY` is wrong or the connection was
   refused.
2. Grep `snapshotted live feed` — the `n_imbalance_records` and
   `n_tbbo_records` counts tell you whether the buffer was populated
   at all.
3. If the buffer was populated but coverage was still thin, the
   universe file may have drifted from the `SymbolMappingMsg` output
   for this session (a ticker rename, a mis-typed symbol). The
   `symbols_mapped` count in `feed_stats` shows how many bindings
   were resolved.
4. If you launched after 15:40, `recent_return` will have truncated —
   see section 5.

### "past MOC submit cutoff" (rc=4)

The wall clock passed `--submit-by` before the orchestrator was ready
to submit. Either the process started too late in the day, the MVO
solve took longer than expected, or `--print-wait-secs` was set high
enough to squeeze the submission window. The target book was persisted
to `live/logs/targets/YYYY-MM-DD.json` before the cutoff check so you
can see what would have been sent. Not applicable to replay.
**Distinct from rc=11**, which fires after entries already filled —
rc=4 is a safe pre-order skip, rc=11 is an active-exposure page.

### "OVERNIGHT EXPOSURE" (rc=11)

**Page immediately.** Entries filled but the MOC exit did not fully
cover the book. The ERROR log line names the concrete panic command
and lists the uncovered / MOC-rejected symbols. Standard remediation:

```bash
# During regular hours — MKT flatten fills immediately.
.venv/bin/python -m live.panic --paper --flatten-with MKT
# Or the live variant:
.venv/bin/python -m live.panic --live \
  --live-confirm YES_I_WANT_LIVE_TRADING --yes
```

Common causes and where to look:

- **MOC exchange rejection** — MOC batch fired past NASDAQ's 15:55:00
  ET entry cutoff (SEC 34-84454, 2018 rule). Grep `MOC rejected at
  exchange` in the log. Fix root cause by starting the run earlier
  (see section 5).
- **Uncovered symbols from a partial `get_positions()` snapshot** —
  TWS event stream had a brownout during the MOC snapshot. Grep
  `COVERAGE GAP` / `uncovered_symbols` — the log names each missing
  symbol.
- **Stale non-terminal `get_order_status`** — the cross-check saw a
  non-terminal read at cross-check time (should always be terminal
  after the MKT settles or the belt-and-braces cancel loop pulls it).
  Grep `non-terminal status at cross-check time`.
  Same TWS-brownout family as above.
- **MOC skipped past `--submit-by`** — the MOC batch fired but only
  some legs made the cutoff. Grep `legs skipped past submit cutoff`.
  Widen the window between `--rebalance-time` + `--print-wait-secs`
  and `--submit-by`, or start earlier.
- **`get_positions()` raised entirely** — TWS dropped between entry
  fills and the MOC snapshot. Grep `get_positions FAILED`. Panic
  then investigate the TWS logs.

After remediation, the next daily run's `_reconcile_from_previous`
will read the persisted target JSON at
`live/logs/targets/YYYY-MM-DD.json` and flag any drift against the
current positions.

### "broker connect failed" (rc=8)

The most common causes:

- Paper mode requested against a live TWS port (7496 / 4001).
  `Broker._LIVE_PORTS` refuses paper on those ports.
- TWS not running, or running on a different port than expected.
- Another orchestrator holding `--broker-client-id 1`; either wait or
  pass a distinct `--broker-client-id`.

Nothing has been placed at rc=8 — safe to fix and retry.

### After an incident

- `live/logs/targets/YYYY-MM-DD.json` — the intended book. Compare
  against `broker.get_positions()` to see what actually landed.
- `live/logs/fills/YYYY-MM-DD.jsonl` — every order the audit heard back
  from. A missing order id means the order was never placed (check the
  main log for a `leg submission failed` or `legs skipped` entry).
- On the next daily run, `_reconcile_from_previous` picks up
  yesterday's target JSON and calls `broker.reconcile_positions`; any
  drift is logged as a warning at startup.

### Do not

- Run two orchestrators against the same TWS with the same
  `--broker-client-id`. TWS accepts the second connection but silently
  loses one of the two order streams.
- **Run `panic` against the same TWS client-id as an in-flight
  orchestrator run.** Panic's `cancel_all()` would pull the
  orchestrator's own LMT entries and MOCs mid-flow, corrupting the
  rc gate. Panic defaults to client-id 99; orchestrator to 1. Keep
  them distinct. See the concurrency warning in section 10.
- Run the retrain and the daily simultaneously against the same
  artifact directory during the writes. Retrain's writes are atomic
  per file, but a run that spans a mid-write moment on a different
  file could read a mixed set (e.g. new loadings, old idio-diag).
  Schedule the retrain during off-hours.
- Edit the `.npy` or `.joblib` artifacts by hand. Retrain them.
- Follow a `--replay-submit` rehearsal with a real paper run without
  flattening the paper account. The replay fills sit there and will
  show up as drift in the next reconciliation.

### Nonsense-book diagnosis

If the daily submits a book that looks wrong (odd GMV, wrong sign,
concentrated in one name), check:

1. `coverage` in the `day_df built` log entry — a thin snapshot with
   only a few names produces a degenerate cross section.
2. `retrain complete` in the retrain log — confirm the last retrain
   actually finished and wrote atomically.
3. `live/logs/targets/YYYY-MM-DD.json` — the exact numbers the
   orchestrator produced. Sanity-check `gmv_dollar` and `vol_cap`
   against your intent.
4. The launch time — see section 5. A late launch silently truncates
   `recent_return`.

## 10. Panic — the kill switch

```bash
.venv/bin/python -m live.panic --help
```

`live/panic.py` is the account-wide kill switch: it connects to TWS,
cancels every open order, and (unless `--cancel-only`) flattens every
non-flat position via per-leg `broker.close_position(sym,
order_type=...)`. Deliberately small and blunt — this is what an
operator runs when something is on fire.

**Scope is ACCOUNT-WIDE, not universe-scoped.** Unlike the daily
orchestrator (which pins out-of-universe positions to their current
qty and never touches manual trades), panic closes every non-flat
position IB reports — including other-strategy books and manual
trades on the same account. Operators of a shared account must know
this before hitting the button.

**CONCURRENCY WARNING — do NOT run panic against the same TWS client-id
while an orchestrator run is in flight.** Panic uses client-id 99 by
default; the orchestrator uses 1. Distinct ids let them coexist against
the same TWS but they must NEVER target the same client-id
concurrently. Panic's `cancel_all()` is a global cancel over the
client's orders — running it against the orchestrator's client-id
mid-flow would pull the orchestrator's own LMT entries and MOC exits
between submission and the acceptance / coverage audits, corrupting
the rc gate and possibly leaving an open book unhedged. The two-leg
model's ordering guarantees only hold when it has exclusive control
of its client-id for the duration of a run.

If a mid-flow panic is unavoidable, panic with client-id 99 (the
default) — it operates on the account, not the orchestrator's client
— and let the orchestrator's rc gate observe the resulting state
naturally.

Two gotchas the script prints up-front:

1. NASDAQ has TWO distinct MOC deadlines the script encodes separately:
   the CANCEL/MODIFY cutoff at **15:50:00 ET** (`_past_moc_cancel_cutoff`
   — past this, `cancel_all()` CANNOT pull already-accepted MOC
   orders) and the ENTRY cutoff at **15:55:00 ET**
   (`_past_moc_entry_cutoff` — SEC release 34-84454, 2018 rule change
   that extended it from the old 15:50 — past this, new MOC
   submissions are typically rejected). `--flatten-with MOC` after
   15:55 may be a no-op; a BEFORE-snapshot MOC after 15:50 is
   unpullable. NYSE keeps a 15:50 entry cutoff (different venue,
   different rule); this kill switch routes to NASDAQ via
   `ACTIVE_ROUTING`.
2. `--flatten-with MKT` outside regular trading hours (weekday
   09:30–16:00 ET, holidays not modelled): MKT orders queue until the
   next open — they do NOT flatten immediately.

### CLI reference

Mode:

- `--paper` — paper mode. Default. Connects to TWS port 7497.
- `--live` — live mode. Requires `--live-confirm=YES_I_WANT_LIVE_TRADING`
  and connects to port 7496. Also triggers an interactive `PANIC`
  prompt (skippable with `--yes`).
- `--live-confirm LIVE_CONFIRM` — the broker confirmation string.
  Must be exactly `YES_I_WANT_LIVE_TRADING` — see [../CLAUDE.md](../CLAUDE.md).

Behaviour:

- `--flatten-with {MKT,MOC}` — order type for the flatten. Default
  `MKT`. MKT fills immediately during regular hours; MOC fills at the
  16:00 auction (clean end-of-day flatten) but is subject to NASDAQ's
  15:55 entry cutoff (and 15:50 cancel/modify cutoff — see gotcha #1).
- `--cancel-only` — cancel open orders but do NOT flatten positions.
- `--dry-run` — connect, print the BEFORE snapshot, place nothing.
- `--yes` — skip the interactive `PANIC` confirmation prompt
  (`--live` only prompts by default). Paper mode without `--yes`
  proceeds silently — this is the "quick oh-no" ergonomics.

Broker:

- `--broker-host BROKER_HOST` — TWS host. Default 127.0.0.1.
- `--broker-port BROKER_PORT` — TWS port. Default auto (7497 paper,
  7496 live).
- `--broker-client-id BROKER_CLIENT_ID` — TWS client id. Default
  **99** — distinct from the daily orchestrator's default (1) so
  panic can connect while the daily is holding its own client id.

### Golden commands

Inspect only — connect, print BEFORE snapshot, place nothing:

```bash
.venv/bin/python -m live.panic --paper --dry-run
```

The everyday panic — cancel every open order and MKT-flatten every
non-flat position during regular trading hours:

```bash
.venv/bin/python -m live.panic --paper
```

Cancel-only variants (useful before a MOC panic near the close, or
when you want to pull queued orders without flattening):

```bash
.venv/bin/python -m live.panic --paper --cancel-only
.venv/bin/python -m live.panic --paper --flatten-with MOC        # end-of-day clean flatten
```

Live counterparts (the confirmation string is required and the
prompt is skippable with `--yes` for scripted use):

```bash
.venv/bin/python -m live.panic --live --live-confirm YES_I_WANT_LIVE_TRADING
.venv/bin/python -m live.panic --live --live-confirm YES_I_WANT_LIVE_TRADING --yes
```

### Exit codes

- `0` — clean. Cancel + flatten completed and the AFTER snapshot shows
  no lingering non-panic orders and (for the flatten path) no
  non-flat positions remain.
- `1` — NOT FLAT / partial failure. Some combination of: lingering
  non-panic orders after `cancel_all`, per-leg flatten failures,
  positions still non-flat in the AFTER snapshot, MOC flatten
  submitted past NASDAQ's 15:55 entry cutoff (likely rejected), OR
  the AFTER snapshot itself couldn't be taken because TWS went
  unreachable ("COULD NOT CONFIRM FLAT"). Investigate the printed
  BEFORE / AFTER snapshots.
- `2` — operator declined the live `PANIC` prompt.
- `8` — broker connect failed. TWS not running, wrong port, or a
  `--broker-client-id` collision.
- `9` — `--live` was requested without the exact `--live-confirm`
  string.

### Post-panic verification

**The AFTER snapshot is the source of truth, not the exit code.** On
a slow TWS, `get_positions()` can under-report. Eyeball the printed
AFTER block: it lists every open order and every non-flat position.
`rc=0` combined with an AFTER snapshot showing 0 orders and 0
non-flat positions is the only genuine "flat" state.

Panic is intentionally a separate binary from the daily
orchestrator so exit-code numbers don't collide: `daily_orchestrator`
uses 0/2/3/4/5/6/7/8/9/11, panic uses 0/1/2/8/9. A wrapper keying on
the orchestrator's rc numbers must not also key on panic's — the same
integer means different things across the two binaries.

## 11. How to test

```bash
.venv/bin/python -m pytest data_collection/tests/ live/tests/ -v
```

All 189+ tests should be green. The suite is fast (about two seconds).
Safety-critical tests are grouped by module below — each one locks
in a specific correctness or safety claim; if one goes red the
corresponding guarantee is broken.

Test files that matter here:

- `live/tests/test_cache.py` — coverage manifest + resolver
  (schema/date/symbol coverage, sidecar invalidation, HIT vs MISS,
  fetch-guard behaviour with and without `--allow-fetch`).
- `live/tests/test_retrain.py` — atomic-write helpers + retrain's
  `--allow-fetch` / `--yes` gating (rc=9, rc=10) + the coverage-plan
  log format regression (parquet name + reason must appear on MISS).
- `live/tests/test_replay.py` — replay path end-to-end through
  `daily_orchestrator` (broker-free default, `--replay-submit`,
  guard-matrix rcs, per-leg cutoff neutralisation, tagged target
  book) + replay-side coverage-plan log format regression.
- `live/tests/test_end_to_end.py` — daily orchestrator end-to-end
  through a mocked broker (coverage assertion, pinning, unqualified
  drops, oversized-leg raise, empty-book refusal, broker-connect
  failure).
- `live/tests/test_panic.py` — panic script full behaviour matrix
  (cancel + flatten ordering, per-leg failures, MOC-cutoff warnings,
  TWS-dropout resilience, live confirm-string / prompt gates).

Feature engineering:

- `test_live_feed.py::test_symbol_mapping_prefers_stype_in_when_out_is_digits`
  — Databento's live gateway always emits `stype_out=instrument_id`; we
  buffer the ticker from `stype_in_symbol` in that case.
- `test_live_feed.py::test_symbol_mapping_uses_stype_out_when_it_is_a_real_ticker`
  — non-gateway replay case still uses the OUT symbol.
- `test_features.py::test_build_feature_frame_single_day_with_reference_stats`
  — a single-day live buffer plus reference stats produces a non-empty
  feature frame with finite ratios (the C2 regression).
- `test_features.py::test_build_feature_frame_single_day_without_reference_stats_is_empty`
  — the same call without reference stats deliberately drops every row,
  documenting the failure mode the daily must not hit.

Daily orchestrator end-to-end (`test_end_to_end.py`):

*Coverage / pinning / prequalify / broker plumbing:*

- `test_daily_orchestrator_refuses_thin_coverage`
  — coverage below `--min-universe-coverage` aborts with rc=2 before
  the broker is contacted.
- `test_daily_orchestrator_pins_out_of_universe_positions`
  — a manual position in a symbol not in the universe is never touched.
- `test_daily_orchestrator_drops_unqualified_symbols`
  — a symbol IB refuses to qualify is dropped from the target, the
  batch continues.
- `test_daily_orchestrator_raises_max_order_qty_for_large_leg`
  — an oversized leg raises `Broker.MAX_ORDER_QTY` for the session
  instead of crashing.
- `test_daily_orchestrator_refuses_empty_share_targets`
  — an empty target book returns rc=5 without connecting to the broker.
- `test_broker_connect_failure_returns_rc_8`
  — an exception out of `_connect_broker` returns rc=8 with no
  traceback and no orders placed.

*Two-leg execution safety:*

- `test_daily_orchestrator_places_two_leg_orders`
  — happy path: LMT entries via `broker.buy`/`broker.sell` with
  `order_type="LMT"`, a numeric `limit_price`, and the default
  tif=DAY, then opposite-direction MOC counter-flatten per name
  sized to the fill; `broker.rebalance()` is never called.
- `test_entries_reach_broker_with_production_defaults`
  — the ONE end-to-end test that runs with the REAL production
  `DEFAULT_MAX_QUOTE_STALENESS_SECS` (30s) and
  `DEFAULT_MAX_ENTRY_REL_SPREAD` (50 bps). Uses a fresh-vs-now
  fixture so a regression in either default (e.g. dropping
  staleness to 1s) fails LOUDLY here.
- `test_two_leg_partial_fill_moc_sized_to_actual_position`
  — **the core safety invariant.** A 50% LMT fill produces a 50% MOC
  counter (sized to actual position), never a full target-diff MOC.
- `test_two_leg_zero_fill_no_moc_for_that_name`
  — an entry that raises during placement does NOT receive a naked
  MOC (position stays flat, MOC batch skips the symbol).
- `test_two_leg_submit_by_hit_between_entry_and_exit_returns_rc_11`
  — cutoff trips AFTER entries filled, MOC batch skipped → rc=11
  (not rc=4 — that's a pre-order safe skip).
- `test_rc_4_and_rc_11_are_distinct`
  — rc=4 (pre-order safe skip) and rc=11 (mid-flow exposure) must
  never collide onto the same code.
- `test_r2_stuck_entries_cancelled_before_moc_snapshot`
  — non-terminal entries get `broker.cancel(oid)` BEFORE the MOC
  snapshot runs; temporal-ordering assertion via the fake broker's
  event log.
- `test_r4_duplicate_positions_dedup_by_symbol`
  — IB returning the same symbol twice at `get_positions()` aggregates
  into ONE MOC order sized to the sum, never two.
- `test_scale_auto_computed_from_lambda`
  — `Optimizer.set_problem` is called with `scale = 1 / args.lam`.
- `test_target_gmv_argparse_error`
  — the removed `--target-gmv` flag is rejected by argparse.
- `test_r7_lambda_zero_returns_rc_9`,
  `test_r7_lambda_negative_returns_rc_9`
  — invalid `--lambda-` returns rc=9 (safe skip band, below the
  `rc >= 11` pager threshold).

*RR1 — coverage cross-check:*

- `test_rr1_partial_snapshot_returns_rc_11`
  — partial `get_positions()` (3 of 10 filled names) → cross-check
  flags 7 uncovered → rc=11 with loud alert.
- `test_rr1_wait_budget_exhausted_with_real_fills_returns_rc_11`
  — `wait_for_fill` timing out while LMTs actually filled (correlated
  slow-TWS brownout) → cross-check re-polls `get_order_status` and
  catches the real fills → rc=11.
- `test_r1_get_positions_raises_returns_rc_11`
  — `get_positions()` raising post-entry → `positions_unknown=True`
  → rc=11.
- `test_r1_empty_snapshot_post_fill_returns_rc_11`
  — stale-empty `get_positions()` after real fills → rc=11.

*RRR1 — MOC exchange-acceptance audit:*

- `test_rrr1_moc_rejected_at_exchange_returns_rc_11`
  — every MOC API-accepted (oid returned) but Cancelled at exchange
  (`filled=0`) → acceptance audit catches → rc=11. Structurally
  pins the audit as load-bearing: first runs with
  `_audit_moc_acceptance` monkeypatched to return `set()` (proves
  pre-fix rc=0), then restores the audit (rc=11).

*RRR2 — I/O failures cannot downgrade rc=11:*

- `test_rrr2_write_target_record_failure_still_returns_rc_11_on_exposure`
  — `_write_target_record` raising with a real exposure signal in
  play → rc still 11 (best-effort wrappers do not propagate).
- `test_rrr2_write_target_record_failure_returns_rc_0_on_clean_run`
  — happy-path run with `_write_target_record` raising → rc still 0
  (persist failure alone is not synthesized into an exposure).

*RRR3 — stale non-terminal `get_order_status`:*

- `test_rrr3_stale_non_terminal_get_order_status_flags_uncovered`
  — TWS event-stream brownout returns non-terminal status without
  raising → cross-check treats as pessimistic full fill → rc=11.

*N1 — CLI plumbing:*

- `test_n1_moc_ack_timeout_flag_threaded_to_audit`
  — `--moc-ack-timeout 90` reaches `_audit_moc_acceptance`'s
  `wall_clock_wait` parameter.

Replay (`test_replay.py`):

- `test_replay_default_is_broker_free`
  — replay without `--replay-submit` never contacts the broker; the
  target record is still persisted under `targets/replay/`.
- `test_replay_submit_places_moc_orders`
  — `--replay-submit` reaches the fake broker and places MOC orders
  even with `--submit-by 00:01` (the per-leg cutoff is neutralised in
  replay).
- `test_replay_submit_without_replay_date_refused`
  — `--replay-submit` outside replay mode returns rc=7, no broker
  contact.
- `test_replay_refuses_live`
  — `--replay-date` + `--live` returns rc=6, no broker contact.
- `test_replay_skips_wall_clock_waits`
  — `_sleep_until` is never called during replay AND orders still
  land with a past `--submit-by`.
- `test_replay_persists_tagged_target_book`
  — the persisted record has `mode="replay"` and lives in
  `targets/replay/`; the top-level directory is untouched so
  reconciliation cannot pick up a rehearsal.
- `test_replay_feed_snapshot_matches_live_feed_columns`
  — column-set parity between `ReplayFeed.snapshot()` and the
  live-side `imbalance_records_to_frame` / `tbbo_records_to_frame`
  constructors.
- `test_replay_feed_start_failure_exits_cleanly`
  — an exception out of `ReplayFeed.start()` returns rc=2 rather than
  propagating.
- `test_replay_feed_default_allow_fetch_is_false`,
  `test_replay_feed_miss_without_allow_fetch_refuses`
  — replay's fetch guard: default is safe; a cold-cache replay
  without `--allow-fetch` raises `CoverageError` and never calls
  `adapter.get_data`.
- `test_replay_orchestrator_allow_fetch_wired_to_replayfeed`,
  `test_replay_orchestrator_default_allow_fetch_false`
  — the orchestrator's `--allow-fetch` flag plumbs through to
  `ReplayFeed.allow_fetch` and defaults to False.
- `test_replay_feed_coverage_plan_names_nearest_parquet_and_reason`
  — regression: the replay call site's coverage-plan INFO log AND
  the `CoverageError` message both name the nearest parquet, the
  reason, and the shortfall on MISS.

Retrain atomic writes + fetch guard (`test_retrain.py`):

- `test_atomic_np_save_writes_real_data`
  — `_atomic_np_save` writes the actual array bytes, not a 0-byte file
  (the R1 regression).
- `test_atomic_np_save_failure_cleans_up_tmp`
  — a failed save leaves no orphaned `.tmp` file behind.
- `test_atomic_joblib_dump_round_trip`,
  `test_atomic_write_parquet_round_trip`
  — sklearn and polars artifacts round-trip through the atomic-write
  helpers.
- `test_retrain_dry_run_never_calls_adapter`
  — `--dry-run` never hits the adapter; guardrail against a dry-run
  that silently pays for a batch job.
- `test_retrain_miss_without_allow_fetch_exits_nonzero`
  — cold cache without `--allow-fetch` returns rc=9 and never fetches.
- `test_retrain_miss_with_allow_fetch_but_no_yes_refuses`
  — `--allow-fetch` alone (no `--yes`, non-TTY) returns rc=10 and
  never fetches.
- `test_retrain_miss_with_allow_fetch_and_yes_fetches`
  — `--allow-fetch --yes` authorises the resolver to fetch.
- `test_retrain_coverage_plan_names_nearest_parquet_and_reason`
  — regression on the operator-facing log: the "coverage plan" INFO
  line must name the closest cached parquet AND the shortfall reason
  so a MISS isn't opaque.

Cache manifest / resolver (`test_cache.py`):

- `test_coverage_hit_full_range`, `test_coverage_miss_partial_date_range_head`,
  `test_coverage_miss_partial_date_range_tail`,
  `test_coverage_miss_symbol_subset`
  — HIT/MISS decisions for the common failure shapes.
- `test_r1_minute_bar_parquet_never_hits_ohlcv_1d_request`
  — the `ohlcv-1d` / `ohlcv-1m` disambiguation. A minute-bar parquet
  cannot silently HIT a daily-bar request (would corrupt the
  covariance fit).
- `test_r2_weekend_tail_hits_from_friday_parquet`,
  `test_r2_weekend_head_hits_from_monday_parquet`
  — weekend-floor semantics: a Friday-ending parquet HITs a request
  whose exclusive end is Saturday/Sunday.
- `test_r3_symbol_only_covers_half_window_miss`
  — per-symbol range check: an IPO / delisting mid-window MISSes even
  though the symbol appears in the parquet.
- `test_r4_nearest_reports_smallest_shortfall`
  — the near-miss log names the parquet with the SMALLEST shortfall,
  not the glob-order first failure.
- `test_r6_sidecar_invalidated_when_size_changes_but_mtime_preserved`
  — sidecar keys on `(mtime, size)` so an in-place rewrite that
  preserves mtime still invalidates.
- `test_resolver_miss_without_allow_fetch_raises`,
  `test_resolver_miss_with_allow_fetch_calls_adapter`
  — the resolver's `allow_fetch` gate.

Panic script (`test_panic.py`):

- `test_default_run_cancels_and_flattens`,
  `test_cancel_precedes_flatten_ordering`
  — the standard flow: cancel then per-leg flatten.
- `test_per_leg_flatten_survives_partial_failure`
  — one leg's exception doesn't skip the rest of the batch or the
  AFTER snapshot.
- `test_lingering_non_panic_order_marks_not_flat`,
  `test_lingering_only_panic_own_orders_still_flat`
  — the AFTER snapshot excludes panic's own newly-placed flatten
  legs when counting "lingering" orders (H2).
- `test_moc_flatten_past_cutoff_returns_nonzero`
  — an MOC panic past NASDAQ's 15:55 entry cutoff returns rc=1, not
  a false clean rc=0.
- `test_after_snapshot_tws_dropout_marks_could_not_confirm`
  — a TWS dropout during the AFTER snapshot exits rc=1 with an
  explicit "COULD NOT CONFIRM FLAT" line; never falsely reports flat.
- `test_live_without_confirm_string_refused`,
  `test_live_prompt_abort_places_nothing`
  — the two live-mode gates: `--live-confirm` string and the
  interactive `PANIC` prompt.
- `test_confirm_msg_stays_in_sync_with_broker`
  — `panic._LIVE_CONFIRM_MSG` cannot drift from
  `Broker.LIVE_CONFIRM_MSG`.

### First-time smoke check

Before the first paper session, run the dry-run smoke to prove the
whole pipeline is wired without touching the network or the broker:

```bash
.venv/bin/python -m live.daily_orchestrator \
  --dry-run --no-live-feed \
  --rebalance-time 00:00 --stream-start-time 00:00 \
  --submit-by 23:59 --on-feed-failure proceed
```

Expected in the log tail (stream handler goes to stderr): `orchestrator
starting`, `loaded active universe`, `loaded forecast pipeline`,
`loaded covariance artifacts`, `MVO solved`, `share targets computed`,
`empty share_targets — refusing to touch the broker` (with rc=5), or in
the `--dry-run` variant `DRY RUN — printing target book`. Any earlier
exit means one of the prerequisites in section 2 is missing.

Second smoke: a broker-free replay of a date already in your historical
cache — this proves the replay path end to end without live-imbalance
airtime:

```bash
.venv/bin/python -m live.daily_orchestrator \
  --replay-date 2026-05-15 --paper
```

Expected: `replay mode: starting historical feed`, then the same
`MVO solved`, `share targets computed`, `target book persisted` chain,
then `REPLAY (no-submit) — book computed and persisted...` and rc=0.
The persisted record lands at
`live/logs/targets/replay/2026-05-15.json`.

## 12. Exit-code reference

Consolidated from sections 4 (daily), 6 (retrain), and 10 (panic).
Panic is a separate binary from the daily orchestrator so rc numbers
don't collide with each other.

### `live.daily_orchestrator`

**Pager predicate: `rc >= 11` is the ONLY code that implies a live
book at close.** Everything in [2, 10] is a safe pre-order refusal
or clean shutdown.

| rc   | Meaning                                                                            |
| ---- | ---------------------------------------------------------------------------------- |
| `0`  | MVO solved; two-leg batch completed AND post-close position poll confirmed managed universe FLAT (or dry-run / broker-free replay summary). Applies to `--replay-submit` too — any run that reached the broker also runs the authoritative post-close verification. |
| `2`  | Live feed / feature build failed (coverage assertion, `LiveFeed.start()` / `ReplayFeed.start()` exception, cold-cache replay without `--allow-fetch`). |
| `3`  | MVO solver returned a non-optimal status.                                          |
| `4`  | Wall clock past `--submit-by` at submit time; NOTHING placed. Replay honours this by default (round-2 change) — pass `--replay-ignore-cutoff` to exercise the order path on a late replay. Safe skip. |
| `5`  | Target book was empty and `--dry-run` was not set. Broker not contacted.           |
| `6`  | `--replay-date` + `--live` was requested. Refused. No broker contact.              |
| `7`  | `--replay-submit` was requested without `--replay-date`. No broker contact.        |
| `8`  | Broker refused the connection (live-port guard, TWS down, `--broker-client-id` collision). |
| `9`  | `--lambda- <= 0` (invalid argument). No broker contact.                             |
| `11` | **OVERNIGHT EXPOSURE — page.** Either (a) a pre-close signal fired an EARLY warning so the operator can run panic before 16:00 (MOC skipped, `get_positions` failed, stale-empty snapshot, cross-check found uncovered symbols, or acceptance audit found exchange rejections; tag: `OVERNIGHT EXPOSURE (pre-close)`), OR (b) the AUTHORITATIVE post-close position poll (16:00-16:05) found a nonzero managed-universe symbol still on the book, or every poll failed pessimistically (tag: `OVERNIGHT EXPOSURE (post-close)`). ERROR log names the concrete panic command and the offending symbols. |

### `live.retrain`

| rc   | Meaning                                                                             |
| ---- | ----------------------------------------------------------------------------------- |
| `0`  | Refit completed and artifacts atomically written (or `--dry-run` plan printed).     |
| `9`  | Coverage MISS on one or more schemas and `--allow-fetch` was not set. Nothing fetched. |
| `10` | Coverage MISS with `--allow-fetch` set but confirmation refused (no `--yes` / no TTY / `no` at prompt / Ctrl-D). Nothing fetched. |

### `live.panic`

| rc  | Meaning                                                                             |
| --- | ----------------------------------------------------------------------------------- |
| `0` | Clean. Cancel + flatten completed; AFTER snapshot shows no lingering non-panic orders and no non-flat positions. |
| `1` | NOT FLAT / partial failure: lingering orders, per-leg failures, positions remain, MOC past cutoff, or "COULD NOT CONFIRM FLAT" (TWS dropout during AFTER snapshot). |
| `2` | Operator declined the `PANIC` prompt (live mode).                                  |
| `8` | Broker connect failed.                                                              |
| `9` | `--live` without the exact `--live-confirm` string.                                 |
