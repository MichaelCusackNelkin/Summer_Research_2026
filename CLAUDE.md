# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A Columbia MAFN summer research project (per `README.md`) building an optimal-portfolio
backtest from three estimated pieces — a return **forecast**, a **covariance** matrix, and
**spread costs** — solved as a portfolio optimization and backtested on a **point-in-time
(PIT) universe**. The original close-auction imbalance homework is the alpha seed; the
current active build-out is the PIT data plumbing that makes a survivorship-aware backtest
possible (universe, covariance, and forecast all refit per rebalance).

> History: the repo began as two parallel projects (an execution project under
> `research-alpha/`/`sim_lib/`, plus this alpha direction). The execution code and the
> `research-alpha/` tree have since been **removed** — ignore any stale references to
> `research-alpha`, `sim_lib`, or `strategy_close_imbalance.py`. `pip` may still list
> `sim_lib` as an editable install pointing at a missing directory; do not import or
> reinstall it.

## Repository layout

- `data_collection/` — the only installable package (`data-collection`, installed editable
  into `.venv`). Two source modules, both with tests:
  - `src/data_collection/data_adapters.py` — the Databento-backed three-layer cached
    adapter (`DatabentoGenericAdapter`). Core infrastructure everything else builds on.
  - `src/data_collection/universe_pipeline.py` — pure-logic PIT universe selection +
    deduped download planning (no I/O, no SDK calls; unit-tested against synthetic frames).
- `build_universes.py` (repo root) — the CLI orchestrator that drives `universe_pipeline`:
  builds PIT universes, estimates Databento cost, and (with `--download`) executes the
  deduped batch plan via the adapter. This is the substantial application code.
- `data.py` (repo root) — small scratch script: materialize-only fetch of
  `imbalance`/`tbbo`/`statistics` for the flat `universe.txt` ticker list over a fixed window.
- `universe.txt` — flat newline-delimited ticker list (~2000 names) used by `data.py` and
  the notebooks. (`universe_russell.txt` was removed.)
- `notebook_studies/` — the research surface (all notebooks, no `.py`): `universe_selection`
  (the prototype `universe_pipeline` mirrors), `alpha_forecast_study`, `optimal_portfolio`,
  `rules_based_backtest`, `closing_imbalance_homework` (the original NYSE-IBF homework), and
  `test` scratch.
- `covariance_research/` — `covariance.ipynb` + `covariance.py` (scratch covariance studies;
  the `.py` is exploratory and not wired into the pipeline).
- `alpha_research/`, `optimization_research/` — currently empty placeholders.
- `universes/` — outputs of `build_universes.py` (see below). Tracked except the adapter cache.
- `docs/security_master_identity_plan.md` — a deferred, ready-to-execute plan (no code yet).
- `data/dbn_cache/generic/` — parquet + raw `.dbn.zst` adapter cache (gitignored: `**/data`).

Notebooks and scripts import the package as a top-level name
(`from data_collection.data_adapters import ...`) — there is no `PYTHONPATH` shim, it relies
on the editable install. `build_universes.py` and `data.py` run from the **repo root**;
`covariance_research/covariance.py` uses `parents[2]` paths and `../../../universe.txt`, so
it expects to run from its own directory.

## Common commands

Use the project venv explicitly — there is no `activate` step:

- Build PIT universes + cost estimate (NO downloads, the safe default):
  `.venv/bin/python build_universes.py --strategy-start 2020-01-01`
- Skip the live cost API call too: add `--skip-cost`.
- Execute the deduped download plan (live Databento batch jobs — **costs money**):
  `.venv/bin/python build_universes.py --strategy-start 2020-01-01 --download`
- Full flag list: `.venv/bin/python build_universes.py --help`. Key knobs: `--top-n`,
  `--lookback-months`, `--rebalance-months`, `--min-price`, `--min-trading-frac`,
  `--adv-agg {median,mean}`, `--schemas`, `--max-concurrency`, `--today`, `--out-dir`.
- Materialize the raw schema parquets for `universe.txt`:
  `.venv/bin/python data.py` (note: `data.py` calls `adapter.clear_cache()` on every run).
- Tests (both modules in `data_collection`): `.venv/bin/python -m pytest data_collection/tests/`
  — `pyproject.toml` sets `asyncio_mode = "auto"`, so `async def test_*` run without a marker.
- Single test:
  `.venv/bin/python -m pytest data_collection/tests/test_universe_pipeline.py::test_select_universe_top_n`
- Re-install after editing `data_collection/pyproject.toml`:
  `.venv/bin/pip install -e ./data_collection`.
- Jupyter: `.venv/bin/jupyter lab` (or run notebooks through the venv `ipykernel`).

`requirements.txt` lists the broad runtime deps; `data_collection/pyproject.toml` is the
authoritative dependency set for the installed package. Python is 3.14.

## Required environment

The adapter, `build_universes.py`, and `data.py` read `DATABENTO_API_KEY` from the
environment via `python-dotenv` loading `./.env`. Note the **inconsistency**: notebooks and
`covariance_research/covariance.py` reference the hyphenated `DATABENTO-API-KEY`, while the
`.env` standard and `build_universes.py` use the underscore `DATABENTO_API_KEY`
(`_resolve_api_key` accepts either). Prefer the underscore form. The cache root the tools
expect is `<repo_root>/data/dbn_cache/generic/`.

## Architecture notes that span multiple files

### Three-layer Databento cache (`data_collection.data_adapters.DatabentoGenericAdapter`)

`get_data(...)` resolves a request through three layers, falling through only on miss:

1. Parquet cache (`<cache_dir>/<key>.parquet`) — what callers normally hit.
2. Raw `.dbn.zst` cache (`<cache_dir>/raw_dbn/*_<key>_*.dbn.zst`) — decoded on demand (per
   file via temp parquets so peak memory is one decoded file), then materialized into layer 1.
3. Databento Batch API — submits a job, persists `<cache_dir>/raw_dbn/<key>.job` as a resume
   marker (a re-run rejoins an in-flight job instead of submitting a duplicate), polls,
   downloads, renames each file to `<job_id>_<key>_<i>.dbn.zst`, then falls through to layer 2.

The key (`_generate_cache_key`) is an MD5 over a JSON dict of normalized tickers, `start`,
`end`, `dataset`, `schema`. Any change to those inputs produces a brand-new cache slot — the
layers are **not** aware of overlapping date ranges or supersets (this is exactly why
`universe_pipeline` does its own interval-level dedup — see below). Defaults:
`dataset="XNAS.ITCH"`, `schema="ohlcv-1m"`; `end_date` is **exclusive** and defaults to today.

- `cache_only=True` refuses the network (raises `FileNotFoundError` on miss) — for batch
  scripts that must never spawn a job.
- `materialize_only=True` ensures the parquet exists and returns `None` instead of eagerly
  reading it — for callers that read back via `scan_data`.
- `clear_cache()` only removes the parquet layer; raw `.dbn.zst` files are preserved.

`scan_data(...)` is the lazy synchronous sibling: returns a `pl.LazyFrame` from
`pl.scan_parquet` and **raises `FileNotFoundError` if the parquet layer is missing** — it
never decodes raw DBN and never hits the API. Materialize once with `get_data(...)` first.
The canonical pattern (used by `_ensure_parquet_async` in `build_universes.py`): try
`scan_data`, and on `FileNotFoundError` call `get_data(..., materialize_only=True)`.

Blocking SDK calls are wrapped in `asyncio.to_thread`, so `get_data` is safe to `await` from
a Jupyter event loop.

### PIT universe pipeline (`universe_pipeline.py` + `build_universes.py`)

The logic/orchestration split is deliberate: **`universe_pipeline.py` is pure** (functions of
in-memory frames + dataclasses, fully unit-testable offline), and **`build_universes.py`
holds all the I/O** (adapter calls, the live cost API, file writes). When changing behavior,
prefer adding to the pure module and keep the CLI thin.

The CLI runs three steps (`amain`):

1. **Universes.** One `ALL_SYMBOLS` OHLCV-1d fetch over `[max(strategy_start, DATA_FLOOR),
   today)`, then `build_pit_universes` walks rebalance dates spaced `REBALANCE` (default 6mo)
   apart. The first rebalance is `floor + LOOKBACK` (default 18mo) so every training window
   sits wholly inside the fetched range — no lookahead. Per rebalance, `select_universe`
   takes the **top-N by trailing dollar-volume ADV** (default `TOP_N=3000`, `adv_agg="median"`)
   after dropping null/non-positive close|volume rows ("edge case K"), names trading <90% of
   sessions (`MIN_TRADING_FRAC`), and last-close < `MIN_PRICE`. Writes `universes.parquet`
   (long format, one row per `(rebalance_date, symbol)`) and `universes.json`.
2. **Cost estimate.** `plan_all_downloads` produces the deduped request plan;
   `estimate_cost` sums Databento `metadata.get_cost` over it (concurrent, `Semaphore`-bounded,
   `get_cost` called **without** the deprecated `mode` arg). Writes `cost_estimate.json`.
3. **Downloads** (only with `--download`). `execute_downloads` runs the plan via the adapter
   (`Semaphore`-bounded), then `consolidate_schema` concats per-schema parquets and dedups on
   each schema's `NATURAL_KEYS`. Without `--download`, a dry-run `download_ledger.planned.json`
   is written so you can inspect what *would* be fetched.

**The dedup/ledger is the core idea.** Because the adapter's cache key is opaque to date
overlap, the pipeline tracks coverage itself. Each `(schema, ticker)` has required coverage
`[max(DATA_FLOOR, rebalance-18mo), rebalance+6mo)`. The `CoverageLedger` records intervals
already requested; `subtract_intervals` (half-open `[start, end)` interval algebra) computes
the missing gaps; tickers sharing an identical gap are grouped into **one** `DownloadRequest`.
With the default layout this collapses to ~2 requests per schema per rebalance (brand-new
names need the full ~24mo span; persisting names need only the new 6mo forward slice).
Properties that matter when editing this:

- The ledger persists to JSON and is **resumable**: `execute_downloads` marks + saves it only
  after each *successful* materialization (under a lock), so a crash mid-batch leaves it
  consistent and the next run refetches only the un-marked gaps. `available_end` clamps every
  window to `today` so no request asks for future data (keeps the ledger honest near the
  present edge).
- `MAX_SYMBOLS_PER_REQUEST = 2000` is Databento's per-call symbol cap (applies to both
  `get_cost` and `batch.submit_job`); `split_oversized_requests` chunks any larger group.
  Coverage is tracked per ticker, not per request, so chunking changes nothing downstream.

### Security identity (deferred)

The pipeline keys securities on **`raw_symbol`** (PIT-correct per day via Databento's
`map_symbols`). It does **not** use `instrument_id` as a cross-time key — on XNAS.ITCH that
integer is remapped daily. The rigorous fix (a Databento **Security Master** identity map
keyed on `bbg_comp_id`/`figi`, handling renames like FB→META and recycled tickers) is fully
designed in `docs/security_master_identity_plan.md` but **not implemented** — it needs a
Reference Data subscription we don't have. Until then, window-scoping neutralizes recycling
and rare mid-window renames conservatively drop a name (no lookahead/survivorship bias).
