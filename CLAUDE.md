# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

The repo started as two parallel research projects (per `README.md`: extending the
Spring 2026 "Statistical Aspects of Finance" execution work, plus a new
cross-sectional alpha direction). The execution project's code (`sim_lib/`,
`research-execution/`) has since been removed from the working tree — the live
work is the alpha project under `research-alpha/`.

- `data_collection/` — installable package `data_collection`. Holds the single
  Databento-backed cached adapter in `src/data_collection/data_adapters.py`; this
  is the only source module and the only library with tests.
- `research-alpha/code/strategy_close_imbalance.py` — the close-auction imbalance
  backtest. This is the substantial code in the repo (~40 KB, all logic in one file).
- `research-alpha/code/main.py` — thin `argparse`/`asyncio.run` CLI wrapper around
  `strategy_close_imbalance.main`.
- `research-alpha/code/notebooks/` — `cross_sectional_alpha.ipynb` (IC / alpha
  studies) and `test.ipynb` scratch.
- `closing_imbalance_homework.ipynb` (repo root) — the original pandas/CSV homework
  (NYSE IBF gz files) that `strategy_close_imbalance.py` mirrors as Q5.
- `research-alpha/report/` — empty placeholder for a Quarto whitepaper.
- `data/dbn_cache/generic/` — parquet + raw `.dbn.zst` cache for the adapter
  (gitignored: `**/data`).

`data_collection` is installed editable into `.venv` (Python 3.14). Notebooks and
scripts import it as a top-level package (`from data_collection.data_adapters import ...`);
there is no `PYTHONPATH` shim. `strategy_close_imbalance.py` is imported by its
sibling `main.py` by bare module name, so run it from `research-alpha/code/`.

> Stale install: `pip` still lists `sim_lib` as an editable install pointing at a
> directory that no longer exists. Do not `import sim_lib` or try to reinstall it —
> the source is gone. Ignore it unless you are deliberately resurrecting that project.

## Common commands

Use the project venv explicitly — there is no `activate` step in instructions:

- Run the backtest CLI: `.venv/bin/python research-alpha/code/main.py --help`
  - Smoke path (synthesize bid/ask from OHLCV-1s, skip the bbo batch fetch):
    `.venv/bin/python research-alpha/code/main.py --smoke` (equivalent to `SMOKE_TEST=1`).
  - With params: `.venv/bin/python research-alpha/code/main.py --commission 0.0035 --capital 1000000`
- Open Jupyter: `.venv/bin/jupyter lab` (or run notebooks through the venv `ipykernel`).
- Tests (only `data_collection` has them): `.venv/bin/python -m pytest data_collection/tests/`
  — `pyproject.toml` sets `asyncio_mode = "auto"`, so `async def test_*` functions run
  without an explicit marker.
- Single test: `.venv/bin/python -m pytest data_collection/tests/test_data_adapters.py::test_cache_dir_is_created`
- Re-install the library after editing its `pyproject.toml`: `.venv/bin/pip install -e ./data_collection`.

`requirements.txt` at the repo root lists the broad runtime deps (polars, databento,
statsmodels, scikit-learn, etc.); `data_collection/pyproject.toml` is the authoritative
dependency set for the installed package.

## Required environment

The adapter and the backtest read `DATABENTO_API_KEY` from the environment (via
`python-dotenv` loading `./.env`). The cache path the backtest expects is
`repo_root/data/dbn_cache/generic/` (`CACHE_DIR` in `strategy_close_imbalance.py`).

## Architecture notes that span multiple files

### Three-layer Databento cache (`data_collection.data_adapters.DatabentoGenericAdapter`)

`get_data(...)` resolves a request through three layers in order, falling through
only on miss:

1. Parquet cache (`<cache_dir>/<key>.parquet`) — what callers normally hit.
2. Raw `.dbn.zst` cache (`<cache_dir>/raw_dbn/*_<key>_*.dbn.zst`) — decoded on demand
   (per-file streaming via temp parquets so peak memory is one decoded file), then
   materialized into layer 1.
3. Databento Batch API — submits a job, persists `<cache_dir>/raw_dbn/<key>.job` as a
   resume marker (so a re-run rejoins an in-flight job instead of submitting a
   duplicate), polls, downloads, renames each file to `<job_id>_<key>_<i>.dbn.zst`,
   then falls through to layer 2.

The key (`_generate_cache_key`) is an MD5 over a JSON-serialized dict of normalized
tickers, `start`, `end`, `dataset`, `schema`. Any change to those inputs produces a
brand-new cache slot — the layers are not aware of overlapping date ranges or supersets.
Defaults are `dataset="XNAS.ITCH"`, `schema="ohlcv-1m"`; `end_date` is exclusive and
defaults to today.

- `cache_only=True` makes the adapter refuse to hit the network (raises
  `FileNotFoundError` on miss) — use in batch scripts that should never spawn a job.
- `materialize_only=True` ensures the parquet exists and returns `None` instead of
  eagerly `read_parquet`-ing it — use when the caller will read it back via `scan_data`.
- `clear_cache()` only removes the parquet layer; raw `.dbn.zst` files are preserved.

`scan_data(...)` is the lazy, synchronous sibling: it returns a `pl.LazyFrame` from
`pl.scan_parquet` and **raises `FileNotFoundError` if the parquet layer is missing** —
it never decodes raw DBN and never hits the API. Materialize the parquet once with
`get_data(...)` before relying on `scan_data(...)`. The strategy's `_ensure_parquet`
helper encodes exactly this pattern: try `scan_data`, and on `FileNotFoundError` call
`get_data` to populate the cache.

The blocking Databento SDK calls are wrapped in `asyncio.to_thread`, so `get_data` is
safe to `await` from a Jupyter event loop.

### Close-auction imbalance backtest (`research-alpha/code/strategy_close_imbalance.py`)

The module's top docstring is the spec — read it first. It is a Databento-driven
re-implementation of the homework's "ride with the published imbalance" strategy.
Shape of the pipeline (`main` orchestrates):

- Pulls four `ALL_SYMBOLS` schemas for the `START_DATE..END_DATE` window via the
  adapter: `Imbalance`, `OHLCV-1s`, `statistics` (and bbo, unless `--smoke`). Each
  loader (`load_imbalance`/`load_ohlcv_1s`/`load_statistics`) converts `ts_event` to
  `America/New_York` and filters to the session.
- Per trading day, `replay_day_snapshot` drives `DBNStore.replay` through a `Strategy`
  callback. The strategy keeps live per-symbol bbo state and **latches** each symbol's
  closing-imbalance snapshot on its *first* imbalance message at-or-after 15:50:00 NY
  (`SNAP_TIME_NY`); later messages for a latched symbol are ignored — this mirrors the
  homework's `groupby('symbol').first()`. Replay stops at `REPLAY_STOP_TIME_NY` (15:58).
- Two filters select the day's tradeable names: `|imb_dollar_qty| > $500k`
  (`MIN_ABS_DOLLAR_IMBALANCE`) and `|imb_dollar / session_$vol| > 1%` (`MIN_ABS_QTY_SIGNAL`).
  There is no top-N ADV universe pre-filter and no rolling-regression alpha overlay.
- Sizing is **equal weight** across the N passing names (`capital / N`); dollar
  neutrality is intentionally dropped. Direction trades *with* the imbalance side
  (long on `'B'`, short on `'A'`). Entries pay one half-spread against the live bbo;
  exits fill at the official close print (statistics `stat_type == 11`).
- `book_trades` turns snapshots into `Trade`/`PanelRow` PnL (commission is
  per-share, capped at `COMMISSION_CAP_FRAC` = 1% of leg notional). `summarize_and_plot`
  and `save_results_parquet` write PNGs and `trades.parquet` / `daily_pnl.parquet` /
  `backtest_panel.parquet` into `research-alpha/code/strategy_close_imbalance_plots/`.

Prices arrive as Databento fixed-point int64 and are scaled by `PX_SCALE = 1e-9`
(correct for XNAS.ITCH). The per-day execution is deliberately replay-driven so the
same `Strategy.on_event` callback could consume live records with minimal change;
only the session-$-volume denominator and the close print are treated as batch lookups.
