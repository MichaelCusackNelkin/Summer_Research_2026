# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A Columbia MAFN summer research project (per `README.md`) building an optimal-portfolio
backtest from three estimated pieces — a return **forecast**, a **covariance** matrix, and
**spread costs** — solved as a portfolio optimization and backtested on a **point-in-time
(PIT) universe**. The original close-auction imbalance homework is the alpha seed.

**The active work on this branch (`covariance-estimation`) is the covariance matrix.** The
goal is a well-conditioned, factor-structured covariance estimate fit on corporate-action–
adjusted daily returns. Everything below is organized around that: the `covariance_research/`
notebook is the live surface, and the `CAX/` corporate-actions downloader exists to feed it
clean (split/dividend-adjusted) prices.

> History: the repo began as two parallel projects (an execution project under
> `research-alpha/`/`sim_lib/`, plus this alpha direction). The execution code and the
> `research-alpha/` tree have since been **removed** — ignore any stale references to
> `research-alpha`, `sim_lib`, or `strategy_close_imbalance.py`. `pip` may still list
> `sim_lib` as an editable install pointing at a missing directory; do not import or
> reinstall it.
>
> This branch also **retired the PIT-download orchestrator** (`build_universes.py`), its
> outputs (`universes/`), the deferred security-master plan (`docs/`), and the
> `universe_selection` notebook. `data_collection/src/data_collection/universe_pipeline.py`
> and its test file survive but are now **orphaned/legacy** — nothing imports the pipeline
> anymore. Treat it as reference, not active code; don't build new work on it without asking.

## Repository layout

- `covariance_research/` — **the active surface.** `covariance.ipynb` builds the covariance
  estimate; `covariance.py` is an earlier scratch script (exploratory, not wired in, and
  contains stale column references like `price`/`date` — prefer the notebook).
- `CAX/` — corporate-actions download + recovery (splits & dividends from **yfinance**),
  the input to the covariance adjustment. See the architecture note below. Outputs
  `corporate_actions.csv` and `missing_corporate_actions.txt` next to the scripts.
- `data_collection/` — the only installable package (`data-collection`, installed editable
  into `.venv`). Core infrastructure everything else builds on:
  - `src/data_collection/data_adapters.py` — the Databento-backed three-layer cached
    adapter (`DatabentoGenericAdapter`). **Still central** — the covariance notebook pulls
    daily bars through it. Has tests.
  - `src/data_collection/universe_pipeline.py` — pure PIT universe-selection logic.
    **Legacy/orphaned** (see history note); still has a passing test file.
- `data.py` (repo root) — small scratch script: materialize-only fetch of
  `imbalance`/`tbbo`/`statistics` for the flat `universe.txt` ticker list over a fixed
  window. (Calls `adapter.clear_cache()` on every run.)
- `universe.txt` — flat newline-delimited ticker list (~2000 names) used by `data.py`,
  `CAX/`, and the notebooks. Uses Bloomberg/CRSP-style class delimiters (e.g. `BRK.B`).
- `notebook_studies/` — the broader research surface (all notebooks, no `.py`):
  `optimal_portfolio` (the backtest that consumes the covariance), `alpha_forecast_study`,
  `rules_based_backtest`, `closing_imbalance_homework` (the original NYSE-IBF homework), and
  `test` scratch.
- `alpha_research/`, `optimization_research/` — currently empty placeholders.
- `data/dbn_cache/generic/` — parquet + raw `.dbn.zst` adapter cache (gitignored: `**/data`).

Notebooks and scripts import the package as a top-level name
(`from data_collection.data_adapters import ...`) — there is no `PYTHONPATH` shim, it relies
on the editable install. **Working-directory matters** (paths are relative):
`covariance.ipynb`/`covariance.py` run from `covariance_research/` (they read
`../CAX/corporate_actions.csv` / `../../../universe.txt`); `CAX/CAX.py` and `data.py` run
from the repo root.

## Common commands

Use the project venv explicitly — there is no `activate` step. Python is 3.14.

- **Download corporate actions** for `universe.txt` over the hard-coded window
  (`2023-06-01 .. 2026-06-11`, inclusive), writing `CAX/corporate_actions.csv`:
  `.venv/bin/python CAX/CAX.py`
  - Subset / recovery: `.venv/bin/python CAX/CAX.py --symbols NVDA,BRK.B --merge`
  - Knobs: `--workers` (default 16; lower to dodge Yahoo throttling), `--cooldown`.
- **Recover the names `CAX.py` couldn't fetch** (reads `CAX/recovery_failed.txt`). Note the
  in-file docstring path is stale — `cax_recover.py` does `import CAX.CAX`, so it must run as
  a **module from the repo root**, not as a script:
  `.venv/bin/python -m CAX.cax_recover`
  (recommended for a cold start after heavy use: `--cooldown 1800 --classify-delay 12`).
- **Tests** (both modules in `data_collection`):
  `.venv/bin/python -m pytest data_collection/tests/`
  — `pyproject.toml` sets `asyncio_mode = "auto"`, so `async def test_*` run without a marker.
- **Single test:**
  `.venv/bin/python -m pytest data_collection/tests/test_universe_pipeline.py::test_select_universe_top_n`
- **Materialize raw schema parquets** for `universe.txt`: `.venv/bin/python data.py`.
- Re-install after editing `data_collection/pyproject.toml`:
  `.venv/bin/pip install -e ./data_collection`.
- Jupyter: `.venv/bin/jupyter lab` (or run notebooks through the venv `ipykernel`).

`requirements.txt` lists the broad runtime deps; `data_collection/pyproject.toml` is the
authoritative dependency set for the installed package.

## Required environment

The adapter, `CAX`, and `data.py` read `DATABENTO_API_KEY` (yfinance in `CAX/` needs no key).
Note the **inconsistency**: notebooks and `covariance_research/covariance.py` reference the
hyphenated `DATABENTO-API-KEY`, while the `.env` standard and the adapter's `_resolve_api_key`
use the underscore `DATABENTO_API_KEY` (`_resolve_api_key` accepts either). Prefer the
underscore form. Loaded via `python-dotenv` from `./.env`. The cache root the tools expect is
`<repo_root>/data/dbn_cache/generic/`.

## Architecture notes that span multiple files

### Covariance estimation pipeline (`covariance_research/covariance.ipynb`)

The active build, end to end inside the notebook:

1. Pull daily bars (`ohlcv-1d`, `XNAS.ITCH`) for `universe.txt` through the adapter's
   `scan_data` (lazy Polars).
2. Load `../CAX/corporate_actions.csv`, fold splits/dividends into a per-symbol cumulative
   **adjustment factor**, join it onto `close`, and form `adj_close`.
3. Compute daily returns off `adj_close`, then **EWMA-vol-standardize** them (causal: the
   vol estimate is shifted so it uses only past info) to get unit-ish-variance residuals.
4. Estimate covariance with a **factor structure**: SVD/PCA on the standardized return
   matrix, keep the top-`k` factors (`k≈10`), reconstruct `best_k_cov = Vᵀ S² V`, and sanity-
   check **condition number** and **PSD** (`eigvalsh ≥ -1e-8`).

When extending this, the open problems from `README.md` are the guide: covariance needs more
daily data and a real factor structure (plus a market-neutrality constraint in the optimizer
downstream).

### Corporate actions (`CAX/`) — input to the adjustment

`CAX.py` fetches splits + dividends for every `universe.txt` ticker from yfinance's
`.actions`, restricted to the hard-coded window, and writes a tidy CSV
(`symbol,date,dividend,split`). The non-obvious engineering is **throttle handling**, because
Yahoo aggressively rate-limits ~2000 names and its failure modes are ambiguous:

- A ticker that returns **zero history rows** is treated as a *retryable* error, **not** an
  empty result — a throttled live name and a genuinely delisted name look identical, so it
  refuses to record "no actions" without positive evidence Yahoo served data. Only a frame
  that came back with real rows-but-no-events is recorded as a trustworthy empty.
- Class delimiters are remapped for Yahoo (`BRK.B` → `BRK-B`), trying the literal form first.
- Concurrent pass (16 threads) with per-ticker exponential backoff, then up to 3 resweeps of
  only the still-errored tickers.

`cax_recover.py` handles the residue: it **classifies** failures into `live` / `nodata`
(404/zero-row — almost always a closed-M&A delisting purged from Yahoo, e.g. DFS, HES, JNPR)
/ `transient`, patiently retries only `transient`, **merges** recovered rows into the CSV
(replacing just those symbols' rows), and writes `missing_corporate_actions.txt` — the known
gap the covariance side must cover another way (recover **splits** for those names from the
Databento raw close series; free dividends are unavailable for them).

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
layers are **not** aware of overlapping date ranges or supersets. Defaults:
`dataset="XNAS.ITCH"`, `schema="ohlcv-1m"`; `end_date` is **exclusive** and defaults to today.

- `cache_only=True` refuses the network (raises `FileNotFoundError` on miss).
- `materialize_only=True` ensures the parquet exists and returns `None` instead of eagerly
  reading it — for callers that read back via `scan_data`.
- `clear_cache()` only removes the parquet layer; raw `.dbn.zst` files are preserved.

`scan_data(...)` is the lazy synchronous sibling: returns a `pl.LazyFrame` from
`pl.scan_parquet` and **raises `FileNotFoundError` if the parquet layer is missing** — it
never decodes raw DBN and never hits the API. Materialize once with `get_data(...)` first.
The canonical pattern: try `scan_data`, and on `FileNotFoundError` call
`get_data(..., materialize_only=True)`. Blocking SDK calls are wrapped in `asyncio.to_thread`,
so `get_data` is safe to `await` from a Jupyter event loop.

### Security identity (note)

The data keys securities on **`raw_symbol`** (PIT-correct per day via Databento's
`map_symbols`), **not** `instrument_id` (which XNAS.ITCH remaps daily). A rigorous Security
Master identity map (handling renames like FB→META and recycled tickers) was designed in the
now-removed `docs/security_master_identity_plan.md`; it needs a Reference Data subscription we
don't have, so it remains unimplemented.
