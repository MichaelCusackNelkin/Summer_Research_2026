# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A Columbia MAFN summer research project (per `README.md`) building an optimal-portfolio
backtest from estimated pieces — a return **forecast**, a **covariance** matrix, **spread
costs**, and a square-root market-**impact** model — solved as a portfolio optimization and
backtested on a **point-in-time (PIT) universe**. The original close-auction imbalance homework
is the alpha seed. The strategy is now **live (paper-traded)**: the same math runs intraday
against Interactive Brokers, entering at 15:50 and flattening at the 16:00 NASDAQ close cross.

**The research surface is four notebook directories** — `forecast/`, `covariance/`, `impact/`,
and `optimizer/` — each **persisting artifacts to disk**; `optimizer/optimal_portfolio.ipynb`
loads all of them and runs the daily MVO backtest. The `CAX/` corporate-actions downloader
feeds the covariance side clean (split/dividend-adjusted) prices. The **`live/`** directory is
the production port of that math (a daily orchestrator + a retrain job that rebuilds the same
artifacts on a schedule), and **`ibkr_broker/`** is the broker wrapper it trades through. The
`README.md` "Notes / Research Direction" list is the current open-problem backlog (condition
number, GARCH idiosyncratic vol, richer forecast features, lambda sweeps, intraday/multi-period
variants).

**Two deliberately disjoint date windows** run through the pipeline (see commit "none-
overlapping train sets"): the covariance is fit on **2023-06-01 .. 2025-06-01** daily bars,
while the forecast and optimizer operate on the **later, non-overlapping 2025-06-11 ..
2026-06-11** window. Keep them separate when editing — the covariance is a train-period risk
model consumed out-of-sample by the optimizer. (In `live/retrain.py` these become rolling
windows: 24 months for covariance, 12 months for forecast/impact, ending yesterday.)

> History: the repo began as two parallel projects (an execution project under
> `research-alpha/`/`sim_lib/`, plus this alpha direction). The execution code and the
> `research-alpha/` tree have since been **removed** — ignore any stale references to
> `research-alpha`, `sim_lib`, or `strategy_close_imbalance.py`. `pip` may still list
> `sim_lib` as an editable install pointing at a missing directory; do not import or
> reinstall it.
>
> Earlier cleanup also **retired the PIT-download orchestrator** (`build_universes.py`), its
> outputs (`universes/`), the deferred security-master plan (`docs/`), and the old
> `universe_selection` notebook (a lightweight `universe_selection.py` script replaced it —
> see below). `data_collection/src/data_collection/universe_pipeline.py` and its test file
> survive but are now **orphaned/legacy** — nothing imports the pipeline anymore. Treat it as
> reference, not active code; don't build new work on it without asking. The old
> `alpha_research/`/`optimization_research/` empty placeholders and the `covariance_research/`
> directory (renamed to `covariance/`) are also gone.

## Repository layout

- `covariance/` — the risk-model surface. `covariance.ipynb` builds the factor covariance and
  **saves its artifacts here** as `.npy`. There are **two vintages of names on disk**: the
  older set (`loadings.npy`, `factor_returns.npy`, `idiosyncratic_diag_vec.npy`,
  `cov_matrix.npy`) and the newer **`_return`-suffixed** set the `live/` system consumes
  (`loadings_return.npy`, `idiosyncratic_diag_return.npy`, `factor_cov_matrix.npy`,
  `cov_matrix_return.npy`, plus `name_vol.npy`). `live/retrain.py --refit covariance` writes
  the `_return` set; the optimizer notebook reads both. `covariance.py` is an earlier scratch
  script (exploratory, not wired in, stale column references like `price`/`date` — prefer the
  notebook).
- `forecast/` — the alpha-model surface. `forecasting.ipynb` builds and cross-validates the
  return forecast and **persists** the fitted sklearn pipeline to **`forecast_pipe.joblib`**
  (a `StandardScaler` + **`Lasso(alpha=1e-4)`**; the model choice moved from Ridge to Lasso).
- `impact/` — the market-impact surface. `impact_estimation.ipynb` calibrates a **square-root
  impact** model and persists `impact_gamma_vec.npy`, `impact_k_vec.npy`, and
  `reference_stats.parquet` (per-symbol ADV / daily vol used as the impact & cost denominators).
- `optimizer/` — `optimal_portfolio.ipynb`, the daily MVO backtest that **loads** the
  forecast (`../forecast/forecast_pipe.joblib`), risk model (`../covariance/*.npy`), and impact
  vectors, then solves a parametrized QP per day. (Moved here out of the old `notebook_studies/`.)
- `live/` — the **production port** of the research pipeline: a daily orchestrator, a retrain
  job, and a kill switch, trading through `ibkr_broker/`. Has its own detailed operator guide
  (`live/README.md`) and a full test suite (`live/tests/`). See the "Live trading system"
  architecture note below.
- `ibkr_broker/` — installable package (`ibkr-broker`, editable install), an `ib_async` wrapper
  exposing a synchronous `Broker` class (`buy`/`sell`/`buy_moc`/`get_positions`/
  `reconcile_positions`/`flatten_all`, etc.). Port map: 7497 paper, 7496 live, 4002/4001
  Gateway; live connections require the confirm string `YES_I_WANT_LIVE_TRADING`.
- `whitepaper/` — `whitepaper.qmd`, the Quarto writeup. (An earlier note here claimed the
  filename was misspelled `wihtepaper.qmd`; it is not, and never was on disk.)
- `CAX/` — corporate-actions download + recovery (splits & dividends from **yfinance**),
  the input to the covariance adjustment. See the architecture note below. Outputs
  `corporate_actions.csv` and `missing_corporate_actions.txt` next to the scripts.
- `data_collection/` — the only installable package (`data-collection`, installed editable
  into `.venv`). Core infrastructure everything else builds on:
  - `src/data_collection/data_adapters.py` — the Databento-backed three-layer cached
    adapter (`DatabentoGenericAdapter`). **Still central** — every notebook pulls bars
    through it. Has tests.
  - `src/data_collection/universe_pipeline.py` — pure PIT universe-selection logic.
    **Legacy/orphaned** (see history note); still has a passing test file.
- `universe_selection.py` (repo root) — small script that picks the top-`K` names by average
  daily dollar volume (`close*volume`) from an `ALL_SYMBOLS` daily-bar scan and writes
  `top_{K}_universe.txt`. Run from the repo root. (Note: the `Path.exists`/`top_k["by"]` calls
  are brittle — treat as a helper, verify output.)
- `top_49_universe.txt`, `top_50_universe.txt`, `top_100_universe.txt`, `top_250_universe.txt`,
  `top_500_universe.txt` — tiered ADV-ranked universes produced by `universe_selection.py`.
  **`top_49_universe.txt` is the active universe** as of 2026-07-21
  (`config.DEFAULT_ACTIVE_UNIVERSE_FILE`) — chosen for paper trading to avoid known universe
  issues while the strategy scales up; the plan is to grow toward 500 later. The on-disk
  covariance / forecast / impact artifacts are all sized for the top-49 set, so changing this
  without rebuilding all three produces shape mismatches at solve time.
  `top_100_universe.txt` is the previous active universe, kept for reference. `universe.txt`
  remains the **broad** pool that `live/retrain.py` scans for the historical fit.
- `universe.txt` — flat newline-delimited ticker list (~2000 names) used by `data.py`, `CAX/`,
  and as the broad pool. Uses Bloomberg/CRSP-style class delimiters (e.g. `BRK.B`).
- `data.py` (repo root) — small scratch script: materialize-only fetch of
  `imbalance`/`tbbo`/`statistics` for the flat `universe.txt` ticker list over a fixed
  window. (Calls `adapter.clear_cache()` on every run.)
- `notebook_studies/` — older research scratch (all notebooks, no `.py`): `alpha_forecast_study`,
  `rules_based_backtest`, `closing_imbalance_homework` (the original NYSE-IBF homework), and
  `test`. Superseded by `forecast/`/`optimizer/` for the live pipeline, kept for reference.
- `data/dbn_cache/generic/` — parquet + raw `.dbn.zst` adapter cache (gitignored: `**/data`).

Notebooks and scripts import the package as a top-level name
(`from data_collection.data_adapters import ...`) — there is no `PYTHONPATH` shim, it relies
on the editable install. **Working-directory matters** (paths are relative): each pipeline
notebook runs from its own directory and reaches up with `../` — e.g. `covariance.ipynb` reads
`../CAX/corporate_actions.csv` and `../top_100_universe.txt`; the optimizer reads
`../forecast/forecast_pipe.joblib` and `../covariance/*.npy`. `CAX/CAX.py`, `data.py`, and
`universe_selection.py` run from the repo root. The `live/` package, by contrast, is invoked
**as a module from the repo root** (`.venv/bin/python -m live.daily_orchestrator`) and resolves
artifact paths relative to the repo root, not a per-directory `../`.

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
- **Rebuild a tiered universe** (ADV top-`K` → `top_{K}_universe.txt`; edit `K`/window in the
  file): `.venv/bin/python universe_selection.py`.
- Re-install after editing `data_collection/pyproject.toml`:
  `.venv/bin/pip install -e ./data_collection`.
- **Install the broker package** (needed for anything in `live/`):
  `.venv/bin/pip install -e ./ibkr_broker`.
- Jupyter: `.venv/bin/jupyter lab` (or run notebooks through the venv `ipykernel`).

### Live trading system (`live/`) — run as modules from the repo root

`live/README.md` is the authoritative operator's guide (CLI flags, exit codes, guard matrix,
timing gotchas). The essentials:

- **Tests (cost 0, run these first):**
  `.venv/bin/python -m pytest data_collection/tests/ live/tests/ ibkr_broker/tests/`
  (the `ibkr_broker` `market_hours`-marked tests need a live TWS/market; unit tests do not).
- **Retrain artifacts** (rebuilds the covariance/forecast/impact sets on disk):
  `.venv/bin/python -m live.retrain --refit all` (or `covariance`/`forecast`/`impact`).
  Add `--dry-run` to print the plan without fetching or writing.
- **Broker-free replay** against cached historical data (validates
  feed→features→alpha→MVO→sizing without touching TWS):
  `.venv/bin/python -m live.daily_orchestrator --replay-date 2026-05-13 --paper`.
- **Paper trading** (real Databento feed, TWS paper port 7497):
  `.venv/bin/python -m live.daily_orchestrator --paper`.
- **Live trading** requires the exact confirm string:
  `.venv/bin/python -m live.daily_orchestrator --live --live-confirm YES_I_WANT_LIVE_TRADING`.
- **Kill switch** (account-wide flatten): `.venv/bin/python -m live.panic --paper`.

**Two guardrails not to trip over.** (1) Historical reads from `retrain` and
`--replay-date` route through a **coverage manifest** (`live/cache.py`) that *refuses to fetch*
from Databento's paid Batch API on a cache miss unless `--allow-fetch` (retrain also needs
`--yes`) — a stale `--end-date` or a drifted universe file raises `CoverageError` instead of
silently billing. (2) **Exit codes carry the pager predicate: `rc >= 11` is the ONLY code that
implies a live book at the close** (overnight exposure — page). Everything in `[2, 10]` is a
safe pre-order refusal.

`requirements.txt` lists the broad runtime deps; `data_collection/pyproject.toml` and
`ibkr_broker/pyproject.toml` are the authoritative dependency sets for the installed packages.

## Required environment

The adapter, `CAX`, and `data.py` read `DATABENTO_API_KEY` (yfinance in `CAX/` needs no key).
Note the **inconsistency**: the pipeline notebooks and `universe_selection.py` reference the
hyphenated `DATABENTO-API-KEY` (via `os.getenv("DATABENTO-API-KEY")`), while the `.env`
standard and the adapter's `_resolve_api_key` use the underscore `DATABENTO_API_KEY`
(`_resolve_api_key` accepts either). Prefer the underscore form. Loaded via `python-dotenv`
from `./.env`. The cache root the tools expect is `<repo_root>/data/dbn_cache/generic/`.

## Architecture notes that span multiple files

The research pipeline is four notebooks that hand artifacts to each other on disk. The forecast,
impact, and optimizer share the same data-loading preamble (imbalance/tbbo/statistics over the
2025-06-11 .. 2026-06-11 window, `top_100_universe.txt`); the covariance runs on its own
earlier daily-bar window. `live/` re-implements the same math for production and is the second
"span-multiple-files" surface — see its own note at the end.

### Covariance / risk model (`covariance/covariance.ipynb`)

End to end inside the notebook, producing a **factor covariance** `Σ_r = Lᵀ Σ_f L + D`:

1. Pull daily bars (`ohlcv-1d`, `XNAS.ITCH`, `2023-06-01 .. 2025-06-01`) for the universe
   through the adapter's `scan_data` (lazy Polars).
2. Load `../CAX/corporate_actions.csv`, fold splits/dividends into a per-symbol cumulative
   **adjustment factor**, join it onto `close`, and form `adj_close`.
3. Compute daily returns off `adj_close`, then **EWMA-vol-standardize** them (causal: the
   vol estimate is shifted so it uses only past info) to get unit-ish-variance residuals.
4. **SVD/PCA** on the standardized return matrix, keep the top-`k` factors (`k≈10`) to get
   loadings `L` and factor returns `F`; form the factor covariance `Σ_f` and an
   **idiosyncratic diagonal** `D` from the residual (per-name) variance, so the reconstruction
   `Lᵀ Σ_f L + D` is full-rank/PSD. Sanity-check **condition number** and PSD
   (`eigvalsh ≥ -1e-8`).
5. **Persist** the artifacts — the optimizer reads these directly. Note the two on-disk
   vintages: the older `loadings.npy` / `factor_returns.npy` / `idiosyncratic_diag_vec.npy` /
   `cov_matrix.npy`, and the `_return`-suffixed set the live system consumes
   (`loadings_return.npy`, `idiosyncratic_diag_return.npy`, `factor_cov_matrix.npy`,
   `cov_matrix_return.npy`, `name_vol.npy`).

Open problems (from `README.md`): condition number is still ~11,000 (want ≈10); the
idiosyncratic term should become **time-varying via GARCH(1,1)** (there is a stub markdown
section for this), and the optimizer wants a **market-neutrality constraint** downstream.

### Return forecast (`forecast/forecasting.ipynb`)

Forecasts the **15:50–16:00 return** (the close-auction window) from close-imbalance features.
Pulls `imbalance`, `tbbo`, and `statistics` schemas over the 2025-06-11 .. 2026-06-11 window,
engineers cross-sectional imbalance features (e.g. rank / pct-of-total net imbalance), and
runs an **IC study** with a `TimeSeriesSplit` (non-overlapping train sets) before fitting a
linear model. The chosen model is **`Lasso(alpha=1e-4)`** (moved off Ridge); the fitted sklearn
`Pipeline` (`StandardScaler` + `Lasso`) is saved to **`forecast_pipe.joblib`**. Later cells
explore **regime detection** — regressing daily IC on day-level features like an absorption
ratio — to find when the imbalance signal flips between momentum and reversion.

### Market impact (`impact/impact_estimation.ipynb`)

Calibrates a **square-root market-impact** model — impact ≈ `gamma · sign(x) · sqrt(|x| / ADV)`
scaled by daily vol — used as a term in the optimizer's QP objective. Persists a per-name
`impact_gamma_vec.npy` (and `impact_k_vec.npy`), plus `reference_stats.parquet` holding the
per-symbol ADV and daily-vol denominators. The optimizer/live path falls back to a scalar
`--impact-gamma` (default 0.5) if the vector is absent, but the reference-stats parquet is a
hard prerequisite for a live/replay run (its absence aborts the coverage assertion).

### Daily MVO backtest (`optimizer/optimal_portfolio.ipynb`)

Loads `../forecast/forecast_pipe.joblib` (alpha), `../covariance/*.npy` (risk), and the impact
vectors, then solves a
**per-day mean–variance QP with cvxpy** over the active set. Days are independent (the book is
flat at every close), so it **fans the days across processes**, compiling the parametrized QP
**once** per worker and re-solving with each day's alpha/costs plugged into `cp.Parameter`s.
Positions are **absolute-dollar** weights (`{date: {symbol: $}}`), not fractions. Downstream
cells do PnL accounting (dollar positions · returns-to-close, net of a spread-cost model),
diagnostics (Sharpe, GMV vs cumulative PnL), and **1-D parameter sweeps** (e.g. GMV vs the
risk-aversion `lambda`) that reuse the already-built per-day QP records.

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

### Live trading system (`live/`) — production port

`live/` re-implements the research math for live (paper) trading and is the second cross-file
surface. `live/README.md` is the authoritative operator's guide; the shape:

- **`daily_orchestrator.py`** (the big one, ~2000 lines) runs once per trading day: opens the
  Databento live feed, snapshots at 15:50 America/New_York, engineers the SAME features the
  forecast was trained on, predicts alpha, solves the MVO, converts dollar targets to signed
  shares, and executes a **two-leg batch**. The strategy is **intraday, not overnight** — the
  book is flat at the 16:00 close.
- **`retrain.py`** rebuilds the covariance/forecast/impact artifacts on a rolling window (24 /
  12 / 12 months) and writes them **atomically** (`.tmp` + `os.replace`), so the daily always
  reads a whole file. It scans the **broad** `universe.txt`, then filters to the active set.
- **`panic.py`** is the account-wide kill switch (flatten everything).
- **`features.py`** holds the shared feature engineering — `ALPHA_FEATURES` order is the
  contract between retrain (fit) and the daily (predict). Change it, retrain, *then* let the
  daily pick up the new pipeline. Also houses `build_factor_covariance`, `build_impact_frame`,
  `calibrate_impact_gamma` — the notebook math ported to functions.
- **`optimizer.py`** — `ForecastModel` / `CostModel` / `ImpactModel` / `Optimizer` and
  `dollar_positions_to_shares`, the cvxpy QP the notebook backtest also expresses.
- **`live_feed.py` / `replay_feed.py`** — the live Databento websocket feed and its
  cache-sourced replay twin; both feed the identical downstream path.
- **`cache.py`** — the **coverage manifest / resolver** (`CoverageManifest`, `CoverageResolver`,
  `CoverageError`) that fingerprints on-disk parquets by schema/symbols/date-range and refuses a
  paid Batch API fetch on a miss unless explicitly authorized. This is the anti-surprise-bill
  guardrail; see the "universe drift trap" in `live/README.md`.
- **`nasdaq_routing.py`** — the single source of `ACTIVE_ROUTING = "SMART"`. Every
  `live/` order routes via IB's SMART router; `NASDAQ_EXCHANGE = "ISLAND"` is
  retained but currently **unused**. Any new broker call site in `live/` MUST import and thread
  `ACTIVE_ROUTING` through rather than typing a literal or leaning on the broker default;
  enforcement is code review. SMART is a **known risk** for the MOC exit leg on any
  non-NASDAQ-primary name — a SMART MOC prints on the primary listing's close cross, not
  NASDAQ's, which is the wrong auction for a model calibrated to the NASDAQ cross. The
  compensating control is `_prequalify_symbols`, which drops any name whose
  `primaryExchange != "NASDAQ"`. That filter is **ACTIVE** (re-enabled 2026-07-21, committed in
  `ac2350d`, alongside the SMART switch). It is a no-op on both current tiered universes —
  49/49 in `top_49_universe.txt`, and 99 of the 100 in `top_100_universe.txt` qualify with all
  99 NASDAQ-primary (ITCI fails IB qualification outright and is dropped before the filter) —
  so nothing changes today; its value is universe-drift protection AND making the SMART routing
  premise load-bearing rather than assumed.
- **Exchange deadlines the order path is built around** (verified 2026-07-21; earlier versions
  of this file and several `live/` docstrings had these wrong). NASDAQ MOC **entry** cutoff is
  **15:55:00 ET** — the 2018 rule change (SEC release 34-84454) lifted the old 15:50 rule.
  NASDAQ MOC orders may not be **cancelled or modified** after **15:50:00 ET**. **NYSE** keeps a
  15:50 MOC entry cutoff, but this strategy trades the NASDAQ close cross exclusively, so 15:55
  is the load-bearing deadline. Hence `config.DEFAULT_SUBMIT_BY_TIME = "15:55"` is correct and
  must not be "fixed" earlier; note that the book is committed once MOCs are away after 15:50,
  since they can no longer be pulled or resized.
- Non-obvious execution invariant worth knowing before touching the order path: **LEG 2 (the
  MOC counter-flatten) is sized from a fresh `broker.get_positions()` after the entry fills,
  NEVER from `target - current`.** This is what prevents a partial entry fill from
  over-flattening into an opposite overnight exposure; `_submit_moc_flatten` structurally
  refuses any target-side argument. Logs land in `live/logs/` (`targets/`, `fills/`, and the
  orchestrator/retrain `.log` files); replay target books are quarantined under
  `live/logs/targets/replay/`.

### Security identity (note)

The data keys securities on **`raw_symbol`** (PIT-correct per day via Databento's
`map_symbols`), **not** `instrument_id` (which XNAS.ITCH remaps daily). A rigorous Security
Master identity map (handling renames like FB→META and recycled tickers) was designed in the
now-removed `docs/security_master_identity_plan.md`; it needs a Reference Data subscription we
don't have, so it remains unimplemented.
