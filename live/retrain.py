"""Retrain job.

Runs on a slower cadence (weekly / monthly) than the daily orchestrator.
Refreshes the historical Databento cache on a rolling window and rebuilds
whichever artifacts the user asks for:

- ``covariance`` : the six ``.npy`` files in ``covariance/`` (loadings, factor
  covariance, idiosyncratic diagonal, cov matrix, name vol) built from
  ``ohlcv-1d`` bars adjusted for splits/dividends via ``CAX/corporate_actions.csv``.
- ``forecast`` : the sklearn Pipeline in ``forecast/forecast_pipe.joblib``,
  fit on the same feature engineering the daily uses (imbalance + tbbo +
  statistics).
- ``impact`` : the per-name gamma vector at ``impact/impact_gamma_vec.npy``
  (see ``impact/impact_estimation.ipynb``) and a per-symbol reference-stats
  parquet at ``impact/reference_stats.parquet`` (per-symbol ADV, vol) that
  the daily orchestrator loads at startup.
- ``all`` : all three.

The retrain is additive: it OVERWRITES derived ``.npy``/``.joblib``/parquet
artifacts but never touches the raw ``.dbn.zst`` cache or the parquet cache
under ``data/dbn_cache/``.

Usage:
    .venv/bin/python -m live.retrain --refit all
    .venv/bin/python -m live.retrain --refit forecast --window-months 6
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import polars as pl
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data_collection.data_adapters import DatabentoGenericAdapter

from live import config
from live.cache import (
    CoverageError,
    CoverageManifest,
    CoverageResolver,
    CoverageStatus,
    format_coverage_decision,
    summarize_manifest,
)
from live.features import (
    ALPHA_FEATURES,
    TARGET_COLUMN,
    apply_corporate_actions,
    build_factor_covariance,
    build_feature_frame,
    build_impact_frame,
    calibrate_impact_gamma,
    compute_auction_prices,
    compute_avg_daily_dollar_volume,
    compute_close_trades,
    compute_daily_vol,
    compute_imbalance_print,
    compute_recent_return,
)

logger = logging.getLogger("live.retrain")



# ---------------------------------------------------------------------------
# Atomic write helpers (H3)
#
# The daily orchestrator reads these files at startup — if retrain crashes
# mid-write the daily could otherwise load a shape-mismatched or truncated
# artifact. Every artifact is staged to a temp path in the same directory
# and atomically ``os.replace``d into place at the very end of a successful
# refit. All-or-nothing per artifact.
# ---------------------------------------------------------------------------

import os
import tempfile


def _atomic_np_save(path: Path, array) -> None:
    """Atomic ``np.save`` — always writes real data to the target path.

    Prior implementation passed ``tmp`` (the string path) into ``np.save``.
    When the string lacks a ``.npy`` suffix numpy silently appends one and
    writes to a DIFFERENT file (``<tmp>.npy``), leaving the original empty
    ``<tmp>`` from ``mkstemp`` in place. ``os.replace(tmp, path)`` then
    moves the empty file over the target and orphans the real data.

    Fix: write through the file DESCRIPTOR ``mkstemp`` returns — numpy
    accepts a file object and never renames.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".npy.tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            np.save(fh, array)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _atomic_joblib_dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".joblib.tmp", dir=str(path.parent))
    try:
        os.close(fd)
        joblib.dump(obj, tmp)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _atomic_write_parquet(path: Path, frame: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".parquet.tmp", dir=str(path.parent))
    try:
        os.close(fd)
        frame.write_parquet(tmp)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_REFIT_CHOICES = ("all", "covariance", "forecast", "impact")


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Refresh cached historical data and rebuild covariance / forecast / impact artifacts."
    )
    p.add_argument(
        "--refit", choices=_REFIT_CHOICES, default="all",
        help="Which artifact set to rebuild. Default: all.",
    )
    p.add_argument(
        "--window-months", type=int, default=12,
        help="Rolling window (months) for the forecast/impact fit. Default: 12.",
    )
    p.add_argument(
        "--covariance-window-days", type=int,
        default=config.DEFAULT_COVARIANCE_WINDOW_DAYS,
        help=f"Days of daily bars for the covariance fit. Default: {config.DEFAULT_COVARIANCE_WINDOW_DAYS}.",
    )
    p.add_argument(
        "--end-date", default=None,
        help="Explicit end date (YYYY-MM-DD, exclusive per Databento). Default: yesterday.",
    )
    p.add_argument(
        "--n-factors", type=int, default=10,
        help="Number of SVD factors for the covariance. Default: 10.",
    )
    p.add_argument(
        "--broad-universe-file", default=config.DEFAULT_BROAD_UNIVERSE_FILE,
        help=f"Broad-universe file. Default: {config.DEFAULT_BROAD_UNIVERSE_FILE}.",
    )
    p.add_argument(
        "--active-universe-file", default=config.DEFAULT_ACTIVE_UNIVERSE_FILE,
        help=f"Active-universe file. Default: {config.DEFAULT_ACTIVE_UNIVERSE_FILE}.",
    )
    p.add_argument(
        "--forecast-alpha", type=float, default=1e-4,
        help="Lasso regularisation strength. Default: 1e-4.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print the plan (windows, symbol counts, artifact paths) but write nothing.",
    )
    p.add_argument(
        "--allow-fetch", action="store_true", default=False,
        help=(
            "Authorise the resolver to submit a Databento Batch API job "
            "when the coverage manifest cannot serve a request. Off by "
            "default — a MISS raises with a helpful message so accidental "
            "batch charges cannot happen from a stale end-date. Must be "
            "combined with --yes (or an interactive prompt) before any "
            "batch job is submitted."
        ),
    )
    p.add_argument(
        "--yes", action="store_true", default=False,
        help=(
            "Non-interactive confirmation to proceed with fetches "
            "authorised by --allow-fetch. Batch API calls require BOTH "
            "--allow-fetch AND --yes to be set (belt-and-braces). "
            "If unset, the resolver will offer an interactive prompt on "
            "a TTY; on a non-interactive terminal the run refuses."
        ),
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Historical fetch helpers
# ---------------------------------------------------------------------------

def _end_date(args: argparse.Namespace) -> str:
    if args.end_date:
        return args.end_date
    return config.yesterday_iso()


def _start_date(end_date_iso: str, days: int) -> str:
    end = date.fromisoformat(end_date_iso)
    return (end - timedelta(days=days)).isoformat()


async def _ensure_covered(
    resolver: CoverageResolver,
    tickers,
    start_date: str,
    end_date: str,
    schema: str,
    required_symbols,
    allow_fetch: bool,
) -> None:
    """Route a materialise-only request through the coverage resolver.

    Coverage HIT (existing parquet fully covers the window/symbols) is a
    no-op — the caller reads the covering parquet directly via
    ``resolver.resolve_lazy``. Coverage MISS raises ``CoverageError``
    unless ``allow_fetch=True``, in which case the resolver invokes the
    adapter's batch-API path just like the old ``_materialize`` helper.
    """
    await resolver.materialize(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        schema=schema,
        required_symbols=required_symbols,
        allow_fetch=allow_fetch,
    )


def _historical_frames(
    resolver: CoverageResolver,
    universe: list[str],
    active_set: list[str],
    start_date: str,
    end_date: str,
) -> dict[str, pl.LazyFrame]:
    """Scan the cache for the three schemas we need, filtered to the active
    set + New York timezone + market hours. Mirrors the notebook preambles.

    Uses ``resolver.resolve_lazy`` so a coverage-HIT parquet (different
    hash slot, wider window / symbol set) can serve the request without
    an unnecessary re-fetch. ``allow_fetch=False`` here — anything the
    resolver would need to fetch was already materialised upstream.
    """
    df_imbalance = (
        resolver.resolve_lazy(
            tickers=universe,
            start_date=start_date, end_date=end_date,
            schema="imbalance",
            required_symbols=active_set,
            allow_fetch=False,
        )
        .select(["ts_event", "symbol", "side", "total_imbalance_qty", "paired_qty", "ref_price"])
        .with_columns(
            pl.col("ts_event").dt.convert_time_zone("America/New_York"),
            pl.col("symbol").cast(pl.Categorical),
        )
        .filter(
            pl.col("ts_event").dt.time().is_between(pl.time(9, 30), pl.time(16, 0)),
            pl.col("symbol").is_in(active_set),
        )
    )
    df_trades = (
        resolver.resolve_lazy(
            tickers=universe,
            start_date=start_date, end_date=end_date,
            schema="tbbo",
            required_symbols=active_set,
            allow_fetch=False,
        )
        .select(["ts_event", "symbol", "price", "size", "ask_px_00", "bid_px_00"])
        .filter(
            pl.col("ts_event").dt.time().is_between(pl.time(13, 30), pl.time(21, 0)),
            pl.col("symbol").is_in(active_set),
        )
        .with_columns(pl.col("ts_event").dt.convert_time_zone("America/New_York"))
        .filter(pl.col("ts_event").dt.time().is_between(pl.time(9, 30), pl.time(16, 0)))
        .with_columns(pl.col("symbol").cast(pl.Categorical))
    )
    df_stats = (
        resolver.resolve_lazy(
            tickers=universe,
            start_date=start_date, end_date=end_date,
            schema="statistics",
            required_symbols=active_set,
            allow_fetch=False,
        )
        .select(["ts_event", "stat_type", "symbol", "price", "quantity"])
        .with_columns(
            pl.col("ts_event").dt.convert_time_zone("America/New_York"),
            pl.col("symbol").cast(pl.Categorical),
        )
        .filter(
            pl.col("stat_type") == 11,
            pl.col("symbol").is_in(active_set),
        )
    )
    return {"imbalance": df_imbalance, "tbbo": df_trades, "statistics": df_stats}


def _log_plan(
    resolver: CoverageResolver,
    requests: list[dict],
) -> list[tuple[dict, "CoverageDecision"]]:
    """Print a HIT/MISS coverage plan for a batch of proposed requests.

    Each ``requests`` entry is the kwargs of a would-be ``resolver.plan``
    call. Returns the (request, decision) pairs so callers can act on
    them without re-planning.
    """
    plan: list[tuple[dict, "CoverageDecision"]] = []
    for req in requests:
        decision = resolver.plan(**req)
        schema = req.get("schema")
        # Format the HIT/MISS decision into the log MESSAGE (not just
        # ``extra``) so the terminal stream handler — which only emits
        # ``%(message)s`` — shows the parquet name, reason and shortfall.
        # Without this the operator sees a bare "coverage plan" line and
        # has to grep the JSON log to find out what's missing.
        line = format_coverage_decision(schema, decision)
        logger.info(
            "coverage plan — %s", line,
            extra={
                "schema": schema,
                "start": req.get("start_date"),
                "end": req.get("end_date"),
                "status": decision.status.value,
                "path": str(decision.covering_path) if decision.covering_path else None,
                "nearest_parquet": (
                    str(decision.nearest_parquet) if decision.nearest_parquet else None
                ),
                "shortfall_distance": decision.shortfall_distance,
                "reason": decision.reason,
            },
        )
        plan.append((req, decision))
    return plan


def _confirm_fetch(args: argparse.Namespace, missing: list[dict]) -> bool:
    """Return True iff the operator has authorised the pending fetches.

    ``--allow-fetch`` alone is not enough for retrain — batch jobs cost
    real money. ``--yes`` (non-interactive) or a live-TTY prompt is the
    second gate. On a non-interactive terminal without ``--yes`` we
    refuse rather than block indefinitely.
    """
    if not missing:
        return True
    if not args.allow_fetch:
        return False
    if args.yes:
        return True
    if not sys.stdin.isatty():
        logger.error(
            "coverage MISS on %d schema(s) and --allow-fetch given but no "
            "--yes and no TTY for prompting; refusing to submit batch jobs",
            len(missing),
            extra={"n_missing": len(missing)},
        )
        return False
    print(
        f"\n{len(missing)} schema(s) would be fetched via the Databento Batch API. "
        "This is a paid operation.",
        file=sys.stderr,
    )
    for req in missing:
        print(
            f"  - schema={req.get('schema')} "
            f"start={req.get('start_date')} end={req.get('end_date')}",
            file=sys.stderr,
        )
    # R5: Ctrl-D / Ctrl-C at the prompt must land on the same clean
    # "refused" path as a "no" answer — not a bare traceback out to
    # ``asyncio.run`` (which would leave the operator staring at a
    # stack trace and no clear indication whether a batch was submitted).
    try:
        resp = input("Proceed? Type 'yes' to confirm: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return resp == "yes"


# ---------------------------------------------------------------------------
# Covariance
# ---------------------------------------------------------------------------

async def _refit_covariance(
    resolver: CoverageResolver,
    active_set: list[str],
    end_date: str,
    days: int,
    n_factors: int,
    dry_run: bool,
    allow_fetch: bool,
) -> None:
    start_date = _start_date(end_date, days)
    logger.info(
        "refitting covariance",
        extra={
            "start": start_date, "end": end_date,
            "n_active": len(active_set), "n_factors": n_factors,
        },
    )
    if dry_run:
        logger.info("dry-run: skipping covariance materialize + fit + write")
        return
    await _ensure_covered(
        resolver, "ALL_SYMBOLS", start_date, end_date,
        schema="ohlcv-1d",
        required_symbols=active_set,
        allow_fetch=allow_fetch,
    )

    df_bars = (
        resolver.resolve_lazy(
            tickers="ALL_SYMBOLS",
            start_date=start_date, end_date=end_date,
            schema="ohlcv-1d",
            required_symbols=active_set,
            allow_fetch=False,
        )
        .select(["ts_event", "symbol", "close"])
        .with_columns(pl.col("symbol").cast(pl.Categorical))
        .filter(pl.col("symbol").is_in(active_set))
        .sort(["ts_event", "symbol"])
        .collect(engine="streaming")
    )
    if df_bars.is_empty():
        raise RuntimeError(
            f"No daily bars found for {len(active_set)} active symbols in [{start_date}, {end_date})."
        )

    cax_path = config.REPO_ROOT / "CAX" / "corporate_actions.csv"
    if not cax_path.exists():
        raise FileNotFoundError(
            f"corporate_actions.csv missing at {cax_path}. Run `python CAX/CAX.py`."
        )
    cax = (
        pl.read_csv(cax_path)
        .sort(["date", "symbol"])
        .with_columns(
            pl.col("date").str.to_datetime(time_unit="ns", time_zone="UTC"),
            pl.col("symbol").cast(pl.Categorical),
        )
    )
    adj_px = apply_corporate_actions(df_bars, cax)

    result = build_factor_covariance(adj_px, active_set, n_factors=n_factors)
    logger.info(
        "covariance fit complete",
        extra={
            "condition_number": result["condition_number"],
            "cov_shape": tuple(result["cov_matrix_return"].shape),
            "loadings_shape": tuple(result["loadings_return"].shape),
        },
    )

    if dry_run:
        logger.info("dry-run: skipping covariance write")
        return

    config.COV_DIR.mkdir(parents=True, exist_ok=True)
    _atomic_np_save(config.COV_DIR / config.COV_LOADINGS_NAME, result["loadings_return"])
    _atomic_np_save(config.COV_DIR / config.COV_EPS_DIAG_NAME, result["idiosyncratic_diag_return"])
    _atomic_np_save(config.COV_DIR / config.COV_FACTOR_COV_NAME, result["factor_cov_matrix"])
    _atomic_np_save(config.COV_DIR / config.COV_MATRIX_NAME, result["cov_matrix_return"])
    _atomic_np_save(config.COV_DIR / config.COV_NAME_VOL_NAME, result["name_vol"])
    logger.info(
        "covariance artifacts written",
        extra={"dir": str(config.COV_DIR)},
    )


# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------

async def _refit_forecast(
    resolver: CoverageResolver,
    universe: list[str],
    active_set: list[str],
    end_date: str,
    window_months: int,
    forecast_alpha: float,
    dry_run: bool,
    allow_fetch: bool,
) -> pl.DataFrame:
    """Refit the sklearn Pipeline and return the assembled training frame
    so the impact-refit step can reuse the same data windows and cached
    schemas.
    """
    days = window_months * 30
    start_date = _start_date(end_date, days)
    logger.info(
        "refitting forecast",
        extra={
            "start": start_date, "end": end_date,
            "n_universe": len(universe), "n_active": len(active_set),
            "lasso_alpha": forecast_alpha,
        },
    )
    if dry_run:
        logger.info("dry-run: skipping forecast materialize + fit + write")
        return None
    for schema in ("imbalance", "tbbo", "statistics"):
        await _ensure_covered(
            resolver, universe, start_date, end_date,
            schema=schema,
            required_symbols=active_set,
            allow_fetch=allow_fetch,
        )

    frames = _historical_frames(resolver, universe, active_set, start_date, end_date)

    close_trades = compute_close_trades(frames["tbbo"]).collect(engine="streaming")
    recent_return = compute_recent_return(frames["tbbo"]).collect(engine="streaming")
    avg_daily_dollar_volume = compute_avg_daily_dollar_volume(frames["tbbo"]).collect(engine="streaming")
    daily_vol = compute_daily_vol(frames["tbbo"]).collect(engine="streaming")
    imb_print = compute_imbalance_print(frames["imbalance"]).collect(engine="streaming")
    auction = compute_auction_prices(frames["statistics"]).collect(engine="streaming")

    training = build_feature_frame(
        imb_print=imb_print,
        close_trades=close_trades,
        avg_daily_dollar_volume=avg_daily_dollar_volume,
        daily_vol=daily_vol,
        recent_return=recent_return,
        auction=auction,
    )
    if training.is_empty():
        raise RuntimeError(
            f"No training rows built in [{start_date}, {end_date}) for {len(active_set)} active symbols."
        )
    logger.info(
        "assembled training frame",
        extra={"n_rows": training.height, "n_days": int(training["date"].n_unique())},
    )

    X = training[ALPHA_FEATURES].to_numpy()
    y = training[TARGET_COLUMN].to_numpy()
    pipe = make_pipeline(StandardScaler(), Lasso(alpha=forecast_alpha))
    pipe.fit(X, y.reshape(-1, 1))

    from scipy.stats import spearmanr
    y_hat = pipe.predict(X).ravel()
    ic, _ = spearmanr(y, y_hat)
    r2 = float(pipe.score(X, y.reshape(-1, 1)))
    logger.info(
        "forecast fit complete",
        extra={"in_sample_ic": float(ic), "in_sample_r2": r2},
    )

    if dry_run:
        logger.info("dry-run: skipping forecast write")
        return training

    config.FORECAST_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FORECAST_DIR / config.FORECAST_PIPE_NAME
    _atomic_joblib_dump(path, pipe)
    logger.info("forecast pipeline written", extra={"path": str(path)})
    return training


# ---------------------------------------------------------------------------
# Impact
# ---------------------------------------------------------------------------

async def _refit_impact(
    resolver: CoverageResolver,
    universe: list[str],
    active_set: list[str],
    end_date: str,
    window_months: int,
    training: Optional[pl.DataFrame],
    dry_run: bool,
    allow_fetch: bool,
) -> None:
    days = window_months * 30
    start_date = _start_date(end_date, days)
    logger.info(
        "refitting impact",
        extra={"start": start_date, "end": end_date, "n_active": len(active_set)},
    )
    if dry_run:
        logger.info("dry-run: skipping impact materialize + fit + write")
        return
    await _ensure_covered(
        resolver, universe, start_date, end_date,
        schema="tbbo",
        required_symbols=active_set,
        allow_fetch=allow_fetch,
    )

    tbbo_scan = resolver.resolve_lazy(
        tickers=universe,
        start_date=start_date, end_date=end_date,
        schema="tbbo",
        required_symbols=active_set,
        allow_fetch=False,
    ).with_columns(
        pl.col("ts_event").dt.convert_time_zone("America/New_York"),
        pl.col("symbol").cast(pl.Categorical),
    )
    fit = build_impact_frame(tbbo_scan, active_set)
    if fit.is_empty():
        raise RuntimeError(
            f"Impact fit frame empty for [{start_date}, {end_date})."
        )
    gamma_vec, diagnostics = calibrate_impact_gamma(fit, active_set)
    logger.info(
        "impact fit complete",
        extra={
            "gamma_median": float(np.median(gamma_vec)),
            "gamma_pool": diagnostics["gamma_pool"],
            "n_fit_symbols": diagnostics["n_fit_symbols"],
        },
    )

    # Reference stats: per-symbol ADV (latest available EWMA row) + latest
    # daily vol. The daily orchestrator joins this into its live day-frame
    # so the impact model has non-null inputs at 15:50.
    tbbo_lf = tbbo_scan
    adv = compute_avg_daily_dollar_volume(tbbo_lf).collect(engine="streaming")
    vol = compute_daily_vol(tbbo_lf).collect(engine="streaming")
    ref_stats = (
        adv.join(vol, on=["date", "symbol"], how="inner")
        .drop_nulls(["ADV", "vol"])
        .sort(["symbol", "date"])
        .group_by("symbol", maintain_order=True)
        .agg(
            pl.col("ADV").last().alias("ADV"),
            pl.col("vol").last().alias("vol"),
            pl.col("date").last().alias("as_of_date"),
        )
    )
    logger.info(
        "reference stats built",
        extra={
            "n_symbols": ref_stats.height,
            "adv_median": float(ref_stats["ADV"].median() or 0),
            "vol_median": float(ref_stats["vol"].median() or 0),
        },
    )

    if dry_run:
        logger.info("dry-run: skipping impact / reference-stats write")
        return

    config.IMPACT_DIR.mkdir(parents=True, exist_ok=True)
    _atomic_np_save(config.IMPACT_DIR / config.IMPACT_GAMMA_NAME, gamma_vec)
    _atomic_write_parquet(config.IMPACT_DIR / config.REF_STATS_NAME, ref_stats)
    logger.info(
        "impact artifacts written",
        extra={
            "gamma_path": str(config.IMPACT_DIR / config.IMPACT_GAMMA_NAME),
            "reference_stats_path": str(config.IMPACT_DIR / config.REF_STATS_NAME),
        },
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def _amain(args: argparse.Namespace) -> int:
    universe = config.load_broad_universe(args.broad_universe_file)
    active_set = config.load_active_universe(args.active_universe_file)
    end_date = _end_date(args)
    logger.info(
        "retrain starting",
        extra={
            "refit": args.refit,
            "end_date": end_date,
            "window_months": args.window_months,
            "n_broad": len(universe),
            "n_active": len(active_set),
            "dry_run": args.dry_run,
            "allow_fetch": args.allow_fetch,
            "yes": args.yes,
        },
    )

    api_key = config.resolve_databento_key()
    adapter = DatabentoGenericAdapter(api_key, config.CACHE_DIR)
    manifest = CoverageManifest(config.CACHE_DIR)
    resolver = CoverageResolver(adapter, manifest=manifest)
    logger.info(
        "cache coverage manifest\n%s",
        summarize_manifest(manifest),
    )

    # Pre-flight coverage plan across everything the caller asked us to
    # refit. If nothing would be fetched, we proceed straight through.
    # Otherwise, either --dry-run just prints the plan or we gate the run
    # on --allow-fetch AND (--yes | interactive-y).
    planned_requests = _build_planned_requests(args, universe, active_set, end_date)
    plan_summary = _log_plan(resolver, planned_requests)
    missing = [req for req, decision in plan_summary if decision.status is CoverageStatus.MISS]

    if args.dry_run:
        logger.info(
            "dry-run: coverage plan printed, nothing fetched or written",
            extra={"n_hit": len(plan_summary) - len(missing), "n_miss": len(missing)},
        )
        return 0

    if missing and not args.allow_fetch:
        logger.error(
            "coverage MISS on %d schema(s); refusing to fetch without --allow-fetch",
            len(missing),
            extra={"n_missing": len(missing)},
        )
        return 9
    if missing and not _confirm_fetch(args, missing):
        logger.error(
            "fetch confirmation not received; refusing to submit batch job(s)",
            extra={"n_missing": len(missing)},
        )
        return 10

    # Passed the confirmation gate above, so this simply mirrors
    # --allow-fetch. On a HIT-only plan it doesn't matter (no fetch will
    # happen); on a MISS plan it authorises the resolver to submit the
    # batch job the operator just confirmed.
    fetch_allowed = args.allow_fetch

    try:
        if args.refit in ("all", "covariance"):
            await _refit_covariance(
                resolver=resolver,
                active_set=active_set,
                end_date=end_date,
                days=args.covariance_window_days,
                n_factors=args.n_factors,
                dry_run=args.dry_run,
                allow_fetch=fetch_allowed,
            )

        training: Optional[pl.DataFrame] = None
        if args.refit in ("all", "forecast"):
            training = await _refit_forecast(
                resolver=resolver,
                universe=universe,
                active_set=active_set,
                end_date=end_date,
                window_months=args.window_months,
                forecast_alpha=args.forecast_alpha,
                dry_run=args.dry_run,
                allow_fetch=fetch_allowed,
            )

        if args.refit in ("all", "impact"):
            await _refit_impact(
                resolver=resolver,
                universe=universe,
                active_set=active_set,
                end_date=end_date,
                window_months=args.window_months,
                training=training,
                dry_run=args.dry_run,
                allow_fetch=fetch_allowed,
            )
    except CoverageError as e:
        logger.error("coverage error during refit: %s", e)
        return 9

    logger.info(
        "retrain complete",
        extra={
            "refit": args.refit,
            "artifacts_dir_forecast": str(config.FORECAST_DIR),
            "artifacts_dir_covariance": str(config.COV_DIR),
            "artifacts_dir_impact": str(config.IMPACT_DIR),
        },
    )
    return 0


def _build_planned_requests(
    args: argparse.Namespace,
    universe: list[str],
    active_set: list[str],
    end_date: str,
) -> list[dict]:
    """Enumerate every ``(tickers, window, schema)`` this run will touch.

    Feeds ``_log_plan`` and the fetch-confirmation gate. Mirrors the
    per-refit fetch shapes exactly so the plan cannot drift from what
    each ``_refit_*`` actually asks for.
    """
    requests: list[dict] = []
    if args.refit in ("all", "covariance"):
        cov_start = _start_date(end_date, args.covariance_window_days)
        requests.append({
            "tickers": "ALL_SYMBOLS",
            "start_date": cov_start, "end_date": end_date,
            "schema": "ohlcv-1d",
            "required_symbols": active_set,
        })
    window_days = args.window_months * 30
    if args.refit in ("all", "forecast"):
        f_start = _start_date(end_date, window_days)
        for schema in ("imbalance", "tbbo", "statistics"):
            requests.append({
                "tickers": universe,
                "start_date": f_start, "end_date": end_date,
                "schema": schema,
                "required_symbols": active_set,
            })
    if args.refit in ("all", "impact"):
        i_start = _start_date(end_date, window_days)
        # Only add tbbo again if forecast didn't already request it.
        already = any(
            r.get("schema") == "tbbo" and r.get("start_date") == i_start
            and r.get("end_date") == end_date
            for r in requests
        )
        if not already:
            requests.append({
                "tickers": universe,
                "start_date": i_start, "end_date": end_date,
                "schema": "tbbo",
                "required_symbols": active_set,
            })
    return requests


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    config.setup_logging("live.retrain")
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    sys.exit(main())
