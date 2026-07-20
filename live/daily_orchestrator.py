"""Daily live-trading orchestrator.

Runs once per trading day. Timeline:

  ~09:30 ET  live feed opens; imbalance + tbbo start buffering
   15:50 ET  snapshot buffers, engineer features, run alpha + MVO,
             convert dollar targets to signed share counts, reconcile
             against current IBKR positions
   15:50 ET  ENTRY batch: per-name MKT sized to ``target - current``
             via ``broker.buy`` / ``broker.sell`` (default tif=DAY).
   15:50 ET  Bounded fill-wait (per-order + global budget tied to the
             MOC cutoff), then belt-and-braces cancel of any entry
             still in a non-terminal state.
   15:50 ET  EXIT batch: per-name MOC sized to the ACTUAL post-entry
             position (never to ``target - current``). This flattens
             the managed universe to zero at the 16:00 auction.
  <=15:55    hard stop for MOC EXIT submission

The two legs together capture the strategy's forecasted 15:50 -> 16:00
close-auction move: the MKT batch enters intraday, the MOC exit lifts
at the closing print, and the book is flat overnight. The MOC size is
ALWAYS ``|current_qty|`` (measured immediately before the MOC send) so
a partial MKT fill can never over-flatten into an opposite overnight
exposure.

Everything broker-side happens through the ``ibkr_broker.Broker``
primitives (``buy`` / ``sell`` / ``get_positions`` /
``get_order_status`` / ``cancel``); this script never talks to TWS
directly.

Safety defaults:
- Paper mode unless ``--live`` AND ``--live-confirm`` are supplied.
- ``--dry-run`` computes the target book but places nothing.
- ``--on-feed-failure abort`` (the default) refuses to place orders if
  the live feed is stale / never connected.

Exit-code taxonomy (orchestrator only; see retrain.py and panic.py for
their own spaces):
  0   success (both legs sent; book will be flat after 16:00 close)
  2   feed abort or feature-build failure (no orders placed)
  3   solver failure (no orders placed)
  4   past MOC submit cutoff BEFORE any orders placed (safe skip)
  5   empty share_targets refusal (no orders placed)
  6   --replay-date + --live refused
  7   --replay-submit without --replay-date
  8   broker connect failed (no orders placed)
  9   --lambda- <= 0 (invalid argument, no orders placed)
  11  entries submitted but MOC exit could not fully cover the book:
      legs skipped past cutoff, get_positions failed, the snapshot came
      back stale-empty, the ground-truth cross-check found uncovered
      symbols (partial-snapshot / budget-exhausted-with-real-fill /
      TWS-brownout stale non-terminal), or the MOC acceptance audit
      found exchange rejections (MOC placed after ~15:50 cutoff).
      OVERNIGHT EXPOSURE POSSIBLE — page the operator.

The pager predicate is ``rc >= 11``: rc=11 is the ONLY code that
implies a live book at close. Everything else in [2, 10] is a safe
pre-order refusal or clean shutdown.

CONCURRENCY NOTE: do NOT run ``python -m live.panic`` against the
same TWS client-id while an orchestrator run is in flight — panic's
``cancel_all()`` will pull the orchestrator's own MKT entries and MOC
exits mid-flow, corrupting the coverage cross-check's ground-truth
poll. Panic uses client-id 99 by default; the orchestrator uses 1.
Distinct ids let them coexist against the same TWS but they should
never target the same client-id concurrently.

Usage:
    .venv/bin/python -m live.daily_orchestrator --paper
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import polars as pl

_UTC = timezone.utc

from live import config
from live.nasdaq_routing import NASDAQ_EXCHANGE
from live import trade_recorder
from live.features import (
    ALPHA_FEATURES,
    build_feature_frame,
    compute_avg_daily_dollar_volume,
    compute_close_trades,
    compute_daily_vol,
    compute_imbalance_print,
    compute_recent_return,
)
from live.live_feed import LiveFeed
from live.replay_feed import ReplayFeed
from live.optimizer import (
    CostModel,
    ForecastModel,
    ImpactModel,
    Optimizer,
    dollar_positions_to_shares,
)

logger = logging.getLogger("live.daily_orchestrator")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Daily live orchestrator: stream -> features -> alpha -> MVO "
            "-> MKT entries -> MOC exits."
        ),
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--paper", action="store_true",
        help="Paper-trading mode (default). Refuses to place live orders.",
    )
    mode.add_argument(
        "--live", action="store_true",
        help=(
            "Live-trading mode. Requires --live-confirm to be the exact "
            f"broker confirmation string ({{confirm_msg}})."
        ),
    )
    p.add_argument(
        "--live-confirm", default="",
        help="Broker confirmation string required with --live.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Compute alpha + solve MVO but skip broker connection & order submission.",
    )
    p.add_argument(
        "--lambda-", dest="lam", type=float, default=config.DEFAULT_LAMBDA,
        help="Risk aversion per dollar (see the notebook lambda). "
             f"Default: {config.DEFAULT_LAMBDA:g}.",
    )
    p.add_argument(
        "--impact-gamma", type=float, default=config.DEFAULT_IMPACT_GAMMA,
        help="Scalar impact gamma used if no per-name vector is loaded. "
             f"Default: {config.DEFAULT_IMPACT_GAMMA}.",
    )
    p.add_argument(
        "--use-impact-vector", action="store_true", default=True,
        help="Load ``impact/impact_gamma_vec.npy`` and multiply into the "
             "objective per name (see notebook follow-up).",
    )
    p.add_argument(
        "--no-impact-vector", dest="use_impact_vector", action="store_false",
        help="Disable per-name impact gamma; use scalar ``--impact-gamma``.",
    )
    p.add_argument(
        "--stream-start-time", default=config.DEFAULT_STREAM_START_TIME,
        help=f"Time (HH:MM ET) to open the live feed. Default: {config.DEFAULT_STREAM_START_TIME}.",
    )
    p.add_argument(
        "--rebalance-time", default=config.DEFAULT_REBALANCE_TIME,
        help=f"Time (HH:MM ET) to snapshot + solve. Default: {config.DEFAULT_REBALANCE_TIME}.",
    )
    p.add_argument(
        "--submit-by", default=config.DEFAULT_SUBMIT_BY_TIME,
        help=(
            "Hard cutoff (HH:MM ET) for MOC EXIT submission. "
            "Default: " + config.DEFAULT_SUBMIT_BY_TIME + ". "
            "Note: NYSE/Nasdaq MOC ENTRY cutoff is 15:50 ET, not 15:55 — "
            "any leg placed after 15:50 is expected to face MOC rejection "
            "on some exchanges. This flag ONLY bounds the MOC exit batch; "
            "the MKT entry batch fires unconditionally so the operator is "
            "never left with a filled entry and no queued exit. If the "
            "cutoff was already past at orchestrator entry (no orders "
            "placed), the process exits rc=4 (safe skip). If the cutoff "
            "trips mid-flow AFTER entries filled, the process exits rc=11 "
            "(OVERNIGHT EXPOSURE) and the target record lists the open "
            "positions."
        ),
    )
    p.add_argument(
        "--on-feed-failure", choices=("abort", "proceed"), default="abort",
        help="What to do if the live feed is stale / never connected. Default: abort.",
    )
    p.add_argument(
        "--universe-file", default=config.DEFAULT_ACTIVE_UNIVERSE_FILE,
        help=f"Active-universe filename in the repo root. Default: {config.DEFAULT_ACTIVE_UNIVERSE_FILE}.",
    )
    p.add_argument(
        "--broker-host", default="127.0.0.1",
        help="TWS host. Default: 127.0.0.1.",
    )
    p.add_argument(
        "--broker-port", type=int, default=None,
        help="TWS port. Default: auto (7497 paper, 7496 live).",
    )
    p.add_argument(
        "--broker-client-id", type=int, default=1,
        help="TWS client id. Use distinct ids per running process.",
    )
    p.add_argument(
        "--reference-stats-path", default=None,
        help=(
            "Override for the reference-stats parquet (per-symbol ADV/vol). "
            f"Default: impact/{config.REF_STATS_NAME}."
        ),
    )
    p.add_argument(
        "--no-live-feed", action="store_true",
        help=(
            "Skip opening the Databento live feed. Only useful with "
            "--dry-run for smoke tests — the alpha will be zero everywhere."
        ),
    )
    p.add_argument(
        "--vol-cap", type=float, default=config.DEFAULT_VOL_CAP,
        help=(
            "Target daily $-volatility cap. The QP is scale-invariant — "
            "vol_cap alone is the effective risk limit; the ``scale`` "
            "hyperparameter is a numerical conditioning factor auto-set "
            f"to 1 / lambda. Default: {config.DEFAULT_VOL_CAP:g}."
        ),
    )
    p.add_argument(
        "--moc-ack-timeout", type=float,
        default=config.DEFAULT_MOC_ACK_TIMEOUT_SECS,
        help=(
            "Bounded wall-clock (seconds) the MOC acceptance audit "
            "waits per-oid for a terminal state after the MOC batch "
            "is sent. An MOC that reaches Cancelled / ApiCancelled / "
            "Inactive with filled=0 within this window is treated as "
            "REJECTED at the exchange and escalates to rc=11. TimeoutError "
            "(still Submitted at deadline) is presumed accepted; MOCs sit "
            "Submitted until 16:00. Shorter = faster operator feedback "
            "but risks false rc=0 if a rejection propagates later than "
            "the window (rare; rejections typically terminal within "
            "seconds). Longer = more reliable rejection detection but "
            "adds up to that many seconds to every happy run's wall "
            f"clock. Default: {config.DEFAULT_MOC_ACK_TIMEOUT_SECS:g}."
        ),
    )
    p.add_argument(
        "--print-wait-secs", type=float, default=10.0,
        help=(
            "Seconds to wait after --rebalance-time before snapshotting the "
            "live buffers. The imbalance print publishes AT 15:50:00+ so a "
            "0s wait races it. Default: 10."
        ),
    )
    p.add_argument(
        "--min-universe-coverage", type=float, default=0.5,
        help=(
            "Minimum fraction of the universe that must appear in the live "
            "snapshot for us to proceed. Falls back to --on-feed-failure "
            "otherwise. Default: 0.5."
        ),
    )
    p.add_argument(
        "--replay-date", default=None,
        help=(
            "Dress-rehearsal mode. Instead of streaming from the Databento "
            "LIVE gateway, replay cached HISTORICAL imbalance + tbbo for "
            "the given YYYY-MM-DD (America/New_York) trading day. Skips the "
            "stream-start and snapshot wall-clock waits and bypasses the "
            "--submit-by cutoff (\"now\" is unrelated to the historical "
            "date). Refused with --live. First replay of a new date may "
            "trigger a one-off Databento historical materialisation of the "
            "parquet cache — cheap historical, not the paid live feed. "
            "By default replay is BROKER-FREE (see --replay-submit); it "
            "still validates features->alpha->MVO->sizing->prequalify. "
            "Caveat: with --replay-submit, MOC orders fill at TODAY's "
            "close, not the historical date's close, so PnL is not "
            "realistic — the point is order-plumbing validation."
        ),
    )
    p.add_argument(
        "--allow-fetch", action="store_true", default=False,
        help=(
            "Authorise the replay feed to submit a Databento Batch API "
            "job when the local cache cannot cover the requested day. "
            "Off by default — a cold-cache replay refuses rather than "
            "silently paying for a fresh historical materialisation. "
            "Only meaningful with --replay-date; ignored for the live "
            "path (websocket feed, no historical cache in the loop)."
        ),
    )
    p.add_argument(
        "--replay-submit", action="store_true", default=False,
        help=(
            "Opt in to placing REAL paper MOC orders during a replay run. "
            "Off by default because paper fills from a replay pollute the "
            "next real paper/live run's diff baseline (broker.get_positions "
            "will look like drift and the next day will trade to unwind). "
            "Only valid in replay mode (--replay-date). Always paper — "
            "combining with --live is refused (rc=6) regardless. If you "
            "enable this, flatten the paper account before the next real "
            "session."
        ),
    )
    return p.parse_args(argv)


# Argv keys that must never be logged in cleartext.
_SENSITIVE_ARG_KEYS: frozenset[str] = frozenset({"live_confirm"})


def _redact_args(cli_args: dict) -> dict:
    """Deep-copy args and mask sensitive keys before logging."""
    redacted = dict(cli_args)
    for k in _SENSITIVE_ARG_KEYS:
        if redacted.get(k):
            redacted[k] = "***REDACTED***"
    return redacted


def _redact_argv(argv: list[str]) -> list[str]:
    """Return argv with the value following any sensitive flag redacted."""
    sensitive_flags = {f"--{k.replace('_', '-')}" for k in _SENSITIVE_ARG_KEYS}
    out: list[str] = []
    skip_next = False
    for tok in argv:
        if skip_next:
            out.append("***REDACTED***")
            skip_next = False
            continue
        if tok in sensitive_flags:
            out.append(tok)
            skip_next = True
        elif any(tok.startswith(f"{flag}=") for flag in sensitive_flags):
            flag, _, _ = tok.partition("=")
            out.append(f"{flag}=***REDACTED***")
        else:
            out.append(tok)
    return out


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------

def _load_forecast_pipe() -> object:
    path = config.FORECAST_DIR / config.FORECAST_PIPE_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"Forecast pipeline missing at {path}. Run `python -m live.retrain`."
        )
    logger.info("loaded forecast pipeline", extra={"path": str(path)})
    return joblib.load(path)


def _load_cov_artifacts() -> dict:
    files = {
        "L": config.COV_DIR / config.COV_LOADINGS_NAME,
        "eps_diag": config.COV_DIR / config.COV_EPS_DIAG_NAME,
        "Sigma_f": config.COV_DIR / config.COV_FACTOR_COV_NAME,
        "Sigma_r": config.COV_DIR / config.COV_MATRIX_NAME,
    }
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Covariance artifacts missing: " + ", ".join(missing) +
            ". Run `python -m live.retrain --refit covariance`."
        )
    arts = {k: np.load(p) for k, p in files.items()}
    logger.info(
        "loaded covariance artifacts",
        extra={"loadings_shape": arts["L"].shape, "cov_shape": arts["Sigma_r"].shape},
    )
    return arts


def _load_impact_vector(use_vector: bool, fallback_scalar: float, N: int) -> np.ndarray | float:
    if not use_vector:
        return fallback_scalar
    path = config.IMPACT_DIR / config.IMPACT_GAMMA_NAME
    if not path.exists():
        logger.warning(
            "impact gamma vector missing; falling back to scalar",
            extra={"path": str(path), "scalar": fallback_scalar},
        )
        return fallback_scalar
    vec = np.load(path)
    if vec.shape != (N,):
        logger.warning(
            "impact gamma vector shape mismatch; falling back to scalar",
            extra={"got": vec.shape, "expected": (N,), "scalar": fallback_scalar},
        )
        return fallback_scalar
    logger.info("loaded per-name impact gamma", extra={"n": N, "median": float(np.median(vec))})
    return vec


def _load_reference_stats(path_override: Optional[str], universe: list[str]) -> pl.DataFrame:
    """Load the per-symbol reference stats (ADV, vol) written by retrain.

    Missing file is a soft failure — we synthesise a table of NaNs so the
    downstream ``merge_reference_stats`` join still runs. The impact model
    will then charge missing names the max coefficient (never-free-to-trade).

    F4 note (replay mode): the retrain-time snapshot was computed as of
    the retrain date, which is likely AFTER the replayed historical date.
    That is mild look-ahead in the impact/cost denominators. The plumbing
    is intentionally identical to the live path (so replay stays a
    dress-rehearsal for the real code path), but the resulting PnL is
    not realistic — order-plumbing validation, not backtest accuracy.
    """
    path = Path(path_override) if path_override else config.IMPACT_DIR / config.REF_STATS_NAME
    if not path.exists():
        logger.warning(
            "reference stats missing; using NaNs (impact model will fall back)",
            extra={"path": str(path)},
        )
        return pl.DataFrame({
            "symbol": universe,
            "ADV": [None] * len(universe),
            "vol": [None] * len(universe),
        })
    stats = pl.read_parquet(path)
    logger.info(
        "loaded reference stats",
        extra={"path": str(path), "n_symbols": stats.height},
    )
    return stats


# ---------------------------------------------------------------------------
# Feature engineering (live)
# ---------------------------------------------------------------------------

def _build_live_day_frame(
    feed: LiveFeed,
    universe: list[str],
    reference_stats: pl.DataFrame,
    min_coverage: float = 0.5,
) -> tuple[pl.DataFrame, dict[str, float]]:
    """Snapshot the live feed and produce the single-day feature frame.

    ``min_coverage`` is the fraction of the universe that must appear in
    the alpha-ready ``day_df`` before we return. Anything less and we raise
    ``RuntimeError`` — silent-empty is unacceptable in the live path.

    Returns
    -------
    (day_df, last_mid_by_sym)
        ``day_df`` has the columns the optimizer's ForecastModel / CostModel /
        ImpactModel need. ``last_mid_by_sym`` is the freshest mid we saw for
        each symbol in the snapshot — used to convert dollar targets to
        shares (falling back to ``entry_price`` when needed).
    """
    frames = feed.snapshot()
    imb = frames.get("imbalance", pl.DataFrame())
    tbbo = frames.get("tbbo", pl.DataFrame())
    logger.info(
        "snapshotted live feed",
        extra={
            "n_imbalance_records": int(imb.height if not imb.is_empty() else 0),
            "n_tbbo_records": int(tbbo.height if not tbbo.is_empty() else 0),
            "feed_stats": feed.stats(),
        },
    )

    if imb.is_empty() or tbbo.is_empty():
        raise RuntimeError(
            f"Live snapshot too thin (imb={imb.height}, tbbo={tbbo.height}); "
            "cannot compute features."
        )

    # Restrict tbbo to the active set — the feed may deliver mappings for
    # names we don't trade if the universe file drifted.
    universe_set = set(universe)
    tbbo = tbbo.filter(pl.col("symbol").is_in(list(universe_set)))
    imb = imb.filter(pl.col("symbol").is_in(list(universe_set)))

    tbbo_lf = tbbo.lazy()
    imb_lf = imb.lazy()

    close_trades = compute_close_trades(tbbo_lf).collect()
    recent_return = compute_recent_return(tbbo_lf).collect()
    avg_daily_dollar_volume = compute_avg_daily_dollar_volume(tbbo_lf).collect()
    daily_vol = compute_daily_vol(tbbo_lf).collect()
    imb_print = compute_imbalance_print(imb_lf).collect()

    # Live has one day of tbbo, so ADV/vol are causal-null. Reference stats
    # are threaded INTO build_feature_frame so they backfill BEFORE the
    # ratio features (paired_dollars/ADV, sigma/sqrt(ADV)) are computed —
    # otherwise the ratios divide by null and derive_features drops every
    # row. Ordering matters here.
    day_df = build_feature_frame(
        imb_print=imb_print,
        close_trades=close_trades,
        avg_daily_dollar_volume=avg_daily_dollar_volume,
        daily_vol=daily_vol,
        recent_return=recent_return,
        auction=None,  # no auction print at 15:50 — that's what we're predicting
        reference_stats=reference_stats,
    )

    # Last mid + last-observed ts per symbol. The mid is used for share
    # sizing (fresher than ``entry_price`` if we kept streaming after
    # the 15:50 print). The ts is threaded into the trade recorder so
    # ``arrival_ts_utc`` / ``arrival_staleness_secs`` can flag thin
    # names whose last print was minutes ago — those rows would
    # otherwise pool with clean rows under an analyst's naive
    # decision-time slippage average.
    last_mid = (
        tbbo
        .sort("ts_event")
        .with_columns(((pl.col("ask_px_00") + pl.col("bid_px_00")) / 2).alias("mid"))
        .group_by("symbol", maintain_order=False)
        .agg(
            pl.col("mid").last().alias("last_mid"),
            pl.col("ts_event").last().alias("last_mid_ts"),
        )
    )
    last_mid_by_sym = dict(zip(last_mid["symbol"].to_list(), last_mid["last_mid"].to_list()))
    last_mid_ts_by_sym = dict(zip(last_mid["symbol"].to_list(), last_mid["last_mid_ts"].to_list()))

    # C2 hard coverage assertion: refuse to hand back a thin day_df. A live
    # snapshot at 15:50 should cover most of the top-100 — anything less
    # than ``min_coverage`` almost certainly means the feed or feature
    # pipeline is broken, and quietly proceeding would send zero-alpha
    # orders (which with close_unlisted=True equals "flatten everything").
    coverage = day_df.height / max(len(universe), 1)
    if coverage < min_coverage:
        raise RuntimeError(
            f"live universe coverage {coverage:.2%} below --min-universe-coverage "
            f"{min_coverage:.2%} (day_df.height={day_df.height}, universe={len(universe)})"
        )
    logger.info(
        "day_df built",
        extra={
            "day_df_rows": day_df.height,
            "universe_size": len(universe),
            "coverage": coverage,
        },
    )
    return day_df, last_mid_by_sym, last_mid_ts_by_sym


def _compute_share_targets(
    dollar_positions: np.ndarray,
    universe: list[str],
    day_df: pl.DataFrame,
    last_mid_by_sym: dict[str, float],
) -> dict[str, int]:
    """Prefer last-observed mid; fall back to entry_price. Names with no
    price at all are dropped (broker cannot size them).
    """
    if day_df.is_empty() or "symbol" not in day_df.columns:
        entry_map: dict[str, float] = {}
    else:
        entry_map = dict(zip(day_df["symbol"].to_list(), day_df["entry_price"].to_list()))
    prices = np.array(
        [
            last_mid_by_sym.get(sym) or entry_map.get(sym) or float("nan")
            for sym in universe
        ],
        dtype=float,
    )
    return dollar_positions_to_shares(dollar_positions, prices, universe)


# ---------------------------------------------------------------------------
# Broker interaction
# ---------------------------------------------------------------------------

def _connect_broker(args: argparse.Namespace):
    from ibkr_broker.broker import Broker
    broker = Broker()
    kwargs = {
        "host": args.broker_host,
        "client_id": args.broker_client_id,
    }
    if args.broker_port is not None:
        kwargs["port"] = args.broker_port
    if args.live:
        if args.live_confirm != Broker.LIVE_CONFIRM_MSG:
            raise RuntimeError(
                "live mode requires --live-confirm='"
                f"{Broker.LIVE_CONFIRM_MSG}' (found {args.live_confirm!r})"
            )
        broker.connect(mode="live", live_confirm=args.live_confirm, **kwargs)
    else:
        broker.connect(mode="paper", **kwargs)
    return broker


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _stop_feed_safe(feed) -> None:
    """Best-effort idempotent ``feed.stop()`` — every early-return
    path in ``main()`` calls this so the feed's daemon thread doesn't
    outlive the process on rc=3/4/5/8. ``LiveFeed.stop`` is already
    idempotent internally; wrap it in a try/except so an exception
    during shutdown doesn't propagate up over the return code.
    """
    if feed is None:
        return
    try:
        feed.stop()
    except Exception:
        logger.exception("feed stop failed (non-fatal)")


def _sleep_until(target: datetime, poll: float = 5.0) -> None:
    """Block until ``target`` (America/New_York timezone).

    Uses coarse polling so signals and Ctrl-C wake the process quickly.
    """
    while True:
        now = config.now_ny()
        remaining = (target - now).total_seconds()
        if remaining <= 0:
            return
        time.sleep(min(remaining, poll))



# ---------------------------------------------------------------------------
# Target-book persistence & broker pre-flight helpers
# ---------------------------------------------------------------------------

_TARGETS_DIR = config.LIVE_LOG_DIR / "targets"
_FILLS_DIR = config.LIVE_LOG_DIR / "fills"
# Per-trade parquet output root. Kept as a module-level so tests can
# monkey-patch to an isolated ``tmp_path`` (same idiom as the two dirs
# above).
_TRADES_DIR = config.LIVE_LOG_DIR / "trades"

# Per-process scratch dir used only when the safeguard in
# ``_resolve_trades_dir`` decides the current run is not producing
# real fills (fake broker, notebook, test harness). Prevents
# fabricated rows from silently appending to ``live/logs/trades/``.
_PYTEST_TRADES_SCRATCH: Optional[Path] = None


def _cleanup_pytest_trades_scratch() -> None:
    """Best-effort atexit cleanup of ``_PYTEST_TRADES_SCRATCH``.

    Gated on ``PYTEST_CURRENT_TEST`` so a PRODUCTION misdetection of
    the broker type (see ``_resolve_trades_dir``) leaves the recorded
    fills on disk for the operator to recover from
    ``/tmp/live-trade-recorder-scratch-*``. Under pytest we still
    rmtree so the scratch dir doesn't accumulate.

    Deliberately silent — this runs at interpreter shutdown, when
    the logging handlers' file streams may already be closed. The
    WARNING that operators would need to see in the production
    misdetection case was already emitted at redirect time in
    ``_resolve_trades_dir``; this function only decides whether to
    delete.
    """
    global _PYTEST_TRADES_SCRATCH
    if _PYTEST_TRADES_SCRATCH is None:
        return
    if not os.environ.get("PYTEST_CURRENT_TEST"):
        # Not a test run — the redirect fired due to a broker-type
        # misdetection in production. DO NOT DELETE. The parquet in
        # scratch may be the ONLY record of the day's trades.
        return
    try:
        import shutil as _shutil
        _shutil.rmtree(_PYTEST_TRADES_SCRATCH, ignore_errors=True)
    except Exception:
        pass


def _is_real_ibkr_broker(broker: object) -> bool:
    """True iff ``broker`` is a live ``ibkr_broker.Broker`` instance.

    Fake test-broker classes (``_FakeBroker`` in the test files, or
    any user-side stub) do NOT satisfy this — the recorder safeguard
    uses this to redirect their output to a scratch dir. Any failure
    to import ``ibkr_broker`` is treated as "not real" (safer than
    silently allowing fabricated rows through).

    Known limitations we deliberately accept:

    - ``MagicMock(spec=Broker)`` fakes ``__class__`` and will pass
      ``isinstance``. Not a realistic production risk (the autouse
      test fixture patches ``_TRADES_DIR`` before this ever runs
      against a mocked broker) but worth naming.
    - A duplicate ``ibkr_broker`` module identity (editable install
      re-imported through a different ``sys.path`` prefix, an
      ``importlib.reload``, or a future subclass) can make the
      isinstance check miss. This is why ``_cleanup_pytest_trades_scratch``
      gates its rmtree on ``PYTEST_CURRENT_TEST`` — a false negative
      here trades to scratch but the file stays on disk for recovery.
    """
    if broker is None:
        return False
    try:
        from ibkr_broker.broker import Broker as _RealBroker
    except Exception:
        return False
    return isinstance(broker, _RealBroker)


def _resolve_trades_dir(broker: object = None) -> Path:
    """Return the effective trade-recorder output root.

    - Production: honours ``_TRADES_DIR`` (defaults to
      ``live/logs/trades``) IF the run is producing real fills, i.e.
      ``broker`` is an actual ``ibkr_broker.Broker`` instance connected
      to TWS.
    - Otherwise (fake broker in a notebook, test harness, or a broker
      class whose ``isinstance`` check misses for any reason):
      redirect to a per-process temp directory. In PRODUCTION this
      also emits a WARNING naming the broker type and scratch path so
      an operator can find the recorded fills if the misdetection
      was legitimate.
    - Tests that WANT to inspect recorder output should still
      monkeypatch ``_TRADES_DIR`` to their own ``tmp_path``; the
      scratch redirect is a defence-in-depth safeguard, not a feature.

    Explicit ``_TRADES_DIR`` overrides (e.g. tests' monkeypatch) are
    always honoured — the caller has clearly said "put it here".
    """
    global _PYTEST_TRADES_SCRATCH
    default = config.LIVE_LOG_DIR / "trades"
    if _TRADES_DIR != default:
        # Caller explicitly picked a location. Trust them.
        return _TRADES_DIR
    if _is_real_ibkr_broker(broker):
        return _TRADES_DIR
    # Redirect. Anything that ends up here is EITHER fake fills (test
    # / notebook, expected) OR a real broker whose type check missed
    # (production, must not be silent).
    if _PYTEST_TRADES_SCRATCH is None:
        _PYTEST_TRADES_SCRATCH = Path(
            tempfile.mkdtemp(prefix="live-trade-recorder-scratch-")
        )
        import atexit as _atexit
        _atexit.register(_cleanup_pytest_trades_scratch)
    # A WARNING (not INFO) so a production misdetection is impossible
    # to miss in the logs. The broker type name is included so the
    # operator can see WHY the check failed (e.g. duplicate module
    # identity via an editable install / sys.path variant).
    logger.warning(
        "trade recorder: redirecting output to scratch dir %s "
        "(broker type: %s). If this fired during a real paper/live "
        "run, the isinstance check missed — check the scratch dir "
        "for the recorded parquet; it will NOT be deleted at exit "
        "unless PYTEST_CURRENT_TEST is set.",
        str(_PYTEST_TRADES_SCRATCH),
        type(broker).__name__ if broker is not None else "None",
    )
    return _PYTEST_TRADES_SCRATCH


def _hash_universe(universe: list[str]) -> str:
    """Short stable hash — lets the reconciler tell whether yesterday's
    saved book still applies (or if the universe file changed under us).
    """
    return hashlib.sha1("\n".join(universe).encode("utf-8")).hexdigest()[:12]


def _write_target_record(record: dict) -> Path:
    """Write today's intended book atomically to
    ``live/logs/targets/YYYY-MM-DD.json`` (paper/live), or to
    ``live/logs/targets/replay/YYYY-MM-DD.json`` when the record's
    ``mode`` is ``"replay"``.

    The replay subdirectory keeps rehearsals out of the reconciliation
    path — ``_load_previous_target_record`` only scans the top level, so
    it never mistakes a replayed book for a real one.
    """
    is_replay = record.get("mode") == "replay"
    root = _TARGETS_DIR / "replay" if is_replay else _TARGETS_DIR
    root.mkdir(parents=True, exist_ok=True)
    # For replay, filename is the REPLAY date (so each historical day gets
    # its own record); for live/paper, filename is TODAY.
    stem = record.get("replay_date") if is_replay else config.now_ny().date().isoformat()
    final = root / f"{stem}.json"
    tmp = root / f".{stem}.json.tmp"
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True, default=str))
    tmp.replace(final)
    logger.info("target book persisted", extra={"path": str(final), "mode": record.get("mode")})
    return final


def _load_previous_target_record() -> Optional[dict]:
    """Return the most recent target-record on disk STRICTLY before today,
    if any.
    """
    if not _TARGETS_DIR.exists():
        return None
    today = config.now_ny().date().isoformat()
    files = sorted(
        (f for f in _TARGETS_DIR.glob("*.json") if f.stem < today),
        reverse=True,
    )
    if not files:
        return None
    try:
        return json.loads(files[0].read_text())
    except Exception as e:
        logger.warning(
            "previous target record unreadable; skipping reconciliation",
            extra={"path": str(files[0]), "error": str(e)},
        )
        return None


def _reconcile_from_previous(broker, prev: dict) -> None:
    """Compare yesterday's target book to today's actual positions and log
    any drift. Delegates to ``broker.reconcile_positions`` (broker.py:900).
    """
    targets = prev.get("targets", {})
    if not targets:
        return
    try:
        drift = broker.reconcile_positions({k: int(v) for k, v in targets.items()})
    except Exception as e:
        logger.warning("reconcile_positions failed", extra={"error": str(e)})
        return
    if drift:
        logger.warning(
            "position drift vs yesterday's target book",
            extra={"n_drift_names": len(drift), "generated_at_ny": prev.get("generated_at_ny")},
        )
    else:
        logger.info(
            "no drift vs yesterday's target book",
            extra={"generated_at_ny": prev.get("generated_at_ny")},
        )


def _prequalify_symbols(broker, symbols: list[str]) -> set[str]:
    """Qualify every symbol through IB before we start firing orders.

    Any symbol IB refuses (delisted, wrong ticker, no market data
    subscription) is dropped from the returned set — the orchestrator omits
    it from the extended target so ``rebalance`` won't try to trade it.

    Uses the broker's own ``_make_contract`` helper so we go through the
    same code path ``_place`` uses at trade time — and we pass the same
    ``NASDAQ_EXCHANGE`` code every entry/exit uses so qualification
    surfaces a NASDAQ-untradeable name here (drop) instead of at order
    submission (a lost session slot). See ``live/nasdaq_routing.py``
    for why NASDAQ-only routing is a hard requirement of this strategy.
    We swallow individual exceptions and log them; one bad name should
    not kill the batch.

    NASDAQ-listed eligibility filter (DISABLED 2026-07-20)
    ------------------------------------------------------

    A ``primaryExchange == "NASDAQ"`` filter previously lived here
    (see the commented-out block below) — the intent was to drop
    NYSE/ARCA-primary names since their MOC prints on the primary
    listing's close-cross, not NASDAQ's. It was rolled back
    together with the broker's route-survival assertion: during a
    2026-07-20 paper replay the broker guard rejected 46 of 46
    requested symbols (100% failure, 0 passed) upstream of this
    filter, so this drop's behaviour was never exercised end-to-end
    on that run. The filter is also currently a no-op on both
    tiered universes: 49/49 in top_49_universe.txt, and 99 of the
    100 in top_100_universe.txt qualify with all 99 NASDAQ-primary
    (ITCI fails IB qualification outright and is dropped by the
    still-active try/except above, never reaching this filter), so
    on the active universe it never dropped a name. Its real value is future
    universe-drift protection — ``universe.txt`` (the broad ~2000
    name pool retrain scans) does include NYSE/ARCA-primary names.
    """
    ok: set[str] = set()
    for sym in symbols:
        try:
            # The qualified contract is only inspected by the
            # commented-out primaryExchange filter below, so we
            # discard the return value. Keep the call itself — it
            # is still how we detect a name IB cannot qualify (bad
            # ticker, no market-data subscription, etc.).
            broker._make_contract(sym, exchange=NASDAQ_EXCHANGE)
        except Exception as e:
            logger.warning(
                "contract qualification failed",
                extra={"symbol": sym, "error": str(e)[:200]},
            )
            continue
        # --- DISABLED 2026-07-20 (see docstring above) --------------------
        # NASDAQ-listed eligibility: drop names whose primary listing
        # is not NASDAQ, because their MOC prints on their home
        # exchange's close-cross, not NASDAQ's. We include the
        # qualified contract's primaryExchange in the log so the
        # operator can see WHY the name was dropped.
        #
        # Rolled back alongside the broker-side route-survival
        # assertion so we can re-run a replay and observe what
        # actually reaches the MOC leg. On both current tiered
        # universes this filter is a no-op: 49/49 in top_49, and 99
        # of the 100 in top_100 qualify with all 99 NASDAQ-primary
        # (ITCI fails IB qualification outright and is dropped by
        # the try/except above, never reaching here), so restoring
        # it only matters as
        # future universe-drift protection — ``universe.txt`` (the
        # broad ~2000-name pool ``retrain`` scans) contains
        # NYSE/ARCA-primary names. To restore, capture the
        # ``_make_contract`` return above into a ``contract``
        # variable again.
        # primary = getattr(contract, "primaryExchange", "") or ""
        # if primary != "NASDAQ":
        #     logger.warning(
        #         "dropping non-NASDAQ-listed symbol from target: MOC "
        #         "at close prints on the primary listing's close-cross, "
        #         "not NASDAQ's — an ISLAND MOC on this name would "
        #         "mis-print or be rejected",
        #         extra={
        #             "symbol": sym,
        #             "primary_exchange": primary,
        #             "expected_primary_exchange": "NASDAQ",
        #         },
        #     )
        #     continue
        # ------------------------------------------------------------------
        ok.add(sym)
    logger.info(
        "contracts pre-qualified",
        extra={"n_ok": len(ok), "n_requested": len(symbols)},
    )
    return ok


def _max_leg_size(extended_targets: dict[str, int], current: dict[str, int]) -> int:
    """Largest absolute per-leg diff |target - current|. Zero if empty."""
    if not extended_targets:
        return 0
    return max(
        (abs(extended_targets[s] - current.get(s, 0)) for s in extended_targets),
        default=0,
    )


def _submit_mkt_entries(
    broker,
    targets: dict[str, int],
    current: dict[str, int],
    managed_universe: set[str],
) -> list[dict]:
    """ENTRY batch: per-name plain MKT sized to ``target - current``.

    Routed via ``broker.buy`` / ``broker.sell`` with ``order_type="MKT"``
    and the primitive default ``tif="DAY"`` — a stuck MKT that never
    fills is bounded by the belt-and-braces cancel loop in main() (any
    entry whose status has not reached a terminal state by the end of
    ``_wait_entry_fills`` gets an explicit ``broker.cancel``), which
    runs before the MOC snapshot so a late fill can't leave an
    unhedged position after the counter-flatten is sized.

    Fires unconditionally with respect to the MOC submit-by cutoff — an
    entered position with no exit is bad, but so is refusing to enter
    because we're worried about the exit; the operator needs to know
    either way (the MOC-skip case is handled downstream by
    ``_submit_moc_flatten``).

    Out-of-universe symbols (``sym in current but not managed_universe``)
    are pinned to their current qty upstream, so they diff to zero here
    and we never touch them.

    Returns a list of dicts, one per attempted entry, with keys:
      symbol, side, requested_qty, order_id (str|None), error (str|None).
    Downstream, ``_wait_entry_fills`` populates ``filled_qty`` and
    ``status`` from ``broker.wait_for_fill``.
    """
    entries: list[dict] = []
    for sym, tgt in sorted(targets.items()):
        diff = tgt - current.get(sym, 0)
        if diff == 0:
            continue
        if sym not in managed_universe:
            # Belt-and-braces: the caller has already pinned out-of-universe
            # names to their current qty (diff==0), but if any leak through
            # we still refuse to touch them.
            continue
        side = "BUY" if diff > 0 else "SELL"
        qty = int(abs(diff))
        try:
            # NASDAQ-only routing (see live/nasdaq_routing.py): the
            # strategy is calibrated to the NASDAQ closing auction —
            # SMART-routed off-venue fills are uncorrelated with the
            # signal we sized, so every entry MKT is forced to the
            # NASDAQ INET book.
            if side == "BUY":
                oid = broker.buy(sym, qty, order_type="MKT",
                                 exchange=NASDAQ_EXCHANGE)
            else:
                oid = broker.sell(sym, qty, order_type="MKT",
                                  exchange=NASDAQ_EXCHANGE)
        except Exception as e:
            logger.warning(
                "MKT entry submission failed",
                extra={"symbol": sym, "side": side, "qty": qty,
                       "error": str(e)[:200]},
            )
            entries.append({
                "symbol": sym, "side": side, "requested_qty": qty,
                "order_id": None, "error": str(e)[:200],
            })
            continue
        logger.info(
            "MKT entry placed",
            extra={"symbol": sym, "side": side, "qty": qty, "order_id": oid},
        )
        entries.append({
            "symbol": sym, "side": side, "requested_qty": qty,
            "order_id": oid, "error": None,
        })
    return entries


# Order statuses considered "terminal" for the entry-fill wait. Matches
# ``ib_async.order.OrderStatus.DoneStates`` semantically (the broker uses
# that set inside ``wait_for_fill``); we duplicate the check here so main
# can decide which entries still need an explicit cancel.
_TERMINAL_ORDER_STATES: frozenset[str] = frozenset({
    "Filled", "Cancelled", "ApiCancelled", "Inactive",
})


def _cancel_non_terminal_entries(broker, entries: list[dict]) -> list[str]:
    """Cancel every entry whose status is not in a terminal state.

    Plain MKT orders in a liquid name at 15:50 should be terminal within
    milliseconds, so under normal conditions this loop is a no-op. It
    exists to bound the "MKT placed but never acked / never filled" edge
    — a broker/TWS blip or an illiquid tape that leaves an order in a
    limbo status. Cancelling here bounds real exposure to what was
    actually filled BEFORE the MOC snapshot reads positions and sizes
    the counter-flatten.

    A panic --cancel-only run mid-flow interacts cleanly with this: any
    order still working at panic time gets pulled by ``cancel_all``;
    already-filled entries stay filled (fine — the MOC snapshot picks
    them up); the MOC batch either hasn't been sent yet (nothing to
    cancel) or is still pre-cutoff (panic pulls them via cancel_all).

    Returns the list of order_ids we sent cancels for.
    """
    cancelled: list[str] = []
    for e in entries:
        oid = e.get("order_id")
        status = e.get("status")
        if oid is None:
            continue
        if status in _TERMINAL_ORDER_STATES:
            continue
        try:
            broker.cancel(oid)
            cancelled.append(oid)
            logger.info(
                "belt-and-braces cancel of non-terminal entry",
                extra={"order_id": oid, "symbol": e.get("symbol"),
                       "status": status},
            )
            # Mark as ApiCancelled so downstream reasoning knows we
            # explicitly stopped it (distinct from a fill).
            e["status"] = "ApiCancelled"
        except Exception as ex:
            logger.warning(
                "belt-and-braces cancel failed — entry may still be working",
                extra={"order_id": oid, "symbol": e.get("symbol"),
                       "error": str(ex)[:200]},
            )
    return cancelled


def _wait_entry_fills(
    broker,
    entries: list[dict],
    submit_by: datetime,
    per_order_timeout: float = config.DEFAULT_ENTRY_FILL_TIMEOUT_SECS,
    safety_buffer_secs: float = 30.0,
) -> list[dict]:
    """Block until each submitted entry order is terminal, bounded by a
    GLOBAL deadline tied to ``submit_by``.

    Rationale (R5): a naive ``N * per_order_timeout`` loop can burn 8+
    minutes on N=100 slow-TWS days and push the MOC snapshot past 15:55.
    Better to stop waiting early — the MOC exit's position snapshot is
    the source of truth for what to flatten, so an incomplete wait is
    HARMLESS but a blown MOC cutoff is not.

    The effective deadline is ``min(now + N*per_order_timeout,
    submit_by - safety_buffer_secs)``. Anything past that stops waiting
    and marks remaining orders ``status="wait_budget_exhausted"``. Those
    orders will be caught by the belt-and-braces cancel loop in main.

    Mutates each dict in ``entries`` in place with ``filled_qty`` and
    ``status``; returns the same list for convenience.
    """
    now = config.now_ny()
    n = max(len(entries), 1)
    naive_deadline = now + timedelta(seconds=n * per_order_timeout)
    cutoff_deadline = submit_by - timedelta(seconds=safety_buffer_secs)
    deadline = min(naive_deadline, cutoff_deadline)
    if deadline <= now:
        # The safety buffer already fires — don't wait on anything;
        # rely on the belt-and-braces cancel loop.
        logger.warning(
            "entry-fill wait deadline already past; skipping all waits",
            extra={"now": now.isoformat(), "deadline": deadline.isoformat(),
                   "submit_by": submit_by.isoformat()},
        )
        for e in entries:
            if e.get("order_id") is None:
                e["filled_qty"] = 0
                e["status"] = "not_submitted"
                continue
            e["filled_qty"] = 0
            e["status"] = "wait_budget_exhausted"
        return entries

    for e in entries:
        oid = e.get("order_id")
        if oid is None:
            e["filled_qty"] = 0
            e["status"] = "not_submitted"
            continue
        remaining = (deadline - config.now_ny()).total_seconds()
        if remaining <= 0:
            logger.warning(
                "entry-fill wait budget exhausted; leaving order for cancel loop",
                extra={"order_id": oid, "symbol": e.get("symbol"),
                       "deadline": deadline.isoformat()},
            )
            e["filled_qty"] = 0
            e["status"] = "wait_budget_exhausted"
            continue
        this_wait = min(per_order_timeout, remaining)
        try:
            info = broker.wait_for_fill(oid, timeout=this_wait)
            e["filled_qty"] = int(getattr(info, "filled", 0) or 0)
            e["status"] = getattr(info, "status", "")
        except TimeoutError:
            logger.warning(
                "MKT entry fill wait timed out",
                extra={"order_id": oid, "symbol": e.get("symbol"),
                       "timeout": this_wait},
            )
            e["filled_qty"] = 0
            e["status"] = "timeout"
        except Exception as ex:
            logger.warning(
                "MKT entry status poll failed",
                extra={"order_id": oid, "symbol": e.get("symbol"),
                       "error": str(ex)[:200]},
            )
            e["filled_qty"] = 0
            e["status"] = "poll_failed"
    return entries


def _submit_moc_flatten(
    broker,
    managed_universe: set[str],
    submit_by: datetime,
) -> tuple[list[dict], list[str], bool]:
    """EXIT batch: MOC counter-orders sized to the ACTUAL current position.

    Snapshots ``broker.get_positions()`` fresh, aggregates by symbol
    (R4: IB can duplicate rows for the same symbol on rare occasions —
    multi-account/advisor books, or a position update racing the
    snapshot — and a naive loop would fire TWO MOC orders and
    over-flatten), restricts to ``managed_universe``, and sends a
    per-symbol MOC in the opposite direction sized to ``abs(net_qty)``.
    Out-of-universe positions are ignored (H1 invariant).

    Safety invariant (structural, not runtime): this function's signature
    does not accept any target-side data (``share_targets``, ``entries``,
    ``target - current``) — the only path to ``qty`` is via the freshly
    snapshotted position book. A partial MKT fill therefore cannot cause
    an over-flatten; a future refactor that adds a target argument would
    need to be rejected at review, since there is no runtime check that
    can catch such a change from inside this function.

    Returns ``(moc_batch, skipped_symbols, positions_unknown)``. Each
    ``moc_batch`` element is ``{"oid": str, "symbol": str, "side": str,
    "qty": int, "signed_qty": int}`` — the caller's coverage
    cross-check uses ``signed_qty`` to verify
    ``expected_held + moc_signed == 0`` per name, and the acceptance
    audit uses ``oid`` + ``symbol`` to report rejected MOCs by name.
    ``positions_unknown=True`` means the snapshot itself failed and
    the caller must treat every filled entry as at-risk.
    """
    try:
        positions = broker.get_positions()
    except Exception as e:
        # R1: on a TWS drop between entry fills and this snapshot, silent
        # empty returns would let main see (order_ids=[], skipped=[]) and
        # exit rc=0 — with the ENTIRE MKT-filled book riding the auction
        # unhedged. Signal the unknown state distinctly.
        logger.error(
            "MOC exit: get_positions FAILED — POSITIONS MAY BE LIVE INTO "
            "THE CLOSE, OPERATOR MUST MANUALLY FLATTEN",
            extra={"error": str(e)[:200]},
        )
        return [], [], True

    # R4: dedupe by symbol. Sum quantities in case IB returned the same
    # symbol twice (an artifact of multi-account or update races). Filter
    # to managed universe and non-zero net quantities.
    agg: dict[str, int] = {}
    for p in positions:
        if p.symbol not in managed_universe:
            continue
        agg[p.symbol] = agg.get(p.symbol, 0) + int(p.quantity)
    flatten_targets = [(sym, qty) for sym, qty in agg.items() if qty != 0]

    moc_batch: list[dict] = []
    skipped: list[str] = []
    for sym, current_qty in sorted(flatten_targets):
        if config.now_ny() > submit_by:
            skipped.append(sym)
            continue
        qty = abs(current_qty)
        side = "SELL" if current_qty > 0 else "BUY"
        # R6: honest sanity checks. The structural invariant (MOC size
        # cannot be tainted by target data) is enforced by this
        # function's signature, not by any runtime assert — see docstring.
        assert qty > 0, f"MOC flatten: expected positive qty, got {qty}"
        assert side in ("BUY", "SELL"), f"MOC flatten: bad side {side!r}"
        try:
            # NASDAQ-only routing (see live/nasdaq_routing.py): the
            # MOC exit MUST print at the NASDAQ close auction — that
            # is literally the auction we forecast. Route direct to
            # the NASDAQ INET book instead of letting SMART pick a
            # different venue's close-cross.
            if side == "BUY":
                oid = broker.buy(sym, qty, order_type="MOC",
                                 exchange=NASDAQ_EXCHANGE)
            else:
                oid = broker.sell(sym, qty, order_type="MOC",
                                  exchange=NASDAQ_EXCHANGE)
        except Exception as e:
            logger.warning(
                "MOC exit submission failed",
                extra={"symbol": sym, "side": side, "qty": qty,
                       "error": str(e)[:200]},
            )
            continue
        # Signed counter-qty: BUY MOC covers a short (negative current_qty)
        # by adding shares (+qty); SELL MOC covers a long (positive
        # current_qty) by removing shares (-qty).
        signed_qty = qty if side == "BUY" else -qty
        moc_batch.append({
            "oid": oid, "symbol": sym, "side": side,
            "qty": qty, "signed_qty": signed_qty,
        })
        logger.info(
            "MOC exit placed",
            extra={"symbol": sym, "side": side, "qty": qty,
                   "current_qty": current_qty, "order_id": oid},
        )

    if skipped:
        logger.error(
            "MOC exit: legs skipped past submit cutoff — POSITIONS LEFT OPEN",
            extra={"n_skipped": len(skipped), "symbols": skipped[:20]},
        )

    return moc_batch, skipped, False


def _audit_moc_acceptance(
    broker,
    moc_batch: list[dict],
    wall_clock_wait: float = config.DEFAULT_MOC_ACK_TIMEOUT_SECS,
) -> set[str]:
    """RRR1: wait up to ``wall_clock_wait`` seconds for MOC EXCHANGE
    ACCEPTANCE and return the set of symbols whose MOC was REJECTED.

    ``broker._place`` (via ``buy``/``sell``) returns synchronously the
    moment TWS accepts the API-level placeOrder call. That is BEFORE
    any exchange ack. An MOC submitted past the ~15:50 exchange entry
    cutoff (e.g. a 15:50:20+ MOC send on a slow day) is API-accepted,
    an oid is returned, but the exchange REJECTS it — the order flips
    to ``Cancelled`` / ``Inactive`` with ``filled=0`` within seconds
    to a minute. Without this audit, the coverage cross-check scores
    the symbol as covered (an oid came back) and rc=0 despite an open
    overnight position.

    We wait per-oid for a terminal state via ``broker.wait_for_fill``:
    - Terminal (Cancelled / ApiCancelled / Inactive) with ``filled=0``
      → REJECTED → add ``symbol`` to the unfilled set.
    - Terminal with ``filled>0`` → early Filled → fine.
    - TimeoutError → still ``Submitted`` at deadline → presumed
      accepted at exchange (MOCs sit ``Submitted`` until 16:00).
    - Exception → pessimistic: add to unfilled set (safer than
      silently assuming acceptance).

    Wall-clock budget is bounded so a run doesn't sit ~8 min waiting
    until 16:00 for the real fill. Residual risk: an MOC that was
    accepted at ack then Cancelled after the audit window is not
    caught here (rare; would need TWS-side pathology or an operator
    ``panic --cancel-only`` running against the same client-id).

    Returns a set of REJECTED symbols. Empty = every MOC was
    accepted (or actually filled early).
    """
    rejected: set[str] = set()
    if not moc_batch:
        return rejected

    per_order_slice = max(1.0, wall_clock_wait / max(len(moc_batch), 1))
    deadline = time.time() + wall_clock_wait

    for leg in moc_batch:
        oid = leg["oid"]
        sym = leg["symbol"]
        remaining = max(0.0, deadline - time.time())
        this_wait = min(per_order_slice, remaining)
        if this_wait <= 0:
            # Budget exhausted — one final non-blocking status poll.
            try:
                info = broker.get_order_status(oid)
            except Exception as e:
                logger.warning(
                    "MOC acceptance audit: status poll failed (treating "
                    "as rejected — pessimistic)",
                    extra={"oid": oid, "symbol": sym, "error": str(e)[:200]},
                )
                rejected.add(sym)
                continue
            status = getattr(info, "status", "")
            filled = int(getattr(info, "filled", 0) or 0)
            if status in _TERMINAL_ORDER_STATES and filled == 0:
                rejected.add(sym)
                logger.error(
                    "MOC REJECTED at exchange",
                    extra={"oid": oid, "symbol": sym,
                           "status": status, "filled": filled},
                )
            continue
        try:
            info = broker.wait_for_fill(oid, timeout=this_wait)
            status = getattr(info, "status", "")
            filled = int(getattr(info, "filled", 0) or 0)
            if status in _TERMINAL_ORDER_STATES and filled == 0:
                rejected.add(sym)
                logger.error(
                    "MOC REJECTED at exchange",
                    extra={"oid": oid, "symbol": sym,
                           "status": status, "filled": filled},
                )
        except TimeoutError:
            # Presumed accepted; MOCs sit Submitted until 16:00.
            pass
        except Exception as e:
            logger.warning(
                "MOC acceptance audit: wait_for_fill failed (treating "
                "as rejected — pessimistic)",
                extra={"oid": oid, "symbol": sym, "error": str(e)[:200]},
            )
            rejected.add(sym)
    return rejected


def _coverage_cross_check(
    broker,
    entries: list[dict],
    current_baseline: dict[str, int],
    moc_signed_by_sym: dict[str, int],
    managed_universe: set[str],
) -> list[str]:
    """RR1: per-symbol GROUND-TRUTH coverage check.

    ``entries['filled_qty']`` is derived from ``_wait_entry_fills``,
    which can lie in two ways under slow-TWS conditions:
      (1) budget-exhausted branches hard-set ``filled_qty=0`` even
          though the MKT truly filled at the exchange;
      (2) truncated wait misses a late-arriving Filled status update.

    And the MOC snapshot can be silently partial (TWS reconnect during
    the ``get_positions()`` request returns 3 of 10 held names).

    Both defeats have the same shape: state derived from polling looks
    "no exposure" while the exchange actually holds an open book. Close
    them by re-polling each submitted entry's TRUE fill via
    ``broker.get_order_status``, computing expected post-entry holdings
    (``current_baseline + signed_fill``), then checking that every
    non-zero managed-universe holding was actually flattened by the
    MOC batch (``expected + moc_signed == 0``).

    Unpollable orders are treated PESSIMISTICALLY as fully filled at
    the requested qty — bias errors toward "flag as exposure" so a
    real gap can't hide.

    Returns the list of uncovered symbols. Non-empty ⇒ operator must
    flatten manually.
    """
    expected_held: dict[str, int] = dict(current_baseline)
    for e in entries:
        oid = e.get("order_id")
        if oid is None:
            continue
        sym = e["symbol"]
        side = e["side"]
        signed_sign = 1 if side == "BUY" else -1
        requested = int(e.get("requested_qty", 0) or 0)
        try:
            st = broker.get_order_status(oid)
            status = getattr(st, "status", "")
            # RRR3: a live MKT entry SHOULD be terminal by cross-check
            # time (Filled / Cancelled / ApiCancelled / Inactive) —
            # either it filled, or the belt-and-braces cancel loop
            # pulled it. A non-terminal read is itself a staleness
            # signal — the local ib_async cache may reflect a stale
            # filled=0 during a TWS event-stream brownout. Treat
            # non-terminal the same as an exception: assume pessimistic
            # full fill. Overcounting exposure > silently missing it.
            if status not in _TERMINAL_ORDER_STATES:
                logger.warning(
                    "coverage cross-check: non-terminal status at "
                    "cross-check time is a staleness signal — assuming "
                    "full fill for exposure accounting",
                    extra={"order_id": oid, "symbol": sym,
                           "status": status},
                )
                filled = requested
            else:
                filled = int(getattr(st, "filled", 0) or 0)
        except Exception as ex:
            # RR1 pessimism: if we can't poll the order, assume it fully
            # filled. Overcounting exposure is safer than missing it.
            logger.warning(
                "coverage cross-check: get_order_status failed, "
                "assuming full fill for exposure accounting",
                extra={"order_id": oid, "symbol": sym,
                       "error": str(ex)[:200]},
            )
            filled = requested
        expected_held[sym] = expected_held.get(sym, 0) + signed_sign * filled

    uncovered: list[str] = []
    for sym, q in expected_held.items():
        if sym not in managed_universe or q == 0:
            continue
        moc_signed = moc_signed_by_sym.get(sym, 0)
        # After MOC, expected residual = q + moc_signed. Non-zero = gap.
        if q + moc_signed != 0:
            uncovered.append(sym)
    return sorted(uncovered)


# Terminal statuses that end a fill audit for a given oid.
_FILL_AUDIT_TERMINAL_STATES: frozenset[str] = frozenset({
    "Filled", "Cancelled", "ApiCancelled", "Inactive",
})

# Per-oid wait cap for ENTRY legs during the fill audit. Entries are
# either terminal by the time the audit runs (they were placed at
# 15:50, filled or cancelled within milliseconds, and passed through
# the belt-and-braces cancel loop) OR stuck non-terminal (halted /
# illiquid tape / PendingCancel that TWS hasn't cleared yet). The
# latter would otherwise burn the audit's whole remaining budget on
# a wait that only settles at 16:00, starving every subsequent MOC's
# rc verdict of a real fill and pushing `broker.disconnect()` past
# 16:05. Cap the entry wait; MOCs get the full remaining budget
# because they legitimately need to block until the close cross.
_ENTRY_AUDIT_TIMEOUT_SECS: float = 10.0


def _fill_audit(
    broker,
    order_ids: list[str],
    timeout_seconds: int = 900,
    recorder: Optional["trade_recorder.TradeRecorder"] = None,
    oid_to_leg_sym: Optional[dict[str, tuple[str, str]]] = None,
) -> None:
    """Poll each order for a terminal state, up to a WHOLE-audit deadline.

    Previously each order got ``timeout_seconds / len(order_ids)``
    seconds — for a ~136-oid book that's ~6.6s per order, and MOC
    orders sit ``Submitted`` until the 16:00 close cross, so every MOC
    timed out and was recorded as ``filled_qty=0``. That's the exit
    leg of a close-auction strategy — visible in a downstream sum as
    "the exit never happened".

    The corrected design gives EACH oid the full remaining budget in
    a sequential ``wait_for_fill`` loop. Entries fill within
    milliseconds and return immediately; the first MOC blocks until
    ~16:00 (when the close cross prints); the remaining MOCs are
    already ``Filled`` by then and their ``wait_for_fill`` returns
    immediately. Total wall-clock ≈ 10 minutes on the happy path
    starting at 15:50.

    ``wait_for_fill``'s internal ``self._ib.sleep(0.2)`` pumps the
    ib_async event loop, so status updates are actually delivered to
    ``broker._trades[oid]`` during the wait (bare ``time.sleep`` would
    stall event delivery).

    Anything still non-terminal at the whole-audit deadline is
    recorded as ``status="timeout"`` and, in the parquet, gets a null
    ``filled_qty`` / ``achieved_dollars`` — a subsequent
    ``sum(achieved_dollars)`` correctly ignores it rather than
    silently treating it as $0.

    All recorder calls are ``safe_call``-wrapped so a raise inside the
    recorder cannot leak into the audit and lose the JSONL log for
    subsequent orders.
    """
    _FILLS_DIR.mkdir(parents=True, exist_ok=True)
    today = config.now_ny().date().isoformat()
    fills_path = _FILLS_DIR / f"{today}.jsonl"
    deadline = time.time() + timeout_seconds
    oid_to_leg_sym = oid_to_leg_sym or {}

    # Audit MOC oids FIRST. MOCs need the whole ~10-minute window
    # (they don't fill until 16:00). Entries then get a capped wait
    # each — they're either already terminal (fast return) or stuck
    # non-terminal, in which case the cap keeps one bad entry from
    # eating the whole audit budget and pushing broker.disconnect()
    # past 16:05.
    moc_ids = [o for o in order_ids if oid_to_leg_sym.get(o, ("", ""))[0] == "moc"]
    entry_ids = [o for o in order_ids if oid_to_leg_sym.get(o, ("", ""))[0] != "moc"]
    ordered_ids = moc_ids + entry_ids

    def _record_terminal_to_recorder(oid: str, info, observed_ts_utc: datetime) -> None:
        if recorder is None:
            return
        leg_sym = oid_to_leg_sym.get(oid)
        if leg_sym is None:
            return
        leg, sym = leg_sym
        # extract_broker_extras is defensive but wrap the whole call
        # in a try so a raise here doesn't cost us the fill row.
        try:
            extras = trade_recorder.extract_broker_extras(broker, oid)
        except Exception:
            logger.exception(
                "extract_broker_extras raised (non-fatal); recording fill without extras",
                extra={"order_id": oid, "symbol": sym},
            )
            extras = {}
        filled_val = getattr(info, "filled", 0)
        try:
            filled_qty: Optional[float] = (
                float(filled_val) if filled_val is not None else 0.0
            )
        except (TypeError, ValueError):
            filled_qty = 0.0
        fill_price_val = getattr(info, "avg_fill_price", None)
        try:
            fill_price: Optional[float] = (
                float(fill_price_val) if fill_price_val is not None else None
            )
        except (TypeError, ValueError):
            fill_price = None
        trade_recorder.safe_call(
            recorder.record_fill,
            leg=leg,
            symbol=sym,
            order_id=oid,
            filled_qty=filled_qty,
            fill_price=fill_price,
            status=getattr(info, "status", None),
            observed_ts_utc=observed_ts_utc,
            extras=extras,
        )

    def _record_timeout_to_recorder(oid: str, status: str, observed_ts_utc: datetime) -> None:
        if recorder is None:
            return
        leg_sym = oid_to_leg_sym.get(oid)
        if leg_sym is None:
            return
        leg, sym = leg_sym
        # Timeout / error — filled_qty stays NULL so downstream sums
        # don't confuse it with a definitive zero-fill.
        trade_recorder.safe_call(
            recorder.record_fill,
            leg=leg,
            symbol=sym,
            order_id=oid,
            filled_qty=None,
            fill_price=None,
            status=status,
            observed_ts_utc=observed_ts_utc,
        )

    n_terminal = 0
    n_timeout = 0
    n_error = 0
    with fills_path.open("a") as fh:
        for oid in ordered_ids:
            leg = oid_to_leg_sym.get(oid, ("moc", ""))[0]
            remaining = max(0.0, deadline - time.time())
            if remaining <= 0:
                logger.warning(
                    "fill audit deadline reached; remaining orders recorded as timeout",
                    extra={"order_id": oid, "leg": leg},
                )
                observed = datetime.now(_UTC)
                fh.write(json.dumps({
                    "ts": config.now_ny().isoformat(),
                    "order_id": oid, "status": "timeout",
                }, default=str) + "\n")
                _record_timeout_to_recorder(oid, "timeout", observed)
                n_timeout += 1
                continue
            # Cap ENTRY waits at ~10s so a stuck non-terminal entry
            # (PendingCancel, halted symbol) can't burn the whole
            # remaining budget. MOCs get the full budget because
            # they legitimately need to wait for the 16:00 print.
            per_wait = (
                min(remaining, _ENTRY_AUDIT_TIMEOUT_SECS)
                if leg == "entry" else remaining
            )
            try:
                # Sequential wait_for_fill. Entries return in ms
                # (or hit the ~10s cap); the first MOC blocks until
                # ~16:00, later MOCs return immediately.
                info = broker.wait_for_fill(oid, timeout=per_wait)
            except TimeoutError:
                observed = datetime.now(_UTC)
                fh.write(json.dumps({
                    "ts": config.now_ny().isoformat(),
                    "order_id": oid, "status": "timeout",
                }, default=str) + "\n")
                _record_timeout_to_recorder(oid, "timeout", observed)
                n_timeout += 1
                continue
            except Exception as e:
                observed = datetime.now(_UTC)
                fh.write(json.dumps({
                    "ts": config.now_ny().isoformat(),
                    "order_id": oid, "status": "error",
                    "error": str(e)[:200],
                }, default=str) + "\n")
                _record_timeout_to_recorder(oid, "error", observed)
                n_error += 1
                continue
            observed = datetime.now(_UTC)
            fh.write(json.dumps({
                "ts": config.now_ny().isoformat(),
                "order_id": getattr(info, "order_id", oid),
                "symbol": getattr(info, "symbol", None),
                "action": getattr(info, "action", None),
                "quantity": getattr(info, "quantity", None),
                "status": getattr(info, "status", None),
                "filled": getattr(info, "filled", None),
                "avg_fill_price": getattr(info, "avg_fill_price", None),
            }, default=str) + "\n")
            _record_terminal_to_recorder(oid, info, observed)
            n_terminal += 1
    logger.info(
        "fills audited",
        extra={
            "n_orders": len(order_ids),
            "n_terminal": n_terminal,
            "n_timeout": n_timeout,
            "n_error": n_error,
            "path": str(fills_path),
        },
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    config.setup_logging("live.daily_orchestrator")
    # Default to paper if neither --paper nor --live was set.
    if not args.live:
        args.paper = True

    logger.info(
        "orchestrator starting",
        extra={
            "argv": _redact_argv(list(sys.argv)),
            "cli_args": _redact_args(vars(args)),
        },
    )

    # Replay mode: dress-rehearse against cached historical data. Refuse
    # combining with --live outright — routing historical alpha into a
    # LIVE account is nonsensical and dangerous. Paper (or dry-run) only.
    replay_mode = args.replay_date is not None
    if replay_mode and args.live:
        logger.error(
            "--replay-date is refused with --live; replay is paper- or dry-run-only",
            extra={"replay_date": args.replay_date},
        )
        return 6
    if args.replay_submit and not replay_mode:
        logger.error("--replay-submit requires --replay-date")
        return 7
    # R7: --lambda- <= 0 would blow up scale = 1/lambda with a bare
    # ZeroDivisionError (0) or produce a nonsensical negative-risk-aversion
    # scale (negative lam). Refuse loudly with a distinct rc so a cron
    # wrapper can distinguish an invalid-arg failure from an infra failure.
    # RR2: rc=9 (safe skip; no orders placed) instead of rc=12 so it
    # stays out of the `rc >= 11` "page an operator" danger band.
    if args.lam <= 0:
        logger.error(
            "--lambda- must be > 0 (negative or zero lambda is not "
            "a valid risk aversion)",
            extra={"got": args.lam},
        )
        return 9
    # F2: replay is BROKER-FREE by default. --replay-submit opts in to the
    # real order-plumbing path (still paper-only). Warn loudly at startup
    # so the operator knows to flatten the paper account before the next
    # real session — replay paper fills pollute the diff baseline.
    if replay_mode and args.replay_submit:
        logger.warning(
            "REPLAY SUBMIT: real paper MKT entries + MOC exits will be "
            "placed. Both legs hit the paper account; fills from a replay "
            "pollute the next real paper/live run's diff baseline — "
            "flatten the paper account before the next real session.",
            extra={"replay_date": args.replay_date},
        )

    universe = config.load_active_universe(args.universe_file)
    N = len(universe)
    logger.info("loaded active universe", extra={"n": N, "file": args.universe_file})

    # Load persisted models. Fail fast — if anything is missing, we do NOT
    # want to place orders based on the wrong artifacts.
    pipe = _load_forecast_pipe()
    cov = _load_cov_artifacts()
    if cov["L"].shape[1] != N:
        raise RuntimeError(
            f"covariance loadings shape {cov['L'].shape} does not match universe size {N}"
        )
    impact_gamma = _load_impact_vector(args.use_impact_vector, args.impact_gamma, N)
    reference_stats = _load_reference_stats(args.reference_stats_path, universe)

    forecast_model = ForecastModel(universe, model=pipe, features=ALPHA_FEATURES)
    cost_model = CostModel(universe)
    impact_model = ImpactModel(universe)

    # Timing.
    stream_hh, stream_mm = config.parse_hhmm(args.stream_start_time)
    reb_hh, reb_mm = config.parse_hhmm(args.rebalance_time)
    sub_hh, sub_mm = config.parse_hhmm(args.submit_by)
    stream_at = config.today_ny(stream_hh, stream_mm)
    rebalance_at = config.today_ny(reb_hh, reb_mm)
    submit_by_at = config.today_ny(sub_hh, sub_mm)
    now = config.now_ny()

    # ------------------------------------------------------------------
    # 1) Open feed (LIVE or REPLAY) unless disabled
    # ------------------------------------------------------------------

    feed = None
    try:
        if replay_mode:
            # Replay: cached historical adapter, no wall-clock wait — the
            # data is already sitting on disk (or one-off materialise).
            from data_collection.data_adapters import DatabentoGenericAdapter
            api_key = config.resolve_databento_key()
            adapter = DatabentoGenericAdapter(api_key, config.CACHE_DIR)
            feed = ReplayFeed(
                adapter=adapter,
                universe=universe,
                replay_date=args.replay_date,
                allow_fetch=args.allow_fetch,
            )
            logger.info(
                "replay mode: starting historical feed",
                extra={"replay_date": args.replay_date, "n_universe": N},
            )
            feed.start()
        elif not args.no_live_feed:
            api_key = config.resolve_databento_key()
            feed = LiveFeed(api_key=api_key, universe=universe)
            if now < stream_at:
                logger.info(
                    "waiting until stream start",
                    extra={"target": stream_at.isoformat(), "now": now.isoformat()},
                )
                _sleep_until(stream_at)
            feed.start()
    except Exception as e:
        # F3: a cold-cache replay of a weekend/holiday, an auth error, or
        # any Databento/adapter exception in ``start()`` reaches here.
        # No broker was contacted yet — return the feed-abort rc instead
        # of letting the traceback escape.
        logger.exception("feed start failed", extra={"error": str(e), "replay_mode": replay_mode})
        return 2

    # ------------------------------------------------------------------
    # 2) Sleep until rebalance time
    # ------------------------------------------------------------------

    # H4: the imbalance print publishes AT the rebalance minute — snapshotting
    # exactly then races the print. Sleep an extra ``--print-wait-secs`` past
    # the wall time before the snapshot. The submit-by cutoff (below) still
    # bounds this from running long.
    #
    # Replay mode skips this wait: the historical parquet already covers the
    # full trading day, so ``now`` is unrelated to the replay date.
    snapshot_at = rebalance_at + timedelta(seconds=args.print_wait_secs)
    if replay_mode:
        logger.info(
            "replay mode: skipping snapshot wall-clock wait",
            extra={"snapshot_at": snapshot_at.isoformat()},
        )
    else:
        logger.info(
            "waiting for snapshot time",
            extra={
                "rebalance_at": rebalance_at.isoformat(),
                "snapshot_at": snapshot_at.isoformat(),
                "print_wait_secs": args.print_wait_secs,
                "now": config.now_ny().isoformat(),
            },
        )
        _sleep_until(snapshot_at)

    # ------------------------------------------------------------------
    # 3) Snapshot + feature engineering
    # ------------------------------------------------------------------

    # Cache the tbbo snapshot at decision time ONLY on the replay
    # path — replay's snapshot covers the full historical day already,
    # so we can pull the T+1..T+5 mids synchronously without keeping
    # any stream alive. In live/paper we let the recorder's background
    # marks thread call ``feed.snapshot()`` once at ~15:55 (T+MAX)
    # and walk that snapshot back to derive every earlier T+N mark;
    # taking a redundant copy at 15:50 would burn ~1s on the critical
    # path (~0.9s/M records) for nothing.
    tbbo_at_decision: pl.DataFrame = pl.DataFrame()
    snapshot_completed_at_utc: Optional[datetime] = None
    last_mid_ts_by_sym: dict[str, datetime] = {}
    try:
        if feed is None:
            day_df = pl.DataFrame()
            last_mid_by_sym: dict[str, float] = {}
        else:
            day_df, last_mid_by_sym, last_mid_ts_by_sym = _build_live_day_frame(
                feed, universe, reference_stats,
                min_coverage=args.min_universe_coverage,
            )
            snapshot_completed_at_utc = datetime.now(_UTC)
            if replay_mode:
                try:
                    tbbo_at_decision = feed.snapshot().get("tbbo", pl.DataFrame())
                except Exception:
                    logger.exception(
                        "replay tbbo snapshot for recorder failed (non-fatal)"
                    )
    except Exception as e:
        logger.exception("feature build failed", extra={"error": str(e)})
        if args.on_feed_failure == "abort":
            if feed is not None:
                try:
                    feed.stop()
                except Exception:
                    logger.exception("feed stop failed (non-fatal)")
            return 2
        day_df = pl.DataFrame()
        last_mid_by_sym = {}

    # For replay / dry-run we stop the feed now — replay tbbo is
    # already fully materialised in ``tbbo_at_decision``, and dry-run
    # never trades. For live/paper the feed stays alive so the
    # recorder's background marks capture can call ``feed.snapshot()``
    # once around 15:55 (T+MAX); the outer ``finally`` at end-of-main
    # stops it on every exit path (including rc=3/4/5/8 early returns).
    if feed is not None and (replay_mode or args.dry_run):
        try:
            feed.stop()
        except Exception:
            logger.exception("feed stop failed (non-fatal)")

    # ------------------------------------------------------------------
    # 4) Alpha + MVO
    # ------------------------------------------------------------------

    alpha_vec = forecast_model.get_alpha(day_df)
    cost_vec = cost_model.get_cost(day_df, config.COMMISSION_PER_SHARE)
    impact_coeff_vec = impact_model.get_impact_coeff(day_df)
    logger.info(
        "alpha computed",
        extra={
            "n_names_with_alpha": int((alpha_vec != 0).sum()),
            "alpha_abs_median": float(np.median(np.abs(alpha_vec))),
            "cost_median": float(np.median(cost_vec)),
            "impact_coeff_median": float(np.median(impact_coeff_vec)),
        },
    )

    optimizer = Optimizer(universe)
    # C3: --vol-cap is the effective risk limit (target daily $-vol); the QP
    # is scale-invariant so ``scale`` cancels between the objective and the
    # ``quad_risk <= vol_cap/scale`` constraint. ``scale`` is purely a
    # numerical conditioning factor; 1/lambda is the expected order of
    # magnitude of the unconstrained optimal book (y* ~= Sigma^-1 alpha / lam)
    # so it's a natural auto-choice. The old --target-gmv flag misleadingly
    # implied it sized the book — it did not; only --vol-cap, --lambda-,
    # and the impact/cost terms determine actual GMV.
    scale = 1.0 / args.lam
    optimizer.set_problem(
        Sigma_r=cov["Sigma_r"],
        Sigma_f=cov["Sigma_f"],
        Loadings=cov["L"],
        eps_diag=cov["eps_diag"],
        lam=args.lam,
        gam=impact_gamma,
        vol_cap=args.vol_cap,
        scale=scale,
    )
    try:
        dollar_positions = optimizer.solve(alpha_vec, cost_vec, impact_coeff_vec)
    except Exception as e:
        logger.exception("solver failed", extra={"error": str(e)})
        _stop_feed_safe(feed)
        return 3
    gmv = float(np.abs(dollar_positions).sum())
    nmv = float(dollar_positions.sum())
    logger.info(
        "MVO solved",
        extra={
            "gmv_dollar": gmv,
            "nmv_dollar": nmv,
            "scale": scale,
            "vol_cap": args.vol_cap,
            "lambda": args.lam,
        },
    )

    # ------------------------------------------------------------------
    # 5) Dollar -> shares
    # ------------------------------------------------------------------

    share_targets = _compute_share_targets(dollar_positions, universe, day_df, last_mid_by_sym)
    logger.info(
        "share targets computed",
        extra={
            "n_names_traded": len(share_targets),
            "gross_shares": int(sum(abs(v) for v in share_targets.values())),
        },
    )

    # ------------------------------------------------------------------
    # 6) Submit or dry-run
    # ------------------------------------------------------------------

    now = config.now_ny()
    # Replay mode: the wall clock is unrelated to the historical date, so the
    # 15:55 cutoff would always fire on a rehearsal run late in the day.
    # Skip it in replay only; live/paper still honour it.
    if not replay_mode and now > submit_by_at:
        # rc=4: past cutoff BEFORE any orders were placed. Distinct from
        # rc=11 (past cutoff AFTER entries filled) — this one is a safe
        # skip; no book, no exposure.
        logger.error(
            "past MOC submit cutoff; refusing to place orders "
            "(safe skip — no book was created)",
            extra={"now": now.isoformat(), "submit_by": submit_by_at.isoformat()},
        )
        _stop_feed_safe(feed)
        return 4

    # L1: refuse to route an empty book into a rebalance() — with
    # close_unlisted=True (however scoped), an empty target dict would
    # flatten the entire managed book. Belt and braces alongside the
    # coverage check above.
    if not share_targets and not args.dry_run:
        logger.error("empty share_targets — refusing to touch the broker")
        _stop_feed_safe(feed)
        return 5

    # M2: persist the intended book to disk before any broker call. This is
    # the "truth" the next morning's reconciliation compares against.
    universe_hash = _hash_universe(universe)
    if replay_mode:
        record_mode = "replay"
    elif args.live:
        record_mode = "live"
    else:
        record_mode = "paper"
    target_record = {
        "generated_at_utc": now.astimezone(_UTC).isoformat(),
        "generated_at_ny": now.isoformat(),
        "mode": record_mode,
        "dry_run": args.dry_run,
        "lambda": args.lam,
        "scale": scale,
        "vol_cap": args.vol_cap,
        "gmv_dollar": gmv,
        "nmv_dollar": nmv,
        "universe_file": args.universe_file,
        "universe_hash": universe_hash,
        "targets": share_targets,
    }
    if replay_mode:
        target_record["replay_date"] = args.replay_date
    _write_target_record(target_record)

    if args.dry_run:
        logger.info(
            "DRY RUN — printing target book, no broker connection",
            extra={"targets": share_targets},
        )
        _stop_feed_safe(feed)
        return 0

    # F2: default replay behavior is broker-free — validate the whole
    # features->alpha->MVO->sizing chain but do not send orders. Only
    # ``--replay-submit`` opts in to the real paper order path.
    if replay_mode and not args.replay_submit:
        logger.info(
            "REPLAY (no-submit) — book computed and persisted, no broker "
            "connection. Pass --replay-submit to exercise the order path.",
            extra={"targets": share_targets},
        )
        _stop_feed_safe(feed)
        return 0

    try:
        broker = _connect_broker(args)
    except Exception as e:
        # Broker refused to connect — most commonly the live-port guard
        # tripping on a paper-mode port mismatch, TWS not running on the
        # expected port, or a bad client_id collision. No orders have been
        # placed; exit with a coded rc rather than a bare traceback.
        logger.exception("broker connect failed", extra={"error": str(e)})
        _stop_feed_safe(feed)
        return 8

    # ------------------------------------------------------------------
    # Trade recorder — captures one parquet row per trade for post-trade
    # slippage / impact analysis. Every call site is wrapped in
    # ``trade_recorder.safe_call`` (or an inline try/except) so a
    # recorder exception (I/O, schema, missing field) cannot propagate
    # into the order path. Created here rather than earlier so a
    # broker-connect failure (rc=8) leaves no half-populated recorder
    # rows to finalize.
    # ------------------------------------------------------------------
    recorder: Optional[trade_recorder.TradeRecorder] = None
    try:
        if replay_mode:
            from datetime import date as _date
            rd = args.replay_date.split("-")
            _replay_date_obj = _date(int(rd[0]), int(rd[1]), int(rd[2]))
            # Anchor the historical decision time at HH:MM then ADD
            # print-wait-secs as a timedelta — ``second=<int>`` on the
            # datetime constructor raises ``ValueError`` for values
            # >= 60, and a swallowed exception would silently disable
            # the recorder for the whole day.
            decision_dt_ny = datetime(
                _replay_date_obj.year, _replay_date_obj.month,
                _replay_date_obj.day, reb_hh, reb_mm,
                0, tzinfo=config.NY_TZ,
            ) + timedelta(seconds=float(args.print_wait_secs))
            recorder_trading_date = args.replay_date
        else:
            decision_dt_ny = snapshot_at
            recorder_trading_date = config.now_ny().date().isoformat()
        decision_ts_utc = decision_dt_ny.astimezone(_UTC)
        recorder = trade_recorder.TradeRecorder(
            mode=record_mode,
            decision_ts_utc=decision_ts_utc,
            universe=universe,
            trading_date=recorder_trading_date,
            base_dir=_resolve_trades_dir(broker=broker),
            replay_date=args.replay_date if replay_mode else None,
            universe_hash=universe_hash,
            snapshot_ts_utc=snapshot_completed_at_utc,
        )
        _decision_context = {
            "mode": record_mode,
            "lambda": float(args.lam),
            "scale": float(scale),
            "vol_cap": float(args.vol_cap),
            "gmv_dollar": float(gmv),
            "nmv_dollar": float(nmv),
            "solver_status": str(getattr(optimizer.problem, "status", "")),
            "objective_value": (
                float(optimizer.problem.value)
                if getattr(optimizer, "problem", None) is not None
                and optimizer.problem.value is not None else None
            ),
            "impact_gamma_kind": (
                "vector" if isinstance(impact_gamma, np.ndarray) else "scalar"
            ),
            "impact_gamma_scalar": (
                float(impact_gamma) if not isinstance(impact_gamma, np.ndarray)
                else None
            ),
            "universe_file": args.universe_file,
            "universe_size": int(len(universe)),
            "universe_hash": universe_hash,
            "replay_date": args.replay_date if replay_mode else None,
            "min_universe_coverage": float(args.min_universe_coverage),
            "print_wait_secs": float(args.print_wait_secs),
            "rebalance_time": args.rebalance_time,
            "submit_by": args.submit_by,
            "artifact_paths": {
                "forecast_pipe": str(
                    config.FORECAST_DIR / config.FORECAST_PIPE_NAME
                ),
                "cov_loadings": str(
                    config.COV_DIR / config.COV_LOADINGS_NAME
                ),
                "cov_matrix": str(
                    config.COV_DIR / config.COV_MATRIX_NAME
                ),
                "impact_gamma_vec": str(
                    config.IMPACT_DIR / config.IMPACT_GAMMA_NAME
                ),
                "reference_stats": (
                    args.reference_stats_path
                    or str(config.IMPACT_DIR / config.REF_STATS_NAME)
                ),
            },
        }
        trade_recorder.safe_call(
            recorder.record_decision_context, _decision_context,
        )
        _arrivals = trade_recorder.arrival_prices_from_day_df(
            day_df, last_mid_by_sym, last_mid_ts_by_sym,
        )
        trade_recorder.safe_call(recorder.record_arrival_prices, _arrivals)
        _targets_map: dict[str, dict] = {}
        for _i, _sym in enumerate(universe):
            _td = float(dollar_positions[_i])
            _ts = share_targets.get(_sym, 0)
            if _td == 0 and _ts == 0:
                continue
            _targets_map[_sym] = {
                "target_dollars": _td,
                "target_shares": _ts,
            }
        trade_recorder.safe_call(recorder.record_targets, _targets_map)
        # Impact-decay-curve marks at T+1..T+5min. In replay,
        # tbbo_at_decision already covers the whole historical day —
        # extract synchronously per offset. In live/paper, spawn a
        # single background thread that snapshots the still-running
        # feed at decision + MAX(offsets) (~15:55) and walks that one
        # snapshot back to derive each earlier mark; the thread is
        # bounded and the main path never blocks on it. See
        # ``schedule_marks_capture_from_feed`` docstring for why
        # one-wake-then-walkback is strictly better than five timers.
        if replay_mode:
            trade_recorder.safe_call(
                recorder.capture_marks_from_tbbo,
                tbbo_at_decision,
                decision_ts_utc,
            )
        elif feed is not None:
            trade_recorder.safe_call(
                recorder.schedule_marks_capture_from_feed,
                feed=feed,
                decision_ts_utc=decision_ts_utc,
            )
    except Exception:
        # A recorder-init failure is non-fatal; the order path continues
        # exactly as before. Leaving ``recorder=None`` disables every
        # later call site.
        logger.exception("trade recorder init failed (non-fatal)")
        recorder = None

    try:
        # M2 (part 2): reconcile against yesterday's saved book, if any.
        prev = _load_previous_target_record()
        if prev is not None:
            _reconcile_from_previous(broker, prev)

        # Actual TWS positions right now — the diff baseline.
        # RRR4: aggregate duplicate rows the same way _submit_moc_flatten
        # does. A last-row-wins dict comprehension here would undercount
        # if IB returns duplicates (multi-account / update-race), biasing
        # the coverage cross-check toward SPURIOUS rc=11 (safe direction
        # but noisy pager). Sum instead.
        current: dict[str, int] = {}
        for p in broker.get_positions():
            current[p.symbol] = current.get(p.symbol, 0) + int(p.quantity)
        universe_set = set(universe)

        # H1: only close positions in the managed universe. Out-of-universe
        # holdings get pinned to their current qty in the extended target so
        # they diff-out to zero. This means we can safely pass
        # ``close_unlisted=False`` to broker.rebalance() and never touch a
        # position we didn't put on.
        extended_targets: dict[str, int] = dict(share_targets)
        for sym, qty in current.items():
            if sym in universe_set:
                extended_targets.setdefault(sym, 0)
            else:
                extended_targets[sym] = qty  # explicit no-op preserves qty
        logger.info(
            "extended target (managed-only close)",
            extra={
                "n_target_names": len(share_targets),
                "n_current_names": len(current),
                "n_out_of_universe_pinned": sum(1 for s in current if s not in universe_set),
            },
        )

        # H2 (part 1): pre-qualify every contract we intend to trade. Any
        # symbol IB refuses to qualify (delisted / bad ticker) is dropped
        # from the target with a warning so a single bad name doesn't kill
        # the whole submission mid-loop.
        managed_symbols = set(share_targets) | (set(current) & universe_set)
        qualified_symbols = _prequalify_symbols(broker, sorted(managed_symbols))
        for sym in list(extended_targets):
            if sym in universe_set and sym not in qualified_symbols:
                logger.warning(
                    "dropping unqualified symbol from target",
                    extra={"symbol": sym, "target_qty": extended_targets.get(sym)},
                )
                extended_targets.pop(sym, None)

        # H2 (part 2): find the largest absolute leg diff and raise
        # MAX_ORDER_QTY if needed. Prefer raising the ceiling with a logged,
        # computed value over silently scaling the whole book.
        max_leg = _max_leg_size(extended_targets, current)
        if max_leg > broker.MAX_ORDER_QTY:
            new_ceiling = int(max_leg * 1.1) + 1  # 10% headroom
            logger.warning(
                "raising broker MAX_ORDER_QTY for this session",
                extra={
                    "old": broker.MAX_ORDER_QTY,
                    "new": new_ceiling,
                    "largest_leg": max_leg,
                },
            )
            broker.MAX_ORDER_QTY = new_ceiling

        # M1: per-leg submit-by check for the MOC EXIT batch. The MKT ENTRY
        # batch always fires (see below): a filled entry with no queued exit
        # would still leave open exposure, so we always try to route the exit;
        # if the wall clock has crossed submit_by by then we exit rc=4 loudly.
        #
        # F1: in replay mode the wall clock is unrelated to the historical
        # date, so ``submit_by_at`` (anchored to TODAY 15:55) would silently
        # skip every leg on an afternoon rehearsal. Compute an effective
        # cutoff comfortably in the future for replay only; live/paper
        # keep the real cutoff.
        if replay_mode:
            effective_submit_by = config.now_ny() + timedelta(hours=1)
            logger.info(
                "replay mode: per-leg submit-by neutralised",
                extra={"effective_submit_by": effective_submit_by.isoformat()},
            )
        else:
            effective_submit_by = submit_by_at

        # --------------------------------------------------------------
        # LEG 1: plain MKT entries sized to (target - current). Fires
        # unconditionally with respect to the MOC cutoff. Default
        # tif=DAY; the belt-and-braces cancel loop below bounds any
        # entry still non-terminal at the end of _wait_entry_fills.
        # --------------------------------------------------------------
        entries = _submit_mkt_entries(
            broker=broker,
            targets=extended_targets,
            current=current,
            managed_universe=universe_set,
        )
        # Recorder — one entry submission row per attempted MKT. Even
        # failed submissions (``order_id is None``) get a row so post-
        # trade analysis sees the intended trade that never made it.
        if recorder is not None:
            _entry_sent_ts = datetime.now(_UTC)
            for _e in entries:
                trade_recorder.safe_call(
                    recorder.record_submission,
                    leg="entry",
                    symbol=_e.get("symbol", ""),
                    side=_e.get("side", ""),
                    requested_qty=int(_e.get("requested_qty", 0) or 0),
                    order_type="MKT",
                    order_id=_e.get("order_id"),
                    sent_ts_utc=_entry_sent_ts,
                    exchange=NASDAQ_EXCHANGE,
                    tif="DAY",
                    error=_e.get("error"),
                )
        logger.info(
            "MKT entry batch submitted",
            extra={
                "n_entries": len(entries),
                "n_ok": sum(1 for e in entries if e.get("order_id")),
                "n_failed": sum(1 for e in entries if e.get("error")),
            },
        )
        # R5: bounded fill wait tied to effective_submit_by so a slow
        # TWS can't drag the MOC snapshot past the cutoff. Ground-truth
        # for the MOC size is always broker.get_positions() below — not
        # this poll.
        _wait_entry_fills(
            broker, entries,
            submit_by=effective_submit_by,
            per_order_timeout=config.DEFAULT_ENTRY_FILL_TIMEOUT_SECS,
        )

        # R2 belt-and-braces: cancel every entry that never reached a
        # terminal state. Plain MKT should fill within milliseconds on a
        # liquid tape, but a placed-but-not-acked / illiquid-hang order
        # can survive. Cancel BEFORE the MOC snapshot so late fills
        # don't leave an unhedged position past the counter-flatten.
        cancelled_oids = _cancel_non_terminal_entries(broker, entries)
        if cancelled_oids:
            logger.info(
                "non-terminal entries cancelled before MOC snapshot",
                extra={"n_cancelled": len(cancelled_oids)},
            )

        # --------------------------------------------------------------
        # LEG 2: MOC counter-flatten sized to ACTUAL positions.
        # Structural invariant: the MOC size is drawn from a fresh
        # broker.get_positions() snapshot, NEVER from `target - current`
        # or from the entry's requested_qty. Partial-fill scenarios
        # therefore cannot over-flatten into an opposite overnight
        # exposure. See _submit_moc_flatten docstring.
        # --------------------------------------------------------------
        (moc_batch, moc_skipped,
         moc_positions_unknown) = _submit_moc_flatten(
            broker=broker,
            managed_universe=universe_set,
            submit_by=effective_submit_by,
        )
        moc_order_ids = [leg["oid"] for leg in moc_batch]
        moc_signed_by_sym = {leg["symbol"]: leg["signed_qty"]
                             for leg in moc_batch}
        moc_oid_to_sym = {leg["oid"]: leg["symbol"] for leg in moc_batch}
        # Recorder — one MOC submission row per successful leg. Skipped
        # symbols (past cutoff / snapshot failed) don't have oids and
        # aren't recorded here; the exposure verdict picks them up in
        # the target-record persistence path.
        if recorder is not None:
            _moc_sent_ts = datetime.now(_UTC)
            for _leg in moc_batch:
                trade_recorder.safe_call(
                    recorder.record_submission,
                    leg="moc",
                    symbol=_leg["symbol"],
                    side=_leg["side"],
                    requested_qty=int(_leg["qty"]),
                    order_type="MOC",
                    order_id=_leg["oid"],
                    sent_ts_utc=_moc_sent_ts,
                    exchange=NASDAQ_EXCHANGE,
                    tif=None,  # MOC has no meaningful TIF at IB
                )
        n_filled_entries = sum(
            1 for e in entries if int(e.get("filled_qty", 0) or 0) > 0
        )
        logger.info(
            "MOC exit batch submitted",
            extra={
                "n_orders": len(moc_order_ids),
                "n_skipped_past_cutoff": len(moc_skipped),
                "positions_unknown": moc_positions_unknown,
                "n_filled_entries": n_filled_entries,
                "order_ids": moc_order_ids,
            },
        )

        # RR1: GROUND-TRUTH coverage cross-check. Independent of
        # _wait_entry_fills (which can hard-set filled_qty=0 on the
        # budget-exhausted branch) AND independent of the MOC snapshot
        # (which can return partial). Re-polls each entry's true fill
        # via broker.get_order_status and verifies every non-zero
        # managed-universe expected holding was actually flattened.
        # RRR5 note: broker.get_order_status is a local self._trades
        # dict read (O(N) microseconds), not a network call — the
        # earlier "~5s for N=100" comment was wrong. Cost is negligible.
        # This runs AFTER the MOC batch was sent, so it does not extend
        # the exposure window either way.
        uncovered_symbols = _coverage_cross_check(
            broker=broker,
            entries=entries,
            current_baseline=current,
            moc_signed_by_sym=moc_signed_by_sym,
            managed_universe=universe_set,
        )

        # R1 additional guard: if entries reported fills but the MOC
        # snapshot returned an empty position book (get_positions
        # succeeded but returned []/partial due to TWS staleness), main
        # would otherwise see (order_ids=[], skipped=[]) and exit rc=0
        # — with a MKT-entered book riding the auction unhedged. Detect
        # and flag it as an exposure condition.
        empty_snapshot_after_fills = (
            n_filled_entries > 0
            and not moc_order_ids
            and not moc_skipped
            and not moc_positions_unknown
        )
        if empty_snapshot_after_fills:
            logger.error(
                "MOC exit: entries filled but zero MOC legs sent — TWS "
                "position snapshot may be stale; POSITIONS MAY BE LIVE "
                "INTO THE CLOSE, OPERATOR MUST MANUALLY FLATTEN",
                extra={"n_filled_entries": n_filled_entries,
                       "n_entries": len(entries)},
            )

        # RRR1: MOC exchange acceptance audit. broker.buy/sell returns
        # the oid the moment TWS accepts the API call — BEFORE the
        # exchange ack. A MOC submitted past the ~15:50 cutoff is
        # API-accepted then Cancelled at exchange. Wait a bounded window
        # for terminal state; anything terminal-not-Filled is a rejection.
        # See _audit_moc_acceptance for the trade-off between short wait
        # (fast operator feedback) and full 15:55->16:00 wait (fully
        # safe but slow).
        try:
            unfilled_moc_syms = _audit_moc_acceptance(
                broker, moc_batch,
                wall_clock_wait=args.moc_ack_timeout,
            )
        except Exception:
            # RRR2 pessimism: if the audit itself crashed we cannot
            # verify acceptance. Treat every MOC as at-risk so the rc
            # gate escalates. Exposure > silent success.
            logger.exception(
                "MOC acceptance audit crashed — treating every MOC as "
                "at-risk (pessimistic)",
            )
            unfilled_moc_syms = {leg["symbol"] for leg in moc_batch}

        # RRR2: decide the exposure verdict from every signal known at
        # this point, BEFORE any I/O that could raise and short-circuit
        # the rc=11 branch (persist + JSONL audit). A raised exception
        # from disk-full / permissions would otherwise propagate,
        # main() exits rc=1, and the pager (keyed on rc >= 11) drops
        # the real exposure page.
        exposure = bool(
            moc_skipped or moc_positions_unknown
            or empty_snapshot_after_fills or uncovered_symbols
            or unfilled_moc_syms
        )

        # Persist the ACTUAL executed record next to the intended one so
        # the operator can reconcile after the fact. Amend the target
        # record on disk in place (same path). Best-effort: a write
        # failure does NOT downgrade an rc=11 exposure verdict.
        target_record["entries"] = [
            {k: v for k, v in e.items() if k != "error" or v is not None}
            for e in entries
        ]
        target_record["moc_batch"] = moc_batch
        target_record["moc_order_ids"] = moc_order_ids
        target_record["moc_signed_by_sym"] = moc_signed_by_sym
        target_record["moc_skipped_past_cutoff"] = moc_skipped
        target_record["moc_positions_unknown"] = moc_positions_unknown
        target_record["empty_snapshot_after_fills"] = empty_snapshot_after_fills
        target_record["uncovered_symbols"] = uncovered_symbols
        target_record["unfilled_moc_symbols"] = sorted(unfilled_moc_syms)
        target_record["cancelled_entry_oids"] = cancelled_oids
        try:
            _write_target_record(target_record)
        except Exception:
            logger.exception("target record write failed (non-fatal)")

        # R3 + RR1 + RRR1: distinct rc for the OVERNIGHT-EXPOSURE state.
        # rc=11 is THE pager predicate (per CLAUDE.md), so NOTHING
        # recording-related may run before this return — a marks
        # wait or a MOC fill audit here would delay the operator's
        # window to run ``live.panic`` before the 16:00 close. This
        # is the second time the pattern has bitten (round 1 removed
        # ``finalize(wait=30)`` from the ``finally``, round 2 moved
        # a 350s wait into the exact same seat). The ``finally``
        # below does the minimal safe cleanup: ``finalize(wait=None)``
        # (fast), ``_stop_feed_safe`` (fast, idempotent),
        # ``broker.disconnect()``.
        if exposure:
            panic_cmd = ("python -m live.panic --paper --flatten-with MKT "
                         "(or --live --live-confirm=... --yes for live)")
            unfilled_list = sorted(unfilled_moc_syms)
            logger.error(
                "OVERNIGHT EXPOSURE — MOC exit incomplete. RUN: %s. "
                "Uncovered symbols: %s. MOC rejected at exchange: %s. "
                "Skipped: %s. Positions unknown: %s. Empty snapshot "
                "after fills: %s.",
                panic_cmd,
                uncovered_symbols[:20] or "(see logs above)",
                unfilled_list[:20] or "(none)",
                moc_skipped[:20],
                moc_positions_unknown,
                empty_snapshot_after_fills,
                extra={
                    "n_skipped": len(moc_skipped),
                    "skipped_symbols": moc_skipped[:20],
                    "positions_unknown": moc_positions_unknown,
                    "empty_snapshot_after_fills": empty_snapshot_after_fills,
                    "n_filled_entries": n_filled_entries,
                    "uncovered_symbols": uncovered_symbols[:20],
                    "n_uncovered": len(uncovered_symbols),
                    "unfilled_moc_symbols": unfilled_list[:20],
                    "n_unfilled_moc": len(unfilled_list),
                    "panic_command": panic_cmd,
                },
            )
            return 11

        # Non-exposure path from here on: recorder finalization work.
        # None of this runs on the rc=11 path (above) so it can't
        # delay the operator page.

        # Stop the live feed EARLY — as soon as the recorder's marks
        # background thread has landed (or given up). Otherwise
        # ``_fill_audit`` blocks to ~16:00 for MOC fills and the
        # Databento buffers accumulate the whole closing auction's
        # worth of tbbo per name in the meantime, which is pure
        # memory growth on a long-running live box. On replay this
        # is a no-op (the feed was already stopped at feature-build
        # time).
        if recorder is not None:
            # Bounded wait: 5min for the T+MAX target instant + some
            # slack for the snapshot itself. If the thread hasn't
            # landed by then we still stop the feed — the per-mark
            # columns just stay null on those rows. This wait is
            # AFTER the rc=11 exposure gate above, so a delay here
            # never gates the operator page.
            trade_recorder.safe_call(
                recorder.wait_for_marks, timeout=350.0,
            )
        _stop_feed_safe(feed)

        # M2 (part 3): JSONL fill audit — best-effort logging only,
        # NOT tied to the rc gate (RRR2). Fills happen at 16:00 close,
        # so we cap the wait at 15 minutes past submit-by to avoid
        # the process lingering past the trading day. The MOC
        # acceptance audit above is the rc-gating check.
        audit_ids = [e["order_id"] for e in entries if e.get("order_id")]
        audit_ids += moc_order_ids
        # (leg, symbol) index for the trade recorder — needed so the
        # audit can attach each returned OrderInfo to the correct row.
        _oid_to_leg_sym: dict[str, tuple[str, str]] = {}
        for _e in entries:
            _oid = _e.get("order_id")
            if _oid is not None:
                _oid_to_leg_sym[str(_oid)] = ("entry", _e["symbol"])
        for _leg in moc_batch:
            _oid_to_leg_sym[str(_leg["oid"])] = ("moc", _leg["symbol"])
        try:
            _fill_audit(
                broker, audit_ids, timeout_seconds=900,
                recorder=recorder, oid_to_leg_sym=_oid_to_leg_sym,
            )
        except Exception:
            logger.exception("fill audit failed (non-fatal)")
    finally:
        # Finalize the recorder WITHOUT waiting for the marks thread
        # — the ``finally`` fires on the rc=11 OVERNIGHT-EXPOSURE
        # page too, and a 30s wait between the exposure detection and
        # the ``return 11`` would delay the operator alert. On the
        # happy path the marks thread has already fired (MOC audit
        # takes it past 15:55 anyway); on any early-exit path the
        # per-mark columns simply stay null. Post-trade analysis is
        # never real-time and can tolerate the loss.
        if recorder is not None:
            try:
                recorder.finalize(wait_for_marks=None)
            except Exception:
                logger.exception(
                    "trade recorder finalize failed (non-fatal)"
                )
        _stop_feed_safe(feed)
        broker.disconnect()

    return 0


if __name__ == "__main__":
    sys.exit(main())
