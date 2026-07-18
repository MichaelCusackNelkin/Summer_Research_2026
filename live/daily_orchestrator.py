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
import sys
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

    # Last mid per symbol for share sizing (fresher than entry_price if we
    # kept streaming after the 15:50 print).
    last_mid = (
        tbbo
        .sort("ts_event")
        .with_columns(((pl.col("ask_px_00") + pl.col("bid_px_00")) / 2).alias("mid"))
        .group_by("symbol", maintain_order=False)
        .agg(pl.col("mid").last().alias("last_mid"))
    )
    last_mid_by_sym = dict(zip(last_mid["symbol"].to_list(), last_mid["last_mid"].to_list()))

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
    return day_df, last_mid_by_sym


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

    NASDAQ-listed eligibility filter
    --------------------------------

    Direct-routing to ``ISLAND`` is not the same as being NASDAQ-listed.
    IB will qualify NYSE- or ARCA-primary names on ``ISLAND`` for
    continuous UTP trading — but the MOC at the close is the
    *primary listing exchange's* closing cross. So a name whose
    ``primaryExchange`` is ``"NYSE"`` (or anything other than
    ``"NASDAQ"``) will pass qualification here, then have its MOC exit
    mis-print / rejected at the 16:00 close-cross. That silently
    breaks the strategy (which is calibrated to the NASDAQ auction
    print).

    We therefore also require ``primaryExchange == "NASDAQ"`` on every
    qualified contract and DROP any non-NASDAQ-listed name with a loud
    warning. This is the correct behaviour for a NASDAQ-only strategy:
    fills off the NASDAQ close-cross are useless.
    """
    ok: set[str] = set()
    for sym in symbols:
        try:
            contract = broker._make_contract(sym, exchange=NASDAQ_EXCHANGE)
        except Exception as e:
            logger.warning(
                "contract qualification failed",
                extra={"symbol": sym, "error": str(e)[:200]},
            )
            continue
        # NASDAQ-listed eligibility: drop names whose primary listing
        # is not NASDAQ, because their MOC prints on their home
        # exchange's close-cross, not NASDAQ's. We include the
        # qualified contract's primaryExchange in the log so the
        # operator can see WHY the name was dropped.
        primary = getattr(contract, "primaryExchange", "") or ""
        if primary != "NASDAQ":
            logger.warning(
                "dropping non-NASDAQ-listed symbol from target: MOC "
                "at close prints on the primary listing's close-cross, "
                "not NASDAQ's — an ISLAND MOC on this name would "
                "mis-print or be rejected",
                extra={
                    "symbol": sym,
                    "primary_exchange": primary,
                    "expected_primary_exchange": "NASDAQ",
                },
            )
            continue
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


def _fill_audit(broker, order_ids: list[str], timeout_seconds: int = 900) -> None:
    """Wait for fills up to ``timeout_seconds`` and append a JSONL entry per
    order to ``live/logs/fills/YYYY-MM-DD.jsonl``.

    The MOC prints at 16:00; we cap the poll so the process doesn't hang
    if TWS never updates one of the terminal states.
    """
    _FILLS_DIR.mkdir(parents=True, exist_ok=True)
    today = config.now_ny().date().isoformat()
    fills_path = _FILLS_DIR / f"{today}.jsonl"
    deadline = time.time() + timeout_seconds
    per_order_wait = max(1.0, timeout_seconds / max(len(order_ids), 1))
    with fills_path.open("a") as fh:
        for oid in order_ids:
            remaining = max(0.0, deadline - time.time())
            wait = min(per_order_wait, remaining)
            if wait <= 0:
                logger.warning("fill audit deadline reached", extra={"order_id": oid})
                break
            try:
                info = broker.wait_for_fill(oid, timeout=wait)
                entry = {
                    "ts": config.now_ny().isoformat(),
                    "order_id": info.order_id,
                    "symbol": info.symbol,
                    "action": info.action,
                    "quantity": info.quantity,
                    "status": info.status,
                    "filled": info.filled,
                    "avg_fill_price": info.avg_fill_price,
                }
            except TimeoutError:
                entry = {
                    "ts": config.now_ny().isoformat(),
                    "order_id": oid,
                    "status": "timeout",
                }
            except Exception as e:
                entry = {
                    "ts": config.now_ny().isoformat(),
                    "order_id": oid,
                    "status": "error",
                    "error": str(e)[:200],
                }
            fh.write(json.dumps(entry, default=str) + "\n")
    logger.info("fills audited", extra={"n_orders": len(order_ids), "path": str(fills_path)})


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

    try:
        if feed is None:
            day_df = pl.DataFrame()
            last_mid_by_sym: dict[str, float] = {}
        else:
            day_df, last_mid_by_sym = _build_live_day_frame(
                feed, universe, reference_stats,
                min_coverage=args.min_universe_coverage,
            )
    except Exception as e:
        logger.exception("feature build failed", extra={"error": str(e)})
        if args.on_feed_failure == "abort":
            if feed is not None:
                feed.stop()
            return 2
        day_df = pl.DataFrame()
        last_mid_by_sym = {}

    if feed is not None:
        feed.stop()

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
        return 4

    # L1: refuse to route an empty book into a rebalance() — with
    # close_unlisted=True (however scoped), an empty target dict would
    # flatten the entire managed book. Belt and braces alongside the
    # coverage check above.
    if not share_targets and not args.dry_run:
        logger.error("empty share_targets — refusing to touch the broker")
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
        return 0

    try:
        broker = _connect_broker(args)
    except Exception as e:
        # Broker refused to connect — most commonly the live-port guard
        # tripping on a paper-mode port mismatch, TWS not running on the
        # expected port, or a bad client_id collision. No orders have been
        # placed; exit with a coded rc rather than a bare traceback.
        logger.exception("broker connect failed", extra={"error": str(e)})
        return 8
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

        # M2 (part 3): JSONL fill audit — best-effort logging only,
        # NOT tied to the rc gate (RRR2). Fills happen at 16:00 close,
        # so we cap the wait at 15 minutes past submit-by to avoid
        # the process lingering past the trading day. The MOC
        # acceptance audit above is the rc-gating check.
        audit_ids = [e["order_id"] for e in entries if e.get("order_id")]
        audit_ids += moc_order_ids
        try:
            _fill_audit(broker, audit_ids, timeout_seconds=900)
        except Exception:
            logger.exception("fill audit failed (non-fatal)")

        # R3 + RR1 + RRR1: distinct rc for the OVERNIGHT-EXPOSURE state.
        # rc=4  → past MOC submit cutoff BEFORE any orders placed (safe
        #         skip; no book).
        # rc=11 → entries submitted but the MOC exit could not fully
        #         cover the resulting book: legs skipped past cutoff,
        #         get_positions failed/returned stale-empty, the
        #         cross-check found uncovered symbols (partial snapshot
        #         / budget-exhausted-with-real-fill / staleness-brownout),
        #         or the MOC acceptance audit found exchange rejections
        #         (submission past ~15:50). The book may be riding
        #         unhedged into the auction — operator must page.
        if exposure:
            # RR4 / RRR1: name the concrete remedy so an operator paged
            # at 15:57 doesn't have to remember the command.
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
    finally:
        broker.disconnect()

    return 0


if __name__ == "__main__":
    sys.exit(main())
