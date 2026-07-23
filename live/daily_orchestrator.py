"""Daily live-trading orchestrator.

Runs once per trading day. Timeline:

  ~09:30 ET  live feed opens; imbalance + tbbo start buffering
   15:50 ET  snapshot buffers, engineer features, run alpha + MVO,
             convert dollar targets to signed share counts, reconcile
             against current IBKR positions
   15:50 ET+ ENTRY batch: per-name LMT sized to ``target - current``,
             priced at the tbbo mid observed in the DECISION-TIME
             snapshot (the same 15:50 ``feed.snapshot()`` that fed
             the MVO, TIME-BOUNDED at the decision moment via
             ``_extract_last_quote_by_sym(max_ts=decision_ts_utc)``
             so the replay path cannot leak the 16:00 closing print
             into the entry price). Same-observation pricing keeps
             the price we solved against and the price we bid at
             consistent — the tradeoff is that the mid is knowingly
             stale by the MVO solve duration, so the staleness
             guard (``DEFAULT_MAX_QUOTE_STALENESS_SECS``) measures
             the tick's age at submission time (wall clock live,
             replay-day decision anchor on replay). Sub-tick
             precision is rounded PASSIVELY per side ($0.01 above
             $1, $0.0001 below): BUY floors DOWN, SELL ceils UP —
             see ``_round_to_tick`` for why passive-per-side is a
             correctness requirement, not a preference. The entry
             is DAY (default ``tif``) and works at the limit UNTIL
             the sweep cancels it just before 15:55 (see
             ``DEFAULT_ENTRY_WORKING_SECS``); this is materially
             longer than the old MKT-era 30s fill-wait and is what
             gives a passive limit a chance to actually cross. A
             symbol whose decision-time tbbo is missing, stale,
             crossed, non-positive, absurdly priced, or wider than
             ``DEFAULT_MAX_ENTRY_REL_SPREAD`` is SKIPPED entirely
             — no entry, no exposure, no MOC (a skipped entry
             carries ``order_id=None`` and ``filled_qty=0`` so both
             the fill-gated dispatcher and the coverage cross-
             check treat it as "did not attempt"). If MORE than
             ``1 - DEFAULT_MIN_ENTRY_SUBMIT_COVERAGE`` of the
             priced-in universe is skipped by the quote guard the
             orchestrator EXITS rc=10 before submitting any
             remaining names — a truncated fragment of a
             risk-balanced solve is worse than no book at all.
             Submitted via ``broker.buy`` / ``broker.sell`` with
             ``order_type="LMT"``, ``limit_price=<numeric>``, and
             the broker default ``tif="DAY"``.
   15:50 ET+ FILL-GATED MOC DISPATCH: a single-threaded polling loop
             concurrently monitors every entry's status. As each entry
             becomes terminal Filled, its opposite-side MOC is
             submitted IMMEDIATELY, sized to the ACTUAL current
             position (fresh ``broker.get_positions()``). An entry
             that fills zero receives NO MOC from this path. A
             staleness check (``baseline + signed_fill == snapshot``)
             defers any lagging symbol to the sweep rather than
             sizing an MOC off a pre-fill position book (round-4
             fix: IB emits execution reports BEFORE position updates,
             so a reducing entry against a nonzero baseline could
             otherwise MOC off the stale +qty and end up short).
   ~15:51    FIRST SWEEP (immediately after dispatcher): cancels any
             entry still non-terminal, verifies dispatcher MOCs are
             still live at the exchange (drops coverage credit for
             any terminal-rejected MOC so the sweep re-fires),
             takes a fresh ``get_positions()``, and issues MOCs for
             every managed-universe symbol whose
             ``held + covered_signed != 0`` (two-sided: TOPS UP an
             under-cover, TRIMS an over-cover). Value: EARLY
             exposure detection — pre-close signals fire in time
             for the operator to run ``live.panic`` before 16:00.
   <=15:55   SECOND SWEEP at the real ``sweep_deadline`` (computed
             dynamically from book size — see
             ``_compute_sweep_deadline``). SLEEP from ~15:51 to the
             deadline is pumped with periodic ``get_positions()``
             calls so the ib_async connection stays alive during
             the wait. On a normal day the second sweep issues
             ZERO orders — its value is confirmation: fresh
             snapshot ~3-4 minutes later has caught up on any IB
             position-propagation lag, re-verifies every prior MOC
             (dispatcher + first sweep) is still live, and re-fires
             for any residual. This is the GUARANTEE — no
             residual/baseline position strands overnight because
             its entry didn't fill today, and no late cancel-race
             fill goes uncovered.
   16:00     NASDAQ closing cross prints.
  <=16:05    POST-CLOSE POSITION POLL: ``broker.get_positions()`` is
             polled with backoff. Any nonzero managed-universe symbol
             after this window = rc=11 (real overnight exposure).
             This is the AUTHORITATIVE exposure verdict; earlier
             signals are early warnings that can still short-circuit
             to rc=11 pre-close so the operator has time to run panic.

The two-leg model captures the strategy's forecasted 15:50 -> 16:00
close-auction move: LMT entries land intraday at the decision-time
tbbo mid, MOC exits lift at the closing print, and the book is flat
overnight. The MOC size is ALWAYS ``|current_qty|`` (measured
immediately before the MOC send) — a partial LMT fill can never
over-flatten into an opposite overnight exposure, and an entry that
never crossed simply leaves the position where it was.

Ground-truth ON EXCHANGE DEADLINES (verified against SEC/Nasdaq/IB):
- NASDAQ MOC ENTRY cutoff: 15:55:00 ET. MOC orders may be entered
  4:00am - 15:55:00 ET. (Nasdaq extended this from 15:50 in the 2018
  rule change, SEC release 34-84454.)
- NASDAQ MOC CANCEL/MODIFY cutoff: 15:50:00 ET.
- NYSE MOC entry cutoff: 15:50:00 ET (different venue, different rule).
- NASDAQ LOC may be entered 15:55-15:58 given a First Reference Price.
- Closing cross runs at 16:00:00.

This strategy targets NASDAQ exclusively (see ``live/nasdaq_routing.py``),
so 15:55 is the load-bearing cutoff.

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
  0   success (both legs sent; post-close position poll confirmed FLAT)
  2   feed abort or feature-build failure (no orders placed)
  3   solver failure (no orders placed)
  4   past MOC submit cutoff BEFORE any orders placed (safe skip)
  5   empty share_targets refusal (no orders placed)
  6   --replay-date + --live refused
  7   --replay-submit without --replay-date
  8   broker connect failed (no orders placed)
  9   --lambda- <= 0 (invalid argument, no orders placed)
  10  LMT entry submit coverage below ``DEFAULT_MIN_ENTRY_SUBMIT_COVERAGE``
      (systemic feed wedge or quote-guard mass-skip; any entries that DID
      land are cancelled before this return so no exposure remains).
  11  OVERNIGHT EXPOSURE POSSIBLE — page the operator. Any of:
        * pre-close signals (EARLY page so the operator has time to
          run panic before 16:00): legs skipped past cutoff (either
          the fill-gated dispatcher's per-tick cutoff refusal OR the
          sweep's own past-cutoff refusal), get_positions failed,
          stale-empty snapshot after fills, the ground-truth
          cross-check found uncovered symbols, the MOC acceptance
          audit found exchange rejections (MOC placed after NASDAQ's
          15:55 cutoff), the sweep's belt-and-braces cancel failed
          for an entry (uncancellable entries may still fill into
          unhedged overnight positions), OR the fill-gated
          dispatcher's per-symbol MOC submission raised at the
          broker (a ``buy``/``sell`` exception may leave the request
          in an indeterminate state at TWS — round-4 fix: this
          signal was previously computed and dropped, reintroducing
          the "neither moc_batch nor skipped" gap that risked a
          double-flatten from the sweep);
        * post-close (16:00-16:05) position poll found a nonzero
          managed-universe symbol still on the book, OR the poll
          itself failed (pessimistic verdict), OR reached its
          deadline without a confirmed FLAT (never got the
          ``DEFAULT_POST_CLOSE_MIN_CONSECUTIVE_FLAT_READS`` needed
          past the ``DEFAULT_POST_CLOSE_POLL_NOT_BEFORE_TIME``
          threshold), OR the poll block itself raised an
          exception (round-3 pessimistic wrap — cannot verify =
          cannot claim flat). This is the AUTHORITATIVE signal
          — it replaces inference and runs for every code path
          that reached the broker (including ``--replay-submit``,
          except a ``--replay-submit`` REHEARSAL that reaches the
          poll BEFORE today's ``DEFAULT_POST_CLOSE_POLL_NOT_BEFORE_TIME``
          wall clock — the poll SKIPS with a WARNING because a
          historical replay's fills cannot be verified against
          today's auction; the operator MUST manually flatten
          the paper account before the next real session).

The pager predicate is ``rc >= 11``: rc=11 is the ONLY code that
implies a live book at close. Everything else in [2, 10] is a safe
pre-order refusal or clean shutdown.

CONCURRENCY NOTE: do NOT run ``python -m live.panic`` against the
same TWS client-id while an orchestrator run is in flight — panic's
``cancel_all()`` will pull the orchestrator's own LMT entries and MOC
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
import math
import os
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from pathlib import Path
from typing import Callable, Optional

import joblib
import numpy as np
import polars as pl

_UTC = timezone.utc

from live import config
from live.nasdaq_routing import ACTIVE_ROUTING
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
            "-> LMT entries -> MOC exits."
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
            "This is NASDAQ's MOC ENTRY cutoff of 15:55:00 ET (SEC "
            "release 34-84454, 2018 rule change). This flag ONLY bounds "
            "the MOC exit batch; the LMT entry batch fires unconditionally "
            "so the operator is never left with a filled entry and no "
            "queued exit. The fill-gated dispatcher fires each MOC "
            "immediately after its entry becomes terminal; the deadline "
            "sweep at ``submit_by - sweep_margin`` (where sweep_margin "
            "scales with book size — see ``_compute_sweep_deadline``) "
            "guarantees every managed-universe held position has an MOC "
            "away before the cutoff. If the cutoff was already past at "
            "orchestrator entry (no orders placed), the process exits "
            "rc=4 (safe skip). If the cutoff trips mid-flow AFTER "
            "entries filled, the process exits rc=11 (OVERNIGHT "
            "EXPOSURE) and the target record lists the open positions."
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
    p.add_argument(
        "--replay-ignore-cutoff", action="store_true", default=False,
        help=(
            "In replay mode, IGNORE the real ``--submit-by`` wall-clock "
            "cutoff (neutralise it to now+1h). Off by default — replay "
            "now HONOURS the real cutoff, so a late-in-the-day rehearsal "
            "correctly surfaces the same rc=4 / rc=11 codes a real run "
            "would. Pass this flag to preserve the earlier "
            "afternoon-rehearsal ergonomics where you deliberately want "
            "to exercise the order path even though ``now`` is past "
            "15:55. Only meaningful with --replay-date."
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

def _extract_last_quote_by_sym(
    tbbo: pl.DataFrame,
    universe: Optional[list[str]] = None,
    max_ts: Optional[datetime] = None,
) -> dict[str, dict]:
    """Extract per-symbol last-observed quote from a tbbo frame.

    Returns ``{sym: {"bid": float, "ask": float, "mid": float,
    "ts": datetime}}`` for every symbol seen in ``tbbo`` (optionally
    filtered to ``universe`` and to ``ts_event <= max_ts``). ``ts``
    is the tbbo record's ``ts_event`` (tz-aware UTC on the live
    path, tz-aware NY on the replay path — the tz mixes cleanly
    because both are anchored to UTC internally by polars). The
    caller (``_validate_quote_for_entry``) converts to UTC before
    the age comparison.

    ``max_ts`` (round-2 look-ahead defence): the LIVE path passes
    None because the feed buffers only what has already arrived.
    The REPLAY path passes the replay-day decision timestamp;
    otherwise the cached parquet delivers the entire trading day
    (including the 16:00 closing print) and ``.last()`` would pick
    a tick from AFTER the 15:50 decision moment. That would price
    LMT entries off the closing cross itself — the exact tick the
    strategy is trying to forecast — turning replay into a
    look-ahead PnL cheat.

    Pre-group filters (round-2 correctness):
      * ``symbol`` in universe (if provided)
      * ``ts_event <= max_ts`` (if provided)
      * ``bid_px_00`` and ``ask_px_00`` both finite AND positive
        AND uncrossed. A name whose LAST tick is a bad print
        (NaN, negative, crossed) but whose second-to-last is
        clean should keep the clean quote — filtering FINITE rows
        BEFORE the last() aggregation is what preserves that. If
        every tick for a symbol is bad, the symbol is naturally
        excluded from the output and the entry gets a
        ``no_usable_quote_at_entry`` skip downstream.

    Aggregation is explicit-and-tie-stable
    (``sort_by("ts_event").last()``): polars 1.40's implicit
    ordering behavior happens to line up today, but pinning the
    intent means a future polars change can't silently return a
    different tick per symbol under duplicate timestamps.

    One definition of "the mid" lives here — ``_build_live_day_frame``
    calls through it once per decision-time snapshot and threads the
    result to both share sizing (``_compute_share_targets``) and LMT
    entry pricing (``_submit_limit_entries``). Broken out for
    testability and to keep the polars aggregation from being
    duplicated inline.
    """
    if tbbo is None or tbbo.is_empty():
        return {}
    if universe is not None:
        tbbo = tbbo.filter(pl.col("symbol").is_in(list(universe)))
        if tbbo.is_empty():
            return {}
    if max_ts is not None:
        # LiveFeed stamps UTC; ReplayFeed converts to
        # America/New_York. Normalise both sides to UTC for the
        # comparison — a polars ``<=`` on datetimes across two
        # different tzs errors out ("could not evaluate '<='
        # comparison"), so we can't rely on tz-implicit
        # broadcasting. The cast is a no-op on the LIVE path.
        if max_ts.tzinfo is None:
            raise ValueError("_extract_last_quote_by_sym: max_ts must be tz-aware")
        max_ts_utc = max_ts.astimezone(_UTC)
        tbbo = tbbo.with_columns(
            pl.col("ts_event").dt.convert_time_zone("UTC")
        ).filter(pl.col("ts_event") <= max_ts_utc)
        if tbbo.is_empty():
            return {}
    # Drop bad prints BEFORE the group_by so one bad final tick
    # does not evict a symbol whose earlier ticks are clean.
    tbbo = tbbo.filter(
        pl.col("bid_px_00").is_finite()
        & pl.col("ask_px_00").is_finite()
        & (pl.col("bid_px_00") > 0)
        & (pl.col("ask_px_00") > 0)
        & (pl.col("ask_px_00") >= pl.col("bid_px_00"))
    )
    if tbbo.is_empty():
        return {}
    last = (
        tbbo
        .with_columns(
            ((pl.col("ask_px_00") + pl.col("bid_px_00")) / 2).alias("mid"),
        )
        .group_by("symbol", maintain_order=False)
        .agg(
            # Explicit ``sort_by("ts_event").last()`` pins the
            # intent: the returned tick is the one with the largest
            # ``ts_event``. Polars 1.40.1's ``.last()`` on an
            # unsorted group_by is documented to return "any" row;
            # today it happens to preserve source order, but that's
            # an implementation detail we shouldn't rely on.
            pl.col("bid_px_00").sort_by("ts_event").last().alias("bid"),
            pl.col("ask_px_00").sort_by("ts_event").last().alias("ask"),
            pl.col("mid").sort_by("ts_event").last().alias("mid"),
            pl.col("ts_event").sort_by("ts_event").last().alias("ts"),
        )
    )
    out: dict[str, dict] = {}
    for row in last.iter_rows(named=True):
        sym = row["symbol"]
        # Categorical dtype -> str
        if not isinstance(sym, str):
            sym = str(sym)
        out[sym] = {
            "bid": row["bid"],
            "ask": row["ask"],
            "mid": row["mid"],
            "ts": row["ts"],
        }
    return out


def _build_live_day_frame(
    feed: LiveFeed,
    universe: list[str],
    reference_stats: pl.DataFrame,
    min_coverage: float = 0.5,
    decision_ts_utc: Optional[datetime] = None,
) -> tuple[pl.DataFrame, dict[str, float], dict[str, datetime], dict[str, dict]]:
    """Snapshot the live feed and produce the single-day feature frame.

    ``min_coverage`` is the fraction of the universe that must appear in
    the alpha-ready ``day_df`` before we return. Anything less and we raise
    ``RuntimeError`` — silent-empty is unacceptable in the live path.

    ``decision_ts_utc`` (round-2 look-ahead defence) bounds the
    tbbo frame at ``ts_event <= decision_ts_utc`` before extracting
    the per-symbol last quote — so the LMT entry pricing sees only
    ticks that would have arrived by the decision moment. On the
    LIVE path pass ``None`` (the feed buffer only holds what has
    arrived); on the REPLAY path pass the replay-day 15:50 anchor
    so the cached whole-day parquet doesn't leak the 16:00 closing
    print into the entry limit price.

    Returns
    -------
    (day_df, last_mid_by_sym, last_mid_ts_by_sym, last_quote_by_sym)
        ``day_df`` has the columns the optimizer's ForecastModel / CostModel /
        ImpactModel need. ``last_mid_by_sym`` is the freshest mid we saw for
        each symbol in the snapshot — used to convert dollar targets to
        shares (falling back to ``entry_price`` when needed).
        ``last_mid_ts_by_sym`` is the ts of the tick that produced the mid,
        threaded into the recorder for ``arrival_ts_utc`` /
        ``arrival_staleness_secs``.
        ``last_quote_by_sym`` is the per-symbol ``{"bid": float, "ask":
        float, "mid": float, "ts": datetime}`` derived from the SAME
        snapshot — the LMT entry batch prices off THIS (not a fresh
        re-snapshot); the staleness guard measures the tick's age at
        submission time. Same observation for optimizer and bid means
        the price we solved against and the price we bid at agree.
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
    # the 15:50 print), the ts is threaded into the trade recorder so
    # ``arrival_ts_utc`` / ``arrival_staleness_secs`` can flag thin
    # names whose last print was minutes ago, AND the full quote
    # (bid/ask/mid/ts) is passed downstream to
    # ``_submit_limit_entries`` so LMT prices are computed off the
    # SAME observation the optimizer used.
    last_quote_by_sym = _extract_last_quote_by_sym(
        tbbo, universe=None, max_ts=decision_ts_utc,
    )
    last_mid_by_sym = {sym: q["mid"] for sym, q in last_quote_by_sym.items()}
    last_mid_ts_by_sym = {sym: q["ts"] for sym, q in last_quote_by_sym.items()}

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
    return day_df, last_mid_by_sym, last_mid_ts_by_sym, last_quote_by_sym


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


def _sleep_until(
    target: datetime,
    poll: float = 5.0,
    pump=None,
    pump_interval: float = 30.0,
) -> None:
    """Block until ``target`` (America/New_York timezone).

    Uses coarse polling so signals and Ctrl-C wake the process quickly.

    ROUND-4 SAFETY FIX (SAFETY reviewer finding 4). ``time.sleep`` does
    NOT service the asyncio event loop; ib_async only pumps inside
    ``ib.sleep()`` (which is what ``Broker.get_positions`` internally
    calls). A long unpumped sleep with the ib_async connection open
    starves the loop while TWS keeps streaming position/order/account
    updates into the socket — the connection can be dropped or wedged,
    and every subsequent ``get_positions`` call would then raise. On
    the rc=0 path that would drive ``_poll_positions_after_close`` into
    its pessimistic EXPOSED branch, spuriously paging rc=11 on a
    genuinely flat book.

    If ``pump`` is supplied, it is invoked every ``pump_interval``
    seconds during the wait to keep the underlying connection alive
    (typically ``lambda: broker.get_positions()`` — that call's
    internal ``ib.sleep(0.5)`` is what actually services the loop, and
    the returned positions double as an early exposure canary). Pump
    exceptions are logged as warnings but do not interrupt the wait —
    a transient TWS blip during a 4-minute sleep must not delay the
    downstream sweep / verdict, and the eventual authoritative check
    (``_poll_positions_after_close``) has its own pessimistic-on-raise
    behaviour.
    """
    last_pump = time.time()
    while True:
        now = config.now_ny()
        remaining = (target - now).total_seconds()
        if remaining <= 0:
            return
        time.sleep(min(remaining, poll))
        if pump is not None and (time.time() - last_pump) >= pump_interval:
            try:
                pump()
            except Exception:
                logger.warning(
                    "pump callback raised during _sleep_until (non-fatal)",
                    exc_info=True,
                )
            last_pump = time.time()


def _wait_for_marks_pumped(
    recorder,
    broker,
    timeout: float,
    pump_interval: float = 30.0,
) -> bool:
    """``recorder.wait_for_marks`` in short slices with a broker pump
    between slices to keep the ib_async connection alive.

    ROUND-4 SAFETY FIX (SAFETY reviewer finding 4). ``wait_for_marks``
    is a ``threading.Event.wait`` — does NOT service the asyncio event
    loop. A ~350s wait right before the authoritative
    ``_poll_positions_after_close`` was the last long unpumped stretch
    in the rc=0 path; on a bad day it could wedge the connection and
    force the poll into its pessimistic-EXPOSED branch, paging rc=11
    on a genuinely flat book.

    Returns True iff the marks thread signalled completion before the
    total timeout. Pump exceptions are logged as warnings; the wait
    continues so a transient TWS blip during this window cannot delay
    the rc verdict.
    """
    if recorder is None:
        return False
    deadline = time.time() + timeout
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            return False
        this_wait = min(pump_interval, remaining)
        try:
            done = recorder.wait_for_marks(timeout=this_wait)
        except Exception:
            logger.warning(
                "recorder.wait_for_marks raised (non-fatal); giving up "
                "on marks wait so post-close poll can proceed",
                exc_info=True,
            )
            return False
        if done:
            return True
        if broker is not None:
            try:
                broker.get_positions()
            except Exception:
                logger.warning(
                    "broker heartbeat during marks wait failed (non-fatal)",
                    exc_info=True,
                )



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
    ``ACTIVE_ROUTING`` code every entry/exit uses so qualification
    surfaces an IB-untradeable name here (drop) instead of at order
    submission (a lost session slot). See ``live/nasdaq_routing.py``
    for the current routing choice and its rationale.
    We swallow individual exceptions and log them; one bad name should
    not kill the batch.

    NASDAQ-primary eligibility filter (ACTIVE)
    ------------------------------------------

    Every qualified contract is inspected for ``primaryExchange ==
    "NASDAQ"``; names failing that check are dropped and logged with
    ``primary_exchange`` set to the observed value so the operator
    can see WHY the name was dropped. Re-enabled 2026-07-21 alongside
    the SMART routing switch — see ``live/nasdaq_routing.py``.

    The filter was previously disabled (rolled back 2026-07-20
    together with the broker's route-survival assertion) after a
    paper replay in which the broker guard rejected 46 of 46
    requested symbols (100% failure, 0 passed) upstream of this
    filter, so this drop's behaviour was never exercised end-to-end
    on that run. It is now re-enabled because under SMART routing
    the NASDAQ-primary premise is load-bearing for the MOC exit leg
    (a SMART MOC on a non-NASDAQ-primary name would route to the
    home exchange's close cross, not NASDAQ's — the wrong auction).

    What this filter **does** buy you: a guarantee that every name
    we send an order for is NASDAQ-primary, so universe drift (the
    broad ``universe.txt`` pool retrain scans does include NYSE /
    ARCA-primary names) can no longer silently introduce a name
    whose MOC belongs to another exchange's close cross.

    What it does **not** buy you: verification that IB's SMART
    router actually routes an MOC for a NASDAQ-primary name to the
    NASDAQ close cross specifically (rather than a different
    NASDAQ-eligible venue's post-close mechanism). That is a
    separate, still-unverified question — see the pre-live paper-
    session gate in the ``live/nasdaq_routing.py`` docstring
    (confirm every ``leg="moc"`` trade-recorder row has
    ``venue in {"ISLAND", "NASDAQ"}``).

    Expected filter behaviour on today's tiered universes:

    * ``top_49_universe.txt`` — **TWS-observed** no-op: 49/49
      qualify with ``primaryExchange == "NASDAQ"`` (recorded in
      ``ibkr_broker/src/ibkr_broker/broker.py:370-372`` from a real
      paper-TWS session).
    * ``top_100_universe.txt`` (the previously active universe;
      ``top_49_universe.txt`` is the currently active universe as of
      2026-07-21) — **expected but not yet observed on this
      system**: 99 of the 100 should qualify with all 99
      NASDAQ-primary (ITCI fails IB qualification outright and is
      dropped by the still-active try/except above, never reaching
      this filter). No paper-TWS observation exists yet for the
      100-name set; the one paper run that would have exercised it
      (2026-07-20) rejected 46 of 46 requested symbols at the broker
      guard upstream of this filter. The ``contracts pre-qualified``
      log line below is the operator signal — the pre-live gate in
      ``live/nasdaq_routing.py`` calls for ``n_ok == n_requested``
      (or ``n_ok == n_requested - 1`` for the ITCI drop) on the
      first paper session to settle the claim.

    Its value is future universe-drift protection.
    """
    ok: set[str] = set()
    for sym in symbols:
        try:
            # Capture the qualified contract so the primary-exchange
            # filter below can read its ``primaryExchange`` field. The
            # call itself also detects a name IB cannot qualify (bad
            # ticker, no market-data subscription, etc.) — that error
            # path drops the name via the ``except`` branch.
            contract = broker._make_contract(sym, exchange=ACTIVE_ROUTING)
        except Exception as e:
            logger.warning(
                "contract qualification failed",
                extra={"symbol": sym, "error": str(e)[:200]},
            )
            continue
        # NASDAQ-primary eligibility: drop names whose primary listing
        # is not NASDAQ. Their MOC prints on their home exchange's
        # close-cross, not NASDAQ's — the strategy is calibrated to
        # the NASDAQ close auction, so an off-auction MOC fill is
        # uncorrelated with the signal we sized. The filter runs
        # regardless of routing choice (SMART or ISLAND): under SMART
        # the router would send the MOC to the primary listing's
        # close cross (i.e. the wrong auction) for a non-NASDAQ-
        # primary name; under ISLAND direct routing IB may reject
        # the MOC or mis-print it. We include ``primary_exchange``
        # in the log so the operator can see WHY the name was
        # dropped.
        primary = getattr(contract, "primaryExchange", "") or ""
        if primary != "NASDAQ":
            logger.warning(
                "dropping non-NASDAQ-listed symbol from target: MOC "
                "at close prints on the primary listing's close-cross, "
                "not NASDAQ's — a MOC on this name would print on the "
                "wrong auction (or be rejected under a direct route)",
                extra={
                    "symbol": sym,
                    "primary_exchange": primary,
                    "expected_primary_exchange": "NASDAQ",
                },
            )
            continue
        ok.add(sym)
    # A drop is financially fail-safe (the missing name just doesn't
    # trade, and any held position still flattens because the MOC leg
    # is sized off ``managed_universe`` — the intersection of the
    # active universe with ``get_positions``, NOT the qualified set),
    # but a smaller-than-expected book still deserves an operator
    # signal. Escalate to WARNING when any name dropped so the
    # aggregate change is visible in the log stream; the per-symbol
    # WARNINGs above still carry the WHY (IB unqualified vs. non-
    # NASDAQ-primary). On today's universes this is expected to be
    # an INFO no-op — see the "runtime no-op" claim in
    # ``live/nasdaq_routing.py`` (still awaiting paper-TWS
    # verification on top_100).
    n_dropped = len(symbols) - len(ok)
    log_fn = logger.warning if n_dropped > 0 else logger.info
    log_fn(
        "contracts pre-qualified",
        extra={
            "n_ok": len(ok),
            "n_requested": len(symbols),
            "n_dropped": n_dropped,
        },
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


def _round_to_tick(price: float, side: str) -> float:
    """Round a price to a valid US-equity tick, PASSIVELY per side.

    Rule NMS 612: prices >= $1.00 quote in $0.01 increments; prices
    below $1.00 quote in $0.0001 increments. IB rejects sub-tick
    limit prices — every limit price we submit MUST be rounded here.

    Rounding direction is PASSIVE:
      * ``side == "BUY"``  → ``floor(price / tick) * tick`` (round DOWN)
      * ``side == "SELL"`` → ``ceil(price / tick) * tick``  (round UP)

    Why passive-per-side (round-2 correctness fix). Half-up rounding
    on a common one-cent spread puts the mid exactly on a half-penny
    and rounds UP for both sides — a BUY at the ask (marketable,
    fills instantly) and a SELL half a cent above the ask (rarely
    fills). That yields a systematically net-long realized book the
    covariance never priced. Passive rounding keeps BOTH sides at or
    inside the mid: buys post at the near-side bid, sells at the
    near-side ask, both are passive limits and either both fill or
    both rest — no directional bias in the realized book.

    Representation: use ``Decimal`` for the tick-count arithmetic so
    ``price / tick`` cannot land at ``149.99999999999997`` and floor
    to the wrong tick. Then cast back to ``float`` for the IB wire
    format (IB's Python client takes floats). Snap the returned
    float to the tick's decimal precision (derived from the tick
    itself, not a hard-coded ``if tick == 0.01`` fork).
    """
    if side not in ("BUY", "SELL"):
        raise ValueError(f"_round_to_tick: side must be BUY or SELL, got {side!r}")
    tick = (
        config.ENTRY_TICK_ABOVE_DOLLAR
        if abs(price) >= 1.0
        else config.ENTRY_TICK_BELOW_DOLLAR
    )
    # Decimal keeps price/tick exact enough that the floor/ceil
    # decision is unaffected by binary-float noise. All-Decimal
    # arithmetic on rounded string inputs is exact; the final cast
    # back to float is lossless for penny-grid prices representable
    # in double precision (which every US-equity price under $10T
    # is).
    p = Decimal(str(price))
    t = Decimal(str(tick))
    if side == "BUY":
        # Round DOWN toward zero for positive prices; equivalent to
        # floor for the price >= 0 case that every real quote hits.
        rounded = (p / t).to_integral_value(rounding=ROUND_DOWN) * t
    else:  # SELL
        rounded = (p / t).to_integral_value(rounding=ROUND_UP) * t
    # Derive the display precision from the tick itself — e.g. tick
    # 0.01 → 2 decimals, tick 0.0001 → 4. Uses ``normalize`` +
    # exponent so a future tick change (e.g. NMS half-cent proposals)
    # does not require another `if` branch here.
    tick_exp = t.normalize().as_tuple().exponent
    ndigits = max(0, -tick_exp) if isinstance(tick_exp, int) else 4
    return float(round(float(rounded), ndigits))


def _validate_quote_for_entry(
    quote: Optional[dict],
    now_utc: datetime,
    max_staleness_secs: float,
    max_rel_spread: float,
) -> tuple[bool, Optional[str], Optional[float]]:
    """Apply the LMT-entry quote guards.

    Returns ``(ok, reason, mid)``.  ``ok=False`` means the caller must
    SKIP this symbol (no entry, no MOC). ``reason`` is a short tag
    logged and stored on the entry record. On success ``mid`` is the
    raw (unrounded) mid; the caller rounds via ``_round_to_tick``.

    Rejects, in this order:
      * missing quote (``quote is None``)
      * missing bid / ask / mid / ts
      * non-numeric or non-finite bid / ask / mid (catches
        Databento's ``UNDEF_PRICE`` sentinel, which pretty-scales to
        NaN today — see the ``ENTRY_MAX_LIMIT_PRICE`` guard for the
        finite-but-huge variant)
      * non-positive bid or ask (bad tick / feed glitch)
      * crossed or locked (ask < bid — the mid is nonsense)
      * mid missing / non-positive
      * absurdly large bid or ask (``> ENTRY_MAX_LIMIT_PRICE`` —
        catches the "scaled sentinel" case where ``pretty_ref_price``
        of Databento's INT64_MAX arrives as ~9.22e9 rather than NaN,
        which would otherwise pass every guard: finite, positive,
        uncrossed, rel_spread==0)
      * stale (``now - ts > max_staleness_secs``)
      * wide (``(ask - bid) / mid > max_rel_spread``)

    ``now_utc`` is compared to ``ts_event`` in UTC. On the live path
    this is ``datetime.now(_UTC)``; on the replay path the caller
    passes the replay-day decision timestamp so a historical tbbo
    doesn't look "months stale" against today's wall clock. A
    NEGATIVE age (``ts_utc > now_utc``) indicates clock skew and is
    logged as a warning; small negative ages (< 1s) are treated as
    fresh; larger backward skews reject as ``quote_stale`` so a
    scrambled clock cannot admit unbounded look-ahead.
    """
    if quote is None:
        return False, "no_usable_quote_at_entry", None
    bid = quote.get("bid")
    ask = quote.get("ask")
    mid = quote.get("mid")
    ts = quote.get("ts")
    if bid is None or ask is None or mid is None or ts is None:
        return False, "quote_missing_fields", None
    try:
        bid_f = float(bid)
        ask_f = float(ask)
        mid_f = float(mid)
    except (TypeError, ValueError):
        return False, "quote_non_numeric", None
    if not (math.isfinite(bid_f) and math.isfinite(ask_f)
            and math.isfinite(mid_f)):
        return False, "quote_non_finite", None
    if bid_f <= 0 or ask_f <= 0:
        return False, "quote_non_positive", None
    if ask_f < bid_f:
        return False, "quote_crossed", None
    if mid_f <= 0:
        return False, "quote_non_positive", None
    # Absolute price sanity bound. Guards against Databento
    # UNDEF_PRICE sentinel (scaled to ~9.22e9) leaking through the
    # non-finite check. Every US-listed equity trades well below
    # ENTRY_MAX_LIMIT_PRICE; Berkshire A around $700k is the highest
    # single-share price in practice.
    if (bid_f > config.ENTRY_MAX_LIMIT_PRICE
            or ask_f > config.ENTRY_MAX_LIMIT_PRICE):
        return False, "quote_absurd_price", None
    # Staleness: coerce ts to UTC for comparison against now_utc.
    try:
        if ts.tzinfo is None:
            # Should never happen — LiveFeed stamps UTC, ReplayFeed
            # converts to America/New_York (also tz-aware). But be
            # defensive.
            return False, "quote_naive_ts", None
        ts_utc = ts.astimezone(_UTC)
    except Exception:
        return False, "quote_bad_ts", None
    age_secs = (now_utc - ts_utc).total_seconds()
    # Clock-skew defence. Negative age = ts is IN THE FUTURE relative
    # to ``now_utc``; on the live path that's a wall-clock skew,
    # on the replay path that's a genuine look-ahead attempt.
    # Tolerate up to 1s of "future" (feed timestamps can beat the
    # local clock by tens of ms on a busy loop); anything larger is
    # treated as stale so we never submit a limit priced off a tick
    # that came after "now".
    if age_secs < -1.0:
        logger.warning(
            "quote_stale (clock skew): tick timestamp is in the "
            "future relative to now_utc",
            extra={
                "age_secs": age_secs,
                "now_utc": now_utc.isoformat(),
                "quote_ts_utc": ts_utc.isoformat(),
            },
        )
        return False, "quote_stale", None
    if age_secs > max_staleness_secs:
        return False, "quote_stale", None
    rel_spread = (ask_f - bid_f) / mid_f
    if rel_spread > max_rel_spread:
        return False, "quote_wide_spread", None
    return True, None, mid_f


def _submit_limit_entries(
    broker,
    targets: dict[str, int],
    current: dict[str, int],
    managed_universe: set[str],
    last_quote_by_sym: dict[str, dict],
    now_utc: datetime,
    max_staleness_secs: float,
    max_rel_spread: float,
    replay_mode: bool = False,
) -> list[dict]:
    """ENTRY batch: per-name LMT priced at the decision-time tbbo
    mid (rounded PASSIVELY per side), sized to ``target - current``.

    Each entry is a plain LMT (not IB MIDPRICE / pegged, not IOC —
    a standard DAY LMT with a numeric ``limit_price``) at the mid of
    the tbbo quote observed in the SAME 15:50 snapshot the MVO
    consumed (see ``_build_live_day_frame``), passively rounded to
    a valid US-equity tick (BUY floors DOWN, SELL ceils UP — see
    ``_round_to_tick``). Routed via ``broker.buy`` / ``broker.sell``
    with ``order_type="LMT"``, ``limit_price=<numeric>``,
    ``exchange=ACTIVE_ROUTING``, and the primitive default
    ``tif="DAY"`` (the DAY is what actually binds — the entry works
    at the limit until the sweep cancels it just before 15:55; see
    ``DEFAULT_ENTRY_WORKING_SECS``).

    A stuck LMT that never crosses (much more likely than under the
    old MKT regime) is bounded by the fill-gated dispatcher
    (``_fill_gated_moc_dispatch``) and the deadline sweep
    (``_sweep_moc_deadline``): the dispatcher polls entries every
    ``poll_interval_secs`` and stops waiting at ``sweep_deadline``,
    and the sweep then cancels any entry still non-terminal via
    ``broker.cancel`` BEFORE it sizes its counter-flatten from a
    fresh ``get_positions``. So an LMT that expires unfilled leaves
    no position and no MOC, and one that partially filled has its
    MOC sized to the ACTUAL filled quantity — the two-leg safety
    invariant is unchanged.

    Fires unconditionally with respect to the MOC submit-by cutoff —
    an entered position with no exit is bad, but so is refusing to
    enter because we're worried about the exit; the operator needs
    to know either way (the MOC-skip case is handled by
    ``_fill_gated_moc_dispatch`` firing per-symbol and
    ``_sweep_moc_deadline`` catching everything left).

    ----
    Quote guards (``_validate_quote_for_entry``).
    ----
    A symbol whose ``last_quote_by_sym`` entry is missing, stale
    (older than ``max_staleness_secs``), crossed / locked (ask <
    bid), non-positive, or wider than ``max_rel_spread`` is SKIPPED:
    the returned record has ``order_id=None`` and
    ``error=<reason>`` (e.g. ``"quote_stale"``,
    ``"quote_wide_spread"``). ``_fill_gated_moc_dispatch`` treats
    ``order_id is None`` as ``status="not_submitted"``,
    ``filled_qty=0``, so no MOC ever fires for a skipped symbol.
    The coverage cross-check skips entries with ``order_id is None``
    entirely, so a skipped LMT does not synthesise phantom expected
    exposure. **No MKT fallback** — the operator's policy is that
    no entry = no exposure, and a fallback would defeat the whole
    point of limiting downside from a bad quote.

    Out-of-universe symbols (``sym in current but not
    managed_universe``) are pinned to their current qty upstream, so
    they diff to zero here and we never touch them.

    Returns a list of dicts, one per attempted entry, with keys:
      symbol, side, requested_qty, order_id (str|None), error (str|None),
      limit_price (float|None). ``limit_price`` is populated on skips
      too when the mid was computable but rejected downstream — this
      is None for a "no quote at all" skip, and populated (with the
      unrounded mid) for a "quote arrived but was crossed / stale /
      etc." skip so post-trade analysis can see what we saw.
    Downstream, ``_fill_gated_moc_dispatch`` populates ``filled_qty``
    and ``status`` from ``broker.get_order_status``; the sweep does
    the same for any entry the dispatcher gave up on.
    """
    entries: list[dict] = []
    n_skipped_by_reason: dict[str, int] = {}
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
        quote = last_quote_by_sym.get(sym)
        # Re-sample "now" per symbol on the LIVE / paper path so a
        # slow first-symbol submission doesn't unfairly stale the
        # last few. On REPLAY the caller pinned ``now_utc`` to the
        # replay-day decision anchor and we want to preserve that
        # exact reference across every symbol (a wall-clock sample
        # here would defeat the whole point of the replay-relative
        # staleness fix).
        now_per_sym = now_utc if replay_mode else datetime.now(_UTC)
        ok, reason, mid = _validate_quote_for_entry(
            quote, now_per_sym, max_staleness_secs, max_rel_spread,
        )
        if not ok:
            n_skipped_by_reason[reason or "unknown"] = (
                n_skipped_by_reason.get(reason or "unknown", 0) + 1
            )
            logger.warning(
                "LMT entry skipped — quote guard rejected",
                extra={
                    "symbol": sym, "side": side, "qty": qty,
                    "reason": reason,
                    "quote_bid": quote.get("bid") if quote else None,
                    "quote_ask": quote.get("ask") if quote else None,
                    "quote_mid": quote.get("mid") if quote else None,
                    "quote_ts": (
                        quote.get("ts").isoformat()
                        if quote and quote.get("ts") is not None
                        else None
                    ),
                },
            )
            entries.append({
                "symbol": sym, "side": side, "requested_qty": qty,
                "order_id": None, "error": reason,
                "limit_price": (
                    float(quote["mid"]) if quote and quote.get("mid") is not None
                    else None
                ),
            })
            continue
        limit_price = _round_to_tick(float(mid), side)
        try:
            # Routing pulled from live/nasdaq_routing.py::ACTIVE_ROUTING.
            # Under SMART, the entry LMT is exposed to the national
            # best bid/offer rather than being confined to the NASDAQ
            # top-of-book — see the module docstring for the rationale.
            if side == "BUY":
                oid = broker.buy(
                    sym, qty,
                    order_type="LMT",
                    limit_price=limit_price,
                    exchange=ACTIVE_ROUTING,
                )
            else:
                oid = broker.sell(
                    sym, qty,
                    order_type="LMT",
                    limit_price=limit_price,
                    exchange=ACTIVE_ROUTING,
                )
        except Exception as e:
            logger.warning(
                "LMT entry submission failed",
                extra={"symbol": sym, "side": side, "qty": qty,
                       "limit_price": limit_price,
                       "error": str(e)[:200]},
            )
            entries.append({
                "symbol": sym, "side": side, "requested_qty": qty,
                "order_id": None, "error": str(e)[:200],
                "limit_price": limit_price,
            })
            continue
        logger.info(
            "LMT entry placed",
            extra={
                "symbol": sym, "side": side, "qty": qty,
                "order_id": oid, "limit_price": limit_price,
                "mid_raw": float(mid),
            },
        )
        entries.append({
            "symbol": sym, "side": side, "requested_qty": qty,
            "order_id": oid, "error": None,
            "limit_price": limit_price,
        })
    if n_skipped_by_reason:
        logger.warning(
            "LMT entry batch: some symbols skipped by quote guard",
            extra={"skipped_by_reason": n_skipped_by_reason},
        )
    return entries


# Order statuses considered "terminal" for the entry-fill wait. Matches
# ``ib_async.order.OrderStatus.DoneStates`` semantically (the broker uses
# that set inside ``wait_for_fill``); we duplicate the check here so main
# can decide which entries still need an explicit cancel.
_TERMINAL_ORDER_STATES: frozenset[str] = frozenset({
    "Filled", "Cancelled", "ApiCancelled", "Inactive",
})


def _cancel_non_terminal_entries(broker, entries: list[dict]) -> list[str]:
    """ORPHANED as of round 2 — superseded by the belt-and-braces cancel
    step inside ``_sweep_moc_deadline``. Retained for reference so a
    reader who chases the module's historical execution model still
    finds the semantics, and so a future rollback can restore the
    two-phase model with a one-line call-site edit. No main()
    call site exercises this function today.

    Cancel every entry whose status is not in a terminal state.

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
    """ORPHANED as of round 2 — superseded by
    ``_fill_gated_moc_dispatch``, which polls entries concurrently
    (single-threaded fast poll) and fires each opposite-side MOC the
    moment the entry becomes terminal. Retained for reference. No
    main() call site exercises this function today; the round-2 sweep
    deadline replaces the ``submit_by - safety_buffer_secs`` bound
    this function computed.

    Block until each submitted entry order is terminal, bounded by a
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
                "entry fill wait timed out",
                extra={"order_id": oid, "symbol": e.get("symbol"),
                       "timeout": this_wait},
            )
            e["filled_qty"] = 0
            e["status"] = "timeout"
        except Exception as ex:
            logger.warning(
                "entry status poll failed",
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
    """ORPHANED as of round 2 — superseded by the pair
    ``_fill_gated_moc_dispatch`` (per-symbol MOC as each entry fills) +
    ``_sweep_moc_deadline`` (deadline top-up sized from a fresh
    get_positions). Retained for reference. No main() call site
    exercises this function today. The signature-refuses-target-side-
    args safety invariant it documented is now enforced identically by
    ``_submit_single_moc`` and ``_sweep_moc_deadline``.

    EXIT batch: MOC counter-orders sized to the ACTUAL current position.

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
    snapshotted position book. A partial LMT fill therefore cannot cause
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
        # exit rc=0 — with the ENTIRE LMT-filled book riding the auction
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
            # Routing pulled from live/nasdaq_routing.py::ACTIVE_ROUTING.
            # SMART on a NASDAQ-primary name still prints the MOC on
            # the NASDAQ close cross (each name routes to its own
            # primary close auction), so the same-auction contract the
            # strategy needs is preserved. See the module docstring.
            if side == "BUY":
                oid = broker.buy(sym, qty, order_type="MOC",
                                 exchange=ACTIVE_ROUTING)
            else:
                oid = broker.sell(sym, qty, order_type="MOC",
                                  exchange=ACTIVE_ROUTING)
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


def _compute_sweep_deadline(
    submit_by: datetime,
    n_managed_symbols: int,
) -> tuple[datetime, float]:
    """Return ``(sweep_deadline, sweep_margin_secs)``.

    The sweep runs before ``submit_by`` (NASDAQ MOC entry cutoff, default
    15:55:00 ET) with enough headroom for its full worst-case cost:

      * two ``get_positions()`` calls (each ~0.5-0.6s of hard-coded
        ib_async sleep — one for the sweep's own snapshot and one
        residual buffer for the dispatcher's final tick)
      * one ``get_order_status`` call per entry (belt-and-braces cancel
        pass)
      * one ``placeOrder`` per not-already-covered symbol (~0.13s)

    Formula:
        margin = max(SWEEP_MIN_MARGIN_SECS,
                     n * (SWEEP_PER_ORDER_SECS + SWEEP_PER_STATUS_SECS)
                     + 2 * SWEEP_GET_POSITIONS_SECS
                     + SWEEP_FIXED_OVERHEAD_SECS)

    Scales with book size so the same code path works at 49 names today
    (floor: 15s) and 500 names later (~121s) without a manual tuning knob.
    """
    per_order = config.SWEEP_PER_ORDER_SECS
    per_status = config.SWEEP_PER_STATUS_SECS
    get_positions = config.SWEEP_GET_POSITIONS_SECS
    overhead = config.SWEEP_FIXED_OVERHEAD_SECS
    floor = config.SWEEP_MIN_MARGIN_SECS
    margin = max(
        floor,
        n_managed_symbols * (per_order + per_status)
        + 2 * get_positions
        + overhead,
    )
    deadline = submit_by - timedelta(seconds=margin)
    return deadline, margin


def _submit_single_moc(
    broker,
    symbol: str,
    current_qty: int,
    submit_by: datetime,
) -> tuple[Optional[dict], str]:
    """Submit ONE opposite-side MOC for ``symbol`` sized to ``|current_qty|``.

    Signature deliberately admits no target-side data — only the ACTUAL
    ``current_qty`` (from a fresh ``broker.get_positions()`` snapshot)
    can size this leg. Same structural invariant every LEG-2 sender
    upholds — a refactor that added a target argument must be
    rejected at review. Additionally, ``submit_by`` is REQUIRED so
    every submission is guarded against the NASDAQ MOC entry cutoff:
    a submit past ``submit_by`` is refused BEFORE ``placeOrder`` is
    called, so we never send a MOC the exchange will silently reject
    past 15:55.

    Returns ``(leg_dict_or_None, outcome)``. ``outcome`` is one of:
      * ``"ok"``          — leg was sent; ``leg_dict`` populated.
      * ``"skipped_cutoff"`` — wall clock is past ``submit_by``;
        nothing sent. Caller must treat as EXPOSURE for that symbol.
      * ``"skipped_zero"`` — ``current_qty`` was 0; nothing to
        flatten (safe).
      * ``"error"``       — broker raised on submission. Caller must
        treat as EXPOSURE.
    """
    if current_qty == 0:
        return None, "skipped_zero"
    if config.now_ny() > submit_by:
        logger.error(
            "MOC exit REFUSED — wall clock past --submit-by cutoff. "
            "POSITION LEFT OPEN.",
            extra={"symbol": symbol, "current_qty": current_qty,
                   "submit_by": submit_by.isoformat(),
                   "now": config.now_ny().isoformat()},
        )
        return None, "skipped_cutoff"
    qty = abs(int(current_qty))
    side = "SELL" if current_qty > 0 else "BUY"
    try:
        if side == "BUY":
            oid = broker.buy(symbol, qty, order_type="MOC",
                             exchange=ACTIVE_ROUTING)
        else:
            oid = broker.sell(symbol, qty, order_type="MOC",
                              exchange=ACTIVE_ROUTING)
    except Exception as e:
        logger.warning(
            "MOC exit submission failed",
            extra={"symbol": symbol, "side": side, "qty": qty,
                   "error": str(e)[:200]},
        )
        return None, "error"
    signed_qty = qty if side == "BUY" else -qty
    logger.info(
        "MOC exit placed",
        extra={"symbol": symbol, "side": side, "qty": qty,
               "current_qty": current_qty, "order_id": oid},
    )
    return {
        "oid": oid, "symbol": symbol, "side": side,
        "qty": qty, "signed_qty": signed_qty,
    }, "ok"


def _snapshot_managed_positions(
    broker,
    managed_universe: set[str],
) -> Optional[dict[str, int]]:
    """One ``broker.get_positions()`` call → managed-universe qty map.

    Returns ``None`` iff ``get_positions()`` raised (positions
    UNKNOWN — caller must escalate to the exposure path). Otherwise a
    fresh ``{symbol: signed_qty}`` for every managed-universe symbol
    that has a nonzero row. Aggregates duplicate rows the same way
    ``_sweep_moc_deadline`` does (multi-account / update-race).

    Consolidating N per-symbol ``_current_position_for`` calls into ONE
    per-tick call is the round-2 fix for the dispatcher's O(N)
    hard-coded 0.5s sleep from ``broker.get_positions()`` — at 500
    names the earlier per-symbol pattern would have added 250s of pure
    sleep to the dispatcher window and blown the 15:55 cutoff.
    """
    try:
        positions = broker.get_positions()
    except Exception as e:
        logger.error(
            "MOC exit: get_positions FAILED during fill-gated dispatch — "
            "POSITIONS MAY BE LIVE INTO THE CLOSE, OPERATOR MUST "
            "MANUALLY FLATTEN",
            extra={"error": str(e)[:200]},
        )
        return None
    agg: dict[str, int] = {}
    for p in positions:
        if p.symbol not in managed_universe:
            continue
        agg[p.symbol] = agg.get(p.symbol, 0) + int(p.quantity)
    return agg


def _fill_gated_moc_dispatch(
    broker,
    entries: list[dict],
    managed_universe: set[str],
    sweep_deadline: datetime,
    submit_by: datetime,
    baseline: Optional[dict[str, int]] = None,
    poll_interval_secs: float = 0.25,
    max_wait_secs: Optional[float] = None,
) -> tuple[list[dict], dict[str, int], list[str], list[str], bool]:
    """CONCURRENTLY monitor entry fills and fire each MOC as soon as
    its entry becomes terminal.

    Design: single-threaded fast-polling loop rather than a
    ThreadPoolExecutor of ``wait_for_fill`` calls. ib_async is NOT
    thread-safe, so multi-thread wait calls would race the event loop.
    Instead, iterate every pending entry once per tick, read its
    terminal-state via ``broker.get_order_status`` (a local self._trades
    dict read — O(N) microseconds per call, not a network round-trip),
    and if terminal Filled with ``filled > 0``, immediately submit the
    counter-side MOC sized from a per-tick ``broker.get_positions()``
    snapshot for that symbol. An entry that reaches terminal state
    with ``filled == 0`` (Cancelled, Inactive, or Filled-zero) receives
    NO MOC from this path — its position stays flat.

    ROUND-2 SCALABILITY FIX. Earlier round-1 code called
    ``broker.get_positions()`` ONCE PER FILLED SYMBOL inside the inner
    loop. ``ibkr_broker.Broker.get_positions`` contains a hard-coded
    ``self._ib.sleep(0.5)``, so N calls costs ~0.5*N seconds regardless
    of anything the strategy does. At the operator's planned 500-name
    book that was 250s of pure sleep, blowing the 15:55 cutoff before
    the sweep even ran. The new pattern is:

      * take ONE ``broker.get_positions()`` at the top of each tick
        via ``_snapshot_managed_positions`` — cost 0.5s/tick, not
        0.5s/name.
      * index that snapshot by symbol; per-terminal-entry lookups are
        O(1).
      * check the wall clock against the ``effective_deadline`` INSIDE
        the inner symbol loop as well as at the top of the tick, so a
        single long pass cannot silently overrun the sweep deadline.

    Concurrency in the operator's sense: each symbol's MOC fires as
    soon as its entry becomes terminal, not after all N entries are
    checked.

    The loop exits when either
    (a) every entry has reached terminal state (Filled or otherwise), or
    (b) the wall clock crosses ``sweep_deadline``.

    In case (b) the remaining non-terminal entries are LEFT for the
    caller's sweep to cancel and reconcile.

    Every ``_submit_single_moc`` call carries ``submit_by`` so a slow
    fill that pushed us past the cutoff mid-tick refuses to submit
    (returning outcome ``"skipped_cutoff"``) rather than firing a MOC
    the exchange will silently reject.

    ROUND-4 SAFETY FIX (SAFETY reviewer finding 2). ``baseline`` is
    the pre-entry position book ``{symbol: signed_qty}`` from the
    orchestrator's initial ``broker.get_positions()`` call. Real IB
    emits the execution report BEFORE the position update, so a
    dispatcher tick that observes ``status=Filled`` may still read the
    PRE-fill position quantity. With a nonzero baseline (residual
    carried in from a prior day, or a manual position that pre-dates
    today's run), a REDUCING entry against that baseline would size
    an MOC to the STALE pre-fill position — a SELL 100 entry against
    baseline +100 filling to 0 would still see current=+100, submit
    SELL MOC 100, and end up with a naked SHORT 100 overnight because
    the true post-fill position was 0. Fix: for each newly-terminal
    entry, compute the deterministic expected post-fill quantity
    ``expected = baseline.get(sym, 0) + (filled if BUY else -filled)``
    and refuse to submit an MOC when ``tick_snapshot[sym] != expected``.
    The symbol stays in ``pending`` so the sweep (which snapshots
    later, giving IB more time to catch up) re-verifies. This is NOT
    the forbidden ``target - current`` sizing — the MOC is still sized
    to the observed snapshot when it agrees with expectations; we only
    use ``baseline + signed_fill`` as a STALENESS CHECK, and the same
    arithmetic is what ``_coverage_cross_check`` treats as ground
    truth. ``baseline=None`` (default) skips the check — used only by
    direct unit tests where the caller doesn't have a baseline handy.
    Main() ALWAYS passes it.

    Returns ``(moc_batch, covered_qty, uncovered_symbols,
    skipped_past_cutoff, positions_unknown)``:

    * ``moc_batch`` — MOC legs the fill-gated path submitted.
    * ``covered_qty`` — ``{symbol: signed_qty_MOC_sent}``. The sweep
      consumes this to compute a residual per symbol (see
      ``_sweep_moc_deadline``) — a fill-gated MOC that under-sized
      relative to the true post-tick position gets a top-up from the
      sweep instead of being silently skipped.
    * ``uncovered_symbols`` — symbols whose entry became terminal
      Filled but whose ``_submit_single_moc`` returned ``"error"``
      (broker ``buy``/``sell`` raised at submission time — the API
      call itself failed, not the sizing snapshot). Snapshot failure
      is signalled separately via ``positions_unknown``. Feeds
      pre-close exposure directly (rc=11) because a submission
      exception can leave the request in an indeterminate state at
      TWS — some paths accept then raise the read timeout AFTER the
      order was queued at the exchange, so a bare "assume nothing
      happened" would defer to the sweep and risk a double-flatten.
    * ``skipped_past_cutoff`` — symbols whose ``_submit_single_moc``
      was refused because the wall clock crossed ``submit_by``. Feeds
      the pre-close exposure signal directly (rc=11).
    * ``positions_unknown`` — True if any ``get_positions()`` raised
      during dispatch. Callers should treat this as an early exposure
      signal.

    Mutates each entry in place: sets ``filled_qty`` and ``status``
    from ``get_order_status`` when known so ``_coverage_cross_check``
    can consume the same rows.
    """
    if baseline is None:
        baseline = {}
    if not entries:
        return [], {}, [], [], False

    pending = {
        e["order_id"]: e
        for e in entries
        if e.get("order_id") is not None
    }
    for e in entries:
        if e.get("order_id") is None:
            e["filled_qty"] = 0
            e["status"] = "not_submitted"

    moc_batch: list[dict] = []
    covered_qty: dict[str, int] = {}
    uncovered: list[str] = []
    skipped_past_cutoff: list[str] = []
    positions_unknown = False

    # Effective deadline. Under the LMT-entry design the dispatcher
    # WORKS UNTIL THE SWEEP DEADLINE by default
    # (``DEFAULT_ENTRY_WORKING_SECS is None``) — a passive limit that
    # hasn't crossed by 15:50:40 is not a wedged submission; it's a
    # limit that hasn't crossed YET, and the closing ramp routinely
    # crosses limits set at 15:50 many minutes later. Capping this at
    # the MKT-era 30s would have the dispatcher walk away and the
    # sweep cancel every LMT before it had a chance to fill. The
    # sweep still cancels any still-non-terminal entry before 15:55,
    # so working through the ramp adds no exposure risk. Callers
    # (dispatcher tests) can pass an explicit ``max_wait_secs`` to
    # cap the wait for a fast-test bound.
    if max_wait_secs is None:
        _cfg_working_secs = getattr(
            config, "DEFAULT_ENTRY_WORKING_SECS", None,
        )
        if _cfg_working_secs is None:
            effective_deadline = sweep_deadline
        else:
            effective_deadline = min(
                sweep_deadline,
                config.now_ny() + timedelta(seconds=float(_cfg_working_secs)),
            )
    else:
        effective_deadline = min(
            sweep_deadline,
            config.now_ny() + timedelta(seconds=max_wait_secs),
        )

    while pending:
        if config.now_ny() >= effective_deadline:
            # Deadline reached — return leaving remaining entries as
            # "still pending". The sweep handles them.
            for oid, e in pending.items():
                e.setdefault("filled_qty", 0)
                e.setdefault("status", "pending_at_sweep_deadline")
                logger.warning(
                    "fill-gated dispatch: sweep deadline reached with "
                    "entry still non-terminal; leaving for sweep",
                    extra={"order_id": oid, "symbol": e.get("symbol")},
                )
            break

        # ROUND-3 SAFETY FIX (SAFETY reviewer finding 1): read order
        # status FIRST (fast, local ib_async cache), then take the
        # position snapshot ONLY AFTER we know some entries are newly
        # terminal. Real IB propagates the position update AFTER the
        # execution report — the earlier ordering (snapshot first,
        # status second) let a REDUCING entry against a nonzero
        # baseline get MOC-sized to the PRE-fill position. The
        # snapshot call also carries a 0.5s ib.sleep, which gives IB
        # time to catch up between the status observation and the
        # position read. If the snapshot is still stale, the sweep's
        # two-sided top-up/trim (below) is the second line of
        # defence.
        newly_terminal: list[tuple[str, dict, int, str]] = []
        deadline_hit_mid_tick = False
        entries_to_pop: list[str] = []
        for oid, e in list(pending.items()):
            # Round-2: check deadline INSIDE the inner loop too. At
            # 500 names a full tick can approach the sweep margin;
            # without this a single pass could silently overrun.
            if config.now_ny() >= effective_deadline:
                deadline_hit_mid_tick = True
                break
            sym = e["symbol"]
            try:
                info = broker.get_order_status(oid)
                status = getattr(info, "status", "") or ""
                filled = int(getattr(info, "filled", 0) or 0)
            except Exception as ex:
                logger.warning(
                    "fill-gated dispatch: get_order_status failed; "
                    "leaving entry for sweep",
                    extra={"order_id": oid, "symbol": sym,
                           "error": str(ex)[:200]},
                )
                # Leave in pending; if it repeats we hit the deadline
                # and the sweep will pick it up.
                continue
            if status not in _TERMINAL_ORDER_STATES:
                continue
            # Terminal.
            e["filled_qty"] = filled
            e["status"] = status
            if filled <= 0:
                # No position to flatten from this entry — leave it
                # alone. Operator's stated behaviour: "A symbol whose
                # entry filled ZERO gets NO MOC leg from the fill-gated
                # path." The sweep will still catch any residual
                # baseline position.
                logger.info(
                    "fill-gated dispatch: entry terminal with 0 fill; "
                    "no MOC from fill-gated path",
                    extra={"order_id": oid, "symbol": sym,
                           "status": status},
                )
                entries_to_pop.append(oid)
                continue
            newly_terminal.append((oid, e, filled, status))

        if deadline_hit_mid_tick:
            # Emit the deadline log and break out — remaining pending
            # entries fall through to the sweep. Do NOT pop the
            # newly_terminal ones on this last partial pass, so the
            # sweep can still cover them by position snapshot.
            for oid, e in pending.items():
                e.setdefault("filled_qty", 0)
                e.setdefault("status", "pending_at_sweep_deadline")
                logger.warning(
                    "fill-gated dispatch: sweep deadline reached mid-tick; "
                    "leaving remaining entries for sweep",
                    extra={"order_id": oid, "symbol": e.get("symbol")},
                )
            break

        # Pop terminal-zero-fill entries now.
        for oid in entries_to_pop:
            pending.pop(oid, None)

        if not newly_terminal:
            # No new fills this tick — sleep and retry without paying
            # the 0.5s get_positions cost. Cheaper polling loop.
            if pending:
                time.sleep(poll_interval_secs)
            continue

        # Snapshot AFTER observing terminal fills, so the 0.5s ib_async
        # sleep inside get_positions gives IB time to propagate the
        # position update for the just-observed fills. This is the
        # SAFETY reviewer's fix (1) — cheapest defensible ordering
        # without changing the concurrency model.
        tick_snapshot = _snapshot_managed_positions(broker, managed_universe)
        if tick_snapshot is None:
            positions_unknown = True
            # We cannot size any MOC this tick — sleep and retry until
            # deadline. Every filled entry stays in ``pending`` so
            # the sweep sees them.
            time.sleep(poll_interval_secs)
            continue

        for oid, e, filled, status in newly_terminal:
            sym = e["symbol"]
            current = tick_snapshot.get(sym, 0)
            # ROUND-4 SAFETY FIX (SAFETY reviewer finding 2): the
            # deterministic expected post-fill quantity for this
            # symbol is baseline + signed_fill. If the tick snapshot
            # disagrees, the position update hasn't propagated yet
            # (IB emits execution reports BEFORE position updates)
            # and sizing an MOC off ``current`` would be wrong. Leave
            # in pending so the sweep re-verifies with a fresh
            # snapshot ~4 minutes later; every managed-universe
            # symbol still holding a position at sweep time gets an
            # MOC via the two-sided residual arithmetic.
            side = e.get("side", "")
            signed_fill = filled if side == "BUY" else -filled
            expected = int(baseline.get(sym, 0)) + int(signed_fill)
            if int(current) != expected:
                logger.warning(
                    "fill-gated dispatch: position snapshot lags fill "
                    "event (expected != current) — deferring MOC to "
                    "the sweep to avoid mis-sizing off a stale book",
                    extra={
                        "order_id": oid, "symbol": sym,
                        "baseline": int(baseline.get(sym, 0)),
                        "side": side, "filled": int(filled),
                        "expected_qty": expected,
                        "snapshot_qty": int(current),
                    },
                )
                # Keep in pending so the sweep re-verifies. Also mark
                # the entry as ``defer_to_sweep_for_moc`` in-place so
                # the sweep's belt-and-braces cancel loop leaves the
                # (already-terminal) entry alone — the entry IS
                # terminal, we just declined to submit the MOC here.
                continue
            if current == 0:
                # Entry filled and position book agrees the net is 0
                # (a reducing entry against a nonzero baseline that
                # exactly cancels). No MOC needed — the position is
                # already flat. The sweep would compute residual=0
                # and skip as well.
                logger.info(
                    "fill-gated dispatch: entry filled and position "
                    "confirmed flat (baseline+signed_fill=0); no MOC "
                    "needed for this symbol",
                    extra={"order_id": oid, "symbol": sym,
                           "filled": filled},
                )
                pending.pop(oid, None)
                continue
            leg, outcome = _submit_single_moc(
                broker, sym, current, submit_by,
            )
            if outcome == "ok" and leg is not None:
                moc_batch.append(leg)
                # Track SIGNED covered qty per symbol.
                covered_qty[sym] = (
                    covered_qty.get(sym, 0) + leg["signed_qty"]
                )
            elif outcome == "skipped_cutoff":
                skipped_past_cutoff.append(sym)
            elif outcome == "error":
                # Submission raised — treat as uncovered so the
                # pre-close signals escalate.
                uncovered.append(sym)
            pending.pop(oid, None)

        if pending:
            time.sleep(poll_interval_secs)

    return moc_batch, covered_qty, uncovered, skipped_past_cutoff, positions_unknown


def _sweep_moc_deadline(
    broker,
    entries: list[dict],
    managed_universe: set[str],
    covered_qty: dict[str, int],
    submit_by: datetime,
    dispatcher_legs: Optional[list[dict]] = None,
) -> tuple[list[dict], list[str], list[str], list[str], bool]:
    """DEADLINE SWEEP.

    Runs at the sweep_deadline. Guarantees every managed-universe symbol
    holding a nonzero position gets an MOC away before ``submit_by``
    (NASDAQ 15:55 cutoff), including residual/baseline positions carried
    in from a prior day and any position whose entry filled after the
    dispatcher gave up. Also TOPS UP or TRIMS any dispatcher MOC whose
    size does not match the true post-fill position (see the "residual"
    section below).

    Steps:
      1. Cancel any entry order still non-terminal at
         ``get_order_status`` time (belt-and-braces — the dispatcher
         should have left few). Cancel failures are surfaced up the
         return tuple (an entry that could not be pulled is by
         definition an uncontrolled position source).
      2. Verify each dispatcher-submitted MOC is still live at the
         exchange (``dispatcher_legs`` carries per-leg oid + symbol +
         signed_qty). A dispatcher MOC that reached terminal
         Cancelled with filled=0 (exchange rejection during the
         cancel race) is EXCLUDED from ``effective_covered_qty``;
         without this the sweep would credit the coverage and skip
         the symbol, leaving it uncovered until the audit fires ~1s
         later — with no re-MOC path. Fixes the ROUND-2 SAFETY
         REVIEWER'S third finding.
      3. Take a FRESH ``broker.get_positions()`` snapshot.
      4. Iterate over EVERY symbol that either shows a nonzero held
         quantity OR appears in ``effective_covered_qty`` (two-sided:
         so the sweep can TRIM an over-cover, not just top up a
         short-cover). Compute
         ``residual = held + effective_covered_qty.get(sym, 0)``.
         This is the SIGNED quantity the sweep must submit to zero
         the book.
         * ``residual == 0`` → dispatcher already covered exactly;
           skip.
         * ``residual > 0`` → net long → submit SELL MOC ``|residual|``.
         * ``residual < 0`` → net short (dispatcher over-covered a
           reducing entry against a nonzero baseline, or the
           snapshot lagged the fill) → submit BUY MOC ``|residual|``
           as an offset. Fixes the ROUND-2 SAFETY REVIEWER'S first
           (critical) finding.
      5. Any leg past ``submit_by`` is recorded as skipped (exposure).

    ``covered_qty`` is signed — ``covered_qty[sym] = -60`` for a
    dispatcher SELL MOC of 60. So for held=100 the residual is
    ``100 + (-60) = 40 → SELL MOC 40`` (top-up); for held=0 the
    residual is ``0 + (-60) = -60 → BUY MOC 60`` (trim, offsets the
    over-flatten).

    Parameters
    ----------
    dispatcher_legs
        The MOC legs the fill-gated dispatcher already submitted (the
        ``fg_moc_batch``). Needed so the sweep can drop a rejected
        dispatcher MOC's credit from ``covered_qty`` before computing
        residuals. Pass ``None`` (default) to skip verification
        entirely — used only by direct unit tests where the caller
        doesn't have a leg list handy; in main() ALWAYS pass the
        real leg list.

    Returns
    -------
    ``(moc_batch, cancelled_oids, cancel_failed_syms, skipped_symbols,
    positions_unknown)``:
      - ``moc_batch`` — MOC legs the sweep submitted (in addition to
        the dispatcher's).
      - ``cancelled_oids`` — order ids the sweep cancelled cleanly.
      - ``cancel_failed_syms`` — symbols whose non-terminal entry
        cancel raised. These entries may still be working and their
        eventual fill is uncontrolled — the caller must OR this into
        the pre-close exposure signal.
      - ``skipped_symbols`` — managed symbols that needed an MOC but
        submit_by was already past.
      - ``positions_unknown`` — True if ``get_positions()`` raised.
    """
    # (1) Cancel any still-non-terminal entry from the fill-gated path.
    cancelled_oids: list[str] = []
    cancel_failed_syms: list[str] = []
    for e in entries:
        oid = e.get("order_id")
        if oid is None:
            continue
        status = e.get("status") or ""
        try:
            info = broker.get_order_status(oid)
            live_status = getattr(info, "status", "") or ""
            live_filled = int(getattr(info, "filled", 0) or 0)
        except Exception:
            live_status = status
            live_filled = int(e.get("filled_qty", 0) or 0)
        if live_status in _TERMINAL_ORDER_STATES:
            # Update the entry's status in case the dispatcher didn't
            # observe the terminal transition. (Round-2 simplification:
            # single get_order_status call, not the earlier duplicated
            # info + info2 read.)
            e["status"] = live_status
            e["filled_qty"] = live_filled
            continue
        try:
            broker.cancel(oid)
            cancelled_oids.append(oid)
            logger.info(
                "sweep: cancelled non-terminal entry",
                extra={"order_id": oid, "symbol": e.get("symbol"),
                       "status": live_status},
            )
            e["status"] = "ApiCancelled"
        except Exception as ex:
            # Round-2 fix: track cancel failures. An entry we could
            # not pull is uncontrolled — it may still fill mid-window
            # and end up as an overnight position no MOC covers.
            sym_cf = e.get("symbol")
            if sym_cf:
                cancel_failed_syms.append(sym_cf)
            logger.warning(
                "sweep: cancel failed — entry may still be working",
                extra={"order_id": oid, "symbol": sym_cf,
                       "error": str(ex)[:200]},
            )

    # (2) Verify each dispatcher MOC's live status. ROUND-3 SAFETY FIX
    # (SAFETY reviewer finding 3): _submit_single_moc returns outcome
    # 'ok' the instant broker.buy/sell returns an oid, which is BEFORE
    # any exchange ack. An MOC rejected at the exchange (short-sale
    # restriction, margin, contract/venue issue) can flip terminal
    # Cancelled with filled=0 within seconds. If we credit the
    # dispatcher's covered_qty regardless, the sweep would skip the
    # symbol (residual == 0 arithmetic) — but the exchange has no
    # cover and the audit only fires later, with no re-MOC path.
    # Drop rejected dispatcher legs from the effective credit here so
    # the sweep can resubmit.
    effective_covered_qty: dict[str, int] = dict(covered_qty)
    if dispatcher_legs:
        for leg in dispatcher_legs:
            oid = leg.get("oid")
            sym = leg.get("symbol")
            signed = int(leg.get("signed_qty", 0) or 0)
            if oid is None or not sym or signed == 0:
                continue
            try:
                info = broker.get_order_status(oid)
                d_status = getattr(info, "status", "") or ""
                d_filled = int(getattr(info, "filled", 0) or 0)
            except Exception as ex:
                # Cannot verify — leave the credit in place (worst
                # case: rely on the coverage cross-check / audit
                # downstream). Not zeroing here avoids a spurious
                # double-flatten on a transient status-poll blip.
                logger.warning(
                    "sweep: dispatcher MOC status poll failed; leaving "
                    "coverage credit in place",
                    extra={"order_id": oid, "symbol": sym,
                           "error": str(ex)[:200]},
                )
                continue
            if d_status in _TERMINAL_ORDER_STATES and d_filled == 0:
                logger.error(
                    "sweep: dispatcher MOC terminal-rejected; dropping "
                    "coverage credit so sweep re-submits",
                    extra={"order_id": oid, "symbol": sym,
                           "status": d_status,
                           "dropped_signed_qty": signed},
                )
                # Undo this leg's contribution.
                effective_covered_qty[sym] = (
                    effective_covered_qty.get(sym, 0) - signed
                )
                if effective_covered_qty[sym] == 0:
                    del effective_covered_qty[sym]

    # (3) Fresh position snapshot.
    try:
        positions = broker.get_positions()
    except Exception as e:
        logger.error(
            "sweep: get_positions FAILED — POSITIONS MAY BE LIVE INTO "
            "THE CLOSE, OPERATOR MUST MANUALLY FLATTEN",
            extra={"error": str(e)[:200]},
        )
        return [], cancelled_oids, cancel_failed_syms, [], True

    # Aggregate duplicate rows (multi-account / update-race).
    agg: dict[str, int] = {}
    for p in positions:
        if p.symbol not in managed_universe:
            continue
        agg[p.symbol] = agg.get(p.symbol, 0) + int(p.quantity)

    # ROUND-3 TWO-SIDED SWEEP (SAFETY reviewer finding 1): iterate
    # over the UNION of "symbols currently held" (``agg``) and
    # "symbols the dispatcher already MOC'd" (``effective_covered_qty``),
    # so a symbol that is FLAT NOW but has a dispatcher MOC still
    # queued (e.g. the reducing-entry-against-baseline case where the
    # snapshot lagged the fill, so dispatcher over-covered) gets a
    # TRIM MOC in the opposite direction — restoring the invariant
    # ``held + Σ signed_MOC == 0``.
    #
    # SAFETY reviewer's critical case: baseline AAPL +100, entry
    # SELL 100 (reducing to 0). IB emits the Filled event before
    # the position update, so the dispatcher's tick snapshot read
    # +100 while status read Filled. dispatcher submitted SELL MOC
    # 100 → covered_qty[AAPL] = -100. True held after fill = 0.
    # Old sweep iterated only ``agg`` and skipped AAPL (not in the
    # dict), leaving the naked SHORT of 100 shares undetected until
    # 16:05. New sweep: sym in union, held = agg.get(AAPL, 0) = 0,
    # residual = 0 + (-100) = -100 < 0 → BUY MOC 100 offset. Net at
    # close: +100 baseline + (-100 entry) + (-100 MOC) + (+100 sweep)
    # = 0. Correct.
    #
    # The dispatcher-partial-plus-late-fill case (holds 100 but
    # dispatcher only covered 60) still works: held=100, covered=-60,
    # residual = 40 > 0 → SELL MOC 40 top-up. Both cases in one loop.
    moc_batch: list[dict] = []
    skipped: list[str] = []
    all_syms = sorted(set(agg) | set(effective_covered_qty))
    for sym in all_syms:
        if sym not in managed_universe:
            # Defensive — effective_covered_qty is keyed by dispatcher
            # symbols which are already managed; this is a belt-and-
            # braces filter.
            continue
        held = agg.get(sym, 0)
        already_signed = effective_covered_qty.get(sym, 0)
        residual = held + already_signed
        if residual == 0:
            # Book zeros out — dispatcher covered exactly, or symbol
            # is fully flat and uncovered.
            continue
        leg, outcome = _submit_single_moc(
            broker, sym, residual, submit_by,
        )
        if outcome == "ok" and leg is not None:
            moc_batch.append(leg)
        elif outcome == "skipped_cutoff":
            skipped.append(sym)
        # "error" is logged inside _submit_single_moc; treat as skipped
        # so pre-close exposure catches it.
        elif outcome == "error":
            skipped.append(sym)

    if skipped:
        logger.error(
            "sweep: MOC legs skipped past submit cutoff — POSITIONS "
            "LEFT OPEN",
            extra={"n_skipped": len(skipped), "symbols": skipped[:20]},
        )

    return moc_batch, cancelled_oids, cancel_failed_syms, skipped, False


def _moc_legs_corroborate_flat(
    broker,
    moc_legs: list[dict],
    book_was_opened: bool,
) -> Optional[bool]:
    """Independent evidence for ``_poll_positions_after_close`` that the
    book MUST be flat, used to disambiguate an empty position response.

    Returns:
    * ``True``  — positive evidence of flat: either no book was ever
      opened (no entry filled, so there is nothing to flatten), or every
      MOC leg reached terminal ``Filled`` with ``filled > 0``.
    * ``False`` — positive evidence AGAINST flat: at least one MOC leg is
      terminal-not-Filled or still open.
    * ``None``  — cannot tell (a status poll raised, or a book was opened
      but no MOC legs exist to corroborate against). Treated by the
      caller exactly like ``False``: not proof, so not flat.

    Deliberately order-state-based rather than position-based: the whole
    point is to cross-check the position stream against a SEPARATE
    source, so a single wiped subscription cannot satisfy both.
    """
    if not book_was_opened:
        return True
    if not moc_legs:
        return None
    for leg in moc_legs:
        oid = leg.get("oid")
        if oid is None:
            return None
        try:
            info = broker.get_order_status(oid)
        except Exception:
            return None
        status = getattr(info, "status", "")
        filled = int(getattr(info, "filled", 0) or 0)
        if status != "Filled" or filled <= 0:
            return False
    return True


def _poll_positions_after_close(
    broker,
    managed_universe: set[str],
    poll_until: datetime,
    not_before: Optional[datetime] = None,
    min_consecutive_flat_reads: int = 2,
    initial_interval_secs: float = config.DEFAULT_POST_CLOSE_POLL_INITIAL_INTERVAL_SECS,
    max_interval_secs: float = config.DEFAULT_POST_CLOSE_POLL_MAX_INTERVAL_SECS,
    corroborator: Optional[Callable[[], Optional[bool]]] = None,
) -> tuple[bool, list[str], Optional[str]]:
    """AUTHORITATIVE post-auction exposure verdict.

    After 16:00 the NASDAQ close cross has printed and MOC fills are
    reflected in ``broker.get_positions()``. IB can take a few seconds
    to a few minutes to propagate the fill; we poll with exponential
    backoff until ``poll_until`` (default 16:05 ET).

    ROUND-2 SAFETY BOUNDS. Two independent guards close the "poll
    returns FLAT before the auction printed" bug the round-1 code had:

    * ``not_before`` — if set, the poll REFUSES to return the FLAT
      verdict until ``config.now_ny() >= not_before``. In prod, this
      is ``today_ny(16, 0) + DEFAULT_POST_CLOSE_POLL_SETTLE_SECS`` so
      no code path can declare FLAT before the 16:00 auction has had
      a chance to print. A run that reaches the poll early (recorder
      short-circuit, empty book by mid-afternoon, etc.) keeps polling
      until it can honestly answer. If EVERY successful poll before
      ``not_before`` is empty AND the loop hits ``poll_until`` before
      ``not_before`` is reached (only possible with a misconfigured
      ``poll_until``), we return PESSIMISTIC EXPOSED to fail closed.

    * ``min_consecutive_flat_reads`` — require N consecutive
      successful all-flat reads before returning FLAT (default 2). A
      single transient empty response from a broker that returned rows
      on a prior poll is suspect, not authoritative; requiring a
      confirmation poll defeats the ``empty_snapshot_after_fills``
      failure mode (a TWS stream that momentarily returns []).

    Rule:
      * Any nonzero managed-universe position at ``poll_until``
        → EXPOSED (rc=11).
      * ``get_positions()`` fails on EVERY attempt in the window
        → PESSIMISTIC EXPOSED (rc=11) — we cannot verify flat, so we
        cannot claim it.
      * All managed-universe positions are zero AND we are past
        ``not_before`` AND we have N consecutive flat reads
        → FLAT (rc=0 gate).

    Returns ``(is_exposed, nonzero_symbols, error_msg)``.
    * ``is_exposed`` — True iff we should return rc=11.
    * ``nonzero_symbols`` — sorted list of managed-universe symbols
      still non-flat at the last successful poll (empty if never
      succeeded or last read was empty-but-pre-not_before).
    * ``error_msg`` — last exception string if all polls failed OR a
      synthetic reason if we exited without a confirmed flat read
      (``"pre_not_before_no_confirmed_flat"``, etc.); ``None`` on a
      genuine FLAT verdict.
    """
    interval = initial_interval_secs
    last_error: Optional[str] = None
    last_nonzero: list[str] = []
    at_least_one_success = False
    consecutive_flat = 0
    while True:
        now_before_poll = config.now_ny()
        try:
            positions = broker.get_positions()
        except Exception as e:
            last_error = str(e)[:200]
            # A raise is not a "flat" read — reset the counter.
            consecutive_flat = 0
            logger.warning(
                "post-close position poll: get_positions failed; retrying",
                extra={"error": last_error,
                       "poll_until": poll_until.isoformat()},
            )
        else:
            at_least_one_success = True
            # Clear any error from an EARLIER failed poll that has since
            # recovered, so the fail-closed reason reported below names
            # the condition that actually held at the deadline rather
            # than a stale connection fault. Set before the
            # uncorroborated-empty branch, which legitimately re-sets it.
            last_error = None
            agg: dict[str, int] = {}
            for p in positions:
                if p.symbol not in managed_universe:
                    continue
                agg[p.symbol] = agg.get(p.symbol, 0) + int(p.quantity)
            last_nonzero = sorted(s for s, q in agg.items() if q != 0)
            # A WHOLLY EMPTY position response is ambiguous and must not
            # be read as proof of flat on its own. ``broker.get_positions``
            # is ``reqPositions(); sleep(0.5); positions()`` — it returns
            # ib_async's CACHED position set and returns [] WITHOUT
            # raising if the subscription was torn down and re-established
            # (e.g. a TWS drop / auto-reconnect across the 16:00 auction)
            # and has not yet repopulated. A genuinely flat book and a
            # wiped cache are byte-identical at this layer, so
            # ``min_consecutive_flat_reads`` does not separate them — a
            # persistently-empty stream satisfies the counter just as a
            # real flat book does. This is the post-close analogue of the
            # pre-close ``empty_snapshot_after_fills`` guard.
            #
            # We therefore require INDEPENDENT corroboration before an
            # empty read may count toward the flat tally. ``corroborator``
            # returns True only on positive evidence the book must be flat
            # (every MOC leg terminal-Filled, or no book was ever opened);
            # False on positive evidence it is not; None when it cannot
            # tell. Anything other than True leaves the tally at zero, so
            # the loop runs to ``poll_until`` and exits through the
            # fail-closed "insufficient_flat_confirmations" branch below.
            # A NON-empty response is trusted as-is: the stream is
            # demonstrably reporting, so "rows present, none of ours
            # nonzero" is real evidence of flat.
            raw_empty = not positions
            if not last_nonzero and raw_empty:
                corro: Optional[bool] = None
                if corroborator is not None:
                    try:
                        corro = corroborator()
                    except Exception as e:
                        # Never let a corroborator bug mask exposure.
                        corro = None
                        logger.warning(
                            "post-close position poll: corroborator "
                            "raised; treating empty read as "
                            "UNCORROBORATED",
                            extra={"error": str(e)[:200]},
                        )
                if corro is not True:
                    consecutive_flat = 0
                    last_error = "uncorroborated_empty_position_read"
                    logger.warning(
                        "post-close position poll: EMPTY position "
                        "response with no corroboration that the book "
                        "is flat — NOT counting toward FLAT (a wiped "
                        "position cache looks identical to a flat "
                        "book); continuing to poll",
                        extra={"corroborated": corro,
                               "poll_until": poll_until.isoformat()},
                    )
                    now = config.now_ny()
                    if now >= poll_until:
                        break
                    sleep_for = min(
                        interval, (poll_until - now).total_seconds(),
                    )
                    if sleep_for > 0:
                        time.sleep(sleep_for)
                    interval = min(interval * 1.5, max_interval_secs)
                    continue
            if not last_nonzero:
                # Empty read. Only count it toward the consecutive-flat
                # tally if we're past not_before — otherwise a stale
                # early read could otherwise satisfy the counter before
                # the auction has printed.
                if not_before is not None and now_before_poll < not_before:
                    logger.info(
                        "post-close position poll: empty read pre-auction; "
                        "not counting toward FLAT confirmation",
                        extra={"not_before": not_before.isoformat(),
                               "now": now_before_poll.isoformat()},
                    )
                    consecutive_flat = 0
                else:
                    consecutive_flat += 1
                    if consecutive_flat >= min_consecutive_flat_reads:
                        logger.info(
                            "post-close position poll: FLAT",
                            extra={"n_managed_symbols": len(managed_universe),
                                   "consecutive_flat_reads": consecutive_flat},
                        )
                        return False, [], None
                    logger.info(
                        "post-close position poll: flat read; awaiting "
                        "confirmation",
                        extra={"consecutive_flat_reads": consecutive_flat,
                               "required": min_consecutive_flat_reads},
                    )
            else:
                consecutive_flat = 0
                logger.warning(
                    "post-close position poll: still non-flat, retrying",
                    extra={"n_nonzero": len(last_nonzero),
                           "sample_symbols": last_nonzero[:20]},
                )
        now = config.now_ny()
        if now >= poll_until:
            break
        sleep_for = min(interval, (poll_until - now).total_seconds())
        if sleep_for > 0:
            time.sleep(sleep_for)
        interval = min(interval * 1.5, max_interval_secs)

    if not at_least_one_success:
        # Every poll raised. Pessimistic verdict — cannot confirm flat.
        logger.error(
            "post-close position poll: every attempt FAILED — cannot "
            "confirm flat, treating as EXPOSED",
            extra={"error": last_error},
        )
        return True, [], last_error
    if last_nonzero:
        logger.error(
            "post-close position poll: managed-universe symbols still "
            "non-flat after deadline — OVERNIGHT EXPOSURE",
            extra={"n_nonzero": len(last_nonzero),
                   "symbols": last_nonzero[:50]},
        )
        return True, last_nonzero, None
    # We reached poll_until with successful reads but no confirmed
    # FLAT (either not_before was never reached, or we didn't get
    # ``min_consecutive_flat_reads`` in a row). Fail closed — do NOT
    # silently declare flat.
    # Prefer the specific reason when we have one — an operator reading
    # ``post_close_poll_error`` needs to distinguish "the position stream
    # went empty and nothing could corroborate it" from a generic
    # shortfall of consecutive reads. ``last_error`` is only set on the
    # uncorroborated-empty path here: a poll that RAISED sets it too,
    # but every successful poll clears it above, so by the time we get
    # here it can only describe the most recent read — not a stale
    # connection fault that already recovered.
    _reason = last_error or "insufficient_flat_confirmations"
    logger.error(
        "post-close position poll: reached deadline without a confirmed "
        "FLAT (either pre-not_before never elapsed or insufficient "
        "consecutive flat reads) — treating as EXPOSED",
        extra={
            "consecutive_flat": consecutive_flat,
            "required": min_consecutive_flat_reads,
            "not_before": not_before.isoformat() if not_before else None,
            "poll_until": poll_until.isoformat(),
            "reason": _reason,
        },
    )
    return True, [], _reason


def _audit_moc_acceptance(
    broker,
    moc_batch: list[dict],
    wall_clock_wait: float = config.DEFAULT_MOC_ACK_TIMEOUT_SECS,
) -> set[str]:
    """RRR1: wait up to ``wall_clock_wait`` seconds for MOC EXCHANGE
    ACCEPTANCE and return the set of symbols whose MOC was REJECTED.

    ``broker._place`` (via ``buy``/``sell``) returns synchronously the
    moment TWS accepts the API-level placeOrder call. That is BEFORE
    any exchange ack. An MOC submitted past NASDAQ's 15:55:00 MOC
    ENTRY cutoff (SEC release 34-84454) is API-accepted, an oid is
    returned, but the exchange REJECTS it — the order flips to
    ``Cancelled`` / ``Inactive`` with ``filled=0`` within seconds to
    a minute. Without this audit, the coverage cross-check scores
    the symbol as covered (an oid came back) and could show rc=0
    despite an open overnight position (the post-close poll would
    still catch it, but this audit is the EARLY warning so the
    operator can run panic before 16:00).

    Note the two distinct NASDAQ deadlines: 15:55:00 is the ENTRY
    cutoff (last moment an MOC will be accepted); 15:50:00 is the
    CANCEL/MODIFY cutoff (past 15:50 an already-accepted MOC cannot
    be pulled). Both matter here: the audit fires slightly after the
    sweep (~15:54-15:55), so a rejection detected here is caught
    AFTER the cancel window has already closed. The audit is
    diagnostic (surface it to rc=11 → operator page) rather than
    corrective.

    We wait per-oid for a terminal state via ``broker.wait_for_fill``:
    - Terminal (Cancelled / ApiCancelled / Inactive) with ``filled=0``
      → REJECTED → add ``symbol`` to the unfilled set.
    - Terminal with ``filled>0`` → early Filled → fine.
    - TimeoutError → still ``Submitted`` at deadline → presumed
      accepted at exchange (MOCs sit ``Submitted`` until 16:00).
    - Exception → pessimistic: add to unfilled set (safer than
      silently assuming acceptance).

    Wall-clock budget is bounded so a run doesn't sit ~5-8 min
    waiting until 16:00 for the real fill. Residual risk: an MOC
    that was accepted at ack then Cancelled after the audit window
    is not caught here — but the post-close position poll catches
    it definitively.

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

    ``entries['filled_qty']`` is derived from
    ``_fill_gated_moc_dispatch`` (via its per-tick
    ``broker.get_order_status`` reads) and from ``_sweep_moc_deadline``
    for entries the dispatcher gave up on. Both can lie in two ways
    under slow-TWS conditions:
      (1) the dispatcher's ``max_wait_secs`` cap or the sweep deadline
          fires before an entry's Filled event lands, and the entry
          is recorded as ``filled_qty=0``, ``status="pending_at_sweep_deadline"``
          / ``"wait_budget_exhausted"`` even though the MKT truly
          filled at the exchange;
      (2) a truncated ``get_order_status`` read misses a late-arriving
          Filled status update.

    And any ``get_positions()`` call can be silently partial (TWS
    reconnect during the request returns 3 of 10 held names).

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
            # RRR3: a live entry SHOULD be terminal by cross-check
            # time (Filled / Cancelled / ApiCancelled / Inactive) —
            # either it filled, or the belt-and-braces cancel loop
            # pulled it. A non-terminal read is itself a staleness
            # signal — the local ib_async cache may reflect a stale
            # filled=0 during a TWS event-stream brownout. Treat
            # non-terminal the same as an exception: assume pessimistic
            # full fill. Overcounting exposure > silently missing it.
            # NOTE: under LMT entries, a genuinely-unfilled DAY LMT
            # is CANCELLED by the sweep before this cross-check runs,
            # so a Cancelled/ApiCancelled with filled=0 here is
            # normal and correctly counts as zero exposure.
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
            "REPLAY SUBMIT: real paper LMT entries + MOC exits will be "
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

    # Round-2 replay-clock fix: compute the DECISION TIMESTAMP up
    # front so downstream can use it consistently for BOTH the
    # look-ahead cap on the tbbo extraction AND the staleness
    # comparison in the entry-quote guards. On the LIVE path this
    # is wall clock ``now``; on the REPLAY path this is the
    # replay-day 15:50 + print_wait_secs, so measuring "tick age"
    # against it correctly rejects a tick from LAST WEEK while
    # accepting one from 15:49 on the replay date. Without this
    # every replay-mode entry would skip ``quote_stale`` (a
    # historical tick is trivially days-to-months old vs today's
    # wall clock) and ``--replay-submit`` would place ZERO orders
    # while exiting rc=0 — the documented rehearsal-before-real-
    # money would be a silent no-op.
    if replay_mode:
        from datetime import date as _date_ctor
        _rd_parts = args.replay_date.split("-")
        _replay_date_obj = _date_ctor(
            int(_rd_parts[0]), int(_rd_parts[1]), int(_rd_parts[2]),
        )
        # Use the datetime constructor + timedelta so
        # ``print_wait_secs`` can safely be >= 60.
        _decision_dt_ny = datetime(
            _replay_date_obj.year, _replay_date_obj.month,
            _replay_date_obj.day, reb_hh, reb_mm, 0,
            tzinfo=config.NY_TZ,
        ) + timedelta(seconds=float(args.print_wait_secs))
        decision_ts_utc = _decision_dt_ny.astimezone(_UTC)
    else:
        decision_ts_utc = datetime.now(_UTC)

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
    last_quote_by_sym: dict[str, dict] = {}
    try:
        if feed is None:
            day_df = pl.DataFrame()
            last_mid_by_sym: dict[str, float] = {}
        else:
            (
                day_df,
                last_mid_by_sym,
                last_mid_ts_by_sym,
                last_quote_by_sym,
            ) = _build_live_day_frame(
                feed, universe, reference_stats,
                min_coverage=args.min_universe_coverage,
                # Round-2 look-ahead defence: bound tbbo at the
                # replay-day decision moment. On the LIVE path this
                # is "now" and the feed buffer only holds pre-now
                # ticks anyway; on the REPLAY path this is the
                # replay-day 15:50 anchor, so the cached whole-day
                # parquet cannot leak the 16:00 closing print into
                # the LMT entry price.
                decision_ts_utc=decision_ts_utc,
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
        last_quote_by_sym = {}

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
    # Round-2 change: replay HONOURS ``--submit-by`` by default so a
    # late-in-the-day rehearsal correctly surfaces the same rc=4 code
    # a real run would. Opt out with ``--replay-ignore-cutoff`` to
    # preserve the earlier afternoon-rehearsal ergonomics (the cutoff
    # is neutralised further down alongside ``effective_submit_by``).
    #
    # This rc=4 check ONLY applies to runs that will actually attempt
    # to place orders. Dry-run and broker-free replay (default replay
    # without ``--replay-submit``) never reach the broker, so the
    # cutoff is not relevant to them.
    replay_cutoff_disabled = replay_mode and args.replay_ignore_cutoff
    broker_free_run = args.dry_run or (replay_mode and not args.replay_submit)
    if (not broker_free_run
            and not replay_cutoff_disabled
            and now > submit_by_at):
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
        # ``decision_ts_utc`` was already computed up-front (before
        # ``_build_live_day_frame``) so both the tbbo-time bound and
        # the entry staleness comparison share ONE decision timestamp.
        # The recorder just consumes it here.
        if replay_mode:
            recorder_trading_date = args.replay_date
        else:
            recorder_trading_date = config.now_ny().date().isoformat()
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

    # Armed immediately before LEG 1 submission. Read ONLY by the
    # last-resort ``except`` on the order-path ``try`` below, to decide
    # whether an unhandled exception is a safe pre-order failure (rc=1)
    # or a possible uncovered book (rc=11). Must be initialised OUTSIDE
    # the ``try`` so the handler can always read it.
    orders_may_be_live = False

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

        # M1: per-leg submit-by check for the MOC EXIT batch. The LMT ENTRY
        # batch always fires (see below): a filled entry with no queued exit
        # would still leave open exposure, so we always try to route the exit;
        # if the wall clock has crossed submit_by by then we exit rc=4 loudly.
        #
        # Replay policy: HONOUR the real wall-clock cutoff by default so a
        # late-in-the-day rehearsal correctly surfaces the same rc=4 / rc=11
        # codes a real run would (the previous silent neutralisation left
        # replay's ``n_skipped_past_cutoff:0`` counter uninformative and let
        # a late replay MOC never be flagged). Opt out with
        # ``--replay-ignore-cutoff`` to preserve the earlier afternoon-
        # rehearsal ergonomics where you deliberately want to exercise the
        # order path even though ``now`` is past 15:55.
        if replay_mode and args.replay_ignore_cutoff:
            effective_submit_by = config.now_ny() + timedelta(hours=1)
            logger.info(
                "replay mode: per-leg submit-by neutralised "
                "(--replay-ignore-cutoff)",
                extra={"effective_submit_by": effective_submit_by.isoformat()},
            )
        else:
            effective_submit_by = submit_by_at

        # --------------------------------------------------------------
        # LEG 1: LMT entries sized to (target - current), priced at
        # the DECISION-TIME tbbo mid — the same 15:50
        # ``feed.snapshot()`` that fed the MVO. Reusing that
        # observation (rather than taking a fresh mid immediately
        # before submission) means the price we solved against and
        # the price we bid at agree; the tradeoff is that the mid
        # is knowingly stale by the MVO solve duration, so the
        # staleness guard (``DEFAULT_MAX_QUOTE_STALENESS_SECS``) is
        # the binding safety knob and is measured against
        # ``quote.ts`` (the tick time) at wall-clock submission time.
        # Fires unconditionally with respect to the MOC cutoff.
        # Default tif=DAY; the fill-gated dispatcher below observes
        # each entry transitioning to terminal state and either
        # fires the MOC (Filled) or leaves it for the sweep
        # (Cancelled / stuck / never crossed). A symbol with no
        # usable decision-time mid is SKIPPED — see
        # ``_submit_limit_entries`` docstring.
        # --------------------------------------------------------------
        # Reference "now" for the entry-quote staleness check.
        # * LIVE / PAPER: ``datetime.now(_UTC)`` — the actual wall
        #   clock at submit time. Tick age = submit_time - tick_ts.
        # * REPLAY: ``decision_ts_utc`` — the replay-day 15:50
        #   anchor. Tick age = replay_decision - tick_ts, so a
        #   historical tick at 15:49 on the replay date reads as
        #   ~1 minute old (fresh), not "months old vs today's
        #   wall clock" (which would skip every symbol).
        #
        # ``_submit_limit_entries`` also re-samples ``now`` per
        # symbol inside its loop so a slow qualification of the
        # first few names doesn't unfairly stale the last few
        # names — see the ``now_utc`` handling there.
        if replay_mode:
            entry_reference_now_utc = decision_ts_utc
        else:
            entry_reference_now_utc = datetime.now(_UTC)
        if snapshot_completed_at_utc is not None:
            _seconds_since_decision = (
                entry_reference_now_utc - snapshot_completed_at_utc
            ).total_seconds()
        else:
            _seconds_since_decision = None
        # Log per-symbol mid staleness AT SUBMISSION so operators can
        # watch how far the price we bid at has drifted from the
        # observation we solved against. Systematic multi-second
        # staleness is an operational signal — a slow solve or a
        # slow snapshot-to-submit path shrinks realized fill rate
        # under LMT entries even if no single quote crosses the
        # ``DEFAULT_MAX_QUOTE_STALENESS_SECS`` threshold.
        _ages = [
            (entry_reference_now_utc - q["ts"].astimezone(_UTC)).total_seconds()
            for sym, q in last_quote_by_sym.items()
            if sym in universe_set
            and q.get("ts") is not None
            and q["ts"].tzinfo is not None
            and (extended_targets.get(sym, 0) - current.get(sym, 0)) != 0
        ]
        _ages_summary = (
            {
                "n": len(_ages),
                "min": round(min(_ages), 3),
                "max": round(max(_ages), 3),
                "median": round(sorted(_ages)[len(_ages) // 2], 3),
            }
            if _ages else {"n": 0}
        )
        logger.info(
            "entry-time quote staleness (decision-snapshot age at submit)",
            extra={
                "secs_since_decision_snapshot": (
                    round(_seconds_since_decision, 3)
                    if _seconds_since_decision is not None else None
                ),
                "quote_age_secs_summary": _ages_summary,
                "n_quotes_available": len(last_quote_by_sym),
                "n_targets_to_price": sum(
                    1 for s, t in extended_targets.items()
                    if s in universe_set and (t - current.get(s, 0)) != 0
                ),
                "staleness_threshold_secs": (
                    config.DEFAULT_MAX_QUOTE_STALENESS_SECS
                ),
                "replay_mode": replay_mode,
            },
        )

        # ARM the exposure gate BEFORE the call, not after: a batch that
        # raises partway through has already put live orders on the book,
        # and a post-call assignment would never run.
        orders_may_be_live = True
        entries = _submit_limit_entries(
            broker=broker,
            targets=extended_targets,
            current=current,
            managed_universe=universe_set,
            last_quote_by_sym=last_quote_by_sym,
            now_utc=entry_reference_now_utc,
            max_staleness_secs=config.DEFAULT_MAX_QUOTE_STALENESS_SECS,
            max_rel_spread=config.DEFAULT_MAX_ENTRY_REL_SPREAD,
            replay_mode=replay_mode,
        )
        # Recorder — one entry submission row per attempted LMT. Even
        # skipped / failed submissions (``order_id is None``) get a row
        # so post-trade analysis sees the intended trade that never
        # made it, plus the limit price and error tag when populated.
        if recorder is not None:
            _entry_sent_ts = datetime.now(_UTC)
            for _e in entries:
                trade_recorder.safe_call(
                    recorder.record_submission,
                    leg="entry",
                    symbol=_e.get("symbol", ""),
                    side=_e.get("side", ""),
                    requested_qty=int(_e.get("requested_qty", 0) or 0),
                    order_type="LMT",
                    order_id=_e.get("order_id"),
                    sent_ts_utc=_entry_sent_ts,
                    exchange=ACTIVE_ROUTING,
                    tif="DAY",
                    error=_e.get("error"),
                    limit_price=_e.get("limit_price"),
                )
        _quote_guard_reasons = {
            "no_usable_quote_at_entry", "quote_missing_fields",
            "quote_non_numeric", "quote_non_finite",
            "quote_non_positive", "quote_crossed",
            "quote_naive_ts", "quote_bad_ts",
            "quote_stale", "quote_wide_spread",
            "quote_absurd_price",
        }
        _n_intended = len(entries)
        _n_submitted = sum(1 for e in entries if e.get("order_id"))
        _n_broker_failed = sum(
            1 for e in entries
            if e.get("order_id") is None
            and e.get("error") not in _quote_guard_reasons
            and e.get("error") is not None
        )
        _n_skipped_quote_guard = sum(
            1 for e in entries
            if e.get("order_id") is None
            and e.get("error") in _quote_guard_reasons
        )
        logger.info(
            "LMT entry batch submitted",
            extra={
                "n_entries": _n_intended,
                "n_ok": _n_submitted,
                "n_failed": _n_broker_failed,
                "n_skipped_quote_guard": _n_skipped_quote_guard,
            },
        )

        # BLOCKER 4 (round 2): mass-skip guard.
        # Rc=5 (empty share_targets) fires BEFORE entry submission and
        # cannot catch a systemic feed wedge that skips e.g. 40 of 49
        # priced-in names. Executing the surviving 9 is worse than
        # skipping the whole run: it's a non-random truncated fragment
        # of a risk-balanced solve entering the closing ramp with a
        # covariance that was fit on the full set. Refuse loudly with
        # rc=10 (pre-order refusal — no exposure — stays out of the
        # ``rc >= 11`` pager band). If any entries DID submit already
        # (the guard skipped a mix of names), CANCEL them before the
        # rc=10 return so those DAY LMTs do not linger and fill into
        # an uncovered position.
        _n_universe_targets = sum(
            1 for s, t in extended_targets.items()
            if s in universe_set and (t - current.get(s, 0)) != 0
        )
        if _n_universe_targets > 0:
            _submit_coverage = _n_submitted / _n_universe_targets
        else:
            _submit_coverage = 1.0
        if _submit_coverage < config.DEFAULT_MIN_ENTRY_SUBMIT_COVERAGE:
            logger.error(
                "LMT entry submit coverage below floor — refusing to route "
                "a truncated fragment of a risk-balanced solve",
                extra={
                    "submit_coverage": round(_submit_coverage, 4),
                    "floor": config.DEFAULT_MIN_ENTRY_SUBMIT_COVERAGE,
                    "n_priced_targets": _n_universe_targets,
                    "n_submitted": _n_submitted,
                    "n_skipped_quote_guard": _n_skipped_quote_guard,
                    "n_broker_failed": _n_broker_failed,
                },
            )
            # Cancel any entries that DID land on the wire.
            _submitted_oids = [
                e.get("order_id") for e in entries
                if e.get("order_id") is not None
            ]
            _n_cancel_ok = 0
            _n_cancel_failed = 0
            for _oid in _submitted_oids:
                try:
                    broker.cancel(_oid)
                    _n_cancel_ok += 1
                except Exception as _ex:
                    _n_cancel_failed += 1
                    logger.warning(
                        "mass-skip cancel: broker.cancel failed",
                        extra={"order_id": _oid, "error": str(_ex)[:200]},
                    )
            logger.error(
                "mass-skip abort: submitted entries cancelled; exiting rc=10",
                extra={
                    "n_cancel_attempted": len(_submitted_oids),
                    "n_cancel_ok": _n_cancel_ok,
                    "n_cancel_failed": _n_cancel_failed,
                },
            )
            _stop_feed_safe(feed)
            return 10

        # --------------------------------------------------------------
        # LEG 2: PER-SYMBOL FILL-GATED MOC + DEADLINE SWEEP
        # --------------------------------------------------------------
        # Concurrent (from the operator's perspective) fill-gated MOC:
        # each entry's opposite-side MOC fires the moment its entry
        # becomes terminal Filled. Sized off a FRESH
        # broker.get_positions() for that symbol so a partial LMT fill
        # cannot over-flatten into an opposite overnight exposure.
        # Runs single-threaded (ib_async is not thread-safe) but polls
        # all entries concurrently in a tight loop — no symbol waits
        # for any other.
        #
        # Sweep deadline is computed dynamically from book size (see
        # ``_compute_sweep_deadline``) so the same code path scales
        # from today's 49-name book to a future 500-name book without
        # a manual tuning knob. Above the sweep deadline, the sweep
        # cancels any remaining unfilled entry, takes a fresh
        # get_positions, and submits an MOC for every managed-universe
        # symbol still holding a nonzero position that isn't already
        # covered — this is what guarantees a residual/baseline
        # position (from a prior day, or from a symbol whose entry
        # didn't fill today) still gets flattened.
        # --------------------------------------------------------------
        # Book size the sweep budgets for = the full managed universe.
        # The sweep must be able to fire an MOC for every possibly-held
        # managed symbol; the actual number of sweep MOCs is usually
        # smaller (dispatcher covers most), but budgeting for the whole
        # universe fails safe.
        n_managed_symbols = len(universe_set)
        sweep_deadline, sweep_margin_secs = _compute_sweep_deadline(
            effective_submit_by, n_managed_symbols,
        )
        logger.info(
            "MOC sweep deadline computed",
            extra={
                "sweep_deadline": sweep_deadline.isoformat(),
                "submit_by": effective_submit_by.isoformat(),
                "sweep_margin_secs": sweep_margin_secs,
                "n_managed_symbols": n_managed_symbols,
            },
        )

        (fg_moc_batch, covered_qty_by_sym,
         fg_uncovered_from_submit_error,
         fg_skipped_past_cutoff,
         fg_positions_unknown) = _fill_gated_moc_dispatch(
            broker=broker,
            entries=entries,
            managed_universe=universe_set,
            sweep_deadline=sweep_deadline,
            submit_by=effective_submit_by,
            baseline=current,
        )

        # ROUND-4 SAFETY FIX (SAFETY reviewer finding 3). Historically
        # the "sweep" fired immediately after the dispatcher — on the
        # normal 49-name day that meant ~15:51 with the real
        # ``sweep_deadline`` at ~15:54:42, leaving the 15:51-15:54:42
        # window completely uncovered: cancel-race entry fills after
        # the first snapshot, dispatcher MOCs that flip terminal-
        # rejected between 15:51 and 15:54, and any lag-deferred
        # symbols (finding 2) all had no belt-and-braces coverage
        # until the post-close poll at 16:00-16:05 — well past both
        # cutoffs (entry 15:55, cancel/modify 15:50). Fix: TWO
        # sweeps.
        #   * FIRST sweep runs immediately after the dispatcher. Its
        #     value is EARLY detection so pre-close signals fire in
        #     time for the operator to run ``live.panic`` before
        #     16:00. Cancels non-terminal entries, verifies dispatcher
        #     MOCs, tops-up / trims / re-fires as needed.
        #   * SLEEP until ``sweep_deadline`` (~15:54:42 for 49 names,
        #     ~15:53:00 for 500 names). Pump the broker every 30s so
        #     the ib_async connection is exercised — a wedged
        #     connection would break the second sweep AND the
        #     authoritative post-close poll (finding 4).
        #   * SECOND sweep at the real deadline. Fresh position
        #     snapshot (well past any plausible IB propagation lag),
        #     verifies EVERY prior MOC (dispatcher + first sweep) is
        #     still live, and re-fires for any residual. On a normal
        #     day this issues zero orders — the value is confirmation
        #     under the ACTUAL "book state we hand to 16:00".
        (sweep_moc_batch, cancelled_oids, cancel_failed_syms,
         sweep_skipped, sweep_positions_unknown) = _sweep_moc_deadline(
            broker=broker,
            entries=entries,
            managed_universe=universe_set,
            covered_qty=covered_qty_by_sym,
            submit_by=effective_submit_by,
            dispatcher_legs=fg_moc_batch,
        )

        # Sleep until the real sweep_deadline with a periodic
        # broker.get_positions() pump. If we are already past the
        # deadline (dispatcher exited slow and first sweep took real
        # time), _sleep_until returns immediately.
        #
        # REPLAY EXEMPTION: in replay mode ``sweep_deadline`` is
        # computed off ``effective_submit_by`` which is either today's
        # wall-clock 15:54 (irrelevant to the historical replay date)
        # or ``now + 1h`` (with --replay-ignore-cutoff — an
        # arbitrary neutralised value). Neither makes sense as a
        # wall-clock wait: the historical feed has no late fills, and
        # a real paper broker's rejections come in seconds, not hours.
        # Skip the sleep in replay — the second sweep still runs
        # immediately so the code path is exercised end-to-end.
        if not replay_mode:
            try:
                _sleep_until(
                    sweep_deadline,
                    pump=lambda: broker.get_positions(),
                    pump_interval=30.0,
                )
            except Exception:
                # A raise in the sleep loop itself should not gate
                # the second sweep — press on. The sweep's own error
                # paths are pessimistic.
                logger.exception(
                    "sleep-to-sweep-deadline raised (non-fatal); "
                    "proceeding with second sweep",
                )

        # Combined covered_qty going into the second sweep = dispatcher
        # legs + first-sweep legs, signed.
        combined_covered_qty: dict[str, int] = dict(covered_qty_by_sym)
        for _leg in sweep_moc_batch:
            _sym = _leg["symbol"]
            combined_covered_qty[_sym] = (
                combined_covered_qty.get(_sym, 0)
                + int(_leg["signed_qty"])
            )
        (sweep2_moc_batch, cancelled2_oids, cancel_failed2_syms,
         sweep2_skipped, sweep2_positions_unknown) = _sweep_moc_deadline(
            broker=broker,
            entries=entries,
            managed_universe=universe_set,
            covered_qty=combined_covered_qty,
            submit_by=effective_submit_by,
            dispatcher_legs=fg_moc_batch + sweep_moc_batch,
        )
        # Second sweep only OBSERVES cancel failures on non-terminal
        # entries — after the first sweep those should be terminal
        # already, so this list is expected empty on a normal day.
        # Union defensively.
        cancelled_oids = list(cancelled_oids) + list(cancelled2_oids)
        cancel_failed_syms = list(cancel_failed_syms) + list(cancel_failed2_syms)
        # Combined skipped-past-cutoff = dispatcher's per-tick
        # refusals + both sweeps' own past-cutoff refusals. Any is an
        # exposure signal.
        moc_skipped: list[str] = (
            list(fg_skipped_past_cutoff)
            + list(sweep_skipped)
            + list(sweep2_skipped)
        )

        moc_batch = (
            list(fg_moc_batch)
            + list(sweep_moc_batch)
            + list(sweep2_moc_batch)
        )
        moc_positions_unknown = bool(
            fg_positions_unknown
            or sweep_positions_unknown
            or sweep2_positions_unknown
        )
        moc_order_ids = [leg["oid"] for leg in moc_batch]
        # ROUND-3 SAFETY FIX (SAFETY + CORRECTNESS reviewer finding 2/6):
        # accumulate signed_qty per symbol rather than last-wins, AND
        # exclude MOC legs that are already visibly terminal-not-Filled
        # at coverage-check time. Two reasons:
        #   * The round-2 dispatcher-then-sweep pipeline deliberately
        #     emits TWO MOC legs per symbol on the partial-fill
        #     top-up path (dispatcher SELL 60 + sweep SELL 40 for a
        #     held 100 = correctly zero at close). A dict
        #     comprehension keyed by symbol would drop the
        #     dispatcher leg, causing _coverage_cross_check to
        #     under-count coverage and spuriously page rc=11 on the
        #     fix's OWN success path.
        #   * The round-3 sweep re-fires when the dispatcher's MOC
        #     was terminal-rejected (SAFETY finding 3). If we
        #     blindly summed every submitted signed_qty here, a
        #     symbol whose dispatcher AND sweep MOCs both got
        #     rejected would show up in moc_signed_by_sym with
        #     twice the intended coverage — the coverage
        #     cross-check would then report the symbol as uncovered
        #     at 2*|held| under-coverage even though the truth
        #     (both rejections leave the position 100% uncovered)
        #     is that exposure equals held. We use a local
        #     get_order_status re-poll per leg (O(N) local dict
        #     reads, negligible) to only count MOCs currently in a
        #     good live state.
        moc_signed_by_sym: dict[str, int] = {}
        for _leg in moc_batch:
            _sym = _leg["symbol"]
            _oid = _leg["oid"]
            _include = True
            try:
                _info = broker.get_order_status(_oid)
                _st = getattr(_info, "status", "") or ""
                _fl = int(getattr(_info, "filled", 0) or 0)
                # Terminal-rejected (Cancelled/Inactive with 0 fills)
                # covers nothing.
                if _st in _TERMINAL_ORDER_STATES and _fl == 0:
                    _include = False
            except Exception:
                # If we can't poll, INCLUDE — pessimistic direction
                # for coverage (assume the MOC is live, so downstream
                # signals see it as coverage; if it isn't, other
                # signals will fire).
                _include = True
            if _include:
                moc_signed_by_sym[_sym] = (
                    moc_signed_by_sym.get(_sym, 0)
                    + int(_leg["signed_qty"])
                )
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
                    exchange=ACTIVE_ROUTING,
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

        # RR1: GROUND-TRUTH coverage cross-check. Independent of the
        # dispatcher (which can leave an entry in
        # ``pending_at_sweep_deadline`` with ``filled_qty=0`` even when
        # the exchange fill landed after the deadline) AND independent
        # of any get_positions snapshot (which can return partial).
        # Re-polls each entry's true fill
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
        # — with a LMT-entered book riding the auction unhedged. Detect
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
        # exchange ack. A MOC submitted past NASDAQ's 15:55:00 ENTRY
        # cutoff (SEC 34-84454) is API-accepted then Cancelled at
        # exchange. Wait a bounded window for terminal state; anything
        # terminal-not-Filled is a rejection. See _audit_moc_acceptance
        # for the trade-off between short wait (fast operator feedback)
        # and full 15:55->16:00 wait (fully safe but slow).
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

        # PRE-CLOSE exposure signals — the EARLY warning signals. These
        # existed in the pre-refactor code as the SOLE authority; now
        # they short-circuit to rc=11 IMMEDIATELY so the operator has
        # time to run ``live.panic`` before 16:00. The AUTHORITATIVE
        # verdict — the post-close position poll — runs later on the
        # non-exposed path. See the "Reconciliation" note below.
        #
        # NOTE for future reviewers: any signal known here that has
        # ALREADY declared exposure is worth paging on RIGHT NOW.
        # Waiting 5-8 minutes for the post-close poll to concur adds
        # zero information (the poll is monotonic vs these signals —
        # each of them is a superset of "definitely exposed" that the
        # poll cannot revoke) and burns the operator's remediation
        # window.
        # Cancel failures during the sweep are a NEW pre-close exposure
        # signal (round-2 fix). An entry we could not cancel may still
        # fill in the [sweep, 16:00] window without an MOC cover —
        # that's a naked overnight position.
        # ROUND-4 SAFETY FIX (SAFETY reviewer finding 1). The
        # dispatcher's ``fg_uncovered_from_submit_error`` list carries
        # symbols whose per-symbol MOC submission RAISED at the broker
        # (a ``broker.buy``/``sell`` exception AFTER TWS accepted the
        # request is possible — a read timeout AFTER the exchange
        # queued the order looks the same to the caller). Ignoring it
        # in the exposure signal reintroduced the "neither in moc_batch
        # nor in skipped" gap: the symbol has no MOC in flight FROM
        # THIS PATH, the sweep would then see held=+q, covered=0, and
        # fire a SECOND MOC — a double-flatten to the naked-short
        # opposite side. Route the signal into the pre-close exposure
        # decision so the operator knows to run panic BEFORE 16:00.
        pre_close_exposure = bool(
            moc_skipped or moc_positions_unknown
            or empty_snapshot_after_fills or uncovered_symbols
            or unfilled_moc_syms or cancel_failed_syms
            or fg_uncovered_from_submit_error
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
        target_record["cancel_failed_symbols"] = sorted(set(cancel_failed_syms))
        target_record["sweep_deadline_ny"] = sweep_deadline.isoformat()
        target_record["sweep_margin_secs"] = sweep_margin_secs
        target_record["covered_by_fill_gated"] = sorted(covered_qty_by_sym)
        # Round-4: persist the fill-gated submission-error list so post-
        # trade analysis can see WHICH symbols the dispatcher failed to
        # queue an MOC for. Feeds the pre-close exposure signal above.
        target_record["fill_gated_submit_errors"] = sorted(
            set(fg_uncovered_from_submit_error)
        )
        try:
            _write_target_record(target_record)
        except Exception:
            logger.exception("target record write failed (non-fatal)")

        # R3 + RR1 + RRR1: distinct rc for the OVERNIGHT-EXPOSURE state.
        # rc=11 is THE pager predicate (per CLAUDE.md), so NOTHING
        # recording-related may run before this return — a marks
        # wait or a MOC fill audit here would delay the operator's
        # window to run ``live.panic`` before the 16:00 close.
        if pre_close_exposure:
            panic_cmd = ("python -m live.panic --paper --flatten-with MKT "
                         "(or --live --live-confirm=... --yes for live)")
            unfilled_list = sorted(unfilled_moc_syms)
            cancel_failed_list = sorted(set(cancel_failed_syms))
            fg_submit_error_list = sorted(set(fg_uncovered_from_submit_error))
            logger.error(
                "OVERNIGHT EXPOSURE (pre-close) — MOC exit incomplete. "
                "RUN: %s. Uncovered symbols: %s. MOC rejected at "
                "exchange: %s. Skipped: %s. Positions unknown: %s. "
                "Empty snapshot after fills: %s. Cancel-failed entries: %s. "
                "Fill-gated submission errors: %s.",
                panic_cmd,
                uncovered_symbols[:20] or "(see logs above)",
                unfilled_list[:20] or "(none)",
                moc_skipped[:20],
                moc_positions_unknown,
                empty_snapshot_after_fills,
                cancel_failed_list[:20] or "(none)",
                fg_submit_error_list[:20] or "(none)",
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
                    "cancel_failed_symbols": cancel_failed_list[:20],
                    "n_cancel_failed": len(cancel_failed_list),
                    "fill_gated_submit_errors": fg_submit_error_list[:20],
                    "n_fill_gated_submit_errors": len(fg_submit_error_list),
                    "panic_command": panic_cmd,
                },
            )
            return 11

        # Non-exposure path from here on: post-close position poll +
        # recorder finalization work. None of this runs on the pre-close
        # rc=11 path (above) so it can't delay the operator page.

        # Stop the live feed EARLY — as soon as the recorder's marks
        # background thread has landed (or given up). Otherwise the
        # Databento buffers accumulate the whole closing auction's
        # worth of tbbo per name in the meantime, which is pure memory
        # growth on a long-running live box. On replay this is a no-op
        # (the feed was already stopped at feature-build time).
        if recorder is not None:
            # Bounded wait: 5min for the T+MAX target instant + some
            # slack for the snapshot itself. This wait is AFTER the
            # pre-close rc=11 gate above, so a delay here never gates
            # the operator page for pre-close signals.
            #
            # ROUND-4 SAFETY FIX (SAFETY reviewer finding 4). Use the
            # pumped variant so a ~350s ``threading.Event.wait`` cannot
            # starve the ib_async event loop with the connection open
            # — a wedged connection would break the post-close poll
            # (now the sole authoritative verdict) and spuriously
            # page rc=11 on a genuinely flat book.
            try:
                _wait_for_marks_pumped(
                    recorder=recorder,
                    broker=broker,
                    timeout=350.0,
                    pump_interval=30.0,
                )
            except Exception:
                logger.exception(
                    "pumped marks wait failed (non-fatal); proceeding to "
                    "post-close poll",
                )
        _stop_feed_safe(feed)

        # POST-CLOSE POSITION POLL — the AUTHORITATIVE exposure verdict.
        # Runs whenever the process reached the broker (both live/paper
        # AND ``--replay-submit`` — the historical feed is unrelated to
        # the wall clock but the orders are real, so the same post-
        # auction verification is required). A broker-free replay
        # (default replay without ``--replay-submit``) short-circuited
        # much earlier at rc=0 and never gets here.
        #
        # Round-2 guards against a stale/early FLAT verdict:
        #   1. ``not_before`` — refuse FLAT until >= 16:00 + settle,
        #      so a code path that reached the poll before the auction
        #      cannot declare flat on a pre-auction empty snapshot.
        #   2. ``min_consecutive_flat_reads`` — need N=2 confirmations
        #      so a transient empty response is not authoritative.
        #   3. ``poll_until`` upper-bound clamp — never spin beyond
        #      ``config.POST_CLOSE_POLL_MAX_WINDOW_MINS`` past NOW,
        #      even if the configured deadline is far in the future
        #      (config typo, clock skew).
        #
        # Round-3 fixes on top:
        #   * The whole poll block is wrapped in try/except; any
        #     exception is treated as pessimistic EXPOSED (SAFETY
        #     reviewer finding 4). Without this, a raise from
        #     config.parse_hhmm / config.today_ny / the poll itself
        #     would propagate to main() and yield rc=1 — below the
        #     rc>=11 pager threshold, silently degrading a genuine
        #     exposure to a non-page.
        #   * If ``now < not_before`` at entry (a rehearsal or a run
        #     that raced ahead), we either sleep until ``not_before``
        #     (live/paper — the auction hasn't printed yet; MUST
        #     wait) or SKIP the poll with a WARNING (replay-submit —
        #     the historical replay date is unrelated to today's
        #     auction; the paper positions are real but no post-
        #     auction print is coming for hours). The old code let
        #     the loop spin the whole MAX_WINDOW then falsely return
        #     "insufficient_flat_confirmations" (CORRECTNESS
        #     reviewer finding 7).
        #   * After any sleep we RE-CLAMP poll_until so the window
        #     never closes before we have room for
        #     ``min_consecutive_flat_reads`` reads.
        post_close_exposed = False
        post_close_nonzero_syms: list[str] = []
        post_close_error: Optional[str] = None
        post_close_skipped_reason: Optional[str] = None
        try:
            post_close_until_hh, post_close_until_mm = config.parse_hhmm(
                config.DEFAULT_POST_CLOSE_POLL_UNTIL_TIME,
            )
            post_close_until_configured = config.today_ny(
                post_close_until_hh, post_close_until_mm,
            )
            not_before_hh, not_before_mm = config.parse_hhmm(
                config.DEFAULT_POST_CLOSE_POLL_NOT_BEFORE_TIME,
            )
            post_close_not_before = config.today_ny(
                not_before_hh, not_before_mm,
            ) + timedelta(
                seconds=config.DEFAULT_POST_CLOSE_POLL_SETTLE_SECS,
            )
            _now_ny = config.now_ny()
            # Replay rehearsal short-circuit: a --replay-submit run
            # that reached here BEFORE the wall-clock auction time
            # cannot verify by definition (real paper orders were
            # placed, but the historical date's close is not today's
            # close). Skip with a WARNING; the operator MUST run
            # panic to flatten the paper account before the next
            # real session (see live/README.md replay section).
            if replay_mode and _now_ny < post_close_not_before:
                post_close_skipped_reason = "replay_pre_not_before"
                logger.warning(
                    "post-close position poll SKIPPED — replay-submit "
                    "rehearsal reached the poll before today's 16:00 "
                    "auction wall clock. Paper positions from this "
                    "rehearsal are LIVE in the paper account; run "
                    "``python -m live.panic --paper`` to flatten "
                    "before the next real session.",
                    extra={
                        "now": _now_ny.isoformat(),
                        "not_before": post_close_not_before.isoformat(),
                        "replay_date": args.replay_date,
                    },
                )
            else:
                if _now_ny < post_close_not_before:
                    # Live/paper: the auction hasn't printed yet.
                    # Sleep to ``not_before`` — this is the honest
                    # answer, and matches the design that the
                    # post-close poll is the AUTHORITATIVE verdict.
                    #
                    # ROUND-4 SAFETY FIX (SAFETY reviewer finding 4).
                    # Pump the broker every 30s so the ib_async loop
                    # is serviced during this potentially-long
                    # ``time.sleep``. Without the pump, a wedged
                    # connection here would flip the entire post-
                    # close poll into its pessimistic-EXPOSED branch,
                    # paging rc=11 on a genuinely flat book.
                    logger.info(
                        "post-close position poll: waiting for "
                        "not_before wall clock",
                        extra={
                            "now": _now_ny.isoformat(),
                            "not_before": post_close_not_before.isoformat(),
                            "wait_secs": (
                                post_close_not_before - _now_ny
                            ).total_seconds(),
                        },
                    )
                    _sleep_until(
                        post_close_not_before,
                        pump=lambda: broker.get_positions(),
                        pump_interval=30.0,
                    )
                    _now_ny = config.now_ny()
                # Compute a poll_until that (a) never exceeds the
                # configured DEFAULT_POST_CLOSE_POLL_UNTIL_TIME, (b)
                # never spins beyond POST_CLOSE_POLL_MAX_WINDOW_MINS
                # from NOW, and (c) is always at least long enough
                # for ``min_consecutive_flat_reads`` reads to have a
                # chance — otherwise we'd fail-closed on the narrow
                # "reached the poll at 16:04:59" window (~5s), which
                # is spurious rc=11.
                min_reads = int(
                    config.DEFAULT_POST_CLOSE_MIN_CONSECUTIVE_FLAT_READS
                )
                min_window_secs = max(
                    2.0,
                    (min_reads + 1)
                    * float(
                        config.DEFAULT_POST_CLOSE_POLL_MAX_INTERVAL_SECS
                    ),
                )
                post_close_max_window_end = _now_ny + timedelta(
                    minutes=config.POST_CLOSE_POLL_MAX_WINDOW_MINS,
                )
                post_close_until = min(
                    post_close_until_configured,
                    post_close_max_window_end,
                )
                # Clamp UP: at least min_window_secs from now, but
                # never beyond the MAX_WINDOW ceiling (the ceiling
                # is the hard safety cap against configuration typos).
                post_close_until = max(
                    post_close_until,
                    _now_ny + timedelta(seconds=min_window_secs),
                )
                post_close_until = min(
                    post_close_until, post_close_max_window_end,
                )
                (post_close_exposed, post_close_nonzero_syms,
                 post_close_error) = _poll_positions_after_close(
                    broker=broker,
                    managed_universe=universe_set,
                    poll_until=post_close_until,
                    not_before=post_close_not_before,
                    min_consecutive_flat_reads=min_reads,
                    # "book_was_opened" must mean "a position could
                    # exist", NOT "an entry filled today". A residual
                    # carried in from a prior day is live even when
                    # nothing filled today — ``_reconcile_from_previous``
                    # only LOGS drift, it never trades — so a nonzero
                    # managed baseline counts as an opened book.
                    # Separately, ``moc_batch`` holds only legs that were
                    # successfully sent: a skipped or errored leg appends
                    # nothing, so "every leg in moc_batch filled" proves
                    # only "every MOC I managed to send filled", NOT
                    # "every position was covered". If anything was
                    # skipped we therefore refuse to corroborate at all.
                    corroborator=lambda: (
                        None if moc_skipped
                        else _moc_legs_corroborate_flat(
                            broker,
                            moc_batch,
                            n_filled_entries > 0 or any(
                                q != 0
                                for s, q in current.items()
                                if s in universe_set
                            ),
                        )
                    ),
                )
        except Exception as exc:
            # Any raise in the poll block cannot silently downgrade to
            # rc=1 — the poll is the sole authoritative post-auction
            # verdict, so inability to run it MUST page.
            logger.exception(
                "post-close position poll block raised — pessimistic "
                "verdict: EXPOSED (cannot verify book)",
            )
            post_close_exposed = True
            post_close_error = f"poll_block_raised: {str(exc)[:200]}"
            post_close_nonzero_syms = []

        target_record["post_close_nonzero_symbols"] = post_close_nonzero_syms
        target_record["post_close_poll_error"] = post_close_error
        target_record["post_close_skipped_reason"] = post_close_skipped_reason
        try:
            _write_target_record(target_record)
        except Exception:
            logger.exception(
                "target record write failed (non-fatal, post-close)"
            )
        if post_close_exposed:
            panic_cmd = (
                "python -m live.panic --paper --flatten-with MKT "
                "(or --live --live-confirm=... --yes for live)"
            )
            logger.error(
                "OVERNIGHT EXPOSURE (post-close) — managed-universe "
                "positions still non-flat after 16:00 auction, or "
                "post-close position poll failed. RUN: %s. "
                "Nonzero symbols: %s. Poll error: %s.",
                panic_cmd,
                post_close_nonzero_syms[:20] or "(unknown — poll failed)",
                post_close_error or "(none)",
                extra={
                    "n_nonzero_post_close": len(post_close_nonzero_syms),
                    "post_close_nonzero_symbols": post_close_nonzero_syms[:20],
                    "post_close_error": post_close_error,
                    "panic_command": panic_cmd,
                },
            )
            return 11

        # M2 (part 3): JSONL fill audit — best-effort logging only,
        # NOT tied to the rc gate (RRR2). MOC fills happen at 16:00
        # close, so cap the wait at 15 minutes past submit-by to avoid
        # the process lingering past the trading day. The post-close
        # position poll above is the rc-gating check.
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
    except Exception as exc:
        # LAST-RESORT EXPOSURE GATE.
        #
        # Everything from the pre-trade reconcile through the post-close
        # position poll lives in this ``try``. It previously had a
        # ``finally`` but NO ``except``, so ANY unhandled raise between
        # entry submission and the rc gate propagated out of main() and
        # the process exited rc=1 — BELOW the ``rc >= 11`` pager
        # predicate — while a filled book sat uncovered. The concrete
        # path: IB hands back a position row whose ``quantity`` is None
        # and ``int(p.quantity)`` raises TypeError inside the sweep,
        # after LEG 1 has already filled.
        #
        # rc=1 is only safe BEFORE any order is live. ``orders_may_be_live``
        # is armed immediately before the LEG 1 submission call (not
        # after it — a partial batch that raises mid-submission has
        # already put orders on the book), so the two cases separate
        # cleanly:
        #   * armed   -> a book may exist with no verified MOC cover.
        #                Page. rc=11.
        #   * unarmed -> nothing was ever submitted. Safe failure, rc=1,
        #                which matches the pre-order refusal codes.
        #
        # This runs BEFORE the ``finally`` below, so the operator page is
        # emitted ahead of recorder finalization / feed teardown, exactly
        # as the in-band rc=11 returns do.
        logger.exception(
            "UNHANDLED EXCEPTION in the order path",
            extra={"orders_may_be_live": orders_may_be_live},
        )
        if orders_may_be_live:
            panic_cmd = (
                "python -m live.panic --paper --flatten-with MKT "
                "(or --live --live-confirm=... --yes for live)"
            )
            logger.error(
                "OVERNIGHT EXPOSURE (unhandled exception) — the order "
                "path raised AFTER entry submission, so the book state "
                "is UNKNOWN and may be live with no MOC cover. RUN: %s. "
                "Exception: %s",
                panic_cmd,
                str(exc)[:300],
                extra={
                    "panic_command": panic_cmd,
                    "exception_type": type(exc).__name__,
                    "exception": str(exc)[:300],
                },
            )
            return 11
        logger.error(
            "order path raised BEFORE any order was submitted — no book "
            "can be live; exiting rc=1 (safe, non-paging)",
            extra={"exception_type": type(exc).__name__},
        )
        return 1
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
        # A bare raise here would DISCARD an in-flight ``return 11`` and
        # propagate out of main() as rc=1 — the exact hole the order-path
        # ``except`` above closes, re-opened through the ``finally``. The
        # risk is correlated, not theoretical: the TWS drop that makes a
        # post-close verdict pessimistic is the same event that makes
        # ``disconnect()`` likely to raise.
        try:
            broker.disconnect()
        except Exception:
            logger.exception("broker disconnect failed (non-fatal)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
