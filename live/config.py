"""Shared configuration for the live pipeline.

Centralises the small handful of "cross-script" constants and helper
functions the two entry points (``daily_orchestrator``, ``retrain``) both
need: repo paths, environment loading, universe loader, artifact filenames.

Deliberately dependency-light so importing it never triggers the heavy
Databento / cvxpy / sklearn import chain.
"""
from __future__ import annotations

import logging
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Absolute repo root. ``config.py`` lives at ``<repo>/live/config.py``, so the
# repo root is two ``.parent`` steps up. We resolve once at import time so
# every downstream ``Path`` is anchored regardless of the caller's cwd — the
# daily orchestrator and retrain scripts are meant to be launched from any
# working directory (systemd, cron, terminal), not just the repo root.
REPO_ROOT: Path = Path(__file__).resolve().parents[1]

CACHE_DIR: Path = REPO_ROOT / "data" / "dbn_cache" / "generic"
COV_DIR: Path = REPO_ROOT / "covariance"
FORECAST_DIR: Path = REPO_ROOT / "forecast"
IMPACT_DIR: Path = REPO_ROOT / "impact"
LIVE_LOG_DIR: Path = REPO_ROOT / "live" / "logs"

# ---------------------------------------------------------------------------
# Artifact filenames
#
# Kept in one place so the daily script and the retrain script never disagree
# on where an artifact lives. Names match what ``covariance/covariance.ipynb``,
# ``forecast/forecasting.ipynb`` and ``impact/impact_estimation.ipynb`` write
# today.
# ---------------------------------------------------------------------------

COV_LOADINGS_NAME = "loadings_return.npy"
COV_EPS_DIAG_NAME = "idiosyncratic_diag_return.npy"
COV_FACTOR_COV_NAME = "factor_cov_matrix.npy"
COV_MATRIX_NAME = "cov_matrix_return.npy"
COV_NAME_VOL_NAME = "name_vol.npy"

FORECAST_PIPE_NAME = "forecast_pipe.joblib"

IMPACT_GAMMA_NAME = "impact_gamma_vec.npy"
IMPACT_K_NAME = "impact_k_vec.npy"

# Per-symbol reference stats (ADV$ + daily return vol) produced by retrain and
# consumed by the daily orchestrator. Small parquet — one row per active-set
# symbol — so the daily script can start without doing a fresh Databento pull.
REF_STATS_NAME = "reference_stats.parquet"

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------

# The saved covariance, forecast, and impact artifacts on disk are all sized
# for the currently-active tiered universe. As of 2026-07-21 that is the
# top-49 file — chosen for paper trading to avoid known universe issues
# (see the notes in ``live/nasdaq_routing.py``); the plan is to scale
# toward 500 later. Changing this without simultaneously rebuilding all
# three artifact sets will produce shape mismatches at solve time.
DEFAULT_ACTIVE_UNIVERSE_FILE = "top_49_universe.txt"

# Broad symbol pool used for historical Databento pulls. The ``CACHE_DIR``
# parquet files are keyed on the ticker list, so the retrain has to pass
# exactly this list to hit the existing cache slots.
DEFAULT_BROAD_UNIVERSE_FILE = "universe.txt"

# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

NY_TZ = ZoneInfo("America/New_York")

# Default MOC submission window. NASDAQ's MOC ENTRY cutoff is 15:55:00 ET
# (SEC release 34-84454 lifted the old 15:50 rule in 2018); MOC orders may
# not be CANCELED OR MODIFIED after 15:50:00 ET. NYSE keeps a 15:50 MOC
# entry cutoff, but this strategy trades the NASDAQ close cross exclusively
# (see ``live/nasdaq_routing.py``), so 15:55 is the load-bearing entry
# deadline. We solve at 15:50 (that is when the imbalance print is
# available) and hard-stop MOC submission at 15:55 — the LAST moment
# NASDAQ still accepts an MOC.
DEFAULT_REBALANCE_TIME = "15:50"
DEFAULT_SUBMIT_BY_TIME = "15:55"
DEFAULT_STREAM_START_TIME = "09:30"

# Commission model used by the offline optimizer notebook — kept in sync
# here so the live optimizer produces the same targets.
COMMISSION_PER_SHARE = 0.0035

# Optimizer hyperparameters, matched to the notebook's "current best" values.
# Real deployments should sweep these on the historical data and override via
# CLI; these are just safe defaults.
DEFAULT_LAMBDA = 5e-5
DEFAULT_VOL_CAP = 5e3
DEFAULT_IMPACT_GAMMA = 0.5

# Two-leg execution: MKT entry at 15:50, MOC counter-flatten at the 16:00
# auction (see daily_orchestrator._submit_mkt_entries /
# _fill_gated_moc_dispatch / _sweep_moc_deadline).
# The strategy predicts the 15:50 -> 16:00 close-auction move; we take that
# move with an MKT entry that lands intraday and an MOC exit that lifts at
# the closing print. Per-order fill-wait bounds the fill-gated MOC
# dispatcher — as soon as any entry becomes terminal Filled, the fill-gated
# path submits its opposite-side MOC IMMEDIATELY. Set to 30s to give slow
# venues headroom; the dispatcher still respects the sweep deadline (see
# ``_compute_sweep_deadline``) so a slow tape can never delay the sweep
# past NASDAQ's 15:55:00 MOC entry cutoff.
DEFAULT_ENTRY_FILL_TIMEOUT_SECS = 30.0

# RRR1: bounded wall-clock wait for MOC acceptance verification. TWS's
# placeOrder returns synchronously BEFORE any exchange ack; a rejected
# MOC (submitted after the ~15:55 NASDAQ exchange entry cutoff) becomes
# terminal Cancelled/Inactive within seconds to a minute. We wait up to
# this many seconds for any MOC to reach terminal-not-Filled state;
# anything still Submitted at the deadline is presumed accepted (will
# fill at 16:00). Trade-off: waiting the full ~5-8 min until 16:00 would
# be stricter but slows operator feedback with no rejection detected. In
# practice the definitive verdict now comes from the post-close position
# poll (see ``DEFAULT_POST_CLOSE_POLL_*``), so this audit is an EARLY
# warning signal, not the authority.
DEFAULT_MOC_ACK_TIMEOUT_SECS = 60.0

# Fill-gated MOC dispatcher — sweep-deadline margin formula.
#
# We must NOT let a slow tape delay any managed-universe symbol's MOC
# past NASDAQ's 15:55:00 entry cutoff. The dispatcher submits each
# symbol's MOC as soon as its entry becomes terminal; a sweep at
# ``submit_by - sweep_margin`` cancels remaining unfilled entries and
# submits an MOC for every managed-universe symbol still holding a
# nonzero position that has not already been covered.
#
# The margin must accommodate:
#   * every ``broker.get_positions()`` call (each contains an ib_async
#     ``self._ib.sleep(0.5)`` — hard-coded broker cost)
#   * every ``broker.get_order_status`` call inside the sweep's
#     belt-and-braces cancel loop
#   * every MOC ``placeOrder`` call (~0.13s observed in paper mode).
#
# Formula (round-2 fix — earlier version budgeted only the placeOrder
# cost and missed the get_positions/status costs, so at 500 names the
# real work would exceed the budgeted margin):
#
#     margin = max(SWEEP_MIN_MARGIN_SECS,
#                  n * (SWEEP_PER_ORDER_SECS + SWEEP_PER_STATUS_SECS)
#                  + 2 * SWEEP_GET_POSITIONS_SECS
#                  + SWEEP_FIXED_OVERHEAD_SECS)
#
# At 49 symbols → 15.0s (floor). At 500 symbols → 500*(0.13+0.1) + 2*0.6
# + 5 ≈ 121s. The dispatcher additionally consolidates its
# per-symbol get_positions calls into ONE per tick (see
# ``_fill_gated_moc_dispatch``) so the pre-sweep window is bounded
# independently of book size.
SWEEP_MIN_MARGIN_SECS = 15.0
SWEEP_PER_ORDER_SECS = 0.13
SWEEP_PER_STATUS_SECS = 0.1
SWEEP_GET_POSITIONS_SECS = 0.6
SWEEP_FIXED_OVERHEAD_SECS = 5.0

# Post-close position verification — poll ``broker.get_positions()``
# after the 16:00 auction print, retry with exponential backoff up to
# ``DEFAULT_POST_CLOSE_POLL_UNTIL_TIME`` ET so IB has time to reflect
# the auction fill. ANY nonzero managed-universe position after the
# deadline → rc=11 (real overnight exposure). A poll that fails
# outright is pessimistic — treat as exposure. This replaces the
# pre-16:00 inference-based verdict as the AUTHORITATIVE exposure
# signal.
#
# Two extra guards on top of the raw poll:
#
# * ``DEFAULT_POST_CLOSE_POLL_NOT_BEFORE_TIME`` — the poll REFUSES to
#   declare FLAT before this wall-clock time (default 16:00 ET plus
#   ``DEFAULT_POST_CLOSE_POLL_SETTLE_SECS``). Without this a poll that
#   reached the code path early (e.g. recorder short-circuit at 15:53)
#   could declare FLAT on a stale/empty response — the exact bug the
#   ``empty_snapshot_after_fills`` guard exists to defend against.
# * ``DEFAULT_POST_CLOSE_MIN_CONSECUTIVE_FLAT_READS`` — require N
#   consecutive successful all-flat reads before returning FLAT (default
#   2). A transient empty/partial from a broker that returned rows on
#   a prior poll is suspect, not authoritative.
DEFAULT_POST_CLOSE_POLL_UNTIL_TIME = "16:05"
DEFAULT_POST_CLOSE_POLL_INITIAL_INTERVAL_SECS = 5.0
DEFAULT_POST_CLOSE_POLL_MAX_INTERVAL_SECS = 30.0
DEFAULT_POST_CLOSE_POLL_NOT_BEFORE_TIME = "16:00"
DEFAULT_POST_CLOSE_POLL_SETTLE_SECS = 30.0
DEFAULT_POST_CLOSE_MIN_CONSECUTIVE_FLAT_READS = 2
# Hard upper bound on wall-clock time spent in the post-close poll —
# guards against the poll deadline being far in the future (a clock
# skew, a run started late, or a config typo) and the loop spinning
# for hours. Independent of NY_TZ — pure wall-clock minutes past
# ``config.now_ny()`` at the moment the poll is entered.
POST_CLOSE_POLL_MAX_WINDOW_MINS = 15.0

# Rolling windows for retrain. Covariance uses a longer window because it
# needs enough daily bars to estimate 10 factors; forecast/impact use a
# shorter window that stays inside the "regime we currently believe in".
DEFAULT_COVARIANCE_WINDOW_DAYS = 730  # ~24 months
DEFAULT_FORECAST_WINDOW_DAYS = 365    # ~12 months
DEFAULT_IMPACT_WINDOW_DAYS = 365      # ~12 months


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def load_env() -> None:
    """Load ``.env`` from the repo root (idempotent)."""
    load_dotenv(REPO_ROOT / ".env")


def resolve_databento_key() -> str:
    """Return the Databento API key, honouring both env-var spellings.

    The offline notebooks use ``DATABENTO-API-KEY`` (with hyphens) while the
    adapter's ``_resolve_api_key`` and standard ``.env`` files use
    ``DATABENTO_API_KEY``. We prefer the underscore form (per ``CLAUDE.md``)
    and fall back to the hyphen form so the same ``.env`` works for both
    surfaces.
    """
    load_env()
    key = os.getenv("DATABENTO_API_KEY") or os.getenv("DATABENTO-API-KEY")
    if not key:
        raise RuntimeError(
            "DATABENTO_API_KEY is not set. Add it to .env (either "
            "DATABENTO_API_KEY=... or the legacy DATABENTO-API-KEY=...)."
        )
    return key


# ---------------------------------------------------------------------------
# Universe helpers
# ---------------------------------------------------------------------------

def load_universe(file_name: str) -> list[str]:
    """Load a newline-delimited ticker file from the repo root."""
    path = REPO_ROOT / file_name
    if not path.exists():
        raise FileNotFoundError(f"Universe file not found: {path}")
    with path.open("r") as fh:
        return [line.strip() for line in fh if line.strip()]


def load_active_universe(file_name: str = DEFAULT_ACTIVE_UNIVERSE_FILE) -> list[str]:
    """Load the active (per-day tradable) universe."""
    return load_universe(file_name)


def load_broad_universe(file_name: str = DEFAULT_BROAD_UNIVERSE_FILE) -> list[str]:
    """Load the broad pool used for historical Databento cache pulls."""
    return load_universe(file_name)


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def now_ny() -> datetime:
    """Current wall-clock time in America/New_York."""
    return datetime.now(NY_TZ)


def parse_hhmm(spec: str) -> tuple[int, int]:
    """Parse ``"HH:MM"`` into ``(hour, minute)``.

    Raises ``ValueError`` on malformed input rather than silently coercing.
    """
    parts = spec.strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"Expected HH:MM, got {spec!r}")
    h, m = int(parts[0]), int(parts[1])
    if not (0 <= h < 24 and 0 <= m < 60):
        raise ValueError(f"Out-of-range HH:MM: {spec!r}")
    return h, m


def today_ny(hh: int, mm: int) -> datetime:
    """Return today's date at ``HH:MM`` America/New_York."""
    today = now_ny().date()
    return datetime(today.year, today.month, today.day, hh, mm, tzinfo=NY_TZ)


def yesterday_iso() -> str:
    """Yesterday's date as ``YYYY-MM-DD`` — Databento's exclusive ``end_date``.

    Databento's historical batch API rejects an end-date beyond the last full
    trading day it has ingested, so retrain jobs default to "yesterday" and
    let the adapter's cache handle the actual end-of-day boundary.
    """
    return (date.today() - timedelta(days=1)).isoformat()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_JSON_FORMAT_LOGGERS: set[str] = set()


class _JsonLineFormatter(logging.Formatter):
    """Minimal JSON-lines formatter.

    We keep it inline (no dependency on ``python-json-logger``) so the live
    scripts have exactly two runtime dependencies beyond what's already
    installed: ``databento`` and ``ib_async``.

    Emits: ``ts``, ``level``, ``logger``, ``msg``. Any ``extra={...}`` passed
    to the log call is merged in.
    """

    _RESERVED = {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "message", "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        import json
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, NY_TZ).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Merge any structured extras passed via ``logger.info("...", extra=...)``
        for k, v in record.__dict__.items():
            if k in self._RESERVED or k.startswith("_"):
                continue
            try:
                json.dumps(v)  # ensure serialisable
            except TypeError:
                v = repr(v)
            payload[k] = v
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(
    name: str,
    log_dir: Optional[Path] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure JSON-lines file logging + a human-readable stdout stream.

    Idempotent: calling twice with the same ``name`` leaves handlers as-is
    (systemd-style unit files that re-import the script won't multiply logs).

    File is ``<log_dir>/<name>-YYYY-MM-DD.log``.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if name in _JSON_FORMAT_LOGGERS:
        return logger

    log_dir = log_dir or LIVE_LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"

    # L2: nightly rotation with a 30-day retention window. Keeps disk usage
    # bounded on a long-running deployment while still leaving enough
    # history to compare intraday runs across a full working month.
    from logging.handlers import TimedRotatingFileHandler
    file_handler = TimedRotatingFileHandler(
        log_path, when="midnight", backupCount=30, utc=False,
    )
    file_handler.setFormatter(_JsonLineFormatter())
    file_handler.setLevel(level)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    stream_handler.setLevel(level)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    _JSON_FORMAT_LOGGERS.add(name)
    return logger
