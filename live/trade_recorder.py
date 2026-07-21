"""Post-trade recording for slippage / impact analysis.

Records one row per attempted trade to a columnar parquet file. Both legs
of the two-leg execution (MKT entries and MOC counter-flatten) are captured
and identified via a ``leg`` column. Separate files are written for
``paper``, ``live``, and ``replay`` runs so replay rehearsals never
contaminate real trading records.

## Design constraints

- **Never breaks trading.** Every public method is intended to be called
  from inside a ``try/except`` at the call site (or via ``safe_call``),
  because a disk-full / pyarrow / schema error must not prevent orders
  from being sent or the book from being flattened.
- **Staged arrival of columns.** A trade record accretes fields in
  temporal stages:
    stage A — decision time (~15:50:10): arrival mid, bid/ask, spread,
              arrival tick ts, target notional, optimizer solve context,
              leg=entry rows.
    stage B — entry submission (~15:50:11): sent_ts, oid, side.
    stage C — entry fills settle (~15:50:12+): fill qty, fill price,
              commission (from ``broker._trades[oid]``).
    stage D — MOC submission (~15:50:13): leg=moc rows created, oid,
              side, sent_ts.
    stage E — impact-decay curve at T+1..T+5min (~15:55:10): a single
              feed snapshot is taken at T+MAX (~15:55), then filtered
              back to observed mid at (or before) each of decision +
              N minutes for N in ``MARK_OFFSETS_MINUTES``. Each mark
              records its own observed ts and staleness so an analyst
              can filter thin-book rows PER MARK. The T+1 mark reflects
              only data at or before T+1, even though it's CAPTURED at
              T+5 wall clock (property of the per-mark filter, not the
              capture time). See ``schedule_marks_capture_from_feed``.
    stage F — MOC fills settle (~16:00): fill qty, fill price for MOC.
    stage G — finalize (end of run): atomic parquet write.
  Each stage merges into an in-memory row dict keyed by (leg, symbol).
  A JSONL journal captures the full payload of every event so a
  mid-session process crash leaves the raw event log on disk (there
  is no automated rebuilder yet — the journal is a manual-recovery
  input, not a live-restore mechanism).
- **Atomic** writes: ``.tmp`` + ``os.replace``. Never a torn file.
- **Run-scoped output**: ``<mode>/<date>/<run_id>.parquet``. A same-day
  re-run (or a crash-then-restart) never destroys the earlier
  parquet. Analysts CAN concatenate every run file for a date, but
  MUST ``GROUP BY run_id`` before any target-side aggregation —
  ``target_dollars`` / ``target_shares`` are per-run intent and a
  naive sum across two runs of the same day double-counts them.
  ``achieved_dollars`` sums cleanly across the concat (distinct
  ``order_id``, no overlap on real fills). A per-row ``run_id``
  column makes runs distinguishable in-frame.
- **Explicit pyarrow schema.** Every column has a stated type. Nulls in
  an all-null column keep the correct type (pyarrow inference on a
  list-of-dicts would demote them to ``null``).
- **Timezone**: every timestamp column is stored as ``timestamp[us,
  tz=UTC]``. Convert display via the reader; internal representation
  is UTC to sidestep DST edge cases. Naive datetimes are REJECTED at
  the API boundary — silent naive==UTC coercion has bitten us before.

## Reading across schema vintages

Use ``trade_recorder.read_trades(paths)`` — not a raw
``pyarrow.dataset`` or ``pl.read_parquet`` — for any glob that spans
schema versions (a date range straddling a deploy). The helper hands
pyarrow the CURRENT schema so v3 files back-fill the new v4 columns
as typed nulls; the naive readers infer from the first fragment and
either silently drop the new columns (analyst reads T+5 only and
calibrates a bogus decay curve) or raise depending on file ordering.
See ``read_trades`` for the failure modes it protects against.

## Column semantics for downstream analysts

- ``decision_ts_utc``: the scheduled 15:50 anchor (``snapshot_at``).
  This is what T+5min is measured against; not the actual snapshot
  completion time.
- ``snapshot_ts_utc``: when the orchestrator actually finished the
  live-feed snapshot. Fill latency = ``fill_ts_utc - snapshot_ts_utc``.
- ``sent_ts_utc``: when the orchestrator called ``broker.buy/sell``.
- ``fill_ts_utc``: TWS's real ``execution.time`` when available;
  ``observed_ts_utc`` (wall clock) is the fallback.
- ``observed_ts_utc``: wall clock when ``_fill_audit`` picked the fill
  up. Always populated on filled/timeout rows.
- ``arrival_ts_utc``: the tbbo tick that produced the arrival mid.
  ``arrival_staleness_secs = (snapshot_ts_utc or decision_ts_utc) -
  arrival_ts_utc``: the baseline is ``snapshot_ts_utc`` when present
  (the actual snapshot completion instant, ~decision + 1-2s) and
  falls back to ``decision_ts_utc`` otherwise; snapshot_ts_utc is
  preferred so a tick delivered just past 15:50:00 doesn't record a
  spuriously-negative staleness (see Round 2 review #7).
- ``arrival_price``: the last mid in the feed's snapshot buffer at
  snapshot completion time. NOT time-filtered to ``decision_ts_utc``
  — the snapshot naturally completes ~1-10s past the decision anchor
  depending on feed latency. So the realized offset between arrival
  and the first mark is not exactly 60s; it varies with per-run
  snapshot latency. Use ``t{n}min_ts_utc - arrival_ts_utc`` as the
  x-axis for a decay-curve fit, NOT ``n * 60``.
- ``t{n}min_ts_utc`` (n in 1..5): the tbbo tick that produced the
  T+Nmin mid. ``t{n}min_staleness_secs = <intended T+N target> -
  t{n}min_ts_utc`` — an illiquid name whose last print was at 15:50:02
  yields the same ``t1..t5min_price`` (== ``arrival_price``, by
  construction) with monotonically-increasing staleness so the analyst
  can filter it out per mark. The impact-decay curve for a single
  trade is ``[t1min_price, t2min_price, ..., t5min_price]``, each
  paired with its own timestamp and staleness gap. **The x-axis
  distances between marks are NOMINALLY 60s each, but the REALIZED
  elapsed times are ``t{n}min_ts_utc - arrival_ts_utc`` — use those
  for the decay fit; both endpoints drift (arrival by snapshot
  latency, marks by book thinness), and fitting on the nominal grid
  biases the decay exponent upward.**
- ``rel_spread_at_decision``: FULL relative spread ``(ask-bid)/mid``,
  NOT half. Same definition as ``features.py::spread_return`` (which
  has an inaccurate half-spread comment in it; that fix is out of
  scope for this module).
- ``bid_at_decision`` / ``ask_at_decision`` are DERIVED from the mid
  and the relative spread — they are a symmetric pair, NOT observed
  quotes. Do not use them for per-row effective-spread analysis.
- ``achieved_dollars``: signed ``filled_qty * fill_price``. Zero only
  on terminal no-fill (Cancelled/Inactive/ApiCancelled with filled=0).
  Null on unknown-terminal status (timeout, error, poll failure) so
  a downstream sum of ``achieved_dollars`` doesn't silently swallow
  fills whose status hasn't propagated yet.
- ``target_dollars`` / ``target_shares``: signed OPTIMIZER target for
  the ENTRY. MOC rows leave these null — the MOC has no notional
  target (its qty is derived from the actual post-entry position, not
  from the entry's optimizer target).
- ``filled_qty`` / ``remaining_qty`` / ``cancelled_qty`` are float64:
  IB reports fractional-share fills as floats and integer truncation
  would silently lose sub-share fills. The strategy uses whole shares
  but the recorder faithfully captures whatever IB serves.
- ``commission``: from CommissionReport, which arrives ASYNC on the
  ib_async event stream. On a very fast run an entry fill's row may
  reach finalize before the report lands — commission is then null.
- ``run_id``: ``<utc-timestamp>_pid<pid>`` id for one orchestrator
  invocation. Same-day re-runs produce distinct parquets; the column
  disambiguates them in a concatenated frame.
"""
from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import polars as pl
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from live import config

logger = logging.getLogger(__name__)

_UTC = timezone.utc

# Version stamped into every row so downstream readers can pin against a
# specific column layout. Bump on ANY schema change (add / rename / drop /
# type change).
#
# v4: added per-minute impact-decay marks at T+1, T+2, T+3, T+4 (T+5
#     already existed in v3). Each mark has a price / observed-ts /
#     staleness triple. The old t5min columns keep their v3 names and
#     semantics so v3 parquets stay readable and any downstream code
#     that reads a t5min column continues to work — T+5 is now one
#     point on the decay curve, not a renamed thing.
SCHEMA_VERSION: int = 4

# Every minute-offset the recorder captures a post-decision mark at.
# Each n in this tuple contributes THREE columns to the schema:
# ``t{n}min_price``, ``t{n}min_ts_utc``, ``t{n}min_staleness_secs``.
# Change this list and every downstream helper — column generation,
# row backfill, timestamp coercion — picks it up.
MARK_OFFSETS_MINUTES: tuple[int, ...] = (1, 2, 3, 4, 5)


def _mark_column_names(offset_minutes: int) -> tuple[str, str, str]:
    """Return ``(price_col, ts_col, staleness_col)`` for a mark offset."""
    return (
        f"t{offset_minutes}min_price",
        f"t{offset_minutes}min_ts_utc",
        f"t{offset_minutes}min_staleness_secs",
    )


def _mark_stash_key_ts(offset_minutes: int) -> str:
    """Key under ``self._arrival_prices[sym]`` for the observed ts of
    a mark. The stash-side and row-side names differ deliberately —
    the row column is ``t{n}min_ts_utc``, the stash key is
    ``t{n}min_observed_ts_utc`` — so a symbol whose marks were
    captured BEFORE its row existed can be backfilled without a
    naming collision with the row column itself.
    """
    return f"t{offset_minutes}min_observed_ts_utc"


def _build_trade_record_schema() -> pa.Schema:
    """Assemble the v4 schema. Every column is spelled out explicitly
    (so an all-null column keeps its declared type through a
    ``pa.Table.from_pylist(..., schema=...)`` build) — the per-mark
    block is generated programmatically to keep it in lockstep with
    ``MARK_OFFSETS_MINUTES``.
    """
    fields: list[tuple[str, pa.DataType]] = [
        ("schema_version", pa.int32()),
        ("mode", pa.string()),           # "paper" | "live" | "replay"
        ("leg", pa.string()),            # "entry" | "moc"
        ("symbol", pa.string()),
        ("side", pa.string()),           # "BUY" | "SELL"
        ("order_type", pa.string()),
        ("tif", pa.string()),
        # ``exchange``: the routing code we REQUESTED at submission time
        # (whatever ``live.nasdaq_routing.ACTIVE_ROUTING`` is set to when
        # the order is placed — "SMART" today, "ISLAND" when this book
        # ran direct-routed). This is the intent, not the outcome; the
        # actual venue the fill printed on lives in ``venue`` (below),
        # populated from ``execution.exchange`` and worth splitting on
        # for any post-trade venue analysis.
        ("exchange", pa.string()),
        ("account", pa.string()),
        ("order_id", pa.string()),
        ("perm_id", pa.int64()),
        ("client_id", pa.int64()),
        ("exec_id", pa.string()),
        # All timestamps are timezone-aware UTC. Convert on read.
        ("decision_ts_utc", pa.timestamp("us", tz="UTC")),
        ("snapshot_ts_utc", pa.timestamp("us", tz="UTC")),
        ("sent_ts_utc", pa.timestamp("us", tz="UTC")),
        ("fill_ts_utc", pa.timestamp("us", tz="UTC")),
        # Where ``fill_ts_utc`` came from:
        #   "execution" — TWS ``execution.time`` from the fill's execution
        #                 report (accurate to the exchange fill instant).
        #   "observed"  — wall-clock fallback (``_fill_audit`` return
        #                 time). With MOC audit resequencing this is
        #                 ~5-10 minutes AFTER the true fill; latency
        #                 computed off "observed" rows is not comparable
        #                 to "execution" rows. Filter on this column.
        ("fill_ts_source", pa.string()),
        ("observed_ts_utc", pa.timestamp("us", tz="UTC")),
        ("arrival_ts_utc", pa.timestamp("us", tz="UTC")),
        # t5min columns stay in their v3 positions to keep the on-disk
        # field ordering for v3-shared columns identical to v3 files.
        # The remaining mark columns (T+1..T+4) are appended below.
        ("t5min_ts_utc", pa.timestamp("us", tz="UTC")),
        ("order_qty", pa.int64()),       # signed by side (+BUY / -SELL)
        ("requested_qty", pa.int64()),   # unsigned qty as sent
        ("filled_qty", pa.float64()),    # IB may report fractional
        ("signed_filled_qty", pa.float64()),
        ("remaining_qty", pa.float64()),
        ("cancelled_qty", pa.float64()),
        ("fill_price", pa.float64()),    # VWAP across partial fills
        ("arrival_price", pa.float64()),
        ("bid_at_decision", pa.float64()),      # DERIVED from mid + spread
        ("ask_at_decision", pa.float64()),      # DERIVED from mid + spread
        ("rel_spread_at_decision", pa.float64()),   # (ask-bid)/mid; NOT half
        ("arrival_staleness_secs", pa.float64()),
        ("t5min_staleness_secs", pa.float64()),
        ("target_dollars", pa.float64()),   # entry rows only; MOC rows null
        ("target_shares", pa.int64()),      # entry rows only; MOC rows null
        ("achieved_dollars", pa.float64()),
        ("t5min_price", pa.float64()),
        ("order_status", pa.string()),
        ("error", pa.string()),
        ("commission", pa.float64()),
        ("commission_currency", pa.string()),
        ("realized_pnl", pa.float64()),
        ("liquidity", pa.string()),
        ("venue", pa.string()),
        ("replay_date", pa.string()),
        ("universe_hash", pa.string()),
        ("run_id", pa.string()),
        ("decision_context_json", pa.string()),
    ]
    seen = {name for name, _ in fields}
    for n in MARK_OFFSETS_MINUTES:
        price_col, ts_col, stale_col = _mark_column_names(n)
        for name, dtype in (
            (price_col, pa.float64()),
            (ts_col, pa.timestamp("us", tz="UTC")),
            (stale_col, pa.float64()),
        ):
            if name in seen:
                # Already spelled out above (T+5 lives in the v3
                # positions for on-disk stability).
                continue
            fields.append((name, dtype))
            seen.add(name)
    return pa.schema(fields)


TRADE_RECORD_SCHEMA: pa.Schema = _build_trade_record_schema()

# Terminal statuses on which we CAN commit ``achieved_dollars=0.0`` for a
# zero-fill row (the trade was definitively rejected / cancelled). Any
# other terminal / non-terminal state leaves ``achieved_dollars`` null so
# a downstream ``sum(achieved_dollars)`` doesn't silently swallow a
# real fill whose status hasn't propagated yet.
_TERMINAL_NO_FILL_STATES: frozenset[str] = frozenset({
    "Cancelled", "ApiCancelled", "Inactive",
})


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

TRADES_DIR = config.LIVE_LOG_DIR / "trades"


def resolve_output_path(
    mode: str,
    trading_date: str,
    run_id: str,
    base_dir: Optional[Path] = None,
) -> Path:
    """Return the final parquet path for the given mode / date / run.

    Layout: ``<base>/<mode>/<trading_date>/<run_id>.parquet``. Each
    orchestrator invocation writes its OWN file, so a mid-session
    crash-then-restart never destroys the earlier run's captured
    fills. Same-day re-runs coexist and can be concat'd downstream.
    """
    base = Path(base_dir) if base_dir else TRADES_DIR
    if mode not in ("paper", "live", "replay"):
        raise ValueError(f"unknown mode {mode!r}")
    return base / mode / trading_date / f"{run_id}.parquet"


def resolve_journal_path(final_path: Path) -> Path:
    """Journal path sits next to the final parquet, hidden."""
    return final_path.parent / f".{final_path.stem}.journal.jsonl"


def make_run_id(started_at_utc: Optional[datetime] = None) -> str:
    """Distinguishing ``<utc-timestamp>_pid<pid>`` id for one run."""
    if started_at_utc is None:
        started_at_utc = datetime.now(_UTC)
    stamp = started_at_utc.strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_pid{os.getpid()}"


# ---------------------------------------------------------------------------
# Row model
# ---------------------------------------------------------------------------

def _null_row() -> dict[str, Any]:
    return {name: None for name in TRADE_RECORD_SCHEMA.names}


def _now_utc() -> datetime:
    return datetime.now(_UTC)


def _to_utc(ts: Optional[datetime]) -> Optional[datetime]:
    """Coerce tz-aware datetimes to UTC.

    A naive datetime is REJECTED (``ValueError``). Passes ``None``
    through so a missing stage stays a stored null.
    """
    if ts is None:
        return None
    if ts.tzinfo is None:
        raise ValueError(
            "trade recorder refuses naive datetime — pass a tz-aware ts "
            "(America/New_York or UTC). Silent naive==UTC coercion has "
            "produced tz bugs in this repo before."
        )
    return ts.astimezone(_UTC)


def _finite(v: Any) -> Optional[float]:
    """Filter NaN/inf out at write time."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f


# ---------------------------------------------------------------------------
# Broker extras extraction
# ---------------------------------------------------------------------------

def extract_broker_extras(broker: Any, oid: str) -> dict[str, Any]:
    """Pull the metadata TWS / ib_async surface on the Trade object.

    Every step is guarded — a raise inside a ``@property`` getter or a
    missing field must not propagate. On failure the recorder still
    writes the row with the fields we could pull from ``OrderInfo``.
    """
    out: dict[str, Any] = {}
    try:
        trades_dict = getattr(broker, "_trades", None)
    except Exception:
        return out
    if trades_dict is None:
        return out
    try:
        trade = trades_dict.get(str(oid))
    except Exception:
        return out
    if trade is None:
        return out
    try:
        order = getattr(trade, "order", None)
    except Exception:
        order = None
    if order is not None:
        for src, dst in (
            ("permId", "perm_id"),
            ("clientId", "client_id"),
            ("tif", "tif"),
            ("account", "account"),
        ):
            try:
                val = getattr(order, src, None)
                if val is None or val == "":
                    continue
                if dst in ("perm_id", "client_id"):
                    out[dst] = int(val)
                else:
                    out[dst] = str(val)
            except Exception:
                continue
    try:
        contract = getattr(trade, "contract", None)
    except Exception:
        contract = None
    if contract is not None:
        try:
            val = getattr(contract, "exchange", None)
            if val:
                out["exchange"] = str(val)
        except Exception:
            pass
    try:
        order_status = getattr(trade, "orderStatus", None)
    except Exception:
        order_status = None
    if order_status is not None:
        try:
            val = getattr(order_status, "remaining", None)
            if val is not None:
                out["remaining_qty"] = float(val)
        except Exception:
            pass

    # Fills: pull commission / liquidity / venue / exec_id from the FIRST
    # fill (there is usually only one for an MKT into a top-of-book).
    # Multi-fill orders aggregate — sum commissions, take first venue.
    try:
        fills = getattr(trade, "fills", None)
    except Exception:
        fills = None
    if fills:
        try:
            total_commission = 0.0
            saw_commission = False
            commission_currency: Optional[str] = None
            realized_pnl: Optional[float] = None
            saw_pnl = False
            first_exec_id: Optional[str] = None
            first_venue: Optional[str] = None
            first_liquidity: Optional[str] = None
            first_fill_ts: Optional[datetime] = None
            for fill in fills:
                exec_obj = getattr(fill, "execution", None)
                if exec_obj is not None:
                    if first_exec_id is None:
                        eid = getattr(exec_obj, "execId", None)
                        if eid:
                            first_exec_id = str(eid)
                    if first_venue is None:
                        ex = getattr(exec_obj, "exchange", None)
                        if ex:
                            first_venue = str(ex)
                    if first_liquidity is None:
                        liq = getattr(exec_obj, "lastLiquidity", None)
                        if liq not in (None, 0, ""):
                            first_liquidity = str(liq)
                    if first_fill_ts is None:
                        ts = getattr(exec_obj, "time", None)
                        if isinstance(ts, datetime):
                            first_fill_ts = ts
                comm = getattr(fill, "commissionReport", None)
                if comm is not None:
                    c = getattr(comm, "commission", None)
                    if c is not None:
                        try:
                            total_commission += float(c)
                            saw_commission = True
                        except (TypeError, ValueError):
                            pass
                    cc = getattr(comm, "currency", None)
                    if cc and commission_currency is None:
                        commission_currency = str(cc)
                    p = getattr(comm, "realizedPNL", None)
                    if p is not None:
                        try:
                            realized_pnl = (realized_pnl or 0.0) + float(p)
                            saw_pnl = True
                        except (TypeError, ValueError):
                            pass
            if saw_commission:
                out["commission"] = total_commission
            if commission_currency is not None:
                out["commission_currency"] = commission_currency
            if saw_pnl and realized_pnl is not None:
                out["realized_pnl"] = realized_pnl
            if first_exec_id is not None:
                out["exec_id"] = first_exec_id
            if first_venue is not None and "venue" not in out:
                out["venue"] = first_venue
            if first_liquidity is not None:
                out["liquidity"] = first_liquidity
            if first_fill_ts is not None:
                # NOTE: extras' ``fill_ts_utc`` OVERRIDES the recorder's
                # wall-clock ``observed_ts_utc`` inside ``record_fill``
                # — this IS TWS's real ``execution.time``, the primary
                # time axis for slippage analysis.
                try:
                    out["fill_ts_utc"] = _to_utc(first_fill_ts)
                except ValueError:
                    # A naive execution.time from ib_async is possible
                    # on some paths — treat as UTC. The observed_ts_utc
                    # column remains authoritative for the outer wall
                    # clock so no info is lost.
                    out["fill_ts_utc"] = first_fill_ts.replace(tzinfo=_UTC)
        except Exception:
            pass
    return out


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------

@dataclass
class _MarksCapture:
    """Handle for the background multi-mark impact-decay-curve thread.

    ``result`` is keyed by ``offset_minutes`` -> per-symbol
    ``(mid, observed_ts)`` — the same shape ``_mid_at_or_before``
    returns for each individual target instant.
    """
    thread: Optional[threading.Thread]
    done: threading.Event
    result: dict[int, dict[str, tuple[float, datetime]]] = field(default_factory=dict)
    target_ts_by_offset: dict[int, datetime] = field(default_factory=dict)


class TradeRecorder:
    """Accumulate per-trade rows and write one parquet at end-of-day.

    Instantiate once per orchestrator run. Every public mutator method
    is safe to call from ``safe_call``; a raise is swallowed by the
    caller and the order path continues unaffected.

    Thread safety: the background marks capture updates rows through
    ``record_marks``, which takes ``self._lock``. All other mutators
    also take the lock.
    """

    def __init__(
        self,
        *,
        mode: str,
        decision_ts_utc: datetime,
        universe: list[str],
        trading_date: str,
        base_dir: Optional[Path] = None,
        replay_date: Optional[str] = None,
        universe_hash: Optional[str] = None,
        run_id: Optional[str] = None,
        snapshot_ts_utc: Optional[datetime] = None,
    ) -> None:
        if mode not in ("paper", "live", "replay"):
            raise ValueError(f"unknown mode {mode!r}")
        self.mode = mode
        self.decision_ts_utc = _to_utc(decision_ts_utc)
        self.snapshot_ts_utc = _to_utc(snapshot_ts_utc) if snapshot_ts_utc else None
        self.universe = list(universe)
        self.trading_date = trading_date
        self.replay_date = replay_date
        self.universe_hash = universe_hash
        self.run_id = run_id or make_run_id()
        self.final_path = resolve_output_path(
            mode, trading_date, self.run_id, base_dir,
        )
        self.journal_path = resolve_journal_path(self.final_path)
        self._lock = threading.RLock()
        self._rows: dict[tuple[str, str], dict[str, Any]] = {}
        self._decision_context: dict[str, Any] = {}
        self._arrival_prices: dict[str, dict[str, Any]] = {}
        self._targets: dict[str, dict[str, Any]] = {}
        self._marks: Optional[_MarksCapture] = None
        self._finalized = False
        # Prep the parent dir. The journal is NOT unlinked here — a
        # crash-and-restart's earlier journal must be preserved for
        # rebuild. Because ``run_id`` is unique per invocation, the
        # journal path is unique too, so a fresh run starts with a
        # fresh (non-existent) journal anyway.
        try:
            self.final_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            logger.exception(
                "trade recorder: parent-dir prep failed (non-fatal)",
                extra={"path": str(self.journal_path)},
            )

    # ------------------------------------------------------------------
    # Journaling
    # ------------------------------------------------------------------

    def _journal(self, event: dict[str, Any]) -> None:
        """Append one event dict to the JSONL journal.

        Skipped once ``finalize()`` has run (the parquet is authoritative
        from that point on, and a later background thread must not
        resurrect a deleted journal).
        """
        if self._finalized:
            return
        try:
            payload = {
                "ts": _now_utc().isoformat(),
                **event,
            }
            with self.journal_path.open("a") as fh:
                fh.write(json.dumps(payload, default=str) + "\n")
        except Exception:
            logger.exception(
                "trade recorder: journal append failed (non-fatal)",
                extra={"path": str(self.journal_path)},
            )

    # ------------------------------------------------------------------
    # Row lookup / merge
    # ------------------------------------------------------------------

    def _row(self, leg: str, symbol: str) -> dict[str, Any]:
        key = (leg, symbol)
        row = self._rows.get(key)
        if row is None:
            row = _null_row()
            row["schema_version"] = SCHEMA_VERSION
            row["mode"] = self.mode
            row["leg"] = leg
            row["symbol"] = symbol
            row["decision_ts_utc"] = self.decision_ts_utc
            row["snapshot_ts_utc"] = self.snapshot_ts_utc
            row["replay_date"] = self.replay_date
            row["universe_hash"] = self.universe_hash
            row["run_id"] = self.run_id
            self._rows[key] = row
            # Backfill decision-time state we may already have.
            arr = self._arrival_prices.get(symbol)
            if arr is not None:
                self._apply_arrival_to_row(row, arr)
            if leg == "entry":
                # MOC's "target" is derived from actual post-entry
                # position; the entry's target does NOT apply to it.
                tgt = self._targets.get(symbol)
                if tgt is not None:
                    self._apply_target_to_row(row, tgt)
            if self._decision_context:
                row["decision_context_json"] = json.dumps(
                    self._decision_context, sort_keys=True, default=str,
                )
        return row

    def _apply_arrival_to_row(
        self, row: dict[str, Any], arr: dict[str, Any],
    ) -> None:
        row["arrival_price"] = _finite(arr.get("mid"))
        row["bid_at_decision"] = _finite(arr.get("bid"))
        row["ask_at_decision"] = _finite(arr.get("ask"))
        row["rel_spread_at_decision"] = _finite(arr.get("rel_spread"))
        arr_ts = arr.get("observed_ts_utc")
        if arr_ts is not None:
            try:
                row["arrival_ts_utc"] = _to_utc(arr_ts)
            except ValueError:
                row["arrival_ts_utc"] = None
            # Baseline: prefer ``snapshot_ts_utc`` (when the snapshot
            # actually completed) over ``decision_ts_utc`` (the
            # scheduled anchor). The snapshot is guaranteed to be AT
            # or AFTER any tick it observed, so this staleness is
            # ≥ 0; the scheduled baseline can go slightly negative
            # for a tick delivered just past 15:50:00.
            baseline = self.snapshot_ts_utc or self.decision_ts_utc
            if baseline and row.get("arrival_ts_utc"):
                row["arrival_staleness_secs"] = (
                    baseline - row["arrival_ts_utc"]
                ).total_seconds()
        # Per-mark decay-curve columns may have been captured before
        # the row existed. Backfill each configured offset from the
        # per-symbol stash. Uniform loop so a change to
        # ``MARK_OFFSETS_MINUTES`` propagates automatically.
        for _n in MARK_OFFSETS_MINUTES:
            price_col, ts_col, stale_col = _mark_column_names(_n)
            stash_ts_key = _mark_stash_key_ts(_n)
            if arr.get(price_col) is not None:
                row[price_col] = _finite(arr.get(price_col))
            if arr.get(stash_ts_key) is not None:
                row[ts_col] = arr.get(stash_ts_key)
            if arr.get(stale_col) is not None:
                row[stale_col] = _finite(arr.get(stale_col))

    def _apply_target_to_row(
        self, row: dict[str, Any], tgt: dict[str, Any],
    ) -> None:
        row["target_dollars"] = _finite(tgt.get("target_dollars"))
        if tgt.get("target_shares") is not None:
            try:
                row["target_shares"] = int(tgt["target_shares"])
            except (TypeError, ValueError):
                pass

    # ------------------------------------------------------------------
    # Stage A: decision-time context and arrival prices
    # ------------------------------------------------------------------

    def record_decision_context(self, context: dict[str, Any]) -> None:
        with self._lock:
            self._decision_context = dict(context)
            ctx_json = json.dumps(self._decision_context, sort_keys=True, default=str)
            for row in self._rows.values():
                row["decision_context_json"] = ctx_json
        self._journal({"kind": "decision_context", "context": context})

    def record_arrival_prices(
        self,
        arrivals: dict[str, dict[str, Any]],
    ) -> None:
        """arrivals = ``{sym: {'bid', 'ask', 'mid', 'rel_spread',
        'observed_ts_utc': datetime}}``.
        """
        with self._lock:
            for sym, prices in arrivals.items():
                stash = self._arrival_prices.setdefault(sym, {})
                stash.update(prices)
                for (leg, s), row in self._rows.items():
                    if s != sym:
                        continue
                    self._apply_arrival_to_row(row, stash)
        # Journal the FULL payload — a crash-then-rebuild needs enough
        # to reconstruct decision-time state per symbol.
        self._journal({"kind": "arrival_prices", "arrivals": arrivals})

    def record_targets(
        self,
        targets: dict[str, dict[str, Any]],
    ) -> None:
        """Applies to ENTRY rows only. MOC's implicit target is a
        flatten sized off the actual post-entry position; the entry's
        optimizer target is not a meaningful MOC target."""
        with self._lock:
            for sym, tgt in targets.items():
                self._targets[sym] = dict(tgt)
                for (leg, s), row in self._rows.items():
                    if s != sym or leg != "entry":
                        continue
                    self._apply_target_to_row(row, tgt)
        self._journal({"kind": "targets", "targets": targets})

    # ------------------------------------------------------------------
    # Stage B / D: submission events
    # ------------------------------------------------------------------

    def record_submission(
        self,
        *,
        leg: str,
        symbol: str,
        side: str,
        requested_qty: int,
        order_type: str,
        order_id: Optional[str],
        sent_ts_utc: Optional[datetime] = None,
        exchange: Optional[str] = None,
        tif: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        with self._lock:
            row = self._row(leg, symbol)
            row["side"] = side
            row["order_type"] = order_type
            row["requested_qty"] = int(requested_qty)
            signed = int(requested_qty) if side == "BUY" else -int(requested_qty)
            row["order_qty"] = signed
            if order_id is not None:
                row["order_id"] = str(order_id)
            row["sent_ts_utc"] = _to_utc(sent_ts_utc) if sent_ts_utc else _now_utc()
            if exchange:
                row["exchange"] = str(exchange)
            if tif:
                row["tif"] = str(tif)
            if error:
                row["error"] = str(error)[:400]
        self._journal({
            "kind": "submission",
            "leg": leg,
            "symbol": symbol,
            "side": side,
            "order_id": order_id,
            "requested_qty": int(requested_qty),
            "order_type": order_type,
            "exchange": exchange,
            "tif": tif,
            "error": error,
            "sent_ts_utc": (
                _to_utc(sent_ts_utc).isoformat() if sent_ts_utc else None
            ),
        })

    # ------------------------------------------------------------------
    # Stage C / F: fill events
    # ------------------------------------------------------------------

    def record_fill(
        self,
        *,
        leg: str,
        symbol: str,
        order_id: Optional[str],
        filled_qty: Optional[float],
        fill_price: Optional[float],
        status: Optional[str] = None,
        observed_ts_utc: Optional[datetime] = None,
        extras: Optional[dict[str, Any]] = None,
    ) -> None:
        """Record fill data.

        ``observed_ts_utc`` is the wall clock at which ``_fill_audit``
        picked the fill up (always populated on filled/timeout rows).
        ``extras`` from ``extract_broker_extras`` may override several
        columns — specifically ``fill_ts_utc`` (TWS's real
        ``execution.time``, which IS the primary time axis for
        slippage analysis and MUST beat the wall clock).

        ``achieved_dollars`` semantics:
        - filled_qty > 0 and fill_price finite -> signed notional.
        - filled_qty == 0 AND status in {Cancelled/ApiCancelled/Inactive}
          -> ``0.0`` (definitively no fill, safe to sum).
        - otherwise -> ``None`` (unknown; timeout/error; downstream
          sums must not silently treat this as $0).
        """
        with self._lock:
            row = self._row(leg, symbol)
            side = row.get("side")
            fq = None if filled_qty is None else float(filled_qty)
            row["filled_qty"] = fq
            if fq is None or side not in ("BUY", "SELL"):
                row["signed_filled_qty"] = None
            else:
                row["signed_filled_qty"] = fq if side == "BUY" else -fq
            fp = _finite(fill_price)
            row["fill_price"] = fp
            if (fq is not None and fq > 0 and fp is not None
                    and side in ("BUY", "SELL")):
                sign = 1.0 if side == "BUY" else -1.0
                row["achieved_dollars"] = sign * fq * fp
            elif (fq is not None and fq == 0
                  and status in _TERMINAL_NO_FILL_STATES):
                row["achieved_dollars"] = 0.0
            else:
                # Timeout / unknown / non-terminal — leave null.
                row["achieved_dollars"] = None
            if status is not None:
                row["order_status"] = str(status)
            # observed_ts_utc: always populated on a fill/timeout row
            # (it's a wall clock — we always know when we picked the
            # row up).
            row["observed_ts_utc"] = (
                _to_utc(observed_ts_utc) if observed_ts_utc else _now_utc()
            )
            # fill_ts_utc: ONLY populated for rows that actually
            # filled. A ``timeout`` / ``Cancelled`` / ``Inactive``
            # row must NOT carry a fill timestamp — downstream
            # latency = ``fill_ts_utc - sent_ts_utc`` would otherwise
            # get a biased right tail from trades that never
            # happened. Extras' ``execution.time`` (below) can still
            # OVERRIDE this on real fills; when that happens we set
            # ``fill_ts_source = "execution"`` there.
            if row.get("fill_ts_utc") is None and fq is not None and fq > 0:
                row["fill_ts_utc"] = row["observed_ts_utc"]
                row["fill_ts_source"] = "observed"
            # cancelled_qty: computable when both requested and filled
            # are known AND status is a terminal cancel.
            req = row.get("requested_qty")
            if (fq is not None and req is not None
                    and status in _TERMINAL_NO_FILL_STATES):
                row["cancelled_qty"] = max(0.0, float(req) - fq)
            if extras:
                for k, v in extras.items():
                    if k not in TRADE_RECORD_SCHEMA.names:
                        continue
                    if v is None:
                        continue
                    # Extras WIN for fill_ts_utc — TWS's real
                    # execution.time is the primary time axis and
                    # MUST NOT lose to the wall-clock fallback.
                    # Also flip ``fill_ts_source`` so downstream can
                    # tell "execution" from "observed" without a
                    # brittle self-comparison of the two timestamps.
                    if k == "fill_ts_utc":
                        row[k] = v
                        row["fill_ts_source"] = "execution"
                        continue
                    # For every OTHER field, only overwrite null.
                    if row.get(k) is not None:
                        continue
                    row[k] = v
            if row.get("order_id") is None and order_id is not None:
                row["order_id"] = str(order_id)
        self._journal({
            "kind": "fill",
            "leg": leg,
            "symbol": symbol,
            "order_id": order_id,
            "filled_qty": filled_qty,
            "fill_price": fill_price,
            "status": status,
            "observed_ts_utc": (
                _to_utc(observed_ts_utc).isoformat()
                if observed_ts_utc else None
            ),
        })

    # ------------------------------------------------------------------
    # Stage E: impact-decay-curve marks at T+1..T+5min
    # ------------------------------------------------------------------

    def record_marks(
        self,
        *,
        marks_by_offset_minutes: dict[int, dict[str, tuple[float, datetime]]],
        target_ts_by_offset_minutes: dict[int, datetime],
    ) -> None:
        """Attach per-symbol observed mids at (or before) each of a
        series of target instants — the post-decision impact-decay
        curve. ``marks_by_offset_minutes[n][sym] = (mid, observed_ts)``
        for offset ``n`` minutes after decision.

        Each mark records the OBSERVED tick timestamp and a staleness
        computed against its OWN target — never a fabricated ts, never
        the outer capture time. A thin name whose only tick was
        pre-decision yields the same price across all 5 marks with
        increasing staleness (and is therefore filterable per-mark).

        Bad inputs are skipped granularly rather than aborting the
        whole call:
          - offset keys outside ``MARK_OFFSETS_MINUTES`` warn+skip
            (the schema has no column);
          - naive ``observed_ts`` on an individual sym warn+skip that
            sym only (symmetric with the per-offset target handling —
            a single bad tick in the snapshot mustn't drop every
            other symbol's mark for that offset).
        """
        normalized_targets: dict[int, datetime] = {}
        for n, tgt in target_ts_by_offset_minutes.items():
            try:
                normalized_targets[int(n)] = _to_utc(tgt)
            except ValueError:
                logger.warning(
                    "trade recorder: naive target ts for T+%dmin — skipping",
                    int(n),
                )
        # Assemble the journal payload while walking the input so
        # dropped syms / offsets don't leak into the journal (an
        # analyst rebuilding from journal must not see a mark for
        # something the row-side skipped).
        journal_marks: dict[str, dict[str, dict[str, Any]]] = {}
        with self._lock:
            for n, mid_and_ts_by_sym in marks_by_offset_minutes.items():
                n_int = int(n)
                if n_int not in MARK_OFFSETS_MINUTES:
                    logger.warning(
                        "trade recorder: unknown mark offset %d — schema has "
                        "no column, skipping",
                        n_int,
                    )
                    continue
                target = normalized_targets.get(n_int)
                if target is None:
                    continue
                price_col, ts_col, stale_col = _mark_column_names(n_int)
                stash_ts_key = _mark_stash_key_ts(n_int)
                per_offset_journal: dict[str, dict[str, Any]] = {}
                for sym, (mid, observed_ts) in mid_and_ts_by_sym.items():
                    try:
                        obs = _to_utc(observed_ts)
                    except ValueError:
                        logger.warning(
                            "trade recorder: naive observed ts for %s at "
                            "T+%dmin — skipping this sym only",
                            sym, n_int,
                        )
                        continue
                    if obs is None:
                        continue
                    price_finite = _finite(mid)
                    stale = (target - obs).total_seconds()
                    arr = self._arrival_prices.setdefault(sym, {})
                    arr[price_col] = price_finite
                    arr[stash_ts_key] = obs
                    arr[stale_col] = stale
                    for (leg, s), row in self._rows.items():
                        if s != sym:
                            continue
                        row[price_col] = price_finite
                        row[ts_col] = obs
                        row[stale_col] = stale
                    per_offset_journal[sym] = {
                        # NaN gets filtered out here symmetric with the
                        # row-side ``_finite`` filter — a bare NaN in
                        # the JSONL journal parses under Python's json
                        # but strict JSON parsers reject it.
                        "mid": price_finite,
                        "observed_ts_utc": obs.isoformat(),
                    }
                if per_offset_journal:
                    journal_marks[str(n_int)] = per_offset_journal
        # Only journal offsets whose target was valid AND that landed
        # on at least one row — matches what an analyst reading the
        # journal would see reconstructed on the row side. The target
        # list uses the same `normalized_targets` map so the two
        # halves of the payload can't disagree.
        self._journal({
            "kind": "marks",
            "target_ts_by_offset_minutes": {
                str(n): t.isoformat()
                for n, t in normalized_targets.items()
            },
            "marks_by_offset_minutes": journal_marks,
        })

    def capture_marks_from_tbbo(
        self,
        tbbo: pl.DataFrame,
        decision_ts_utc: datetime,
        offsets_minutes: tuple[int, ...] = MARK_OFFSETS_MINUTES,
    ) -> None:
        """Synchronous walk-back for each offset — used on the replay
        path where the historical tbbo already covers the whole day
        in memory.

        Every offset filters ``ts_event <= (decision + offset)`` on
        the same frame, so no look-ahead across marks is possible.
        A pre-decision-only tick yields the same price at every
        offset with monotonically-increasing staleness.
        """
        decision = _to_utc(decision_ts_utc)
        marks_by_offset: dict[int, dict[str, tuple[float, datetime]]] = {}
        target_ts_by_offset: dict[int, datetime] = {}
        for n in offsets_minutes:
            n_int = int(n)
            target_ts = decision + timedelta(minutes=n_int)
            target_ts_by_offset[n_int] = target_ts
            try:
                mids = _mid_at_or_before(tbbo, target_ts)
            except Exception:
                logger.exception(
                    "trade recorder: T+%dmin replay extraction failed "
                    "(non-fatal)",
                    n_int,
                )
                continue
            marks_by_offset[n_int] = mids
        self.record_marks(
            marks_by_offset_minutes=marks_by_offset,
            target_ts_by_offset_minutes=target_ts_by_offset,
        )

    def schedule_marks_capture_from_feed(
        self,
        feed: Any,
        decision_ts_utc: datetime,
        offsets_minutes: tuple[int, ...] = MARK_OFFSETS_MINUTES,
        max_wait_secs: Optional[float] = None,
    ) -> None:
        """Background thread — SINGLE snapshot at ``decision + max(offsets)``
        (typically T+5), then walk-back through the same frame for
        each earlier offset.

        Design rationale (worth calling out because it looks like
        it should be five timers):
        - One thread, one wake, one ``feed.snapshot()`` call — the
          feed's buffer at T+5 already contains every tick from the
          whole day, so a per-offset ``_mid_at_or_before`` filter
          gives the T+1 last-mid without a T+1-wall-clock capture.
        - No additional feed lifetime — the feed still stops right
          after this one capture (via ``wait_for_marks``).
        - No look-ahead — each mark filters ``ts_event <= its own
          target``, not the capture time. The T+1 mark is CAPTURED
          at T+5 wall clock but reflects only data at or before T+1.
        - The rc=11 overnight-exposure path is unchanged: the wait
          budget for the thread is set by the caller
          (``max_wait_secs``) and the main path never blocks on it
          — the orchestrator uses ``wait_for_marks(timeout=350)``
          AFTER the exposure check, so a recording delay never
          gates the operator page.

        The parent's ``finally`` intentionally does NOT wait for
        this thread. If it lands after ``finalize()``, ``_journal``
        and ``record_marks`` become no-ops via the ``_finalized``
        guard.
        """
        decision = _to_utc(decision_ts_utc)
        target_ts_by_offset: dict[int, datetime] = {
            int(n): decision + timedelta(minutes=int(n))
            for n in offsets_minutes
        }
        if not target_ts_by_offset:
            return
        max_target = max(target_ts_by_offset.values())
        done = threading.Event()
        capture = _MarksCapture(
            thread=None, done=done, target_ts_by_offset=target_ts_by_offset,
        )

        def _worker():
            try:
                budget = (
                    max_wait_secs if max_wait_secs is not None
                    else float("inf")
                )
                start_wall = time.time()
                while True:
                    now = _now_utc()
                    remaining = (max_target - now).total_seconds()
                    if remaining <= 0:
                        break
                    elapsed = time.time() - start_wall
                    if elapsed >= budget:
                        break
                    # Cap each sleep chunk at BOTH the remaining budget
                    # AND ~5s so a stopped feed / early cancel returns
                    # promptly, and a short unit-test budget
                    # (max_wait_secs=0.1) doesn't get stuck for 5s
                    # before re-checking the budget.
                    sleep_secs = min(remaining, 5.0)
                    if budget != float("inf"):
                        sleep_secs = min(sleep_secs, max(0.001, budget - elapsed))
                    time.sleep(sleep_secs)
                if self._finalized:
                    return
                snapshot = feed.snapshot()
                tbbo = snapshot.get("tbbo", pl.DataFrame())
                if tbbo.is_empty():
                    logger.warning(
                        "trade recorder: marks snapshot returned empty tbbo",
                    )
                    return
                # Walk back: one filter per target on the same frame.
                marks_by_offset: dict[
                    int, dict[str, tuple[float, datetime]]
                ] = {}
                for n, target_ts in target_ts_by_offset.items():
                    try:
                        marks_by_offset[n] = _mid_at_or_before(tbbo, target_ts)
                    except Exception:
                        logger.exception(
                            "trade recorder: T+%dmin walk-back failed "
                            "(non-fatal)",
                            n,
                        )
                        continue
                capture.result = marks_by_offset
                self.record_marks(
                    marks_by_offset_minutes=marks_by_offset,
                    target_ts_by_offset_minutes=target_ts_by_offset,
                )
            except Exception:
                logger.exception(
                    "trade recorder: marks capture thread failed (non-fatal)",
                )
            finally:
                done.set()

        t = threading.Thread(
            target=_worker,
            name="trade-recorder-marks",
            daemon=True,
        )
        capture.thread = t
        self._marks = capture
        t.start()
        logger.info(
            "trade recorder: marks capture scheduled",
            extra={
                "max_target_ts_utc": max_target.isoformat(),
                "offsets_minutes": list(target_ts_by_offset.keys()),
            },
        )

    def wait_for_marks(self, timeout: Optional[float] = None) -> bool:
        """Block until the background marks thread finishes (or
        ``timeout`` elapses). Returns True iff the thread signalled
        completion; False if the wait timed out or there is no
        scheduled capture.

        The orchestrator uses this to stop the Databento feed as
        soon as the marks snapshot is in, instead of holding it
        open through the 10-minute ``_fill_audit`` wait. Called
        only AFTER the rc=11 exposure gate so a bounded wait here
        never delays the operator page.
        """
        if self._marks is None:
            return False
        try:
            return bool(self._marks.done.wait(timeout=timeout))
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------

    def finalize(
        self,
        *,
        wait_for_marks: Optional[float] = None,
    ) -> Optional[Path]:
        """Materialise the parquet atomically. Idempotent.

        ``wait_for_marks`` defaults to ``None`` — do NOT block in the
        common case. The orchestrator's ``finally`` calls this on
        every exit path (including the rc=11 OVERNIGHT-EXPOSURE page),
        and a wait would delay the operator alert. Pass a positive
        number only in tests that need to synchronise the background
        marks thread; production callers should never pass one.
        """
        with self._lock:
            if self._finalized:
                return self.final_path
        # Wait for the marks thread OUTSIDE the lock so it can still
        # call ``record_marks`` (which takes the lock). Also do not
        # set ``_finalized = True`` yet — the thread's post-sleep
        # early-out checks that flag to avoid resurrecting a deleted
        # journal after the parquet is written; setting it here would
        # race with a wait we actually want to succeed.
        if self._marks is not None and wait_for_marks:
            try:
                self._marks.done.wait(timeout=wait_for_marks)
            except Exception:
                logger.exception(
                    "trade recorder: marks wait failed (non-fatal)",
                )
        with self._lock:
            if self._finalized:
                # Race with a concurrent finalize on another thread.
                return self.final_path
            self._finalized = True
            rows_list = [self._materialise_row(r) for r in self._rows.values()]
        if not rows_list:
            logger.info(
                "trade recorder: no rows to write; skipping parquet",
                extra={"path": str(self.final_path)},
            )
            self._unlink_journal_if_safe()
            return None
        try:
            table = pa.Table.from_pylist(rows_list, schema=TRADE_RECORD_SCHEMA)
        except Exception:
            logger.exception(
                "trade recorder: pyarrow table build failed — journal kept for recovery",
                extra={"n_rows": len(rows_list), "path": str(self.final_path)},
            )
            return None
        try:
            _atomic_write_arrow(self.final_path, table)
            logger.info(
                "trade recorder: wrote parquet",
                extra={
                    "path": str(self.final_path),
                    "n_rows": table.num_rows,
                    "n_cols": table.num_columns,
                    "mode": self.mode,
                    "run_id": self.run_id,
                },
            )
        except Exception:
            logger.exception(
                "trade recorder: parquet write failed (journal kept for recovery)",
                extra={"path": str(self.final_path)},
            )
            return None
        self._unlink_journal_if_safe()
        return self.final_path

    def _unlink_journal_if_safe(self) -> None:
        try:
            if self.journal_path.exists():
                self.journal_path.unlink()
        except OSError:
            pass

    def _materialise_row(self, row: dict[str, Any]) -> dict[str, Any]:
        """Freeze a row for pyarrow — every schema column present, every
        timestamp tz-aware UTC.
        """
        out: dict[str, Any] = {}
        for name in TRADE_RECORD_SCHEMA.names:
            out[name] = row.get(name)
        ts_cols: list[str] = [
            "decision_ts_utc", "snapshot_ts_utc", "sent_ts_utc",
            "fill_ts_utc", "observed_ts_utc", "arrival_ts_utc",
        ]
        for _n in MARK_OFFSETS_MINUTES:
            _, ts_col, _ = _mark_column_names(_n)
            ts_cols.append(ts_col)
        for k in ts_cols:
            v = out.get(k)
            if v is None:
                continue
            try:
                out[k] = _to_utc(v)
            except ValueError:
                # Naive slipped in from somewhere — treat as UTC rather
                # than drop the row.
                if isinstance(v, datetime):
                    out[k] = v.replace(tzinfo=_UTC)
        out["schema_version"] = SCHEMA_VERSION
        out["mode"] = self.mode
        return out


# ---------------------------------------------------------------------------
# Helpers used by the orchestrator call sites
# ---------------------------------------------------------------------------

def safe_call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    """Invoke ``fn(*args, **kwargs)`` and swallow every exception with a
    logged ``logger.exception`` line."""
    try:
        fn(*args, **kwargs)
    except Exception:
        logger.exception("trade recorder call raised (non-fatal)")


def arrival_prices_from_day_df(
    day_df: pl.DataFrame,
    last_mid_by_sym: Optional[dict[str, float]] = None,
    last_mid_ts_by_sym: Optional[dict[str, datetime]] = None,
) -> dict[str, dict[str, Any]]:
    """Derive per-symbol arrival snapshot at the 15:50 decision moment.

    Reads ``entry_price`` and ``spread_return`` from ``day_df`` (both
    populated by ``features.compute_close_trades``). ``spread_return``
    is the FULL relative spread ``(ask - bid) / mid`` — not half; the
    stored column is named ``rel_spread_at_decision`` to make this
    explicit. See ``features.py::compute_close_trades`` for a stale
    half-spread comment (out of scope for this module to touch).

    The bid / ask are DERIVED symmetrically from the mid and the
    relative spread — they are NOT observed quotes, and per-row
    effective-spread analysis is invalid.

    ``last_mid_ts_by_sym`` maps ``{sym: last-observed tbbo ts}`` and is
    used to populate ``arrival_ts_utc`` / ``arrival_staleness_secs``.
    A missing entry leaves the row's staleness null; downstream can
    drop or flag it.
    """
    out: dict[str, dict[str, Any]] = {}
    if day_df is None or day_df.is_empty():
        return out
    if "symbol" not in day_df.columns or "entry_price" not in day_df.columns:
        return out
    symbols = day_df["symbol"].to_list()
    entry_prices = day_df["entry_price"].to_list()
    rel_spreads = (
        day_df["spread_return"].to_list()
        if "spread_return" in day_df.columns else [None] * len(symbols)
    )
    for sym, ep, sr in zip(symbols, entry_prices, rel_spreads):
        mid = None
        if last_mid_by_sym is not None:
            mid = last_mid_by_sym.get(sym)
        if mid is None or mid == 0:
            mid = ep
        if mid is None or not math.isfinite(float(mid)) or float(mid) <= 0:
            continue
        mid_f = float(mid)
        rel_spread = (
            float(sr) if sr is not None and math.isfinite(float(sr)) else None
        )
        if rel_spread is not None:
            half = 0.5 * rel_spread * mid_f
            bid = mid_f - half
            ask = mid_f + half
        else:
            bid = None
            ask = None
        entry = {
            "bid": bid,
            "ask": ask,
            "mid": mid_f,
            "rel_spread": rel_spread,
        }
        if last_mid_ts_by_sym is not None:
            ts = last_mid_ts_by_sym.get(sym)
            if isinstance(ts, datetime):
                entry["observed_ts_utc"] = ts
        out[sym] = entry
    return out


def _mid_at_or_before(
    tbbo: pl.DataFrame,
    target_ts_utc: datetime,
) -> dict[str, tuple[float, datetime]]:
    """Return per-symbol ``(mid, observed_ts)`` for the latest tbbo tick
    at or before ``target_ts_utc``.

    The OBSERVED ts is returned so ``record_marks`` can compute a
    staleness gap — a thin name whose last print was minutes ago must
    NOT be recorded as if the price were fresh at the target instant.
    """
    if tbbo is None or tbbo.is_empty():
        return {}
    if "ts_event" not in tbbo.columns or "symbol" not in tbbo.columns:
        return {}
    ts_col = pl.col("ts_event")
    df = tbbo
    dtype = df.schema.get("ts_event")
    if isinstance(dtype, pl.Datetime) and dtype.time_zone not in (None, "UTC"):
        df = df.with_columns(ts_col.dt.convert_time_zone("UTC"))
    elif isinstance(dtype, pl.Datetime) and dtype.time_zone is None:
        df = df.with_columns(ts_col.dt.replace_time_zone("UTC"))
    target_ts = _to_utc(target_ts_utc)
    df = (
        df.filter(pl.col("ts_event") <= target_ts)
        .with_columns(
            ((pl.col("ask_px_00") + pl.col("bid_px_00")) / 2).alias("mid")
        )
        .sort("ts_event")
        .group_by("symbol", maintain_order=False)
        .agg(
            pl.col("mid").last().alias("mid"),
            pl.col("ts_event").last().alias("observed_ts"),
        )
    )
    out: dict[str, tuple[float, datetime]] = {}
    for sym, mid, obs_ts in zip(
        df["symbol"].to_list(),
        df["mid"].to_list(),
        df["observed_ts"].to_list(),
    ):
        if mid is None or not math.isfinite(float(mid)):
            continue
        if obs_ts is None:
            continue
        if obs_ts.tzinfo is None:
            obs_ts = obs_ts.replace(tzinfo=_UTC)
        out[str(sym)] = (float(mid), obs_ts.astimezone(_UTC))
    return out


def _atomic_write_arrow(path: Path, table: pa.Table) -> None:
    """``.tmp`` file + ``os.replace`` — never a torn parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".parquet.tmp", dir=str(path.parent))
    try:
        os.close(fd)
        pq.write_table(table, tmp)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Cross-vintage reader — MUST be used for any glob spanning schema
# versions (e.g. a date range straddling a deploy).
# ---------------------------------------------------------------------------

def read_trades(
    paths: Iterable[Any],
    schema: pa.Schema = TRADE_RECORD_SCHEMA,
) -> pa.Table:
    """Read one or more trade parquets across schema vintages.

    Passing the CURRENT ``schema`` explicitly makes pyarrow back-fill
    any columns a v3 (or older) file lacks as **typed nulls**, so a
    concat over a version boundary retains every v4 column. This is
    the ONLY safe reader for a multi-file glob whose files may span
    the v3 -> v4 boundary:

    - ``pyarrow.dataset(paths).to_table()`` (no ``schema=``) infers
      from the FIRST fragment — depending on filesystem ordering,
      you get either 49 columns (v3 first, v4 columns silently
      dropped, no warning) or 61 (v4 first, v3 rows show typed
      nulls for the new columns). The **silent-drop path** produces
      an analyst-facing decay-curve fit calibrated on T+5 only,
      which is precisely the single-point failure v4 exists to
      escape.
    - ``pl.read_parquet([v3, v4])`` raises ``SchemaError``; the
      remedy it suggests (``extra_columns="ignore"``) also silently
      drops the new columns.

    Use this helper. If a Polars frame is wanted, wrap the return
    with ``pl.from_arrow(read_trades(paths))``.

    ``paths``: any iterable of paths (str or ``Path``). One path
    is fine — the same code path handles single-file reads.
    ``schema``: defaults to the CURRENT ``TRADE_RECORD_SCHEMA``.
    Pin an older schema only when you deliberately want an older
    view (e.g. reproducing a historical downstream aggregation).

    **Distinguishing null-from-vintage from null-from-failure**: for
    a row whose ``t{n}min_price`` is null, filter on
    ``schema_version < 4`` to identify the "column didn't exist"
    case (v3 row, back-filled by this helper), and
    ``schema_version == 4`` to identify the "mark genuinely didn't
    populate" case (e.g. no tick at or before the target, or the
    background thread never ran). Downstream calibration must
    exclude both, but the two mean different things about the day.
    """
    str_paths = [str(p) for p in paths]
    if not str_paths:
        return pa.Table.from_pylist([], schema=schema)
    dataset = ds.dataset(str_paths, format="parquet", schema=schema)
    return dataset.to_table()


__all__ = [
    "MARK_OFFSETS_MINUTES",
    "SCHEMA_VERSION",
    "TRADE_RECORD_SCHEMA",
    "TRADES_DIR",
    "TradeRecorder",
    "arrival_prices_from_day_df",
    "extract_broker_extras",
    "make_run_id",
    "read_trades",
    "resolve_journal_path",
    "resolve_output_path",
    "safe_call",
]
