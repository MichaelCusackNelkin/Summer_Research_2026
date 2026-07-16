"""Coverage-aware wrapper around the Databento parquet cache.

The ``DatabentoGenericAdapter`` (``data_collection.data_adapters``) keys its
parquet cache on an MD5 of the EXACT ``{tickers, start_date, end_date,
dataset, schema}`` tuple. A request whose end-date is one day off misses
even when the same data is already on disk under a different hash slot —
which is how the live scripts have been quietly triggering fresh
Databento batch-API jobs. This module sits on top of the adapter and
answers coverage queries against what is actually on disk (schema
inferred from column names, symbol set via ``unique()``, date range from
``ts_event`` min/max) so callers can:

* HIT — serve the request from an existing parquet, filtered on the fly,
  without writing a new cache slot.
* MISS — refuse politely and tell the operator what would need to be
  fetched. Callers must opt in to fetching via ``allow_fetch=True``.

Everything in this module is intentionally live/-local. The adapter
itself keeps its exact-hash keying — a proper fix would rework the
adapter's cache-key scheme, but that's out of scope here.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Iterable, Optional, Union

import polars as pl

from data_collection.data_adapters import DatabentoGenericAdapter

logger = logging.getLogger(__name__)


# Column-set fingerprints for each Databento schema we know how to serve
# from an existing parquet. Enough to disambiguate — not a full schema
# match. Order-independent because we test with subset (``fp <= cols``).
#
# Note: ``ohlcv-1d`` and ``ohlcv-1m`` share this column set. We infer the
# generic ``ohlcv-*`` label here and disambiguate on timestamp cadence
# inside ``_scan_parquet`` (daily bars have exactly one time-of-day —
# midnight UTC — while minute bars have many). This matters because
# ``_refit_covariance`` requests ``ohlcv-1d`` and a minute-bar parquet
# HITting would silently corrupt the covariance fit.
_SCHEMA_FINGERPRINTS: dict[str, frozenset[str]] = {
    "imbalance": frozenset({"total_imbalance_qty", "paired_qty", "ref_price"}),
    "tbbo": frozenset({"ask_px_00", "bid_px_00"}),
    "statistics": frozenset({"stat_type"}),
    "ohlcv-*": frozenset({"open", "high", "low", "close", "volume"}),
    "definition": frozenset({"raw_symbol", "security_update_action"}),
}


def _infer_schema(columns: Iterable[str]) -> Optional[str]:
    """Infer a Databento schema label from a column-name set.

    Returns ``None`` when nothing matches — the manifest silently skips
    those parquets so unknown/foreign files never cause a false HIT.
    The special ``"ohlcv-*"`` label is a placeholder — callers must
    disambiguate to ``ohlcv-1d`` / ``ohlcv-1m`` from the ts_event
    cadence before storing the manifest entry.
    """
    cols = set(columns)
    for schema, fp in _SCHEMA_FINGERPRINTS.items():
        if fp <= cols:
            return schema
    return None


class CoverageStatus(str, Enum):
    HIT = "HIT"
    MISS = "MISS"


@dataclass(frozen=True)
class ParquetEntry:
    """Manifest row — one existing parquet's contents summarised.

    ``symbol_ranges`` stores per-symbol ``(start_date, end_date)``
    coverage. A required symbol only counts as covered if its own range
    brackets the request window — otherwise IPO / delisting mid-window
    would produce silent false HITs. Empty dict means per-symbol data
    was not collected (older sidecar entries, or a scan error).
    """
    path: Path
    schema: str
    symbols: frozenset[str]
    start_date: date  # inclusive, from min(ts_event).date()
    end_date: date    # inclusive, from max(ts_event).date()
    mtime_ns: int
    # R6: size + mtime keys the sidecar so an overwrite that preserves
    # mtime still invalidates.
    size_bytes: int = 0
    # Per-symbol coverage — required-symbol HIT requires each symbol's
    # own range to cover the request window (R3). Excluded from hash /
    # equality because dicts are unhashable and the (path, mtime, size)
    # tuple is already sufficient identity.
    symbol_ranges: Optional[dict[str, tuple[date, date]]] = field(
        default=None, hash=False, compare=False,
    )


@dataclass(frozen=True)
class CoverageDecision:
    status: CoverageStatus
    covering_path: Optional[Path] = None
    reason: str = ""
    # MISS-only: the parquet that came closest to serving the request, and
    # the integer shortfall (days at head/tail, or count of missing required
    # symbols). Populated by ``CoverageManifest.check`` so callers can build
    # a self-diagnostic log line / exception message without re-inspecting
    # the manifest. Both ``None`` when there is no candidate at all (no
    # cached parquet matches the schema).
    nearest_parquet: Optional[Path] = None
    shortfall_distance: Optional[int] = None


class CoverageError(RuntimeError):
    """A coverage MISS was hit without ``allow_fetch=True``."""


class CoverageManifest:
    """In-memory manifest of the adapter's parquet cache.

    Rebuilds lazily on ``load()``. A sidecar JSON at
    ``<cache_dir>/.coverage_manifest.json`` stores per-parquet mtimes so
    entries only get re-scanned when the file on disk actually changed.
    The sidecar is a pure optimisation — deleting it forces a rebuild.
    """

    _SIDECAR_NAME = ".coverage_manifest.json"

    def __init__(self, cache_dir: Union[str, Path], persist: bool = True) -> None:
        self.cache_dir = Path(cache_dir)
        self.sidecar_path = self.cache_dir / self._SIDECAR_NAME
        self.persist = persist
        self._entries: dict[str, ParquetEntry] = {}
        self._loaded = False

    def load(self) -> "CoverageManifest":
        """Populate the manifest. Idempotent — cheap on repeat calls."""
        if self._loaded:
            return self
        cached = self._read_sidecar()
        for parquet_path in sorted(self.cache_dir.glob("*.parquet")):
            try:
                stat = parquet_path.stat()
            except OSError as e:
                logger.warning("could not stat parquet; skipping",
                               extra={"path": str(parquet_path), "error": str(e)})
                continue
            key = parquet_path.name
            cached_entry = cached.get(key)
            # R6: invalidate on any (mtime, size) delta — mtime alone is
            # fooled by any tool that preserves atime/mtime on rewrite.
            if (cached_entry
                and cached_entry.get("mtime_ns") == stat.st_mtime_ns
                and cached_entry.get("size_bytes") == stat.st_size):
                restored = self._restore_from_sidecar(
                    parquet_path, cached_entry, stat.st_mtime_ns, stat.st_size,
                )
                if restored is not None:
                    self._entries[key] = restored
                    continue
            entry = self._scan_parquet(parquet_path, stat.st_mtime_ns, stat.st_size)
            if entry is not None:
                self._entries[key] = entry
        if self.persist:
            self._write_sidecar()
        self._loaded = True
        return self

    def entries(self) -> list[ParquetEntry]:
        self.load()
        return list(self._entries.values())

    def refresh(self) -> "CoverageManifest":
        """Force a re-scan on next ``load()``."""
        self._entries.clear()
        self._loaded = False
        return self

    def check(
        self,
        *,
        tickers: Union[str, Iterable[str]],
        start_date: str,
        end_date: str,
        schema: str,
        required_symbols: Optional[Iterable[str]] = None,
    ) -> CoverageDecision:
        """Return HIT/MISS for a proposed adapter request.

        Parameters
        ----------
        tickers
            The value the caller would pass to the adapter. Wildcard
            strings (``"ALL_SYMBOLS"``) skip the symbol-subset check
            unless ``required_symbols`` is supplied.
        start_date, end_date
            ISO ``YYYY-MM-DD``. ``end_date`` is exclusive per Databento
            convention, so the last day of data the request needs is
            ``end_date - 1``.
        schema
            One of the schemas ``_SCHEMA_FINGERPRINTS`` recognises.
        required_symbols
            Symbols that MUST be present in the parquet for a HIT. When
            ``None``, the check falls back to ``tickers`` if that is a
            concrete list, else skips the symbol-subset check.

        Notes
        -----
        Date-range comparisons are floored to the nearest weekday: a
        parquet ending on a Friday counts as covering a request whose
        exclusive end lands on the following Saturday or Sunday, since
        Databento would return zero new rows for those weekend days
        anyway. Exchange holidays remain a residual gap — a request
        needing a holiday-adjacent day whose closest weekday isn't in
        the parquet will still MISS. Solving that needs a trading
        calendar which is out of scope here.

        Coverage is an endpoint check (per-parquet and per-required-
        symbol). Interior-day contiguity is NOT verified: contiguous
        batch pulls make interior gaps unlikely in practice, and adding
        an interior scan would inflate scan cost for little real-world
        value. A parquet with a genuine hole in the middle of the
        window will therefore still HIT — accept this as a known
        limitation.
        """
        self.load()
        try:
            req_start = date.fromisoformat(start_date)
            req_end = date.fromisoformat(end_date)
        except ValueError:
            return CoverageDecision(
                CoverageStatus.MISS,
                reason=f"malformed dates: start={start_date!r} end={end_date!r}",
            )
        if req_end <= req_start:
            return CoverageDecision(
                CoverageStatus.MISS,
                reason=f"empty window {start_date} to {end_date}",
            )
        last_needed = req_end - timedelta(days=1)
        # R2: floor endpoints to the nearest trading weekday. A parquet
        # ending Fri covers a request whose ``last_needed`` falls on the
        # weekend; a parquet starting Mon covers a request whose
        # ``req_start`` falls on the preceding weekend.
        effective_last_needed = _prev_weekday(last_needed)
        effective_req_start = _next_weekday(req_start)
        req_syms = _resolve_required(tickers, required_symbols)

        candidates = [e for e in self._entries.values() if e.schema == schema]
        if not candidates:
            return CoverageDecision(
                CoverageStatus.MISS,
                reason=f"no cached parquet has schema={schema!r}",
            )

        # R4: track the smallest shortfall across candidates so the
        # near-miss log points at the ACTUAL closest parquet, not just
        # the glob-order first failure.
        best_near_miss: Optional[tuple[int, str, ParquetEntry]] = None

        def _worse_than_best(shortfall: int) -> bool:
            return best_near_miss is not None and shortfall >= best_near_miss[0]

        for entry in candidates:
            if entry.start_date > effective_req_start:
                shortfall_d = (entry.start_date - effective_req_start).days
                miss_reason = (
                    f"parquet starts {entry.start_date}, request needs {req_start} "
                    f"(short {shortfall_d}d at head)"
                )
                if not _worse_than_best(shortfall_d):
                    best_near_miss = (shortfall_d, miss_reason, entry)
                continue
            if entry.end_date < effective_last_needed:
                shortfall_d = (effective_last_needed - entry.end_date).days
                miss_reason = (
                    f"parquet ends {entry.end_date}, request needs {last_needed} "
                    f"(short {shortfall_d}d at tail)"
                )
                if not _worse_than_best(shortfall_d):
                    best_near_miss = (shortfall_d, miss_reason, entry)
                continue
            if req_syms is not None and not req_syms.issubset(entry.symbols):
                missing = sorted(req_syms - entry.symbols)
                miss_reason = (
                    f"parquet missing {len(missing)} required symbol(s), "
                    f"e.g. {missing[:5]}"
                )
                # Use missing-symbol count as the distance metric.
                if not _worse_than_best(len(missing)):
                    best_near_miss = (len(missing), miss_reason, entry)
                continue
            # R3: parquet-wide dates check passed AND every required
            # symbol appears somewhere. Now confirm each required
            # symbol's own coverage brackets the (weekday-floored)
            # window. Skip if per-symbol ranges weren't recorded — an
            # older manifest entry falls back to the endpoint-only check.
            if req_syms is not None and entry.symbol_ranges:
                worst_shortfall = 0
                worst_reason: Optional[str] = None
                for sym in req_syms:
                    rng = entry.symbol_ranges.get(sym)
                    if rng is None:
                        continue  # subset check above already accepted it
                    sym_start, sym_end = rng
                    sym_effective_start = _next_weekday(req_start)
                    sym_effective_last = _prev_weekday(last_needed)
                    if sym_start > sym_effective_start:
                        d = (sym_start - sym_effective_start).days
                        if d > worst_shortfall:
                            worst_shortfall = d
                            worst_reason = (
                                f"symbol {sym!r} starts {sym_start} in this parquet, "
                                f"request needs {req_start} (short {d}d at head — "
                                f"likely IPO/listing after window start)"
                            )
                    if sym_end < sym_effective_last:
                        d = (sym_effective_last - sym_end).days
                        if d > worst_shortfall:
                            worst_shortfall = d
                            worst_reason = (
                                f"symbol {sym!r} ends {sym_end} in this parquet, "
                                f"request needs {last_needed} (short {d}d at tail — "
                                f"likely delisting inside window)"
                            )
                if worst_reason is not None:
                    if not _worse_than_best(worst_shortfall):
                        best_near_miss = (worst_shortfall, worst_reason, entry)
                    continue
            return CoverageDecision(
                status=CoverageStatus.HIT,
                covering_path=entry.path,
                reason=(
                    f"schema={entry.schema} "
                    f"dates=[{entry.start_date}..{entry.end_date}] "
                    f"n_symbols={len(entry.symbols)}"
                ),
            )

        if best_near_miss is not None:
            shortfall, reason, near_entry = best_near_miss
            return CoverageDecision(
                CoverageStatus.MISS,
                reason=reason,
                nearest_parquet=near_entry.path,
                shortfall_distance=shortfall,
            )
        return CoverageDecision(
            CoverageStatus.MISS,
            reason=f"no cached parquet matches schema={schema!r}",
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _read_sidecar(self) -> dict[str, dict]:
        if not self.persist or not self.sidecar_path.exists():
            return {}
        try:
            payload = json.loads(self.sidecar_path.read_text())
            if not isinstance(payload, dict):
                return {}
            return payload
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(
                "coverage manifest sidecar unreadable; rebuilding",
                extra={"path": str(self.sidecar_path), "error": str(e)},
            )
            return {}

    def _write_sidecar(self) -> None:
        payload: dict[str, dict] = {}
        for key, entry in self._entries.items():
            payload[key] = {
                "schema": entry.schema,
                "symbols": sorted(entry.symbols),
                "start_date": entry.start_date.isoformat(),
                "end_date": entry.end_date.isoformat(),
                "mtime_ns": entry.mtime_ns,
                "size_bytes": entry.size_bytes,
                # Per-symbol coverage — small (a few hundred names, two
                # ISO dates each) so we persist it verbatim rather than
                # re-scan on every process start.
                "symbol_ranges": (
                    None if entry.symbol_ranges is None
                    else {sym: [s.isoformat(), e.isoformat()]
                          for sym, (s, e) in entry.symbol_ranges.items()}
                ),
            }
        tmp = self.sidecar_path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(payload))
            tmp.replace(self.sidecar_path)
        except OSError as e:
            logger.warning(
                "could not persist coverage sidecar",
                extra={"path": str(self.sidecar_path), "error": str(e)},
            )

    @staticmethod
    def _restore_from_sidecar(
        parquet_path: Path,
        payload: dict,
        mtime_ns: int,
        size_bytes: int,
    ) -> Optional[ParquetEntry]:
        try:
            raw_ranges = payload.get("symbol_ranges")
            symbol_ranges: Optional[dict[str, tuple[date, date]]] = None
            if isinstance(raw_ranges, dict):
                symbol_ranges = {
                    sym: (date.fromisoformat(s), date.fromisoformat(e))
                    for sym, (s, e) in raw_ranges.items()
                }
            return ParquetEntry(
                path=parquet_path,
                schema=payload["schema"],
                symbols=frozenset(payload["symbols"]),
                start_date=date.fromisoformat(payload["start_date"]),
                end_date=date.fromisoformat(payload["end_date"]),
                mtime_ns=mtime_ns,
                size_bytes=size_bytes,
                symbol_ranges=symbol_ranges,
            )
        except (KeyError, ValueError) as e:
            logger.warning(
                "cached manifest entry unusable; re-scanning",
                extra={"path": str(parquet_path), "error": str(e)},
            )
            return None

    @staticmethod
    def _scan_parquet(
        path: Path, mtime_ns: int, size_bytes: int,
    ) -> Optional[ParquetEntry]:
        """Peek a parquet cheaply: columns, per-symbol date range.

        Uses lazy scans so we never load row data — only the ``symbol``
        and ``ts_event`` columns, grouped for a per-symbol min/max in a
        single streaming pass. Also collects the count of distinct
        ``ts_event`` times so we can disambiguate ohlcv-1d (one unique
        time = midnight UTC) from ohlcv-1m (many).
        """
        try:
            columns = pl.scan_parquet(str(path)).collect_schema().names()
        except Exception as e:  # noqa: BLE001 — polars raises many types
            logger.warning(
                "could not read parquet schema; skipping",
                extra={"path": str(path), "error": str(e)},
            )
            return None
        if "symbol" not in columns or "ts_event" not in columns:
            logger.info(
                "parquet lacks symbol/ts_event; skipping coverage",
                extra={"path": str(path), "columns_head": columns[:5]},
            )
            return None
        schema = _infer_schema(columns)
        if schema is None:
            logger.info(
                "parquet has unrecognised schema fingerprint; skipping coverage",
                extra={"path": str(path), "columns_head": columns[:5]},
            )
            return None
        # Per-symbol coverage. One streaming pass, group-by symbol.
        try:
            per_sym = (
                pl.scan_parquet(str(path))
                .group_by("symbol")
                .agg(
                    pl.col("ts_event").min().alias("min_ts"),
                    pl.col("ts_event").max().alias("max_ts"),
                )
                .collect(engine="streaming")
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "could not read parquet metadata; skipping",
                extra={"path": str(path), "error": str(e)},
            )
            return None

        if per_sym.is_empty():
            return None

        symbol_ranges: dict[str, tuple[date, date]] = {}
        overall_min: Optional[date] = None
        overall_max: Optional[date] = None
        for sym, min_ts, max_ts in per_sym.iter_rows():
            if sym is None or min_ts is None or max_ts is None:
                continue
            min_d = _to_utc_date(min_ts)
            max_d = _to_utc_date(max_ts)
            symbol_ranges[sym] = (min_d, max_d)
            if overall_min is None or min_d < overall_min:
                overall_min = min_d
            if overall_max is None or max_d > overall_max:
                overall_max = max_d

        if overall_min is None or overall_max is None:
            return None

        # R1: ohlcv-1d vs ohlcv-1m disambiguation. Daily bars have a
        # single ts_event time per row (midnight UTC); minute bars have
        # many. A minute-bar parquet HITting an ohlcv-1d request would
        # silently corrupt the covariance fit — label it as ohlcv-1m so
        # no live caller can HIT it. The check is cheap (one aggregate,
        # one column).
        if schema == "ohlcv-*":
            try:
                n_unique_times = (
                    pl.scan_parquet(str(path))
                    .select(pl.col("ts_event").dt.time().n_unique().alias("n"))
                    .collect()[0, 0]
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "could not check ohlcv cadence; conservatively labelling as ohlcv-1m",
                    extra={"path": str(path), "error": str(e)},
                )
                schema = "ohlcv-1m"
            else:
                schema = "ohlcv-1d" if n_unique_times <= 1 else "ohlcv-1m"
                logger.info(
                    "ohlcv cadence disambiguated",
                    extra={
                        "path": str(path),
                        "n_unique_times": int(n_unique_times or 0),
                        "resolved_schema": schema,
                    },
                )

        return ParquetEntry(
            path=path,
            schema=schema,
            symbols=frozenset(symbol_ranges.keys()),
            start_date=overall_min,
            end_date=overall_max,
            mtime_ns=mtime_ns,
            size_bytes=size_bytes,
            symbol_ranges=symbol_ranges,
        )


def _resolve_required(
    tickers: Union[str, Iterable[str]],
    override: Optional[Iterable[str]],
) -> Optional[frozenset[str]]:
    """Determine which symbols must be present for a HIT.

    Explicit ``override`` wins. Otherwise falls back to ``tickers`` for
    concrete lists and skips the check for wildcard strings.
    """
    if override is not None:
        return frozenset(s.upper() for s in override)
    if isinstance(tickers, str):
        if tickers.upper() == "ALL_SYMBOLS":
            return None
        return frozenset({tickers.upper()})
    return frozenset(str(t).upper() for t in tickers)


def _prev_weekday(d: date) -> date:
    """Return the last weekday (Mon-Fri) on-or-before ``d``.

    Used to floor the tail bound of a coverage request: Databento would
    return no rows for a Saturday or Sunday, so a parquet ending on the
    preceding Friday is a legitimate match for a request whose exclusive
    end lands on a weekend. Exchange holidays (Thanksgiving, Christmas,
    etc.) are NOT accounted for — that requires a trading calendar which
    is out of scope here.
    """
    # Monday=0..Sunday=6.
    while d.weekday() > 4:
        d -= timedelta(days=1)
    return d


def _next_weekday(d: date) -> date:
    """Return the next weekday (Mon-Fri) on-or-after ``d``."""
    while d.weekday() > 4:
        d += timedelta(days=1)
    return d


def _to_utc_date(ts: object) -> date:
    if isinstance(ts, datetime):
        if ts.tzinfo is not None:
            ts = ts.astimezone(timezone.utc)
        return ts.date()
    if isinstance(ts, date):
        return ts
    # Fallback for exotic types.
    return datetime.fromisoformat(str(ts)).date()


# ---------------------------------------------------------------------------
# Resolver — the policy wrapper live scripts actually call.
# ---------------------------------------------------------------------------


class CoverageResolver:
    """Adapter-facing policy layer for coverage-aware reads.

    Every historical data read from the live scripts should go through
    this resolver rather than call the adapter directly. Behaviour:

    1. Fast path: ``adapter.scan_data(...)``. On an exact-hash cache
       HIT the adapter returns a LazyFrame and we're done.
    2. On the adapter's ``FileNotFoundError``, consult the coverage
       manifest for an existing parquet whose contents cover the request.
    3. Coverage HIT: build a filtered ``pl.LazyFrame`` off the existing
       parquet — no new cache slot, no Databento call, no duplicated data.
    4. Coverage MISS: refuse by default. ``allow_fetch=True`` authorises
       the resolver to invoke ``adapter.get_data(materialize_only=True)``;
       the caller's script (``retrain.py``) layers additional interactive
       or ``--yes`` gates on top for batch-API-cost protection.
    """

    def __init__(
        self,
        adapter: DatabentoGenericAdapter,
        manifest: Optional[CoverageManifest] = None,
    ) -> None:
        self.adapter = adapter
        self.manifest = manifest or CoverageManifest(adapter.cache_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(
        self,
        *,
        tickers: Union[str, Iterable[str]],
        start_date: str,
        end_date: str,
        dataset: str = "XNAS.ITCH",
        schema: str = "ohlcv-1m",
        required_symbols: Optional[Iterable[str]] = None,
    ) -> CoverageDecision:
        """Return a coverage decision without doing any I/O.

        Used by ``--dry-run`` paths to preview whether a request will HIT
        the manifest or trigger a fetch. Also short-circuits with a
        synthetic HIT if the adapter's exact-hash-keyed parquet already
        exists, since callers can use that verbatim.
        """
        adapter_path = self._adapter_hash_path(
            tickers, start_date, end_date, dataset, schema,
        )
        if adapter_path.exists():
            return CoverageDecision(
                status=CoverageStatus.HIT,
                covering_path=adapter_path,
                reason="adapter hash-key cache hit",
            )
        return self.manifest.check(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            schema=schema,
            required_symbols=required_symbols,
        )

    def resolve_lazy(
        self,
        *,
        tickers: Union[str, Iterable[str]],
        start_date: str,
        end_date: str,
        dataset: str = "XNAS.ITCH",
        schema: str = "ohlcv-1m",
        required_symbols: Optional[Iterable[str]] = None,
        allow_fetch: bool = False,
    ) -> pl.LazyFrame:
        """Return a filtered ``pl.LazyFrame`` for the requested slice.

        - Fast path: adapter's exact-hash parquet exists → adapter scan.
        - Coverage HIT: filter an existing parquet to the request window.
        - Coverage MISS + ``allow_fetch=False``: raise ``CoverageError``.
        - Coverage MISS + ``allow_fetch=True``: materialise via Databento.

        On coverage HIT the returned LazyFrame is pre-filtered so callers
        see the same shape they'd see from an exact-hash adapter scan:
        ``symbol IN tickers`` (unless ``ALL_SYMBOLS``) and
        ``start_date <= ts_event < end_date``.
        """
        try:
            return self.adapter.scan_data(
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                dataset=dataset,
                schema=schema,
            )
        except FileNotFoundError:
            pass

        decision = self.manifest.check(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            schema=schema,
            required_symbols=required_symbols,
        )
        if decision.status is CoverageStatus.HIT:
            logger.info(
                "coverage HIT — serving from existing parquet, no fetch",
                extra={
                    "path": str(decision.covering_path),
                    "schema": schema,
                    "start": start_date,
                    "end": end_date,
                    "reason": decision.reason,
                },
            )
            assert decision.covering_path is not None
            return _filtered_scan(
                decision.covering_path, tickers, start_date, end_date,
            )

        if not allow_fetch:
            raise CoverageError(
                _miss_message(schema, tickers, start_date, end_date, decision)
            )

        logger.warning(
            "coverage MISS with --allow-fetch — submitting Databento batch job",
            extra={
                "schema": schema, "start": start_date, "end": end_date,
                "reason": decision.reason,
            },
        )
        asyncio.run(
            self.adapter.get_data(
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                dataset=dataset,
                schema=schema,
                materialize_only=True,
            )
        )
        self.manifest.refresh()
        return self.adapter.scan_data(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            dataset=dataset,
            schema=schema,
        )

    async def materialize(
        self,
        *,
        tickers: Union[str, Iterable[str]],
        start_date: str,
        end_date: str,
        dataset: str = "XNAS.ITCH",
        schema: str = "ohlcv-1m",
        required_symbols: Optional[Iterable[str]] = None,
        allow_fetch: bool = False,
    ) -> CoverageDecision:
        """Ensure the request can be served without an unauthorised fetch.

        HIT: no-op. MISS + ``allow_fetch=False``: raise. MISS +
        ``allow_fetch=True``: call ``adapter.get_data(materialize_only=True)``
        and refresh the manifest.
        """
        adapter_path = self._adapter_hash_path(
            tickers, start_date, end_date, dataset, schema,
        )
        if adapter_path.exists():
            return CoverageDecision(
                status=CoverageStatus.HIT,
                covering_path=adapter_path,
                reason="adapter hash-key cache hit",
            )

        decision = self.manifest.check(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            schema=schema,
            required_symbols=required_symbols,
        )
        if decision.status is CoverageStatus.HIT:
            logger.info(
                "coverage HIT (materialize) — no fetch needed",
                extra={
                    "path": str(decision.covering_path),
                    "schema": schema,
                    "start": start_date,
                    "end": end_date,
                },
            )
            return decision

        if not allow_fetch:
            raise CoverageError(
                _miss_message(schema, tickers, start_date, end_date, decision)
            )

        logger.warning(
            "coverage MISS + --allow-fetch — materialising via Databento batch",
            extra={
                "schema": schema, "start": start_date, "end": end_date,
                "reason": decision.reason,
            },
        )
        await self.adapter.get_data(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            dataset=dataset,
            schema=schema,
            materialize_only=True,
        )
        self.manifest.refresh()
        return CoverageDecision(status=CoverageStatus.HIT, reason="fetched")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _adapter_hash_path(
        self,
        tickers: Union[str, Iterable[str]],
        start_date: str,
        end_date: str,
        dataset: str,
        schema: str,
    ) -> Path:
        # NB: the adapter's ``_generate_cache_key`` normalises tickers
        # differently for str vs iterable inputs (str -> [tickers],
        # iterable -> sorted-uppercase). If a caller passes tickers in a
        # form the adapter's own path wouldn't have used at fetch time,
        # this hash will diverge from the true adapter cache slot and
        # ``adapter_path.exists()`` returns False. That is FAIL-SAFE: it
        # merely skips the fast path and falls through to the manifest —
        # no unintended fetch can happen. If the fast path becomes a
        # correctness requirement rather than an optimisation, fix by
        # calling the adapter's ``scan_data`` and catching
        # ``FileNotFoundError`` instead of guessing the path.
        cache_key = self.adapter._generate_cache_key(
            tickers, start_date, end_date, dataset, schema,
        )
        return self.adapter._get_cache_path(cache_key)


def _filtered_scan(
    path: Path,
    tickers: Union[str, Iterable[str]],
    start_date: str,
    end_date: str,
) -> pl.LazyFrame:
    """Scan ``path`` filtered to the caller's request window.

    Polars pushes column/predicate filters into the parquet reader so
    this is cheap even for the 100+ GB tbbo parquet — only the requested
    symbols' row groups get materialised.
    """
    lf = pl.scan_parquet(str(path), low_memory=True)

    if isinstance(tickers, str):
        if tickers.upper() != "ALL_SYMBOLS":
            lf = lf.filter(pl.col("symbol") == tickers.upper())
    else:
        sym_list = sorted({str(t).upper() for t in tickers})
        if sym_list:
            lf = lf.filter(pl.col("symbol").is_in(sym_list))

    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    lf = lf.filter(
        (pl.col("ts_event") >= start_dt) & (pl.col("ts_event") < end_dt)
    )
    return lf


def _miss_message(
    schema: str,
    tickers: Union[str, Iterable[str]],
    start_date: str,
    end_date: str,
    decision: "CoverageDecision",
) -> str:
    """Build a self-diagnostic message for a MISS.

    Includes the nearest cached parquet's filename, the shortfall distance
    (days at head/tail or count of missing required symbols), and the
    per-parquet reason so ``CoverageError`` messages surface every fact an
    operator would otherwise have to hand-inspect the parquet cache for.
    """
    if decision.nearest_parquet is not None:
        near = (
            f"Nearest parquet {decision.nearest_parquet.name}: "
            f"{decision.reason} (shortfall={decision.shortfall_distance})"
        )
    else:
        near = decision.reason
    return (
        f"Coverage MISS: schema={schema} tickers=<{_describe_tickers(tickers)}> "
        f"in [{start_date}, {end_date}). {near}. Nothing was fetched. "
        f"Pass --allow-fetch to authorise a Databento batch job, or --dry-run "
        f"to preview the plan."
    )


def format_coverage_decision(schema: str, decision: "CoverageDecision") -> str:
    """One-line HIT/MISS summary for operator-facing logs.

    Format:
      HIT:  ``<schema>: HIT (<parquet_filename>)``
      MISS: ``<schema>: MISS (nearest <parquet_filename>: <reason>, shortfall=<N>)``
      MISS (no candidate): ``<schema>: MISS (<reason>)``

    Kept in ``cache.py`` so ``retrain.py`` and ``replay_feed.py`` cannot
    drift on the wording of the plan lines the operator relies on.
    """
    if decision.status is CoverageStatus.HIT:
        parquet = (
            decision.covering_path.name if decision.covering_path is not None
            else "?"
        )
        return f"{schema}: HIT ({parquet})"
    if decision.nearest_parquet is not None:
        return (
            f"{schema}: MISS (nearest {decision.nearest_parquet.name}: "
            f"{decision.reason}, shortfall={decision.shortfall_distance})"
        )
    return f"{schema}: MISS ({decision.reason})"


def _describe_tickers(tickers: Union[str, Iterable[str]]) -> str:
    if isinstance(tickers, str):
        return tickers
    tickers = list(tickers)
    if not tickers:
        return "<empty>"
    if len(tickers) <= 3:
        return ",".join(tickers)
    return f"{tickers[0]},...,{tickers[-1]} (n={len(tickers)})"


def summarize_manifest(manifest: CoverageManifest) -> str:
    """One-line-per-parquet summary — used by ``--dry-run`` plan output."""
    entries = sorted(
        manifest.entries(), key=lambda e: (e.schema, e.start_date, e.path.name),
    )
    if not entries:
        return "  (no parquets in cache)"
    return "\n".join(
        f"  {e.path.name}: schema={e.schema} "
        f"dates=[{e.start_date}..{e.end_date}] n_symbols={len(e.symbols)}"
        for e in entries
    )
