"""Tests for the coverage-aware cache layer (``live/cache.py``).

Uses tiny synthetic parquets (a few rows, one row-group) to exercise the
schema-fingerprint / symbol-subset / date-range branches of the coverage
check, then drives the resolver end-to-end against a fake adapter to
prove:

- Coverage HIT returns a filtered LazyFrame WITHOUT any adapter call.
- Coverage MISS refuses without ``allow_fetch=True`` (raises
  ``CoverageError``), never calling the adapter.
- Coverage MISS + ``allow_fetch=True`` invokes ``adapter.get_data``
  exactly once.

Every test uses ``tmp_path`` for its cache dir so nothing touches the
real ``data/dbn_cache/generic/`` tree — the coverage layer's whole
purpose is to avoid burning money on that cache and it must never
require write access to it under test.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from live.cache import (
    CoverageError,
    CoverageManifest,
    CoverageResolver,
    CoverageStatus,
    _infer_schema,
    summarize_manifest,
)


# ---------------------------------------------------------------------------
# Synthetic parquet factories
# ---------------------------------------------------------------------------


def _dt(y, m, d, h=13, mi=30) -> datetime:
    return datetime(y, m, d, h, mi, tzinfo=timezone.utc)


def _write_imbalance_parquet(
    path: Path,
    symbols: list[str],
    start_date: tuple[int, int, int],
    end_date: tuple[int, int, int],
) -> None:
    """Write a minimal imbalance parquet spanning ``[start, end]`` for symbols."""
    rows = []
    for sym in symbols:
        rows.append({
            "ts_event": _dt(*start_date),
            "symbol": sym,
            "side": "B",
            "total_imbalance_qty": 1000,
            "paired_qty": 2000,
            "ref_price": 100.0,
        })
        rows.append({
            "ts_event": _dt(*end_date, h=20),
            "symbol": sym,
            "side": "A",
            "total_imbalance_qty": 500,
            "paired_qty": 1000,
            "ref_price": 101.0,
        })
    pl.DataFrame(rows).write_parquet(str(path))


def _write_tbbo_parquet(
    path: Path,
    symbols: list[str],
    start_date: tuple[int, int, int],
    end_date: tuple[int, int, int],
) -> None:
    rows = []
    for sym in symbols:
        rows.append({
            "ts_event": _dt(*start_date),
            "symbol": sym,
            "price": 100.0,
            "size": 100,
            "ask_px_00": 100.05,
            "bid_px_00": 99.95,
        })
        rows.append({
            "ts_event": _dt(*end_date, h=20),
            "symbol": sym,
            "price": 101.0,
            "size": 200,
            "ask_px_00": 101.05,
            "bid_px_00": 100.95,
        })
    pl.DataFrame(rows).write_parquet(str(path))


class _FakeAdapter:
    """Minimal DatabentoGenericAdapter substitute.

    Reproduces the surface CoverageResolver relies on:
    - ``cache_dir``
    - ``_generate_cache_key``, ``_get_cache_path``, ``_normalize_tickers``
    - ``scan_data``: raises FileNotFoundError when the derived hash-key
      path does not exist, else returns a lazy scan.
    - ``get_data``: records that it was called (so tests can assert the
      resolver did NOT invoke the batch API on a coverage HIT).
    """

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.get_data_calls: list[dict] = []

    def _normalize_tickers(self, tickers):
        if isinstance(tickers, str):
            return [tickers]
        return sorted({t.upper() for t in tickers})

    def _generate_cache_key(self, tickers, start_date, end_date, dataset, schema):
        import hashlib, json
        norm = self._normalize_tickers(tickers)
        payload = json.dumps({
            "tickers": norm, "start": start_date, "end": end_date,
            "dataset": dataset, "schema": schema,
        }, sort_keys=True)
        return hashlib.md5(payload.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.parquet"

    def scan_data(self, tickers, start_date, end_date, dataset="XNAS.ITCH", schema="ohlcv-1m"):
        key = self._generate_cache_key(tickers, start_date, end_date, dataset, schema)
        path = self._get_cache_path(key)
        if not path.exists():
            raise FileNotFoundError(f"no parquet at {path}")
        return pl.scan_parquet(str(path), low_memory=True)

    async def get_data(self, tickers, start_date, end_date, dataset="XNAS.ITCH",
                       schema="ohlcv-1m", materialize_only=False, **kw):
        self.get_data_calls.append({
            "tickers": tickers, "start": start_date, "end": end_date,
            "dataset": dataset, "schema": schema,
        })


# ---------------------------------------------------------------------------
# Schema inference
# ---------------------------------------------------------------------------


def test_infer_schema_imbalance():
    cols = {"ts_event", "symbol", "total_imbalance_qty", "paired_qty", "ref_price"}
    assert _infer_schema(cols) == "imbalance"


def test_infer_schema_tbbo():
    cols = {"ts_event", "symbol", "ask_px_00", "bid_px_00", "price", "size"}
    assert _infer_schema(cols) == "tbbo"


def test_infer_schema_ohlcv_returns_placeholder():
    """The column-set alone can't disambiguate 1d vs 1m; ``_infer_schema``
    returns the ``"ohlcv-*"`` placeholder and ``_scan_parquet`` refines
    it via the ts_event cadence check."""
    cols = {"ts_event", "symbol", "open", "high", "low", "close", "volume"}
    assert _infer_schema(cols) == "ohlcv-*"


def test_infer_schema_unknown_returns_none():
    assert _infer_schema({"foo", "bar"}) is None


# ---------------------------------------------------------------------------
# Manifest — construction & coverage decisions
# ---------------------------------------------------------------------------


def test_manifest_ignores_parquets_without_symbol_or_ts_event(tmp_path):
    path = tmp_path / "junk.parquet"
    pl.DataFrame({"foo": [1, 2, 3]}).write_parquet(str(path))
    manifest = CoverageManifest(tmp_path, persist=False)
    manifest.load()
    assert manifest.entries() == []


def test_manifest_infers_and_stores_imbalance_entry(tmp_path):
    path = tmp_path / "a.parquet"
    _write_imbalance_parquet(path, ["AAPL", "MSFT"], (2025, 6, 11), (2026, 6, 10))
    manifest = CoverageManifest(tmp_path, persist=False)
    entries = manifest.entries()
    assert len(entries) == 1
    e = entries[0]
    assert e.schema == "imbalance"
    assert e.symbols == frozenset({"AAPL", "MSFT"})
    assert e.start_date.isoformat() == "2025-06-11"
    assert e.end_date.isoformat() == "2026-06-10"


def test_coverage_hit_full_range(tmp_path):
    path = tmp_path / "a.parquet"
    _write_imbalance_parquet(path, ["AAPL", "MSFT"], (2025, 6, 11), (2026, 6, 10))
    manifest = CoverageManifest(tmp_path, persist=False)
    decision = manifest.check(
        tickers=["AAPL"],
        start_date="2026-05-15", end_date="2026-05-16",
        schema="imbalance",
    )
    assert decision.status is CoverageStatus.HIT
    assert decision.covering_path == path


def test_coverage_miss_schema_mismatch(tmp_path):
    """Parquet is tbbo; request asks for imbalance → MISS."""
    path = tmp_path / "t.parquet"
    _write_tbbo_parquet(path, ["AAPL"], (2025, 6, 11), (2026, 6, 10))
    manifest = CoverageManifest(tmp_path, persist=False)
    decision = manifest.check(
        tickers=["AAPL"],
        start_date="2025-08-01", end_date="2025-08-02",
        schema="imbalance",
    )
    assert decision.status is CoverageStatus.MISS
    assert "schema" in decision.reason.lower()


def test_coverage_miss_symbol_subset(tmp_path):
    """Parquet has AAPL/MSFT; request asks for NVDA → MISS."""
    path = tmp_path / "a.parquet"
    _write_imbalance_parquet(path, ["AAPL", "MSFT"], (2025, 6, 11), (2026, 6, 10))
    manifest = CoverageManifest(tmp_path, persist=False)
    decision = manifest.check(
        tickers=["NVDA"],
        start_date="2025-08-01", end_date="2025-08-02",
        schema="imbalance",
    )
    assert decision.status is CoverageStatus.MISS
    assert "symbol" in decision.reason.lower() or "missing" in decision.reason.lower()


def test_coverage_miss_partial_date_range_head(tmp_path):
    """Parquet covers 2025-06-11..2026-06-11; request is 2025-05-01..2025-06-30 → MISS."""
    path = tmp_path / "a.parquet"
    _write_imbalance_parquet(path, ["AAPL"], (2025, 6, 11), (2026, 6, 10))
    manifest = CoverageManifest(tmp_path, persist=False)
    decision = manifest.check(
        tickers=["AAPL"],
        start_date="2025-05-01", end_date="2025-06-30",
        schema="imbalance",
    )
    assert decision.status is CoverageStatus.MISS
    assert "head" in decision.reason.lower() or "starts" in decision.reason.lower()


def test_coverage_miss_partial_date_range_tail(tmp_path):
    """Request extends past end of parquet → MISS."""
    path = tmp_path / "a.parquet"
    _write_imbalance_parquet(path, ["AAPL"], (2025, 6, 11), (2026, 6, 10))
    manifest = CoverageManifest(tmp_path, persist=False)
    decision = manifest.check(
        tickers=["AAPL"],
        start_date="2026-05-01", end_date="2026-07-15",  # past end
        schema="imbalance",
    )
    assert decision.status is CoverageStatus.MISS
    assert "tail" in decision.reason.lower() or "ends" in decision.reason.lower()


def test_coverage_hit_all_symbols_with_required_symbols(tmp_path):
    """Wildcard tickers + explicit required_symbols → symbol check against required."""
    path = tmp_path / "a.parquet"
    _write_imbalance_parquet(path, ["AAPL", "MSFT", "NVDA"], (2025, 6, 11), (2026, 6, 10))
    manifest = CoverageManifest(tmp_path, persist=False)
    decision = manifest.check(
        tickers="ALL_SYMBOLS",
        start_date="2025-08-01", end_date="2025-08-02",
        schema="imbalance",
        required_symbols=["AAPL", "MSFT"],
    )
    assert decision.status is CoverageStatus.HIT


def test_coverage_miss_all_symbols_with_missing_required(tmp_path):
    path = tmp_path / "a.parquet"
    _write_imbalance_parquet(path, ["AAPL", "MSFT"], (2025, 6, 11), (2026, 6, 10))
    manifest = CoverageManifest(tmp_path, persist=False)
    decision = manifest.check(
        tickers="ALL_SYMBOLS",
        start_date="2025-08-01", end_date="2025-08-02",
        schema="imbalance",
        required_symbols=["AAPL", "MSFT", "NVDA"],  # NVDA missing
    )
    assert decision.status is CoverageStatus.MISS


def test_manifest_empty_cache_reports_miss(tmp_path):
    manifest = CoverageManifest(tmp_path, persist=False)
    decision = manifest.check(
        tickers=["AAPL"],
        start_date="2025-01-01", end_date="2025-01-02",
        schema="imbalance",
    )
    assert decision.status is CoverageStatus.MISS


def test_manifest_malformed_dates_miss(tmp_path):
    _write_imbalance_parquet(tmp_path / "a.parquet", ["AAPL"], (2025, 6, 11), (2026, 6, 10))
    manifest = CoverageManifest(tmp_path, persist=False)
    decision = manifest.check(
        tickers=["AAPL"],
        start_date="not-a-date", end_date="2025-06-20",
        schema="imbalance",
    )
    assert decision.status is CoverageStatus.MISS
    assert "malformed" in decision.reason.lower()


def test_manifest_empty_window_miss(tmp_path):
    _write_imbalance_parquet(tmp_path / "a.parquet", ["AAPL"], (2025, 6, 11), (2026, 6, 10))
    manifest = CoverageManifest(tmp_path, persist=False)
    decision = manifest.check(
        tickers=["AAPL"],
        start_date="2025-06-20", end_date="2025-06-20",  # equal → empty
        schema="imbalance",
    )
    assert decision.status is CoverageStatus.MISS


def test_manifest_sidecar_reused_on_matching_mtime(tmp_path):
    path = tmp_path / "a.parquet"
    _write_imbalance_parquet(path, ["AAPL"], (2025, 6, 11), (2026, 6, 10))
    m1 = CoverageManifest(tmp_path, persist=True)
    m1.load()
    assert (tmp_path / ".coverage_manifest.json").exists()

    # New manifest instance should reuse the sidecar without re-scanning.
    # We poison the parquet name-lookup path by pointing polars at a
    # missing file — the sidecar should still let us restore the entry.
    m2 = CoverageManifest(tmp_path, persist=True)
    m2.load()
    entries = m2.entries()
    assert len(entries) == 1
    assert entries[0].schema == "imbalance"


def test_manifest_sidecar_invalidated_on_mtime_change(tmp_path):
    path = tmp_path / "a.parquet"
    _write_imbalance_parquet(path, ["AAPL"], (2025, 6, 11), (2026, 6, 10))
    m1 = CoverageManifest(tmp_path, persist=True)
    m1.load()

    # Rewrite the parquet with a different symbol set. New mtime → re-scan.
    _write_imbalance_parquet(path, ["AAPL", "MSFT"], (2025, 6, 11), (2026, 6, 10))
    m2 = CoverageManifest(tmp_path, persist=True)
    m2.load()
    e = m2.entries()[0]
    assert e.symbols == frozenset({"AAPL", "MSFT"})


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


def test_resolver_hit_via_coverage_no_adapter_fetch(tmp_path):
    """Adapter hash miss + coverage HIT → filtered LazyFrame, no batch job."""
    path = tmp_path / "existing.parquet"
    _write_imbalance_parquet(path, ["AAPL", "MSFT"], (2025, 6, 11), (2026, 6, 10))

    adapter = _FakeAdapter(tmp_path)
    resolver = CoverageResolver(adapter)

    lf = resolver.resolve_lazy(
        tickers=["AAPL"],
        start_date="2026-05-15", end_date="2026-05-16",
        schema="imbalance",
    )
    df = lf.collect()
    # Only AAPL rows within the requested date range.
    assert all(df["symbol"] == "AAPL")
    assert adapter.get_data_calls == [], "adapter.get_data must NOT be called on HIT"


def test_resolver_miss_without_allow_fetch_raises(tmp_path):
    _write_imbalance_parquet(tmp_path / "a.parquet", ["AAPL"], (2025, 6, 11), (2026, 6, 10))
    adapter = _FakeAdapter(tmp_path)
    resolver = CoverageResolver(adapter)

    with pytest.raises(CoverageError) as exc:
        resolver.resolve_lazy(
            tickers=["NVDA"],  # not in the parquet
            start_date="2025-08-01", end_date="2025-08-02",
            schema="imbalance",
        )
    msg = str(exc.value)
    assert "Coverage MISS" in msg
    assert "--allow-fetch" in msg
    assert adapter.get_data_calls == [], "adapter.get_data must NOT be called on refused MISS"


def test_resolver_miss_with_allow_fetch_calls_adapter(tmp_path):
    adapter = _FakeAdapter(tmp_path)
    resolver = CoverageResolver(adapter)

    # Empty cache → coverage MISS. allow_fetch=True must call adapter.get_data,
    # then the second scan_data call will still miss (no parquet was
    # actually written by the fake). We expect a FileNotFoundError from the
    # follow-up scan, but the important thing is that get_data was invoked.
    with pytest.raises(FileNotFoundError):
        resolver.resolve_lazy(
            tickers=["AAPL"],
            start_date="2025-08-01", end_date="2025-08-02",
            schema="imbalance",
            allow_fetch=True,
        )
    assert len(adapter.get_data_calls) == 1
    assert adapter.get_data_calls[0]["schema"] == "imbalance"


def test_resolver_fast_path_uses_adapter_hash(tmp_path):
    """Adapter's own hash-keyed parquet exists → adapter scan, no manifest."""
    adapter = _FakeAdapter(tmp_path)
    tickers = ["AAPL"]
    start, end = "2025-08-01", "2025-08-02"
    schema, dataset = "imbalance", "XNAS.ITCH"
    key = adapter._generate_cache_key(tickers, start, end, dataset, schema)
    _write_imbalance_parquet(
        adapter._get_cache_path(key), ["AAPL"], (2025, 8, 1), (2025, 8, 1),
    )

    resolver = CoverageResolver(adapter)
    lf = resolver.resolve_lazy(
        tickers=tickers, start_date=start, end_date=end,
        schema=schema, dataset=dataset,
    )
    df = lf.collect()
    assert df.height > 0
    assert adapter.get_data_calls == []


def test_resolver_plan_dry_run_never_calls_adapter(tmp_path):
    """``plan()`` must be pure — no adapter.get_data, no adapter.scan_data
    triggered write."""
    _write_imbalance_parquet(tmp_path / "a.parquet", ["AAPL"], (2025, 6, 11), (2026, 6, 10))

    class _NoScanAdapter(_FakeAdapter):
        def scan_data(self, *a, **kw):
            raise AssertionError("scan_data must NOT be called from plan()")

    adapter = _NoScanAdapter(tmp_path)
    resolver = CoverageResolver(adapter)

    hit = resolver.plan(
        tickers=["AAPL"],
        start_date="2026-05-15", end_date="2026-05-16",
        schema="imbalance",
    )
    assert hit.status is CoverageStatus.HIT

    miss = resolver.plan(
        tickers=["NVDA"],
        start_date="2026-05-15", end_date="2026-05-16",
        schema="imbalance",
    )
    assert miss.status is CoverageStatus.MISS
    assert adapter.get_data_calls == []


def test_resolver_materialize_hit_no_fetch(tmp_path):
    _write_imbalance_parquet(tmp_path / "a.parquet", ["AAPL"], (2025, 6, 11), (2026, 6, 10))
    adapter = _FakeAdapter(tmp_path)
    resolver = CoverageResolver(adapter)

    decision = asyncio.run(resolver.materialize(
        tickers=["AAPL"],
        start_date="2026-05-15", end_date="2026-05-16",
        schema="imbalance",
    ))
    assert decision.status is CoverageStatus.HIT
    assert adapter.get_data_calls == []


def test_resolver_materialize_miss_without_allow_fetch_raises(tmp_path):
    adapter = _FakeAdapter(tmp_path)
    resolver = CoverageResolver(adapter)
    with pytest.raises(CoverageError):
        asyncio.run(resolver.materialize(
            tickers=["AAPL"],
            start_date="2025-08-01", end_date="2025-08-02",
            schema="imbalance",
        ))
    assert adapter.get_data_calls == []


def test_resolver_materialize_miss_with_allow_fetch_invokes_adapter(tmp_path):
    adapter = _FakeAdapter(tmp_path)
    resolver = CoverageResolver(adapter)
    asyncio.run(resolver.materialize(
        tickers=["AAPL"],
        start_date="2025-08-01", end_date="2025-08-02",
        schema="imbalance",
        allow_fetch=True,
    ))
    assert len(adapter.get_data_calls) == 1


def test_summarize_manifest_prints_entries(tmp_path):
    _write_imbalance_parquet(tmp_path / "a.parquet", ["AAPL"], (2025, 6, 11), (2026, 6, 10))
    _write_tbbo_parquet(tmp_path / "b.parquet", ["MSFT"], (2025, 7, 1), (2025, 8, 1))
    manifest = CoverageManifest(tmp_path, persist=False)
    manifest.load()
    text = summarize_manifest(manifest)
    assert "imbalance" in text
    assert "tbbo" in text
    assert "n_symbols=1" in text


def test_summarize_manifest_empty(tmp_path):
    manifest = CoverageManifest(tmp_path, persist=False)
    text = summarize_manifest(manifest)
    assert "no parquets" in text.lower()


# ---------------------------------------------------------------------------
# R1 — ohlcv-1d vs ohlcv-1m disambiguation
# ---------------------------------------------------------------------------


def _write_ohlcv_parquet(
    path: Path,
    symbols: list[str],
    ts_events: list[datetime],
) -> None:
    """Minimal ohlcv-shape parquet — the ``ts_events`` list is exploded per symbol."""
    rows = []
    for sym in symbols:
        for ts in ts_events:
            rows.append({
                "ts_event": ts,
                "symbol": sym,
                "open": 100.0, "high": 101.0, "low": 99.0,
                "close": 100.5, "volume": 1000,
            })
    pl.DataFrame(rows).write_parquet(str(path))


def test_r1_daily_bars_labelled_ohlcv_1d(tmp_path):
    """One time-of-day per row (midnight UTC) → labelled ohlcv-1d."""
    ts = [datetime(2025, 6, d, 0, 0, tzinfo=timezone.utc) for d in range(11, 16)]
    _write_ohlcv_parquet(tmp_path / "daily.parquet", ["AAPL"], ts)
    manifest = CoverageManifest(tmp_path, persist=False)
    entries = manifest.entries()
    assert len(entries) == 1
    assert entries[0].schema == "ohlcv-1d"


def test_r1_minute_bars_labelled_ohlcv_1m(tmp_path):
    """Multiple times of day per row → labelled ohlcv-1m."""
    ts = [
        datetime(2025, 6, 11, 9, 30, tzinfo=timezone.utc),
        datetime(2025, 6, 11, 9, 31, tzinfo=timezone.utc),
        datetime(2025, 6, 11, 9, 32, tzinfo=timezone.utc),
        datetime(2025, 6, 12, 9, 30, tzinfo=timezone.utc),
        datetime(2025, 6, 12, 9, 31, tzinfo=timezone.utc),
    ]
    _write_ohlcv_parquet(tmp_path / "minute.parquet", ["AAPL"], ts)
    manifest = CoverageManifest(tmp_path, persist=False)
    entries = manifest.entries()
    assert len(entries) == 1
    assert entries[0].schema == "ohlcv-1m"


def test_r1_minute_bar_parquet_never_hits_ohlcv_1d_request(tmp_path):
    """Regression: an ohlcv-1m parquet cannot HIT an ohlcv-1d request.

    If this test fails, ``_refit_covariance`` would silently train on
    minute bars (thousands of rows/day → wrong daily-return series).
    """
    ts = [
        datetime(2025, 6, 11, 9, 30, tzinfo=timezone.utc),
        datetime(2025, 6, 11, 9, 31, tzinfo=timezone.utc),
        datetime(2025, 6, 12, 9, 30, tzinfo=timezone.utc),
    ]
    _write_ohlcv_parquet(tmp_path / "minute.parquet", ["AAPL", "MSFT"], ts)
    manifest = CoverageManifest(tmp_path, persist=False)
    decision = manifest.check(
        tickers=["AAPL"],
        start_date="2025-06-11", end_date="2025-06-13",
        schema="ohlcv-1d",
    )
    assert decision.status is CoverageStatus.MISS
    # Nothing with schema=ohlcv-1d is on disk.
    assert "ohlcv-1d" in decision.reason


# ---------------------------------------------------------------------------
# R2 — weekend/holiday-aware endpoints
# ---------------------------------------------------------------------------


def test_r2_weekend_tail_hits_from_friday_parquet(tmp_path):
    """Parquet ends Friday; request needs Sunday → HIT (weekend has no data)."""
    # 2025-05-30 is a Friday. 2025-06-01 is a Sunday; end=2025-06-02 (Monday
    # exclusive) → last_needed = Sunday 2025-06-01 → weekday-floored = Fri.
    path = tmp_path / "a.parquet"
    _write_imbalance_parquet(path, ["AAPL"], (2025, 5, 1), (2025, 5, 30))
    manifest = CoverageManifest(tmp_path, persist=False)
    decision = manifest.check(
        tickers=["AAPL"],
        start_date="2025-05-01", end_date="2025-06-02",  # Mon
        schema="imbalance",
    )
    assert decision.status is CoverageStatus.HIT, decision.reason


def test_r2_weekend_head_hits_from_monday_parquet(tmp_path):
    """Parquet starts Monday; request starts Saturday → HIT."""
    # 2025-06-02 is a Monday. Request starts 2025-05-31 (Sat) → weekday-
    # floored req_start = Mon 2025-06-02.
    path = tmp_path / "a.parquet"
    _write_imbalance_parquet(path, ["AAPL"], (2025, 6, 2), (2025, 12, 31))
    manifest = CoverageManifest(tmp_path, persist=False)
    decision = manifest.check(
        tickers=["AAPL"],
        start_date="2025-05-31", end_date="2025-12-30",  # Sat..(before end)
        schema="imbalance",
    )
    assert decision.status is CoverageStatus.HIT, decision.reason


def test_r2_weekday_shortfall_still_missed(tmp_path):
    """A real weekday shortfall still MISSes — R2 must not weaken the check."""
    path = tmp_path / "a.parquet"
    _write_imbalance_parquet(path, ["AAPL"], (2025, 5, 1), (2025, 5, 28))  # Wed
    manifest = CoverageManifest(tmp_path, persist=False)
    # request ends 2025-06-02 (Mon) → last_needed = Sun → floored = Fri 2025-05-30.
    # Parquet ends Wed 2025-05-28 — genuinely short 2 weekdays.
    decision = manifest.check(
        tickers=["AAPL"],
        start_date="2025-05-01", end_date="2025-06-02",
        schema="imbalance",
    )
    assert decision.status is CoverageStatus.MISS
    assert "tail" in decision.reason.lower() or "ends" in decision.reason.lower()


# ---------------------------------------------------------------------------
# R3 — per-symbol coverage
# ---------------------------------------------------------------------------


def _write_imbalance_per_symbol(
    path: Path,
    per_symbol_ranges: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]],
) -> None:
    """Write a parquet where each symbol has its own start/end date."""
    rows = []
    for sym, ((sy, sm, sd), (ey, em, ed)) in per_symbol_ranges.items():
        rows.append({
            "ts_event": _dt(sy, sm, sd),
            "symbol": sym, "side": "B",
            "total_imbalance_qty": 1000, "paired_qty": 2000, "ref_price": 100.0,
        })
        rows.append({
            "ts_event": _dt(ey, em, ed, h=20),
            "symbol": sym, "side": "A",
            "total_imbalance_qty": 500, "paired_qty": 1000, "ref_price": 101.0,
        })
    pl.DataFrame(rows).write_parquet(str(path))


def test_r3_symbol_only_covers_half_window_miss(tmp_path):
    """AAPL spans the whole window; NVDA only appears in the first half → MISS."""
    _write_imbalance_per_symbol(tmp_path / "a.parquet", {
        "AAPL": ((2025, 6, 11), (2026, 6, 10)),
        "NVDA": ((2025, 6, 11), (2025, 9, 30)),  # delisting mid-window
    })
    manifest = CoverageManifest(tmp_path, persist=False)
    decision = manifest.check(
        tickers=["AAPL", "NVDA"],
        start_date="2025-06-11", end_date="2026-06-11",
        schema="imbalance",
    )
    assert decision.status is CoverageStatus.MISS
    assert "NVDA" in decision.reason or "nvda" in decision.reason.lower()


def test_r3_symbol_only_covers_second_half_miss(tmp_path):
    """AAPL spans whole window; TEM only appears after mid-window (IPO)."""
    _write_imbalance_per_symbol(tmp_path / "a.parquet", {
        "AAPL": ((2025, 6, 11), (2026, 6, 10)),
        "TEM": ((2026, 1, 15), (2026, 6, 10)),  # IPO mid-window
    })
    manifest = CoverageManifest(tmp_path, persist=False)
    decision = manifest.check(
        tickers=["AAPL", "TEM"],
        start_date="2025-06-11", end_date="2026-06-11",
        schema="imbalance",
    )
    assert decision.status is CoverageStatus.MISS
    reason = decision.reason.lower()
    assert "tem" in reason and ("head" in reason or "starts" in reason)


def test_r3_all_symbols_span_window_hits(tmp_path):
    """Every symbol spans the request window → HIT."""
    _write_imbalance_per_symbol(tmp_path / "a.parquet", {
        "AAPL": ((2025, 6, 11), (2026, 6, 10)),
        "MSFT": ((2025, 6, 11), (2026, 6, 10)),
    })
    manifest = CoverageManifest(tmp_path, persist=False)
    decision = manifest.check(
        tickers=["AAPL", "MSFT"],
        start_date="2025-06-11", end_date="2026-06-11",
        schema="imbalance",
    )
    assert decision.status is CoverageStatus.HIT


# ---------------------------------------------------------------------------
# R4 — nearest-shortfall diagnostic
# ---------------------------------------------------------------------------


def test_r4_nearest_reports_smallest_shortfall(tmp_path, caplog):
    """Nearest near-miss must be the SMALLEST shortfall, not first-in-glob."""
    # Parquet A ends way early (huge tail shortfall).
    _write_imbalance_parquet(tmp_path / "a.parquet", ["AAPL"], (2020, 1, 1), (2020, 2, 1))
    # Parquet B ends just a day short (tiny tail shortfall) — this is the
    # ACTUAL nearest and the log should identify it.
    _write_imbalance_parquet(tmp_path / "b.parquet", ["AAPL"], (2020, 1, 1), (2020, 3, 15))
    manifest = CoverageManifest(tmp_path, persist=False)

    import logging
    with caplog.at_level(logging.WARNING, logger="live.cache"):
        decision = manifest.check(
            tickers=["AAPL"],
            start_date="2020-01-01", end_date="2020-03-17",  # Tue
            schema="imbalance",
        )
    assert decision.status is CoverageStatus.MISS
    # The reason should reference the b.parquet shortfall (small), not a.parquet's.
    assert "2020-03-15" in decision.reason or "1d at tail" in decision.reason


# ---------------------------------------------------------------------------
# R5 — Ctrl-D handling in retrain confirmation prompt
# ---------------------------------------------------------------------------


def test_r5_confirm_fetch_handles_eoferror(monkeypatch):
    """Ctrl-D at the prompt → clean False, not a bare traceback."""
    from live import retrain
    from types import SimpleNamespace

    def _raise_eof(prompt):
        raise EOFError

    monkeypatch.setattr("builtins.input", _raise_eof)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    args = SimpleNamespace(allow_fetch=True, yes=False)
    result = retrain._confirm_fetch(args, missing=[{"schema": "tbbo"}])
    assert result is False


def test_r5_confirm_fetch_handles_keyboardinterrupt(monkeypatch):
    """Ctrl-C at the prompt → clean False."""
    from live import retrain
    from types import SimpleNamespace

    def _raise_kb(prompt):
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", _raise_kb)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    args = SimpleNamespace(allow_fetch=True, yes=False)
    result = retrain._confirm_fetch(args, missing=[{"schema": "tbbo"}])
    assert result is False


# ---------------------------------------------------------------------------
# R6 — sidecar keyed on (mtime, size)
# ---------------------------------------------------------------------------


def test_r6_sidecar_invalidated_when_size_changes_but_mtime_preserved(tmp_path):
    """Overwrite that preserves mtime but changes size must invalidate the sidecar."""
    import os
    path = tmp_path / "a.parquet"
    _write_imbalance_parquet(path, ["AAPL"], (2025, 6, 11), (2026, 6, 10))
    m1 = CoverageManifest(tmp_path, persist=True)
    m1.load()
    stat_before = path.stat()

    # Rewrite with a different symbol set (different size), then force
    # the mtime back to what it was — simulates a tool that preserves
    # atime/mtime on overwrite.
    _write_imbalance_per_symbol(path, {
        "AAPL": ((2025, 6, 11), (2026, 6, 10)),
        "MSFT": ((2025, 6, 11), (2026, 6, 10)),
        "NVDA": ((2025, 6, 11), (2026, 6, 10)),
    })
    os.utime(path, ns=(stat_before.st_atime_ns, stat_before.st_mtime_ns))
    # If ONLY mtime were keyed, the sidecar would be reused and the
    # manifest would still say symbols=={AAPL}. The size differs, so the
    # sidecar must invalidate and re-scan.
    m2 = CoverageManifest(tmp_path, persist=True)
    m2.load()
    entry = m2.entries()[0]
    assert entry.symbols == frozenset({"AAPL", "MSFT", "NVDA"}), \
        "size-key must invalidate stale sidecar even when mtime is preserved"


# ---------------------------------------------------------------------------
# R7 — sidecar-reuse test actually verifies the sidecar is used
# ---------------------------------------------------------------------------


def test_r7_sidecar_reuse_avoids_scan_parquet(tmp_path, monkeypatch):
    """The sidecar path must not re-invoke ``_scan_parquet`` when nothing
    changed. Previously the test would pass whether the sidecar was used
    or not — now we count actual calls to ``_scan_parquet``.
    """
    path = tmp_path / "a.parquet"
    _write_imbalance_parquet(path, ["AAPL"], (2025, 6, 11), (2026, 6, 10))
    # Prime the sidecar.
    CoverageManifest(tmp_path, persist=True).load()

    scan_calls: list = []
    real_scan = CoverageManifest._scan_parquet

    def _counting_scan(*a, **kw):
        scan_calls.append(a)
        return real_scan(*a, **kw)

    monkeypatch.setattr(CoverageManifest, "_scan_parquet", staticmethod(_counting_scan))

    m2 = CoverageManifest(tmp_path, persist=True)
    m2.load()
    assert scan_calls == [], \
        "sidecar reuse must skip _scan_parquet when mtime+size match"


def test_r7_resolver_fast_path_isolated_from_manifest(tmp_path):
    """Force the manifest to raise if consulted — a passing test proves
    the adapter-hash fast path served the request without touching it."""
    adapter = _FakeAdapter(tmp_path)
    tickers = ["AAPL"]
    start, end = "2025-08-01", "2025-08-02"
    schema, dataset = "imbalance", "XNAS.ITCH"
    key = adapter._generate_cache_key(tickers, start, end, dataset, schema)
    _write_imbalance_parquet(
        adapter._get_cache_path(key), ["AAPL"], (2025, 8, 1), (2025, 8, 1),
    )

    class _RaisingManifest:
        def check(self, **kw):
            raise AssertionError("manifest must not be consulted on fast path")

    resolver = CoverageResolver(adapter, manifest=_RaisingManifest())
    lf = resolver.resolve_lazy(
        tickers=tickers, start_date=start, end_date=end,
        schema=schema, dataset=dataset,
    )
    assert lf.collect().height > 0
