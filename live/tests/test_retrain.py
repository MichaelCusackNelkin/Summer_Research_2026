"""Regression tests for the atomic-write helpers in ``live/retrain.py``.

The R1 review found that ``_atomic_np_save`` produced 0-byte files: it
handed a string path lacking ``.npy`` to ``np.save``, which silently
appended the suffix and wrote the real data to a different file, then
``os.replace`` moved the empty original into the target path.

The existing test suite mocked out ``_load_cov_artifacts`` and never
touched these helpers, which is how the bug slipped through. These tests
call each helper directly and round-trip the result.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from live.retrain import (
    _atomic_joblib_dump,
    _atomic_np_save,
    _atomic_write_parquet,
)


def _leaked_tmp_files(dir_: Path) -> list[str]:
    """List anything in ``dir_`` that looks like a leaked temp file.

    We match on ``.tmp`` anywhere in the name because our helpers use
    ``.npy.tmp`` / ``.joblib.tmp`` / ``.parquet.tmp`` suffixes.
    """
    return [p.name for p in dir_.iterdir() if ".tmp" in p.name]


def test_atomic_np_save_writes_real_data(tmp_path):
    """R1 regression: the file must contain the actual array bytes."""
    path = tmp_path / "x.npy"
    array = np.array([1.0, 2.0, 3.0])
    _atomic_np_save(path, array)
    assert path.exists(), "target file was not created"
    assert path.stat().st_size > 0, "target file is empty (R1 regression)"
    loaded = np.load(path)
    assert np.array_equal(loaded, array)
    assert _leaked_tmp_files(tmp_path) == [], "temp file left behind"


def test_atomic_np_save_multidim_round_trip(tmp_path):
    """A more realistic shape — matches the covariance artifacts."""
    path = tmp_path / "cov.npy"
    array = np.random.RandomState(0).randn(100, 100)
    _atomic_np_save(path, array)
    loaded = np.load(path)
    assert loaded.shape == array.shape
    assert np.allclose(loaded, array)


def test_atomic_np_save_overwrites_existing(tmp_path):
    path = tmp_path / "x.npy"
    _atomic_np_save(path, np.array([9.0]))
    assert path.stat().st_size > 0
    _atomic_np_save(path, np.array([1.0, 2.0, 3.0, 4.0]))
    assert np.array_equal(np.load(path), np.array([1.0, 2.0, 3.0, 4.0]))
    assert _leaked_tmp_files(tmp_path) == []


def test_atomic_np_save_failure_cleans_up_tmp(tmp_path):
    """If the write raises, no half-baked tmp file is left."""
    path = tmp_path / "x.npy"

    class _BadArray:
        # Something np.save will reject
        def __array__(self, *a, **kw):
            raise RuntimeError("boom")

    with pytest.raises(Exception):
        _atomic_np_save(path, _BadArray())
    assert not path.exists(), "target should not exist after failure"
    assert _leaked_tmp_files(tmp_path) == [], "temp file left behind after failure"


def test_atomic_joblib_dump_round_trip(tmp_path):
    path = tmp_path / "pipe.joblib"
    pipe = make_pipeline(StandardScaler(), Ridge(alpha=1e-4))
    X = np.random.RandomState(0).randn(50, 3)
    y = X @ np.array([0.1, -0.1, 0.2])
    pipe.fit(X, y)

    _atomic_joblib_dump(path, pipe)
    assert path.exists()
    assert path.stat().st_size > 0
    loaded = joblib.load(path)
    assert np.allclose(loaded.predict(X), pipe.predict(X))
    assert _leaked_tmp_files(tmp_path) == []


def test_atomic_write_parquet_round_trip(tmp_path):
    path = tmp_path / "ref.parquet"
    frame = pl.DataFrame({
        "symbol": ["AAA", "BBB", "CCC"],
        "ADV": [1e9, 5e8, 2e9],
        "vol": [0.02, 0.03, 0.025],
    })
    _atomic_write_parquet(path, frame)
    assert path.exists()
    assert path.stat().st_size > 0
    loaded = pl.read_parquet(path)
    assert loaded.equals(frame)
    assert _leaked_tmp_files(tmp_path) == []


def test_atomic_write_parquet_overwrites_existing(tmp_path):
    path = tmp_path / "ref.parquet"
    _atomic_write_parquet(path, pl.DataFrame({"symbol": ["A"], "ADV": [1.0], "vol": [0.1]}))
    frame2 = pl.DataFrame({"symbol": ["B"], "ADV": [2.0], "vol": [0.2]})
    _atomic_write_parquet(path, frame2)
    assert pl.read_parquet(path).equals(frame2)
    assert _leaked_tmp_files(tmp_path) == []


# ---------------------------------------------------------------------------
# --allow-fetch / --yes gating
#
# These tests drive ``retrain.main`` end-to-end with a fake adapter so we can
# assert that the coverage layer refuses (or authorises) fetches according
# to CLI flags. All artifacts + universes are monkey-patched so the tests
# don't need real cache files.
# ---------------------------------------------------------------------------


import os
from live import config, retrain
from live.cache import CoverageResolver, CoverageStatus


class _RecordingFakeAdapter:
    """Fake adapter — records every batch-API call. If ``get_data`` is
    invoked, the test fails a downstream ``assert``.
    """

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
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

    def _get_cache_path(self, key):
        return self.cache_dir / f"{key}.parquet"

    def scan_data(self, tickers, start_date, end_date, dataset="XNAS.ITCH", schema="ohlcv-1m"):
        key = self._generate_cache_key(tickers, start_date, end_date, dataset, schema)
        path = self._get_cache_path(key)
        if not path.exists():
            raise FileNotFoundError(str(path))
        return pl.scan_parquet(str(path), low_memory=True)

    async def get_data(self, tickers, start_date, end_date, dataset="XNAS.ITCH",
                       schema="ohlcv-1m", materialize_only=False, **kw):
        self.get_data_calls.append({
            "tickers": tickers, "start": start_date, "end": end_date,
            "dataset": dataset, "schema": schema,
        })


@pytest.fixture
def _fake_adapter_env(tmp_path, monkeypatch):
    """Common wiring: fresh empty cache dir, fake adapter, minimal universe."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    adapter = _RecordingFakeAdapter(cache_dir)
    monkeypatch.setattr(retrain, "DatabentoGenericAdapter", lambda *a, **kw: adapter)
    monkeypatch.setattr(config, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(retrain.config, "CACHE_DIR", cache_dir)
    monkeypatch.setenv("DATABENTO_API_KEY", "dummy")

    # Small universes so we don't read the real files.
    monkeypatch.setattr(retrain.config, "load_broad_universe", lambda _f: ["AAPL", "MSFT"])
    monkeypatch.setattr(retrain.config, "load_active_universe", lambda _f: ["AAPL", "MSFT"])
    return adapter, cache_dir


def test_retrain_dry_run_never_calls_adapter(_fake_adapter_env):
    adapter, _ = _fake_adapter_env
    rc = retrain.main(["--refit", "all", "--dry-run", "--end-date", "2026-07-15"])
    assert rc == 0
    assert adapter.get_data_calls == [], "dry-run must NEVER hit the adapter"


def test_retrain_miss_without_allow_fetch_exits_nonzero(_fake_adapter_env):
    """Empty cache → all schemas MISS → refuse without --allow-fetch."""
    adapter, _ = _fake_adapter_env
    rc = retrain.main(["--refit", "forecast", "--end-date", "2026-07-15"])
    assert rc == 9, f"expected rc=9 (refused fetch), got {rc}"
    assert adapter.get_data_calls == []


def test_retrain_miss_with_allow_fetch_but_no_yes_refuses(monkeypatch, _fake_adapter_env):
    """--allow-fetch alone (no --yes) refuses on a non-TTY. Batch API costs
    real money — belt AND suspenders."""
    adapter, _ = _fake_adapter_env
    # Force non-TTY so no interactive prompt is offered.
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    rc = retrain.main([
        "--refit", "forecast", "--end-date", "2026-07-15", "--allow-fetch",
    ])
    assert rc == 10, f"expected rc=10 (no confirmation), got {rc}"
    assert adapter.get_data_calls == []


def test_retrain_coverage_plan_names_nearest_parquet_and_reason(
    monkeypatch, tmp_path,
):
    """Operator-facing regression: the ``coverage plan`` INFO line must name
    the closest parquet AND the reason so a MISS isn't opaque.

    Setup: cache holds a tbbo parquet covering AAPL+MSFT over the target
    window. Retrain requests the same window with CRWV in the required set
    → single-symbol MISS whose ``format_coverage_decision`` output should
    include the parquet filename and ``missing 1 required symbol(s)``.
    """
    import logging
    from datetime import datetime, timezone
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    # A parquet the manifest can fingerprint as ``tbbo`` (via the
    # ask_px_00/bid_px_00 signature) covering the whole 12-month window
    # for AAPL+MSFT only. CRWV is deliberately absent so the required-set
    # check misses.
    start = datetime(2025, 7, 15, tzinfo=timezone.utc)
    end = datetime(2026, 7, 14, 20, tzinfo=timezone.utc)
    rows = []
    for sym in ("AAPL", "MSFT"):
        for ts in (start, end):
            rows.append({
                "ts_event": ts,
                "symbol": sym,
                "price": 100.0,
                "size": 100,
                "ask_px_00": 100.05,
                "bid_px_00": 99.95,
            })
    tbbo_path = cache_dir / "cached-tbbo.parquet"
    pl.DataFrame(rows).write_parquet(str(tbbo_path))

    adapter = _RecordingFakeAdapter(cache_dir)
    # Reuse the same cache_dir the fake adapter created; the fixture
    # normally makes its own but we want ours (with the seeded parquet).
    adapter.cache_dir = cache_dir
    monkeypatch.setattr(retrain, "DatabentoGenericAdapter", lambda *a, **kw: adapter)
    monkeypatch.setattr(config, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(retrain.config, "CACHE_DIR", cache_dir)
    monkeypatch.setenv("DATABENTO_API_KEY", "dummy")
    monkeypatch.setattr(
        retrain.config, "load_broad_universe",
        lambda _f: ["AAPL", "MSFT", "CRWV"],
    )
    monkeypatch.setattr(
        retrain.config, "load_active_universe",
        lambda _f: ["AAPL", "MSFT", "CRWV"],
    )

    # ``live.retrain``'s configured stream logger has ``propagate=False``,
    # so root-attached caplog can't see its records. Attach a handler
    # directly on the retrain logger for this test — records land in
    # ``captured`` regardless of propagation.
    captured: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            captured.append(record)

    handler = _Capture(level=logging.INFO)
    retrain.logger.addHandler(handler)
    try:
        rc = retrain.main([
            "--refit", "impact", "--dry-run", "--end-date", "2026-07-15",
        ])
    finally:
        retrain.logger.removeHandler(handler)
    assert rc == 0
    plan_lines = [
        r.getMessage() for r in captured
        if "coverage plan" in r.getMessage()
    ]
    assert plan_lines, f"no coverage plan log; got: {[r.getMessage() for r in captured]}"
    miss_line = next(
        (line for line in plan_lines if "MISS" in line),
        None,
    )
    assert miss_line is not None, f"expected a MISS line, got: {plan_lines}"
    # Both the parquet filename AND the reason must appear in the message.
    assert "cached-tbbo.parquet" in miss_line, miss_line
    assert "missing 1 required symbol(s)" in miss_line, miss_line
    assert "CRWV" in miss_line, miss_line
    assert "shortfall=1" in miss_line, miss_line


def test_retrain_miss_with_allow_fetch_and_yes_fetches(monkeypatch, _fake_adapter_env):
    """--allow-fetch + --yes together authorises the batch API call.

    We stub out the actual refit implementations because the fake adapter
    doesn't produce real parquets; the point is to prove the gate lets
    the fetch happen and to reach the refit body.
    """
    adapter, _ = _fake_adapter_env

    async def _noop_refit_forecast(**kw):
        # Simulate an unrelated failure inside the refit body — irrelevant.
        # What we care about is that fetch_allowed reached this call.
        assert kw.get("allow_fetch") is True, \
            "resolver should be authorised to fetch after --yes"
        return None

    monkeypatch.setattr(retrain, "_refit_forecast", _noop_refit_forecast)

    rc = retrain.main([
        "--refit", "forecast", "--end-date", "2026-07-15",
        "--allow-fetch", "--yes",
    ])
    assert rc == 0

