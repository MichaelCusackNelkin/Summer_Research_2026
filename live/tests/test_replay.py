"""End-to-end tests for the ``--replay-date`` dress-rehearsal path.

Mirrors the shape of ``test_end_to_end.py`` but through ``ReplayFeed``
instead of ``LiveFeed``, and drives ``daily_orchestrator.main`` end-to-end
with a mocked broker so the ``stream -> features -> alpha -> MVO -> orders``
chain is exercised on cached historical data.

Guarantees under test:

- Replay mode produces a non-empty book and places MOC orders through the
  broker (same primitives paper trading uses).
- ``--replay-date`` + ``--live`` is refused with a distinct non-zero rc
  and no broker connection is attempted.
- Replay skips both the stream-start and snapshot wall-clock waits — a
  run at an arbitrary wall-clock time (including "past" the notional
  submit-by) still completes without hitting ``_sleep_until``.
- The persisted target record is tagged ``mode="replay"`` and written to
  the ``targets/replay/`` subdirectory so it can never be picked up by
  the live/paper reconciliation.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from live import config
from live import daily_orchestrator
from live.features import ALPHA_FEATURES
from live.replay_feed import ReplayFeed


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakePosition:
    __slots__ = ("symbol", "quantity", "avg_cost")

    def __init__(self, symbol, quantity, avg_cost=100.0):
        self.symbol = symbol
        self.quantity = quantity
        self.avg_cost = avg_cost


class _FakeOrderStatus:
    def __init__(self, order_id, symbol, action, requested, filled, status="Filled"):
        self.order_id = order_id
        self.symbol = symbol
        self.action = action
        self.quantity = requested
        self.status = status
        self.filled = filled
        self.avg_fill_price = 100.0


class _FakeBroker:
    """Same shape as test_end_to_end._FakeBroker — kept local so the two
    files can evolve independently. Models fills into an internal
    position book so the MOC counter-flatten's ``get_positions()``
    snapshot behaves like a real broker after the MKT entry batch.
    """

    LIVE_CONFIRM_MSG = "YES_I_WANT_LIVE_TRADING"
    MAX_ORDER_QTY = 10_000

    def __init__(self):
        self.connected = False
        self.disconnected = False
        self.buy_calls: list[tuple[str, int, str, dict]] = []
        self.sell_calls: list[tuple[str, int, str, dict]] = []
        self.qualified_symbols: list[str] = []
        self._positions: dict[str, int] = {}
        self._orders: dict[str, dict] = {}
        self._order_counter = 0

    def connect(self, mode="paper", **kwargs):
        self.connected = True

    def disconnect(self):
        self.disconnected = True

    def get_positions(self):
        return [_FakePosition(s, q) for s, q in self._positions.items() if q != 0]

    def _make_contract(self, symbol, exchange="SMART", currency="USD"):
        self.qualified_symbols.append(symbol)
        return object()

    def _record_fill(self, oid, symbol, action, requested, order_type):
        filled = int(requested)  # full-fill replay broker
        signed = filled if action == "BUY" else -filled
        self._positions[symbol] = self._positions.get(symbol, 0) + signed
        self._orders[oid] = {
            "symbol": symbol, "action": action, "requested": int(requested),
            "filled": filled, "order_type": order_type, "status": "Filled",
        }

    def buy(self, symbol, quantity, order_type="MKT", **kwargs):
        self.buy_calls.append((symbol, int(quantity), order_type, dict(kwargs)))
        self._order_counter += 1
        oid = f"oid-{self._order_counter}"
        self._record_fill(oid, symbol, "BUY", int(quantity), order_type)
        return oid

    def sell(self, symbol, quantity, order_type="MKT", **kwargs):
        self.sell_calls.append((symbol, int(quantity), order_type, dict(kwargs)))
        self._order_counter += 1
        oid = f"oid-{self._order_counter}"
        self._record_fill(oid, symbol, "SELL", int(quantity), order_type)
        return oid

    def cancel(self, order_id):
        info = self._orders.get(order_id)
        if info is not None and info["status"] not in {
            "Filled", "Cancelled", "ApiCancelled", "Inactive",
        }:
            info["status"] = "ApiCancelled"

    def reconcile_positions(self, expected):
        return {}

    def wait_for_fill(self, order_id, timeout=30.0):
        info = self._orders.get(order_id)
        if info is None:
            return _FakeOrderStatus(order_id, "UNKNOWN", "BUY", 0, 0, status="Unknown")
        return _FakeOrderStatus(
            order_id=order_id, symbol=info["symbol"], action=info["action"],
            requested=info["requested"], filled=info["filled"], status=info["status"],
        )

    def get_order_status(self, order_id):
        info = self._orders.get(order_id)
        if info is None:
            raise ValueError(f"unknown order {order_id}")
        return _FakeOrderStatus(
            order_id=order_id, symbol=info["symbol"], action=info["action"],
            requested=info["requested"], filled=info["filled"], status=info["status"],
        )


# ---------------------------------------------------------------------------
# Synthetic historical parquet source
# ---------------------------------------------------------------------------


def _synthetic_imbalance_frame(replay_date: date, universe: list[str]) -> pl.DataFrame:
    """Same shape as ``imbalance_records_to_frame`` output."""
    ts_ny = datetime(
        replay_date.year, replay_date.month, replay_date.day, 15, 50, 1,
        tzinfo=config.NY_TZ,
    )
    rows = []
    for i, sym in enumerate(universe):
        base = 100.0 + i * 3
        rows.append({
            "ts_event": ts_ny,
            "symbol": sym,
            "side": "B" if (i % 2) == 0 else "A",
            "total_imbalance_qty": 100_000 + 1_000 * i,
            "paired_qty": 500_000,
            "ref_price": base + 0.5,
        })
    return pl.DataFrame(rows).with_columns(
        pl.col("ts_event").dt.convert_time_zone("America/New_York"),
        pl.col("symbol").cast(pl.Categorical),
    )


def _synthetic_tbbo_frame(replay_date: date, universe: list[str]) -> pl.DataFrame:
    rows = []
    for i, sym in enumerate(universe):
        base = 100.0 + i * 3
        for hh, mm, ss in [(15, 40, 0), (15, 45, 0), (15, 49, 58), (15, 50, 30)]:
            ts_ny = datetime(
                replay_date.year, replay_date.month, replay_date.day,
                hh, mm, ss, tzinfo=config.NY_TZ,
            )
            rows.append({
                "ts_event": ts_ny,
                "symbol": sym,
                "price": base + 0.01 * mm,
                "size": 100,
                "ask_px_00": base + 0.01 * mm + 0.005,
                "bid_px_00": base + 0.01 * mm - 0.005,
            })
    return pl.DataFrame(rows).with_columns(
        pl.col("ts_event").dt.convert_time_zone("America/New_York"),
        pl.col("symbol").cast(pl.Categorical),
    )


def _make_stub_replay_feed(replay_date_iso: str, universe: list[str]) -> ReplayFeed:
    """Bypass ReplayFeed's adapter entirely by pre-baking the frames."""
    class _NoAdapter:
        pass

    feed = ReplayFeed(
        adapter=_NoAdapter(),
        universe=universe,
        replay_date=replay_date_iso,
    )
    replay_date = feed.replay_date

    imb = _synthetic_imbalance_frame(replay_date, universe)
    tbbo = _synthetic_tbbo_frame(replay_date, universe)

    def _snapshot():
        return {"imbalance": imb, "tbbo": tbbo}

    def _start():
        feed._started = True

    def _stop(timeout: float = 0.0):
        feed._stopped = True

    feed.start = _start  # type: ignore[method-assign]
    feed.stop = _stop  # type: ignore[method-assign]
    feed.snapshot = _snapshot  # type: ignore[method-assign]
    return feed


# ---------------------------------------------------------------------------
# Shared plumbing (mirrors test_end_to_end helpers)
# ---------------------------------------------------------------------------


def _tiny_pipeline():
    N_features = len(ALPHA_FEATURES)
    X = np.random.RandomState(0).randn(500, N_features)
    y = X @ np.linspace(0.1, -0.1, N_features) * 1e-3
    pipe = make_pipeline(StandardScaler(), Ridge(alpha=1e-4))
    pipe.fit(X, y)
    return pipe


def _in_memory_artifacts(universe: list[str]):
    N = len(universe)
    rng = np.random.RandomState(1)
    L = rng.randn(10, N) * 0.01
    Sigma_f = np.eye(10) * 1e-4
    eps_diag = np.full(N, 1e-4)
    Sigma_r = L.T @ Sigma_f @ L + np.diag(eps_diag)
    ref_stats = pl.DataFrame({
        "symbol": universe,
        "ADV": np.full(N, 1e9),
        "vol": np.full(N, 0.02),
    })
    return {"L": L, "Sigma_f": Sigma_f, "Sigma_r": Sigma_r,
            "eps_diag": eps_diag, "ref_stats": ref_stats}


def _patch_artifacts(monkeypatch, universe: list[str]):
    arts = _in_memory_artifacts(universe)
    monkeypatch.setattr(daily_orchestrator, "_load_forecast_pipe", lambda: _tiny_pipeline())
    monkeypatch.setattr(
        daily_orchestrator, "_load_cov_artifacts",
        lambda: {"L": arts["L"], "Sigma_f": arts["Sigma_f"],
                 "Sigma_r": arts["Sigma_r"], "eps_diag": arts["eps_diag"]},
    )
    monkeypatch.setattr(
        daily_orchestrator, "_load_impact_vector",
        lambda use_vector, fallback_scalar, N: fallback_scalar,
    )
    monkeypatch.setattr(
        daily_orchestrator, "_load_reference_stats",
        lambda path, universe: arts["ref_stats"],
    )


def _patch_target_dirs(monkeypatch, tmp_path):
    """Redirect target-book writes to tmp_path so tests don't pollute
    ``live/logs/targets/`` on disk."""
    monkeypatch.setattr(daily_orchestrator, "_TARGETS_DIR", tmp_path / "targets")
    monkeypatch.setattr(daily_orchestrator, "_FILLS_DIR", tmp_path / "fills")


def _base_argv(replay_date: str) -> list[str]:
    # No --stream-start-time / --rebalance-time / --submit-by / --print-wait-secs
    # overrides — replay must skip those waits regardless of what they're set to.
    return [
        "--replay-date", replay_date,
        "--paper",
        "--vol-cap", "5000",
        "--on-feed-failure", "abort",
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_replay_default_is_broker_free(monkeypatch, tmp_path):
    """F2: replay without --replay-submit must not touch the broker."""
    universe = config.load_active_universe()
    stub_feed = _make_stub_replay_feed("2026-05-15", universe)
    monkeypatch.setattr(daily_orchestrator, "ReplayFeed", lambda **kw: stub_feed)
    monkeypatch.setenv("DATABENTO_API_KEY", "dummy-key")
    _patch_artifacts(monkeypatch, universe)
    _patch_target_dirs(monkeypatch, tmp_path)

    def _blowup(_args):
        raise AssertionError("broker connect happened without --replay-submit")

    monkeypatch.setattr(daily_orchestrator, "_connect_broker", _blowup)

    # No --replay-submit, no --dry-run. This is the DEFAULT replay path.
    rc = daily_orchestrator.main(_base_argv("2026-05-15"))
    assert rc == 0, f"expected rc=0 (broker-free replay), got {rc}"
    # Target book still persisted.
    replay_files = list((tmp_path / "targets" / "replay").glob("*.json"))
    assert replay_files, "target record should still be persisted"


def test_replay_submit_places_moc_orders(monkeypatch, tmp_path):
    """F2: --replay-submit exercises the full per-leg MOC path.

    F1 regression: the per-leg cutoff must be neutralised in replay, so this
    test is deterministic regardless of wall-clock time of day. We
    additionally pass ``--submit-by 00:01`` so the un-fixed code would
    silently skip every leg.
    """
    universe = config.load_active_universe()
    stub_feed = _make_stub_replay_feed("2026-05-15", universe)
    monkeypatch.setattr(daily_orchestrator, "ReplayFeed", lambda **kw: stub_feed)
    monkeypatch.setenv("DATABENTO_API_KEY", "dummy-key")
    _patch_artifacts(monkeypatch, universe)
    _patch_target_dirs(monkeypatch, tmp_path)

    fake_broker = _FakeBroker()

    def _fake_connect(_args):
        fake_broker.connect(mode="paper")
        return fake_broker

    monkeypatch.setattr(daily_orchestrator, "_connect_broker", _fake_connect)

    argv = _base_argv("2026-05-15") + ["--replay-submit", "--submit-by", "00:01"]
    rc = daily_orchestrator.main(argv)
    assert rc == 0, f"expected rc=0, got {rc}"
    assert fake_broker.connected
    assert fake_broker.disconnected
    total_orders = len(fake_broker.buy_calls) + len(fake_broker.sell_calls)
    assert total_orders > 0, (
        "no MOC orders reached the broker — the per-leg submit-by check "
        "was NOT neutralised in replay (F1 regression)."
    )


def test_replay_submit_without_replay_date_refused(monkeypatch, tmp_path):
    """F2: --replay-submit outside replay mode is refused with a distinct rc."""
    universe = config.load_active_universe()
    monkeypatch.setenv("DATABENTO_API_KEY", "dummy-key")
    _patch_artifacts(monkeypatch, universe)
    _patch_target_dirs(monkeypatch, tmp_path)

    def _blowup(_args):
        raise AssertionError("broker connect happened despite --replay-submit misuse")

    monkeypatch.setattr(daily_orchestrator, "_connect_broker", _blowup)

    argv = ["--paper", "--replay-submit"]
    rc = daily_orchestrator.main(argv)
    assert rc == 7, f"expected rc=7 (replay-submit without replay-date), got {rc}"


def test_replay_refuses_live(monkeypatch, tmp_path):
    """--replay-date + --live is refused with a distinct rc and NO broker
    connection is attempted."""
    universe = config.load_active_universe()
    monkeypatch.setenv("DATABENTO_API_KEY", "dummy-key")
    _patch_artifacts(monkeypatch, universe)
    _patch_target_dirs(monkeypatch, tmp_path)

    def _blowup(_args):
        raise AssertionError("broker connect happened despite --replay-date + --live")

    monkeypatch.setattr(daily_orchestrator, "_connect_broker", _blowup)

    argv = [
        "--replay-date", "2026-05-15",
        "--live", "--live-confirm", "YES_I_WANT_LIVE_TRADING",
    ]
    rc = daily_orchestrator.main(argv)
    assert rc == 6, f"expected rc=6 (replay+live refusal), got {rc}"


def test_replay_skips_wall_clock_waits(monkeypatch, tmp_path):
    """Replay must NOT call _sleep_until AND must still submit real orders
    with a past --submit-by. Guards against F1 (per-leg cutoff regression).
    """
    universe = config.load_active_universe()
    stub_feed = _make_stub_replay_feed("2026-05-15", universe)
    monkeypatch.setattr(daily_orchestrator, "ReplayFeed", lambda **kw: stub_feed)
    monkeypatch.setenv("DATABENTO_API_KEY", "dummy-key")
    _patch_artifacts(monkeypatch, universe)
    _patch_target_dirs(monkeypatch, tmp_path)

    sleep_calls: list = []

    def _boom_sleep(*a, **kw):
        sleep_calls.append((a, kw))
        raise AssertionError("_sleep_until called during replay run")

    monkeypatch.setattr(daily_orchestrator, "_sleep_until", _boom_sleep)

    fake_broker = _FakeBroker()

    def _fake_connect(_args):
        fake_broker.connect(mode="paper")
        return fake_broker

    monkeypatch.setattr(daily_orchestrator, "_connect_broker", _fake_connect)

    # Past --submit-by (00:01 ET) is a stress test: live/paper would exit
    # rc=4 pre-submit and skip every leg per-leg; replay must ignore both
    # cutoffs, place real orders through the broker, and never sleep.
    argv = _base_argv("2026-05-15") + ["--replay-submit", "--submit-by", "00:01"]
    rc = daily_orchestrator.main(argv)
    assert rc == 0, f"expected rc=0, got {rc}"
    assert sleep_calls == [], "replay should never call _sleep_until"
    total_orders = len(fake_broker.buy_calls) + len(fake_broker.sell_calls)
    assert total_orders > 0, (
        "no orders placed — the per-leg submit-by check tripped and "
        "silently skipped every leg (F1 regression)."
    )


def test_replay_persists_tagged_target_book(monkeypatch, tmp_path):
    """The persisted record is tagged mode=replay and lands in the
    replay subdirectory so reconciliation cannot pick it up."""
    universe = config.load_active_universe()
    stub_feed = _make_stub_replay_feed("2026-05-15", universe)
    monkeypatch.setattr(daily_orchestrator, "ReplayFeed", lambda **kw: stub_feed)
    monkeypatch.setenv("DATABENTO_API_KEY", "dummy-key")
    _patch_artifacts(monkeypatch, universe)
    _patch_target_dirs(monkeypatch, tmp_path)

    # Dry-run so we don't need the broker path.
    rc = daily_orchestrator.main(_base_argv("2026-05-15") + ["--dry-run"])
    assert rc == 0

    replay_dir = tmp_path / "targets" / "replay"
    assert replay_dir.exists(), "replay subdirectory not created"
    files = list(replay_dir.glob("*.json"))
    assert len(files) == 1, f"expected 1 replay record, got {files}"
    assert files[0].name == "2026-05-15.json"
    # The top-level targets dir must NOT contain the replay record.
    top_files = list((tmp_path / "targets").glob("*.json"))
    assert top_files == [], f"replay leaked into top-level targets: {top_files}"

    import json
    record = json.loads(files[0].read_text())
    assert record["mode"] == "replay"
    assert record["replay_date"] == "2026-05-15"
    assert record["targets"], "expected non-empty target book"


# ---------------------------------------------------------------------------
# ReplayFeed unit tests (no daily orchestrator involvement)
# ---------------------------------------------------------------------------


def test_replay_feed_snapshot_matches_live_feed_columns(monkeypatch):
    """Column-name parity: ReplayFeed.snapshot() output must be
    indistinguishable from live's ``imbalance_records_to_frame`` /
    ``tbbo_records_to_frame`` to _build_live_day_frame.
    """
    from live.features import imbalance_records_to_frame, tbbo_records_to_frame
    universe = ["AAA", "BBB", "CCC"]
    replay_date = date(2026, 5, 15)
    imb = _synthetic_imbalance_frame(replay_date, universe)
    tbbo = _synthetic_tbbo_frame(replay_date, universe)

    class _StubAdapter:
        def scan_data(self, tickers, start_date, end_date, dataset, schema):
            # Return the pre-baked frames as LazyFrames.
            return imb.lazy() if schema == "imbalance" else tbbo.lazy()

    class _StubResolver:
        """Minimal resolver stub — bypasses the manifest/adapter dance so
        the test focuses on ReplayFeed's column plumbing."""
        async def materialize(self, **kw):
            return None
        def resolve_lazy(self, *, schema, **kw):
            return imb.lazy() if schema == "imbalance" else tbbo.lazy()
        def plan(self, **kw):
            from live.cache import CoverageDecision, CoverageStatus
            return CoverageDecision(status=CoverageStatus.HIT, reason="stub")

    feed = ReplayFeed(adapter=_StubAdapter(), universe=universe,
                      replay_date=replay_date, resolver=_StubResolver())
    feed.start()
    snap = feed.snapshot()

    # Compare column sets against the live-side frame constructors.
    imb_ref_cols = set(imbalance_records_to_frame([]).columns)
    tbbo_ref_cols = set(tbbo_records_to_frame([]).columns)
    assert set(snap["imbalance"].columns) == imb_ref_cols
    assert set(snap["tbbo"].columns) == tbbo_ref_cols
    assert snap["imbalance"].height == len(universe)
    # tbbo has 4 timestamps per symbol
    assert snap["tbbo"].height == 4 * len(universe)


def test_replay_feed_refuses_unsupported_schema():
    class _StubAdapter:
        def scan_data(self, **kw):
            raise AssertionError("should not be called")

    with pytest.raises(ValueError):
        ReplayFeed(adapter=_StubAdapter(), universe=["A"],
                   replay_date="2026-05-15", schemas=("mbp-1",))



def test_replay_feed_default_allow_fetch_is_false(monkeypatch):
    """The default must be safe — never fetch without explicit consent.

    This is the guardrail against the bug the whole cache layer was
    written to prevent: a --replay-date whose data is not on disk (or
    inside a coverage-HIT parquet) must not silently trigger a Databento
    historical batch job.
    """
    class _EmptyAdapter:
        cache_dir = None  # doesn't matter — resolver never gets invoked

    feed = ReplayFeed(
        adapter=_EmptyAdapter(),
        universe=["AAA"],
        replay_date="2026-05-15",
    )
    assert feed.allow_fetch is False, \
        "ReplayFeed must default to allow_fetch=False (guard against silent batch jobs)"


def test_replay_feed_miss_without_allow_fetch_refuses(tmp_path):
    """Cold cache + allow_fetch=False → CoverageError, no adapter call.

    Uses the real ``CoverageResolver`` path (not a monkey-patched stub)
    so this regression-tests the actual coverage guard.
    """
    from live.cache import CoverageError, CoverageResolver

    class _EmptyAdapter:
        def __init__(self, cache_dir):
            self.cache_dir = cache_dir
            self.get_data_calls: list = []

        def _normalize_tickers(self, tickers):
            return sorted({t.upper() for t in tickers} if not isinstance(tickers, str) else [tickers.upper()])

        def _generate_cache_key(self, tickers, start, end, dataset, schema):
            import hashlib, json
            payload = json.dumps({
                "tickers": self._normalize_tickers(tickers), "start": start,
                "end": end, "dataset": dataset, "schema": schema,
            }, sort_keys=True)
            return hashlib.md5(payload.encode()).hexdigest()

        def _get_cache_path(self, key):
            return self.cache_dir / f"{key}.parquet"

        def scan_data(self, **kw):
            raise FileNotFoundError("empty cache")

        async def get_data(self, **kw):
            self.get_data_calls.append(kw)

    adapter = _EmptyAdapter(tmp_path)
    feed = ReplayFeed(
        adapter=adapter, universe=["AAA"],
        replay_date="2026-05-15", allow_fetch=False,
    )

    with pytest.raises(CoverageError):
        feed.start()
    assert adapter.get_data_calls == [], \
        "adapter.get_data must NOT be called when allow_fetch=False"


def test_replay_feed_coverage_plan_names_nearest_parquet_and_reason(tmp_path):
    """Regression: the per-schema INFO log at the replay call site must
    name the closest cached parquet AND the shortfall reason so a MISS
    at replay time isn't opaque.

    Setup: seed the cache with an ``imbalance`` parquet that covers the
    replay date's window but is missing the requested symbol
    (``CRWV``). Attach a handler to ``live.replay_feed`` and assert the
    logged line contains both the parquet filename and the
    "missing 1 required symbol(s)" phrase.
    """
    import logging
    from datetime import datetime, timezone

    import polars as pl

    from live import replay_feed as replay_feed_mod
    from live.replay_feed import ReplayFeed

    # Seed one imbalance parquet with AAPL only; CRWV is the required
    # symbol so the manifest MISS will read "missing 1 required symbol(s),
    # e.g. ['CRWV']".
    start = datetime(2026, 5, 15, 13, 30, tzinfo=timezone.utc)
    end = datetime(2026, 5, 15, 20, 00, tzinfo=timezone.utc)
    rows = []
    for ts in (start, end):
        rows.append({
            "ts_event": ts,
            "symbol": "AAPL",
            "side": "B",
            "total_imbalance_qty": 1000,
            "paired_qty": 2000,
            "ref_price": 100.0,
        })
    imb_path = tmp_path / "seed-imbalance.parquet"
    pl.DataFrame(rows).write_parquet(str(imb_path))

    class _EmptyishAdapter:
        def __init__(self, cache_dir):
            self.cache_dir = cache_dir
            self.get_data_calls: list = []

        def _normalize_tickers(self, tickers):
            if isinstance(tickers, str):
                return [tickers]
            return sorted({t.upper() for t in tickers})

        def _generate_cache_key(self, tickers, start, end, dataset, schema):
            import hashlib, json
            payload = json.dumps({
                "tickers": self._normalize_tickers(tickers), "start": start,
                "end": end, "dataset": dataset, "schema": schema,
            }, sort_keys=True)
            return hashlib.md5(payload.encode()).hexdigest()

        def _get_cache_path(self, key):
            return self.cache_dir / f"{key}.parquet"

        def scan_data(self, **kw):
            raise FileNotFoundError("no exact-hash slot")

        async def get_data(self, **kw):
            self.get_data_calls.append(kw)

    feed = ReplayFeed(
        adapter=_EmptyishAdapter(tmp_path),
        universe=["CRWV"],
        replay_date="2026-05-15",
        schemas=("imbalance",),
        allow_fetch=False,
    )

    captured: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            captured.append(record)

    handler = _Capture(level=logging.INFO)
    replay_feed_mod.logger.addHandler(handler)
    prev_level = replay_feed_mod.logger.level
    replay_feed_mod.logger.setLevel(logging.INFO)
    try:
        from live.cache import CoverageError
        with pytest.raises(CoverageError) as exc_info:
            feed.start()
    finally:
        replay_feed_mod.logger.removeHandler(handler)
        replay_feed_mod.logger.setLevel(prev_level)

    plan_lines = [
        r.getMessage() for r in captured
        if "coverage plan" in r.getMessage()
    ]
    assert plan_lines, (
        f"no coverage plan log at replay call site; "
        f"got: {[r.getMessage() for r in captured]}"
    )
    miss_line = next((l for l in plan_lines if "MISS" in l), None)
    assert miss_line is not None, f"expected MISS line, got: {plan_lines}"
    assert "seed-imbalance.parquet" in miss_line, miss_line
    assert "missing 1 required symbol(s)" in miss_line, miss_line
    assert "CRWV" in miss_line, miss_line
    assert "shortfall=1" in miss_line, miss_line

    # The CoverageError message itself must also be self-diagnostic so an
    # operator staring at a bare traceback still knows what went wrong.
    err = str(exc_info.value)
    assert "seed-imbalance.parquet" in err, err
    assert "shortfall=1" in err, err


def test_replay_orchestrator_allow_fetch_wired_to_replayfeed(monkeypatch, tmp_path):
    """The --allow-fetch flag on daily_orchestrator plumbs through to
    ReplayFeed's allow_fetch attribute."""
    universe = config.load_active_universe()
    monkeypatch.setenv("DATABENTO_API_KEY", "dummy-key")
    _patch_artifacts(monkeypatch, universe)
    _patch_target_dirs(monkeypatch, tmp_path)

    captured: dict = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        return _make_stub_replay_feed(kwargs["replay_date"], universe)

    monkeypatch.setattr(daily_orchestrator, "ReplayFeed", _capture)

    rc = daily_orchestrator.main(_base_argv("2026-05-15") + ["--allow-fetch", "--dry-run"])
    assert rc == 0
    assert captured.get("allow_fetch") is True


def test_replay_orchestrator_default_allow_fetch_false(monkeypatch, tmp_path):
    universe = config.load_active_universe()
    monkeypatch.setenv("DATABENTO_API_KEY", "dummy-key")
    _patch_artifacts(monkeypatch, universe)
    _patch_target_dirs(monkeypatch, tmp_path)

    captured: dict = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        return _make_stub_replay_feed(kwargs["replay_date"], universe)

    monkeypatch.setattr(daily_orchestrator, "ReplayFeed", _capture)

    rc = daily_orchestrator.main(_base_argv("2026-05-15") + ["--dry-run"])
    assert rc == 0
    assert captured.get("allow_fetch") is False, \
        "orchestrator must default --allow-fetch to False"


def test_replay_feed_start_failure_exits_cleanly(monkeypatch, tmp_path):
    """F3: an exception out of ReplayFeed.start() (e.g. cold-cache weekend
    date, adapter auth error) must return the feed-abort rc, not blow up
    with a traceback.
    """
    universe = config.load_active_universe()
    monkeypatch.setenv("DATABENTO_API_KEY", "dummy-key")
    _patch_artifacts(monkeypatch, universe)
    _patch_target_dirs(monkeypatch, tmp_path)

    class _ExplodingFeed:
        def start(self):
            raise RuntimeError("simulated cold-cache Databento failure")

        def stop(self, timeout: float = 0.0):
            pass

    monkeypatch.setattr(daily_orchestrator, "ReplayFeed", lambda **kw: _ExplodingFeed())

    def _blowup(_args):
        raise AssertionError("broker should never be contacted on feed failure")

    monkeypatch.setattr(daily_orchestrator, "_connect_broker", _blowup)

    rc = daily_orchestrator.main(_base_argv("2026-05-15") + ["--replay-submit"])
    assert rc == 2, f"expected rc=2 (feed abort), got {rc}"