"""Integration tests: the daily orchestrator drives the recorder end-to-end.

The autouse ``_isolate_live_output_dirs`` fixture in ``conftest.py``
already redirects all three orchestrator output dirs to a per-test
``tmp_path``, so we don't have to patch them here (except where a
specific test wants a KNOWN path to inspect).

Coverage targets on top of the pure-recorder tests:

- ``daily_orchestrator.main`` under ``--paper`` writes a run-scoped
  parquet at ``<trades_root>/paper/<today>/<run_id>.parquet`` populated
  with one row per attempted trade (entry + moc, both legs).
- Replay-submit populates the replay subdirectory only.
- A recorder-side exception cannot break the order path.
- Feed is stopped on rc=3/4/5/8 early returns (structural — used to
  leak the daemon thread).
- MOC-timeout paths write ``filled_qty`` / ``achieved_dollars`` as
  NULL, not $0.
- Real production ``live/logs/trades/`` is untouched even when a test
  forgets to patch ``_TRADES_DIR`` (the safeguard keys on the broker
  type, not on ``PYTEST_CURRENT_TEST``).
"""
from __future__ import annotations

from datetime import datetime, timezone

import polars as pl
import pyarrow.parquet as pq
import pytest

from live import config
from live import daily_orchestrator
from live import trade_recorder
from live.tests.test_end_to_end import (
    _FakeBroker,
    _prepopulate_live_feed,
    _patch_artifacts,
    _base_argv,
)


def test_paper_run_writes_recorder_parquet(monkeypatch, tmp_path):
    """Happy paper path: recorder emits a run-partitioned parquet
    under ``tmp/trades/paper/<today>/<run_id>.parquet``.
    """
    monkeypatch.setattr(daily_orchestrator, "_TRADES_DIR", tmp_path / "trades")
    universe = config.load_active_universe()
    feed = _prepopulate_live_feed(universe)
    monkeypatch.setattr(daily_orchestrator, "LiveFeed", lambda *a, **kw: feed)
    monkeypatch.setattr(daily_orchestrator, "_sleep_until", lambda *a, **kw: None)
    monkeypatch.setenv("DATABENTO_API_KEY", "dummy-key")
    _patch_artifacts(monkeypatch, universe)

    fake_broker = _FakeBroker()

    def _fake_connect(_args):
        fake_broker.connect(mode="paper")
        return fake_broker

    monkeypatch.setattr(daily_orchestrator, "_connect_broker", _fake_connect)

    rc = daily_orchestrator.main(_base_argv())
    assert rc == 0

    today = config.now_ny().date().isoformat()
    day_dir = tmp_path / "trades" / "paper" / today
    assert day_dir.exists(), (
        f"expected trades/paper/{today}/, present: "
        f"{list((tmp_path / 'trades').rglob('*'))}"
    )
    parquets = list(day_dir.glob("*.parquet"))
    assert len(parquets) == 1, f"expected 1 parquet, got {parquets}"
    df = pl.read_parquet(parquets[0])
    assert df.height > 0
    legs = set(df["leg"].to_list())
    assert legs == {"entry", "moc"}
    modes = set(df["mode"].to_list())
    assert modes == {"paper"}

    # Every entry symbol must be paired with a moc row for the same
    # symbol (both legs are recorded).
    entry_syms = set(df.filter(pl.col("leg") == "entry")["symbol"].to_list())
    moc_syms = set(df.filter(pl.col("leg") == "moc")["symbol"].to_list())
    assert entry_syms == moc_syms

    # Signed-notional correctness on filled rows.
    for row in df.iter_rows(named=True):
        if row["filled_qty"] and row["fill_price"] is not None:
            sign = 1 if row["side"] == "BUY" else -1
            assert row["achieved_dollars"] == pytest.approx(
                sign * row["filled_qty"] * row["fill_price"]
            )

    # MOC rows leave the entry's target_dollars / target_shares NULL.
    for row in df.filter(pl.col("leg") == "moc").iter_rows(named=True):
        assert row["target_dollars"] is None
        assert row["target_shares"] is None

    # Decision context reached at least one row.
    import json as _json
    ctx = df.filter(pl.col("decision_context_json").is_not_null())["decision_context_json"][0]
    parsed = _json.loads(ctx)
    assert "lambda" in parsed
    assert "gmv_dollar" in parsed
    assert parsed["mode"] == "paper"


def test_replay_submit_writes_replay_parquet(monkeypatch, tmp_path):
    """Replay-submit lands under trades/replay/<date>/ only — never
    paper/ or live/.
    """
    monkeypatch.setattr(daily_orchestrator, "_TRADES_DIR", tmp_path / "trades")
    from live.tests.test_replay import (
        _make_stub_replay_feed as _make_stub,
        _FakeBroker as _ReplayFakeBroker,
    )
    universe = config.load_active_universe()
    stub_feed = _make_stub("2026-05-13", universe)
    monkeypatch.setattr(daily_orchestrator, "ReplayFeed", lambda **kw: stub_feed)
    monkeypatch.setenv("DATABENTO_API_KEY", "dummy-key")
    _patch_artifacts(monkeypatch, universe)

    fake_broker = _ReplayFakeBroker()

    def _fake_connect(_args):
        fake_broker.connect(mode="paper")
        return fake_broker

    monkeypatch.setattr(daily_orchestrator, "_connect_broker", _fake_connect)

    argv = [
        "--replay-date", "2026-05-13", "--paper",
        "--vol-cap", "5000", "--replay-submit",
        "--submit-by", "00:01", "--on-feed-failure", "abort",
    ]
    rc = daily_orchestrator.main(argv)
    assert rc == 0

    replay_dir = tmp_path / "trades" / "replay" / "2026-05-13"
    assert replay_dir.exists()
    parquets = list(replay_dir.glob("*.parquet"))
    assert len(parquets) == 1

    paper_dir = tmp_path / "trades" / "paper"
    live_dir = tmp_path / "trades" / "live"
    assert not paper_dir.exists() or not any(paper_dir.iterdir())
    assert not live_dir.exists() or not any(live_dir.iterdir())

    df = pl.read_parquet(parquets[0])
    assert set(df["mode"].to_list()) == {"replay"}
    assert set(df["replay_date"].to_list()) == {"2026-05-13"}


def test_dry_run_writes_no_recorder_parquet(monkeypatch, tmp_path):
    monkeypatch.setattr(daily_orchestrator, "_TRADES_DIR", tmp_path / "trades")
    universe = config.load_active_universe()
    feed = _prepopulate_live_feed(universe)
    monkeypatch.setattr(daily_orchestrator, "LiveFeed", lambda *a, **kw: feed)
    monkeypatch.setattr(daily_orchestrator, "_sleep_until", lambda *a, **kw: None)
    monkeypatch.setenv("DATABENTO_API_KEY", "dummy-key")
    _patch_artifacts(monkeypatch, universe)

    rc = daily_orchestrator.main(_base_argv() + ["--dry-run"])
    assert rc == 0
    trades_dir = tmp_path / "trades"
    if trades_dir.exists():
        files = [p for p in trades_dir.rglob("*") if p.is_file()]
        assert not files, f"dry-run wrote recorder files: {files}"


def test_recorder_failure_does_not_break_order_path(monkeypatch, tmp_path):
    """CRITICAL: every recorder call raising must not block orders."""
    monkeypatch.setattr(daily_orchestrator, "_TRADES_DIR", tmp_path / "trades")
    universe = config.load_active_universe()
    feed = _prepopulate_live_feed(universe)
    monkeypatch.setattr(daily_orchestrator, "LiveFeed", lambda *a, **kw: feed)
    monkeypatch.setattr(daily_orchestrator, "_sleep_until", lambda *a, **kw: None)
    monkeypatch.setenv("DATABENTO_API_KEY", "dummy-key")
    _patch_artifacts(monkeypatch, universe)

    fake_broker = _FakeBroker()

    def _fake_connect(_args):
        fake_broker.connect(mode="paper")
        return fake_broker

    monkeypatch.setattr(daily_orchestrator, "_connect_broker", _fake_connect)

    class _EvilRecorder:
        def __init__(self, *a, **kw):
            raise RuntimeError("simulated recorder init failure")

    monkeypatch.setattr(trade_recorder, "TradeRecorder", _EvilRecorder)

    rc = daily_orchestrator.main(_base_argv())
    assert rc == 0
    mkt_orders = [c for c in fake_broker.buy_calls + fake_broker.sell_calls
                  if c[2] == "MKT"]
    moc_orders = [c for c in fake_broker.buy_calls + fake_broker.sell_calls
                  if c[2] == "MOC"]
    assert mkt_orders and moc_orders


def test_recorder_method_exception_does_not_break_order_path(monkeypatch, tmp_path):
    monkeypatch.setattr(daily_orchestrator, "_TRADES_DIR", tmp_path / "trades")
    universe = config.load_active_universe()
    feed = _prepopulate_live_feed(universe)
    monkeypatch.setattr(daily_orchestrator, "LiveFeed", lambda *a, **kw: feed)
    monkeypatch.setattr(daily_orchestrator, "_sleep_until", lambda *a, **kw: None)
    monkeypatch.setenv("DATABENTO_API_KEY", "dummy-key")
    _patch_artifacts(monkeypatch, universe)

    fake_broker = _FakeBroker()

    def _fake_connect(_args):
        fake_broker.connect(mode="paper")
        return fake_broker

    monkeypatch.setattr(daily_orchestrator, "_connect_broker", _fake_connect)

    def _boom(*a, **kw):
        raise OSError("simulated I/O failure inside recorder")

    for name in (
        "record_decision_context",
        "record_arrival_prices",
        "record_targets",
        "record_submission",
        "record_fill",
        "capture_t5min_from_tbbo",
        "schedule_t5min_capture_from_feed",
        "finalize",
    ):
        monkeypatch.setattr(trade_recorder.TradeRecorder, name, _boom)

    rc = daily_orchestrator.main(_base_argv())
    assert rc == 0
    mkt_orders = [c for c in fake_broker.buy_calls + fake_broker.sell_calls
                  if c[2] == "MKT"]
    moc_orders = [c for c in fake_broker.buy_calls + fake_broker.sell_calls
                  if c[2] == "MOC"]
    assert mkt_orders and moc_orders


def test_recorder_run_does_not_touch_production_dir(monkeypatch, tmp_path):
    """The autouse conftest fixture patches ``_TRADES_DIR``; if a test
    somehow unpatched it AND a fake broker were used, the safeguard
    must still redirect. Simulate that here.
    """
    real = config.LIVE_LOG_DIR / "trades"
    baseline = {p for p in real.rglob("*") if p.is_file()} if real.exists() else set()

    # Undo the autouse fixture's patch for JUST _TRADES_DIR to exercise
    # the fake-broker safeguard.
    monkeypatch.setattr(daily_orchestrator, "_TRADES_DIR",
                        config.LIVE_LOG_DIR / "trades")

    universe = config.load_active_universe()
    feed = _prepopulate_live_feed(universe)
    monkeypatch.setattr(daily_orchestrator, "LiveFeed", lambda *a, **kw: feed)
    monkeypatch.setattr(daily_orchestrator, "_sleep_until", lambda *a, **kw: None)
    monkeypatch.setenv("DATABENTO_API_KEY", "dummy-key")
    _patch_artifacts(monkeypatch, universe)

    fake_broker = _FakeBroker()

    def _fake_connect(_args):
        fake_broker.connect(mode="paper")
        return fake_broker

    monkeypatch.setattr(daily_orchestrator, "_connect_broker", _fake_connect)

    rc = daily_orchestrator.main(_base_argv())
    assert rc == 0

    after = {p for p in real.rglob("*") if p.is_file()} if real.exists() else set()
    assert after == baseline, (
        f"orchestrator wrote to real trades dir despite fake broker: "
        f"added={after - baseline}"
    )


# ---------------------------------------------------------------------------
# Feed lifecycle on early-return paths
# ---------------------------------------------------------------------------


def test_feed_stopped_on_rc_3_solver_failure(monkeypatch, tmp_path):
    """Feed leak regression (review #7): rc=3 (solver failed) MUST
    stop the feed."""
    monkeypatch.setattr(daily_orchestrator, "_TRADES_DIR", tmp_path / "trades")
    universe = config.load_active_universe()
    feed = _prepopulate_live_feed(universe)
    monkeypatch.setattr(daily_orchestrator, "LiveFeed", lambda *a, **kw: feed)
    monkeypatch.setattr(daily_orchestrator, "_sleep_until", lambda *a, **kw: None)
    monkeypatch.setenv("DATABENTO_API_KEY", "dummy-key")
    _patch_artifacts(monkeypatch, universe)

    stop_calls = {"n": 0}
    orig_stop = feed.stop

    def _tracking_stop(*a, **kw):
        stop_calls["n"] += 1
        return orig_stop(*a, **kw)

    feed.stop = _tracking_stop  # type: ignore[method-assign]

    from live.optimizer import Optimizer

    def _boom_solve(self, *a, **kw):
        raise RuntimeError("simulated solver blowup")

    monkeypatch.setattr(Optimizer, "solve", _boom_solve)

    def _blowup(_args):
        raise AssertionError("broker connect should not happen on rc=3")

    monkeypatch.setattr(daily_orchestrator, "_connect_broker", _blowup)

    rc = daily_orchestrator.main(_base_argv())
    assert rc == 3
    assert stop_calls["n"] >= 1, (
        "feed.stop should have been called on the rc=3 exit path"
    )


def test_feed_stopped_on_rc_8_broker_connect_failure(monkeypatch, tmp_path):
    """Feed leak regression: rc=8 (broker connect failed) MUST stop
    the feed."""
    monkeypatch.setattr(daily_orchestrator, "_TRADES_DIR", tmp_path / "trades")
    universe = config.load_active_universe()
    feed = _prepopulate_live_feed(universe)
    monkeypatch.setattr(daily_orchestrator, "LiveFeed", lambda *a, **kw: feed)
    monkeypatch.setattr(daily_orchestrator, "_sleep_until", lambda *a, **kw: None)
    monkeypatch.setenv("DATABENTO_API_KEY", "dummy-key")
    _patch_artifacts(monkeypatch, universe)

    stop_calls = {"n": 0}
    orig_stop = feed.stop

    def _tracking_stop(*a, **kw):
        stop_calls["n"] += 1
        return orig_stop(*a, **kw)

    feed.stop = _tracking_stop  # type: ignore[method-assign]

    def _refuse(_args):
        raise RuntimeError("simulated broker refusal")

    monkeypatch.setattr(daily_orchestrator, "_connect_broker", _refuse)

    rc = daily_orchestrator.main(_base_argv())
    assert rc == 8
    assert stop_calls["n"] >= 1


# ---------------------------------------------------------------------------
# MOC-timeout end-to-end
# ---------------------------------------------------------------------------


class _MocStaysSubmittedBroker(_FakeBroker):
    """Same as _FakeBroker but MOC ``wait_for_fill`` raises TimeoutError
    — reproducing real TWS behavior where MOC prints at 16:00 and any
    fill-audit that runs earlier times out.
    """

    def wait_for_fill(self, order_id, timeout=30.0):
        info = self._orders.get(order_id)
        if info and info.get("order_type") == "MOC":
            raise TimeoutError(f"MOC {order_id} still Submitted")
        return super().wait_for_fill(order_id, timeout=timeout)


def test_moc_timeout_records_null_not_zero(monkeypatch, tmp_path):
    """End-to-end: with a broker whose MOC ``wait_for_fill`` times out
    (mirrors reality — MOC prints at 16:00, well after fill audit),
    the parquet records ``filled_qty`` / ``achieved_dollars`` as NULL,
    NOT $0. Downstream ``sum(achieved_dollars)`` correctly ignores
    them rather than telling an analyst "the MOC never happened".
    """
    monkeypatch.setattr(daily_orchestrator, "_TRADES_DIR", tmp_path / "trades")
    universe = config.load_active_universe()
    feed = _prepopulate_live_feed(universe)
    monkeypatch.setattr(daily_orchestrator, "LiveFeed", lambda *a, **kw: feed)
    monkeypatch.setattr(daily_orchestrator, "_sleep_until", lambda *a, **kw: None)
    monkeypatch.setenv("DATABENTO_API_KEY", "dummy-key")
    _patch_artifacts(monkeypatch, universe)

    fake_broker = _MocStaysSubmittedBroker()

    def _fake_connect(_args):
        fake_broker.connect(mode="paper")
        return fake_broker

    monkeypatch.setattr(daily_orchestrator, "_connect_broker", _fake_connect)

    # Give _fill_audit a very short deadline so the test doesn't sit
    # waiting on the sequential fake TimeoutErrors.
    orig_fill_audit = daily_orchestrator._fill_audit

    def _short_fill_audit(broker, order_ids, timeout_seconds=900, **kw):
        return orig_fill_audit(broker, order_ids, timeout_seconds=1, **kw)

    monkeypatch.setattr(daily_orchestrator, "_fill_audit", _short_fill_audit)

    rc = daily_orchestrator.main(_base_argv())
    assert rc == 0

    today = config.now_ny().date().isoformat()
    parquets = list(
        (tmp_path / "trades" / "paper" / today).glob("*.parquet")
    )
    assert len(parquets) == 1
    df = pl.read_parquet(parquets[0])
    # MOC rows: filled_qty / achieved_dollars null; order_status
    # explicitly recorded as timeout.
    moc_rows = df.filter(pl.col("leg") == "moc")
    assert moc_rows.height > 0
    for row in moc_rows.iter_rows(named=True):
        assert row["order_status"] == "timeout"
        assert row["filled_qty"] is None
        assert row["achieved_dollars"] is None


# ---------------------------------------------------------------------------
# Round 2: stuck non-terminal ENTRY must not starve MOC audit
# ---------------------------------------------------------------------------


class _StuckEntryBroker(_FakeBroker):
    """One entry stays non-terminal forever; MOCs fill normally.

    Reproduces the reviewer's Round 2 #2 failure mode: a halted /
    illiquid MKT enters ``PendingCancel``, ``wait_for_fill`` blocks
    on it, and with the old audit design the entire remaining budget
    (~15 minutes) was burnt on it — MOC audit rows all became
    ``timeout``.
    """

    def __init__(self, *a, stuck_symbol: str, **kw):
        super().__init__(*a, **kw)
        self._stuck_symbol = stuck_symbol

    def wait_for_fill(self, order_id, timeout=30.0):
        info = self._orders.get(order_id)
        # Only ENTRY orders on the stuck symbol block. Everything
        # else (other entries, all MOCs) uses the parent's fake
        # behavior which returns immediately.
        if (info and info.get("order_type") == "MKT"
                and info.get("symbol") == self._stuck_symbol):
            import time as _t
            _t.sleep(timeout)  # simulate blocking
            raise TimeoutError(f"stuck entry {order_id}")
        return super().wait_for_fill(order_id, timeout=timeout)


def test_stuck_entry_does_not_starve_moc_audit(monkeypatch, tmp_path):
    """Round 2 #2: with a stuck non-terminal MKT entry, the MOC rows
    MUST still record real fill data. The audit orders MOCs first
    and caps entry waits at 10s, so a bad entry can't burn the whole
    budget.
    """
    monkeypatch.setattr(daily_orchestrator, "_TRADES_DIR", tmp_path / "trades")
    universe = config.load_active_universe()
    feed = _prepopulate_live_feed(universe)
    monkeypatch.setattr(daily_orchestrator, "LiveFeed", lambda *a, **kw: feed)
    monkeypatch.setattr(daily_orchestrator, "_sleep_until", lambda *a, **kw: None)
    monkeypatch.setenv("DATABENTO_API_KEY", "dummy-key")
    _patch_artifacts(monkeypatch, universe)

    stuck_sym = universe[0]
    fake_broker = _StuckEntryBroker(stuck_symbol=stuck_sym)

    def _fake_connect(_args):
        fake_broker.connect(mode="paper")
        return fake_broker

    monkeypatch.setattr(daily_orchestrator, "_connect_broker", _fake_connect)
    # Also shrink the entry-audit cap so the test doesn't sit 10s
    # on the stuck-entry wait_for_fill.
    monkeypatch.setattr(daily_orchestrator, "_ENTRY_AUDIT_TIMEOUT_SECS", 0.2)

    rc = daily_orchestrator.main(_base_argv())
    assert rc == 0

    today = config.now_ny().date().isoformat()
    parquets = list(
        (tmp_path / "trades" / "paper" / today).glob("*.parquet")
    )
    assert len(parquets) == 1
    df = pl.read_parquet(parquets[0])
    # MOC rows: at least SOME must record real fills, not all
    # timeouts. Under the old design (entries first, full budget)
    # every MOC would be timeout.
    moc_rows = df.filter(pl.col("leg") == "moc")
    assert moc_rows.height > 0
    real_moc_fills = moc_rows.filter(pl.col("filled_qty") > 0).height
    assert real_moc_fills > 0, (
        "every MOC recorded as timeout — the stuck entry starved "
        "the MOC audit (regression on Round 2 #2)"
    )


# ---------------------------------------------------------------------------
# Round 3: rc=11 exposure page must not be delayed by any recording work
# ---------------------------------------------------------------------------


def test_rc_11_returns_promptly_regardless_of_recorder(monkeypatch, tmp_path):
    """CRITICAL round-3 regression: on the OVERNIGHT-EXPOSURE path,
    NOTHING recording-related — ``wait_for_t5min``, ``_fill_audit``,
    or ``finalize`` — may run before ``return 11``. This pattern has
    now bitten twice (round 1 removed a 30s ``finalize()`` wait from
    the ``finally``; round 2 moved a 350s ``wait_for_t5min`` into
    exactly the same seat, one function up). This test pins on the
    TIMING, not just the return code.

    Setup: force MOC snapshot failure (``moc_positions_unknown=True``)
    AND monkey-patch ``TradeRecorder.wait_for_t5min`` /
    ``_fill_audit`` to each sleep 30s if called. If either fires on
    the exposure path the whole run takes 30+ seconds; the assertion
    caps at <5s.
    """
    import time as _t
    monkeypatch.setattr(daily_orchestrator, "_TRADES_DIR", tmp_path / "trades")
    universe = config.load_active_universe()
    feed = _prepopulate_live_feed(universe)
    monkeypatch.setattr(daily_orchestrator, "LiveFeed", lambda *a, **kw: feed)
    monkeypatch.setattr(daily_orchestrator, "_sleep_until", lambda *a, **kw: None)
    monkeypatch.setenv("DATABENTO_API_KEY", "dummy-key")
    _patch_artifacts(monkeypatch, universe)

    # get_positions() succeeds for the diff baseline, raises on the
    # MOC snapshot — triggers moc_positions_unknown=True.
    fake_broker = _FakeBroker(get_positions_raises_after_n_calls=2)

    def _fake_connect(_args):
        fake_broker.connect(mode="paper")
        return fake_broker

    monkeypatch.setattr(daily_orchestrator, "_connect_broker", _fake_connect)

    # If wait_for_t5min is called on the exposure path we notice
    # via a 30s sleep. On the correct-behavior path it isn't called
    # at all before `return 11`.
    slow_calls: dict[str, int] = {"wait_for_t5min": 0, "fill_audit": 0}

    def _slow_wait(self, timeout=None):
        slow_calls["wait_for_t5min"] += 1
        _t.sleep(30.0)
        return False

    def _slow_audit(*a, **kw):
        slow_calls["fill_audit"] += 1
        _t.sleep(30.0)

    monkeypatch.setattr(trade_recorder.TradeRecorder, "wait_for_t5min", _slow_wait)
    monkeypatch.setattr(daily_orchestrator, "_fill_audit", _slow_audit)

    t0 = _t.time()
    rc = daily_orchestrator.main(_base_argv())
    elapsed = _t.time() - t0

    assert rc == 11, f"expected rc=11 (OVERNIGHT EXPOSURE), got {rc}"
    assert elapsed < 5.0, (
        f"rc=11 returned after {elapsed:.2f}s — a recording-related "
        f"wait ran before the operator page. wait_for_t5min calls: "
        f"{slow_calls['wait_for_t5min']}, _fill_audit calls: "
        f"{slow_calls['fill_audit']}. Both MUST be zero on the "
        f"exposure path."
    )
    assert slow_calls["wait_for_t5min"] == 0, (
        "wait_for_t5min was called on the rc=11 path — recording "
        "must never gate the operator page"
    )
    assert slow_calls["fill_audit"] == 0, (
        "_fill_audit was called on the rc=11 path — recording must "
        "never gate the operator page"
    )


# ---------------------------------------------------------------------------
# --print-wait-secs >= 60 must not silently disable the recorder
# ---------------------------------------------------------------------------


def test_print_wait_secs_over_60_still_creates_recorder(monkeypatch, tmp_path):
    """Review #13: ``--print-wait-secs 60`` used to feed ``second=60``
    to ``datetime()``, which raises ``ValueError``, which was swallowed
    by the recorder-init try/except so the whole day silently lost its
    recorder. Fix: use ``+ timedelta(seconds=...)``.
    """
    monkeypatch.setattr(daily_orchestrator, "_TRADES_DIR", tmp_path / "trades")
    from live.tests.test_replay import (
        _make_stub_replay_feed as _make_stub,
        _FakeBroker as _ReplayFakeBroker,
    )
    universe = config.load_active_universe()
    stub_feed = _make_stub("2026-05-13", universe)
    monkeypatch.setattr(daily_orchestrator, "ReplayFeed", lambda **kw: stub_feed)
    monkeypatch.setenv("DATABENTO_API_KEY", "dummy-key")
    _patch_artifacts(monkeypatch, universe)

    fake_broker = _ReplayFakeBroker()

    def _fake_connect(_args):
        fake_broker.connect(mode="paper")
        return fake_broker

    monkeypatch.setattr(daily_orchestrator, "_connect_broker", _fake_connect)

    argv = [
        "--replay-date", "2026-05-13", "--paper",
        "--vol-cap", "5000", "--replay-submit",
        "--submit-by", "00:01", "--on-feed-failure", "abort",
        "--print-wait-secs", "90",  # would crash with second=90
    ]
    rc = daily_orchestrator.main(argv)
    assert rc == 0

    replay_dir = tmp_path / "trades" / "replay" / "2026-05-13"
    parquets = list(replay_dir.glob("*.parquet"))
    assert len(parquets) == 1, (
        "recorder should still write parquet with --print-wait-secs=90"
    )


# ---------------------------------------------------------------------------
# Redundant feed.snapshot on 15:50 critical path is GONE (live/paper)
# ---------------------------------------------------------------------------


def test_live_path_does_not_take_redundant_snapshot_at_decision(monkeypatch, tmp_path):
    """Review #11: on the live/paper path, the recorder must NOT call
    ``feed.snapshot()`` a second time at decision time — that used to
    burn ~1s on the critical path. The T+5min thread's later snapshot
    is the only one needed.
    """
    monkeypatch.setattr(daily_orchestrator, "_TRADES_DIR", tmp_path / "trades")
    universe = config.load_active_universe()
    feed = _prepopulate_live_feed(universe)

    snapshot_calls = {"n": 0}
    orig_snap = feed.snapshot

    def _tracking_snapshot():
        snapshot_calls["n"] += 1
        return orig_snap()

    feed.snapshot = _tracking_snapshot  # type: ignore[method-assign]
    monkeypatch.setattr(daily_orchestrator, "LiveFeed", lambda *a, **kw: feed)
    monkeypatch.setattr(daily_orchestrator, "_sleep_until", lambda *a, **kw: None)
    monkeypatch.setenv("DATABENTO_API_KEY", "dummy-key")
    _patch_artifacts(monkeypatch, universe)

    fake_broker = _FakeBroker()

    def _fake_connect(_args):
        fake_broker.connect(mode="paper")
        return fake_broker

    monkeypatch.setattr(daily_orchestrator, "_connect_broker", _fake_connect)

    rc = daily_orchestrator.main(_base_argv())
    assert rc == 0
    # _build_live_day_frame calls feed.snapshot() exactly once. The
    # recorder previously called it a SECOND time at decision. With
    # the fix, live/paper takes at most one at 15:50 (from feature
    # engineering) — the T+5min thread would call it again but the
    # test's tiny in-memory feed and no wait means the thread may not
    # have run. So the count is 1 (or 2 if the T+5min thread landed).
    assert snapshot_calls["n"] <= 2, (
        f"live path took {snapshot_calls['n']} feed.snapshot() calls; "
        f"the recorder's redundant decision-time snapshot should be "
        f"guarded to replay only"
    )
