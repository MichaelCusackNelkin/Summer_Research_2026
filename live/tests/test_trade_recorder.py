"""Tests for the post-trade recorder.

Coverage targets (per the feature brief AND the two code reviews):

- schema / type stability across empty and full rows
- VWAP computation across partial fills
- signed-notional correctness for sell rows
- ``achieved_dollars`` semantics: signed on real fill, ``0.0`` on
  terminal cancel, NULL on timeout / unknown (a downstream sum
  correctly ignores unknowns rather than pretending they're $0)
- MOC rows do NOT inherit the entry's ``target_dollars`` /
  ``target_shares`` (they have no notional target — the MOC size is
  derived from the actual post-entry position, not the entry's target)
- ``t5min_ts_utc`` is the OBSERVED tick ts, not the intended target;
  ``t5min_staleness_secs`` is non-zero for illiquid names whose last
  tbbo tick was earlier than the target — this is where the
  square-root impact calibration lives, so silently pretending the
  price was fresh at 15:55 would bias impact toward zero decay
- extras win for ``fill_ts_utc`` (TWS's real ``execution.time`` MUST
  beat the wall-clock fallback — the recorder's primary time axis)
- ``_to_utc`` REJECTS naive datetimes (no silent naive==UTC coercion)
- ``rel_spread_at_decision`` is the FULL relative spread, not half
- ``--print-wait-secs >= 60`` no longer silently disables the recorder
- per-run partitioning: same-day re-run doesn't destroy the earlier
  parquet; each run writes its own ``<run_id>.parquet``
- paper / live / replay file separation (each mode -> own subdir)
- recorder exception cannot propagate into the order path
- structural test-isolation: without an explicit override, the
  recorder redirects to a scratch dir when the broker isn't a real
  ``ibkr_broker.Broker`` — so a notebook with a fake broker (not just
  a pytest run) is also blocked from appending fabricated fills to
  ``live/logs/trades/``
"""
from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pyarrow.parquet as pq
import pytest

from live import config
from live import daily_orchestrator
from live import trade_recorder


_UTC = timezone.utc


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _snapshot_real_trades_dir_before_tests() -> set[Path]:
    real = config.LIVE_LOG_DIR / "trades"
    if not real.exists():
        return set()
    return {p for p in real.rglob("*") if p.is_file()}


_REAL_TRADES_BASELINE = _snapshot_real_trades_dir_before_tests()


def _decision_ts() -> datetime:
    return datetime(2026, 5, 13, 15, 50, 10, tzinfo=timezone(timedelta(hours=-4)))


def _universe() -> list[str]:
    return ["AAA", "BBB", "CCC"]


def _base_arrivals() -> dict[str, dict]:
    dt = _decision_ts().astimezone(_UTC)
    return {
        "AAA": {"bid": 99.99, "ask": 100.01, "mid": 100.0, "rel_spread": 0.0002,
                "observed_ts_utc": dt},
        "BBB": {"bid": 49.99, "ask": 50.01, "mid": 50.0, "rel_spread": 0.0004,
                "observed_ts_utc": dt},
        "CCC": {"bid": 199.98, "ask": 200.02, "mid": 200.0, "rel_spread": 0.0002,
                "observed_ts_utc": dt},
    }


def _new_recorder(
    mode: str,
    base_dir: Path,
    replay_date: str | None = None,
    run_id: str = "test-run",
) -> trade_recorder.TradeRecorder:
    return trade_recorder.TradeRecorder(
        mode=mode,
        decision_ts_utc=_decision_ts(),
        universe=_universe(),
        trading_date="2026-05-13",
        base_dir=base_dir,
        replay_date=replay_date,
        universe_hash="abc123",
        run_id=run_id,
    )


# ---------------------------------------------------------------------------
# Schema / type stability
# ---------------------------------------------------------------------------


def test_schema_columns_present_and_typed(tmp_path):
    rec = _new_recorder("paper", tmp_path)
    rec.record_decision_context({"lambda": 5e-5, "gmv": 1e6})
    rec.record_arrival_prices(_base_arrivals())
    rec.record_targets({
        "AAA": {"target_dollars": 5000.0, "target_shares": 50},
    })
    rec.record_submission(
        leg="entry", symbol="AAA", side="BUY", requested_qty=50,
        order_type="MKT", order_id="oid-1",
        sent_ts_utc=_decision_ts() + timedelta(seconds=1),
        exchange="ISLAND", tif="DAY",
    )
    rec.record_fill(
        leg="entry", symbol="AAA", order_id="oid-1",
        filled_qty=50, fill_price=100.02,
        status="Filled",
        observed_ts_utc=_decision_ts() + timedelta(seconds=2),
    )
    path = rec.finalize(wait_for_t5min=None)
    assert path is not None
    table = pq.read_table(path)
    for name, dtype in zip(
        trade_recorder.TRADE_RECORD_SCHEMA.names,
        trade_recorder.TRADE_RECORD_SCHEMA.types,
    ):
        assert name in table.schema.names, f"missing column {name}"
        got = table.schema.field(name).type
        assert got == dtype, f"{name}: expected {dtype}, got {got}"
    assert trade_recorder.SCHEMA_VERSION == 3


def test_finalize_produces_no_file_when_empty(tmp_path):
    rec = _new_recorder("paper", tmp_path)
    result = rec.finalize(wait_for_t5min=None)
    assert result is None
    assert not (tmp_path / "paper" / "2026-05-13" / "test-run.parquet").exists()


def test_all_null_column_keeps_type(tmp_path):
    rec = _new_recorder("paper", tmp_path)
    rec.record_submission(
        leg="entry", symbol="AAA", side="BUY", requested_qty=1,
        order_type="MKT", order_id="oid-x",
    )
    rec.record_fill(
        leg="entry", symbol="AAA", order_id="oid-x",
        filled_qty=1, fill_price=100.0, status="Filled",
    )
    path = rec.finalize(wait_for_t5min=None)
    table = pq.read_table(path)
    for col in ("commission", "realized_pnl", "t5min_price",
                "t5min_staleness_secs", "arrival_staleness_secs"):
        assert table.column(col).null_count == table.num_rows, (
            f"{col} should be all null in a minimal row"
        )
        assert table.schema.field(col).type.equals(
            trade_recorder.TRADE_RECORD_SCHEMA.field(col).type
        )


# ---------------------------------------------------------------------------
# Naive-datetime rejection
# ---------------------------------------------------------------------------


def test_to_utc_rejects_naive_datetime():
    naive = datetime(2026, 5, 13, 15, 50, 10)
    with pytest.raises(ValueError):
        trade_recorder._to_utc(naive)


def test_recorder_init_rejects_naive_decision_ts(tmp_path):
    naive = datetime(2026, 5, 13, 15, 50, 10)
    with pytest.raises(ValueError):
        trade_recorder.TradeRecorder(
            mode="paper", decision_ts_utc=naive, universe=["AAA"],
            trading_date="2026-05-13", base_dir=tmp_path,
        )


# ---------------------------------------------------------------------------
# VWAP across partial fills / signed notional
# ---------------------------------------------------------------------------


def test_vwap_multiple_fills_uses_broker_avg_fill_price(tmp_path):
    rec = _new_recorder("paper", tmp_path)
    rec.record_submission(
        leg="entry", symbol="AAA", side="BUY", requested_qty=100,
        order_type="MKT", order_id="oid-1",
    )
    rec.record_fill(
        leg="entry", symbol="AAA", order_id="oid-1",
        filled_qty=50, fill_price=100.00, status="Filled",
    )
    rec.record_fill(
        leg="entry", symbol="AAA", order_id="oid-1",
        filled_qty=100, fill_price=100.05, status="Filled",
    )
    path = rec.finalize(wait_for_t5min=None)
    df = pl.read_parquet(path)
    row = df.row(0, named=True)
    assert row["filled_qty"] == 100
    assert row["fill_price"] == pytest.approx(100.05)
    assert row["signed_filled_qty"] == 100
    assert row["achieved_dollars"] == pytest.approx(10005.0)


def test_signed_notional_sell_is_negative(tmp_path):
    rec = _new_recorder("paper", tmp_path)
    rec.record_targets({
        "AAA": {"target_dollars": -8000.0, "target_shares": -80},
    })
    rec.record_submission(
        leg="entry", symbol="AAA", side="SELL", requested_qty=80,
        order_type="MKT", order_id="oid-s",
    )
    rec.record_fill(
        leg="entry", symbol="AAA", order_id="oid-s",
        filled_qty=80, fill_price=100.10, status="Filled",
    )
    path = rec.finalize(wait_for_t5min=None)
    df = pl.read_parquet(path)
    row = df.row(0, named=True)
    assert row["side"] == "SELL"
    assert row["order_qty"] == -80
    assert row["requested_qty"] == 80
    assert row["signed_filled_qty"] == -80
    assert row["fill_price"] == pytest.approx(100.10)
    assert row["achieved_dollars"] == pytest.approx(-8008.0)
    assert row["target_dollars"] == pytest.approx(-8000.0)
    assert row["target_shares"] == -80


# ---------------------------------------------------------------------------
# achieved_dollars null vs 0.0
# ---------------------------------------------------------------------------


def test_zero_fill_terminal_cancel_records_zero_notional(tmp_path):
    rec = _new_recorder("paper", tmp_path)
    rec.record_submission(
        leg="entry", symbol="AAA", side="BUY", requested_qty=10,
        order_type="MKT", order_id="oid-c",
    )
    rec.record_fill(
        leg="entry", symbol="AAA", order_id="oid-c",
        filled_qty=0, fill_price=None, status="Cancelled",
    )
    path = rec.finalize(wait_for_t5min=None)
    row = pl.read_parquet(path).row(0, named=True)
    assert row["filled_qty"] == 0
    assert row["signed_filled_qty"] == 0
    assert row["achieved_dollars"] == 0.0
    assert row["cancelled_qty"] == 10.0
    assert row["order_status"] == "Cancelled"


def test_timeout_leaves_achieved_dollars_null(tmp_path):
    """A timeout during fill audit MUST NOT be recorded as achieved=$0
    — that's the exit-leg-never-happened bug from review #2.
    """
    rec = _new_recorder("paper", tmp_path)
    rec.record_submission(
        leg="moc", symbol="AAA", side="SELL", requested_qty=50,
        order_type="MOC", order_id="oid-t",
    )
    rec.record_fill(
        leg="moc", symbol="AAA", order_id="oid-t",
        filled_qty=None, fill_price=None, status="timeout",
    )
    path = rec.finalize(wait_for_t5min=None)
    row = pl.read_parquet(path).row(0, named=True)
    assert row["filled_qty"] is None
    assert row["signed_filled_qty"] is None
    assert row["achieved_dollars"] is None
    assert row["order_status"] == "timeout"


# ---------------------------------------------------------------------------
# MOC rows must NOT inherit the entry's target
# ---------------------------------------------------------------------------


def test_moc_row_target_columns_are_null(tmp_path):
    """Review #3: recording the entry's target on the MOC row would
    tell an analyst the MOC achieved 100% of the wrong number.
    """
    rec = _new_recorder("paper", tmp_path)
    rec.record_targets({
        "AAA": {"target_dollars": -8000.0, "target_shares": -80},
    })
    rec.record_submission(
        leg="entry", symbol="AAA", side="SELL", requested_qty=80,
        order_type="MKT", order_id="oid-e",
    )
    rec.record_fill(
        leg="entry", symbol="AAA", order_id="oid-e",
        filled_qty=80, fill_price=100.0, status="Filled",
    )
    rec.record_submission(
        leg="moc", symbol="AAA", side="BUY", requested_qty=80,
        order_type="MOC", order_id="oid-m",
    )
    rec.record_fill(
        leg="moc", symbol="AAA", order_id="oid-m",
        filled_qty=80, fill_price=100.05, status="Filled",
    )
    path = rec.finalize(wait_for_t5min=None)
    df = pl.read_parquet(path)
    entry = df.filter(pl.col("leg") == "entry").row(0, named=True)
    moc = df.filter(pl.col("leg") == "moc").row(0, named=True)
    assert entry["target_dollars"] == pytest.approx(-8000.0)
    assert entry["target_shares"] == -80
    assert moc["target_dollars"] is None
    assert moc["target_shares"] is None
    assert moc["achieved_dollars"] == pytest.approx(80 * 100.05)


def test_moc_target_null_when_recorded_after_moc_row_exists(tmp_path):
    """Reverse ordering: record_targets fires AFTER the MOC row is
    materialised. The gate lives in record_targets AND in _row's
    lazy-apply — both must skip MOC rows.
    """
    rec = _new_recorder("paper", tmp_path)
    rec.record_submission(
        leg="moc", symbol="AAA", side="BUY", requested_qty=80,
        order_type="MOC", order_id="oid-m",
    )
    rec.record_targets({
        "AAA": {"target_dollars": -8000.0, "target_shares": -80},
    })
    path = rec.finalize(wait_for_t5min=None)
    moc = pl.read_parquet(path).filter(pl.col("leg") == "moc").row(0, named=True)
    assert moc["target_dollars"] is None
    assert moc["target_shares"] is None


# ---------------------------------------------------------------------------
# Fill-timestamp precedence: extras' execution.time must win
# ---------------------------------------------------------------------------


def test_fill_ts_utc_extras_beat_wall_clock(tmp_path):
    """Review #5: extras (real execution.time) MUST win over the
    wall-clock fallback."""
    rec = _new_recorder("paper", tmp_path)
    rec.record_submission(
        leg="entry", symbol="AAA", side="BUY", requested_qty=1,
        order_type="MKT", order_id="oid-1",
    )
    wall_clock_ts = datetime(2026, 5, 13, 19, 50, 30, tzinfo=_UTC)
    real_exec_ts = datetime(2026, 5, 13, 19, 50, 12, 500_000, tzinfo=_UTC)
    rec.record_fill(
        leg="entry", symbol="AAA", order_id="oid-1",
        filled_qty=1, fill_price=100.0, status="Filled",
        observed_ts_utc=wall_clock_ts,
        extras={"fill_ts_utc": real_exec_ts, "exec_id": "EX-1"},
    )
    path = rec.finalize(wait_for_t5min=None)
    row = pl.read_parquet(path).row(0, named=True)
    assert row["fill_ts_utc"] == real_exec_ts
    assert row["observed_ts_utc"] == wall_clock_ts
    assert row["exec_id"] == "EX-1"


def test_fill_ts_utc_defaults_to_observed_when_no_extras(tmp_path):
    rec = _new_recorder("paper", tmp_path)
    rec.record_submission(
        leg="entry", symbol="AAA", side="BUY", requested_qty=1,
        order_type="MKT", order_id="oid-1",
    )
    wall_clock_ts = datetime(2026, 5, 13, 19, 50, 30, tzinfo=_UTC)
    rec.record_fill(
        leg="entry", symbol="AAA", order_id="oid-1",
        filled_qty=1, fill_price=100.0, status="Filled",
        observed_ts_utc=wall_clock_ts,
    )
    row = pl.read_parquet(rec.finalize(wait_for_t5min=None)).row(0, named=True)
    assert row["fill_ts_utc"] == wall_clock_ts
    assert row["observed_ts_utc"] == wall_clock_ts
    # fill_ts_source names the origin of fill_ts_utc.
    assert row["fill_ts_source"] == "observed"


def test_fill_ts_source_is_execution_when_extras_present(tmp_path):
    """Round 3 LOW #2: an explicit ``fill_ts_source`` column tells
    the analyst whether ``fill_ts_utc`` is from TWS's
    ``execution.time`` ("execution") or the wall-clock fallback
    ("observed"). Without this the only signal is a brittle
    ``fill_ts_utc == observed_ts_utc`` self-comparison.
    """
    rec = _new_recorder("paper", tmp_path)
    rec.record_submission(
        leg="entry", symbol="AAA", side="BUY", requested_qty=1,
        order_type="MKT", order_id="oid-1",
    )
    wall_clock_ts = datetime(2026, 5, 13, 19, 50, 30, tzinfo=_UTC)
    real_exec_ts = datetime(2026, 5, 13, 19, 50, 12, 500_000, tzinfo=_UTC)
    rec.record_fill(
        leg="entry", symbol="AAA", order_id="oid-1",
        filled_qty=1, fill_price=100.0, status="Filled",
        observed_ts_utc=wall_clock_ts,
        extras={"fill_ts_utc": real_exec_ts},
    )
    row = pl.read_parquet(rec.finalize(wait_for_t5min=None)).row(0, named=True)
    assert row["fill_ts_utc"] == real_exec_ts
    assert row["fill_ts_source"] == "execution"


def test_fill_ts_source_null_on_no_fill_rows(tmp_path):
    rec = _new_recorder("paper", tmp_path)
    rec.record_submission(
        leg="entry", symbol="AAA", side="BUY", requested_qty=10,
        order_type="MKT", order_id="oid-c",
    )
    rec.record_fill(
        leg="entry", symbol="AAA", order_id="oid-c",
        filled_qty=0, fill_price=None, status="Cancelled",
    )
    row = pl.read_parquet(rec.finalize(wait_for_t5min=None)).row(0, named=True)
    assert row["fill_ts_utc"] is None
    assert row["fill_ts_source"] is None


# ---------------------------------------------------------------------------
# T+5min: staleness + no fabricated ts
# ---------------------------------------------------------------------------


def test_t5min_records_observed_ts_not_target(tmp_path):
    """Review #4: for a THIN name whose last tbbo tick was 15:50:02,
    ``t5min_price == arrival_price`` (by construction) but
    ``t5min_ts_utc`` must be the observed tick, NOT the intended
    target. ``t5min_staleness_secs`` is non-zero. Post-trade impact
    calibration filters on this — silently pretending the price was
    fresh at 15:55 would bias impact toward zero decay in the tail.
    """
    ny = timezone(timedelta(hours=-4))
    decision_utc = datetime(2026, 5, 13, 15, 50, 10, tzinfo=ny).astimezone(_UTC)
    tbbo = pl.DataFrame([{
        "ts_event": datetime(2026, 5, 13, 15, 50, 2, tzinfo=ny).astimezone(_UTC),
        "symbol": "AAA",
        "price": 100.0,
        "size": 100,
        "ask_px_00": 100.005,
        "bid_px_00": 99.995,
    }])
    rec = _new_recorder("replay", tmp_path, replay_date="2026-05-13")
    rec.record_arrival_prices({
        "AAA": {
            "bid": 99.995, "ask": 100.005, "mid": 100.0,
            "rel_spread": 0.0001,
            "observed_ts_utc": datetime(2026, 5, 13, 15, 50, 2, tzinfo=ny).astimezone(_UTC),
        },
    })
    rec.record_submission(
        leg="entry", symbol="AAA", side="BUY", requested_qty=10,
        order_type="MKT", order_id="oid-a",
    )
    rec.record_fill(
        leg="entry", symbol="AAA", order_id="oid-a",
        filled_qty=10, fill_price=100.0, status="Filled",
    )
    rec.capture_t5min_from_tbbo(tbbo, decision_utc)
    rec.finalize(wait_for_t5min=None)
    row = pl.read_parquet(rec.final_path).row(0, named=True)

    target_ts = decision_utc + timedelta(minutes=5)
    tick_ts = datetime(2026, 5, 13, 15, 50, 2, tzinfo=ny).astimezone(_UTC)

    assert row["t5min_price"] == pytest.approx(row["arrival_price"])
    assert row["t5min_ts_utc"] == tick_ts
    assert row["t5min_ts_utc"] != target_ts
    expected_stale = (target_ts - tick_ts).total_seconds()
    assert row["t5min_staleness_secs"] == pytest.approx(expected_stale)
    assert row["t5min_staleness_secs"] > 300


def test_t5min_fresh_tick_has_zero_staleness_and_moves(tmp_path):
    ny = timezone(timedelta(hours=-4))
    decision_utc = datetime(2026, 5, 13, 15, 50, 10, tzinfo=ny).astimezone(_UTC)
    target_ts = decision_utc + timedelta(minutes=5)
    tbbo = pl.DataFrame([{
        "ts_event": target_ts,
        "symbol": "AAA",
        "price": 100.5,
        "size": 100,
        "ask_px_00": 100.505,
        "bid_px_00": 100.495,
    }])
    rec = _new_recorder("replay", tmp_path, replay_date="2026-05-13")
    rec.record_arrival_prices({
        "AAA": {
            "bid": 99.995, "ask": 100.005, "mid": 100.0,
            "rel_spread": 0.0001,
            "observed_ts_utc": decision_utc,
        },
    })
    rec.record_submission(
        leg="entry", symbol="AAA", side="BUY", requested_qty=10,
        order_type="MKT", order_id="oid-a",
    )
    rec.record_fill(
        leg="entry", symbol="AAA", order_id="oid-a",
        filled_qty=10, fill_price=100.0, status="Filled",
    )
    rec.capture_t5min_from_tbbo(tbbo, decision_utc)
    rec.finalize(wait_for_t5min=None)
    row = pl.read_parquet(rec.final_path).row(0, named=True)
    assert row["t5min_price"] == pytest.approx(100.5)
    assert row["t5min_price"] != row["arrival_price"]
    assert row["t5min_staleness_secs"] == pytest.approx(0.0, abs=1e-6)


def test_t5min_background_thread_updates_rows(tmp_path):
    decision_utc = datetime.now(_UTC)
    tbbo_rows = []
    for sym in ("AAA", "BBB"):
        base = 100.0 if sym == "AAA" else 50.0
        tbbo_rows.append({
            "ts_event": decision_utc,
            "symbol": sym,
            "price": base,
            "size": 100,
            "ask_px_00": base + 0.05,
            "bid_px_00": base - 0.05,
        })
    tbbo = pl.DataFrame(tbbo_rows)

    class _FakeFeed:
        def snapshot(self):
            return {"tbbo": tbbo, "imbalance": pl.DataFrame()}
        def stop(self, timeout=None):
            pass

    rec = _new_recorder("paper", tmp_path)
    rec.record_submission(
        leg="entry", symbol="AAA", side="BUY", requested_qty=1,
        order_type="MKT", order_id="oid-aaa",
    )
    rec.record_fill(
        leg="entry", symbol="AAA", order_id="oid-aaa",
        filled_qty=1, fill_price=100.0, status="Filled",
    )
    rec.schedule_t5min_capture_from_feed(
        _FakeFeed(), decision_utc, offset=timedelta(milliseconds=50),
    )
    rec.finalize(wait_for_t5min=5.0)
    row = pl.read_parquet(rec.final_path).filter(pl.col("symbol") == "AAA").row(0, named=True)
    assert row["t5min_price"] == pytest.approx(100.0)


def test_t5min_thread_survives_snapshot_failure(tmp_path):
    class _EvilFeed:
        def snapshot(self):
            raise RuntimeError("feed exploded")
        def stop(self, timeout=None):
            pass

    rec = _new_recorder("paper", tmp_path)
    rec.record_submission(
        leg="entry", symbol="AAA", side="BUY", requested_qty=1,
        order_type="MKT", order_id="oid-1",
    )
    rec.record_fill(
        leg="entry", symbol="AAA", order_id="oid-1",
        filled_qty=1, fill_price=100.0, status="Filled",
    )
    rec.schedule_t5min_capture_from_feed(
        _EvilFeed(), datetime.now(_UTC), offset=timedelta(milliseconds=10),
    )
    rec.finalize(wait_for_t5min=2.0)
    df = pl.read_parquet(rec.final_path)
    assert df["t5min_price"].to_list() == [None]


def test_t5min_thread_after_finalize_is_a_noop(tmp_path):
    """Review #10: a late thread must not journal or update rows
    after finalize has run."""
    class _FakeFeed:
        def snapshot(self):
            return {"tbbo": pl.DataFrame([{
                "ts_event": datetime.now(_UTC),
                "symbol": "AAA",
                "price": 100.0, "size": 100,
                "ask_px_00": 100.01, "bid_px_00": 99.99,
            }])}
        def stop(self, timeout=None):
            pass

    rec = _new_recorder("paper", tmp_path)
    rec.record_submission(
        leg="entry", symbol="AAA", side="BUY", requested_qty=1,
        order_type="MKT", order_id="oid-1",
    )
    rec.record_fill(
        leg="entry", symbol="AAA", order_id="oid-1",
        filled_qty=1, fill_price=100.0, status="Filled",
    )
    rec.schedule_t5min_capture_from_feed(
        _FakeFeed(), datetime.now(_UTC), offset=timedelta(seconds=30),
    )
    rec.finalize(wait_for_t5min=None)
    time.sleep(0.1)
    assert not rec.journal_path.exists()


# ---------------------------------------------------------------------------
# Finalize does NOT block by default
# ---------------------------------------------------------------------------


def test_finalize_default_does_not_block_on_t5min_thread(tmp_path):
    """Review #1: ``finalize()`` (default) MUST NOT wait for the
    background thread. On the rc=11 OVERNIGHT-EXPOSURE path a 30s
    delay to the operator page is unacceptable.
    """
    class _SleepyFeed:
        def snapshot(self):
            return {"tbbo": pl.DataFrame()}
        def stop(self, timeout=None):
            pass

    rec = _new_recorder("paper", tmp_path)
    rec.record_submission(
        leg="entry", symbol="AAA", side="BUY", requested_qty=1,
        order_type="MKT", order_id="oid-1",
    )
    rec.record_fill(
        leg="entry", symbol="AAA", order_id="oid-1",
        filled_qty=1, fill_price=100.0, status="Filled",
    )
    rec.schedule_t5min_capture_from_feed(
        _SleepyFeed(), datetime.now(_UTC), offset=timedelta(seconds=60),
    )
    t0 = time.time()
    rec.finalize()
    elapsed = time.time() - t0
    assert elapsed < 2.0, (
        f"finalize blocked for {elapsed:.2f}s — must not wait for T+5min "
        f"by default (would delay a rc=11 operator page in prod)"
    )


# ---------------------------------------------------------------------------
# Run-scoped partitioning
# ---------------------------------------------------------------------------


def test_two_runs_same_day_produce_two_parquets(tmp_path):
    """Review #8: a crash-then-restart on the SAME day must NOT
    destroy the earlier run's record.
    """
    rec1 = _new_recorder("paper", tmp_path, run_id="20260513T195020Z_pid111")
    rec1.record_submission(
        leg="entry", symbol="AAA", side="BUY", requested_qty=1,
        order_type="MKT", order_id="oid-first",
    )
    rec1.record_fill(
        leg="entry", symbol="AAA", order_id="oid-first",
        filled_qty=1, fill_price=100.0, status="Filled",
    )
    rec1.finalize(wait_for_t5min=None)

    rec2 = _new_recorder("paper", tmp_path, run_id="20260513T195530Z_pid222")
    rec2.record_submission(
        leg="entry", symbol="AAA", side="BUY", requested_qty=1,
        order_type="MKT", order_id="oid-second",
    )
    rec2.record_fill(
        leg="entry", symbol="AAA", order_id="oid-second",
        filled_qty=1, fill_price=101.0, status="Filled",
    )
    rec2.finalize(wait_for_t5min=None)

    parquets = sorted((tmp_path / "paper" / "2026-05-13").glob("*.parquet"))
    assert len(parquets) == 2
    df = pl.concat([pl.read_parquet(p) for p in parquets])
    assert set(df["run_id"].to_list()) == {
        "20260513T195020Z_pid111", "20260513T195530Z_pid222",
    }
    assert set(df["order_id"].to_list()) == {"oid-first", "oid-second"}


def test_journal_survives_a_finalize_failure(monkeypatch, tmp_path):
    rec = _new_recorder("paper", tmp_path)
    rec.record_submission(
        leg="entry", symbol="AAA", side="BUY", requested_qty=1,
        order_type="MKT", order_id="oid-1",
    )
    rec.record_fill(
        leg="entry", symbol="AAA", order_id="oid-1",
        filled_qty=1, fill_price=100.0, status="Filled",
    )
    def _boom(path, table):
        raise OSError("simulated disk full")
    monkeypatch.setattr(trade_recorder, "_atomic_write_arrow", _boom)
    result = rec.finalize(wait_for_t5min=None)
    assert result is None
    assert trade_recorder.resolve_journal_path(rec.final_path).exists()


def test_journal_records_full_payloads_not_just_counts(tmp_path):
    """Review #9: journal must be actually rebuildable — the crashed
    day cannot recover ``target_dollars`` from ``n_symbols=100``.
    """
    ny = timezone(timedelta(hours=-4))
    dt = datetime(2026, 5, 13, 15, 50, 10, tzinfo=ny).astimezone(_UTC)
    rec = _new_recorder("paper", tmp_path)
    rec.record_arrival_prices({
        "AAA": {"bid": 99.99, "ask": 100.01, "mid": 100.0,
                "rel_spread": 0.0002, "observed_ts_utc": dt},
    })
    rec.record_targets({"AAA": {"target_dollars": 5000.0, "target_shares": 50}})
    rec.record_t5min_mark(
        mid_and_ts_by_sym={"AAA": (100.05, dt + timedelta(minutes=5))},
        target_ts_utc=dt + timedelta(minutes=5),
    )
    events = []
    with rec.journal_path.open("r") as fh:
        for line in fh:
            events.append(json.loads(line))
    kinds = {e["kind"] for e in events}
    assert {"arrival_prices", "targets", "t5min"}.issubset(kinds)
    arr = next(e for e in events if e["kind"] == "arrival_prices")
    assert arr["arrivals"]["AAA"]["mid"] == 100.0
    tgt = next(e for e in events if e["kind"] == "targets")
    assert tgt["targets"]["AAA"]["target_dollars"] == 5000.0
    t5 = next(e for e in events if e["kind"] == "t5min")
    assert t5["mid_and_ts_by_sym"]["AAA"]["mid"] == 100.05


# ---------------------------------------------------------------------------
# Mode separation
# ---------------------------------------------------------------------------


def test_paper_live_replay_write_to_separate_dirs(tmp_path):
    for mode in ("paper", "live", "replay"):
        rec = trade_recorder.TradeRecorder(
            mode=mode,
            decision_ts_utc=_decision_ts(),
            universe=_universe(),
            trading_date="2026-05-13",
            base_dir=tmp_path,
            replay_date="2026-05-13" if mode == "replay" else None,
            run_id=f"run-{mode}",
        )
        rec.record_submission(
            leg="entry", symbol="AAA", side="BUY", requested_qty=1,
            order_type="MKT", order_id=f"oid-{mode}",
        )
        rec.record_fill(
            leg="entry", symbol="AAA", order_id=f"oid-{mode}",
            filled_qty=1, fill_price=100.0, status="Filled",
        )
        rec.finalize(wait_for_t5min=None)
    for mode in ("paper", "live", "replay"):
        assert (tmp_path / mode / "2026-05-13" / f"run-{mode}.parquet").exists()


def test_resolve_output_path_rejects_unknown_mode():
    with pytest.raises(ValueError):
        trade_recorder.resolve_output_path("shadow", "2026-05-13", "run-1")


# ---------------------------------------------------------------------------
# safe_call
# ---------------------------------------------------------------------------


def test_safe_call_swallows_all_exceptions():
    def _boom():
        raise RuntimeError("simulated recorder failure")
    trade_recorder.safe_call(_boom)


# ---------------------------------------------------------------------------
# rel_spread_at_decision — full relative spread, not half
# ---------------------------------------------------------------------------


def test_rel_spread_column_stores_full_relative_spread(tmp_path):
    """Review #12: the stored value is (ask - bid) / mid — the FULL
    relative spread. Column renamed to ``rel_spread_at_decision`` to
    make this explicit.
    """
    bid, ask = 99.90, 100.10
    mid = (bid + ask) / 2
    real_full_rel_spread = (ask - bid) / mid  # 0.002
    rec = _new_recorder("paper", tmp_path)
    rec.record_arrival_prices({
        "AAA": {
            "bid": bid, "ask": ask, "mid": mid,
            "rel_spread": real_full_rel_spread,
            "observed_ts_utc": _decision_ts().astimezone(_UTC),
        },
    })
    rec.record_submission(
        leg="entry", symbol="AAA", side="BUY", requested_qty=1,
        order_type="MKT", order_id="oid-1",
    )
    rec.record_fill(
        leg="entry", symbol="AAA", order_id="oid-1",
        filled_qty=1, fill_price=100.0, status="Filled",
    )
    row = pl.read_parquet(rec.finalize(wait_for_t5min=None)).row(0, named=True)
    assert row["rel_spread_at_decision"] == pytest.approx(0.002)
    # Pin: NOT the half.
    assert row["rel_spread_at_decision"] != pytest.approx(0.001)


# ---------------------------------------------------------------------------
# arrival_ts_utc / staleness
# ---------------------------------------------------------------------------


def test_arrival_ts_and_staleness_populated(tmp_path):
    """Review #6: a thin name's arrival ts must be flaggable via
    ``arrival_staleness_secs``.
    """
    ny = timezone(timedelta(hours=-4))
    decision_utc = datetime(2026, 5, 13, 15, 50, 10, tzinfo=ny).astimezone(_UTC)
    stale_tick = datetime(2026, 5, 13, 15, 41, 0, tzinfo=ny).astimezone(_UTC)
    rec = _new_recorder("paper", tmp_path)
    rec.record_arrival_prices({
        "AAA": {
            "bid": 99.99, "ask": 100.01, "mid": 100.0,
            "rel_spread": 0.0002, "observed_ts_utc": stale_tick,
        },
    })
    rec.record_submission(
        leg="entry", symbol="AAA", side="BUY", requested_qty=1,
        order_type="MKT", order_id="oid-1",
    )
    rec.record_fill(
        leg="entry", symbol="AAA", order_id="oid-1",
        filled_qty=1, fill_price=100.0, status="Filled",
    )
    row = pl.read_parquet(rec.finalize(wait_for_t5min=None)).row(0, named=True)
    assert row["arrival_ts_utc"] == stale_tick
    expected = (decision_utc - stale_tick).total_seconds()
    assert row["arrival_staleness_secs"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# arrival_prices_from_day_df
# ---------------------------------------------------------------------------


def test_arrival_prices_from_day_df_computes_bid_ask_from_rel_spread():
    ny = timezone(timedelta(hours=-4))
    day_df = pl.DataFrame({
        "symbol": ["AAA", "BBB"],
        "entry_price": [100.0, 50.0],
        "spread_return": [0.002, 0.004],
    })
    last_mid = {"AAA": 100.01}
    last_ts = {"AAA": datetime(2026, 5, 13, 15, 49, 58, tzinfo=ny).astimezone(_UTC)}
    arrivals = trade_recorder.arrival_prices_from_day_df(day_df, last_mid, last_ts)
    assert arrivals["AAA"]["mid"] == pytest.approx(100.01)
    assert arrivals["AAA"]["ask"] == pytest.approx(100.01 + 0.002 * 100.01 / 2)
    assert arrivals["AAA"]["bid"] == pytest.approx(100.01 - 0.002 * 100.01 / 2)
    assert arrivals["AAA"]["rel_spread"] == pytest.approx(0.002)
    assert arrivals["AAA"]["observed_ts_utc"] == last_ts["AAA"]
    assert arrivals["BBB"]["mid"] == pytest.approx(50.0)
    assert "observed_ts_utc" not in arrivals["BBB"]


def test_arrival_prices_from_empty_day_df_returns_empty_dict():
    assert trade_recorder.arrival_prices_from_day_df(pl.DataFrame(), None, None) == {}


# ---------------------------------------------------------------------------
# extract_broker_extras
# ---------------------------------------------------------------------------


def test_extract_broker_extras_handles_missing_trade():
    class _NoTrades:
        pass
    assert trade_recorder.extract_broker_extras(_NoTrades(), "oid-x") == {}


def test_extract_broker_extras_pulls_commission_and_exec_id():
    exec_ = SimpleNamespace(
        execId="EX-42", exchange="ISLAND",
        lastLiquidity="R", time=datetime(2026, 5, 13, 15, 50, 12, tzinfo=_UTC),
    )
    comm = SimpleNamespace(commission=0.35, currency="USD", realizedPNL=None)
    fill = SimpleNamespace(execution=exec_, commissionReport=comm)
    trade = SimpleNamespace(
        order=SimpleNamespace(
            permId=123456789, clientId=1, tif="DAY", account="DU12345",
        ),
        contract=SimpleNamespace(exchange="ISLAND"),
        orderStatus=SimpleNamespace(remaining=0),
        fills=[fill],
    )
    broker = SimpleNamespace(_trades={"oid-1": trade})
    extras = trade_recorder.extract_broker_extras(broker, "oid-1")
    assert extras["perm_id"] == 123456789
    assert extras["client_id"] == 1
    assert extras["tif"] == "DAY"
    assert extras["account"] == "DU12345"
    assert extras["exchange"] == "ISLAND"
    assert extras["commission"] == pytest.approx(0.35)
    assert extras["commission_currency"] == "USD"
    assert extras["exec_id"] == "EX-42"
    assert extras["venue"] == "ISLAND"
    assert extras["liquidity"] == "R"
    assert extras["remaining_qty"] == 0.0
    assert extras["fill_ts_utc"] == datetime(2026, 5, 13, 15, 50, 12, tzinfo=_UTC)


def test_extract_broker_extras_swallows_walk_failures():
    class _RaisingTrade:
        @property
        def order(self):
            raise RuntimeError("attribute error")
    broker = SimpleNamespace(_trades={"oid-1": _RaisingTrade()})
    extras = trade_recorder.extract_broker_extras(broker, "oid-1")
    assert isinstance(extras, dict)


# ---------------------------------------------------------------------------
# Structural test-isolation
# ---------------------------------------------------------------------------


def test_resolve_trades_dir_redirects_with_fake_broker():
    """The safeguard must key on 'is this a real Broker' — a notebook
    with a fake broker at production ``_TRADES_DIR`` triggers the
    scratch redirect. Under this test we intentionally unpatch
    ``_TRADES_DIR`` back to the production default to exercise it.
    """
    class _FakeBroker:
        pass
    default = config.LIVE_LOG_DIR / "trades"
    _saved_dir = daily_orchestrator._TRADES_DIR
    _saved_scratch = daily_orchestrator._PYTEST_TRADES_SCRATCH
    try:
        daily_orchestrator._TRADES_DIR = default
        daily_orchestrator._PYTEST_TRADES_SCRATCH = None
        resolved = daily_orchestrator._resolve_trades_dir(broker=_FakeBroker())
        assert resolved != default
        assert str(resolved).startswith(("/tmp/", "/var/folders/", "/private/"))
    finally:
        daily_orchestrator._TRADES_DIR = _saved_dir
        daily_orchestrator._PYTEST_TRADES_SCRATCH = _saved_scratch


def test_resolve_trades_dir_honours_explicit_override(monkeypatch, tmp_path):
    monkeypatch.setattr(daily_orchestrator, "_TRADES_DIR", tmp_path / "trades")
    class _FakeBroker:
        pass
    resolved = daily_orchestrator._resolve_trades_dir(broker=_FakeBroker())
    assert resolved == tmp_path / "trades"


# ---------------------------------------------------------------------------
# Round 2 review regressions
# ---------------------------------------------------------------------------


def test_resolve_trades_dir_warns_on_redirect(monkeypatch, caplog):
    """Round 2 #1: the scratch redirect must NOT be silent. A
    production broker-type misdetection is the exact scenario this
    warning exists to make visible.
    """
    import logging as _logging
    class _FakeBroker:
        pass
    default = config.LIVE_LOG_DIR / "trades"
    _saved_dir = daily_orchestrator._TRADES_DIR
    _saved_scratch = daily_orchestrator._PYTEST_TRADES_SCRATCH
    try:
        daily_orchestrator._TRADES_DIR = default
        daily_orchestrator._PYTEST_TRADES_SCRATCH = None
        # Capture the module logger.
        records: list[_logging.LogRecord] = []
        class _H(_logging.Handler):
            def emit(self, record):
                records.append(record)
        h = _H(level=_logging.NOTSET)
        daily_orchestrator.logger.addHandler(h)
        try:
            daily_orchestrator._resolve_trades_dir(broker=_FakeBroker())
        finally:
            daily_orchestrator.logger.removeHandler(h)
    finally:
        daily_orchestrator._TRADES_DIR = _saved_dir
        daily_orchestrator._PYTEST_TRADES_SCRATCH = _saved_scratch
    warnings = [r for r in records if r.levelno == _logging.WARNING]
    assert warnings, "expected a WARNING on scratch redirect"
    msg = " ".join(r.getMessage() for r in warnings)
    assert "scratch" in msg.lower()
    assert "_FakeBroker" in msg or "broker type" in msg.lower()


def test_cleanup_pytest_scratch_skips_delete_outside_pytest(monkeypatch, tmp_path):
    """Round 2 #1: the atexit rmtree must be gated on
    ``PYTEST_CURRENT_TEST``. A production misdetection would otherwise
    silently delete the ONLY record of the day's fills.
    """
    scratch = tmp_path / "scratch-under-test"
    scratch.mkdir()
    (scratch / "important.parquet").write_text("real-fills")
    monkeypatch.setattr(daily_orchestrator, "_PYTEST_TRADES_SCRATCH", scratch)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    daily_orchestrator._cleanup_pytest_trades_scratch()
    # Not-under-pytest: preserved.
    assert scratch.exists()
    assert (scratch / "important.parquet").exists()


def test_cleanup_pytest_scratch_deletes_under_pytest(monkeypatch, tmp_path):
    """Under pytest we still want the scratch dir cleaned up so it
    doesn't accumulate.
    """
    scratch = tmp_path / "scratch-under-test"
    scratch.mkdir()
    (scratch / "junk.parquet").write_text("test data")
    monkeypatch.setattr(daily_orchestrator, "_PYTEST_TRADES_SCRATCH", scratch)
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "test_x (call)")
    daily_orchestrator._cleanup_pytest_trades_scratch()
    assert not scratch.exists()


def test_fill_ts_utc_null_on_timeout_row(tmp_path):
    """Round 2 #3: a timeout / cancelled / no-fill row must NOT carry
    a fill timestamp — downstream latency computed as
    ``fill_ts_utc - sent_ts_utc`` on a timeout would be a fabricated
    tens-of-seconds datapoint from a trade that never happened.
    """
    rec = _new_recorder("paper", tmp_path)
    rec.record_submission(
        leg="moc", symbol="AAA", side="SELL", requested_qty=50,
        order_type="MOC", order_id="oid-t",
    )
    obs = datetime(2026, 5, 13, 20, 5, 0, tzinfo=_UTC)
    rec.record_fill(
        leg="moc", symbol="AAA", order_id="oid-t",
        filled_qty=None, fill_price=None, status="timeout",
        observed_ts_utc=obs,
    )
    row = pl.read_parquet(rec.finalize(wait_for_t5min=None)).row(0, named=True)
    assert row["observed_ts_utc"] == obs
    assert row["fill_ts_utc"] is None, (
        "timeout row should NOT carry a fill_ts_utc"
    )
    assert row["filled_qty"] is None
    assert row["achieved_dollars"] is None


def test_fill_ts_utc_null_on_zero_cancel_row(tmp_path):
    rec = _new_recorder("paper", tmp_path)
    rec.record_submission(
        leg="entry", symbol="AAA", side="BUY", requested_qty=10,
        order_type="MKT", order_id="oid-c",
    )
    obs = datetime(2026, 5, 13, 20, 5, 0, tzinfo=_UTC)
    rec.record_fill(
        leg="entry", symbol="AAA", order_id="oid-c",
        filled_qty=0, fill_price=None, status="Cancelled",
        observed_ts_utc=obs,
    )
    row = pl.read_parquet(rec.finalize(wait_for_t5min=None)).row(0, named=True)
    assert row["fill_ts_utc"] is None, (
        "Cancelled-with-zero-fill row should NOT carry a fill_ts_utc"
    )
    # achieved_dollars is 0.0 (terminal cancel) — the reviewer's other
    # rule; NOT affected here.
    assert row["achieved_dollars"] == 0.0


def test_arrival_staleness_baseline_is_snapshot_ts_utc(tmp_path):
    """Round 2 #7: baseline is ``snapshot_ts_utc`` (the actual
    snapshot completion) so a tick that arrived between the scheduled
    15:50:10 and the actual snapshot doesn't produce a negative
    staleness.
    """
    ny = timezone(timedelta(hours=-4))
    scheduled = datetime(2026, 5, 13, 15, 50, 10, tzinfo=ny).astimezone(_UTC)
    # Snapshot actually completed at 15:50:11.5 (typical 1-2s delay).
    snapshot_completed = scheduled + timedelta(seconds=1, milliseconds=500)
    # Tick delivered AFTER 15:50:10 but BEFORE snapshot completion.
    tick_ts = scheduled + timedelta(milliseconds=500)  # 15:50:10.5

    rec = trade_recorder.TradeRecorder(
        mode="paper", decision_ts_utc=scheduled,
        universe=["AAA"], trading_date="2026-05-13",
        base_dir=tmp_path, snapshot_ts_utc=snapshot_completed,
        run_id="test-r",
    )
    rec.record_arrival_prices({
        "AAA": {"bid": 99.99, "ask": 100.01, "mid": 100.0,
                "rel_spread": 0.0002, "observed_ts_utc": tick_ts},
    })
    rec.record_submission(
        leg="entry", symbol="AAA", side="BUY", requested_qty=1,
        order_type="MKT", order_id="oid-1",
    )
    rec.record_fill(
        leg="entry", symbol="AAA", order_id="oid-1",
        filled_qty=1, fill_price=100.0, status="Filled",
    )
    row = pl.read_parquet(rec.finalize(wait_for_t5min=None)).row(0, named=True)
    # Baseline is snapshot (15:50:11.5) - tick (15:50:10.5) = 1.0s.
    # Previously used decision (15:50:10) - tick (15:50:10.5) = -0.5s.
    assert row["arrival_staleness_secs"] == pytest.approx(1.0)
    assert row["arrival_staleness_secs"] >= 0


def test_wait_for_t5min_helper(tmp_path):
    """The orchestrator uses this to stop the feed as soon as the
    T+5min capture finishes, instead of holding it open through the
    fill audit.
    """
    class _FakeFeed:
        def snapshot(self):
            return {"tbbo": pl.DataFrame([{
                "ts_event": datetime.now(_UTC),
                "symbol": "AAA",
                "price": 100.0, "size": 100,
                "ask_px_00": 100.01, "bid_px_00": 99.99,
            }])}
        def stop(self, timeout=None):
            pass

    rec = _new_recorder("paper", tmp_path)
    # No thread scheduled yet -> returns False.
    assert rec.wait_for_t5min(timeout=0.1) is False
    rec.schedule_t5min_capture_from_feed(
        _FakeFeed(), datetime.now(_UTC),
        offset=timedelta(milliseconds=10),
    )
    assert rec.wait_for_t5min(timeout=5.0) is True


def test_real_live_logs_trades_dir_untouched_by_this_module():
    real = config.LIVE_LOG_DIR / "trades"
    if not real.exists():
        assert real not in _REAL_TRADES_BASELINE
        return
    current = {p for p in real.rglob("*") if p.is_file()}
    added = current - _REAL_TRADES_BASELINE
    assert not added, (
        f"tests wrote to the real production trades dir: {sorted(added)}. "
    )
