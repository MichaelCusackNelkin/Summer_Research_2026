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
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from live import config
from live import daily_orchestrator
from live import trade_recorder


_UTC = timezone.utc


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


# Capture the REAL live/logs path at module load — before any test
# fixture has a chance to patch ``config.LIVE_LOG_DIR`` to tmp_path.
# The end-of-suite leak check compares against this baseline, not
# against a moving target.
_REAL_LIVE_LOG_DIR: Path = config.LIVE_LOG_DIR


def _snapshot_real_trades_dir_before_tests() -> set[Path]:
    real = _REAL_LIVE_LOG_DIR / "trades"
    if not real.exists():
        return set()
    return {p for p in real.rglob("*") if p.is_file()}


def _snapshot_real_logs_files_before_tests() -> dict[Path, int]:
    """Byte-length snapshot of every .log file at the REAL LIVE_LOG_DIR.
    A test that leaked into ``live.daily_orchestrator.log`` shows up
    as a size delta on the SAME file, not a new file — the file
    already exists in production, so ``set(rglob)`` diffing misses
    the leak. Byte-lengths do not.
    """
    if not _REAL_LIVE_LOG_DIR.exists():
        return {}
    out: dict[Path, int] = {}
    for p in _REAL_LIVE_LOG_DIR.rglob("*.log*"):
        if p.is_file():
            try:
                out[p] = p.stat().st_size
            except OSError:
                pass
    return out


_REAL_TRADES_BASELINE = _snapshot_real_trades_dir_before_tests()
_REAL_LOG_SIZES_BASELINE = _snapshot_real_logs_files_before_tests()


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
    path = rec.finalize()
    assert path is not None
    table = pq.read_table(path)
    for name, dtype in zip(
        trade_recorder.TRADE_RECORD_SCHEMA.names,
        trade_recorder.TRADE_RECORD_SCHEMA.types,
    ):
        assert name in table.schema.names, f"missing column {name}"
        got = table.schema.field(name).type
        assert got == dtype, f"{name}: expected {dtype}, got {got}"
    assert trade_recorder.SCHEMA_VERSION == 4
    # v4 must include per-minute mark columns for T+1..T+5.
    for _n in trade_recorder.MARK_OFFSETS_MINUTES:
        for _col in (f"t{_n}min_price", f"t{_n}min_ts_utc",
                     f"t{_n}min_staleness_secs"):
            assert _col in table.schema.names, (
                f"v4 missing mark column {_col}"
            )


def test_finalize_produces_no_file_when_empty(tmp_path):
    rec = _new_recorder("paper", tmp_path)
    result = rec.finalize()
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
    path = rec.finalize()
    table = pq.read_table(path)
    cols_to_check = ["commission", "realized_pnl", "arrival_staleness_secs"]
    # Every mark column must keep its declared type through an
    # all-null column — pyarrow inference on a list-of-dicts would
    # otherwise demote all-null columns to ``null``, breaking cross-
    # day concatenation.
    for _n in trade_recorder.MARK_OFFSETS_MINUTES:
        cols_to_check.extend([
            f"t{_n}min_price", f"t{_n}min_ts_utc",
            f"t{_n}min_staleness_secs",
        ])
    for col in cols_to_check:
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
    path = rec.finalize()
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
    path = rec.finalize()
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
    path = rec.finalize()
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
    path = rec.finalize()
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
    path = rec.finalize()
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
    path = rec.finalize()
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
    path = rec.finalize()
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
    row = pl.read_parquet(rec.finalize()).row(0, named=True)
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
    row = pl.read_parquet(rec.finalize()).row(0, named=True)
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
    row = pl.read_parquet(rec.finalize()).row(0, named=True)
    assert row["fill_ts_utc"] is None
    assert row["fill_ts_source"] is None


# ---------------------------------------------------------------------------
# Impact-decay curve staleness + no fabricated ts — see the
# `v4 impact-decay-curve marks` block below for the full test suite.
# The two tests kept here specifically pin per-symbol degradation to
# nulls (they're written against the multi-mark record_marks path
# rather than a decision-time snapshot).
# ---------------------------------------------------------------------------


def test_marks_thin_symbol_records_observed_ts_not_target(tmp_path):
    """Review #4: for a THIN name whose last tbbo tick was 15:50:02,
    ``t5min_price == arrival_price`` (by construction) but
    ``t5min_ts_utc`` must be the observed tick, NOT the intended
    target. ``t5min_staleness_secs`` is non-zero. Post-trade impact
    calibration filters on this — silently pretending the price was
    fresh at 15:55 would bias impact toward zero decay in the tail.

    Ported to the v4 multi-mark API so all five marks record the
    same observed_ts / staleness pattern for a thin symbol.
    """
    ny = timezone(timedelta(hours=-4))
    decision_utc = datetime(2026, 5, 13, 15, 50, 10, tzinfo=ny).astimezone(_UTC)
    tick_ts = datetime(2026, 5, 13, 15, 50, 2, tzinfo=ny).astimezone(_UTC)
    tbbo = pl.DataFrame([{
        "ts_event": tick_ts, "symbol": "AAA",
        "price": 100.0, "size": 100,
        "ask_px_00": 100.005, "bid_px_00": 99.995,
    }])
    rec = _new_recorder("replay", tmp_path, replay_date="2026-05-13")
    rec.record_arrival_prices({
        "AAA": {
            "bid": 99.995, "ask": 100.005, "mid": 100.0,
            "rel_spread": 0.0001, "observed_ts_utc": tick_ts,
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
    rec.capture_marks_from_tbbo(tbbo, decision_utc)
    rec.finalize()
    row = pl.read_parquet(rec.final_path).row(0, named=True)
    # T+5 mark pins the original v3 behavior.
    assert row["t5min_price"] == pytest.approx(row["arrival_price"])
    assert row["t5min_ts_utc"] == tick_ts
    target_ts_5 = decision_utc + timedelta(minutes=5)
    assert row["t5min_staleness_secs"] == pytest.approx(
        (target_ts_5 - tick_ts).total_seconds()
    )
    assert row["t5min_staleness_secs"] > 300
    # And every earlier mark holds the same observed_ts (the tick is
    # the only one available) with monotonically-smaller staleness.
    for n in (1, 2, 3, 4):
        assert row[f"t{n}min_ts_utc"] == tick_ts
        assert row[f"t{n}min_price"] == pytest.approx(100.0)
    stalenesses = [row[f"t{n}min_staleness_secs"] for n in (1, 2, 3, 4, 5)]
    assert stalenesses == sorted(stalenesses)


def test_marks_fresh_tick_has_zero_staleness_and_moves(tmp_path):
    """A tbbo tick EXACTLY at each target instant gives zero
    staleness and a moving price across marks — regression guard
    against a fabricated-fresh-tick shortcut.
    """
    ny = timezone(timedelta(hours=-4))
    decision_utc = datetime(2026, 5, 13, 15, 50, 10, tzinfo=ny).astimezone(_UTC)
    # Fresh ticks at each target instant, drifting price.
    rows = []
    for n in (1, 2, 3, 4, 5):
        rows.append({
            "ts_event": decision_utc + timedelta(minutes=n),
            "symbol": "AAA",
            "price": 100.0 + n * 0.10, "size": 100,
            "ask_px_00": 100.005 + n * 0.10,
            "bid_px_00": 99.995 + n * 0.10,
        })
    tbbo = pl.DataFrame(rows)
    rec = _new_recorder("replay", tmp_path, replay_date="2026-05-13")
    rec.record_arrival_prices({
        "AAA": {
            "bid": 99.995, "ask": 100.005, "mid": 100.0,
            "rel_spread": 0.0001, "observed_ts_utc": decision_utc,
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
    rec.capture_marks_from_tbbo(tbbo, decision_utc)
    rec.finalize()
    row = pl.read_parquet(rec.final_path).row(0, named=True)
    for n in (1, 2, 3, 4, 5):
        assert row[f"t{n}min_price"] == pytest.approx(100.0 + n * 0.10)
        assert row[f"t{n}min_staleness_secs"] == pytest.approx(0.0, abs=1e-6)
    # And the price moved off arrival on every mark.
    assert row["t5min_price"] != row["arrival_price"]


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
    rec1.finalize()

    rec2 = _new_recorder("paper", tmp_path, run_id="20260513T195530Z_pid222")
    rec2.record_submission(
        leg="entry", symbol="AAA", side="BUY", requested_qty=1,
        order_type="MKT", order_id="oid-second",
    )
    rec2.record_fill(
        leg="entry", symbol="AAA", order_id="oid-second",
        filled_qty=1, fill_price=101.0, status="Filled",
    )
    rec2.finalize()

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
    result = rec.finalize()
    assert result is None
    assert trade_recorder.resolve_journal_path(rec.final_path).exists()


def test_journal_records_full_payloads_not_just_counts(tmp_path):
    """Review #9: journal must be actually rebuildable — the crashed
    day cannot recover ``target_dollars`` from ``n_symbols=100``.
    Ported to the v4 record_marks API; kinds are
    ``{'arrival_prices', 'targets', 'marks'}``.
    """
    ny = timezone(timedelta(hours=-4))
    dt = datetime(2026, 5, 13, 15, 50, 10, tzinfo=ny).astimezone(_UTC)
    rec = _new_recorder("paper", tmp_path)
    rec.record_arrival_prices({
        "AAA": {"bid": 99.99, "ask": 100.01, "mid": 100.0,
                "rel_spread": 0.0002, "observed_ts_utc": dt},
    })
    rec.record_targets({"AAA": {"target_dollars": 5000.0, "target_shares": 50}})
    rec.record_marks(
        marks_by_offset_minutes={
            5: {"AAA": (100.05, dt + timedelta(minutes=5))},
        },
        target_ts_by_offset_minutes={5: dt + timedelta(minutes=5)},
    )
    events = []
    with rec.journal_path.open("r") as fh:
        for line in fh:
            events.append(json.loads(line))
    kinds = {e["kind"] for e in events}
    assert {"arrival_prices", "targets", "marks"}.issubset(kinds)
    arr = next(e for e in events if e["kind"] == "arrival_prices")
    assert arr["arrivals"]["AAA"]["mid"] == 100.0
    tgt = next(e for e in events if e["kind"] == "targets")
    assert tgt["targets"]["AAA"]["target_dollars"] == 5000.0
    marks_ev = next(e for e in events if e["kind"] == "marks")
    assert marks_ev["marks_by_offset_minutes"]["5"]["AAA"]["mid"] == 100.05


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
        rec.finalize()
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
    row = pl.read_parquet(rec.finalize()).row(0, named=True)
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
    row = pl.read_parquet(rec.finalize()).row(0, named=True)
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
    row = pl.read_parquet(rec.finalize()).row(0, named=True)
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
    row = pl.read_parquet(rec.finalize()).row(0, named=True)
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
    row = pl.read_parquet(rec.finalize()).row(0, named=True)
    # Baseline is snapshot (15:50:11.5) - tick (15:50:10.5) = 1.0s.
    # Previously used decision (15:50:10) - tick (15:50:10.5) = -0.5s.
    assert row["arrival_staleness_secs"] == pytest.approx(1.0)
    assert row["arrival_staleness_secs"] >= 0


# ---------------------------------------------------------------------------
# v4 impact-decay-curve marks (T+1..T+5min)
#
# These tests pin the new multi-mark surface. Constraints they defend:
#   - all 5 marks are populated from a single feed snapshot at T+MAX
#     (~15:55) — the design uses one wake, one thread, one snapshot,
#     then walk-back per offset (see ``schedule_marks_capture_from_feed``
#     docstring). Five wall-clock timers would burn extra feed lifetime
#     and add concurrency for zero information gain.
#   - each mark reflects ONLY data at or before its own target — a
#     property of ``_mid_at_or_before(tbbo, decision + N*min)``, NOT of
#     the outer capture time. If the T+1 filter ever slipped to include
#     ticks after T+1, the decay curve would be biased toward zero
#     (temporary + permanent impact would look permanent-only).
#   - each mark records its OBSERVED tick ts and a staleness gap
#     computed against its OWN target. A thin symbol that only prints
#     once yields the same price across all 5 marks with monotonically
#     INCREASING staleness — so an analyst can filter thin marks
#     per-offset instead of dropping the whole row.
#   - v3 t5min COLUMNS stay readable (same names / types / positions in
#     the schema); the v3 t5min METHODS were dead code and are gone.
# ---------------------------------------------------------------------------


def test_marks_all_five_populated_from_one_snapshot(tmp_path):
    """One synchronous walk-back over a tbbo that has one tick per
    target instant — all five marks come back populated with the
    right price. This is the happy path.
    """
    ny = timezone(timedelta(hours=-4))
    decision_utc = datetime(2026, 5, 13, 15, 50, 0, tzinfo=ny).astimezone(_UTC)
    rows = []
    for n in (1, 2, 3, 4, 5):
        ts = decision_utc + timedelta(minutes=n)
        # Each minute the mid drifts by +0.10.
        base = 100.0 + n * 0.10
        rows.append({
            "ts_event": ts, "symbol": "AAA",
            "price": base, "size": 100,
            "ask_px_00": base + 0.005, "bid_px_00": base - 0.005,
        })
    tbbo = pl.DataFrame(rows)
    rec = _new_recorder("replay", tmp_path, replay_date="2026-05-13")
    rec.record_arrival_prices({
        "AAA": {"bid": 99.995, "ask": 100.005, "mid": 100.0,
                "rel_spread": 0.0001, "observed_ts_utc": decision_utc},
    })
    rec.record_submission(
        leg="entry", symbol="AAA", side="BUY", requested_qty=10,
        order_type="MKT", order_id="oid-a",
    )
    rec.record_fill(
        leg="entry", symbol="AAA", order_id="oid-a",
        filled_qty=10, fill_price=100.0, status="Filled",
    )
    rec.capture_marks_from_tbbo(tbbo, decision_utc)
    rec.finalize()
    row = pl.read_parquet(rec.final_path).row(0, named=True)
    for n in (1, 2, 3, 4, 5):
        assert row[f"t{n}min_price"] == pytest.approx(100.0 + n * 0.10), (
            f"t{n}min_price mismatch: got {row[f't{n}min_price']}"
        )
        expected_ts = decision_utc + timedelta(minutes=n)
        assert row[f"t{n}min_ts_utc"] == expected_ts
        assert row[f"t{n}min_staleness_secs"] == pytest.approx(0.0, abs=1e-6)


def test_marks_each_reflects_only_data_at_or_before_its_own_target(tmp_path):
    """Load-bearing invariant against look-ahead: build a tbbo whose
    ticks fall strictly between minute boundaries, so each mark has
    an unambiguous "last tick at or before N minutes" answer, and
    check it. If ``_mid_at_or_before`` ever leaks ticks past a
    target the wrong price gets picked up here.

    Timing (all offsets from ``decision_utc``):
      +30s  -> price 100.10   (at-or-before T+1)
      +90s  -> price 100.20   (at-or-before T+2, not T+1)
      +150s -> price 100.30   (at-or-before T+3, not T+2)
      +210s -> price 100.40   (at-or-before T+4, not T+3)
      +270s -> price 100.50   (at-or-before T+5, not T+4)
    So the mark curve MUST be [100.10, 100.20, 100.30, 100.40, 100.50].
    """
    ny = timezone(timedelta(hours=-4))
    decision_utc = datetime(2026, 5, 13, 15, 50, 0, tzinfo=ny).astimezone(_UTC)
    ticks = [(30, 100.10), (90, 100.20), (150, 100.30),
             (210, 100.40), (270, 100.50)]
    rows = []
    for secs, price in ticks:
        rows.append({
            "ts_event": decision_utc + timedelta(seconds=secs),
            "symbol": "AAA",
            "price": price, "size": 100,
            "ask_px_00": price + 0.005, "bid_px_00": price - 0.005,
        })
    tbbo = pl.DataFrame(rows)
    rec = _new_recorder("replay", tmp_path, replay_date="2026-05-13")
    rec.record_arrival_prices({
        "AAA": {"bid": 99.995, "ask": 100.005, "mid": 100.0,
                "rel_spread": 0.0001, "observed_ts_utc": decision_utc},
    })
    rec.record_submission(
        leg="entry", symbol="AAA", side="BUY", requested_qty=10,
        order_type="MKT", order_id="oid-a",
    )
    rec.record_fill(
        leg="entry", symbol="AAA", order_id="oid-a",
        filled_qty=10, fill_price=100.0, status="Filled",
    )
    rec.capture_marks_from_tbbo(tbbo, decision_utc)
    rec.finalize()
    row = pl.read_parquet(rec.final_path).row(0, named=True)
    expected = {1: 100.10, 2: 100.20, 3: 100.30, 4: 100.40, 5: 100.50}
    for n, price in expected.items():
        assert row[f"t{n}min_price"] == pytest.approx(price), (
            f"t{n}min_price: expected {price}, got {row[f't{n}min_price']} "
            f"— filter may be leaking ticks past its own target"
        )
    # And observed timestamps are the tick times, NOT the target
    # instants — no fabricated fresh-tick stamps.
    for n, secs in ((1, 30), (2, 90), (3, 150), (4, 210), (5, 270)):
        assert row[f"t{n}min_ts_utc"] == decision_utc + timedelta(seconds=secs)


def test_marks_staleness_correct_sign_and_units_per_mark(tmp_path):
    """Staleness = target - observed_ts. Positive for a tick that
    landed BEFORE its target; units are seconds. Verify per mark
    from the tick pattern in the previous test.
    """
    ny = timezone(timedelta(hours=-4))
    decision_utc = datetime(2026, 5, 13, 15, 50, 0, tzinfo=ny).astimezone(_UTC)
    ticks = [(30, 100.10), (90, 100.20), (150, 100.30),
             (210, 100.40), (270, 100.50)]
    rows = []
    for secs, price in ticks:
        rows.append({
            "ts_event": decision_utc + timedelta(seconds=secs),
            "symbol": "AAA",
            "price": price, "size": 100,
            "ask_px_00": price + 0.005, "bid_px_00": price - 0.005,
        })
    tbbo = pl.DataFrame(rows)
    rec = _new_recorder("replay", tmp_path, replay_date="2026-05-13")
    rec.record_submission(
        leg="entry", symbol="AAA", side="BUY", requested_qty=1,
        order_type="MKT", order_id="oid-a",
    )
    rec.record_fill(
        leg="entry", symbol="AAA", order_id="oid-a",
        filled_qty=1, fill_price=100.0, status="Filled",
    )
    rec.capture_marks_from_tbbo(tbbo, decision_utc)
    rec.finalize()
    row = pl.read_parquet(rec.final_path).row(0, named=True)
    # T+1 target = decision+60s; last-at-or-before tick = decision+30s
    # so staleness = 60 - 30 = 30s. Etc.
    expected_stale = {
        1: 60 - 30, 2: 120 - 90, 3: 180 - 150, 4: 240 - 210, 5: 300 - 270,
    }
    for n, s in expected_stale.items():
        assert row[f"t{n}min_staleness_secs"] == pytest.approx(float(s)), (
            f"t{n}min_staleness_secs: expected {s}s, got "
            f"{row[f't{n}min_staleness_secs']}"
        )
        assert row[f"t{n}min_staleness_secs"] > 0


def test_marks_thin_symbol_same_price_increasing_staleness(tmp_path):
    """A thin name whose ONLY print of the day landed at 15:41 —
    every mark uses that same price (== arrival_price) but the
    staleness gap grows minute over minute. Analyst filters on
    ``t{n}min_staleness_secs > threshold`` to drop the mark for
    that individual name/offset instead of dropping the whole row.

    This is the exact filter the impact-decay calibration needs;
    without it a thin symbol would silently pin decay at 0.
    """
    ny = timezone(timedelta(hours=-4))
    decision_utc = datetime(2026, 5, 13, 15, 50, 0, tzinfo=ny).astimezone(_UTC)
    # Single tick at 15:41 — 9 minutes before decision.
    single_tick_ts = datetime(2026, 5, 13, 15, 41, 0, tzinfo=ny).astimezone(_UTC)
    tbbo = pl.DataFrame([{
        "ts_event": single_tick_ts, "symbol": "THIN",
        "price": 42.0, "size": 100,
        "ask_px_00": 42.005, "bid_px_00": 41.995,
    }])
    rec = _new_recorder("replay", tmp_path, replay_date="2026-05-13")
    rec.record_arrival_prices({
        "THIN": {"bid": 41.995, "ask": 42.005, "mid": 42.0,
                 "rel_spread": 0.00024,
                 "observed_ts_utc": single_tick_ts},
    })
    rec.record_submission(
        leg="entry", symbol="THIN", side="BUY", requested_qty=5,
        order_type="MKT", order_id="oid-thin",
    )
    rec.record_fill(
        leg="entry", symbol="THIN", order_id="oid-thin",
        filled_qty=5, fill_price=42.0, status="Filled",
    )
    rec.capture_marks_from_tbbo(tbbo, decision_utc)
    rec.finalize()
    row = pl.read_parquet(rec.final_path).row(0, named=True)
    # Every mark uses the pre-decision tick.
    stalenesses = []
    for n in (1, 2, 3, 4, 5):
        assert row[f"t{n}min_price"] == pytest.approx(42.0), (
            f"thin symbol t{n}min_price != last tick price"
        )
        assert row[f"t{n}min_price"] == pytest.approx(row["arrival_price"])
        assert row[f"t{n}min_ts_utc"] == single_tick_ts
        stalenesses.append(row[f"t{n}min_staleness_secs"])
    # Staleness grows by 60s per offset.
    for i in range(1, len(stalenesses)):
        assert stalenesses[i] > stalenesses[i - 1], (
            f"staleness not monotonically increasing: {stalenesses}"
        )
        diff = stalenesses[i] - stalenesses[i - 1]
        assert diff == pytest.approx(60.0), (
            f"minute-over-minute staleness grew by {diff}s, expected 60s"
        )
    # And it's positive and large enough to filter — a T+1 tick that
    # was actually from 9 minutes before decision reads as ~600s stale.
    assert stalenesses[0] > 500


def test_marks_schema_v4_roundtrips_with_all_null_columns(tmp_path):
    """A row with NO marks recorded must still round-trip through
    parquet with every mark column present and correctly typed.
    Pyarrow inference on a list of dicts would demote all-null
    columns to ``null`` and refuse to concat across days; the
    explicit schema is what protects against that.
    """
    rec = _new_recorder("paper", tmp_path)
    rec.record_submission(
        leg="entry", symbol="AAA", side="BUY", requested_qty=1,
        order_type="MKT", order_id="oid-x",
    )
    rec.record_fill(
        leg="entry", symbol="AAA", order_id="oid-x",
        filled_qty=1, fill_price=100.0, status="Filled",
    )
    path = rec.finalize()
    table = pq.read_table(path)
    # Every mark column present and all-null.
    for n in trade_recorder.MARK_OFFSETS_MINUTES:
        for col in (f"t{n}min_price", f"t{n}min_ts_utc",
                    f"t{n}min_staleness_secs"):
            assert col in table.schema.names
            assert table.column(col).null_count == table.num_rows
            assert table.schema.field(col).type.equals(
                trade_recorder.TRADE_RECORD_SCHEMA.field(col).type
            )
    # schema_version stamped as 4 on every row.
    df = pl.read_parquet(path)
    assert (df["schema_version"] == 4).all()


def test_marks_replay_path_produces_curve(tmp_path):
    """Replay path (``capture_marks_from_tbbo``) — synchronous, walks
    the historical tbbo. Two symbols, five marks each, all populated.
    """
    ny = timezone(timedelta(hours=-4))
    decision_utc = datetime(2026, 5, 13, 15, 50, 0, tzinfo=ny).astimezone(_UTC)
    rows = []
    for sym, base in (("AAA", 100.0), ("BBB", 50.0)):
        for n in (1, 2, 3, 4, 5):
            price = base + n * 0.05
            rows.append({
                "ts_event": decision_utc + timedelta(minutes=n),
                "symbol": sym,
                "price": price, "size": 100,
                "ask_px_00": price + 0.005,
                "bid_px_00": price - 0.005,
            })
    tbbo = pl.DataFrame(rows)
    rec = _new_recorder("replay", tmp_path, replay_date="2026-05-13")
    for sym, base in (("AAA", 100.0), ("BBB", 50.0)):
        rec.record_submission(
            leg="entry", symbol=sym, side="BUY", requested_qty=5,
            order_type="MKT", order_id=f"oid-{sym}",
        )
        rec.record_fill(
            leg="entry", symbol=sym, order_id=f"oid-{sym}",
            filled_qty=5, fill_price=base, status="Filled",
        )
    rec.capture_marks_from_tbbo(tbbo, decision_utc)
    rec.finalize()
    df = pl.read_parquet(rec.final_path)
    for sym, base in (("AAA", 100.0), ("BBB", 50.0)):
        row = df.filter(pl.col("symbol") == sym).row(0, named=True)
        for n in (1, 2, 3, 4, 5):
            assert row[f"t{n}min_price"] == pytest.approx(base + n * 0.05)


def test_marks_background_thread_exercises_walkback_and_mid_basis(tmp_path):
    """The single-snapshot walk-back MUST reach the per-mark
    resolution path in the threaded code — not just the sync one.

    Setup that actually exercises it: put ``decision`` in the PAST
    (6 min ago) so every offset target has already elapsed; the
    thread's wake-until-max-target loop breaks out on the first
    iteration and immediately takes ``feed.snapshot()``, then walks
    back through the same frame for each offset. With four ticks
    interleaved between the mark targets, each mark resolves to a
    DIFFERENT tick — reproducing what would happen in production
    when the thread wakes at T+5 and walks back through the day.

    Also pins two properties an earlier version silently violated:

    - **Mid basis, not last trade price.** Every fixture in this
      file happens to make ``price = (ask+bid)/2``, so substituting
      ``pl.col("price")`` for the mid computation in the recorder
      would pass every test. Here the quote is INTENTIONALLY
      asymmetric — ``price=100.0, bid=99.90, ask=100.30`` → mid
      100.10 — so a price-vs-mid regression would flip the assertion.

    - **Sleep-budget bound.** With ``max_wait_secs=0.05`` and the
      old ``time.sleep(min(remaining, 5.0))`` loop the thread would
      block for 5s before rechecking the budget. Assert the
      threaded call returns within <1s to pin that fix.
    """
    ny = timezone(timedelta(hours=-4))
    decision_utc = datetime.now(_UTC) - timedelta(minutes=6)
    # Ticks between minute boundaries so each mark picks a distinct
    # last-at-or-before. Asymmetric quote (mid != price) so the mid
    # basis is actually tested — see docstring.
    ticks = [
        (30, 100.10, 99.90, 100.30),   # last-at-or-before T+1
        (90, 100.20, 100.00, 100.40),  # last-at-or-before T+2
        (150, 100.30, 100.10, 100.50), # last-at-or-before T+3
        (210, 100.40, 100.20, 100.60), # last-at-or-before T+4
        (270, 100.50, 100.30, 100.70), # last-at-or-before T+5
    ]
    tbbo_rows = []
    for offset_s, price, bid, ask in ticks:
        tbbo_rows.append({
            "ts_event": decision_utc + timedelta(seconds=offset_s),
            "symbol": "ASYM",
            "price": price, "size": 100,
            "ask_px_00": ask, "bid_px_00": bid,
        })
    tbbo = pl.DataFrame(tbbo_rows)

    class _FakeFeed:
        def snapshot(self):
            return {"tbbo": tbbo, "imbalance": pl.DataFrame()}
        def stop(self, timeout=None):
            pass

    rec = _new_recorder("paper", tmp_path)
    rec.record_submission(
        leg="entry", symbol="ASYM", side="BUY", requested_qty=1,
        order_type="MKT", order_id="oid-asym",
    )
    rec.record_fill(
        leg="entry", symbol="ASYM", order_id="oid-asym",
        filled_qty=1, fill_price=100.10, status="Filled",
    )

    # Thread wakes ~immediately (decision is 6 min in the past, so
    # remaining <= 0 on first iteration). max_wait_secs is a
    # belt-and-braces cap in case _now_utc drift is somehow negative.
    rec.schedule_marks_capture_from_feed(
        _FakeFeed(), decision_utc, max_wait_secs=0.05,
    )
    t0 = time.time()
    assert rec.wait_for_marks(timeout=10.0) is True
    elapsed = time.time() - t0
    assert elapsed < 1.0, (
        f"threaded walk-back took {elapsed:.2f}s; the sleep-budget "
        f"fix in _worker (bound each sleep chunk by remaining budget) "
        f"is required so max_wait_secs actually caps wall time"
    )

    rec.finalize()
    row = pl.read_parquet(rec.final_path).row(0, named=True)
    # Each mark picks a DIFFERENT tick from the same snapshot ->
    # the mid basis is 100.10, 100.20, 100.30, 100.40, 100.50 —
    # NOT the price column (which would trivially match here too;
    # asymmetric quote proves the recorder is doing (ask+bid)/2).
    expected_mids = {1: 100.10, 2: 100.20, 3: 100.30, 4: 100.40, 5: 100.50}
    for n, mid in expected_mids.items():
        assert row[f"t{n}min_price"] == pytest.approx(mid), (
            f"t{n}min_price = {row[f't{n}min_price']} — either the "
            f"walk-back didn't produce distinct per-mark resolution or "
            f"the recorder substituted 'price' for '(ask+bid)/2'"
        )
    # And staleness is 30s for every mark (each mark's tick is
    # 30s before its target — 30, 60-30 for T+1; 90 tick vs 120
    # target for T+2, etc.).
    for n in (1, 2, 3, 4, 5):
        assert row[f"t{n}min_staleness_secs"] == pytest.approx(30.0)


def test_marks_wait_for_marks_helper(tmp_path):
    """``wait_for_marks`` returns False when no thread is scheduled,
    True once the background capture signals completion — pinned
    against a tight wall-time bound so the sleep-budget fix stays in.
    """
    # Asymmetric quote so a stray mid-vs-price regression still bites.
    class _FakeFeed:
        def snapshot(self):
            return {"tbbo": pl.DataFrame([{
                "ts_event": datetime.now(_UTC),
                "symbol": "AAA",
                "price": 100.0, "size": 100,
                "ask_px_00": 100.20, "bid_px_00": 99.60,
            }])}
        def stop(self, timeout=None):
            pass

    rec = _new_recorder("paper", tmp_path)
    assert rec.wait_for_marks(timeout=0.1) is False
    rec.schedule_marks_capture_from_feed(
        _FakeFeed(), datetime.now(_UTC) - timedelta(minutes=6),
        max_wait_secs=0.05,
    )
    t0 = time.time()
    assert rec.wait_for_marks(timeout=10.0) is True
    # If the sleep-budget fix is reverted, this loop would burn one
    # ~5s sleep chunk before rechecking the budget. The <1s bound
    # deterministically catches that regression instead of racing.
    assert time.time() - t0 < 1.0, (
        "wait_for_marks returned in > 1s — sleep chunk overshot "
        "max_wait_secs (revert of the budget-bound in _worker?)"
    )


def test_marks_finalize_default_does_not_block(tmp_path):
    """``finalize()`` (default) MUST NOT wait for the background marks
    thread — the rc=11 finally calls it and a 30s delay to the
    operator page is unacceptable. Same protection as v3 t5min.
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
    # Real-scale offset (5min) with no max_wait cap — the thread
    # would sleep for 5 min if the wait were mandatory.
    rec.schedule_marks_capture_from_feed(
        _SleepyFeed(), datetime.now(_UTC),
    )
    t0 = time.time()
    rec.finalize()
    elapsed = time.time() - t0
    assert elapsed < 2.0, (
        f"finalize blocked for {elapsed:.2f}s — must not wait for marks "
        f"by default (would delay a rc=11 operator page in prod)"
    )


def test_marks_thread_after_finalize_is_a_noop(tmp_path):
    """A late marks thread must not journal or update rows after
    ``finalize()`` has run. Same protection as v3 t5min.
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
    rec.record_submission(
        leg="entry", symbol="AAA", side="BUY", requested_qty=1,
        order_type="MKT", order_id="oid-1",
    )
    rec.record_fill(
        leg="entry", symbol="AAA", order_id="oid-1",
        filled_qty=1, fill_price=100.0, status="Filled",
    )
    rec.schedule_marks_capture_from_feed(
        _FakeFeed(), datetime.now(_UTC),
    )
    rec.finalize()
    time.sleep(0.1)
    assert not rec.journal_path.exists()


def test_marks_thread_survives_snapshot_failure(tmp_path):
    """A raising feed.snapshot() must NOT propagate out of the
    background thread. All mark columns stay null on the row; the
    parquet still writes cleanly.
    """
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
    rec.schedule_marks_capture_from_feed(
        _EvilFeed(), datetime.now(_UTC), max_wait_secs=0.05,
    )
    assert rec.wait_for_marks(timeout=5.0) is True
    rec.finalize()
    df = pl.read_parquet(rec.final_path)
    for n in (1, 2, 3, 4, 5):
        assert df[f"t{n}min_price"].to_list() == [None], (
            f"t{n}min_price should be null when feed.snapshot() raises"
        )


def test_marks_journal_records_full_per_mark_payloads(tmp_path):
    """The journal must capture all five marks in the recovery event
    — an analyst rebuilding the day from the JSONL needs the full
    per-mark payload, not just a count. Two ticks — one at
    decision (before every mark target), one at decision+3min
    (before T+3, T+4, T+5). So T+1, T+2 see the decision tick
    (mid 100.0); T+3..T+5 see the decision+3min tick (mid 100.5).
    """
    ny = timezone(timedelta(hours=-4))
    decision_utc = datetime(2026, 5, 13, 15, 50, 0, tzinfo=ny).astimezone(_UTC)
    tbbo = pl.DataFrame([
        {
            "ts_event": decision_utc,
            "symbol": "AAA",
            "price": 100.0, "size": 100,
            "ask_px_00": 100.005, "bid_px_00": 99.995,
        },
        {
            "ts_event": decision_utc + timedelta(minutes=3),
            "symbol": "AAA",
            "price": 100.5, "size": 100,
            "ask_px_00": 100.505, "bid_px_00": 100.495,
        },
    ])
    rec = _new_recorder("replay", tmp_path, replay_date="2026-05-13")
    rec.capture_marks_from_tbbo(tbbo, decision_utc)
    events = []
    with rec.journal_path.open("r") as fh:
        for line in fh:
            events.append(json.loads(line))
    marks_events = [e for e in events if e["kind"] == "marks"]
    assert marks_events, "expected a 'marks' journal event"
    ev = marks_events[0]
    # Every offset has a target ts.
    for n in (1, 2, 3, 4, 5):
        assert str(n) in ev["target_ts_by_offset_minutes"]
        assert ev["target_ts_by_offset_minutes"][str(n)] is not None
    # T+1, T+2 see the decision-time tick.
    for n in (1, 2):
        payload = ev["marks_by_offset_minutes"][str(n)]
        assert "AAA" in payload
        assert payload["AAA"]["mid"] == pytest.approx(100.0)
    # T+3, T+4, T+5 see the T+3 tick.
    for n in (3, 4, 5):
        payload = ev["marks_by_offset_minutes"][str(n)]
        assert "AAA" in payload
        assert payload["AAA"]["mid"] == pytest.approx(100.5)


def test_marks_backfill_when_row_created_after_marks_recorded(tmp_path):
    """Order-of-operations: marks recorded FIRST, then the entry row
    is created via a later submission. The per-mark stash must
    backfill onto the new row.

    Uses a decision-time tick so every offset target
    (decision + N min) qualifies for the walk-back — otherwise the
    filter correctly returns empty and we'd be testing nothing.
    """
    ny = timezone(timedelta(hours=-4))
    decision_utc = datetime(2026, 5, 13, 15, 50, 0, tzinfo=ny).astimezone(_UTC)
    tbbo = pl.DataFrame([{
        "ts_event": decision_utc,
        "symbol": "AAA",
        "price": 100.0, "size": 100,
        "ask_px_00": 100.005, "bid_px_00": 99.995,
    }])
    rec = _new_recorder("replay", tmp_path, replay_date="2026-05-13")
    # Marks first, no rows yet.
    rec.capture_marks_from_tbbo(tbbo, decision_utc)
    # THEN the submission — this materialises the entry row via
    # ``_row("entry", "AAA")``, which must backfill from the stash.
    rec.record_submission(
        leg="entry", symbol="AAA", side="BUY", requested_qty=1,
        order_type="MKT", order_id="oid-1",
    )
    rec.record_fill(
        leg="entry", symbol="AAA", order_id="oid-1",
        filled_qty=1, fill_price=100.0, status="Filled",
    )
    rec.finalize()
    row = pl.read_parquet(rec.final_path).row(0, named=True)
    for n in (1, 2, 3, 4, 5):
        # Every offset uses the pre-decision tick; the stash was
        # populated at capture time and had to backfill onto the row
        # that was created LATER via record_submission.
        assert row[f"t{n}min_price"] == pytest.approx(100.0), (
            f"t{n}min_price not backfilled — the stash was empty when the "
            f"row was created"
        )
        # And the observed ts / staleness backfilled too.
        assert row[f"t{n}min_ts_utc"] == decision_utc
        assert row[f"t{n}min_staleness_secs"] == pytest.approx(float(60 * n))


def test_marks_unknown_offset_is_skipped_not_dropped(tmp_path):
    """A caller passing an offset outside ``MARK_OFFSETS_MINUTES``
    (say n=6) gets a warning and a no-op for that offset — the
    schema has no column for it. Other offsets in the same call
    still land.
    """
    ny = timezone(timedelta(hours=-4))
    decision_utc = datetime(2026, 5, 13, 15, 50, 0, tzinfo=ny).astimezone(_UTC)
    rec = _new_recorder("replay", tmp_path, replay_date="2026-05-13")
    rec.record_submission(
        leg="entry", symbol="AAA", side="BUY", requested_qty=1,
        order_type="MKT", order_id="oid-1",
    )
    rec.record_fill(
        leg="entry", symbol="AAA", order_id="oid-1",
        filled_qty=1, fill_price=100.0, status="Filled",
    )
    # Include valid n=3 alongside invalid n=6.
    rec.record_marks(
        marks_by_offset_minutes={
            3: {"AAA": (100.30, decision_utc + timedelta(minutes=3))},
            6: {"AAA": (100.60, decision_utc + timedelta(minutes=6))},
        },
        target_ts_by_offset_minutes={
            3: decision_utc + timedelta(minutes=3),
            6: decision_utc + timedelta(minutes=6),
        },
    )
    rec.finalize()
    row = pl.read_parquet(rec.final_path).row(0, named=True)
    assert row["t3min_price"] == pytest.approx(100.30)
    # And no t6min_* column exists at all — schema is closed.
    assert "t6min_price" not in row


def test_marks_walkback_no_lookahead_across_marks(tmp_path):
    """Load-bearing: even though the background thread CAPTURES at
    T+MAX (~15:55), each mark's price reflects only ticks at or
    before ITS own target. Construct a tbbo where a big price
    jump happens at T+3+delta, and check that T+1, T+2 do NOT
    see the jumped price.
    """
    ny = timezone(timedelta(hours=-4))
    decision_utc = datetime(2026, 5, 13, 15, 50, 0, tzinfo=ny).astimezone(_UTC)
    rows = [
        {"ts_event": decision_utc + timedelta(seconds=30),
         "symbol": "AAA", "price": 100.10, "size": 100,
         "ask_px_00": 100.105, "bid_px_00": 100.095},
        # BIG jump at T+3+1s — T+1 and T+2 must NOT pick this up.
        {"ts_event": decision_utc + timedelta(minutes=3, seconds=1),
         "symbol": "AAA", "price": 150.0, "size": 100,
         "ask_px_00": 150.005, "bid_px_00": 149.995},
        {"ts_event": decision_utc + timedelta(minutes=4, seconds=30),
         "symbol": "AAA", "price": 155.0, "size": 100,
         "ask_px_00": 155.005, "bid_px_00": 154.995},
    ]
    tbbo = pl.DataFrame(rows)
    rec = _new_recorder("replay", tmp_path, replay_date="2026-05-13")
    rec.record_submission(
        leg="entry", symbol="AAA", side="BUY", requested_qty=1,
        order_type="MKT", order_id="oid-1",
    )
    rec.record_fill(
        leg="entry", symbol="AAA", order_id="oid-1",
        filled_qty=1, fill_price=100.0, status="Filled",
    )
    rec.capture_marks_from_tbbo(tbbo, decision_utc)
    rec.finalize()
    row = pl.read_parquet(rec.final_path).row(0, named=True)
    # T+1, T+2, T+3 marks see ONLY the pre-jump tick at T+30s.
    for n in (1, 2, 3):
        assert row[f"t{n}min_price"] == pytest.approx(100.10), (
            f"t{n}min_price = {row[f't{n}min_price']} — the "
            f"post-T+3 jump leaked past its own target (look-ahead bug)"
        )
    # T+4, T+5 see the post-jump ticks.
    assert row["t4min_price"] == pytest.approx(150.0)
    assert row["t5min_price"] == pytest.approx(155.0)


def test_marks_naive_per_sym_ts_skips_that_sym_only(tmp_path):
    """Round 4 finding #10b: a naive ``observed_ts`` on ONE symbol
    must not abort the whole offset's other symbols — the target
    path handles a naive target by warn+skipping per offset, and
    the sym loop must do the same per sym for symmetry.
    """
    ny = timezone(timedelta(hours=-4))
    decision_utc = datetime(2026, 5, 13, 15, 50, 0, tzinfo=ny).astimezone(_UTC)
    good_ts = decision_utc + timedelta(minutes=3)
    naive_ts = datetime(2026, 5, 13, 15, 53, 0)  # no tzinfo
    rec = _new_recorder("paper", tmp_path)
    for sym, base in (("A", 100.0), ("B", 50.0)):
        rec.record_submission(
            leg="entry", symbol=sym, side="BUY", requested_qty=1,
            order_type="MKT", order_id=f"o-{sym}",
        )
        rec.record_fill(
            leg="entry", symbol=sym, order_id=f"o-{sym}",
            filled_qty=1, fill_price=base, status="Filled",
        )
    rec.record_marks(
        marks_by_offset_minutes={
            3: {
                "A": (100.5, naive_ts),   # skip this sym
                "B": (50.5, good_ts),     # this MUST still land
            },
        },
        target_ts_by_offset_minutes={
            3: decision_utc + timedelta(minutes=3),
        },
    )
    rec.finalize()
    df = pl.read_parquet(rec.final_path)
    a = df.filter(pl.col("symbol") == "A").row(0, named=True)
    b = df.filter(pl.col("symbol") == "B").row(0, named=True)
    assert a["t3min_price"] is None
    assert b["t3min_price"] == pytest.approx(50.5)


def test_marks_journal_omits_dropped_offsets(tmp_path):
    """Round 4 finding #10a: an offset whose target ts was naive
    (skipped on the row side) must NOT show up in the journal's
    ``marks_by_offset_minutes`` map. The journal must reflect what
    actually landed on rows, not what was passed in.
    """
    ny = timezone(timedelta(hours=-4))
    decision_utc = datetime(2026, 5, 13, 15, 50, 0, tzinfo=ny).astimezone(_UTC)
    naive_target = datetime(2026, 5, 13, 15, 52, 0)  # no tzinfo -> skip
    good_target = decision_utc + timedelta(minutes=5)
    rec = _new_recorder("paper", tmp_path)
    rec.record_marks(
        marks_by_offset_minutes={
            2: {"AAA": (99.5, decision_utc + timedelta(minutes=2))},
            5: {"AAA": (100.5, decision_utc + timedelta(minutes=5))},
        },
        target_ts_by_offset_minutes={
            2: naive_target,   # naive -> whole offset skipped
            5: good_target,
        },
    )
    events = []
    with rec.journal_path.open("r") as fh:
        for line in fh:
            events.append(json.loads(line))
    marks_events = [e for e in events if e["kind"] == "marks"]
    assert marks_events
    ev = marks_events[0]
    # Offset 2 was dropped (naive target) — must not appear in
    # marks_by_offset_minutes.
    assert "2" not in ev["marks_by_offset_minutes"], (
        "journal recorded a mark for offset 2 whose target was "
        "dropped on the row side — journal MUST reflect what landed"
    )
    # Offset 5 landed correctly.
    assert "5" in ev["marks_by_offset_minutes"]
    assert ev["marks_by_offset_minutes"]["5"]["AAA"]["mid"] == 100.5


def test_marks_use_mid_not_last_trade_price(tmp_path):
    """Mid-vs-price basis regression guard. Every OTHER fixture in
    this file builds a quote that's symmetric around ``price``, so
    ``mid == price`` identically — a bug that substituted
    ``pl.col('price')`` for ``(ask + bid) / 2`` in the recorder
    would pass everywhere else. This test uses an INTENTIONALLY
    asymmetric quote so the two differ.

    Setup: last-trade price 100.00 (a downtick), bid 99.90, ask
    100.30 → mid 100.10 (the recorder MUST record the 100.10, not
    the 100.00). The impact-decay curve is comparable to
    ``arrival_price`` — also a mid — so mixing bases would silently
    corrupt the calibration.
    """
    ny = timezone(timedelta(hours=-4))
    decision_utc = datetime(2026, 5, 13, 15, 50, 0, tzinfo=ny).astimezone(_UTC)
    # One tick per mark, all asymmetric: mid != price.
    rows = []
    for n in (1, 2, 3, 4, 5):
        rows.append({
            "ts_event": decision_utc + timedelta(minutes=n),
            "symbol": "ASYM",
            "price": 100.0 + n * 0.10,       # last-trade (downtick pattern)
            "size": 100,
            "bid_px_00": 99.90 + n * 0.10,
            "ask_px_00": 100.30 + n * 0.10,  # mid = 100.10 + n*0.10
        })
    tbbo = pl.DataFrame(rows)
    rec = _new_recorder("replay", tmp_path, replay_date="2026-05-13")
    rec.record_submission(
        leg="entry", symbol="ASYM", side="BUY", requested_qty=1,
        order_type="MKT", order_id="oid-asym",
    )
    rec.record_fill(
        leg="entry", symbol="ASYM", order_id="oid-asym",
        filled_qty=1, fill_price=100.10, status="Filled",
    )
    rec.capture_marks_from_tbbo(tbbo, decision_utc)
    rec.finalize()
    row = pl.read_parquet(rec.final_path).row(0, named=True)
    for n in (1, 2, 3, 4, 5):
        expected_mid = 100.10 + n * 0.10
        expected_price = 100.0 + n * 0.10
        assert row[f"t{n}min_price"] == pytest.approx(expected_mid), (
            f"t{n}min_price = {row[f't{n}min_price']}; expected mid "
            f"{expected_mid}, NOT last-trade price {expected_price}"
        )
        # And explicitly pin: NOT the price column.
        assert row[f"t{n}min_price"] != pytest.approx(expected_price)


def test_marks_mixed_liquidity_thin_sym_stays_null(tmp_path):
    """Two symbols in the same snapshot: LIQU ticks every minute,
    SILENT has no ticks in the whole window. SILENT's marks must
    stay NULL rather than forward-fill from ``arrival_price``.

    Analyst rule: null means "no price to observe at this offset";
    the decay calibration MUST exclude such rows for that offset,
    not carry the arrival mid as a fake decay observation (which
    would pull decay estimates toward zero).
    """
    ny = timezone(timedelta(hours=-4))
    decision_utc = datetime(2026, 5, 13, 15, 50, 0, tzinfo=ny).astimezone(_UTC)
    rows = []
    for n in (1, 2, 3, 4, 5):
        rows.append({
            "ts_event": decision_utc + timedelta(minutes=n),
            "symbol": "LIQU",
            "price": 100.0 + n * 0.10, "size": 100,
            "bid_px_00": 99.90 + n * 0.10,
            "ask_px_00": 100.30 + n * 0.10,
        })
    tbbo = pl.DataFrame(rows)
    rec = _new_recorder("replay", tmp_path, replay_date="2026-05-13")
    # SILENT has an ARRIVAL price on record (from a stale
    # pre-decision tick), but no ticks in the walk-back window.
    rec.record_arrival_prices({
        "SILENT": {
            "bid": 41.99, "ask": 42.01, "mid": 42.0,
            "rel_spread": 0.00048,
            "observed_ts_utc": decision_utc - timedelta(minutes=10),
        },
    })
    for sym, base in (("LIQU", 100.0), ("SILENT", 42.0)):
        rec.record_submission(
            leg="entry", symbol=sym, side="BUY", requested_qty=1,
            order_type="MKT", order_id=f"oid-{sym}",
        )
        rec.record_fill(
            leg="entry", symbol=sym, order_id=f"oid-{sym}",
            filled_qty=1, fill_price=base, status="Filled",
        )
    rec.capture_marks_from_tbbo(tbbo, decision_utc)
    rec.finalize()
    df = pl.read_parquet(rec.final_path)
    liqu_row = df.filter(pl.col("symbol") == "LIQU").row(0, named=True)
    silent_row = df.filter(pl.col("symbol") == "SILENT").row(0, named=True)
    # LIQU: every mark populated.
    for n in (1, 2, 3, 4, 5):
        assert liqu_row[f"t{n}min_price"] == pytest.approx(100.10 + n * 0.10)
    # SILENT: every mark is null; arrival is still there.
    assert silent_row["arrival_price"] == pytest.approx(42.0)
    for n in (1, 2, 3, 4, 5):
        assert silent_row[f"t{n}min_price"] is None, (
            f"SILENT t{n}min_price = {silent_row[f't{n}min_price']} — "
            f"marks must stay null for a symbol with no ticks in the "
            f"walk-back window; forward-filling arrival would bias "
            f"decay estimates toward zero"
        )
        assert silent_row[f"t{n}min_ts_utc"] is None
        assert silent_row[f"t{n}min_staleness_secs"] is None


def test_read_trades_backfills_v3_parquets_as_typed_nulls(tmp_path):
    """The cross-vintage reader must give a v3 file typed nulls for
    the v4-only columns rather than silently dropping them (which
    ``pyarrow.dataset(...).to_table()`` without ``schema=`` does,
    depending on which fragment happens to be discovered first).

    Simulate a v3 parquet by dropping the T+1..T+4 columns from a
    fresh v4 write, then read a mixed [v3, v4] glob and confirm
    all 61 v4 columns show up.
    """
    # Build a v4 parquet and a v3-shaped one (v4 columns dropped).
    # Tick at decision so ALL five marks resolve (T+1..T+5 all >= tick).
    ny = timezone(timedelta(hours=-4))
    decision_utc = datetime(2026, 5, 13, 15, 50, 0, tzinfo=ny).astimezone(_UTC)
    tbbo = pl.DataFrame([{
        "ts_event": decision_utc,
        "symbol": "AAA",
        "price": 40.0, "size": 100,
        "bid_px_00": 39.99, "ask_px_00": 40.01,
    }])
    rec_v4 = _new_recorder(
        "paper", tmp_path, run_id="run-v4",
    )
    rec_v4.record_submission(
        leg="entry", symbol="AAA", side="BUY", requested_qty=1,
        order_type="MKT", order_id="oid-v4",
    )
    rec_v4.record_fill(
        leg="entry", symbol="AAA", order_id="oid-v4",
        filled_qty=1, fill_price=40.0, status="Filled",
    )
    rec_v4.capture_marks_from_tbbo(tbbo, decision_utc)
    v4_path = rec_v4.finalize()
    assert v4_path is not None

    # Simulate a v3 parquet: read the v4 file, drop the four
    # v4-only mark blocks, rewrite. This reproduces the layout a
    # SCHEMA_VERSION=3 recorder would have written on disk.
    v4_table = pq.read_table(v4_path)
    v3_only_columns = [
        c for c in v4_table.column_names
        if not any(c.startswith(f"t{n}min_") for n in (1, 2, 3, 4))
    ]
    v3_table = v4_table.select(v3_only_columns)
    # Stamp schema_version=3 to mark the vintage — this is how the
    # read side distinguishes v3 from v4 rows in the combined table.
    v3_table = v3_table.set_column(
        v3_table.column_names.index("schema_version"),
        "schema_version",
        pa.array([3] * v3_table.num_rows, type=pa.int32()),
    )
    v3_path = tmp_path / "paper" / "2026-05-13" / "run-v3.parquet"
    pq.write_table(v3_table, v3_path)

    # The real assertion: read_trades gives every v4 column, back-
    # filled as typed nulls on the v3 rows.
    combined = trade_recorder.read_trades([v3_path, v4_path])
    for c in trade_recorder.TRADE_RECORD_SCHEMA.names:
        assert c in combined.column_names, (
            f"read_trades dropped v4 column {c}"
        )
    # The v3 row's new-column values are typed nulls (not string
    # 'None', not skipped). The v4 row's t1min_price came from the
    # capture — check it's still there.
    df = pl.from_arrow(combined).sort("schema_version")
    assert df.height == 2
    v3_row = df.filter(pl.col("schema_version") == 3).row(0, named=True)
    v4_row = df.filter(pl.col("schema_version") == 4).row(0, named=True)
    assert v3_row["t1min_price"] is None
    assert v4_row["t1min_price"] is not None
    # And the column keeps its declared pyarrow type.
    for n in (1, 2, 3, 4, 5):
        assert combined.schema.field(f"t{n}min_price").type.equals(
            trade_recorder.TRADE_RECORD_SCHEMA.field(f"t{n}min_price").type
        )


# ---------------------------------------------------------------------------


def test_real_live_logs_trades_dir_untouched_by_this_module():
    """Check against the REAL trades dir (captured at module load)
    — the conftest fixture patches ``config.LIVE_LOG_DIR`` to
    tmp_path, so ``config.LIVE_LOG_DIR / 'trades'`` at test-body
    time would be a per-test tmp path, not the real one.
    """
    real = _REAL_LIVE_LOG_DIR / "trades"
    if not real.exists():
        return
    current = {p for p in real.rglob("*") if p.is_file()}
    added = current - _REAL_TRADES_BASELINE
    assert not added, (
        f"tests wrote to the real production trades dir: {sorted(added)}"
    )


def test_real_live_logs_dir_log_files_not_written_by_tests():
    """Round 4 finding #8: the autouse conftest fixture patches
    ``config.LIVE_LOG_DIR`` AND clears the ``_JSON_FORMAT_LOGGERS``
    cache so ``setup_logging`` rebuilds its ``TimedRotatingFileHandler``
    at the per-test tmp_path. If that plumbing regresses, the
    integration tests silently append ~300 KB of orchestrator noise
    to the REAL ``live/logs/*.log`` files on every run.

    Byte-length diff catches the leak; ``set(rglob)`` diffing does
    not (the files already exist in production, so the leak is a
    size delta on the same file, not a new file).
    """
    if not _REAL_LIVE_LOG_DIR.exists():
        return
    current = _snapshot_real_logs_files_before_tests()
    grew: list[str] = []
    for p, sz in current.items():
        baseline = _REAL_LOG_SIZES_BASELINE.get(p)
        if baseline is None:
            # A NEW .log file appeared under the real dir during the
            # test suite — direct leak.
            grew.append(f"new file {p} ({sz} bytes)")
        elif sz > baseline:
            grew.append(
                f"{p}: {baseline} -> {sz} bytes "
                f"(+{sz - baseline})"
            )
    assert not grew, (
        "tests wrote to real production live/logs/ .log files:\n"
        + "\n".join(grew)
        + "\n\nCheck that live/tests/conftest.py's autouse fixture "
          "patches config.LIVE_LOG_DIR AND clears "
          "config._JSON_FORMAT_LOGGERS so setup_logging rebuilds "
          "the FileHandler at tmp_path."
    )
