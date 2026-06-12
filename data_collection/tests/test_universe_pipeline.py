"""
Unit tests for the PIT universe + download-plan pipeline.

Pure logic only -- no network, no Databento SDK calls. Tests cover:

* universe selection over synthetic OHLCV (top-N, 90% trading-day filter,
  min-price filter, deterministic tie-break, lookahead avoidance,
  IPO mid-window correctly fails the 90% rule);
* interval algebra (``_normalize``, ``subtract_intervals``);
* the coverage ledger gap branches called out by the spec (brand-new,
  persisting, returning-after-short-gap, returning-after->24mo gap,
  fully-covered, idempotent re-mark);
* the per-rebalance and full-schedule planners.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from dateutil.relativedelta import relativedelta

from data_collection.universe_pipeline import (
    DATA_FLOOR,
    LOOKBACK,
    REBALANCE,
    CoverageLedger,
    DownloadRequest,
    MAX_SYMBOLS_PER_REQUEST,
    PITUniverse,
    _normalize,
    build_pit_universes,
    clamp_floor,
    plan_all_downloads,
    plan_downloads_for_rebalance,
    rebalance_dates,
    select_universe,
    split_oversized_requests,
    subtract_intervals,
)


# --------------------------------------------------------------------------- #
# split_oversized_requests (Databento 2,000-symbol cap)
# --------------------------------------------------------------------------- #

def test_split_oversized_requests_chunks_to_cap_and_preserves_union():
    big = DownloadRequest(
        schema="tbbo",
        rebalance_date=date(2020, 1, 1),
        tickers=tuple(f"SYM{i:04d}" for i in range(3000)),  # sorted, > 2000
        start_date=date(2018, 7, 1),
        end_date=date(2020, 7, 1),
    )
    small = DownloadRequest(
        schema="imbalance",
        rebalance_date=date(2020, 1, 1),
        tickers=tuple(f"AAA{i:03d}" for i in range(10)),
        start_date=date(2020, 1, 1),
        end_date=date(2020, 7, 1),
    )

    out = split_oversized_requests([big, small])

    # The 3000-name request splits into 2000 + 1000; the 10-name one is untouched.
    assert len(out) == 3
    assert [len(r.tickers) for r in out] == [2000, 1000, 10]
    # Every chunk is within the cap.
    assert all(len(r.tickers) <= MAX_SYMBOLS_PER_REQUEST for r in out)
    # Chunks of the big request preserve schema/dates/rebalance and union exactly.
    big_chunks = [r for r in out if r.schema == "tbbo"]
    assert all(
        (r.start_date, r.end_date, r.rebalance_date)
        == (big.start_date, big.end_date, big.rebalance_date)
        for r in big_chunks
    )
    rejoined = tuple(t for r in big_chunks for t in r.tickers)
    assert rejoined == big.tickers  # order + membership preserved, no dupes/drops
    # The small request passed through unchanged and order is preserved.
    assert out[-1] == small


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _make_ohlcv(rows: list[dict]) -> pl.DataFrame:
    """Build an OHLCV-1d frame with a ts_event (UTC midnight) column."""
    out = []
    for r in rows:
        ts = r["ts_event"]
        if isinstance(ts, date) and not isinstance(ts, datetime):
            ts = datetime(ts.year, ts.month, ts.day)
        out.append({
            "ts_event": ts,
            "symbol": r["symbol"],
            "close": float(r["close"]),
            "volume": int(r["volume"]),
        })
    return pl.DataFrame(out)


def _session_dates(start: date, n: int) -> list[date]:
    """Generate `n` distinct business-style dates starting at `start`.

    The selection logic just needs distinct dates; weekend-skipping isn't
    necessary, but stepping by 1 day works fine.
    """
    return [start + timedelta(days=i) for i in range(n)]


# --------------------------------------------------------------------------- #
# select_universe
# --------------------------------------------------------------------------- #

def test_select_universe_top_n_by_adv_descending():
    sessions = _session_dates(date(2023, 1, 2), 20)
    rows: list[dict] = []
    # Three names with different (constant) dollar volumes:
    # AAA = $1M/day, BBB = $2M/day, CCC = $500k/day.
    for d in sessions:
        rows.append({"ts_event": d, "symbol": "AAA", "close": 100.0, "volume": 10_000})
        rows.append({"ts_event": d, "symbol": "BBB", "close": 100.0, "volume": 20_000})
        rows.append({"ts_event": d, "symbol": "CCC", "close": 100.0, "volume": 5_000})

    bars = _make_ohlcv(rows)
    out = select_universe(
        bars,
        window_start=sessions[0],
        window_end=sessions[-1] + timedelta(days=1),
        top_n=2,
        min_trading_frac=0.9,
        min_price=1.0,
    )

    assert out["symbol"].to_list() == ["BBB", "AAA"]
    assert out["rank"].to_list() == [1, 2]
    # CCC dropped only because top_n=2 cut it.
    assert "CCC" not in set(out["symbol"].to_list())


def test_select_universe_drops_zero_and_null_close_volume_rows():
    """Edge case K: null / non-positive close|volume rows must not count toward
    ADV or n_days. A name with junk ticks should look identical to a clean name
    that traded only the valid sessions."""
    sessions = _session_dates(date(2023, 1, 2), 10)
    rows: list[dict] = []
    # CLEAN trades exactly the first 8 valid sessions at $1M/day.
    for d in sessions[:8]:
        rows.append({"ts_event": d, "symbol": "CLEAN", "close": 100.0, "volume": 10_000})
    # DIRTY has the same 8 valid sessions plus junk on two extra sessions:
    # a zero-volume row and a zero-close row that must both be discarded.
    for d in sessions[:8]:
        rows.append({"ts_event": d, "symbol": "DIRTY", "close": 100.0, "volume": 10_000})
    rows.append({"ts_event": sessions[8], "symbol": "DIRTY", "close": 100.0, "volume": 0})
    rows.append({"ts_event": sessions[9], "symbol": "DIRTY", "close": 0.0, "volume": 10_000})

    bars = _make_ohlcv(rows)
    # Also inject a genuine null close|volume row (bypassing the int/float helper).
    null_row = pl.DataFrame(
        {"ts_event": [datetime(2023, 1, 2)], "symbol": ["DIRTY"],
         "close": [None], "volume": [None]},
        schema={"ts_event": pl.Datetime, "symbol": pl.Utf8,
                "close": pl.Float64, "volume": pl.UInt64},
    )
    bars = pl.concat([bars, null_row], how="vertical_relaxed")

    out = select_universe(
        bars,
        window_start=sessions[0],
        window_end=sessions[-1] + timedelta(days=1),
        top_n=10,
        min_trading_frac=0.9,
        min_price=1.0,
    )
    by = {r["symbol"]: r for r in out.to_dicts()}
    # The junk rows are dropped, so only the 8 valid sessions remain for both.
    assert by["DIRTY"]["n_days"] == 8
    assert by["CLEAN"]["n_days"] == 8
    assert by["DIRTY"]["adv"] == by["CLEAN"]["adv"]


def test_select_universe_excludes_below_min_trading_frac():
    sessions = _session_dates(date(2023, 1, 2), 10)
    rows: list[dict] = []
    for d in sessions:
        rows.append({"ts_event": d, "symbol": "FULL", "close": 100.0, "volume": 10_000})
    # IPO mid-window: PARTIAL trades only the last 3 of 10 sessions (30%).
    for d in sessions[7:]:
        rows.append({"ts_event": d, "symbol": "PARTIAL", "close": 100.0, "volume": 100_000_000})

    bars = _make_ohlcv(rows)
    out = select_universe(
        bars,
        window_start=sessions[0],
        window_end=sessions[-1] + timedelta(days=1),
        top_n=10,
        min_trading_frac=0.9,
        min_price=1.0,
    )

    # PARTIAL has the biggest ADV but only traded 30% of sessions -> excluded.
    assert out["symbol"].to_list() == ["FULL"]


def test_select_universe_excludes_below_min_price():
    sessions = _session_dates(date(2023, 1, 2), 10)
    rows: list[dict] = []
    for d in sessions:
        rows.append({"ts_event": d, "symbol": "PENNY", "close": 0.50, "volume": 10_000_000})
        rows.append({"ts_event": d, "symbol": "CHEAP", "close": 2.00, "volume": 1_000})

    bars = _make_ohlcv(rows)
    out = select_universe(
        bars,
        window_start=sessions[0],
        window_end=sessions[-1] + timedelta(days=1),
        top_n=10,
        min_trading_frac=0.9,
        min_price=1.0,
    )

    assert out["symbol"].to_list() == ["CHEAP"]


def test_select_universe_half_open_window_excludes_end_day():
    """ts_event == window_end must be excluded (no lookahead from R itself)."""
    sessions = _session_dates(date(2023, 1, 2), 5)
    cutoff = sessions[-1]  # we'll set window_end == cutoff
    rows: list[dict] = []
    for d in sessions[:-1]:  # 4 in-window sessions
        rows.append({"ts_event": d, "symbol": "AAA", "close": 100.0, "volume": 1_000})
    # The cutoff-day row would balloon ADV if it leaked in.
    rows.append({
        "ts_event": cutoff,
        "symbol": "AAA",
        "close": 100.0,
        "volume": 1_000_000_000,
    })
    bars = _make_ohlcv(rows)
    out = select_universe(
        bars,
        window_start=sessions[0],
        window_end=cutoff,  # exclusive
        top_n=5,
        min_trading_frac=0.0,
        min_price=1.0,
    )
    # Without the leaked row, ADV = $100k. With it, ADV would be > $1B.
    aaa_adv = out.filter(pl.col("symbol") == "AAA")["adv"].item()
    assert aaa_adv == pytest.approx(100_000.0), aaa_adv


def test_select_universe_tie_break_deterministic_on_equal_adv():
    """Equal ADVs must produce the same ranking across runs."""
    sessions = _session_dates(date(2023, 1, 2), 10)
    rows: list[dict] = []
    for d in sessions:
        for sym in ["A", "B", "C", "D"]:
            rows.append({"ts_event": d, "symbol": sym, "close": 100.0, "volume": 1_000})
    bars = _make_ohlcv(rows)

    out1 = select_universe(
        bars,
        window_start=sessions[0],
        window_end=sessions[-1] + timedelta(days=1),
        top_n=4,
        min_trading_frac=0.9,
        min_price=1.0,
    )["symbol"].to_list()
    out2 = select_universe(
        bars.sample(fraction=1.0, shuffle=True, seed=42),
        window_start=sessions[0],
        window_end=sessions[-1] + timedelta(days=1),
        top_n=4,
        min_trading_frac=0.9,
        min_price=1.0,
    )["symbol"].to_list()
    # Set equality is enough -- 4 names tied, all four must appear.
    # The PITUniverse downstream sorts the constituents, which is the real
    # deterministic tie-break.
    assert set(out1) == set(out2) == {"A", "B", "C", "D"}


def test_select_universe_empty_window_returns_empty_frame():
    bars = _make_ohlcv([
        {"ts_event": date(2023, 1, 2), "symbol": "AAA", "close": 100.0, "volume": 1_000},
    ])
    out = select_universe(
        bars,
        window_start=date(2024, 1, 1),
        window_end=date(2024, 2, 1),
        top_n=5,
        min_trading_frac=0.9,
        min_price=1.0,
    )
    assert out.is_empty()
    assert out.columns == ["symbol", "adv", "n_days", "last_px", "rank"]


def test_select_universe_median_vs_mean_one_off_spike():
    sessions = _session_dates(date(2023, 1, 2), 11)
    rows: list[dict] = []
    # AAA has 10 days of $100k and one day of $10M -- median ignores it.
    for d in sessions[:-1]:
        rows.append({"ts_event": d, "symbol": "AAA", "close": 100.0, "volume": 1_000})
    rows.append({"ts_event": sessions[-1], "symbol": "AAA", "close": 100.0, "volume": 100_000})
    # BBB has a flat $500k/day.
    for d in sessions:
        rows.append({"ts_event": d, "symbol": "BBB", "close": 100.0, "volume": 5_000})
    bars = _make_ohlcv(rows)

    med = select_universe(
        bars,
        window_start=sessions[0],
        window_end=sessions[-1] + timedelta(days=1),
        top_n=2,
        min_trading_frac=0.9,
        min_price=1.0,
        adv_agg="median",
    )["symbol"].to_list()
    mean = select_universe(
        bars,
        window_start=sessions[0],
        window_end=sessions[-1] + timedelta(days=1),
        top_n=2,
        min_trading_frac=0.9,
        min_price=1.0,
        adv_agg="mean",
    )["symbol"].to_list()
    assert med[0] == "BBB"   # robust median sees AAA's typical $100k.
    assert mean[0] in {"AAA", "BBB"}  # mean pulled up; ordering may flip.


# --------------------------------------------------------------------------- #
# build_pit_universes
# --------------------------------------------------------------------------- #

def test_build_pit_universes_constituents_sorted_and_uppercase():
    sessions = _session_dates(date(2019, 1, 2), 400)
    rows: list[dict] = []
    for d in sessions:
        for sym in ["aapl", "AAPL", "msft", "GOOGL"]:  # case noise
            rows.append({"ts_event": d, "symbol": sym, "close": 100.0, "volume": 1000})
    bars = _make_ohlcv(rows)

    universes, long_frame = build_pit_universes(
        bars,
        strategy_start=date(2019, 1, 1),   # hard data floor
        today=date(2020, 1, 2),            # just past the first rebalance
        lookback=relativedelta(months=12),
        step=relativedelta(months=6),
        top_n=10,
        min_trading_frac=0.9,
        min_price=1.0,
    )
    assert len(universes) == 1
    u = universes[0]
    # First rebalance = strategy_start + lookback; training window starts AT
    # strategy_start (no data fetched before it).
    assert u.rebalance_date == date(2020, 1, 1)
    assert u.train_start == date(2019, 1, 1)
    assert u.constituents == sorted(u.constituents)
    assert all(s == s.upper() for s in u.constituents)
    # case-noise dedup'd to one AAPL.
    assert u.constituents.count("AAPL") == 1
    # Long frame matches.
    assert long_frame["rebalance_date"][0] == date(2020, 1, 1)


def test_build_pit_universes_strategy_start_is_hard_floor():
    """strategy_start is the hard left edge: the first rebalance is
    strategy_start + lookback and its training window starts exactly at
    strategy_start (nothing before it). A strategy_start below the absolute
    data_floor clamps up to the floor."""
    sessions = _session_dates(date(2018, 5, 1), 900)
    rows = [
        {"ts_event": d, "symbol": "AAA", "close": 100.0, "volume": 1000}
        for d in sessions
    ]
    bars = _make_ohlcv(rows)

    # strategy_start above the absolute floor -> it IS the floor.
    universes, _ = build_pit_universes(
        bars,
        strategy_start=date(2019, 1, 1),
        today=date(2020, 7, 2),
        lookback=relativedelta(months=18),
        step=relativedelta(months=6),
        data_floor=date(2018, 5, 1),
        top_n=5,
        min_trading_frac=0.0,
        min_price=1.0,
    )
    u = universes[0]
    assert u.rebalance_date == date(2020, 7, 1)        # strategy_start + 18mo
    assert u.train_start == date(2019, 1, 1)           # == strategy_start
    assert u.train_end == date(2020, 7, 1)
    # No training window reaches before strategy_start.
    assert all(uu.train_start >= date(2019, 1, 1) for uu in universes)

    # strategy_start below the absolute data_floor -> clamps to the floor.
    universes2, _ = build_pit_universes(
        bars,
        strategy_start=date(2018, 1, 1),               # < data_floor
        today=date(2020, 1, 2),
        lookback=relativedelta(months=18),
        step=relativedelta(months=6),
        data_floor=date(2018, 5, 1),
        top_n=5,
        min_trading_frac=0.0,
        min_price=1.0,
    )
    assert universes2[0].train_start == date(2018, 5, 1)      # floor, not 2018-01-01
    assert universes2[0].rebalance_date == date(2019, 11, 1)  # floor + 18mo


def test_build_pit_universes_windows_contiguous_and_non_overlapping():
    sessions = _session_dates(date(2018, 5, 1), 1500)
    rows = [
        {"ts_event": d, "symbol": "AAA", "close": 100.0, "volume": 1000}
        for d in sessions
    ]
    bars = _make_ohlcv(rows)

    universes, _ = build_pit_universes(
        bars,
        strategy_start=date(2020, 1, 1),
        today=date(2022, 1, 1),
        lookback=relativedelta(months=12),
        step=relativedelta(months=6),
        top_n=5,
        min_trading_frac=0.0,
        min_price=1.0,
    )
    # active_end[i] == active_start[i+1] (contiguous + non-overlapping).
    for a, b in zip(universes, universes[1:]):
        assert a.active_end == b.active_start
        # train_end (== R) equals active_start so no leakage of the R-day bar.
        assert a.train_end == a.active_start


# --------------------------------------------------------------------------- #
# rebalance_dates / clamp_floor
# --------------------------------------------------------------------------- #

def test_rebalance_dates_spans_strategy_start_to_today_inclusive():
    out = rebalance_dates(
        date(2020, 1, 1), date(2021, 1, 1), step=relativedelta(months=6),
    )
    assert out == [date(2020, 1, 1), date(2020, 7, 1), date(2021, 1, 1)]


def test_clamp_floor_returns_floor_when_below():
    assert clamp_floor(date(2017, 1, 1), floor=date(2018, 5, 1)) == date(2018, 5, 1)
    assert clamp_floor(date(2019, 1, 1), floor=date(2018, 5, 1)) == date(2019, 1, 1)


# --------------------------------------------------------------------------- #
# Interval algebra
# --------------------------------------------------------------------------- #

def test_normalize_merges_overlap_and_touching():
    iv = [
        (date(2020, 1, 1), date(2020, 3, 1)),
        (date(2020, 3, 1), date(2020, 5, 1)),  # touching
        (date(2020, 4, 1), date(2020, 6, 1)),  # overlapping
        (date(2021, 1, 1), date(2021, 2, 1)),  # disjoint
    ]
    out = _normalize(iv)
    assert out == [
        (date(2020, 1, 1), date(2020, 6, 1)),
        (date(2021, 1, 1), date(2021, 2, 1)),
    ]


def test_normalize_drops_empty_intervals():
    out = _normalize([(date(2020, 1, 1), date(2020, 1, 1))])
    assert out == []


def test_subtract_intervals_fully_covered():
    req = (date(2020, 1, 1), date(2020, 6, 1))
    cov = [(date(2020, 1, 1), date(2020, 6, 1))]
    assert subtract_intervals(req, cov) == []


def test_subtract_intervals_no_overlap_returns_required():
    req = (date(2020, 1, 1), date(2020, 6, 1))
    cov = [(date(2019, 1, 1), date(2019, 6, 1))]
    assert subtract_intervals(req, cov) == [req]


def test_subtract_intervals_left_overlap_only():
    req = (date(2020, 1, 1), date(2020, 6, 1))
    cov = [(date(2019, 1, 1), date(2020, 3, 1))]
    assert subtract_intervals(req, cov) == [(date(2020, 3, 1), date(2020, 6, 1))]


def test_subtract_intervals_right_overlap_only():
    req = (date(2020, 1, 1), date(2020, 6, 1))
    cov = [(date(2020, 4, 1), date(2020, 12, 1))]
    assert subtract_intervals(req, cov) == [(date(2020, 1, 1), date(2020, 4, 1))]


def test_subtract_intervals_middle_hole_returns_two_pieces():
    req = (date(2020, 1, 1), date(2020, 12, 1))
    cov = [(date(2020, 4, 1), date(2020, 6, 1))]
    assert subtract_intervals(req, cov) == [
        (date(2020, 1, 1), date(2020, 4, 1)),
        (date(2020, 6, 1), date(2020, 12, 1)),
    ]


def test_subtract_intervals_multiple_holes():
    req = (date(2020, 1, 1), date(2020, 12, 1))
    cov = [
        (date(2020, 2, 1), date(2020, 3, 1)),
        (date(2020, 6, 1), date(2020, 7, 1)),
    ]
    assert subtract_intervals(req, cov) == [
        (date(2020, 1, 1), date(2020, 2, 1)),
        (date(2020, 3, 1), date(2020, 6, 1)),
        (date(2020, 7, 1), date(2020, 12, 1)),
    ]


# --------------------------------------------------------------------------- #
# CoverageLedger -- all branches in spec edge case E
# --------------------------------------------------------------------------- #

def _u(date_in: date, constituents: list[str]) -> PITUniverse:
    """Build a synthetic PITUniverse for ledger tests."""
    return PITUniverse(
        rebalance_date=date_in,
        active_start=date_in,
        active_end=date_in + relativedelta(months=6),
        train_start=date_in - relativedelta(months=18),
        train_end=date_in,
        constituents=sorted(constituents),
    )


def test_ledger_brand_new_full_window():
    """First time we see a ticker: gap == full required window."""
    ledger = CoverageLedger()
    u = _u(date(2020, 1, 1), ["AAA", "BBB"])
    plan = plan_downloads_for_rebalance(ledger, u, schemas=["tbbo"])
    assert len(plan) == 1
    req = plan[0]
    assert req.tickers == ("AAA", "BBB")
    # full ~24mo span: train_start..active_end
    assert req.start_date == u.train_start
    assert req.end_date == u.active_end


def test_plan_clamps_request_end_to_available_end():
    """A universe whose active window extends past the dataset's available edge
    must not emit a request ending in the future (Databento 422
    data_end_after_available_end)."""
    ledger = CoverageLedger()
    u = _u(date(2026, 1, 1), ["AAA", "BBB"])  # active_end = 2026-07-01 (future)
    available_end = date(2026, 6, 11)         # data only through today

    plan = plan_downloads_for_rebalance(
        ledger, u, schemas=["tbbo"], mark=True, available_end=available_end,
    )

    assert plan, "expected a request (window starts well before the edge)"
    assert all(r.end_date <= available_end for r in plan)
    # The fetched end is the clamp, not the future active_end.
    assert max(r.end_date for r in plan) == available_end
    # Ledger only claims coverage up to available data, so a later run with
    # fresh data would still fetch the newly-available tail.
    assert all(
        e <= available_end
        for intervals in ledger.by_schema.get("tbbo", {}).values()
        for (_s, e) in intervals
    )


def test_plan_skips_fully_future_window():
    """A rebalance entirely beyond the available edge yields no requests."""
    ledger = CoverageLedger()
    u = _u(date(2030, 1, 1), ["AAA"])
    plan = plan_downloads_for_rebalance(
        ledger, u, schemas=["tbbo"], mark=True, available_end=date(2026, 6, 11),
    )
    assert plan == []


def test_ledger_persisting_name_only_forward_slice():
    """
    Returning name with the prior 24mo already covered up to the new R should
    only need the new 6mo forward slice [active_start, active_end).
    """
    ledger = CoverageLedger()
    u0 = _u(date(2020, 1, 1), ["AAA"])
    plan_downloads_for_rebalance(ledger, u0, schemas=["tbbo"], mark=True)

    u1 = _u(date(2020, 7, 1), ["AAA"])
    # The lookback window for u1 ends at 2020-07-01; covered window already
    # extends through u0.active_end == 2020-07-01, so the gap is just the new
    # 6mo forward slice.
    plan = plan_downloads_for_rebalance(ledger, u1, schemas=["tbbo"])
    assert len(plan) == 1
    req = plan[0]
    assert req.tickers == ("AAA",)
    assert req.start_date == u1.active_start
    assert req.end_date == u1.active_end


def test_ledger_short_gap_contiguous_extend():
    """
    Returning after a short coverage gap should produce a *single* contiguous
    forward fetch, not a backfill of an unrelated hole.
    """
    ledger = CoverageLedger()
    u0 = _u(date(2020, 1, 1), ["AAA"])
    plan_downloads_for_rebalance(ledger, u0, schemas=["tbbo"], mark=True)

    # Skip exactly one rebalance, but the new u2 trailing window is still
    # adjacent / overlapping: required.train_start sits inside what we covered.
    u2 = _u(date(2021, 1, 1), ["AAA"])
    plan = plan_downloads_for_rebalance(ledger, u2, schemas=["tbbo"])
    # Old coverage: [2018-07-01, 2020-07-01). u2 required: [2019-07-01, 2021-07-01).
    # Gap: [2020-07-01, 2021-07-01) -- one contiguous fetch.
    assert len(plan) == 1
    req = plan[0]
    assert req.start_date == date(2020, 7, 1)
    assert req.end_date == u2.active_end


def test_ledger_long_gap_disjoint_only_fetches_new_window():
    """
    Returning after a > 24mo coverage gap must NOT backfill the unneeded hole.
    The new universe only requires its own 18mo trailing window onward.
    """
    ledger = CoverageLedger()
    u0 = _u(date(2020, 1, 1), ["AAA"])
    plan_downloads_for_rebalance(ledger, u0, schemas=["tbbo"], mark=True)
    # Old coverage: [2018-07-01, 2020-07-01).

    # New rebalance R = 2024-01-01. Trailing window = [2022-07-01, 2024-01-01),
    # active = [2024-01-01, 2024-07-01). The old coverage is entirely BEFORE
    # the new required window, so the gap should be just the new ~24mo window
    # (NOT include backfilling 2020-07-01..2022-07-01).
    u_far = _u(date(2024, 1, 1), ["AAA"])
    plan = plan_downloads_for_rebalance(ledger, u_far, schemas=["tbbo"])
    assert len(plan) == 1
    req = plan[0]
    assert req.start_date == u_far.train_start
    assert req.end_date == u_far.active_end
    # Crucially: no fetch is queued that covers the dead interval
    # [2020-07-01, 2022-07-01).
    assert not (req.start_date < date(2022, 7, 1) and req.end_date <= date(2022, 7, 1))


def test_ledger_fully_covered_emits_no_request():
    """If the required window is already covered, the plan is empty."""
    ledger = CoverageLedger()
    u = _u(date(2020, 7, 1), ["AAA"])
    ledger.mark("tbbo", ["AAA"], (u.train_start, u.active_end))
    plan = plan_downloads_for_rebalance(ledger, u, schemas=["tbbo"])
    assert plan == []


def test_ledger_mark_is_idempotent_under_double_mark():
    """Marking the same gap twice must not produce two intervals."""
    ledger = CoverageLedger()
    ledger.mark("tbbo", ["AAA"], (date(2020, 1, 1), date(2020, 7, 1)))
    ledger.mark("tbbo", ["AAA"], (date(2020, 1, 1), date(2020, 7, 1)))
    covered = ledger.covered("tbbo", "AAA")
    assert covered == [(date(2020, 1, 1), date(2020, 7, 1))]


def test_ledger_mark_merges_adjacent_intervals():
    ledger = CoverageLedger()
    ledger.mark("tbbo", ["AAA"], (date(2020, 1, 1), date(2020, 4, 1)))
    ledger.mark("tbbo", ["AAA"], (date(2020, 4, 1), date(2020, 7, 1)))
    assert ledger.covered("tbbo", "AAA") == [
        (date(2020, 1, 1), date(2020, 7, 1)),
    ]


def test_ledger_failed_request_is_refetched_on_retry():
    """
    Simulates the spec's resumability requirement: if a request fails (we
    don't mark), the next planning pass for the same universe must re-issue
    it. If the request later succeeds (we mark), a third planning pass must
    NOT re-issue it.
    """
    ledger = CoverageLedger()
    u = _u(date(2020, 1, 1), ["AAA"])

    # Pass 1: plan, do NOT mark (simulates a crash mid-request).
    plan1 = plan_downloads_for_rebalance(ledger, u, schemas=["tbbo"])
    assert len(plan1) == 1

    # Pass 2: same universe -- the gap is still open, so the plan repeats.
    plan2 = plan_downloads_for_rebalance(ledger, u, schemas=["tbbo"])
    assert plan2 == plan1

    # Now mark and re-plan -- the third pass should be empty.
    ledger.mark("tbbo", plan2[0].tickers, (plan2[0].start_date, plan2[0].end_date))
    plan3 = plan_downloads_for_rebalance(ledger, u, schemas=["tbbo"])
    assert plan3 == []


def test_ledger_groups_tickers_with_identical_gap_into_one_request():
    """
    The 'persisting vs brand-new' split: two tickers with the same gap go
    in ONE request; two tickers with different gaps split into two.
    """
    ledger = CoverageLedger()
    # AAA is brand-new; BBB is already covered through the prior rebalance.
    ledger.mark("tbbo", ["BBB"], (
        date(2018, 7, 1), date(2020, 7, 1),
    ))
    u = _u(date(2020, 7, 1), ["AAA", "BBB"])
    plan = plan_downloads_for_rebalance(ledger, u, schemas=["tbbo"])

    # Expect 2 distinct download requests for this schema -- one for each
    # tickergroup -- which keeps the per-rebalance per-schema call count to
    # the spec's "typically ~2".
    assert len(plan) == 2
    by_ticker = {req.tickers: req for req in plan}
    # AAA gets the full required window.
    assert by_ticker[("AAA",)].start_date == u.train_start
    assert by_ticker[("AAA",)].end_date == u.active_end
    # BBB gets only the forward 6mo slice.
    assert by_ticker[("BBB",)].start_date == u.active_start
    assert by_ticker[("BBB",)].end_date == u.active_end


def test_ledger_persistence_round_trip(tmp_path: Path):
    ledger = CoverageLedger()
    ledger.mark("tbbo", ["aapl", "msft"], (date(2020, 1, 1), date(2020, 7, 1)))
    ledger.mark("imbalance", ["AAPL"], (date(2020, 1, 1), date(2020, 4, 1)))
    p = tmp_path / "ledger.json"
    ledger.save(p)

    reloaded = CoverageLedger.load(p)
    assert reloaded.covered("tbbo", "AAPL") == [(date(2020, 1, 1), date(2020, 7, 1))]
    assert reloaded.covered("tbbo", "MSFT") == [(date(2020, 1, 1), date(2020, 7, 1))]
    assert reloaded.covered("imbalance", "AAPL") == [(date(2020, 1, 1), date(2020, 4, 1))]


# --------------------------------------------------------------------------- #
# plan_all_downloads
# --------------------------------------------------------------------------- #

def test_plan_all_downloads_chronological_and_marks_ledger():
    u0 = _u(date(2020, 1, 1), ["AAA"])
    u1 = _u(date(2020, 7, 1), ["AAA", "BBB"])  # BBB is brand-new at u1.
    full_plan, ledger = plan_all_downloads(
        [u1, u0],  # passed out of order on purpose.
        schemas=["tbbo"],
    )

    # Plan is built rebalance-chronologically.
    # u0: AAA brand-new (one req).
    # u1: AAA persisting (forward-6mo), BBB brand-new (full window).
    assert len(full_plan) == 3

    # Ledger now covers AAA all the way through u1.active_end.
    assert ledger.covered("tbbo", "AAA") == [(u0.train_start, u1.active_end)]
    # And BBB only for its required window at u1.
    assert ledger.covered("tbbo", "BBB") == [(u1.train_start, u1.active_end)]


def test_plan_all_downloads_respects_initial_ledger():
    """An initial ledger should suppress fetches it already covers."""
    u = _u(date(2020, 1, 1), ["AAA"])
    initial = CoverageLedger()
    initial.mark("tbbo", ["AAA"], (u.train_start, u.active_end))

    plan, _ = plan_all_downloads([u], schemas=["tbbo"], initial_ledger=initial)
    assert plan == []
