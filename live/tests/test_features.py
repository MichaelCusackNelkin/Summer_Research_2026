"""Tests for feature engineering. Uses tiny synthetic frames — no network."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import polars as pl
import pytest

from live import features


def _tbbo_rows_for_day(day: str, symbol: str, mids: list[tuple[str, float, float]]):
    """Helper: build tbbo dict rows for one symbol on one day.

    ``mids`` is a list of (HH:MM:SS, mid_price, spread). Ask = mid+spread/2,
    bid = mid-spread/2, trade price = mid.
    """
    out = []
    for hms, mid, spread in mids:
        ts = datetime.fromisoformat(f"{day}T{hms}-04:00")
        out.append({
            "ts_event": ts.astimezone(timezone.utc),
            "symbol": symbol,
            "price": mid,
            "size": 100,
            "ask_px_00": mid + spread / 2,
            "bid_px_00": mid - spread / 2,
        })
    return out


def _imb_rows(day: str, records: list[tuple[str, str, int, int, float]]):
    """``records`` = (symbol, side, total_imb, paired, ref_price)."""
    ts = datetime.fromisoformat(f"{day}T15:50:01-04:00").astimezone(timezone.utc)
    return [
        {
            "ts_event": ts,
            "symbol": sym,
            "side": side,
            "total_imbalance_qty": tot,
            "paired_qty": paired,
            "ref_price": ref,
        }
        for (sym, side, tot, paired, ref) in records
    ]


def test_imbalance_records_to_frame_empty():
    frame = features.imbalance_records_to_frame([])
    assert frame.is_empty()
    assert set(frame.columns) == {"ts_event", "symbol", "side", "total_imbalance_qty", "paired_qty", "ref_price"}


def test_tbbo_records_to_frame_empty():
    frame = features.tbbo_records_to_frame([])
    assert frame.is_empty()
    assert set(frame.columns) == {"ts_event", "symbol", "price", "size", "ask_px_00", "bid_px_00"}


def test_norm_rank_normalises_to_pm_one():
    # Two names on same day so ranks are 1,2 -> normalized to -1, +1
    df = pl.DataFrame({
        "date": [datetime.fromisoformat("2026-06-15").date()] * 2,
        "symbol": ["AAA", "BBB"],
        "net_signed_imb": [-1.0, 1.0],
    })
    out = df.with_columns(features._norm_rank_expr("net_signed_imb").alias("nr"))
    vals = sorted(out["nr"].to_list())
    assert vals == [-1.0, 1.0]


def _single_day_live_scaffold(target: str = "2026-06-15"):
    """The exact shape the live orchestrator sees at 15:50: ONE day of tbbo,
    ONE 15:50 imbalance print. No warm-up rows. Returns
    (tbbo_frame, imb_frame, reference_stats).
    """
    syms_bases = [("AAA", 100.0), ("BBB", 50.0), ("CCC", 200.0)]
    tbbo_records = []
    for sym, base in syms_bases:
        spread = 0.02 * base / 100
        tbbo_records.extend(_tbbo_rows_for_day(target, sym, [
            ("15:40:00", base, spread),
            ("15:45:00", base * 1.001, spread),
            ("15:49:58", base * 1.002, spread),
            ("15:50:30", base * 1.0025, spread),
        ]))
    tbbo = features.tbbo_records_to_frame(tbbo_records)
    imb = features.imbalance_records_to_frame(_imb_rows(target, [
        ("AAA", "B", 100000, 500000, 100.2),
        ("BBB", "A", 50000, 200000, 50.1),
        ("CCC", "B", 30000, 300000, 200.4),
    ]))
    reference_stats = pl.DataFrame({
        "symbol": ["AAA", "BBB", "CCC"],
        "ADV": [1e9, 5e8, 2e9],
        "vol": [0.02, 0.03, 0.025],
    })
    return tbbo, imb, reference_stats


def _collect_pieces(tbbo, imb):
    return {
        "close_trades": features.compute_close_trades(tbbo.lazy()).collect(),
        "recent_return": features.compute_recent_return(tbbo.lazy()).collect(),
        "avg_daily_dollar_volume": features.compute_avg_daily_dollar_volume(tbbo.lazy()).collect(),
        "daily_vol": features.compute_daily_vol(tbbo.lazy()).collect(),
        "imb_print": features.compute_imbalance_print(imb.lazy()).collect(),
    }


def test_build_feature_frame_single_day_with_reference_stats():
    """C2 regression: single-day live buffer + reference_stats must produce
    a non-empty frame with finite ``auction_dollar_volume_ratio``.
    """
    target = "2026-06-15"
    tbbo, imb, ref_stats = _single_day_live_scaffold(target)
    pieces = _collect_pieces(tbbo, imb)

    frame = features.build_feature_frame(
        auction=None,
        reference_stats=ref_stats,
        **pieces,
    )
    assert frame.height == 3, f"expected 3 rows, got {frame.height}"
    for col in features.ALPHA_FEATURES:
        assert col in frame.columns, f"missing {col}"
    ratios = frame["auction_dollar_volume_ratio"].to_numpy()
    assert np.isfinite(ratios).all(), f"non-finite ratios: {ratios}"
    assert frame["ADV"].null_count() == 0
    assert frame["vol"].null_count() == 0


def test_build_feature_frame_single_day_without_reference_stats_is_empty():
    """C2 documentation: no reference_stats + single-day buffer = empty
    frame. The live orchestrator MUST pass reference_stats.
    """
    tbbo, imb, _ = _single_day_live_scaffold("2026-06-15")
    pieces = _collect_pieces(tbbo, imb)
    frame = features.build_feature_frame(auction=None, **pieces)
    assert frame.height == 0, (
        "single-day buffer with no reference_stats should drop every row "
        "(ADV/vol nulls). If this changed, verify the live orchestrator "
        "still passes reference_stats."
    )


def test_build_feature_frame_end_to_end():
    """Historical / retrain path: multi-day buffer, no reference_stats,
    causal EWMA vol non-null on the target day.
    """
    # Target day needs >= 2 preceding days so causal EWMA vol (shift 1 of
    # squared-returns EWMA) is non-null on the target day.
    target = "2026-06-15"
    warmups = ["2026-06-10", "2026-06-11", "2026-06-12"]
    tbbo_records = []
    syms_bases = [("AAA", 100.0), ("BBB", 50.0), ("CCC", 200.0)]

    for day_i, day in enumerate(warmups + [target]):
        for sym, base in syms_bases:
            base_i = base * (1 + 0.001 * day_i)
            spread = 0.02 * base / 100
            tbbo_records.extend(_tbbo_rows_for_day(day, sym, [
                ("15:40:00", base_i, spread),
                ("15:45:00", base_i * 1.001, spread),
                ("15:49:58", base_i * 1.002, spread),
                ("15:50:30", base_i * 1.0025, spread),
            ]))
    tbbo = features.tbbo_records_to_frame(tbbo_records)

    imb_records = _imb_rows(target, [
        ("AAA", "B", 100000, 500000, 100.2),
        ("BBB", "A", 50000, 200000, 50.1),
        ("CCC", "B", 30000, 300000, 200.4),
    ])
    imb = features.imbalance_records_to_frame(imb_records)

    close_trades = features.compute_close_trades(tbbo.lazy()).collect()
    recent_return = features.compute_recent_return(tbbo.lazy()).collect()
    adv = features.compute_avg_daily_dollar_volume(tbbo.lazy()).collect()
    vol = features.compute_daily_vol(tbbo.lazy()).collect()
    imb_print = features.compute_imbalance_print(imb.lazy()).collect()

    frame = features.build_feature_frame(
        imb_print=imb_print,
        close_trades=close_trades,
        avg_daily_dollar_volume=adv,
        daily_vol=vol,
        recent_return=recent_return,
    )
    assert not frame.is_empty()
    on_day = frame.filter(pl.col("date") == datetime.fromisoformat(target).date())
    assert on_day.height == 3
    for col in features.ALPHA_FEATURES:
        assert col in on_day.columns, f"missing {col}"
    assert "entry_price" in on_day.columns
    assert "spread_return" in on_day.columns
    assert on_day["ADV"].null_count() == 0
    assert on_day["vol"].null_count() == 0


def test_merge_reference_stats_overrides_nulls():
    day = "2026-06-15"
    features_df = pl.DataFrame({
        "symbol": ["AAA", "BBB"],
        "date": [datetime.fromisoformat(day).date()] * 2,
        "ADV": [None, 1234.0],
        "vol": [None, 0.01],
    }).with_columns(pl.col("symbol").cast(pl.Categorical))
    ref = pl.DataFrame({
        "symbol": ["AAA", "BBB"],
        "ADV": [5000.0, 9999.0],
        "vol": [0.02, 0.03],
    })
    merged = features.merge_reference_stats(features_df, ref)
    row_a = merged.filter(pl.col("symbol") == "AAA").to_dicts()[0]
    row_b = merged.filter(pl.col("symbol") == "BBB").to_dicts()[0]
    assert row_a["ADV"] == 5000.0  # null -> filled from ref
    assert row_a["vol"] == 0.02
    assert row_b["ADV"] == 1234.0  # existing non-null preserved
    assert row_b["vol"] == 0.01
