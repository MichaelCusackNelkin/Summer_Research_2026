"""Unit tests for LEG 1 LMT entry pricing and quote guards.

Covers:
- ``_round_to_tick``: US-equity tick regime, sign handling, half-up rounding.
- ``_validate_quote_for_entry``: skip on missing / crossed / stale / wide
  / non-positive quotes; accept normal quotes.
- ``_submit_limit_entries``: pricing off the mid, buy vs sell ordering,
  ACTIVE_ROUTING threading, skip → recorder ``error`` propagation,
  no MKT fallback, no broker call on skip.
- Integration: MOC exits remain MOC (untouched by the LMT changes).

Tests here EXPLICITLY monkey-patch the staleness threshold up (or
provide fresh timestamps) rather than rely on the conftest's
"disabled for tests" default, so they exercise the guards intended
behavior.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import pytest

from live import config, daily_orchestrator
from live.daily_orchestrator import (
    _extract_last_quote_by_sym,
    _round_to_tick,
    _submit_limit_entries,
    _validate_quote_for_entry,
)
from live.nasdaq_routing import ACTIVE_ROUTING

_UTC = timezone.utc


class _FakeBroker:
    """Minimal broker fake — records buy/sell kwargs verbatim.

    Only implements what ``_submit_limit_entries`` needs (buy, sell,
    MAX_ORDER_QTY). Does NOT model fills or positions — those tests
    live in ``test_end_to_end.py``.
    """

    MAX_ORDER_QTY = 10_000

    def __init__(self, raise_on: Optional[set[str]] = None) -> None:
        self.buy_calls: list[tuple[str, int, str, dict]] = []
        self.sell_calls: list[tuple[str, int, str, dict]] = []
        self._raise_on = raise_on or set()
        self._counter = 0

    def buy(self, symbol, quantity, order_type="MKT", **kwargs):
        if symbol in self._raise_on:
            raise RuntimeError(f"simulated buy rejection for {symbol}")
        self._counter += 1
        oid = f"buy-{self._counter}"
        self.buy_calls.append((symbol, int(quantity), order_type, dict(kwargs)))
        return oid

    def sell(self, symbol, quantity, order_type="MKT", **kwargs):
        if symbol in self._raise_on:
            raise RuntimeError(f"simulated sell rejection for {symbol}")
        self._counter += 1
        oid = f"sell-{self._counter}"
        self.sell_calls.append((symbol, int(quantity), order_type, dict(kwargs)))
        return oid


# ---------------------------------------------------------------------------
# _round_to_tick — passive per side (round-2 correctness fix)
# ---------------------------------------------------------------------------


def test_round_to_tick_penny_grid_passive_direction_buy():
    """BUY floors DOWN to the next tick."""
    # Mid on the half-penny → BUY rounds to 150.00 (bid side).
    assert _round_to_tick(150.005, "BUY") == pytest.approx(150.00)
    # Half-a-penny above the tick → BUY still floors down.
    assert _round_to_tick(150.0049999, "BUY") == pytest.approx(150.00)
    # A hair above the tick → BUY still floors down.
    assert _round_to_tick(150.00500001, "BUY") == pytest.approx(150.00)
    # On-grid stays on-grid.
    for p in (100.00, 100.01, 100.99, 250.55):
        assert _round_to_tick(p, "BUY") == pytest.approx(p), (
            f"BUY {p} idempotent"
        )


def test_round_to_tick_penny_grid_passive_direction_sell():
    """SELL ceils UP to the next tick."""
    # Mid on the half-penny → SELL rounds to 150.01 (ask side).
    assert _round_to_tick(150.005, "SELL") == pytest.approx(150.01)
    # A hair above the tick → SELL ceils to next.
    assert _round_to_tick(150.00500001, "SELL") == pytest.approx(150.01)
    # On-grid stays on-grid.
    for p in (100.00, 100.01, 100.99, 250.55):
        assert _round_to_tick(p, "SELL") == pytest.approx(p), (
            f"SELL {p} idempotent"
        )


def test_round_to_tick_one_cent_spread_common_case():
    """BLOCKER 1: on the most common quote (1-cent spread with mid
    on the half-penny), BUY and SELL land at bid and ask respectively
    — a PASSIVE fill from both sides. Under round-1's half-up
    behaviour BOTH sides rounded UP to the ask, marketable buys and
    unfillable sells, yielding a systematically net-long realized
    book the covariance never priced.
    """
    bid, ask = 100.00, 100.01
    mid = (bid + ask) / 2  # 100.005 — the half-penny problem case
    buy_lmt = _round_to_tick(mid, "BUY")
    sell_lmt = _round_to_tick(mid, "SELL")
    assert buy_lmt == pytest.approx(bid), (
        f"BUY on half-penny mid must round DOWN to bid {bid}; "
        f"got {buy_lmt}"
    )
    assert sell_lmt == pytest.approx(ask), (
        f"SELL on half-penny mid must round UP to ask {ask}; "
        f"got {sell_lmt}"
    )
    # Also assert exact grid membership (no float noise like
    # 100.00000000000001).
    assert buy_lmt * 100 == round(buy_lmt * 100)
    assert sell_lmt * 100 == round(sell_lmt * 100)


def test_round_to_tick_two_cent_spread_still_passive():
    """Wider spread — mid exactly on the penny. Both sides are
    already on the grid; both return the mid, and the passive
    intent still holds (BUY posts at mid, SELL posts at mid; both
    are inside the touch).
    """
    bid, ask = 100.00, 100.02
    mid = 100.01
    assert _round_to_tick(mid, "BUY") == pytest.approx(100.01)
    assert _round_to_tick(mid, "SELL") == pytest.approx(100.01)


def test_round_to_tick_fine_grid_below_dollar_buy_sell():
    """$0.0001 grid below $1, passive per side."""
    # Mid = 0.50005 → BUY 0.5000, SELL 0.5001.
    assert _round_to_tick(0.50005, "BUY") == pytest.approx(0.5000)
    assert _round_to_tick(0.50005, "SELL") == pytest.approx(0.5001)
    # On-grid.
    assert _round_to_tick(0.5000, "BUY") == pytest.approx(0.5000)
    assert _round_to_tick(0.9999, "SELL") == pytest.approx(0.9999)


def test_round_to_tick_boundary_dollar():
    # Exactly $1.00 uses the ABOVE-DOLLAR grid ($0.01) for both sides.
    assert _round_to_tick(1.0, "BUY") == pytest.approx(1.00)
    assert _round_to_tick(1.0, "SELL") == pytest.approx(1.00)
    # 0.995 with fine grid: BUY 0.9950, SELL 0.9950 (on-grid at 4dp).
    assert _round_to_tick(0.9950, "BUY") == pytest.approx(0.9950)
    assert _round_to_tick(0.9950, "SELL") == pytest.approx(0.9950)


def test_round_to_tick_rejects_bad_side():
    with pytest.raises(ValueError):
        _round_to_tick(150.005, "long")  # not BUY/SELL
    with pytest.raises(ValueError):
        _round_to_tick(150.005, "")  # empty


def test_round_to_tick_float_representation_robust():
    """Round-2 reviewer B: half-up rounding wasn't reliable in
    binary float — ``_round_to_tick(1.005) -> 1.0`` under the old
    code. The Decimal-based implementation MUST not fall over on
    known-hazardous float values.
    """
    # 1.005 with BUY floor → 1.00. Under any half-up float
    # implementation this ambiguously landed at either 1.00 or 1.01.
    # With passive floor we deterministically get 1.00.
    assert _round_to_tick(1.005, "BUY") == pytest.approx(1.00)
    # SELL ceil → 1.01.
    assert _round_to_tick(1.005, "SELL") == pytest.approx(1.01)
    # 100.0499999 → BUY 100.04, SELL 100.05.
    assert _round_to_tick(100.0499999, "BUY") == pytest.approx(100.04)
    assert _round_to_tick(100.0499999, "SELL") == pytest.approx(100.05)


# ---------------------------------------------------------------------------
# _validate_quote_for_entry
# ---------------------------------------------------------------------------


def _make_quote(bid, ask, ts=None):
    if ts is None:
        ts = datetime.now(_UTC)
    mid = (bid + ask) / 2
    return {"bid": bid, "ask": ask, "mid": mid, "ts": ts}


def test_validate_accepts_normal_quote():
    now = datetime.now(_UTC)
    q = _make_quote(150.00, 150.02, ts=now - timedelta(seconds=1))
    ok, reason, mid = _validate_quote_for_entry(
        q, now, max_staleness_secs=30.0, max_rel_spread=0.005,
    )
    assert ok is True
    assert reason is None
    assert mid == pytest.approx(150.01)


def test_validate_rejects_missing_quote():
    now = datetime.now(_UTC)
    ok, reason, mid = _validate_quote_for_entry(
        None, now, max_staleness_secs=30.0, max_rel_spread=0.005,
    )
    assert ok is False
    assert reason == "no_usable_quote_at_entry"
    assert mid is None


def test_validate_rejects_missing_fields():
    now = datetime.now(_UTC)
    q = {"bid": 150.00, "ask": None, "mid": 150.01, "ts": now}
    ok, reason, mid = _validate_quote_for_entry(
        q, now, max_staleness_secs=30.0, max_rel_spread=0.005,
    )
    assert ok is False
    assert reason == "quote_missing_fields"


def test_validate_rejects_crossed_quote():
    now = datetime.now(_UTC)
    # ask < bid — crossed / locked
    q = {"bid": 150.01, "ask": 150.00, "mid": 150.005, "ts": now}
    ok, reason, mid = _validate_quote_for_entry(
        q, now, max_staleness_secs=30.0, max_rel_spread=0.005,
    )
    assert ok is False
    assert reason == "quote_crossed"


def test_validate_rejects_non_positive_bid():
    now = datetime.now(_UTC)
    q = {"bid": 0.0, "ask": 150.02, "mid": 75.01, "ts": now}
    ok, reason, _ = _validate_quote_for_entry(
        q, now, max_staleness_secs=30.0, max_rel_spread=0.005,
    )
    assert ok is False
    assert reason == "quote_non_positive"


def test_validate_rejects_non_positive_ask():
    now = datetime.now(_UTC)
    q = {"bid": 100.0, "ask": -1.0, "mid": 49.5, "ts": now}
    ok, reason, _ = _validate_quote_for_entry(
        q, now, max_staleness_secs=30.0, max_rel_spread=0.005,
    )
    assert ok is False
    assert reason == "quote_non_positive"


def test_validate_rejects_stale_quote():
    now = datetime.now(_UTC)
    # Quote 60s old, threshold 30s.
    q = _make_quote(150.00, 150.02, ts=now - timedelta(seconds=60))
    ok, reason, _ = _validate_quote_for_entry(
        q, now, max_staleness_secs=30.0, max_rel_spread=0.005,
    )
    assert ok is False
    assert reason == "quote_stale"


def test_validate_accepts_at_staleness_threshold():
    # ≤ threshold is OK; > threshold is stale.
    now = datetime.now(_UTC)
    q = _make_quote(150.00, 150.02, ts=now - timedelta(seconds=30))
    ok, reason, _ = _validate_quote_for_entry(
        q, now, max_staleness_secs=30.0, max_rel_spread=0.005,
    )
    assert ok is True, f"expected ok at threshold; reason={reason}"


def test_validate_rejects_wide_spread():
    now = datetime.now(_UTC)
    # 100 bps spread — wider than 50 bps threshold.
    q = _make_quote(100.00, 101.00, ts=now)
    ok, reason, _ = _validate_quote_for_entry(
        q, now, max_staleness_secs=30.0, max_rel_spread=0.005,
    )
    assert ok is False
    assert reason == "quote_wide_spread"


def test_validate_rejects_naive_ts():
    now = datetime.now(_UTC)
    q = {
        "bid": 150.00, "ask": 150.02, "mid": 150.01,
        "ts": datetime.now(),  # naive
    }
    ok, reason, _ = _validate_quote_for_entry(
        q, now, max_staleness_secs=30.0, max_rel_spread=0.005,
    )
    assert ok is False
    assert reason == "quote_naive_ts"


# ---------------------------------------------------------------------------
# Databento sentinel handling (round-2 reviewer B, "also fix")
# ---------------------------------------------------------------------------


def test_validate_rejects_nan_bid_and_ask_from_undef_price():
    """Databento ``UNDEF_PRICE`` is ``INT64_MAX``. Today the
    ``pretty_bid_px_00`` / ``pretty_ask_px_00`` accessors scale
    UNDEF_PRICE to NaN — the code depends on that undocumented
    third-party behavior. Pin the behavior we depend on.
    """
    now = datetime.now(_UTC)
    q = {
        "bid": float("nan"), "ask": float("nan"),
        "mid": float("nan"),
        "ts": now,
    }
    ok, reason, _ = _validate_quote_for_entry(
        q, now, max_staleness_secs=30.0, max_rel_spread=0.005,
    )
    assert ok is False
    assert reason == "quote_non_finite"


def test_validate_rejects_scaled_sentinel_finite_but_absurd():
    """If Databento ever changes ``pretty_*`` to return the SCALED
    sentinel (INT64_MAX * 1e-9 ≈ 9.22e9) instead of NaN, the quote
    would be finite AND positive AND uncrossed AND ``rel_spread == 0``
    — every other guard would pass. Without ``ENTRY_MAX_LIMIT_PRICE``
    we'd emit a BUY LMT at ~$9.2 billion, which TWS would
    happily accept for a fractional-share account. Guard that.
    """
    now = datetime.now(_UTC)
    scaled_sentinel = 9.223372036854776e9  # INT64_MAX * 1e-9
    q = {
        "bid": scaled_sentinel, "ask": scaled_sentinel,
        "mid": scaled_sentinel, "ts": now,
    }
    ok, reason, _ = _validate_quote_for_entry(
        q, now, max_staleness_secs=30.0, max_rel_spread=0.005,
    )
    assert ok is False
    assert reason == "quote_absurd_price", (
        f"expected 'quote_absurd_price' for scaled UNDEF_PRICE "
        f"sentinel; got {reason!r} — the ENTRY_MAX_LIMIT_PRICE "
        f"sanity bound is not firing"
    )


def test_validate_accepts_high_priced_name_below_bound():
    """Berkshire A around $700k is legitimate and MUST pass the
    sanity bound. This pins the ``ENTRY_MAX_LIMIT_PRICE`` at a
    generous-enough value.
    """
    now = datetime.now(_UTC)
    # $700k / share (Berkshire A ballpark).
    q = _make_quote(700_000.00, 700_050.00, ts=now)
    ok, reason, mid = _validate_quote_for_entry(
        q, now,
        max_staleness_secs=30.0,
        max_rel_spread=0.005,
    )
    assert ok is True, (
        f"legitimate $700k BRK-A quote must not be rejected as "
        f"absurd; got reason={reason!r}"
    )
    assert mid == pytest.approx(700_025.00)


def test_validate_clock_skew_future_ts_rejects_gracefully():
    """A tick ts more than 1s in the future relative to ``now_utc``
    is treated as stale (clock-skew defence). Small skews (< 1s)
    are tolerated as feed-vs-local-clock jitter.
    """
    now = datetime.now(_UTC)
    # 500ms in the future → tolerated.
    q = _make_quote(100.00, 100.02, ts=now + timedelta(milliseconds=500))
    ok, reason, _ = _validate_quote_for_entry(
        q, now, max_staleness_secs=30.0, max_rel_spread=0.005,
    )
    assert ok is True, (
        f"500ms of clock skew must NOT reject as stale; got "
        f"reason={reason!r}"
    )
    # 5s in the future → rejected.
    q = _make_quote(100.00, 100.02, ts=now + timedelta(seconds=5))
    ok, reason, _ = _validate_quote_for_entry(
        q, now, max_staleness_secs=30.0, max_rel_spread=0.005,
    )
    assert ok is False
    assert reason == "quote_stale"


# ---------------------------------------------------------------------------
# _submit_limit_entries — end-to-end (broker fake only)
# ---------------------------------------------------------------------------


def _now():
    return datetime.now(_UTC)


def test_submit_limit_entries_buy_prices_at_rounded_mid():
    broker = _FakeBroker()
    quotes = {"AAPL": _make_quote(150.00, 150.02, ts=_now())}
    entries = _submit_limit_entries(
        broker=broker,
        targets={"AAPL": 100},
        current={"AAPL": 0},  # diff = +100 → BUY
        managed_universe={"AAPL"},
        last_quote_by_sym=quotes,
        now_utc=_now(),
        max_staleness_secs=30.0,
        max_rel_spread=0.005,
    )
    assert len(entries) == 1
    e = entries[0]
    assert e["symbol"] == "AAPL"
    assert e["side"] == "BUY"
    assert e["requested_qty"] == 100
    assert e["order_id"] is not None
    assert e["error"] is None
    # Mid = 150.01, on penny grid → limit price 150.01 exactly.
    assert e["limit_price"] == pytest.approx(150.01)
    # Broker was called with LMT + limit_price + ACTIVE_ROUTING.
    assert len(broker.buy_calls) == 1
    sym, qty, ot, kw = broker.buy_calls[0]
    assert (sym, qty, ot) == ("AAPL", 100, "LMT")
    assert kw["limit_price"] == pytest.approx(150.01)
    assert kw["exchange"] == ACTIVE_ROUTING
    assert not broker.sell_calls


def test_submit_limit_entries_sell_prices_at_rounded_mid():
    broker = _FakeBroker()
    quotes = {"MSFT": _make_quote(400.00, 400.02, ts=_now())}
    entries = _submit_limit_entries(
        broker=broker,
        targets={"MSFT": 0},
        current={"MSFT": 50},  # diff = -50 → SELL
        managed_universe={"MSFT"},
        last_quote_by_sym=quotes,
        now_utc=_now(),
        max_staleness_secs=30.0,
        max_rel_spread=0.005,
    )
    assert len(entries) == 1
    e = entries[0]
    assert (e["side"], e["requested_qty"]) == ("SELL", 50)
    assert e["limit_price"] == pytest.approx(400.01)
    assert not broker.buy_calls
    sym, qty, ot, kw = broker.sell_calls[0]
    assert (sym, qty, ot, kw["limit_price"]) == ("MSFT", 50, "LMT", pytest.approx(400.01))
    assert kw["exchange"] == ACTIVE_ROUTING


def test_submit_limit_entries_skips_stale_quote_no_broker_call():
    broker = _FakeBroker()
    now = _now()
    quotes = {"AAPL": _make_quote(150.00, 150.02, ts=now - timedelta(seconds=999))}
    entries = _submit_limit_entries(
        broker=broker,
        targets={"AAPL": 100}, current={"AAPL": 0},
        managed_universe={"AAPL"},
        last_quote_by_sym=quotes,
        now_utc=now,
        max_staleness_secs=30.0,
        max_rel_spread=0.005,
    )
    assert len(entries) == 1
    e = entries[0]
    assert e["symbol"] == "AAPL"
    assert e["side"] == "BUY"
    assert e["order_id"] is None
    assert e["error"] == "quote_stale"
    # Limit price is populated with the raw mid so post-trade analysis
    # sees what we would have priced at.
    assert e["limit_price"] == pytest.approx(150.01)
    # No broker call.
    assert not broker.buy_calls
    assert not broker.sell_calls


def test_submit_limit_entries_skips_crossed_quote_no_broker_call():
    broker = _FakeBroker()
    now = _now()
    quotes = {
        "AAPL": {"bid": 150.02, "ask": 150.00, "mid": 150.01, "ts": now},
    }
    entries = _submit_limit_entries(
        broker=broker,
        targets={"AAPL": 100}, current={"AAPL": 0},
        managed_universe={"AAPL"},
        last_quote_by_sym=quotes,
        now_utc=now,
        max_staleness_secs=30.0,
        max_rel_spread=0.005,
    )
    assert entries[0]["error"] == "quote_crossed"
    assert entries[0]["order_id"] is None
    assert not broker.buy_calls
    assert not broker.sell_calls


def test_submit_limit_entries_skips_missing_quote_no_broker_call():
    broker = _FakeBroker()
    entries = _submit_limit_entries(
        broker=broker,
        targets={"AAPL": 100}, current={"AAPL": 0},
        managed_universe={"AAPL"},
        last_quote_by_sym={},  # no quote at all for AAPL
        now_utc=_now(),
        max_staleness_secs=30.0,
        max_rel_spread=0.005,
    )
    assert len(entries) == 1
    e = entries[0]
    assert e["order_id"] is None
    assert e["error"] == "no_usable_quote_at_entry"
    assert e["limit_price"] is None  # no quote means no mid to record
    assert not broker.buy_calls
    assert not broker.sell_calls


def test_submit_limit_entries_skips_wide_spread_no_broker_call():
    broker = _FakeBroker()
    now = _now()
    quotes = {"AAPL": _make_quote(100.00, 101.00, ts=now)}  # 100 bps
    entries = _submit_limit_entries(
        broker=broker,
        targets={"AAPL": 100}, current={"AAPL": 0},
        managed_universe={"AAPL"},
        last_quote_by_sym=quotes,
        now_utc=now,
        max_staleness_secs=30.0,
        max_rel_spread=0.005,  # 50 bps
    )
    assert entries[0]["error"] == "quote_wide_spread"
    assert not broker.buy_calls
    assert not broker.sell_calls


def test_submit_limit_entries_no_mkt_fallback_on_bad_quote():
    """Verify NO MKT order EVER goes out when a quote guard rejects.

    This is the operator's explicit policy: no entry = no exposure.
    A regression to a "fall back to MKT on bad quote" pattern would
    defeat the whole point of the guard.
    """
    broker = _FakeBroker()
    now = _now()
    # A mix: one good, one bad. Only the good one should hit the
    # broker; the bad one should stay skipped with NO MKT emitted.
    quotes = {
        "AAPL": _make_quote(150.00, 150.02, ts=now),
        # NVDA has a stale quote — must skip.
        "NVDA": _make_quote(500.00, 500.02, ts=now - timedelta(seconds=999)),
    }
    entries = _submit_limit_entries(
        broker=broker,
        targets={"AAPL": 100, "NVDA": 50},
        current={"AAPL": 0, "NVDA": 0},
        managed_universe={"AAPL", "NVDA"},
        last_quote_by_sym=quotes,
        now_utc=now,
        max_staleness_secs=30.0,
        max_rel_spread=0.005,
    )
    all_broker_calls = broker.buy_calls + broker.sell_calls
    # Exactly ONE broker call — for AAPL. NVDA must not appear.
    assert len(all_broker_calls) == 1
    assert all_broker_calls[0][0] == "AAPL"
    # Every broker call is LMT — no MKT anywhere.
    for _sym, _qty, ot, _kw in all_broker_calls:
        assert ot == "LMT", (
            f"MKT fallback detected on broker call: {all_broker_calls} "
            f"— skip policy violated"
        )


def test_submit_limit_entries_zero_diff_no_entry():
    broker = _FakeBroker()
    quotes = {"AAPL": _make_quote(150.00, 150.02, ts=_now())}
    entries = _submit_limit_entries(
        broker=broker,
        targets={"AAPL": 100}, current={"AAPL": 100},  # diff = 0
        managed_universe={"AAPL"},
        last_quote_by_sym=quotes,
        now_utc=_now(),
        max_staleness_secs=30.0,
        max_rel_spread=0.005,
    )
    assert entries == []
    assert not broker.buy_calls
    assert not broker.sell_calls


def test_submit_limit_entries_ignores_out_of_universe():
    broker = _FakeBroker()
    quotes = {"XYZ": _make_quote(150.00, 150.02, ts=_now())}
    entries = _submit_limit_entries(
        broker=broker,
        targets={"XYZ": 100},
        current={"XYZ": 0},
        managed_universe=set(),  # empty — XYZ is out of universe
        last_quote_by_sym=quotes,
        now_utc=_now(),
        max_staleness_secs=30.0,
        max_rel_spread=0.005,
    )
    assert entries == []
    assert not broker.buy_calls


def test_submit_limit_entries_records_broker_error_as_error_field():
    """If the broker itself raises (rejection at TWS), the entry
    dict must record the exception message and populate
    ``limit_price`` so post-trade sees what we tried to bid at.
    """
    broker = _FakeBroker(raise_on={"AAPL"})
    quotes = {"AAPL": _make_quote(150.00, 150.02, ts=_now())}
    entries = _submit_limit_entries(
        broker=broker,
        targets={"AAPL": 100}, current={"AAPL": 0},
        managed_universe={"AAPL"},
        last_quote_by_sym=quotes,
        now_utc=_now(),
        max_staleness_secs=30.0,
        max_rel_spread=0.005,
    )
    assert len(entries) == 1
    e = entries[0]
    assert e["order_id"] is None
    assert "simulated" in (e["error"] or "")
    assert e["limit_price"] == pytest.approx(150.01)


# ---------------------------------------------------------------------------
# _extract_last_quote_by_sym — shared helper
# ---------------------------------------------------------------------------


def test_extract_last_quote_by_sym_last_tick_wins():
    """A per-symbol group aggregation should return the LATEST
    bid/ask/mid/ts, not the first.
    """
    import polars as pl
    frame = pl.DataFrame({
        "ts_event": [
            datetime(2026, 6, 15, 15, 40, 0, tzinfo=_UTC),
            datetime(2026, 6, 15, 15, 50, 0, tzinfo=_UTC),
        ],
        "symbol": ["AAPL", "AAPL"],
        "price": [100.0, 150.0],
        "size": [100, 100],
        "ask_px_00": [100.01, 150.02],
        "bid_px_00": [99.99, 150.00],
    })
    q = _extract_last_quote_by_sym(frame)
    assert "AAPL" in q
    assert q["AAPL"]["bid"] == pytest.approx(150.00)
    assert q["AAPL"]["ask"] == pytest.approx(150.02)
    assert q["AAPL"]["mid"] == pytest.approx(150.01)
    assert q["AAPL"]["ts"] == datetime(2026, 6, 15, 15, 50, 0, tzinfo=_UTC)


def test_extract_last_quote_by_sym_universe_filter():
    import polars as pl
    frame = pl.DataFrame({
        "ts_event": [
            datetime(2026, 6, 15, 15, 50, 0, tzinfo=_UTC),
            datetime(2026, 6, 15, 15, 50, 0, tzinfo=_UTC),
        ],
        "symbol": ["AAPL", "NVDA"],
        "price": [150.0, 500.0],
        "size": [100, 100],
        "ask_px_00": [150.02, 500.02],
        "bid_px_00": [150.00, 500.00],
    })
    q = _extract_last_quote_by_sym(frame, universe=["AAPL"])
    assert set(q) == {"AAPL"}


def test_extract_last_quote_by_sym_none_frame_returns_empty():
    """Round-2 empty-branch coverage."""
    assert _extract_last_quote_by_sym(None) == {}
    assert _extract_last_quote_by_sym(None, universe=["AAPL"]) == {}


def test_extract_last_quote_by_sym_empty_frame_returns_empty():
    import polars as pl
    empty = pl.DataFrame(schema={
        "ts_event": pl.Datetime("us", "UTC"),
        "symbol": pl.Utf8,
        "price": pl.Float64,
        "size": pl.Int64,
        "ask_px_00": pl.Float64,
        "bid_px_00": pl.Float64,
    })
    assert _extract_last_quote_by_sym(empty) == {}
    assert _extract_last_quote_by_sym(empty, universe=["AAPL"]) == {}


def test_extract_last_quote_by_sym_universe_filter_yielding_empty():
    """When ``universe`` filters out every row, return empty (not a
    stale entry from before the filter)."""
    import polars as pl
    frame = pl.DataFrame({
        "ts_event": [datetime(2026, 6, 15, 15, 50, 0, tzinfo=_UTC)],
        "symbol": ["AAPL"],
        "price": [150.0],
        "size": [100],
        "ask_px_00": [150.02],
        "bid_px_00": [150.00],
    })
    assert _extract_last_quote_by_sym(frame, universe=["NVDA"]) == {}


def test_extract_last_quote_by_sym_max_ts_prevents_lookahead():
    """BLOCKER 3b: with a whole-day tbbo frame, ``max_ts`` bounds
    extraction at the decision moment. Without this, replay
    ``_collect_tbbo`` returns the whole trading day and ``.last()``
    picks the 16:00 closing print — the exact tick the strategy is
    trying to forecast.
    """
    import polars as pl
    # Simulate the "whole day" a replay parquet would deliver: one
    # tick at 15:45 (pre-decision), one at 15:50 (decision), one at
    # 16:00 (POST-decision — the closing print!).
    frame = pl.DataFrame({
        "ts_event": [
            datetime(2026, 6, 15, 15, 45, 0, tzinfo=_UTC),
            datetime(2026, 6, 15, 15, 50, 0, tzinfo=_UTC),
            datetime(2026, 6, 15, 16, 0, 0, tzinfo=_UTC),
        ],
        "symbol": ["AAPL", "AAPL", "AAPL"],
        "price": [100.0, 150.0, 999.0],  # 999 = closing print, THE forecast target
        "size": [100, 100, 100],
        "ask_px_00": [100.02, 150.02, 999.02],
        "bid_px_00": [100.00, 150.00, 999.00],
    })
    # Without max_ts: whole-day .last() gives the 16:00 tick.
    q = _extract_last_quote_by_sym(frame)
    assert q["AAPL"]["mid"] == pytest.approx(999.01), (
        f"pre-fix behaviour: without max_ts the closing print bleeds "
        f"into the entry price. Got mid={q['AAPL']['mid']}"
    )
    # WITH max_ts = 15:50: the 16:00 tick is filtered out.
    q_bounded = _extract_last_quote_by_sym(
        frame, max_ts=datetime(2026, 6, 15, 15, 50, 0, tzinfo=_UTC),
    )
    assert q_bounded["AAPL"]["mid"] == pytest.approx(150.01), (
        f"max_ts must exclude the 16:00 tick to prevent look-ahead. "
        f"Got mid={q_bounded['AAPL']['mid']}"
    )
    # And WITH max_ts = 15:44 (BEFORE any tick): symbol drops out.
    q_pre = _extract_last_quote_by_sym(
        frame, max_ts=datetime(2026, 6, 15, 15, 44, 0, tzinfo=_UTC),
    )
    assert q_pre == {}


def test_extract_last_quote_by_sym_bad_last_tick_falls_back_to_clean_tick():
    """Round-2 reviewer B #8: a bad final print (NaN, crossed,
    negative) must NOT evict a symbol whose earlier ticks are clean.
    Filter finite / positive / uncrossed BEFORE the group_by so the
    last-good tick wins.
    """
    import polars as pl
    # Two clean ticks then one bad (NaN) final print.
    frame = pl.DataFrame({
        "ts_event": [
            datetime(2026, 6, 15, 15, 45, 0, tzinfo=_UTC),
            datetime(2026, 6, 15, 15, 49, 58, tzinfo=_UTC),
            datetime(2026, 6, 15, 15, 50, 0, tzinfo=_UTC),  # BAD final
        ],
        "symbol": ["AAPL", "AAPL", "AAPL"],
        "price": [100.0, 150.0, 200.0],
        "size": [100, 100, 100],
        # Final tick has NaN bid — used to be a NaN-in-the-output bug.
        "ask_px_00": [100.02, 150.02, float("nan")],
        "bid_px_00": [100.00, 150.00, float("nan")],
    })
    q = _extract_last_quote_by_sym(frame)
    assert "AAPL" in q, (
        "a single bad final tick must NOT evict a symbol whose earlier "
        "ticks are clean"
    )
    # Should reflect the LAST GOOD tick (15:49:58), not the bad one.
    assert q["AAPL"]["mid"] == pytest.approx(150.01)


# ---------------------------------------------------------------------------
# MOC exit path is unaffected by the LMT changes
# ---------------------------------------------------------------------------


def test_unfilled_entry_works_until_sweep_deadline():
    """BLOCKER 2 (round 2): under passive LMTs the dispatcher must
    NOT walk away at 30s (the MKT-era ``DEFAULT_ENTRY_FILL_TIMEOUT_SECS``).
    ``DEFAULT_ENTRY_WORKING_SECS`` defaults to None (= work until
    sweep deadline). Prove that an entry still non-terminal at 30s
    survives PAST 30s and is then cancelled by the sweep before
    15:55.

    We simulate a fixed clock so we can prove ordering without
    actually sleeping for real seconds. The dispatcher polls
    ``config.now_ny`` on every iteration; we drive it via
    monkeypatch.
    """
    from datetime import datetime as _dt, timedelta as _td
    from live import config as _cfg, daily_orchestrator as _do

    fake_broker = _FakeBroker()

    # Submit a stuck entry directly (simulating a passive limit
    # that hasn't crossed yet). The fake FakeBroker in
    # test_limit_entries doesn't model stuck orders, so build a
    # tiny local one.
    class _StuckBroker(_FakeBroker):
        def buy(self, symbol, quantity, order_type="MKT", **kwargs):
            oid = super().buy(symbol, quantity, order_type=order_type, **kwargs)
            self._stuck_oid = oid
            return oid

        def get_order_status(self, order_id):
            # Non-terminal Submitted, filled=0 — the LMT hasn't
            # crossed yet.
            class _S:
                status = "Submitted"
                filled = 0
            return _S()

        def get_positions(self):
            return []

        def cancel(self, oid):
            self.cancel_calls = getattr(self, "cancel_calls", [])
            self.cancel_calls.append(oid)

    stuck = _StuckBroker()
    oid = stuck.buy("AAPL", 100, order_type="LMT",
                    limit_price=150.00, exchange="SMART")
    entries = [{
        "symbol": "AAPL", "side": "BUY", "requested_qty": 100,
        "order_id": oid, "error": None, "limit_price": 150.00,
    }]

    # Drive the clock via monkey-patch so we don't sleep for real.
    now_ref = [_dt(2026, 6, 15, 15, 50, 0, tzinfo=_cfg.NY_TZ)]

    def _advance_now():
        # Every call advances 5 seconds.
        now_ref[0] = now_ref[0] + _td(seconds=5)
        return now_ref[0]

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(_cfg, "now_ny", _advance_now)
        # The PRODUCTION default is "work until the sweep deadline"
        # (``DEFAULT_ENTRY_WORKING_SECS is None``). Under this test
        # suite the conftest patches it to 0.5s for speed, so we
        # override that patch back to None just for this test to
        # exercise the true production behaviour.
        monkeypatch.setattr(_cfg, "DEFAULT_ENTRY_WORKING_SECS", None, raising=False)
        # Sweep deadline well past what the old MKT-era 30s cap
        # would have permitted — proves the dispatcher will now
        # work all the way to the sweep deadline instead of
        # walking away at 30s. Use a small (2s) real sweep_deadline
        # so the test loop terminates quickly; the point is the
        # dispatcher DOES loop past t+30s (of its virtual clock),
        # not that we actually wait 4 minutes.
        from datetime import timezone as _tz
        real_now = _dt(2026, 6, 15, 15, 50, 0, tzinfo=_cfg.NY_TZ)
        now_ref[0] = real_now
        # ROUND-3 HARDENING: sweep_deadline is pushed to 300s
        # (5 min) past dispatch-start in virtual time. If a 30s
        # cap were reintroduced (``effective_deadline =
        # min(sweep_deadline, now + 30s)``), the dispatcher would
        # exit at ~30-35s virtual and ``elapsed_virtual_secs``
        # would land in that neighborhood. Under the correct
        # working-until-sweep behavior the loop only exits when
        # virtual time crosses sweep_deadline, i.e. ~300s virtual.
        # The threshold assertion below (``>= 250``) discriminates
        # cleanly: cap=30s fails, no-cap passes.
        sweep_deadline = real_now + _td(seconds=300)
        submit_by = real_now + _td(seconds=360)

        (moc_batch, covered_qty, uncovered, skipped, positions_unknown) = (
            _do._fill_gated_moc_dispatch(
                broker=stuck,
                entries=entries,
                managed_universe={"AAPL"},
                sweep_deadline=sweep_deadline,
                submit_by=submit_by,
                baseline={"AAPL": 0},
                poll_interval_secs=0.001,  # keep real sleeps tiny
                # Do NOT pass max_wait_secs — production leaves it
                # None so ``DEFAULT_ENTRY_WORKING_SECS`` (also None
                # after our override above) governs, meaning "work
                # until sweep_deadline".
            )
        )
        # The virtual clock must have advanced ALL THE WAY to the
        # sweep_deadline. If a 30s cap were reintroduced the loop
        # would exit around t=30-35s virtual — well below the 250s
        # threshold below. Reviewer A's mutation proof: re-introducing
        # ``effective_deadline = min(sweep_deadline, now + 30s)``
        # leaves elapsed at ~30-35s virtual, which the old
        # ``>= 30.0`` bar still passed silently. The new ``>= 250``
        # bar catches it.
        elapsed_virtual_secs = (now_ref[0] - real_now).total_seconds()
        assert elapsed_virtual_secs >= 250.0, (
            f"dispatcher exited well before sweep_deadline "
            f"(elapsed={elapsed_virtual_secs}s virtual, expected ~300s). "
            f"A 30s cap on the effective deadline would land here — "
            f"the old ``DEFAULT_ENTRY_FILL_TIMEOUT_SECS`` 30s cap is "
            f"still binding for LMT entries, or some other pre-sweep "
            f"cap has been introduced. The passive LMT design relies on "
            f"the dispatcher working the LMT UNTIL the sweep."
        )
    finally:
        monkeypatch.undo()

    # The dispatcher must NOT have cancelled the entry itself — that
    # is the sweep's job. It should have exited via sweep_deadline
    # with the entry left non-terminal for the sweep.
    assert entries[0]["status"] == "pending_at_sweep_deadline", (
        f"expected entry to be left for sweep; got status="
        f"{entries[0].get('status')!r}. If this reads as "
        f"'wait_budget_exhausted' or similar, DEFAULT_ENTRY_WORKING_SECS "
        f"is capping the dispatcher below the sweep deadline again."
    )
    # And no MOC — the entry never filled.
    assert moc_batch == []

    # Now simulate the sweep: it cancels the still-non-terminal entry.
    (sweep_moc, cancelled_oids, cancel_failed, skipped_sweep, positions_unknown) = (
        _do._sweep_moc_deadline(
            broker=stuck,
            entries=entries,
            managed_universe={"AAPL"},
            covered_qty={},
            submit_by=submit_by,
        )
    )
    assert oid in cancelled_oids, (
        "sweep must cancel the still-working LMT entry before 15:55"
    )
    assert sweep_moc == [], (
        "no position → no sweep MOC"
    )


def test_moc_exits_untouched_by_limit_entry_migration():
    """MOC exits (LEG 2) must still submit as ``order_type='MOC'`` —
    the LMT migration only touches LEG 1.

    Rather than orchestrate a whole run here, exercise
    ``_submit_single_moc`` directly and pin its ``order_type``.
    """
    broker = _FakeBroker()
    # ``_submit_single_moc`` submits a SELL MOC for a positive current
    # qty (flatten a long) and a BUY MOC for a negative current qty
    # (flatten a short). Signed argument: +100 → SELL 100 MOC.
    from datetime import timedelta
    submit_by = datetime.now(config.NY_TZ) + timedelta(hours=1)
    leg, outcome = daily_orchestrator._submit_single_moc(
        broker=broker,
        symbol="AAPL",
        current_qty=100,
        submit_by=submit_by,
    )
    assert outcome == "ok"
    assert leg is not None
    # Verify: MOC, not LMT.
    assert broker.sell_calls, "expected a SELL MOC to flatten +100 long"
    sym, qty, ot, kw = broker.sell_calls[0]
    assert ot == "MOC", (
        f"MOC exit MUST use order_type='MOC'; got {ot!r} — LEG 2 "
        f"was disturbed by the LMT migration"
    )
    assert "limit_price" not in kw or kw.get("limit_price") is None, (
        f"MOC exit MUST NOT carry a limit_price; got {kw!r}"
    )
    assert kw["exchange"] == ACTIVE_ROUTING
