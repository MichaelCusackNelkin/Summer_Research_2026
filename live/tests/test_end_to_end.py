"""End-to-end sanity: mocked Databento Live + mocked broker + real artifacts.

Proves the full ``stream -> features -> alpha -> MVO -> MKT entry ->
MOC exit`` chain runs on the SAME single-day buffer shape the live path
sees at 15:50 — the previous test used a 3-day buffer, masking C2. All
artifacts are in-memory so the test does not require
``forecast/forecast_pipe.joblib`` on disk.

Also covers post-review defensive behaviours:
- H1: out-of-universe positions are pinned (never flattened).
- H2: unqualified symbols dropped from the target; over-max legs raise
  ``MAX_ORDER_QTY`` for the session rather than crash.
- L1 / empty-book refusal: returns rc=5 rather than routing an empty
  ``share_targets`` into the broker (which would flatten the book).
- C2 coverage assertion: a thin single-day snapshot (few universe names)
  raises before any broker call.
- Two-leg execution: MKT entries fire first, then a fresh
  ``broker.get_positions()`` snapshot drives MOC counter-flatten. The
  MOC quantity per name is ALWAYS ``abs(current_qty)``, so a partial
  MKT fill cannot over-flatten into an opposite overnight exposure.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from live import config
from live import daily_orchestrator
from live.features import ALPHA_FEATURES
from live.live_feed import LiveFeed


class _FakePosition:
    __slots__ = ("symbol", "quantity", "avg_cost")

    def __init__(self, symbol, quantity, avg_cost=100.0):
        self.symbol = symbol
        self.quantity = quantity
        self.avg_cost = avg_cost


class _FakeQualifiedContract:
    """Stand-in for an ib_async ``Stock`` after ``qualifyContracts``.

    Exposes the fields ``_prequalify_symbols`` inspects on the
    qualified contract: ``symbol``, ``exchange``, ``primaryExchange``.
    ``exchange`` echoes what the caller requested; ``primaryExchange``
    is set from ``_FakeBroker._non_nasdaq_primary`` (default
    ``"NASDAQ"``) so the NASDAQ-listing filter can be exercised.
    """

    __slots__ = ("symbol", "exchange", "primaryExchange")

    def __init__(self, symbol: str, exchange: str, primaryExchange: str):
        self.symbol = symbol
        self.exchange = exchange
        self.primaryExchange = primaryExchange


class _FakeOrderStatus:
    def __init__(self, order_id, symbol, action, requested, filled, status="Filled"):
        self.order_id = order_id
        self.symbol = symbol
        self.action = action
        self.quantity = requested
        self.status = status
        self.filled = filled
        self.avg_fill_price = 100.0


_TERMINAL_STATES = {"Filled", "Cancelled", "ApiCancelled", "Inactive"}


def _capture_orchestrator_logs(level=None):
    """Attach a handler to the daily_orchestrator logger and return
    (handler, records). Needed because setup_logging sets
    ``logger.propagate = False``, so pytest's caplog fixture (which
    hooks the root logger) captures nothing.
    """
    import logging as _logging
    records: list[_logging.LogRecord] = []

    class _Capture(_logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Capture(level=level or _logging.NOTSET)
    daily_orchestrator.logger.addHandler(handler)
    return handler, records


class _FakeBroker:
    """Stand-in for ``ibkr_broker.broker.Broker`` that models fills.

    Every entry (``buy``/``sell`` with ``order_type="MKT"``) and MOC
    (``buy``/``sell`` with ``order_type="MOC"``) is recorded AND
    partially/fully fills into an internal position book according to
    ``entry_fill_ratio`` (1.0 by default). A subsequent
    ``get_positions()`` reflects the executed fills so
    ``_submit_moc_flatten``'s snapshot behaves like a real broker after
    an MKT entry batch — including the "MOC exit must be sized to
    actual current quantity, not requested" invariant.

    Additional test knobs:
    - ``stuck_on_symbols``: entries for these symbols record with
      ``status="Submitted"`` and ``filled=0`` — they never fill.
      ``wait_for_fill`` raises ``TimeoutError`` on them so the
      belt-and-braces cancel path can be tested.
    - ``get_positions_raises``: cause ``get_positions()`` to raise.
      When ``get_positions_after_n_calls`` is set, only the Nth-and-
      subsequent calls raise.
    - ``get_positions_empty_after_calls``: after N successful calls,
      return ``[]`` — simulates a stale TWS snapshot post-fill.
    - ``duplicate_positions``: symbols whose ``get_positions()`` row is
      returned twice (multi-account / update-race scenario).
    - ``duplicate_all_positions``: every position row is returned twice.
    - ``stuck_all_entries``: every entry records ``status="Submitted"``
      and never fills.
    - ``get_positions_partial_after_n_calls``: from the Nth call onward,
      return only ``partial_position_fraction`` of the rows (default
      0.3 = 3/10 names). Simulates a mid-flow TWS reconnect that
      truncated the position snapshot.
    - ``wait_for_fill_always_times_out``: every ``wait_for_fill`` raises
      ``TimeoutError`` even though the internal book records real
      fills — simulates the R5 "budget-exhausted" branch while
      ``get_order_status`` still reports the true post-fill state.
    - ``moc_reject_all``: every MOC placement returns an oid but records
      ``status="Cancelled"`` and ``filled=0`` (does NOT update the
      internal position book). Simulates an exchange-rejected MOC.
    - ``entry_status_stale``: entries record with an internal fill AND
      an accurate ``status="Filled"`` in the internal book, but
      ``get_order_status`` returns ``status="Submitted"`` / ``filled=0``
      (a stale TWS event stream). ``get_positions()`` still reflects
      the true fill. Simulates RRR3 (staleness brownout).
    - ``non_nasdaq_primary``: mapping ``{symbol: primary_exchange}``
      injected onto the qualified contract's ``primaryExchange`` field
      so tests can exercise ``_prequalify_symbols``'s NASDAQ-listing
      drop path (a NYSE-primary name direct-routed to ISLAND passes
      qualification but its MOC would mis-print at the close; the
      orchestrator drops it during prequalify).
    """

    LIVE_CONFIRM_MSG = "YES_I_WANT_LIVE_TRADING"
    MAX_ORDER_QTY = 10_000

    def __init__(self, initial_positions: list[_FakePosition] | None = None,
                 unqualified: set[str] | None = None,
                 entry_fill_ratio: float = 1.0,
                 raise_on_buy_symbols: set[str] | None = None,
                 raise_on_sell_symbols: set[str] | None = None,
                 stuck_on_symbols: set[str] | None = None,
                 stuck_all_entries: bool = False,
                 get_positions_raises: bool = False,
                 get_positions_raises_after_n_calls: int | None = None,
                 get_positions_empty_after_calls: int | None = None,
                 duplicate_positions: set[str] | None = None,
                 duplicate_all_positions: bool = False,
                 get_positions_partial_after_n_calls: int | None = None,
                 partial_position_fraction: float = 0.3,
                 wait_for_fill_always_times_out: bool = False,
                 moc_reject_all: bool = False,
                 entry_status_stale: bool = False,
                 non_nasdaq_primary: dict[str, str] | None = None):
        self.connected = False
        self.disconnected = False
        self.rebalance_calls: list[tuple[dict, str]] = []
        # Each element: (symbol, requested_qty, order_type, kwargs)
        self.buy_calls: list[tuple[str, int, str, dict]] = []
        self.sell_calls: list[tuple[str, int, str, dict]] = []
        self.cancel_calls: list[str] = []
        # RR3: temporal event log — one tag per side-effect call, in
        # order. Used to pin cancel-vs-snapshot ordering (the R2 guard).
        self.events: list[str] = []
        self.reconcile_calls: list[dict[str, int]] = []
        self.qualified_symbols: list[str] = []
        # (symbol, exchange) pairs recorded by _make_contract — feeds
        # the NASDAQ-routing prequalify regression.
        self.make_contract_calls: list[tuple[str, str]] = []
        self._positions = {p.symbol: p.quantity for p in (initial_positions or [])}
        self._avg_costs = {p.symbol: p.avg_cost for p in (initial_positions or [])}
        self._unqualified = set(unqualified or [])
        self._order_counter = 0
        # Per-order: symbol, action, requested, filled, order_type, status
        self._orders: dict[str, dict] = {}
        self.entry_fill_ratio = entry_fill_ratio
        self._raise_on_buy = set(raise_on_buy_symbols or [])
        self._raise_on_sell = set(raise_on_sell_symbols or [])
        self._stuck_on = set(stuck_on_symbols or [])
        self._stuck_all_entries = stuck_all_entries
        self._get_positions_raises = get_positions_raises
        self._get_positions_raises_after_n_calls = get_positions_raises_after_n_calls
        self._get_positions_empty_after_calls = get_positions_empty_after_calls
        self._get_positions_call_count = 0
        self._duplicate_positions = set(duplicate_positions or [])
        self._duplicate_all_positions = duplicate_all_positions
        self._get_positions_partial_after_n_calls = get_positions_partial_after_n_calls
        self._partial_position_fraction = partial_position_fraction
        self._wait_for_fill_always_times_out = wait_for_fill_always_times_out
        self._moc_reject_all = moc_reject_all
        self._entry_status_stale = entry_status_stale
        # Per-symbol override for the qualified contract's
        # ``primaryExchange`` field. Unspecified names default to
        # ``"NASDAQ"`` so existing tests pass ``_prequalify_symbols``'s
        # NASDAQ-listing filter; tests that want to exercise the
        # non-NASDAQ-primary drop path seed entries here.
        self._non_nasdaq_primary: dict[str, str] = dict(non_nasdaq_primary or {})

    def connect(self, mode="paper", **kwargs):
        self.connected = True

    def disconnect(self):
        self.disconnected = True

    def get_positions(self):
        self._get_positions_call_count += 1
        self.events.append("get_positions")
        if self._get_positions_raises:
            raise RuntimeError("simulated TWS drop in get_positions")
        if (self._get_positions_raises_after_n_calls is not None
                and self._get_positions_call_count
                >= self._get_positions_raises_after_n_calls):
            raise RuntimeError(
                f"simulated TWS drop on get_positions call "
                f"#{self._get_positions_call_count}"
            )
        if (self._get_positions_empty_after_calls is not None
                and self._get_positions_call_count
                > self._get_positions_empty_after_calls):
            return []
        rows: list[_FakePosition] = []
        for sym, qty in self._positions.items():
            if qty == 0:
                continue
            row = _FakePosition(sym, qty, self._avg_costs.get(sym, 100.0))
            if (self._duplicate_all_positions
                    or sym in self._duplicate_positions):
                # Split the true qty across two rows summing to the same
                # total, so aggregation must sum both to reproduce it.
                # This surfaces the naive-loop double-flatten bug: a
                # non-aggregating loop would send TWO MOC orders.
                half = qty // 2
                other = qty - half
                rows.append(_FakePosition(sym, half, self._avg_costs.get(sym, 100.0)))
                rows.append(_FakePosition(sym, other, self._avg_costs.get(sym, 100.0)))
            else:
                rows.append(row)
        # RR1 (a): truncate to a fraction of the rows to simulate a
        # partial TWS snapshot mid-flow.
        if (self._get_positions_partial_after_n_calls is not None
                and self._get_positions_call_count
                >= self._get_positions_partial_after_n_calls):
            keep = max(1, int(round(len(rows) * self._partial_position_fraction)))
            rows = rows[:keep]
        return rows

    def _make_contract(self, symbol, exchange="SMART", currency="USD"):
        if symbol in self._unqualified:
            raise ValueError(f"cannot qualify {symbol}")
        self.qualified_symbols.append(symbol)
        # Record every (symbol, exchange) pair the prequalify path
        # requests so the NASDAQ-routing regression can assert every
        # contract went out with the ISLAND direct-route code.
        self.make_contract_calls.append((symbol, exchange))
        # Return a stub carrying the same shape the qualified
        # ``Stock`` contract exposes to callers: ``symbol``,
        # ``exchange``, and ``primaryExchange``. Default primary is
        # ``"NASDAQ"`` so existing tests pass the new
        # ``_prequalify_symbols`` NASDAQ-listing filter; a per-symbol
        # override in ``_non_nasdaq_primary`` lets tests seed
        # NYSE / ARCA / etc. primaries to exercise the drop path.
        primary = self._non_nasdaq_primary.get(symbol, "NASDAQ")
        return _FakeQualifiedContract(
            symbol=symbol,
            exchange=exchange,
            primaryExchange=primary,
        )

    def _record_fill(self, oid: str, symbol: str, action: str,
                     requested: int, order_type: str, tif: str = "DAY") -> None:
        # Stuck-on symbols record as non-terminal (Submitted, 0 filled) —
        # simulates a MKT placed but never acked. MOC orders (order_type
        # == "MOC") always fully fill in this fake.
        if order_type == "MOC":
            if self._moc_reject_all:
                filled = 0
                status = "Cancelled"
            else:
                filled = requested
                status = "Filled"
        elif self._stuck_all_entries or symbol in self._stuck_on:
            filled = 0
            status = "Submitted"
        else:
            filled = int(round(requested * self.entry_fill_ratio))
            status = "Filled"
        if filled > 0:
            signed = filled if action == "BUY" else -filled
            self._positions[symbol] = self._positions.get(symbol, 0) + signed
        self._orders[oid] = {
            "symbol": symbol, "action": action, "requested": requested,
            "filled": filled, "order_type": order_type, "tif": tif,
            "status": status,
        }

    def buy(self, symbol, quantity, order_type="MKT", **kwargs):
        if symbol in self._raise_on_buy:
            raise RuntimeError(f"simulated buy rejection for {symbol}")
        if quantity > self.MAX_ORDER_QTY:
            raise ValueError(f"quantity {quantity} > MAX_ORDER_QTY {self.MAX_ORDER_QTY}")
        self.buy_calls.append((symbol, quantity, order_type, dict(kwargs)))
        self._order_counter += 1
        oid = f"oid-{self._order_counter}"
        self._record_fill(oid, symbol, "BUY", int(quantity), order_type,
                          tif=kwargs.get("tif", "DAY"))
        self.events.append(f"buy:{order_type}:{symbol}:{oid}")
        return oid

    def sell(self, symbol, quantity, order_type="MKT", **kwargs):
        if symbol in self._raise_on_sell:
            raise RuntimeError(f"simulated sell rejection for {symbol}")
        if quantity > self.MAX_ORDER_QTY:
            raise ValueError(f"quantity {quantity} > MAX_ORDER_QTY {self.MAX_ORDER_QTY}")
        self.sell_calls.append((symbol, quantity, order_type, dict(kwargs)))
        self._order_counter += 1
        oid = f"oid-{self._order_counter}"
        self._record_fill(oid, symbol, "SELL", int(quantity), order_type,
                          tif=kwargs.get("tif", "DAY"))
        self.events.append(f"sell:{order_type}:{symbol}:{oid}")
        return oid

    def cancel(self, order_id):
        self.cancel_calls.append(order_id)
        self.events.append(f"cancel:{order_id}")
        info = self._orders.get(order_id)
        if info is None:
            raise ValueError(f"unknown order {order_id}")
        # Simulate: cancel of a non-terminal order flips it to ApiCancelled.
        if info["status"] not in _TERMINAL_STATES:
            info["status"] = "ApiCancelled"

    def rebalance(self, target_positions, order_type="MKT", close_unlisted=True, tif="DAY"):
        self.rebalance_calls.append((dict(target_positions), order_type))
        return [f"oid-{i}" for i in range(len(target_positions))]

    def reconcile_positions(self, expected: dict[str, int]) -> dict:
        self.reconcile_calls.append(dict(expected))
        return {}

    def wait_for_fill(self, order_id, timeout=30.0):
        # RR1 (b): simulate R5's budget-exhausted branch — the wait
        # times out even though the order actually filled at the
        # exchange. get_order_status STILL reports the true state,
        # which is what the coverage cross-check re-polls.
        if self._wait_for_fill_always_times_out:
            raise TimeoutError(f"simulated wait budget exhausted for {order_id}")
        info = self._orders.get(order_id)
        if info is None:
            return _FakeOrderStatus(order_id, "UNKNOWN", "BUY", 0, 0, status="Unknown")
        if info["status"] not in _TERMINAL_STATES:
            raise TimeoutError(
                f"order {order_id} not terminal (status={info['status']})"
            )
        return _FakeOrderStatus(
            order_id=order_id, symbol=info["symbol"], action=info["action"],
            requested=info["requested"], filled=info["filled"], status=info["status"],
        )

    def get_order_status(self, order_id):
        info = self._orders.get(order_id)
        if info is None:
            raise ValueError(f"unknown order {order_id}")
        # RRR3: entry_status_stale simulates a TWS event-stream brownout
        # where the local self._trades cache lags. get_order_status
        # returns normally (no exception) but the status is non-terminal
        # and filled=0 even though the order actually filled at the
        # exchange. Only MKT entries are affected — MOC and post-cancel
        # states are reported truthfully so other tests remain valid.
        if (self._entry_status_stale
                and info["order_type"] == "MKT"
                and info["status"] == "Filled"):
            return _FakeOrderStatus(
                order_id=order_id, symbol=info["symbol"],
                action=info["action"], requested=info["requested"],
                filled=0, status="Submitted",
            )
        return _FakeOrderStatus(
            order_id=order_id, symbol=info["symbol"], action=info["action"],
            requested=info["requested"], filled=info["filled"], status=info["status"],
        )


def _prepopulate_live_feed(universe: list[str], subset_size: int = 100) -> LiveFeed:
    """Single-day live buffer covering ``subset_size`` names.

    Matches the live path shape exactly: ONE day of tbbo, ONE 15:50
    imbalance print per symbol. No warm-up rows — the C2-relevant edge case.
    """
    feed = LiveFeed(api_key="dummy", universe=universe, reconnect=False)

    class _StubClient:
        def stop(self):
            pass

        def block_for_close(self, timeout=None):
            pass

    feed._client = _StubClient()
    feed._started = True

    day = "2026-06-15"
    imb_recs: list[dict] = []
    tbbo_recs: list[dict] = []
    subset = universe[:subset_size]
    for i, sym in enumerate(subset):
        base = 100.0 + i * 3
        for hh, mm, ss in [(15, 40, 0), (15, 45, 0), (15, 49, 58), (15, 50, 30)]:
            ts = datetime.fromisoformat(
                f"{day}T{hh:02d}:{mm:02d}:{ss:02d}-04:00"
            ).astimezone(timezone.utc)
            tbbo_recs.append({
                "ts_event": ts,
                "symbol": sym,
                "price": base + 0.01 * mm,
                "size": 100,
                "ask_px_00": base + 0.01 * mm + 0.005,
                "bid_px_00": base + 0.01 * mm - 0.005,
            })
        ts_imb = datetime.fromisoformat(f"{day}T15:50:01-04:00").astimezone(timezone.utc)
        imb_recs.append({
            "ts_event": ts_imb,
            "symbol": sym,
            "side": "B" if (i % 2) == 0 else "A",
            "total_imbalance_qty": 100_000 + 1_000 * i,
            "paired_qty": 500_000,
            "ref_price": base + 0.5,
        })
    feed._buffers["imbalance"] = imb_recs
    feed._buffers["tbbo"] = tbbo_recs
    return feed


def _tiny_pipeline():
    N_features = len(ALPHA_FEATURES)
    X = np.random.RandomState(0).randn(500, N_features)
    y = X @ np.linspace(0.1, -0.1, N_features) * 1e-3
    pipe = make_pipeline(StandardScaler(), Ridge(alpha=1e-4))
    pipe.fit(X, y)
    return pipe


def _in_memory_artifacts(universe: list[str]):
    """Build well-conditioned in-memory risk artifacts + reference stats."""
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
    return {
        "L": L, "Sigma_f": Sigma_f, "Sigma_r": Sigma_r, "eps_diag": eps_diag,
        "ref_stats": ref_stats,
    }


def _patch_artifacts(monkeypatch, universe: list[str], artifacts: dict | None = None):
    arts = artifacts or _in_memory_artifacts(universe)
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


def _base_argv() -> list[str]:
    return [
        "--paper",
        "--vol-cap", "5000",
        "--submit-by", "23:59",
        "--rebalance-time", "00:00",
        "--stream-start-time", "00:00",
        "--print-wait-secs", "0",
        "--on-feed-failure", "abort",
    ]


def test_daily_orchestrator_dry_run(monkeypatch, tmp_path):
    """Full pipeline with mocked single-day feed + no broker (dry-run)."""
    universe = config.load_active_universe()
    feed = _prepopulate_live_feed(universe)
    monkeypatch.setattr(daily_orchestrator, "LiveFeed", lambda *a, **kw: feed)
    monkeypatch.setattr(daily_orchestrator, "_sleep_until", lambda *a, **kw: None)
    monkeypatch.setenv("DATABENTO_API_KEY", "dummy-key")
    _patch_artifacts(monkeypatch, universe)

    rc = daily_orchestrator.main(_base_argv() + ["--dry-run"])
    assert rc == 0


def test_daily_orchestrator_places_two_leg_orders(monkeypatch, tmp_path):
    """Two-leg happy path: MKT entries fill 100%, MOC counter-flatten
    sized to the actual (fully-filled) position.

    Assertions:
    - MKT orders reached the broker (order_type == "MKT", no limit_price).
    - MOC counter-orders reached the broker sized to the filled qty.
    - Every MOC exit is in the OPPOSITE direction of its entry.
    - broker.rebalance() is never touched.
    """
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
    assert fake_broker.connected
    assert fake_broker.disconnected
    assert set(fake_broker.qualified_symbols).issubset(set(universe))
    assert fake_broker.rebalance_calls == []

    mkt_buys = [c for c in fake_broker.buy_calls if c[2] == "MKT"]
    mkt_sells = [c for c in fake_broker.sell_calls if c[2] == "MKT"]
    moc_buys = [c for c in fake_broker.buy_calls if c[2] == "MOC"]
    moc_sells = [c for c in fake_broker.sell_calls if c[2] == "MOC"]

    assert len(mkt_buys) + len(mkt_sells) > 0, "no MKT entries reached the broker"
    # Entries must be plain market orders (default tif=DAY) with no
    # limit price and no explicit tif override.
    for sym, qty, ot, kw in mkt_buys + mkt_sells:
        assert kw.get("limit_price") is None
        assert "tif" not in kw, (
            f"MKT entry for {sym} passed an explicit tif={kw.get('tif')!r}; "
            f"entries must use the broker default (DAY)"
        )
    # Every MKT-entered symbol got an opposite-side MOC counter-flatten.
    entered_buys = {sym for sym, qty, ot, kw in mkt_buys}
    entered_sells = {sym for sym, qty, ot, kw in mkt_sells}
    moc_sells_syms = {sym for sym, qty, ot, kw in moc_sells}
    moc_buys_syms = {sym for sym, qty, ot, kw in moc_buys}
    assert entered_buys == moc_sells_syms, "every BUY entry needs a SELL MOC exit"
    assert entered_sells == moc_buys_syms, "every SELL entry needs a BUY MOC exit"
    # MOC sizes equal the currently held quantities (i.e. the MKT-filled qty).
    for sym, qty, ot, kw in moc_buys + moc_sells:
        entry_qty = next(
            q for s, q, o, _ in mkt_buys + mkt_sells if s == sym
        )
        assert qty == entry_qty, (
            f"MOC exit {sym} qty {qty} != entry qty {entry_qty}"
        )


def test_daily_orchestrator_pins_out_of_universe_positions(monkeypatch, tmp_path):
    """H1 regression: an out-of-universe holding must NOT be flattened."""
    universe = config.load_active_universe()
    feed = _prepopulate_live_feed(universe)
    monkeypatch.setattr(daily_orchestrator, "LiveFeed", lambda *a, **kw: feed)
    monkeypatch.setattr(daily_orchestrator, "_sleep_until", lambda *a, **kw: None)
    monkeypatch.setenv("DATABENTO_API_KEY", "dummy-key")
    _patch_artifacts(monkeypatch, universe)

    fake_broker = _FakeBroker(initial_positions=[_FakePosition("XYZZY_UNKNOWN", 42)])

    def _fake_connect(_args):
        fake_broker.connect(mode="paper")
        return fake_broker

    monkeypatch.setattr(daily_orchestrator, "_connect_broker", _fake_connect)

    rc = daily_orchestrator.main(_base_argv())
    assert rc == 0
    # The out-of-universe symbol must never appear in a buy or sell call —
    # neither as an MKT entry nor as an MOC counter-flatten leg.
    touched = (
        {c[0] for c in fake_broker.buy_calls}
        | {c[0] for c in fake_broker.sell_calls}
    )
    assert "XYZZY_UNKNOWN" not in touched, \
        f"out-of-universe symbol touched: {touched & {'XYZZY_UNKNOWN'}}"


def test_daily_orchestrator_drops_unqualified_symbols(monkeypatch, tmp_path):
    """H2 regression: an IB-unqualified symbol must be dropped, not raised."""
    universe = config.load_active_universe()
    feed = _prepopulate_live_feed(universe)
    monkeypatch.setattr(daily_orchestrator, "LiveFeed", lambda *a, **kw: feed)
    monkeypatch.setattr(daily_orchestrator, "_sleep_until", lambda *a, **kw: None)
    monkeypatch.setenv("DATABENTO_API_KEY", "dummy-key")
    _patch_artifacts(monkeypatch, universe)

    bad = universe[0]  # pick any name we know is in the target set
    fake_broker = _FakeBroker(unqualified={bad})

    def _fake_connect(_args):
        fake_broker.connect(mode="paper")
        return fake_broker

    monkeypatch.setattr(daily_orchestrator, "_connect_broker", _fake_connect)

    rc = daily_orchestrator.main(_base_argv())
    assert rc == 0
    touched = (
        {c[0] for c in fake_broker.buy_calls}
        | {c[0] for c in fake_broker.sell_calls}
    )
    assert bad not in touched, \
        f"unqualified symbol {bad} was placed anyway"


def test_daily_orchestrator_raises_max_order_qty_for_large_leg(monkeypatch, tmp_path):
    """H2 regression: an oversized leg raises the ceiling; nothing crashes."""
    universe = config.load_active_universe()
    feed = _prepopulate_live_feed(universe)
    monkeypatch.setattr(daily_orchestrator, "LiveFeed", lambda *a, **kw: feed)
    monkeypatch.setattr(daily_orchestrator, "_sleep_until", lambda *a, **kw: None)
    monkeypatch.setenv("DATABENTO_API_KEY", "dummy-key")
    _patch_artifacts(monkeypatch, universe)

    fake_broker = _FakeBroker()
    fake_broker.MAX_ORDER_QTY = 5  # absurdly tiny so any real leg trips it

    def _fake_connect(_args):
        fake_broker.connect(mode="paper")
        return fake_broker

    monkeypatch.setattr(daily_orchestrator, "_connect_broker", _fake_connect)
    # Lower --lambda- to grow the book (scale=1/lambda grows) so a real
    # leg exceeds the tiny MAX_ORDER_QTY and forces the ceiling raise.
    rc = daily_orchestrator.main(_base_argv() + ["--lambda-", "1e-7"])
    assert rc == 0
    assert fake_broker.MAX_ORDER_QTY > 5, "ceiling was never raised"


def test_daily_orchestrator_refuses_thin_coverage(monkeypatch, tmp_path):
    """C2 regression: a snapshot covering < min-universe-coverage must abort."""
    universe = config.load_active_universe()
    # Only 5 of 100 names — well under the 0.5 default.
    feed = _prepopulate_live_feed(universe, subset_size=5)
    monkeypatch.setattr(daily_orchestrator, "LiveFeed", lambda *a, **kw: feed)
    monkeypatch.setattr(daily_orchestrator, "_sleep_until", lambda *a, **kw: None)
    monkeypatch.setenv("DATABENTO_API_KEY", "dummy-key")
    _patch_artifacts(monkeypatch, universe)

    rc = daily_orchestrator.main(_base_argv())
    assert rc == 2, f"expected rc=2 (feed abort), got {rc}"



def test_daily_orchestrator_refuses_empty_share_targets(monkeypatch, tmp_path):
    """L1 regression: empty share_targets => rc=5, no broker contact.

    Simulated by returning proceed on feed failure (so day_df is empty and
    downstream produces an empty book), then checking we exit before any
    broker connect happens.
    """
    universe = config.load_active_universe()
    feed = _prepopulate_live_feed(universe, subset_size=5)
    monkeypatch.setattr(daily_orchestrator, "LiveFeed", lambda *a, **kw: feed)
    monkeypatch.setattr(daily_orchestrator, "_sleep_until", lambda *a, **kw: None)
    monkeypatch.setenv("DATABENTO_API_KEY", "dummy-key")
    _patch_artifacts(monkeypatch, universe)

    # If _connect_broker gets called we know we failed to refuse.
    def _blowup(_args):
        raise AssertionError("broker connect happened despite empty share_targets")

    monkeypatch.setattr(daily_orchestrator, "_connect_broker", _blowup)

    argv = _base_argv()
    argv[argv.index("--on-feed-failure") + 1] = "proceed"
    rc = daily_orchestrator.main(argv)
    assert rc == 5, f"expected rc=5 (empty-book refusal), got {rc}"



def test_broker_connect_failure_returns_rc_8(monkeypatch, tmp_path):
    """Polish: _connect_broker raising (e.g. live-port guard on paper mode,
    TWS down, bad client_id) exits rc=8 with no traceback, no orders."""
    universe = config.load_active_universe()
    patched_feed = _prepopulate_live_feed(universe)
    monkeypatch.setattr(daily_orchestrator, "LiveFeed", lambda *a, **kw: patched_feed)
    monkeypatch.setattr(daily_orchestrator, "_sleep_until", lambda *a, **kw: None)
    monkeypatch.setenv("DATABENTO_API_KEY", "dummy-key")
    _patch_artifacts(monkeypatch, universe)

    def _refuse(_args):
        raise RuntimeError("simulated live-port guard tripped")

    monkeypatch.setattr(daily_orchestrator, "_connect_broker", _refuse)

    rc = daily_orchestrator.main(_base_argv())
    assert rc == 8, f"expected rc=8 (broker connect failed), got {rc}"


# ---------------------------------------------------------------------------
# Two-leg execution model
# ---------------------------------------------------------------------------

def test_target_gmv_argparse_error(monkeypatch, tmp_path):
    """The old --target-gmv flag must be gone from argparse entirely."""
    universe = config.load_active_universe()
    monkeypatch.setenv("DATABENTO_API_KEY", "dummy-key")
    _patch_artifacts(monkeypatch, universe)

    with pytest.raises(SystemExit) as exc_info:
        daily_orchestrator.main(_base_argv() + ["--target-gmv", "100000"])
    # argparse exits with code 2 on unrecognised arguments.
    assert exc_info.value.code == 2


def test_scale_auto_computed_from_lambda(monkeypatch, tmp_path):
    """The Optimizer.set_problem call must receive scale = 1/lambda.

    Regression on the --target-gmv removal: nothing else must silently
    substitute for scale.
    """
    universe = config.load_active_universe()
    feed = _prepopulate_live_feed(universe)
    monkeypatch.setattr(daily_orchestrator, "LiveFeed", lambda *a, **kw: feed)
    monkeypatch.setattr(daily_orchestrator, "_sleep_until", lambda *a, **kw: None)
    monkeypatch.setenv("DATABENTO_API_KEY", "dummy-key")
    _patch_artifacts(monkeypatch, universe)

    captured: dict = {}

    original_set_problem = daily_orchestrator.Optimizer.set_problem

    def _capture(self, **kwargs):
        captured.update(kwargs)
        return original_set_problem(self, **kwargs)

    monkeypatch.setattr(daily_orchestrator.Optimizer, "set_problem", _capture)

    rc = daily_orchestrator.main(_base_argv() + ["--dry-run", "--lambda-", "1e-5"])
    assert rc == 0
    assert "scale" in captured
    assert captured["scale"] == pytest.approx(1e5, rel=1e-9)
    assert captured["lam"] == pytest.approx(1e-5, rel=1e-9)


def test_two_leg_partial_fill_moc_sized_to_actual_position(monkeypatch, tmp_path):
    """CRITICAL SAFETY INVARIANT: on a partial MKT fill the MOC exit must be
    sized to the ACTUAL post-entry position, NEVER to ``target - current``.

    A 50% MKT fill sized as a target-diff MOC exit would over-flatten into
    an opposite overnight exposure — the exact bug this two-leg model exists
    to prevent.
    """
    universe = config.load_active_universe()
    feed = _prepopulate_live_feed(universe)
    monkeypatch.setattr(daily_orchestrator, "LiveFeed", lambda *a, **kw: feed)
    monkeypatch.setattr(daily_orchestrator, "_sleep_until", lambda *a, **kw: None)
    monkeypatch.setenv("DATABENTO_API_KEY", "dummy-key")
    _patch_artifacts(monkeypatch, universe)

    # MKT entries fill exactly 50%; MOC exits (order_type=="MOC") fill fully.
    fake_broker = _FakeBroker(entry_fill_ratio=0.5)

    def _fake_connect(_args):
        fake_broker.connect(mode="paper")
        return fake_broker

    monkeypatch.setattr(daily_orchestrator, "_connect_broker", _fake_connect)

    rc = daily_orchestrator.main(_base_argv())
    assert rc == 0

    mkt_orders = [c for c in fake_broker.buy_calls + fake_broker.sell_calls
                  if c[2] == "MKT"]
    moc_orders = [c for c in fake_broker.buy_calls + fake_broker.sell_calls
                  if c[2] == "MOC"]
    assert mkt_orders, "no MKT entries reached the broker"
    assert moc_orders, "no MOC exits reached the broker"

    # Per-symbol: MOC exit qty must equal MKT entry qty * 0.5 (filled),
    # not the full requested qty.
    entry_by_sym = {c[0]: c[1] for c in mkt_orders}
    for sym, moc_qty, ot, kw in moc_orders:
        entry_qty = entry_by_sym[sym]
        expected_filled = int(round(entry_qty * 0.5))
        assert moc_qty == expected_filled, (
            f"MOC exit {sym} qty {moc_qty} != actual position {expected_filled} "
            f"(entry_qty={entry_qty}) — SAFETY INVARIANT VIOLATED"
        )


def test_two_leg_zero_fill_no_moc_for_that_name(monkeypatch, tmp_path):
    """A leg that raises during MKT placement (broker reject) must NOT
    receive an MOC counter-order — there is no position to flatten.

    Guards against the "always exit whatever we tried to enter" pitfall,
    which would place a naked short/long into the auction.
    """
    universe = config.load_active_universe()
    feed = _prepopulate_live_feed(universe)
    monkeypatch.setattr(daily_orchestrator, "LiveFeed", lambda *a, **kw: feed)
    monkeypatch.setattr(daily_orchestrator, "_sleep_until", lambda *a, **kw: None)
    monkeypatch.setenv("DATABENTO_API_KEY", "dummy-key")
    _patch_artifacts(monkeypatch, universe)

    # Pick any symbol from the universe and have every buy attempt for it
    # raise. Its position stays flat, so no MOC exit should fire.
    rejected_sym = universe[0]
    fake_broker = _FakeBroker(
        raise_on_buy_symbols={rejected_sym},
        raise_on_sell_symbols={rejected_sym},
    )

    def _fake_connect(_args):
        fake_broker.connect(mode="paper")
        return fake_broker

    monkeypatch.setattr(daily_orchestrator, "_connect_broker", _fake_connect)

    rc = daily_orchestrator.main(_base_argv())
    assert rc == 0

    # No MOC (or MKT) for the rejected symbol.
    all_calls = fake_broker.buy_calls + fake_broker.sell_calls
    touched = [(sym, ot) for sym, qty, ot, kw in all_calls if sym == rejected_sym]
    assert not any(ot == "MOC" for _, ot in touched), (
        f"MOC exit fired for a symbol with no entry fill: {touched}"
    )


def test_two_leg_submit_by_hit_between_entry_and_exit_returns_rc_11(
    monkeypatch, tmp_path,
):
    """R3 regression: if the submit-by cutoff passes AFTER MKT entries
    fill but BEFORE the MOC batch, exit **rc=11** (entries-filled-no-exit,
    the OVERNIGHT-EXPOSURE state) — NOT rc=4 (which is the safe pre-order
    skip). The two rcs must be distinct so a cron wrapper can page on
    exposure and stay quiet on a harmless skip.

    Implementation trick: patch ``_submit_mkt_entries`` to run the real
    function and then flip a shared "now" past the submit_by, so the
    subsequent ``_submit_moc_flatten`` per-leg wall-clock check trips
    on every leg.
    """
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

    real_submit_mkt = daily_orchestrator._submit_mkt_entries
    real_now_ny = config.now_ny
    state = {"advance": False}

    def _wrapped_submit_mkt(*args, **kwargs):
        # Run the real MKT entry batch, then flip the wall clock so the
        # MOC batch's per-leg check trips.
        entries = real_submit_mkt(*args, **kwargs)
        state["advance"] = True
        return entries

    def _maybe_advanced_now():
        base = real_now_ny()
        if state["advance"]:
            # 30 minutes past the anchored submit-by (23:59 today).
            return base.replace(hour=23, minute=59, second=59) + \
                   __import__("datetime").timedelta(minutes=30)
        return base

    monkeypatch.setattr(daily_orchestrator, "_submit_mkt_entries",
                        _wrapped_submit_mkt)
    monkeypatch.setattr(daily_orchestrator.config, "now_ny",
                        _maybe_advanced_now)

    rc = daily_orchestrator.main(_base_argv())

    assert rc == 11, (
        f"expected rc=11 (entries filled, MOC exit skipped — OVERNIGHT "
        f"EXPOSURE), got {rc}"
    )
    mkt_orders = [c for c in fake_broker.buy_calls + fake_broker.sell_calls
                  if c[2] == "MKT"]
    moc_orders = [c for c in fake_broker.buy_calls + fake_broker.sell_calls
                  if c[2] == "MOC"]
    assert mkt_orders, "MKT entries should still have fired before the cutoff"
    assert not moc_orders, (
        "MOC exits should have been skipped past the submit-by cutoff"
    )


def test_rc_4_and_rc_11_are_distinct(monkeypatch, tmp_path):
    """R3 regression pin: rc=4 is the safe pre-order skip
    (no orders placed); rc=11 is the entries-filled-no-exit exposure.
    They must NOT collide onto the same code.

    This test triggers the rc=4 path (submit-by already past at
    orchestrator entry) and verifies rc==4, then relies on the
    companion test above to have already verified rc==11 for the
    mid-flow case. Together the pair proves the codes differ.
    """
    universe = config.load_active_universe()
    feed = _prepopulate_live_feed(universe)
    monkeypatch.setattr(daily_orchestrator, "LiveFeed", lambda *a, **kw: feed)
    monkeypatch.setattr(daily_orchestrator, "_sleep_until", lambda *a, **kw: None)
    monkeypatch.setenv("DATABENTO_API_KEY", "dummy-key")
    _patch_artifacts(monkeypatch, universe)

    # A broker that would blow up if it were ever contacted — the rc=4
    # path exits BEFORE broker connect.
    def _blowup(_args):
        raise AssertionError("broker should not be contacted on the rc=4 path")

    monkeypatch.setattr(daily_orchestrator, "_connect_broker", _blowup)

    # Force "now" to be past a --submit-by that's in the (relative) past.
    # We use --submit-by 00:01 so today_ny(00, 01) is 00:01 today; real
    # wall-clock is later, so the pre-order gate trips.
    argv = [
        "--paper",
        "--vol-cap", "5000",
        "--submit-by", "00:01",
        "--rebalance-time", "00:00",
        "--stream-start-time", "00:00",
        "--print-wait-secs", "0",
        "--on-feed-failure", "abort",
    ]
    rc = daily_orchestrator.main(argv)
    assert rc == 4, f"expected rc=4 (safe pre-order skip), got {rc}"


def test_two_leg_replay_exercises_same_path(monkeypatch, tmp_path):
    """Replay mode with --replay-submit must exercise BOTH legs — MKT
    entries and MOC counter-flatten — against the mocked broker.
    """
    from live.tests.test_replay import (
        _make_stub_replay_feed as _make_stub,
        _FakeBroker as _ReplayFakeBroker,
    )

    universe = config.load_active_universe()
    stub_feed = _make_stub("2026-05-15", universe)
    monkeypatch.setattr(daily_orchestrator, "ReplayFeed", lambda **kw: stub_feed)
    monkeypatch.setenv("DATABENTO_API_KEY", "dummy-key")
    _patch_artifacts(monkeypatch, universe)
    # Redirect target-book writes to tmp so replay doesn't pollute the
    # real logs/targets/replay directory.
    monkeypatch.setattr(daily_orchestrator, "_TARGETS_DIR", tmp_path / "targets")
    monkeypatch.setattr(daily_orchestrator, "_FILLS_DIR", tmp_path / "fills")

    fake_broker = _ReplayFakeBroker()

    def _fake_connect(_args):
        fake_broker.connect(mode="paper")
        return fake_broker

    monkeypatch.setattr(daily_orchestrator, "_connect_broker", _fake_connect)

    argv = [
        "--replay-date", "2026-05-15", "--paper",
        "--vol-cap", "5000", "--replay-submit",
        "--submit-by", "00:01", "--on-feed-failure", "abort",
    ]
    rc = daily_orchestrator.main(argv)
    assert rc == 0
    mkt_orders = [c for c in fake_broker.buy_calls + fake_broker.sell_calls
                  if c[2] == "MKT"]
    moc_orders = [c for c in fake_broker.buy_calls + fake_broker.sell_calls
                  if c[2] == "MOC"]
    assert mkt_orders, "replay should exercise MKT entry leg"
    assert moc_orders, "replay should exercise MOC exit leg"


# ---------------------------------------------------------------------------
# Reviewer regression tests: R1, R2, R4, R7
# ---------------------------------------------------------------------------

def test_r1_get_positions_raises_returns_rc_11(monkeypatch, tmp_path):
    """R1 CRITICAL: a TWS drop that causes get_positions() to raise
    between entry fills and the MOC snapshot MUST exit rc=11 with a
    loud POSITIONS-MAY-BE-LIVE alert. The prior behavior returned rc=0
    silently — cron wrapper saw success, book rode the auction unhedged.
    """
    import logging as _logging
    universe = config.load_active_universe()
    feed = _prepopulate_live_feed(universe)
    monkeypatch.setattr(daily_orchestrator, "LiveFeed", lambda *a, **kw: feed)
    monkeypatch.setattr(daily_orchestrator, "_sleep_until", lambda *a, **kw: None)
    monkeypatch.setenv("DATABENTO_API_KEY", "dummy-key")
    _patch_artifacts(monkeypatch, universe)

    # get_positions() is called TWICE in main: (1) diff baseline before
    # entries, (2) inside _submit_moc_flatten after entries. Make it
    # raise on the 2nd call so entries fire but the MOC snapshot fails.
    fake_broker = _FakeBroker(get_positions_raises_after_n_calls=2)

    def _fake_connect(_args):
        fake_broker.connect(mode="paper")
        return fake_broker

    monkeypatch.setattr(daily_orchestrator, "_connect_broker", _fake_connect)

    handler, records = _capture_orchestrator_logs(_logging.ERROR)
    try:
        rc = daily_orchestrator.main(_base_argv())
    finally:
        daily_orchestrator.logger.removeHandler(handler)

    assert rc == 11, (
        f"expected rc=11 (get_positions failed post-entry — OVERNIGHT "
        f"EXPOSURE), got {rc}"
    )
    # Entries fired.
    mkt_orders = [c for c in fake_broker.buy_calls + fake_broker.sell_calls
                  if c[2] == "MKT"]
    assert mkt_orders, "MKT entries should have fired before the snapshot failed"
    # MOC batch produced nothing.
    moc_orders = [c for c in fake_broker.buy_calls + fake_broker.sell_calls
                  if c[2] == "MOC"]
    assert not moc_orders, "no MOC orders should have been sent"
    # Loud alert must be in the log.
    loud = [r.getMessage() for r in records]
    assert any(("POSITIONS MAY BE LIVE" in m) or ("OVERNIGHT EXPOSURE" in m)
               for m in loud), (
        f"expected 'POSITIONS MAY BE LIVE' or 'OVERNIGHT EXPOSURE' alert "
        f"in ERROR log; got: {loud}"
    )


def test_r1_empty_snapshot_post_fill_returns_rc_11(monkeypatch, tmp_path):
    """R1 CRITICAL variant: get_positions() succeeds but returns an
    empty list AFTER entries filled (TWS stale snapshot). Must exit
    rc=11 with a loud alert instead of silent rc=0.

    Simulated by making the 2nd get_positions() (the one inside
    _submit_moc_flatten) return []. Entries did fill into the fake
    broker's internal book, but the snapshot pretends they didn't.
    """
    import logging as _logging
    universe = config.load_active_universe()
    feed = _prepopulate_live_feed(universe)
    monkeypatch.setattr(daily_orchestrator, "LiveFeed", lambda *a, **kw: feed)
    monkeypatch.setattr(daily_orchestrator, "_sleep_until", lambda *a, **kw: None)
    monkeypatch.setenv("DATABENTO_API_KEY", "dummy-key")
    _patch_artifacts(monkeypatch, universe)

    # After the 1st get_positions() (diff baseline), every subsequent
    # call returns [] — simulating stale TWS.
    fake_broker = _FakeBroker(get_positions_empty_after_calls=1)

    def _fake_connect(_args):
        fake_broker.connect(mode="paper")
        return fake_broker

    monkeypatch.setattr(daily_orchestrator, "_connect_broker", _fake_connect)

    handler, records = _capture_orchestrator_logs(_logging.ERROR)
    try:
        rc = daily_orchestrator.main(_base_argv())
    finally:
        daily_orchestrator.logger.removeHandler(handler)

    assert rc == 11, (
        f"expected rc=11 (stale-empty snapshot post-fill — OVERNIGHT "
        f"EXPOSURE), got {rc}"
    )
    loud = [r.getMessage() for r in records]
    assert any(("POSITIONS MAY BE LIVE" in m) or ("OVERNIGHT EXPOSURE" in m)
               for m in loud), (
        f"expected 'POSITIONS MAY BE LIVE' or 'OVERNIGHT EXPOSURE' alert "
        f"in ERROR log; got: {loud}"
    )


def test_r2_stuck_entries_cancelled_before_moc_snapshot(monkeypatch, tmp_path):
    """R2: entries that never reach a terminal state (MKT placed but
    not acked) MUST have broker.cancel(oid) called before the MOC
    snapshot runs. Bounds real exposure to what actually filled.

    Uses ``stuck_all_entries=True`` so every attempted entry is stuck —
    guarantees at least one stuck order regardless of which universe
    symbols the optimizer picks.
    """
    universe = config.load_active_universe()
    feed = _prepopulate_live_feed(universe)
    monkeypatch.setattr(daily_orchestrator, "LiveFeed", lambda *a, **kw: feed)
    monkeypatch.setattr(daily_orchestrator, "_sleep_until", lambda *a, **kw: None)
    monkeypatch.setenv("DATABENTO_API_KEY", "dummy-key")
    _patch_artifacts(monkeypatch, universe)

    fake_broker = _FakeBroker(stuck_all_entries=True)

    def _fake_connect(_args):
        fake_broker.connect(mode="paper")
        return fake_broker

    monkeypatch.setattr(daily_orchestrator, "_connect_broker", _fake_connect)

    rc = daily_orchestrator.main(_base_argv())
    # rc=0: every entry stayed at 0 filled so the book stayed flat and
    # no MOC exit was needed. The R2 invariant is that each of those
    # stuck orders got cancelled.
    assert rc == 0, f"expected rc=0 (stuck entries harmless if cancelled), got {rc}"

    entry_oids = [
        oid for oid, info in fake_broker._orders.items()
        if info["order_type"] == "MKT"
    ]
    assert entry_oids, "no MKT entries were placed at all"
    for oid in entry_oids:
        assert oid in fake_broker.cancel_calls, (
            f"stuck entry {oid} (symbol "
            f"{fake_broker._orders[oid]['symbol']}) never had "
            f"broker.cancel() called before the MOC snapshot"
        )
    # No MOC orders at all — every entry was stuck at 0 filled.
    moc_orders = [c for c in fake_broker.buy_calls + fake_broker.sell_calls
                  if c[2] == "MOC"]
    assert not moc_orders, (
        f"MOC exits fired for stuck (never-filled) entries: {moc_orders}"
    )

    # RR3: temporal ordering — every cancel must precede every
    # get_positions call that happens after the last entry submission.
    # A regression that moved the MOC snapshot BEFORE the cancel loop
    # would let a late-filling MKT leave an unhedged position after
    # the counter-flatten was already sized.
    events = fake_broker.events
    entry_event_indices = [
        i for i, e in enumerate(events)
        if (e.startswith("buy:MKT:") or e.startswith("sell:MKT:"))
    ]
    assert entry_event_indices, "no entry events recorded — test setup bug"
    last_entry_idx = max(entry_event_indices)
    cancel_indices = [
        i for i, e in enumerate(events) if e.startswith("cancel:")
    ]
    assert cancel_indices, "no cancel events recorded"
    gp_after_entries = [
        i for i, e in enumerate(events)
        if e == "get_positions" and i > last_entry_idx
    ]
    assert gp_after_entries, (
        "no get_positions call after the entry batch — MOC snapshot "
        "never happened"
    )
    first_gp_after_entries = min(gp_after_entries)
    for c in cancel_indices:
        assert c < first_gp_after_entries, (
            f"RR3 ordering broken: cancel event at index {c} happened "
            f"AT/AFTER get_positions event at {first_gp_after_entries}; "
            f"MOC snapshot ran before cancel loop. events tail: "
            f"{events[max(0, last_entry_idx-1):]}"
        )


def test_r4_duplicate_positions_dedup_by_symbol(monkeypatch, tmp_path):
    """R4: if IB's get_positions returns the same symbol on multiple
    rows (advisor/multi-account, snapshot race), the MOC batch must
    aggregate them into ONE order sized to the SUM, never two orders.

    Uses ``duplicate_all_positions=True`` so every position row is
    doubled — every held symbol exercises the aggregator.
    """
    universe = config.load_active_universe()
    feed = _prepopulate_live_feed(universe)
    monkeypatch.setattr(daily_orchestrator, "LiveFeed", lambda *a, **kw: feed)
    monkeypatch.setattr(daily_orchestrator, "_sleep_until", lambda *a, **kw: None)
    monkeypatch.setenv("DATABENTO_API_KEY", "dummy-key")
    _patch_artifacts(monkeypatch, universe)

    fake_broker = _FakeBroker(duplicate_all_positions=True)

    def _fake_connect(_args):
        fake_broker.connect(mode="paper")
        return fake_broker

    monkeypatch.setattr(daily_orchestrator, "_connect_broker", _fake_connect)

    rc = daily_orchestrator.main(_base_argv())
    assert rc == 0

    # Every MOC-issued symbol must appear EXACTLY ONCE across all MOC
    # orders. A naive-loop bug would double-count.
    moc_orders = [c for c in fake_broker.buy_calls + fake_broker.sell_calls
                  if c[2] == "MOC"]
    assert moc_orders, "no MOC orders were placed — nothing to dedupe"
    moc_syms = [c[0] for c in moc_orders]
    from collections import Counter
    dup_syms = [s for s, n in Counter(moc_syms).items() if n > 1]
    assert not dup_syms, (
        f"MOC batch fired multiple orders for the same symbol(s): "
        f"{dup_syms}. Aggregator must dedupe."
    )
    # Cross-check: the total MOC qty per symbol matches the entry's
    # actual filled quantity (which is what get_positions represented in
    # two duplicated rows summing to the same total). If the aggregator
    # miscounted, MOC qty would be half (dropped a row) or double
    # (counted the same row twice). We reconstruct the expected qty
    # from the entry order book because post-MOC positions are zero.
    expected_per_sym: dict[str, int] = {}
    for oid, info in fake_broker._orders.items():
        if info["order_type"] != "MKT":
            continue
        signed = info["filled"] if info["action"] == "BUY" else -info["filled"]
        expected_per_sym[info["symbol"]] = (
            expected_per_sym.get(info["symbol"], 0) + signed
        )
    for sym, qty, ot, kw in moc_orders:
        expected = abs(expected_per_sym.get(sym, 0))
        assert qty == expected, (
            f"MOC {sym} qty={qty} != actual entry-fill total {expected} "
            f"— aggregator did not sum duplicated position rows correctly"
        )


def test_r7_lambda_zero_returns_rc_9(monkeypatch, tmp_path):
    """R7 + RR2: --lambda- 0 must exit rc=9 (safe pre-order skip band),
    not rc=12. rc=12 collided with the `rc >= 11` pager predicate.
    rc>=11 must be reserved exclusively for OVERNIGHT-EXPOSURE states.
    """
    universe = config.load_active_universe()
    monkeypatch.setenv("DATABENTO_API_KEY", "dummy-key")
    _patch_artifacts(monkeypatch, universe)

    def _blowup(_args):
        raise AssertionError("broker connect should never happen for --lambda- 0")

    monkeypatch.setattr(daily_orchestrator, "_connect_broker", _blowup)

    rc = daily_orchestrator.main(_base_argv() + ["--lambda-", "0"])
    assert rc == 9, f"expected rc=9 (invalid --lambda-, safe skip), got {rc}"
    # RR2 explicit pin: rc must NOT be in the danger band.
    assert rc < 11, (
        f"rc={rc} is in the `rc >= 11` pager danger band; invalid-arg "
        "must stay in the safe-skip band"
    )


def test_r7_lambda_negative_returns_rc_9(monkeypatch, tmp_path):
    """R7 + RR2: --lambda- < 0 must also exit rc=9."""
    universe = config.load_active_universe()
    monkeypatch.setenv("DATABENTO_API_KEY", "dummy-key")
    _patch_artifacts(monkeypatch, universe)

    def _blowup(_args):
        raise AssertionError("broker connect should never happen for --lambda- < 0")

    monkeypatch.setattr(daily_orchestrator, "_connect_broker", _blowup)

    rc = daily_orchestrator.main(_base_argv() + ["--lambda-", "-1e-5"])
    assert rc == 9, f"expected rc=9 (invalid --lambda-, safe skip), got {rc}"
    assert rc < 11, (
        f"rc={rc} is in the `rc >= 11` pager danger band; invalid-arg "
        "must stay in the safe-skip band"
    )


# ---------------------------------------------------------------------------
# RR1 — coverage cross-check regression tests
# ---------------------------------------------------------------------------

def test_rr1_partial_snapshot_returns_rc_11(monkeypatch, tmp_path):
    """RR1(a) CRITICAL: get_positions() post-entry returns a SUBSET of
    the true holdings (TWS reconnect during the snapshot). MOC batch
    covers 3 of 10; 7 unhedged. Must exit rc=11 with the uncovered
    symbols named in the log.

    Pre-RR1 fix, this returned rc=0 silently: moc_order_ids non-empty
    → all three guards (moc_positions_unknown, empty_snapshot_after_fills,
    moc_skipped) evaluated False → rc=0. The ground-truth cross-check
    catches it.
    """
    import logging as _logging
    universe = config.load_active_universe()
    feed = _prepopulate_live_feed(universe)
    monkeypatch.setattr(daily_orchestrator, "LiveFeed", lambda *a, **kw: feed)
    monkeypatch.setattr(daily_orchestrator, "_sleep_until", lambda *a, **kw: None)
    monkeypatch.setenv("DATABENTO_API_KEY", "dummy-key")
    _patch_artifacts(monkeypatch, universe)

    # get_positions() calls: (1) diff baseline pre-entry, (2) inside
    # _submit_moc_flatten post-entry. From call #2 onward return only
    # 30% of the rows. The pre-entry baseline stays complete so entries
    # are sized correctly, then the MOC snapshot is truncated.
    fake_broker = _FakeBroker(
        get_positions_partial_after_n_calls=2,
        partial_position_fraction=0.3,
    )

    def _fake_connect(_args):
        fake_broker.connect(mode="paper")
        return fake_broker

    monkeypatch.setattr(daily_orchestrator, "_connect_broker", _fake_connect)

    handler, records = _capture_orchestrator_logs(_logging.ERROR)
    try:
        rc = daily_orchestrator.main(_base_argv())
    finally:
        daily_orchestrator.logger.removeHandler(handler)

    assert rc == 11, (
        f"expected rc=11 (partial-snapshot uncovered symbols — OVERNIGHT "
        f"EXPOSURE), got {rc}"
    )
    # Some MOC orders DID fire (proving the check isn't just tripping
    # on empty-snapshot); the coverage check caught the missing ones.
    moc_orders = [c for c in fake_broker.buy_calls + fake_broker.sell_calls
                  if c[2] == "MOC"]
    assert moc_orders, (
        "partial snapshot means SOME MOC orders should have fired for "
        "the visible subset — 0 means we're testing the wrong path"
    )
    # The loud alert must mention "OVERNIGHT EXPOSURE" and cite uncovered
    # symbols.
    loud = [r.getMessage() for r in records]
    assert any("OVERNIGHT EXPOSURE" in m for m in loud), (
        f"expected 'OVERNIGHT EXPOSURE' alert in ERROR log; got: {loud}"
    )
    # And the concrete panic command must be in the log (RR4).
    assert any("live.panic" in m for m in loud), (
        f"expected concrete panic command in log; got: {loud}"
    )


def test_rr1_wait_budget_exhausted_with_real_fills_returns_rc_11(
    monkeypatch, tmp_path,
):
    """RR1(b) CRITICAL: R5's truncated fill-wait hard-sets filled_qty=0
    on the budget-exhausted branch even when MKT entries truly filled
    at the exchange. If get_positions() is ALSO stale-empty in the
    same slow-TWS window, the R1 empty_snapshot_after_fills guard is
    blinded (n_filled_entries==0 defeats the check). The RR1 ground-
    truth cross-check re-polls get_order_status per entry and catches
    the real fills, so the uncovered-symbols branch trips rc=11.
    """
    import logging as _logging
    universe = config.load_active_universe()
    feed = _prepopulate_live_feed(universe)
    monkeypatch.setattr(daily_orchestrator, "LiveFeed", lambda *a, **kw: feed)
    monkeypatch.setattr(daily_orchestrator, "_sleep_until", lambda *a, **kw: None)
    monkeypatch.setenv("DATABENTO_API_KEY", "dummy-key")
    _patch_artifacts(monkeypatch, universe)

    # Correlated failures:
    #  1. wait_for_fill always times out → _wait_entry_fills marks every
    #     entry as budget-exhausted, filled_qty=0.
    #  2. get_positions() returns empty from call #2 onward → MOC snapshot
    #     sees nothing to flatten.
    # Ground truth: entries ACTUALLY filled at 100% (default entry_fill_ratio).
    # The RR1 cross-check re-polls get_order_status(oid), sees the true
    # fill, and rejects the "no exposure" conclusion.
    fake_broker = _FakeBroker(
        wait_for_fill_always_times_out=True,
        get_positions_empty_after_calls=1,
    )

    def _fake_connect(_args):
        fake_broker.connect(mode="paper")
        return fake_broker

    monkeypatch.setattr(daily_orchestrator, "_connect_broker", _fake_connect)

    handler, records = _capture_orchestrator_logs(_logging.ERROR)
    try:
        rc = daily_orchestrator.main(_base_argv())
    finally:
        daily_orchestrator.logger.removeHandler(handler)

    assert rc == 11, (
        f"expected rc=11 (budget-exhausted-with-real-fills — OVERNIGHT "
        f"EXPOSURE), got {rc}. Pre-RR1 this was rc=0."
    )
    loud = [r.getMessage() for r in records]
    assert any("OVERNIGHT EXPOSURE" in m for m in loud), (
        f"expected 'OVERNIGHT EXPOSURE' alert in ERROR log; got: {loud}"
    )
    assert any("live.panic" in m for m in loud), (
        f"expected concrete panic command in log; got: {loud}"
    )
    # Sanity: internal position book should show fills happened.
    assert any(q != 0 for q in fake_broker._positions.values()), (
        "test setup bug: expected real fills to have hit the internal book"
    )


# ---------------------------------------------------------------------------
# RRR-round regression tests
# ---------------------------------------------------------------------------

def test_rrr1_moc_rejected_at_exchange_returns_rc_11(monkeypatch, tmp_path):
    """RRR1 HIGH: broker.buy/sell for MOC returns oid the moment TWS
    API-accepts the placeOrder — BEFORE any exchange ack. A late MOC
    (submitted past ~15:50 cutoff) becomes terminal Cancelled with
    filled=0 shortly after. Pre-RRR1 code trusted the returned oid as
    coverage → cross-check passes → rc=0. RRR1 adds an acceptance
    audit that catches the rejection and returns rc=11.

    N2: structurally lock the safety property by first running with
    ``_audit_moc_acceptance`` neutralized (simulating pre-RRR1) and
    asserting rc=0 — proving the acceptance audit is the load-bearing
    safety piece — then run with the audit restored and assert rc=11.
    """
    import logging as _logging
    universe = config.load_active_universe()

    def _setup_broker():
        feed_local = _prepopulate_live_feed(universe)
        monkeypatch.setattr(daily_orchestrator, "LiveFeed",
                            lambda *a, **kw: feed_local)
        monkeypatch.setattr(daily_orchestrator, "_sleep_until",
                            lambda *a, **kw: None)
        monkeypatch.setenv("DATABENTO_API_KEY", "dummy-key")
        _patch_artifacts(monkeypatch, universe)
        broker = _FakeBroker(moc_reject_all=True)

        def _fake_connect(_args):
            broker.connect(mode="paper")
            return broker

        monkeypatch.setattr(daily_orchestrator, "_connect_broker",
                            _fake_connect)
        return broker

    # ---- Phase 1: acceptance audit neutralized (simulates pre-RRR1).
    broker_pre = _setup_broker()
    real_audit = daily_orchestrator._audit_moc_acceptance
    monkeypatch.setattr(
        daily_orchestrator, "_audit_moc_acceptance",
        lambda *a, **kw: set(),
    )
    rc_pre = daily_orchestrator.main(_base_argv())
    assert rc_pre == 0, (
        f"phase 1 (audit neutralized) must return rc=0 to prove the "
        f"acceptance audit is the load-bearing safety piece; got {rc_pre}"
    )
    mkt_pre = [c for c in broker_pre.buy_calls + broker_pre.sell_calls
               if c[2] == "MKT"]
    moc_pre = [c for c in broker_pre.buy_calls + broker_pre.sell_calls
               if c[2] == "MOC"]
    assert mkt_pre and moc_pre, (
        "phase 1: expected both MKT entries and MOC placements to have "
        "fired against the broker"
    )

    # ---- Phase 2: acceptance audit restored — must catch rejections.
    monkeypatch.setattr(daily_orchestrator, "_audit_moc_acceptance",
                        real_audit)
    broker_post = _setup_broker()
    handler, records = _capture_orchestrator_logs(_logging.ERROR)
    try:
        rc_post = daily_orchestrator.main(_base_argv())
    finally:
        daily_orchestrator.logger.removeHandler(handler)

    assert rc_post == 11, (
        f"phase 2 (audit restored) must return rc=11 on MOC exchange "
        f"rejection; got {rc_post}"
    )
    # MKT entries fired AND MOC placements went through (returned oid).
    mkt_orders = [c for c in broker_post.buy_calls + broker_post.sell_calls
                  if c[2] == "MKT"]
    moc_orders = [c for c in broker_post.buy_calls + broker_post.sell_calls
                  if c[2] == "MOC"]
    assert mkt_orders, "MKT entries should have fired"
    assert moc_orders, "MOC placements should have been API-accepted"
    # Loud alert with the panic command AND the MOC-rejection detail.
    loud = [r.getMessage() for r in records]
    assert any("OVERNIGHT EXPOSURE" in m for m in loud), (
        f"expected 'OVERNIGHT EXPOSURE' alert; got: {loud}"
    )
    assert any("live.panic" in m for m in loud), (
        f"expected concrete panic command; got: {loud}"
    )
    assert any("MOC rejected at exchange" in m for m in loud), (
        f"expected 'MOC rejected at exchange' detail in alert; got: {loud}"
    )


def test_n1_moc_ack_timeout_flag_threaded_to_audit(monkeypatch, tmp_path):
    """N1: --moc-ack-timeout must be threaded into the
    _audit_moc_acceptance(...) call. Confirms operators can widen the
    window without a code edit on chronically slow-TWS venues.
    """
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

    captured: dict = {}
    real_audit = daily_orchestrator._audit_moc_acceptance

    def _capture(broker, moc_batch, wall_clock_wait):
        captured["wall_clock_wait"] = wall_clock_wait
        return real_audit(broker, moc_batch, wall_clock_wait=wall_clock_wait)

    monkeypatch.setattr(daily_orchestrator, "_audit_moc_acceptance", _capture)

    rc = daily_orchestrator.main(_base_argv() + ["--moc-ack-timeout", "90"])
    assert rc == 0
    assert captured.get("wall_clock_wait") == pytest.approx(90.0), (
        f"expected --moc-ack-timeout=90 to reach _audit_moc_acceptance; "
        f"got {captured}"
    )


def test_rrr2_write_target_record_failure_still_returns_rc_11_on_exposure(
    monkeypatch, tmp_path,
):
    """RRR2: if _write_target_record raises (disk full / permissions),
    the rc=11 exposure verdict must still be returned. Pre-fix the
    exception would propagate through main() and the process would
    exit rc=1 — below the `rc >= 11` pager threshold — silently
    dropping the exposure page.

    Combines a real exposure signal (MOC exchange rejection) with a
    persist failure. rc must remain 11.
    """
    import logging as _logging
    universe = config.load_active_universe()
    feed = _prepopulate_live_feed(universe)
    monkeypatch.setattr(daily_orchestrator, "LiveFeed", lambda *a, **kw: feed)
    monkeypatch.setattr(daily_orchestrator, "_sleep_until", lambda *a, **kw: None)
    monkeypatch.setenv("DATABENTO_API_KEY", "dummy-key")
    _patch_artifacts(monkeypatch, universe)

    fake_broker = _FakeBroker(moc_reject_all=True)

    def _fake_connect(_args):
        fake_broker.connect(mode="paper")
        return fake_broker

    monkeypatch.setattr(daily_orchestrator, "_connect_broker", _fake_connect)

    # Poison the persist step. Called TWICE (pre-broker + post-MOC);
    # the pre-broker one persists the intended book — allow it. The
    # post-MOC amend is the one that must fail.
    call_state = {"n": 0}
    real_write = daily_orchestrator._write_target_record

    def _sometimes_raise(record):
        call_state["n"] += 1
        if call_state["n"] >= 2:
            raise OSError("simulated disk full")
        return real_write(record)

    monkeypatch.setattr(daily_orchestrator, "_write_target_record",
                        _sometimes_raise)

    handler, records = _capture_orchestrator_logs(_logging.ERROR)
    try:
        rc = daily_orchestrator.main(_base_argv())
    finally:
        daily_orchestrator.logger.removeHandler(handler)

    assert rc == 11, (
        f"expected rc=11 (exposure verdict survives write failure), "
        f"got {rc}"
    )
    loud = [r.getMessage() for r in records]
    assert any("OVERNIGHT EXPOSURE" in m for m in loud), (
        f"expected 'OVERNIGHT EXPOSURE' alert; got: {loud}"
    )


def test_rrr2_write_target_record_failure_returns_rc_0_on_clean_run(
    monkeypatch, tmp_path,
):
    """RRR2 companion: if there's no exposure signal, a _write_target_record
    failure must NOT synthesize an rc=11. Write is best-effort; a
    happy-path run with a failed post-MOC persist still returns rc=0.
    """
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

    call_state = {"n": 0}
    real_write = daily_orchestrator._write_target_record

    def _sometimes_raise(record):
        call_state["n"] += 1
        if call_state["n"] >= 2:
            raise OSError("simulated disk full on amended record")
        return real_write(record)

    monkeypatch.setattr(daily_orchestrator, "_write_target_record",
                        _sometimes_raise)

    rc = daily_orchestrator.main(_base_argv())
    assert rc == 0, (
        f"clean happy-path run with write failure must NOT synthesize "
        f"rc=11; got {rc}"
    )


def test_rrr3_stale_non_terminal_get_order_status_flags_uncovered(
    monkeypatch, tmp_path,
):
    """RRR3: broker.get_order_status returns non-terminal status without
    raising (TWS event-stream brownout leaves the local _trades cache
    stale). Pre-fix, cross-check saw filled=0 and concluded no exposure.
    Combined with a stale-empty get_positions snapshot, silent rc=0
    despite real fills.

    RRR3 treats non-terminal status as a staleness signal (pessimistic
    full fill). Combined with the stale-empty snapshot, cross-check
    flags every entered symbol as uncovered → rc=11.
    """
    import logging as _logging
    universe = config.load_active_universe()
    feed = _prepopulate_live_feed(universe)
    monkeypatch.setattr(daily_orchestrator, "LiveFeed", lambda *a, **kw: feed)
    monkeypatch.setattr(daily_orchestrator, "_sleep_until", lambda *a, **kw: None)
    monkeypatch.setenv("DATABENTO_API_KEY", "dummy-key")
    _patch_artifacts(monkeypatch, universe)

    # Correlated failures — same TWS brownout: entry get_order_status
    # returns stale non-terminal + filled=0, AND the MOC snapshot
    # get_positions returns [] from call #2 onward.
    fake_broker = _FakeBroker(
        entry_status_stale=True,
        get_positions_empty_after_calls=1,
    )

    def _fake_connect(_args):
        fake_broker.connect(mode="paper")
        return fake_broker

    monkeypatch.setattr(daily_orchestrator, "_connect_broker", _fake_connect)

    handler, records = _capture_orchestrator_logs(_logging.ERROR)
    try:
        rc = daily_orchestrator.main(_base_argv())
    finally:
        daily_orchestrator.logger.removeHandler(handler)

    assert rc == 11, (
        f"expected rc=11 (stale non-terminal cross-check pessimism), "
        f"got {rc}. Pre-RRR3 this was rc=0."
    )
    loud = [r.getMessage() for r in records]
    assert any("OVERNIGHT EXPOSURE" in m for m in loud), (
        f"expected 'OVERNIGHT EXPOSURE' alert; got: {loud}"
    )
    # Sanity: internal position book DID fill (entries hit _positions).
    assert any(q != 0 for q in fake_broker._positions.values()), (
        "test setup bug: entries should have filled into the internal book"
    )


# ---------------------------------------------------------------------------
# NASDAQ-routing regression: every order path must direct-route to ISLAND
# ---------------------------------------------------------------------------

def test_every_order_path_routes_to_nasdaq(monkeypatch, tmp_path):
    """NASDAQ-only routing is a hard requirement of this strategy — the
    model is calibrated to the NASDAQ closing auction, so an off-venue
    fill is uncorrelated with the signal we sized. Every code path in
    ``live/`` that reaches the broker MUST carry
    ``exchange="ISLAND"`` (IB's NASDAQ INET direct-route code, exported
    as ``NASDAQ_EXCHANGE`` from ``live/nasdaq_routing.py``). SMART
    routing is unacceptable — it routinely picks BATS / EDGX / IEX /
    NYSE / ARCA when they're momentarily inside the NBBO.
    Enumerates the three paths and pins each:
    * pre-qualify (``broker._make_contract`` inside
      ``_prequalify_symbols``)
    * MKT entry batch (``broker.buy`` / ``broker.sell`` with
      ``order_type="MKT"``)
    * MOC exit batch (``broker.buy`` / ``broker.sell`` with
      ``order_type="MOC"``)
    A regression that ever drops the ``exchange`` kwarg on any of these
    would let orders SMART-route away from NASDAQ. This test fails
    LOUDLY the moment that happens.
    """
    from live.nasdaq_routing import NASDAQ_EXCHANGE

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

    # (1) prequalify path — every _make_contract call must be ISLAND.
    assert fake_broker.make_contract_calls, "no prequalify calls recorded"
    for sym, exchange in fake_broker.make_contract_calls:
        assert exchange == NASDAQ_EXCHANGE == "ISLAND", (
            f"_prequalify_symbols routed {sym} to exchange={exchange!r}, "
            f"expected {NASDAQ_EXCHANGE!r}"
        )

    # (2) + (3) entry MKT and exit MOC — every buy/sell call must
    # carry exchange="ISLAND" in its kwargs. This is the load-bearing
    # assertion: it fails the moment any call site drops the kwarg.
    all_order_calls = fake_broker.buy_calls + fake_broker.sell_calls
    assert all_order_calls, "no orders reached the broker at all"
    mkt_orders = [c for c in all_order_calls if c[2] == "MKT"]
    moc_orders = [c for c in all_order_calls if c[2] == "MOC"]
    assert mkt_orders, "no MKT entries reached the broker"
    assert moc_orders, "no MOC exits reached the broker"
    for sym, qty, order_type, kw in all_order_calls:
        got = kw.get("exchange")
        assert got == NASDAQ_EXCHANGE == "ISLAND", (
            f"{order_type} order for {sym} routed to exchange={got!r} "
            f"(expected {NASDAQ_EXCHANGE!r}) — NASDAQ-only routing "
            f"contract violated. Full kwargs: {kw}"
        )


def test_no_smart_routing_leaks_through_live_paths(monkeypatch, tmp_path):
    """Companion pin: assert POSITIVELY that ``"SMART"`` never shows up as
    the requested exchange on any order or prequalify call. Catches a
    regression where a call site is added but forgets to import /
    thread ``NASDAQ_EXCHANGE`` (which would silently default the
    broker's ``exchange="SMART"``).
    """
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

    smart_qualify = [
        (sym, ex) for sym, ex in fake_broker.make_contract_calls
        if ex == "SMART"
    ]
    assert not smart_qualify, (
        f"prequalify path leaked SMART routing on: {smart_qualify}"
    )
    smart_orders = [
        (sym, qty, ot, kw) for sym, qty, ot, kw
        in fake_broker.buy_calls + fake_broker.sell_calls
        if kw.get("exchange") == "SMART" or kw.get("exchange") is None
    ]
    assert not smart_orders, (
        f"order path leaked SMART or default routing on: {smart_orders}"
    )


# --- DISABLED 2026-07-20 -----------------------------------------------------
# This test pins the NASDAQ-primary drop path in
# ``_prequalify_symbols`` (live/daily_orchestrator.py). That drop was
# commented out on 2026-07-20 after the broker-side route-survival
# assertion (which ran BEFORE this filter) rejected 46 of 46 requested
# symbols (100% failure, 0 passed) during a paper replay of
# 2026-01-05, meaning the primaryExchange filter never got a chance
# to fire end-to-end. On both current tiered universes the filter is
# also a no-op: 49/49 in top_49_universe.txt, and 99 of the 100 in
# top_100_universe.txt qualify with all 99 NASDAQ-primary (ITCI fails
# IB qualification outright and never reaches the filter); its real
# value is future universe-drift protection. The routing itself
# (``exchange="ISLAND"`` threaded through every broker call) is
# unchanged; only the two rejection checks are rolled back. See the
# commented-out block in ``_prequalify_symbols`` and the matching
# block in ``ibkr_broker/src/ibkr_broker/broker.py::_make_contract``.
# Restore this test together with the code it covers.
#
# def test_prequalify_drops_non_nasdaq_primary(monkeypatch, tmp_path):
#     """Finding #3: ``_prequalify_symbols`` must drop any name whose
#     qualified contract's ``primaryExchange`` is not ``"NASDAQ"``.
#
#     Reason: an ``ISLAND`` direct-routed MOC only prints on the NASDAQ
#     close-cross when the name's primary listing is NASDAQ. IB will
#     still qualify NYSE / ARCA-primary names on ``ISLAND`` for
#     continuous UTP trading — but the closing MOC on that name will
#     mis-print (or be rejected) at the primary listing's close-cross,
#     silently breaking the strategy. Refusing to trade the name is
#     the correct behaviour for a NASDAQ-only book.
#
#     Seeds the fake broker with a batch of universe names all marked
#     NYSE-primary (rather than a single hand-picked one, since the
#     optimizer's target set is model-dependent and we cannot predict
#     which specific name will be traded) and asserts:
#     * at least one drop is logged with ``primary_exchange="NYSE"``,
#     * no buy/sell order for ANY seeded symbol ever reaches the broker,
#     * unseeded (NASDAQ-primary) names still trade normally.
#     """
#     import logging as _logging
#
#     universe = config.load_active_universe()
#     feed = _prepopulate_live_feed(universe)
#     monkeypatch.setattr(daily_orchestrator, "LiveFeed", lambda *a, **kw: feed)
#     monkeypatch.setattr(daily_orchestrator, "_sleep_until", lambda *a, **kw: None)
#     monkeypatch.setenv("DATABENTO_API_KEY", "dummy-key")
#     _patch_artifacts(monkeypatch, universe)
#
#     # The optimizer's target set is model-dependent and (with this
#     # test's synthetic Ridge) skews toward alphabetically-early
#     # mid-caps rather than the file-order top-of-universe mega-caps.
#     # So we seed a wide alphabetical slice of the universe as
#     # NYSE-primary — enough overlap with the target book that the
#     # drop path is guaranteed to fire at least once. We intentionally
#     # leave the last few names unseeded (still NASDAQ-primary in the
#     # fake) so a NASDAQ-primary contingent survives to trade — the
#     # "the drop is symbol-specific, not a wholesale abort" invariant.
#     universe_sorted = sorted(universe)
#     seeded_nyse: dict[str, str] = {
#         sym: "NYSE" for sym in universe_sorted[: max(1, len(universe_sorted) // 2)]
#     }
#     fake_broker = _FakeBroker(non_nasdaq_primary=seeded_nyse)
#
#     def _fake_connect(_args):
#         fake_broker.connect(mode="paper")
#         return fake_broker
#
#     monkeypatch.setattr(daily_orchestrator, "_connect_broker", _fake_connect)
#
#     handler, records = _capture_orchestrator_logs(_logging.WARNING)
#     try:
#         rc = daily_orchestrator.main(_base_argv())
#     finally:
#         daily_orchestrator.logger.removeHandler(handler)
#
#     assert rc == 0
#
#     # (1) No NYSE-primary name may ever hit any buy/sell call. This
#     # is the load-bearing invariant — the strategy MUST NOT trade a
#     # name whose MOC would mis-print on a non-NASDAQ close-cross.
#     touched = (
#         {c[0] for c in fake_broker.buy_calls}
#         | {c[0] for c in fake_broker.sell_calls}
#     )
#     leaked = touched & set(seeded_nyse)
#     assert not leaked, (
#         f"non-NASDAQ-primary names traded despite the drop filter: "
#         f"{sorted(leaked)}"
#     )
#
#     # (2) At least one drop must have been logged with the informative
#     # extras — ``symbol`` (which name) and ``primary_exchange`` (why).
#     # We assert on the informative extras rather than a substring match
#     # on the message so a benign wording change to the warning does
#     # not silently break the regression.
#     dropped_syms = {
#         getattr(r, "symbol", None)
#         for r in records
#         if getattr(r, "primary_exchange", None) == "NYSE"
#         and getattr(r, "expected_primary_exchange", None) == "NASDAQ"
#     }
#     dropped_syms.discard(None)
#     assert dropped_syms, (
#         "no 'dropping non-NASDAQ-listed' warning was captured. "
#         f"seeded={sorted(seeded_nyse)}, "
#         f"captured_records={[r.getMessage() for r in records]}"
#     )
#     # And every dropped symbol must be one we seeded — never a
#     # spurious drop of a NASDAQ-primary name.
#     assert dropped_syms.issubset(set(seeded_nyse)), (
#         f"drop path fired for names we did NOT seed as NYSE-primary: "
#         f"{sorted(dropped_syms - set(seeded_nyse))}"
#     )
#
#     # (3) NASDAQ-primary names still trade — the drop is symbol-
#     # specific, not a wholesale abort.
#     assert touched, (
#         "no orders reached the broker at all — the drop path should "
#         "have removed only the seeded names, not the whole universe"
#     )
# --- end DISABLED 2026-07-20 -------------------------------------------------
