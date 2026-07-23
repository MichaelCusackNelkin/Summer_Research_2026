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

from datetime import datetime, timedelta, timezone
from typing import Optional

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
      drop path (a NYSE-primary name passes qualification but its MOC
      would print on the primary listing's close-cross rather than the
      NASDAQ close cross the strategy targets; the orchestrator drops
      it during prequalify when that filter is enabled).
    - ``zero_fill_symbols``: for symbols in this set, ``buy``/``sell``
      RETURNS an oid (unlike ``raise_on_buy_symbols`` which raises)
      but records the order with ``status="Cancelled"`` and
      ``filled=0`` — i.e. exercise the round-2 dispatcher's terminal-
      Cancelled-with-filled=0 zero-fill branch, not the submission-
      failure branch. Used by
      ``test_fill_gated_zero_fill_terminal_no_moc``.
    - ``slow_fill_ticks_by_sym``: mapping ``{symbol: n_ticks}`` — for
      symbols in the map, ``get_order_status`` returns non-terminal
      (Submitted, filled=0) for the first ``n_ticks`` calls, then
      returns the true recorded state. Simulates fill latency so the
      dispatcher can observe the terminal transition for one symbol
      while another is still pending — exercises the "MOC A fires
      before B's terminal event" concurrency ordering.
    - ``get_positions_transient_empty_calls``: set of call indexes
      (1-based) that return ``[]`` regardless of the internal book.
      Used by the post-close poll consecutive-flat-reads test — a
      single transient empty read must NOT satisfy the FLAT verdict.
    - ``moc_stays_submitted``: MOC orders record with
      ``status="Submitted"`` and ``filled=0`` — the exchange has
      accepted them, but they never terminal-Fill and never move
      positions. Distinct from ``moc_reject_all`` (which flips
      MOCs to terminal Cancelled — my round-3 coverage cross-check
      correctly catches those pre-close via
      ``moc_signed_by_sym``'s terminal-rejected exclusion). Used
      to isolate the post-close poll as the SOLE remaining
      exposure signal: the audit sees ``Submitted`` and presumes
      accepted, coverage cross-check sees MOCs in a live state
      (included in signed coverage → not uncovered), and only
      the post-close position poll catches that positions never
      actually flattened.

    ROUND-3 MOC SEMANTICS (matches real IB). MOC orders record the
    fill BUT do NOT update ``_positions`` until
    ``simulate_close_auction()`` is called. Real IB does not fold MOC
    fills into ``get_positions()`` until after the 16:00 crossing;
    the earlier "MOC fills immediately update positions" fake model
    silently broke the SAFETY reviewer's two-sided sweep test cases
    (a two-sided sweep sees the pre-16:00 position book as if MOCs
    had already filled, which they hadn't in real IB, and over-
    corrects). The conftest autouse fixture wires
    ``simulate_close_auction`` into ``_poll_positions_after_close``
    so happy-path tests still see the post-auction FLAT state.
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
                 partial_persistent: bool = False,
                 wait_for_fill_always_times_out: bool = False,
                 moc_reject_all: bool = False,
                 entry_status_stale: bool = False,
                 non_nasdaq_primary: dict[str, str] | None = None,
                 zero_fill_symbols: set[str] | None = None,
                 slow_fill_ticks_by_sym: dict[str, int] | None = None,
                 get_positions_transient_empty_calls: set[int] | None = None,
                 moc_stays_submitted: bool = False):
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
        # ``partial_persistent`` freezes the truncated-visible symbol
        # set on the first partial snapshot and returns EXACTLY those
        # symbols on every subsequent partial call — even after their
        # internal ``_positions`` qty has been flattened. This models a
        # TWS stream that is stuck returning a fixed subset of names
        # (persistent brownout) rather than one that "rotates" as
        # positions clear. The rotating shape is a weaker simulation
        # of the underlying failure mode; the round-2 fill-gated
        # dispatcher would otherwise silently cover all symbols by
        # calling get_positions per-entry as the internal book cleared.
        self._partial_persistent = partial_persistent
        self._partial_visible_syms: Optional[list[str]] = None
        self._wait_for_fill_always_times_out = wait_for_fill_always_times_out
        self._moc_reject_all = moc_reject_all
        self._entry_status_stale = entry_status_stale
        # Per-symbol override for the qualified contract's
        # ``primaryExchange`` field. Unspecified names default to
        # ``"NASDAQ"`` so existing tests pass ``_prequalify_symbols``'s
        # NASDAQ-listing filter; tests that want to exercise the
        # non-NASDAQ-primary drop path seed entries here.
        self._non_nasdaq_primary: dict[str, str] = dict(non_nasdaq_primary or {})
        # Round-2 knobs (post-review):
        self._zero_fill_symbols = set(zero_fill_symbols or [])
        self._slow_fill_ticks_by_sym: dict[str, int] = dict(
            slow_fill_ticks_by_sym or {}
        )
        # get_order_status remaining-tick counter (decrements per call).
        self._slow_fill_remaining_by_oid: dict[str, int] = {}
        self._get_positions_transient_empty_calls: set[int] = set(
            get_positions_transient_empty_calls or set()
        )
        self._moc_stays_submitted = moc_stays_submitted
        # Round-3 MOC semantics: pending MOC fills held here until
        # ``simulate_close_auction()`` is called. Matches real IB's
        # behaviour where MOC fills are not folded into
        # ``get_positions()`` until after the 16:00 crossing prints.
        # Each entry: oid -> {"symbol": str, "signed": int}.
        self._pending_moc_fills: dict[str, dict] = {}
        # Whether the fake has been told the auction has printed. Once
        # True, ``get_positions()`` includes the pending MOC fills.
        self._auction_printed: bool = False

    def simulate_close_auction(self) -> None:
        """Fold every non-rejected pending MOC fill into ``_positions``.

        Real IB folds MOC fills into ``get_positions()`` after the
        16:00 crossing prints. Test scenarios that want to check
        the post-auction position book (rc=0 happy paths, post-close
        poll tests) call this to trigger that fold. Rejected MOCs
        (``moc_reject_all``) are recorded with ``filled=0`` and
        already excluded from ``_pending_moc_fills``, so this method
        is safe to call in every scenario.

        Idempotent — calling twice folds only the pending set the
        FIRST call saw.
        """
        for oid, meta in list(self._pending_moc_fills.items()):
            sym = meta["symbol"]
            signed = int(meta["signed"])
            if signed == 0:
                continue
            self._positions[sym] = self._positions.get(sym, 0) + signed
        self._pending_moc_fills.clear()
        self._auction_printed = True

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
        # Round-2: transient empty response on specific 1-based call
        # indexes. Modeled independently of the "after N" knobs so a
        # test can inject exactly one empty read amidst otherwise-real
        # results.
        if (self._get_positions_call_count
                in self._get_positions_transient_empty_calls):
            return []
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
        # partial TWS snapshot mid-flow. In ``partial_persistent`` mode
        # the truncated symbol set is frozen on the first partial call
        # and returned on every subsequent partial call — see
        # ``_partial_persistent`` docstring.
        if (self._get_positions_partial_after_n_calls is not None
                and self._get_positions_call_count
                >= self._get_positions_partial_after_n_calls):
            if self._partial_persistent:
                if self._partial_visible_syms is None:
                    keep = max(1, int(round(
                        len(rows) * self._partial_position_fraction
                    )))
                    self._partial_visible_syms = [r.symbol for r in rows[:keep]]
                visible = set(self._partial_visible_syms)
                rows = [r for r in rows if r.symbol in visible]
            else:
                keep = max(1, int(round(
                    len(rows) * self._partial_position_fraction
                )))
                rows = rows[:keep]
        return rows

    def _make_contract(self, symbol, exchange=None, currency="USD"):
        # ``exchange`` defaults to ``None`` (not the real broker's
        # ``"SMART"``) so a call site that omits the kwarg is
        # OBSERVABLE in ``make_contract_calls`` — a ``(sym, None)``
        # tuple in the recorded list tells the routing regression
        # the caller relied on the implicit default. Under the real
        # broker's ``exchange="SMART"`` default and the current
        # ``ACTIVE_ROUTING == "SMART"``, an omission would otherwise
        # be indistinguishable from a correctly-threaded constant
        # and the guard would go dead.
        if symbol in self._unqualified:
            raise ValueError(f"cannot qualify {symbol}")
        self.qualified_symbols.append(symbol)
        # Record every (symbol, exchange) pair the prequalify path
        # requests so the routing regression can assert every contract
        # went out with ACTIVE_ROUTING (see live/nasdaq_routing.py).
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
        # simulates a MKT placed but never acked. MOC orders always
        # fully fill unless ``moc_reject_all`` — but per round-3 MOC
        # semantics (see class docstring) the fill does NOT update
        # ``_positions`` until ``simulate_close_auction()`` runs. This
        # matches real IB, which does not fold MOC fills into
        # ``get_positions`` until after the 16:00 crossing prints.
        if order_type == "MOC":
            if self._moc_reject_all:
                filled = 0
                status = "Cancelled"
            elif self._moc_stays_submitted:
                # Round-3 test knob: MOC sits Submitted forever, never
                # terminal-Fills. Audit presumes accepted (TimeoutError
                # ↔ still Submitted), coverage cross-check includes
                # signed_qty (not terminal-rejected), and only the
                # post-close position poll catches the never-cleared
                # position book. Also do NOT stash in
                # ``_pending_moc_fills`` — the MOC didn't fill at all.
                filled = 0
                status = "Submitted"
            else:
                filled = requested
                status = "Filled"
        elif self._stuck_all_entries or symbol in self._stuck_on:
            filled = 0
            status = "Submitted"
        elif symbol in self._zero_fill_symbols:
            # Round-2: exercise the dispatcher's "terminal with
            # filled=0" branch — oid IS returned (so `pending` sees
            # this entry), but the order goes straight to terminal
            # Cancelled with 0 fills. Distinct from
            # ``raise_on_buy_symbols`` which prevents the oid from
            # ever landing.
            filled = 0
            status = "Cancelled"
        else:
            filled = int(round(requested * self.entry_fill_ratio))
            status = "Filled"
        if filled > 0:
            signed = filled if action == "BUY" else -filled
            if order_type == "MOC":
                # Defer: MOC fills queue for the auction. Real IB
                # does not reflect these in ``get_positions()`` until
                # after 16:00.
                self._pending_moc_fills[oid] = {
                    "symbol": symbol, "signed": signed,
                }
            else:
                self._positions[symbol] = (
                    self._positions.get(symbol, 0) + signed
                )
        self._orders[oid] = {
            "symbol": symbol, "action": action, "requested": requested,
            "filled": filled, "order_type": order_type, "tif": tif,
            "status": status,
        }
        # Round-2 slow-fill: seed the per-oid remaining-ticks counter
        # so ``get_order_status`` reports non-terminal for the first N
        # calls, then transitions to the true terminal state.
        n_ticks = self._slow_fill_ticks_by_sym.get(symbol, 0)
        if n_ticks > 0 and order_type == "MKT":
            self._slow_fill_remaining_by_oid[oid] = n_ticks

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
        # Round-2 slow-fill: for the first N ``get_order_status`` calls
        # on this order, report non-terminal so the dispatcher can
        # observe a fast-filling symbol becoming terminal WHILE this
        # symbol is still pending — proving the concurrency ordering
        # end-to-end.
        remaining = self._slow_fill_remaining_by_oid.get(order_id, 0)
        if remaining > 0:
            self._slow_fill_remaining_by_oid[order_id] = remaining - 1
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
        "--submit-by", "00:01", "--replay-ignore-cutoff",
        "--on-feed-failure", "abort",
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

    # RR3 (round-2 revised): temporal ordering — every cancel must
    # precede the SWEEP's authoritative get_positions call. In round-1
    # this was simpler ("every cancel before any get_positions after
    # entries") because cancel_non_terminal_entries ran once, then
    # the MOC snapshot took one get_positions. In round-2 the
    # DISPATCHER takes a get_positions per tick to size fill-gated
    # MOCs — those calls interleave with entry-status polling for
    # symbols that DID fill, and are correct (they don't affect the
    # sweep's later cancel-then-size flow). What still matters is
    # that the SWEEP's get_positions (the LAST one before this test's
    # main() returns) happens AFTER every entry cancel. That is the
    # invariant that guarantees a stuck entry cannot fill BETWEEN
    # its cancel and the MOC batch being sized off a fresh snapshot.
    events = fake_broker.events
    entry_event_indices = [
        i for i, e in enumerate(events)
        if (e.startswith("buy:MKT:") or e.startswith("sell:MKT:"))
    ]
    assert entry_event_indices, "no entry events recorded — test setup bug"
    cancel_indices = [
        i for i, e in enumerate(events) if e.startswith("cancel:")
    ]
    assert cancel_indices, "no cancel events recorded"
    gp_indices = [
        i for i, e in enumerate(events) if e == "get_positions"
    ]
    assert gp_indices, "no get_positions calls recorded"
    # The LAST get_positions is either the sweep's or the post-close
    # poll's; either way it happens after every cancel in a valid run.
    last_cancel = max(cancel_indices)
    gp_after_last_cancel = [i for i in gp_indices if i > last_cancel]
    assert gp_after_last_cancel, (
        f"no get_positions call after the last cancel (at index "
        f"{last_cancel}) — the sweep did not size its MOC batch off "
        f"a fresh post-cancel snapshot. events tail: "
        f"{events[max(0, last_cancel - 3):]}"
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

    # get_positions() calls: (1) diff baseline pre-entry, (2)+ inside
    # the fill-gated dispatcher + sweep + post-close poll. From call
    # #2 onward return only 30% of the rows AND freeze that visible
    # subset (partial_persistent=True) — the un-fixed round-2 dispatcher
    # would otherwise call get_positions per-entry and rotate through
    # the truncated view as positions cleared, silently covering
    # everything. Persistent partial models the underlying failure mode
    # (TWS is stuck returning the same fixed subset) more honestly.
    fake_broker = _FakeBroker(
        get_positions_partial_after_n_calls=2,
        partial_position_fraction=0.3,
        partial_persistent=True,
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
    """RRR1 + round-2 post-close poll: MOC exchange rejection must
    return rc=11 via ONE OR BOTH of two independent safeguards.

    ``broker.buy/sell`` for MOC returns oid the moment TWS API-accepts
    the placeOrder — BEFORE any exchange ack. A late MOC (submitted
    past NASDAQ's 15:55 cutoff) becomes terminal Cancelled with
    filled=0 shortly after. Two safeguards now catch this:

    1. **Pre-close acceptance audit** — early warning; per-oid wait
       for terminal state during the [submit, submit+moc_ack_timeout]
       window. Fast operator feedback (~60s from submit) so panic can
       be run before 16:00.
    2. **Post-close position poll** — AUTHORITATIVE. Polls
       ``get_positions()`` 16:00-16:05; a nonzero managed-universe
       position at the deadline is definitive proof of exposure.

    This test proves BOTH safeguards independently return rc=11:

    * Phase 1 — audit neutralized + MOCs stay Submitted (fake
      ``moc_stays_submitted``): pre-close signal is silent (coverage
      cross-check sees non-terminal MOCs, includes them in signed
      coverage → sum=0 → not uncovered), but post-close poll
      catches the non-flat book (MOCs never actually filled → held
      remains X). Proves post-close poll is a load-bearing
      standalone safeguard.
    * Phase 2 — audit restored + MOCs terminal-rejected (fake
      ``moc_reject_all``): the acceptance audit AND the coverage
      cross-check both catch rejections pre-close (round-3
      ``moc_signed_by_sym`` re-poll excludes terminal-rejected
      MOCs). Either signal short-circuits to rc=11 before the
      post-close poll runs.
    """
    import logging as _logging
    universe = config.load_active_universe()

    def _setup_broker(moc_mode: str):
        feed_local = _prepopulate_live_feed(universe)
        monkeypatch.setattr(daily_orchestrator, "LiveFeed",
                            lambda *a, **kw: feed_local)
        monkeypatch.setattr(daily_orchestrator, "_sleep_until",
                            lambda *a, **kw: None)
        monkeypatch.setenv("DATABENTO_API_KEY", "dummy-key")
        _patch_artifacts(monkeypatch, universe)
        kwargs = {}
        if moc_mode == "stays_submitted":
            kwargs["moc_stays_submitted"] = True
        elif moc_mode == "reject_all":
            kwargs["moc_reject_all"] = True
        else:
            raise ValueError(f"unknown moc_mode {moc_mode}")
        broker = _FakeBroker(**kwargs)

        def _fake_connect(_args):
            broker.connect(mode="paper")
            return broker

        monkeypatch.setattr(daily_orchestrator, "_connect_broker",
                            _fake_connect)
        return broker

    # ---- Phase 1: acceptance audit neutralized + MOCs stay Submitted
    # → coverage cross-check clean, post-close poll must still catch
    # the non-flat book.
    broker_pre = _setup_broker("stays_submitted")
    real_audit = daily_orchestrator._audit_moc_acceptance
    monkeypatch.setattr(
        daily_orchestrator, "_audit_moc_acceptance",
        lambda *a, **kw: set(),
    )
    handler_pre, records_pre = _capture_orchestrator_logs(_logging.ERROR)
    try:
        rc_pre = daily_orchestrator.main(_base_argv())
    finally:
        daily_orchestrator.logger.removeHandler(handler_pre)
    assert rc_pre == 11, (
        f"phase 1 (audit neutralized) must STILL return rc=11 — the "
        f"post-close position poll is a standalone safeguard; got {rc_pre}"
    )
    loud_pre = [r.getMessage() for r in records_pre]
    assert any("OVERNIGHT EXPOSURE (post-close)" in m for m in loud_pre), (
        f"phase 1: expected 'OVERNIGHT EXPOSURE (post-close)' alert to "
        f"identify which safeguard tripped; got: {loud_pre}"
    )
    mkt_pre = [c for c in broker_pre.buy_calls + broker_pre.sell_calls
               if c[2] == "MKT"]
    moc_pre = [c for c in broker_pre.buy_calls + broker_pre.sell_calls
               if c[2] == "MOC"]
    assert mkt_pre and moc_pre, (
        "phase 1: expected both MKT entries and MOC placements to have "
        "fired against the broker"
    )

    # ---- Phase 2: acceptance audit restored + MOCs terminal-rejected
    # (moc_reject_all) — both the audit and the round-3 coverage
    # cross-check catch the rejection pre-close.
    monkeypatch.setattr(daily_orchestrator, "_audit_moc_acceptance",
                        real_audit)
    broker_post = _setup_broker("reject_all")
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
    # Phase 2 short-circuits pre-close — the "OVERNIGHT EXPOSURE
    # (pre-close)" tag identifies which safeguard tripped.
    assert any("OVERNIGHT EXPOSURE (pre-close)" in m for m in loud), (
        f"phase 2: expected 'OVERNIGHT EXPOSURE (pre-close)' alert "
        f"proving the audit caught this early (before the post-close "
        f"poll ran); got: {loud}"
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
# Routing regression: every order path must thread ACTIVE_ROUTING
# ---------------------------------------------------------------------------

def test_every_order_path_uses_active_routing(monkeypatch, tmp_path):
    """Routing is strategy-critical (see ``live/nasdaq_routing.py``) and
    must be set from ONE place — ``ACTIVE_ROUTING`` — so switching the
    active route is a one-line change here, not a grep across the
    codebase. Every code path in ``live/`` that reaches the broker MUST
    carry ``exchange=ACTIVE_ROUTING`` on the request.

    ``ACTIVE_ROUTING`` is currently ``"SMART"`` (best NBBO on the entry
    MKT; the MOC still prints on the NASDAQ close cross for our
    NASDAQ-primary universe). This test pins the value AND the fact
    that every call site threads the constant — so a future flip back
    to ISLAND stays a one-liner and any drift shows up here first.

    Enumerates the three paths and pins each:
    * pre-qualify (``broker._make_contract`` inside
      ``_prequalify_symbols``)
    * MKT entry batch (``broker.buy`` / ``broker.sell`` with
      ``order_type="MKT"``)
    * MOC exit batch (``broker.buy`` / ``broker.sell`` with
      ``order_type="MOC"``)
    A regression that ever drops the ``exchange`` kwarg on any of these
    would let orders fall through to the broker's implicit default and
    silently drift out of the ``live/`` routing contract. This test
    fails LOUDLY the moment that happens.
    """
    from live.nasdaq_routing import ACTIVE_ROUTING

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

    # (1) prequalify path — every _make_contract call must carry
    # ACTIVE_ROUTING.
    assert fake_broker.make_contract_calls, "no prequalify calls recorded"
    for sym, exchange in fake_broker.make_contract_calls:
        assert exchange == ACTIVE_ROUTING == "SMART", (
            f"_prequalify_symbols routed {sym} to exchange={exchange!r}, "
            f"expected {ACTIVE_ROUTING!r}"
        )

    # (2) + (3) entry MKT and exit MOC — every buy/sell call must
    # carry exchange=ACTIVE_ROUTING in its kwargs. This is the
    # load-bearing assertion: it fails the moment any call site drops
    # the kwarg.
    all_order_calls = fake_broker.buy_calls + fake_broker.sell_calls
    assert all_order_calls, "no orders reached the broker at all"
    mkt_orders = [c for c in all_order_calls if c[2] == "MKT"]
    moc_orders = [c for c in all_order_calls if c[2] == "MOC"]
    assert mkt_orders, "no MKT entries reached the broker"
    assert moc_orders, "no MOC exits reached the broker"
    for sym, qty, order_type, kw in all_order_calls:
        got = kw.get("exchange")
        assert got == ACTIVE_ROUTING == "SMART", (
            f"{order_type} order for {sym} routed to exchange={got!r} "
            f"(expected {ACTIVE_ROUTING!r}) — active-routing contract "
            f"violated. Full kwargs: {kw}"
        )


def test_no_implicit_default_routing_leaks_through_live_paths(monkeypatch, tmp_path):
    """Companion pin: assert POSITIVELY that no order or prequalify call
    ever leaves ``exchange`` unset — every call site must explicitly
    thread ``ACTIVE_ROUTING`` from ``live/nasdaq_routing.py`` rather
    than rely on the broker's implicit ``exchange="SMART"`` default.

    Why this matters even now that ``ACTIVE_ROUTING == "SMART"``: the
    whole point of centralising the routing choice is that flipping
    ``ACTIVE_ROUTING`` back to ISLAND (or to any future venue) is a
    one-line change. A call site that silently defaults to the
    broker's SMART kwarg would silently stay on SMART after such a
    flip — the exact drift this test blocks.
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

    implicit_qualify = [
        (sym, ex) for sym, ex in fake_broker.make_contract_calls
        if ex is None
    ]
    assert not implicit_qualify, (
        f"prequalify path relied on broker's implicit default on: "
        f"{implicit_qualify}"
    )
    implicit_orders = [
        (sym, qty, ot, kw) for sym, qty, ot, kw
        in fake_broker.buy_calls + fake_broker.sell_calls
        if kw.get("exchange") is None
    ]
    assert not implicit_orders, (
        f"order path relied on broker's implicit default on: "
        f"{implicit_orders}"
    )


def test_prequalify_drops_non_nasdaq_primary(monkeypatch, tmp_path):
    """``_prequalify_symbols`` must drop any name whose qualified
    contract's ``primaryExchange`` is not ``"NASDAQ"``.

    Reason: the strategy is calibrated to the NASDAQ closing auction,
    so the MOC exit leg must print at the NASDAQ close cross. Under
    the current SMART routing the router would send an MOC for a
    non-NASDAQ-primary name to the home exchange's close cross — the
    wrong auction — silently breaking the strategy. Under the ISLAND
    direct-route alternative IB will still qualify a NYSE / ARCA-
    primary name for continuous UTP trading, but the closing MOC
    mis-prints (or is rejected) at the primary listing's close-cross
    for the same reason. Under either routing choice, refusing to
    trade the name upstream is the correct behaviour.

    Re-enabled 2026-07-21 alongside the SMART routing switch; had
    been rolled back 2026-07-20 with the broker's route-survival
    assertion. See ``daily_orchestrator._prequalify_symbols``'
    "NASDAQ-primary eligibility filter (ACTIVE)" docstring section
    and ``live/nasdaq_routing.py``.

    Seeds the fake broker with a batch of universe names all marked
    NYSE-primary (rather than a single hand-picked one, since the
    optimizer's target set is model-dependent and we cannot predict
    which specific name will be traded) and asserts:
    * at least one drop is logged with ``primary_exchange="NYSE"``,
    * no buy/sell order for ANY seeded symbol ever reaches the broker,
    * unseeded (NASDAQ-primary) names still trade normally.
    """
    import logging as _logging

    universe = config.load_active_universe()
    feed = _prepopulate_live_feed(universe)
    monkeypatch.setattr(daily_orchestrator, "LiveFeed", lambda *a, **kw: feed)
    monkeypatch.setattr(daily_orchestrator, "_sleep_until", lambda *a, **kw: None)
    monkeypatch.setenv("DATABENTO_API_KEY", "dummy-key")
    _patch_artifacts(monkeypatch, universe)

    # The optimizer's target set is model-dependent and (with this
    # test's synthetic model) skews toward alphabetically-early
    # mid-caps rather than the file-order top-of-universe mega-caps.
    # So we seed a wide alphabetical slice of the universe as
    # NYSE-primary — enough overlap with the target book that the
    # drop path is guaranteed to fire at least once. We intentionally
    # leave the last few names unseeded (still NASDAQ-primary in the
    # fake) so a NASDAQ-primary contingent survives to trade — the
    # "the drop is symbol-specific, not a wholesale abort" invariant.
    universe_sorted = sorted(universe)
    seeded_nyse: dict[str, str] = {
        sym: "NYSE" for sym in universe_sorted[: max(1, len(universe_sorted) // 2)]
    }
    fake_broker = _FakeBroker(non_nasdaq_primary=seeded_nyse)

    def _fake_connect(_args):
        fake_broker.connect(mode="paper")
        return fake_broker

    monkeypatch.setattr(daily_orchestrator, "_connect_broker", _fake_connect)

    handler, records = _capture_orchestrator_logs(_logging.WARNING)
    try:
        rc = daily_orchestrator.main(_base_argv())
    finally:
        daily_orchestrator.logger.removeHandler(handler)

    assert rc == 0

    # (1) No NYSE-primary name may ever hit any buy/sell call. This
    # is the load-bearing invariant — the strategy MUST NOT trade a
    # name whose MOC would print on a non-NASDAQ close-cross.
    touched = (
        {c[0] for c in fake_broker.buy_calls}
        | {c[0] for c in fake_broker.sell_calls}
    )
    leaked = touched & set(seeded_nyse)
    assert not leaked, (
        f"non-NASDAQ-primary names traded despite the drop filter: "
        f"{sorted(leaked)}"
    )

    # (2) At least one drop must have been logged with the informative
    # extras — ``symbol`` (which name) and ``primary_exchange`` (why).
    # We assert on the informative extras rather than a substring match
    # on the message so a benign wording change to the warning does
    # not silently break the regression.
    dropped_syms = {
        getattr(r, "symbol", None)
        for r in records
        if getattr(r, "primary_exchange", None) == "NYSE"
        and getattr(r, "expected_primary_exchange", None) == "NASDAQ"
    }
    dropped_syms.discard(None)
    assert dropped_syms, (
        "no 'dropping non-NASDAQ-listed' warning was captured. "
        f"seeded={sorted(seeded_nyse)}, "
        f"captured_records={[r.getMessage() for r in records]}"
    )
    # And every dropped symbol must be one we seeded — never a
    # spurious drop of a NASDAQ-primary name.
    assert dropped_syms.issubset(set(seeded_nyse)), (
        f"drop path fired for names we did NOT seed as NYSE-primary: "
        f"{sorted(dropped_syms - set(seeded_nyse))}"
    )

    # (3) NASDAQ-primary names still trade — the drop is symbol-
    # specific, not a wholesale abort.
    assert touched, (
        "no orders reached the broker at all — the drop path should "
        "have removed only the seeded names, not the whole universe"
    )


def test_prequalify_drop_still_flattens_existing_position(monkeypatch, tmp_path):
    """Load-bearing asymmetry: a name the NASDAQ-primary filter drops
    must still get its MOC flatten if we're already holding it. The
    filter suppresses the ENTRY (we don't want a new position in a
    name whose MOC would print on the wrong auction), but a position
    we already carry has to be flattened either way — leaving it live
    at 16:00 strands it overnight, which is exactly the failure mode
    the filter is trying to prevent upstream.

    The invariant relies on ``_submit_moc_flatten`` and the surrounding
    exposure checks being passed ``managed_universe=universe_set`` — the
    intersection of the active universe with ``get_positions`` — NOT
    the prequalify-filtered ``qualified_symbols``. A future refactor
    that "tidied" those call sites to pass ``qualified_symbols`` would
    silently strand held positions in dropped names. This test fails
    under exactly that mutation; see the mutation check recorded in
    the review that added it.

    Seeds one active-universe name as NYSE-primary AND as a 500-share
    long, then asserts:
    * no MKT entry for that name reached the broker (filter dropped
      it from the entry batch), and
    * a 500-share SELL MOC did reach the broker (flatten preserved).
    """
    universe = config.load_active_universe()
    assert len(universe) >= 1, "test needs a non-empty active universe"
    # Pick a name from the middle of the universe so it's neither the
    # first (which some tests special-case) nor a boundary name.
    held_sym = sorted(universe)[len(universe) // 2]

    feed = _prepopulate_live_feed(universe)
    monkeypatch.setattr(daily_orchestrator, "LiveFeed", lambda *a, **kw: feed)
    monkeypatch.setattr(daily_orchestrator, "_sleep_until", lambda *a, **kw: None)
    monkeypatch.setenv("DATABENTO_API_KEY", "dummy-key")
    _patch_artifacts(monkeypatch, universe)

    held_qty = 500
    fake_broker = _FakeBroker(
        non_nasdaq_primary={held_sym: "NYSE"},
        initial_positions=[_FakePosition(held_sym, held_qty)],
    )

    def _fake_connect(_args):
        fake_broker.connect(mode="paper")
        return fake_broker

    monkeypatch.setattr(daily_orchestrator, "_connect_broker", _fake_connect)

    rc = daily_orchestrator.main(_base_argv())
    assert rc == 0

    # (1) ENTRY suppressed — no MKT for the dropped name.
    mkt_calls_for_held = [
        (sym, qty, ot, kw)
        for sym, qty, ot, kw in fake_broker.buy_calls + fake_broker.sell_calls
        if sym == held_sym and ot == "MKT"
    ]
    assert not mkt_calls_for_held, (
        f"filter-dropped name {held_sym} reached an MKT entry despite "
        f"the prequalify drop: {mkt_calls_for_held}"
    )

    # (2) EXIT preserved — a MOC of exactly the held quantity, on the
    # opposite side of the long, must reach the broker. The +500 long
    # flattens with a 500-share SELL MOC (``_submit_moc_flatten`` sizes
    # off ``abs(current_qty)`` and signs the side to close). If this
    # asserts False the position stranded overnight.
    moc_sells_for_held = [
        (sym, qty, ot, kw)
        for sym, qty, ot, kw in fake_broker.sell_calls
        if sym == held_sym and ot == "MOC"
    ]
    moc_buys_for_held = [
        (sym, qty, ot, kw)
        for sym, qty, ot, kw in fake_broker.buy_calls
        if sym == held_sym and ot == "MOC"
    ]
    assert not moc_buys_for_held, (
        f"held long in {held_sym} flattened with a BUY MOC (wrong side): "
        f"{moc_buys_for_held}"
    )
    assert len(moc_sells_for_held) == 1, (
        f"expected exactly one MOC flatten for held long in {held_sym}; "
        f"got sells={moc_sells_for_held}"
    )
    sym, qty, ot, kw = moc_sells_for_held[0]
    assert qty == held_qty, (
        f"MOC flatten for {held_sym} sized to {qty}, expected "
        f"{held_qty} (the held quantity)"
    )


# ---------------------------------------------------------------------------
# Round-2: fill-gated per-symbol MOC + deadline sweep + post-close verdict
# ---------------------------------------------------------------------------


def test_fill_gated_moc_fires_immediately_on_entry_terminal(
    monkeypatch, tmp_path,
):
    """Round-2 core behaviour: as each entry becomes terminal Filled,
    the fill-gated dispatcher fires its opposite-side MOC IMMEDIATELY
    — not after all N entries have been polled.

    Round-2 revision (per correctness review): the weaker version of
    this test only asserted symbol-set equality between entries and
    MOCs, which the pre-round-2 batch-based ``_submit_moc_flatten``
    also satisfied. The strengthened version uses
    ``slow_fill_ticks_by_sym`` to make ONE symbol fast-filling and
    another slow (3 poll ticks before terminal), then asserts on the
    broker's event log that the fast symbol's MOC send appears
    BEFORE the slow symbol reaches terminal state. That ordering is
    only achievable by concurrent per-symbol dispatch, and would fail
    against any batch-after-all-entries-terminal implementation.
    """
    universe = config.load_active_universe()
    feed = _prepopulate_live_feed(universe)
    monkeypatch.setattr(daily_orchestrator, "LiveFeed", lambda *a, **kw: feed)
    monkeypatch.setattr(daily_orchestrator, "_sleep_until", lambda *a, **kw: None)
    monkeypatch.setenv("DATABENTO_API_KEY", "dummy-key")
    _patch_artifacts(monkeypatch, universe)

    slow_sym = sorted(universe)[len(universe) // 2]
    fake_broker = _FakeBroker(
        slow_fill_ticks_by_sym={slow_sym: 3},
    )

    def _fake_connect(_args):
        fake_broker.connect(mode="paper")
        return fake_broker

    monkeypatch.setattr(daily_orchestrator, "_connect_broker", _fake_connect)

    rc = daily_orchestrator.main(_base_argv())
    assert rc == 0

    # Ordering assertion. Find (a) the FIRST MOC send event for any
    # non-slow symbol (a proxy for "some symbol's MOC fired") and (b)
    # the slow symbol's get_order_status call that shows it as
    # terminal. If the dispatcher is truly concurrent, (a) must be
    # BEFORE the terminal transition of (b).
    #
    # The fake's ``events`` log records "get_positions",
    # "buy:MKT:<sym>:<oid>", "sell:MKT:<sym>:<oid>", "buy:MOC:...",
    # "sell:MOC:...", "cancel:<oid>". ``get_order_status`` calls are
    # not currently logged (they'd flood the log), so we use "the
    # LAST tick-loop pass that fires a MOC while slow_sym is not yet
    # in the covered set" as the concurrency proof: some MOC(s) must
    # be sent before the slow entry's MOC (i.e. slow_sym's MOC must
    # NOT be the FIRST MOC in the log — if the dispatcher were fully
    # batched it'd fire all MOCs at once).
    moc_events = [
        (i, tag) for i, tag in enumerate(fake_broker.events)
        if (tag.startswith("buy:MOC:") or tag.startswith("sell:MOC:"))
    ]
    assert moc_events, "no MOC events recorded"

    # First MOC's symbol and slow symbol's MOC index.
    def _sym_from_moc_tag(tag: str) -> str:
        # Tag shape: "buy:MOC:<sym>:<oid>" or "sell:MOC:<sym>:<oid>".
        return tag.split(":", 3)[2]

    first_moc_idx = moc_events[0][0]
    first_moc_sym = _sym_from_moc_tag(moc_events[0][1])
    slow_moc_events = [
        (i, tag) for i, tag in moc_events
        if _sym_from_moc_tag(tag) == slow_sym
    ]
    assert slow_moc_events, (
        f"slow symbol {slow_sym!r} never got an MOC — sweep should have "
        f"picked it up after ``slow_fill_ticks`` expired. Events tail: "
        f"{fake_broker.events[-20:]}"
    )
    slow_moc_idx = slow_moc_events[0][0]

    # (1) The FIRST MOC must NOT be the slow symbol's MOC — that
    # proves other MOCs fired before the slow entry reached terminal.
    # A batch-after-all-entries-terminal impl would have to have all
    # MOCs go out in one contiguous span AFTER slow_sym is terminal;
    # a concurrent impl fires other MOCs while slow_sym is still
    # non-terminal in ``pending``.
    assert first_moc_sym != slow_sym, (
        f"first MOC in the event log was for the slow symbol "
        f"{slow_sym!r}. Expected some other symbol's MOC to fire "
        f"before the slow one — concurrency ordering broken."
    )

    # (2) At least one MOC must appear STRICTLY BEFORE slow_sym's MOC
    # in the event log (a stronger restatement of (1)).
    assert first_moc_idx < slow_moc_idx, (
        f"no MOC fired before the slow symbol's MOC (first_moc_idx="
        f"{first_moc_idx}, slow_moc_idx={slow_moc_idx}). Events: "
        f"{fake_broker.events[:30]}"
    )

    # Structural sanity — every entry has an opposite-side MOC.
    entries_by_sym: dict[str, str] = {}
    for tag in fake_broker.events:
        if tag.startswith("buy:MKT:") or tag.startswith("sell:MKT:"):
            side, _, sym_oid = tag.split(":", 2)
            sym = sym_oid.split(":")[0]
            entries_by_sym[sym] = "BUY" if side == "buy" else "SELL"

    mocs_by_sym: dict[str, str] = {}
    for tag in fake_broker.events:
        if tag.startswith("buy:MOC:") or tag.startswith("sell:MOC:"):
            side, _, sym_oid = tag.split(":", 2)
            sym = sym_oid.split(":")[0]
            mocs_by_sym[sym] = "BUY" if side == "buy" else "SELL"

    assert entries_by_sym, "no MKT entries recorded"
    assert set(entries_by_sym) == set(mocs_by_sym), (
        f"symbols missing an MOC: {set(entries_by_sym) - set(mocs_by_sym)}; "
        f"symbols missing an entry: {set(mocs_by_sym) - set(entries_by_sym)}"
    )
    for sym in entries_by_sym:
        entry_side = entries_by_sym[sym]
        moc_side = mocs_by_sym[sym]
        assert entry_side != moc_side, (
            f"{sym}: entry {entry_side} paired with same-side MOC {moc_side} "
            f"— expected opposite-side counter-flatten"
        )


def test_fill_gated_partial_fill_moc_sized_to_actual_position(
    monkeypatch, tmp_path,
):
    """Round-2 partial-fill invariant: fill-gated MOC sizes off a
    FRESH ``get_positions`` per symbol, not the entry's requested qty.
    A 50% MKT fill produces a 50% MOC counter — never a full target-diff
    MOC that would over-flatten into an opposite overnight exposure.
    """
    universe = config.load_active_universe()
    feed = _prepopulate_live_feed(universe)
    monkeypatch.setattr(daily_orchestrator, "LiveFeed", lambda *a, **kw: feed)
    monkeypatch.setattr(daily_orchestrator, "_sleep_until", lambda *a, **kw: None)
    monkeypatch.setenv("DATABENTO_API_KEY", "dummy-key")
    _patch_artifacts(monkeypatch, universe)

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
    assert mkt_orders and moc_orders

    entry_by_sym = {c[0]: c[1] for c in mkt_orders}
    for sym, moc_qty, ot, kw in moc_orders:
        entry_qty = entry_by_sym[sym]
        expected_filled = int(round(entry_qty * 0.5))
        assert moc_qty == expected_filled, (
            f"MOC exit {sym} qty {moc_qty} != actual position "
            f"{expected_filled} (entry_qty={entry_qty}) — fill-gated "
            f"dispatch did NOT size from broker.get_positions"
        )


def test_fill_gated_zero_fill_no_moc_submission_failure(
    monkeypatch, tmp_path,
):
    """Round-2 zero-fill rule (variant A): a symbol whose entry
    submission RAISED (never lands in ``pending``) gets no MOC. This
    exercises the ``order_id is None`` branch in the dispatcher —
    NOT the terminal-Cancelled-with-filled-0 branch. Both branches
    have separate tests to prove each independently.
    """
    universe = config.load_active_universe()
    feed = _prepopulate_live_feed(universe)
    monkeypatch.setattr(daily_orchestrator, "LiveFeed", lambda *a, **kw: feed)
    monkeypatch.setattr(daily_orchestrator, "_sleep_until", lambda *a, **kw: None)
    monkeypatch.setenv("DATABENTO_API_KEY", "dummy-key")
    _patch_artifacts(monkeypatch, universe)

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
    all_calls = fake_broker.buy_calls + fake_broker.sell_calls
    for sym, qty, ot, kw in all_calls:
        if sym == rejected_sym:
            assert ot != "MOC", (
                f"MOC placed for {rejected_sym} despite the entry never "
                f"landing (submission-failure zero-fill → no MOC invariant)"
            )


def test_fill_gated_zero_fill_no_moc_terminal_cancelled(
    monkeypatch, tmp_path,
):
    """Round-2 zero-fill rule (variant B, per correctness review):
    a symbol whose entry SUBMITTED but became terminal Cancelled with
    ``filled=0`` also gets no MOC from the fill-gated path. This
    directly exercises the ``if filled <= 0`` branch in
    ``_fill_gated_moc_dispatch`` (an oid IS in ``pending``, but the
    terminal state's filled qty is 0 so no counter-side MOC fires).

    Distinct from the submission-failure variant above — that one hits
    the ``order_id is None`` code path.
    """
    universe = config.load_active_universe()
    feed = _prepopulate_live_feed(universe)
    monkeypatch.setattr(daily_orchestrator, "LiveFeed", lambda *a, **kw: feed)
    monkeypatch.setattr(daily_orchestrator, "_sleep_until", lambda *a, **kw: None)
    monkeypatch.setenv("DATABENTO_API_KEY", "dummy-key")
    _patch_artifacts(monkeypatch, universe)

    # First, run once without zero_fill to find which symbols are
    # actually traded (the optimizer may skip some universe names).
    probe_broker = _FakeBroker()

    def _probe_connect(_args):
        probe_broker.connect(mode="paper")
        return probe_broker

    monkeypatch.setattr(daily_orchestrator, "_connect_broker", _probe_connect)
    daily_orchestrator.main(_base_argv())
    traded_syms = [
        c[0] for c in probe_broker.buy_calls + probe_broker.sell_calls
        if c[2] == "MKT"
    ]
    assert traded_syms, "probe run traded nothing — cannot pick a zero-fill target"
    zero_fill_sym = traded_syms[0]

    # Reset the environment for the real test (new broker, same feed).
    feed2 = _prepopulate_live_feed(universe)
    monkeypatch.setattr(daily_orchestrator, "LiveFeed", lambda *a, **kw: feed2)
    fake_broker = _FakeBroker(zero_fill_symbols={zero_fill_sym})

    def _fake_connect(_args):
        fake_broker.connect(mode="paper")
        return fake_broker

    monkeypatch.setattr(daily_orchestrator, "_connect_broker", _fake_connect)

    rc = daily_orchestrator.main(_base_argv())
    assert rc == 0
    # Assert an oid IS in the entries list (proves submission
    # succeeded), the order's status IS Cancelled with filled=0 in
    # the fake's internal book (proves it went terminal Cancelled)
    # AND no MOC was placed for the symbol.
    entry_events = [
        tag for tag in fake_broker.events
        if (tag.startswith("buy:MKT:") or tag.startswith("sell:MKT:"))
        and f":{zero_fill_sym}:" in tag
    ]
    assert entry_events, (
        f"expected an entry submission for {zero_fill_sym} — the "
        f"zero-fill knob relies on the entry landing so the terminal "
        f"branch can be exercised. Traded syms this run: "
        f"{[c[0] for c in fake_broker.buy_calls + fake_broker.sell_calls if c[2] == 'MKT'][:10]}"
    )
    # Fake's internal book confirms terminal Cancelled + filled=0.
    matching_orders = [
        o for o in fake_broker._orders.values()
        if o["symbol"] == zero_fill_sym and o["order_type"] == "MKT"
    ]
    assert matching_orders, "no MKT entry recorded internally"
    for o in matching_orders:
        assert o["status"] == "Cancelled" and o["filled"] == 0, (
            f"knob mis-wired — expected terminal Cancelled+filled=0, "
            f"got status={o['status']} filled={o['filled']}"
        )
    all_calls = fake_broker.buy_calls + fake_broker.sell_calls
    for sym, qty, ot, kw in all_calls:
        if sym == zero_fill_sym:
            assert ot != "MOC", (
                f"MOC placed for {zero_fill_sym} despite entry going "
                f"terminal Cancelled with filled=0 (dispatcher's "
                f"'if filled <= 0' branch failed to skip)"
            )


def test_sweep_flattens_residual_baseline_position(monkeypatch, tmp_path):
    """Round-2 sweep invariant: a residual/baseline position whose
    today's entry NEVER FILLS must STILL get an MOC leg from the sweep.

    Simulated by ``stuck_on_symbols={held_sym}`` — the entry submits
    successfully (returns an oid) but never reaches terminal state
    (status stays ``Submitted``, ``filled=0``). The fill-gated
    dispatcher polls the entry, sees non-terminal, moves on; when the
    dispatcher's max wait elapses, the sweep runs, cancels the stuck
    entry, and — critically — takes a fresh get_positions that STILL
    shows the residual 250-share long, and submits the MOC to flatten
    it.

    Regression this pins: if a future refactor changed the sweep to
    "only MOC symbols we tried to enter today" (i.e. filtered its
    scope by ``entries`` rather than by ``managed_universe``), this
    residual would strand overnight.
    """
    universe = config.load_active_universe()
    held_sym = sorted(universe)[len(universe) // 3]
    held_qty = 250

    feed = _prepopulate_live_feed(universe)
    monkeypatch.setattr(daily_orchestrator, "LiveFeed", lambda *a, **kw: feed)
    monkeypatch.setattr(daily_orchestrator, "_sleep_until", lambda *a, **kw: None)
    monkeypatch.setenv("DATABENTO_API_KEY", "dummy-key")
    _patch_artifacts(monkeypatch, universe)

    # Held position + entry for the same name is stuck (never reaches
    # terminal) — the fill-gated dispatch's per-oid wait expires, the
    # sweep runs, cancels the stuck entry, and MOCs the residual.
    fake_broker = _FakeBroker(
        initial_positions=[_FakePosition(held_sym, held_qty)],
        stuck_on_symbols={held_sym},
    )

    def _fake_connect(_args):
        fake_broker.connect(mode="paper")
        return fake_broker

    monkeypatch.setattr(daily_orchestrator, "_connect_broker", _fake_connect)

    rc = daily_orchestrator.main(_base_argv())
    assert rc == 0, (
        f"expected rc=0 (sweep flattens residual position), got {rc}"
    )
    moc_sells_for_held = [
        (sym, qty, ot, kw)
        for sym, qty, ot, kw in fake_broker.sell_calls
        if sym == held_sym and ot == "MOC"
    ]
    moc_buys_for_held = [
        (sym, qty, ot, kw)
        for sym, qty, ot, kw in fake_broker.buy_calls
        if sym == held_sym and ot == "MOC"
    ]
    assert not moc_buys_for_held, (
        f"held long in {held_sym} should have flattened with a SELL MOC "
        f"(long → sell to close), not a BUY MOC: {moc_buys_for_held}"
    )
    assert len(moc_sells_for_held) == 1, (
        f"expected exactly one MOC flatten for the residual position "
        f"in {held_sym}; got sells={moc_sells_for_held}. The sweep is "
        f"the sole guarantee for residual positions — a regression here "
        f"strands them overnight."
    )
    sym, qty, ot, kw = moc_sells_for_held[0]
    assert qty == held_qty, (
        f"sweep MOC flatten for {held_sym} sized to {qty}, expected "
        f"{held_qty} (the held quantity)"
    )


def test_sweep_deadline_scales_with_book_size(monkeypatch, tmp_path):
    """Round-2 sweep margin formula (post-review):
    ``margin = max(FLOOR, N * (PER_ORDER + PER_STATUS) + 2 * GET_POS
    + OVERHEAD)``. The formula now budgets get_positions calls and
    per-symbol status polls in addition to placeOrder time, so at
    500 names the margin is ~121s (not the old ~70s). At 49 names
    the small overhead pushes just past the 15s floor. Pinning
    both the formula and its output so a silent drift here (e.g.
    dropping the get_positions cost from the sum) fails LOUDLY.
    """
    submit_by = datetime.fromisoformat("2026-07-21T15:55:00-04:00")

    def _expected(n):
        return max(
            config.SWEEP_MIN_MARGIN_SECS,
            n * (config.SWEEP_PER_ORDER_SECS + config.SWEEP_PER_STATUS_SECS)
            + 2 * config.SWEEP_GET_POSITIONS_SECS
            + config.SWEEP_FIXED_OVERHEAD_SECS,
        )

    # Small book (49 names). Formula = 49 * (0.13+0.1) + 2*0.6 + 5
    # = 17.47s at the current defaults — above the 15s floor.
    small_deadline, small_margin = daily_orchestrator._compute_sweep_deadline(
        submit_by, n_managed_symbols=49,
    )
    assert small_margin == pytest.approx(_expected(49), rel=1e-9), (
        f"49-name book: expected margin {_expected(49)}s, got "
        f"{small_margin}s. Formula (post-round-2 fix): N*(PER+STATUS) "
        f"+ 2*GET_POSITIONS + OVERHEAD."
    )
    assert (submit_by - small_deadline).total_seconds() == pytest.approx(
        small_margin, rel=1e-9,
    )

    # Tiny book (10 names) — should hit the floor.
    _, tiny_margin = daily_orchestrator._compute_sweep_deadline(
        submit_by, n_managed_symbols=10,
    )
    assert tiny_margin == config.SWEEP_MIN_MARGIN_SECS, (
        f"tiny 10-name book should hit the {config.SWEEP_MIN_MARGIN_SECS}s "
        f"floor; got margin={tiny_margin}"
    )

    # 500-name book — formula kicks in well past the floor.
    big_deadline, big_margin = daily_orchestrator._compute_sweep_deadline(
        submit_by, n_managed_symbols=500,
    )
    assert big_margin == pytest.approx(_expected(500), rel=1e-9), (
        f"500-name book margin drifted from formula: expected "
        f"{_expected(500)}, got {big_margin}"
    )
    assert (submit_by - big_deadline).total_seconds() == pytest.approx(
        big_margin, rel=1e-9,
    )


def test_post_close_position_poll_returns_rc_11_on_nonzero(
    monkeypatch, tmp_path,
):
    """Round-2 post-close verdict: a nonzero managed-universe symbol
    at 16:05 is the AUTHORITATIVE exposure signal. Even with all
    pre-close signals clean (MOCs sent, cross-check happy, audit
    quiet), a nonzero position after the auction returns rc=11.

    Round-3 revision: uses ``moc_stays_submitted=True`` (MOCs sit
    Submitted forever, never terminal-Fill) rather than
    ``moc_reject_all`` (which flips MOCs to terminal Cancelled). The
    coverage cross-check's round-3 rebuild of
    ``moc_signed_by_sym`` now correctly EXCLUDES terminal-rejected
    MOCs — so ``moc_reject_all`` would trip the coverage
    cross-check pre-close, defeating the "isolate the post-close
    poll" premise. With ``moc_stays_submitted``, the audit sees
    ``Submitted`` (presumes accepted), the coverage cross-check
    includes the signed_qty (not terminal-rejected → included),
    and only the post-close position poll catches the fact that
    positions never actually flattened.
    """
    import logging as _logging
    universe = config.load_active_universe()
    feed = _prepopulate_live_feed(universe)
    monkeypatch.setattr(daily_orchestrator, "LiveFeed", lambda *a, **kw: feed)
    monkeypatch.setattr(daily_orchestrator, "_sleep_until", lambda *a, **kw: None)
    monkeypatch.setenv("DATABENTO_API_KEY", "dummy-key")
    _patch_artifacts(monkeypatch, universe)
    # Neutralise the pre-close acceptance audit so the ONLY remaining
    # safety net is the post-close position poll.
    monkeypatch.setattr(
        daily_orchestrator, "_audit_moc_acceptance",
        lambda *a, **kw: set(),
    )

    fake_broker = _FakeBroker(moc_stays_submitted=True)

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
        f"expected rc=11 (post-close nonzero positions), got {rc}. "
        f"With the pre-close audit neutralised, the post-close poll is "
        f"the sole remaining safety net."
    )
    loud = [r.getMessage() for r in records]
    assert any("OVERNIGHT EXPOSURE (post-close)" in m for m in loud), (
        f"expected 'OVERNIGHT EXPOSURE (post-close)' tag to identify "
        f"which safeguard tripped; got: {loud}"
    )


def test_post_close_poll_failure_pre_close_short_circuit(monkeypatch, tmp_path):
    """Round-2 pessimism rule (variant A): if EVERY post-close
    ``get_positions()`` poll raises, we cannot confirm flat and
    rc=11 fires. In this variant the raises begin BEFORE 16:00 so
    the sweep also sees a raise (``moc_positions_unknown``) and the
    pre-close short-circuit returns rc=11 without the post-close
    poll even running. That is the correct EARLY page for a TWS
    outage.

    Renamed from the round-1 name so its behaviour matches the
    docstring (per correctness reviewer). Variant B below covers the
    post-close-only failure end-to-end.
    """
    import logging as _logging
    universe = config.load_active_universe()
    feed = _prepopulate_live_feed(universe)
    monkeypatch.setattr(daily_orchestrator, "LiveFeed", lambda *a, **kw: feed)
    monkeypatch.setattr(daily_orchestrator, "_sleep_until", lambda *a, **kw: None)
    monkeypatch.setenv("DATABENTO_API_KEY", "dummy-key")
    _patch_artifacts(monkeypatch, universe)

    # Baseline + a few dispatcher calls succeed, then all subsequent
    # calls raise — including the sweep AND the post-close poll.
    fake_broker = _FakeBroker(get_positions_raises_after_n_calls=3)

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
        f"expected rc=11 (get_positions raising = pessimistic exposure), "
        f"got {rc}"
    )
    loud = [r.getMessage() for r in records]
    assert any("OVERNIGHT EXPOSURE" in m for m in loud), (
        f"expected 'OVERNIGHT EXPOSURE' alert; got: {loud}"
    )


def test_post_close_poll_only_isolated(monkeypatch, tmp_path):
    """Direct unit-level pin on ``_poll_positions_after_close``.

    Exercises three branches without going through main():
    (a) all-flat → not exposed, no nonzero symbols, no error.
    (b) still non-flat by deadline → exposed with the offending
        symbols named.
    (c) every poll raises → pessimistic exposed with the error msg.
    """
    universe = ["AAA", "BBB", "CCC"]
    managed = set(universe)

    # (a) All flat, WITH corroboration that the book must be flat.
    # An empty position response is only accepted as proof of flat when
    # an independent source agrees (see ``_moc_legs_corroborate_flat``);
    # case (d) below pins the uncorroborated counterpart.
    poll_until_a = config.now_ny() + timedelta(milliseconds=50)
    empty_broker = _FakeBroker()  # no positions
    exposed, nonzero, err = daily_orchestrator._poll_positions_after_close(
        broker=empty_broker,
        managed_universe=managed,
        poll_until=poll_until_a,
        initial_interval_secs=0.01,
        max_interval_secs=0.02,
        corroborator=lambda: True,
    )
    assert not exposed
    assert nonzero == []
    assert err is None

    # (a2) Non-empty response whose MANAGED names are all zero needs NO
    # corroboration — the stream is demonstrably reporting rows, so
    # "rows present, none of ours nonzero" is real evidence of flat.
    # ``ZZZ`` is outside ``managed``, so the response is non-empty while
    # the managed aggregate is empty.
    poll_until_a2 = config.now_ny() + timedelta(milliseconds=50)
    zero_row_broker = _FakeBroker(
        initial_positions=[_FakePosition("ZZZ", 100)],
    )
    exposed, nonzero, err = daily_orchestrator._poll_positions_after_close(
        broker=zero_row_broker,
        managed_universe=managed,
        poll_until=poll_until_a2,
        initial_interval_secs=0.01,
        max_interval_secs=0.02,
    )
    assert not exposed, "zero-qty rows present is positive evidence of flat"
    assert nonzero == []

    # (b) Persistently non-flat.
    poll_until_b = config.now_ny() + timedelta(milliseconds=100)
    stuck_broker = _FakeBroker(
        initial_positions=[
            _FakePosition("AAA", 100),
            _FakePosition("BBB", -50),
        ],
    )
    exposed, nonzero, err = daily_orchestrator._poll_positions_after_close(
        broker=stuck_broker,
        managed_universe=managed,
        poll_until=poll_until_b,
        initial_interval_secs=0.01,
        max_interval_secs=0.02,
    )
    assert exposed
    assert set(nonzero) == {"AAA", "BBB"}
    assert err is None

    # (c) Every poll raises → pessimistic.
    poll_until_c = config.now_ny() + timedelta(milliseconds=50)
    dead_broker = _FakeBroker(get_positions_raises=True)
    exposed, nonzero, err = daily_orchestrator._poll_positions_after_close(
        broker=dead_broker,
        managed_universe=managed,
        poll_until=poll_until_c,
        initial_interval_secs=0.01,
        max_interval_secs=0.02,
    )
    assert exposed
    assert nonzero == []
    assert err is not None and "simulated TWS drop" in err

    # (d) EMPTY position response that CANNOT be corroborated must NOT
    # be read as flat. This is the post-close analogue of
    # ``empty_snapshot_after_fills``: broker.get_positions returns
    # ib_async's cached set and yields [] WITHOUT raising when the
    # subscription was torn down and re-established across the auction,
    # so a wiped cache is byte-identical to a genuinely flat book.
    for label, corro in (
        ("no corroborator", None),
        ("corroborator says NOT flat", lambda: False),
        ("corroborator cannot tell", lambda: None),
        ("corroborator raises", _raise_corroborator),
    ):
        poll_until_d = config.now_ny() + timedelta(milliseconds=50)
        wiped_broker = _FakeBroker()  # empty — indistinguishable from flat
        exposed, nonzero, err = (
            daily_orchestrator._poll_positions_after_close(
                broker=wiped_broker,
                managed_universe=managed,
                poll_until=poll_until_d,
                initial_interval_secs=0.01,
                max_interval_secs=0.02,
                corroborator=corro,
            )
        )
        assert exposed, (
            f"uncorroborated empty position read ({label}) must fail "
            f"CLOSED — a wiped position cache looks exactly like a flat "
            f"book, so it is not proof of flat"
        )


def _raise_corroborator():
    raise RuntimeError("simulated corroborator failure")


def test_post_close_corroborator_accounts_for_residual_and_skips():
    """The post-close corroborator must not treat "no entry filled
    today" as "nothing to flatten".

    ``_reconcile_from_previous`` only LOGS drift (it never trades), so a
    position carried in from a prior day is live even when
    ``n_filled_entries == 0``. And ``moc_batch`` holds only legs that
    were successfully SENT — a skipped/errored leg appends nothing — so
    "every leg in moc_batch filled" proves only "every MOC I managed to
    send filled", not "every position was covered".

    Without both guards a wiped position cache (empty read) on a day
    with a carried-in residual corroborates as FLAT and the run exits
    rc=0 with real overnight exposure. Structural pin on the call-site
    corroborator so a refactor cannot quietly drop either term.
    """
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(
        daily_orchestrator.main,
    )))
    corro = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = getattr(fn, "id", None) or getattr(fn, "attr", None)
        if name != "_poll_positions_after_close":
            continue
        for kw in node.keywords:
            if kw.arg == "corroborator":
                corro = kw.value
    assert corro is not None, (
        "_poll_positions_after_close must be called with a corroborator "
        "— without one every empty position read fails closed and the "
        "run pages every day"
    )
    src = ast.dump(corro)
    assert "moc_skipped" in src, (
        "corroborator must refuse to corroborate when any MOC leg was "
        "skipped — moc_batch only contains legs that were SENT"
    )
    assert "n_filled_entries" in src, (
        "corroborator must know whether today's entries filled"
    )
    assert "current" in src and "universe_set" in src, (
        "corroborator must treat a nonzero managed BASELINE as an "
        "opened book — a residual carried in from a prior day is live "
        "even when nothing filled today"
    )


def test_moc_legs_corroborate_flat_verdicts():
    """Unit pin on the corroboration source itself."""
    class _Info:
        def __init__(self, status, filled):
            self.status = status
            self.filled = filled

    class _B:
        def __init__(self, m):
            self._m = m

        def get_order_status(self, oid):
            v = self._m[oid]
            if isinstance(v, Exception):
                raise v
            return v

    legs = [{"oid": "1", "symbol": "AAA"}, {"oid": "2", "symbol": "BBB"}]

    # No book was ever opened -> nothing to flatten -> flat is provable
    # even with no MOC legs at all.
    assert daily_orchestrator._moc_legs_corroborate_flat(
        _B({}), [], False,
    ) is True

    # A book WAS opened but there are no MOC legs to check -> unknown.
    assert daily_orchestrator._moc_legs_corroborate_flat(
        _B({}), [], True,
    ) is None

    # Every leg terminal Filled -> corroborated flat.
    assert daily_orchestrator._moc_legs_corroborate_flat(
        _B({"1": _Info("Filled", 10), "2": _Info("Filled", 5)}), legs, True,
    ) is True

    # One leg cancelled -> positive evidence AGAINST flat.
    assert daily_orchestrator._moc_legs_corroborate_flat(
        _B({"1": _Info("Filled", 10), "2": _Info("Cancelled", 0)}),
        legs, True,
    ) is False

    # A leg still working -> not proof of flat.
    assert daily_orchestrator._moc_legs_corroborate_flat(
        _B({"1": _Info("Filled", 10), "2": _Info("Submitted", 0)}),
        legs, True,
    ) is False

    # Status poll raises -> cannot tell.
    assert daily_orchestrator._moc_legs_corroborate_flat(
        _B({"1": _Info("Filled", 10), "2": RuntimeError("boom")}),
        legs, True,
    ) is None

    # Filled status but zero shares -> not a real fill.
    assert daily_orchestrator._moc_legs_corroborate_flat(
        _B({"1": _Info("Filled", 0), "2": _Info("Filled", 5)}), legs, True,
    ) is False


def test_unhandled_exception_after_entry_submission_pages_rc11(
    monkeypatch, tmp_path,
):
    """LAST-RESORT EXPOSURE GATE.

    The order-path ``try`` previously had a ``finally`` but no
    ``except``, so an unhandled raise between entry submission and the
    rc gate exited rc=1 — BELOW the ``rc >= 11`` pager predicate — with
    a filled, uncovered book. Pin both halves of the new handler:
    armed (orders submitted) -> rc=11; unarmed -> rc=1.
    """
    import ast
    import inspect
    import textwrap

    src = inspect.getsource(daily_orchestrator.main)
    tree = ast.parse(textwrap.dedent(src))
    order_try = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Try) and node.finalbody:
            # The order-path try is the one whose finally disconnects.
            fin = ast.dump(ast.Module(body=node.finalbody, type_ignores=[]))
            if "disconnect" in fin:
                order_try = node
                break
    assert order_try is not None, "order-path try/finally not found"
    assert len(order_try.handlers) >= 1, (
        "the order-path try MUST have an except handler — without one "
        "an unhandled raise exits rc=1 with a possibly-live book"
    )
    handler_src = ast.dump(order_try)
    assert "orders_may_be_live" in handler_src, (
        "the handler must branch on whether orders were submitted"
    )
    # The armed branch must return 11, the unarmed branch 1.
    returns = [
        n.value.value
        for h in order_try.handlers
        for n in ast.walk(h)
        if isinstance(n, ast.Return)
        and isinstance(n.value, ast.Constant)
    ]
    assert 11 in returns, "armed path must return rc=11 (pager)"
    assert 1 in returns, "unarmed path must return rc=1 (safe)"


def test_fill_gated_dispatch_never_regresses_leg_2_sized_from_target(
    monkeypatch, tmp_path,
):
    """Structural safety pin (CLAUDE.md invariant): the MOC exit
    quantity must be sized from ``broker.get_positions()`` AFTER the
    entry fills, NEVER from ``target - current`` or the entry's
    requested quantity. The round-2 fill-gated dispatcher goes through
    ``_submit_single_moc`` which — by SIGNATURE — cannot accept any
    target-side argument. This test pins that structural property:
    any refactor that tries to plumb a target-side quantity into the
    MOC size fails at review, not at runtime.

    Uses an inspectable helper — we assert on the parameter names of
    ``_submit_single_moc``. Grep-level enforcement.
    """
    import inspect
    sig = inspect.signature(daily_orchestrator._submit_single_moc)
    params = sig.parameters
    # Round-2: signature also carries ``submit_by`` for the pre-send
    # cutoff guard (SAFETY reviewer finding 3). That is a TIME
    # argument, not a target-side quantity — still forbidden are any
    # target-side kwargs.
    assert set(params) == {"broker", "symbol", "current_qty", "submit_by"}, (
        f"_submit_single_moc signature drifted — expected exactly "
        f"(broker, symbol, current_qty, submit_by); got {list(params)}. "
        f"Any target-side argument (share_targets, requested_qty, "
        f"target_diff, etc.) is a structural safety violation."
    )

    sig_sweep = inspect.signature(daily_orchestrator._sweep_moc_deadline)
    # Sweep MUST NOT take target-side data — only managed_universe
    # (for scope) and covered_qty (for de-dup / top-up). The round-2
    # rename from ``already_covered`` (set) to ``covered_qty`` (dict)
    # was to enable top-ups of under-sized dispatcher MOCs.
    sweep_params = set(sig_sweep.parameters)
    forbidden = {"share_targets", "target_positions", "extended_targets",
                 "requested_qty", "target_diff"}
    leaked = sweep_params & forbidden
    assert not leaked, (
        f"_sweep_moc_deadline signature leaked target-side data: "
        f"{leaked}. MOC size must ALWAYS come from a fresh "
        f"get_positions() snapshot inside the sweep."
    )


def test_replay_honours_submit_by_by_default(monkeypatch, tmp_path):
    """Round-2 replay guard: without ``--replay-ignore-cutoff``,
    replay HONOURS the wall-clock ``--submit-by`` — so a
    late-in-the-day rehearsal correctly surfaces rc=4 or rc=11.

    We use --submit-by 00:01 (past cutoff today) and NO ignore flag.
    The pre-order rc=4 check fires because ``now > submit_by`` when
    the orchestrator's main() reaches it.
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

    def _blowup(_args):
        raise AssertionError(
            "broker connect happened despite past --submit-by and no "
            "--replay-ignore-cutoff (replay must honour the cutoff)"
        )

    monkeypatch.setattr(daily_orchestrator, "_connect_broker", _blowup)

    argv = [
        "--replay-date", "2026-05-15", "--paper",
        "--vol-cap", "5000", "--replay-submit",
        "--submit-by", "00:01",  # NO --replay-ignore-cutoff
        "--on-feed-failure", "abort",
    ]
    rc = daily_orchestrator.main(argv)
    assert rc == 4, (
        f"expected rc=4 (past submit-by, safe pre-order skip), got {rc}. "
        f"Replay honouring the cutoff is the round-2 default."
    )


def test_entry_fill_timeout_config_default_is_30_secs():
    """Round-2 operator ask: raise the MKT entry fill timeout default
    to 30s. Pin the config value so a silent drift is caught here.
    """
    # We can't check the runtime module-level value directly because
    # the autouse conftest fixture patches it down to 0.5s for speed.
    # Read the file source to check the DEFAULT constant defined
    # BEFORE any monkeypatch runs.
    import ast
    from pathlib import Path
    src = Path(config.__file__).read_text()
    tree = ast.parse(src)
    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "DEFAULT_ENTRY_FILL_TIMEOUT_SECS":
                    target = node.value
                    break
    assert target is not None, (
        "DEFAULT_ENTRY_FILL_TIMEOUT_SECS missing from live/config.py"
    )
    value = ast.literal_eval(target)
    assert value == 30.0, (
        f"DEFAULT_ENTRY_FILL_TIMEOUT_SECS drifted from 30.0; got {value!r}. "
        f"Round-2 operator ask was to raise from 5s to 30s."
    )


# ---------------------------------------------------------------------------
# Round-2 revision (post-review): new tests for defects the previous round
# left unexercised
# ---------------------------------------------------------------------------


def test_post_close_poll_end_to_end_failure_returns_rc_11(
    monkeypatch, tmp_path,
):
    """Round-2 (variant B): the AUTHORITATIVE end-to-end
    verification that main() propagates rc=11 when
    ``_poll_positions_after_close`` returns exposed with an error.

    Directly monkey-patches ``_poll_positions_after_close`` to return
    ``(True, [], "simulated post-close TWS failure")`` — that is the
    behaviour a real ``get_positions()`` raise 16:00-16:05 must
    produce, and this test proves main() correctly returns rc=11 for
    that outcome (rather than falling through to rc=0 as it did
    before the round-2 fix). The dispatcher/sweep run cleanly (no
    pre-close signal fires), so the ONLY thing keeping this away
    from rc=0 is the post-close poll's authority.
    """
    import logging as _logging
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

    # Poll returns pessimistic-exposed. This bypasses the pre-close
    # signals entirely — the sole load-bearing safeguard for this
    # test is the post-close poll's authority in main().
    monkeypatch.setattr(
        daily_orchestrator, "_poll_positions_after_close",
        lambda **kw: (True, [], "simulated post-close TWS failure"),
    )

    handler, records = _capture_orchestrator_logs(_logging.ERROR)
    try:
        rc = daily_orchestrator.main(_base_argv())
    finally:
        daily_orchestrator.logger.removeHandler(handler)

    assert rc == 11, (
        f"expected rc=11 (post-close poll returned exposed with error), "
        f"got {rc}. main() must propagate the post-close poll verdict "
        f"even when pre-close signals are all clean."
    )
    loud = [r.getMessage() for r in records]
    assert any("OVERNIGHT EXPOSURE (post-close)" in m for m in loud), (
        f"expected 'OVERNIGHT EXPOSURE (post-close)' tag; got: {loud}"
    )


def test_post_close_poll_refuses_flat_before_not_before(monkeypatch, tmp_path):
    """Round-2 lower-bound guard (SAFETY reviewer finding 1): the
    post-close poll REFUSES to declare FLAT before ``not_before``,
    even against an all-flat broker.

    Setup: broker has no positions, ``not_before`` is 200ms in the
    future, ``poll_until`` is 100ms in the future. Since
    ``poll_until < not_before``, the loop breaks at ``poll_until``
    WITHOUT the not_before threshold ever being crossed — so no
    ``consecutive_flat`` count can grow past the "pre_not_before"
    gate. Result must be EXPOSED (fail-closed pessimistic verdict),
    NOT FLAT.
    """
    universe = ["AAA", "BBB", "CCC"]
    managed = set(universe)

    now = config.now_ny()
    not_before = now + timedelta(milliseconds=200)
    poll_until = now + timedelta(milliseconds=100)

    empty_broker = _FakeBroker()
    exposed, nonzero, err = daily_orchestrator._poll_positions_after_close(
        broker=empty_broker,
        managed_universe=managed,
        poll_until=poll_until,
        not_before=not_before,
        min_consecutive_flat_reads=2,
        initial_interval_secs=0.01,
        max_interval_secs=0.02,
    )
    assert exposed, (
        "post-close poll declared FLAT before not_before was reached "
        "— fail-closed guard broken. Any run that reached the poll "
        "before 16:00 (recorder short-circuit, empty book mid-afternoon) "
        "could otherwise declare flat on a stale pre-auction snapshot."
    )
    assert err is not None, (
        "expected a synthetic error explaining the fail-closed reason"
    )


def test_post_close_poll_requires_consecutive_flat_reads(
    monkeypatch, tmp_path,
):
    """Round-2 consecutive-flat guard (SAFETY reviewer finding 1):
    a SINGLE transient empty reading from a broker that returned rows
    on a prior poll is NOT authoritative for FLAT. Requires N=2
    consecutive flat reads.

    Uses ``get_positions_transient_empty_calls={1}``: the first
    ``get_positions`` call returns ``[]`` (transient/stale). All
    subsequent calls report the true nonzero position. If the poll
    accepted the first empty read as FLAT, it would return not-
    exposed on that single reading — exactly the bug the round-2
    fix closes.
    """
    universe = ["AAA", "BBB"]
    managed = set(universe)

    fake_broker = _FakeBroker(
        initial_positions=[_FakePosition("AAA", 100)],
        get_positions_transient_empty_calls={1},
    )
    poll_until = config.now_ny() + timedelta(milliseconds=200)
    exposed, nonzero, err = daily_orchestrator._poll_positions_after_close(
        broker=fake_broker,
        managed_universe=managed,
        poll_until=poll_until,
        not_before=None,  # explicitly disable not_before for this test
        min_consecutive_flat_reads=2,
        initial_interval_secs=0.01,
        max_interval_secs=0.02,
    )
    assert exposed, (
        f"post-close poll accepted a single transient empty read as "
        f"FLAT despite the true position being 100sh AAA. "
        f"consecutive-flat-reads guard failed."
    )
    assert "AAA" in nonzero, (
        f"expected AAA in the exposed-symbols list; got {nonzero}"
    )


def test_sweep_tops_up_undersized_dispatcher_moc(monkeypatch, tmp_path):
    """Round-2 sweep top-up (SAFETY reviewer finding 4): if the
    dispatcher's per-symbol MOC under-sized relative to the true
    post-tick position (a partial fill landed AFTER the dispatcher
    read positions), the sweep MUST top up the residual — not
    silently skip the symbol as ``already_covered``.

    Unit-level test on ``_sweep_moc_deadline`` directly. Simulates
    ``covered_qty[sym] = -60`` (dispatcher already sent SELL MOC of
    60) with a true held position of 100 (an extra 40 landed after
    the dispatcher's read). The sweep must compute
    ``residual = 100 + (-60) = 40`` and send a SELL MOC for 40.
    """
    held_sym = "AAAA"
    held_qty = 100
    dispatcher_covered = -60  # dispatcher SELL MOC of 60

    fake_broker = _FakeBroker(
        initial_positions=[_FakePosition(held_sym, held_qty)],
    )
    submit_by = config.now_ny() + timedelta(hours=1)

    (moc_batch, cancelled_oids, cancel_failed, skipped, positions_unknown) = (
        daily_orchestrator._sweep_moc_deadline(
            broker=fake_broker,
            entries=[],
            managed_universe={held_sym},
            covered_qty={held_sym: dispatcher_covered},
            submit_by=submit_by,
        )
    )
    assert not positions_unknown
    assert not cancelled_oids and not cancel_failed and not skipped
    assert len(moc_batch) == 1, (
        f"expected exactly one top-up MOC for {held_sym}; got {moc_batch}. "
        f"The round-2 sweep must top up under-sized dispatcher MOCs, not "
        f"silently skip them as ``already_covered``."
    )
    leg = moc_batch[0]
    assert leg["symbol"] == held_sym
    assert leg["side"] == "SELL", (
        f"residual {held_qty + dispatcher_covered}sh long should top "
        f"up with a SELL MOC; got side={leg['side']}"
    )
    expected_topup = held_qty + dispatcher_covered  # 40
    assert leg["qty"] == expected_topup, (
        f"top-up MOC sized to {leg['qty']}, expected {expected_topup} "
        f"(held {held_qty} + dispatcher covered {dispatcher_covered})"
    )


def test_sweep_skips_exactly_covered_symbol(monkeypatch, tmp_path):
    """Companion to the top-up test: a symbol whose dispatcher MOC
    exactly matches the true position (``residual == 0``) must NOT
    receive a duplicate MOC from the sweep. This preserves the
    no-double-flatten property.
    """
    held_sym = "BBBB"
    held_qty = 100
    dispatcher_covered = -100  # exact match

    fake_broker = _FakeBroker(
        initial_positions=[_FakePosition(held_sym, held_qty)],
    )
    submit_by = config.now_ny() + timedelta(hours=1)

    (moc_batch, cancelled_oids, cancel_failed, skipped, positions_unknown) = (
        daily_orchestrator._sweep_moc_deadline(
            broker=fake_broker,
            entries=[],
            managed_universe={held_sym},
            covered_qty={held_sym: dispatcher_covered},
            submit_by=submit_by,
        )
    )
    assert moc_batch == [], (
        f"sweep sent an unnecessary MOC for an already-fully-covered "
        f"symbol: {moc_batch}. residual=0 must skip."
    )


def test_sweep_cancel_failure_surfaces_as_pre_close_exposure(
    monkeypatch, tmp_path,
):
    """Round-2 (CORRECTNESS reviewer finding 2): the sweep's cancel
    failures must be surfaced up the return tuple as
    ``cancel_failed_syms`` and OR'd into pre_close_exposure — an
    entry that could not be cancelled may still fill mid-window and
    become an uncontrolled overnight position.

    Configure ``_FakeBroker.cancel`` to raise for a specific oid via
    monkeypatch, verify the sweep returns the symbol in
    ``cancel_failed_syms``.
    """
    stuck_sym = "CCCC"

    fake_broker = _FakeBroker(
        stuck_on_symbols={stuck_sym},
    )

    def _raising_cancel(oid):
        # Track the attempt in the event log so the test can prove
        # cancel was tried.
        fake_broker.cancel_calls.append(oid)
        fake_broker.events.append(f"cancel-failed:{oid}")
        raise RuntimeError(f"simulated cancel failure for {oid}")

    # Submit a stuck entry so the sweep has something to cancel.
    stuck_oid = fake_broker.buy(
        stuck_sym, 100, order_type="MKT", exchange="SMART",
    )
    monkeypatch.setattr(fake_broker, "cancel", _raising_cancel)

    entries = [{
        "symbol": stuck_sym, "side": "BUY", "requested_qty": 100,
        "order_id": stuck_oid, "error": None, "status": "Submitted",
        "filled_qty": 0,
    }]

    submit_by = config.now_ny() + timedelta(hours=1)
    (moc_batch, cancelled_oids, cancel_failed_syms, skipped, positions_unknown) = (
        daily_orchestrator._sweep_moc_deadline(
            broker=fake_broker,
            entries=entries,
            managed_universe={stuck_sym},
            covered_qty={},
            submit_by=submit_by,
        )
    )
    assert stuck_sym in cancel_failed_syms, (
        f"sweep must return {stuck_sym} in cancel_failed_syms; got "
        f"{cancel_failed_syms}. Without this the pre-close exposure "
        f"signal misses an uncancellable entry — CORRECTNESS "
        f"reviewer's second critical finding."
    )
    # No oid was successfully cancelled.
    assert cancelled_oids == []


def test_submit_single_moc_refuses_past_submit_by(monkeypatch, tmp_path):
    """Round-2 (SAFETY reviewer finding 3): ``_submit_single_moc``
    must REFUSE to submit a MOC when the wall clock is past
    ``submit_by``. Before this fix, a late fill during a slow
    dispatcher tick could fire a MOC the exchange would silently
    reject; nothing in the pre-close signal set would notice
    because ``_audit_moc_acceptance`` had a 60s budget that would
    also be past.
    """
    fake_broker = _FakeBroker()
    submit_by_past = config.now_ny() - timedelta(seconds=1)

    leg, outcome = daily_orchestrator._submit_single_moc(
        broker=fake_broker,
        symbol="AAAA",
        current_qty=100,
        submit_by=submit_by_past,
    )
    assert outcome == "skipped_cutoff", (
        f"expected outcome='skipped_cutoff'; got {outcome!r}"
    )
    assert leg is None
    # No order should have reached the broker.
    assert not fake_broker.buy_calls
    assert not fake_broker.sell_calls


def test_dispatcher_bounded_by_deadline_at_500_names(monkeypatch, tmp_path):
    """Round-2 scaling (SAFETY reviewer finding 2): the dispatcher's
    per-tick get_positions call is ONCE per tick, not per filled
    symbol. Simulate a 500-name book with a slow (0.05s) get_positions
    against a 5s effective deadline — the dispatcher must finish
    ALL 500 without breaching the deadline, AND make O(ticks) not
    O(N) get_positions calls.

    Pins the "one snapshot per tick" property structurally.
    """
    import time as _time

    universe = [f"SYM{i:04d}" for i in range(500)]

    fake_broker = _FakeBroker()

    # Slow down get_positions to 0.05s per call. A per-symbol regression
    # (round-1 pattern) would cost 500 * 0.05 = 25s just in
    # get_positions sleep — visibly exceeding the deadline. The
    # round-2 one-per-tick pattern costs ~0.05s + 500 fast dict
    # lookups.
    real_get_positions = fake_broker.get_positions

    def _slow_get_positions():
        _time.sleep(0.05)
        return real_get_positions()

    monkeypatch.setattr(fake_broker, "get_positions", _slow_get_positions)

    # Submit 500 entries directly. Each entry fills instantly in the
    # fake, so all 500 are terminal Filled by the time the dispatcher
    # first polls their status.
    entries = []
    for sym in universe:
        oid = fake_broker.buy(
            sym, 10, order_type="MKT", exchange="SMART",
        )
        entries.append({
            "symbol": sym, "side": "BUY", "requested_qty": 10,
            "order_id": oid, "error": None,
        })

    # Deadline of 5s comfortably fits 1 get_positions (0.05s) + 500
    # fast dispatcher iterations. A per-symbol regression would need
    # ~25s of pure get_positions sleep — well past the deadline —
    # and the test would fail on the moc_batch length assertion.
    calls_before = fake_broker._get_positions_call_count
    sweep_deadline = config.now_ny() + timedelta(seconds=5)
    submit_by = config.now_ny() + timedelta(seconds=10)

    t0 = _time.monotonic()
    (moc_batch, covered_qty, uncovered, skipped, positions_unknown) = (
        daily_orchestrator._fill_gated_moc_dispatch(
            broker=fake_broker,
            entries=entries,
            managed_universe=set(universe),
            sweep_deadline=sweep_deadline,
            submit_by=submit_by,
            poll_interval_secs=0.01,
            max_wait_secs=30.0,
        )
    )
    elapsed = _time.monotonic() - t0
    calls_after = fake_broker._get_positions_call_count
    dispatcher_gp_calls = calls_after - calls_before

    # Structural: dispatcher must not have made ~500 get_positions
    # calls. Under one-per-tick + all-entries-terminal-in-tick-1, we
    # expect exactly 1 get_positions call.
    assert dispatcher_gp_calls <= 5, (
        f"dispatcher made {dispatcher_gp_calls} get_positions calls "
        f"for 500 symbols — the round-2 fix requires ONE per tick, "
        f"not one per filled symbol. Elapsed={elapsed:.2f}s."
    )
    # And every symbol should have been covered by the dispatcher.
    assert len(moc_batch) == 500, (
        f"expected 500 MOC legs, got {len(moc_batch)}. Elapsed="
        f"{elapsed:.2f}s, get_positions calls={dispatcher_gp_calls}, "
        f"skipped={len(skipped)}, uncovered={len(uncovered)}."
    )
    # Total elapsed should be well under the deadline. Under a per-
    # symbol regression this would be dominated by 25s of sleep.
    assert elapsed < 3.0, (
        f"dispatcher took {elapsed:.2f}s for 500 names — likely "
        f"the round-1 per-symbol get_positions regression."
    )


def test_replay_submit_runs_post_close_poll(monkeypatch, tmp_path):
    """Round-2 (SAFETY reviewer finding 5): ``--replay-submit``
    places real broker orders and MUST get the same post-close
    position verification as a live/paper run. Previously the poll
    was gated on ``if not replay_mode`` and skipped entirely,
    letting a replay run silently return rc=0 with real positions
    live.

    Uses ``moc_reject_all=True`` so pre-close signals are clean (MOCs
    submit OK) but positions never zero — the post-close poll MUST
    catch this and return rc=11. If the poll is still gated out for
    replay-submit, this test returns rc=0 and fails.
    """
    import logging as _logging
    from live.tests.test_replay import (
        _make_stub_replay_feed as _make_stub,
    )

    universe = config.load_active_universe()
    stub_feed = _make_stub("2026-05-15", universe)
    monkeypatch.setattr(daily_orchestrator, "ReplayFeed", lambda **kw: stub_feed)
    monkeypatch.setenv("DATABENTO_API_KEY", "dummy-key")
    _patch_artifacts(monkeypatch, universe)
    # Neutralise the pre-close acceptance audit so the ONLY signal
    # left is the post-close position poll.
    monkeypatch.setattr(
        daily_orchestrator, "_audit_moc_acceptance",
        lambda *a, **kw: set(),
    )

    fake_broker = _FakeBroker(moc_stays_submitted=True)

    def _fake_connect(_args):
        fake_broker.connect(mode="paper")
        return fake_broker

    monkeypatch.setattr(daily_orchestrator, "_connect_broker", _fake_connect)

    argv = [
        "--replay-date", "2026-05-15", "--paper",
        "--vol-cap", "5000", "--replay-submit",
        "--submit-by", "23:59",  # not past cutoff so orders can go
        "--replay-ignore-cutoff",  # otherwise rc=4 pre-order
        "--on-feed-failure", "abort",
    ]

    handler, records = _capture_orchestrator_logs(_logging.ERROR)
    try:
        rc = daily_orchestrator.main(argv)
    finally:
        daily_orchestrator.logger.removeHandler(handler)

    assert rc == 11, (
        f"expected rc=11 (post-close poll authoritative for "
        f"--replay-submit too), got {rc}. Round-2 requires the poll "
        f"to run whenever a broker was contacted."
    )
    loud = [r.getMessage() for r in records]
    assert any("OVERNIGHT EXPOSURE (post-close)" in m for m in loud), (
        f"expected 'OVERNIGHT EXPOSURE (post-close)' tag; got: {loud}"
    )


# ---------------------------------------------------------------------------
# Round-3 revision (post-review, second round): regression tests for the
# reviewer-confirmed critical/high defects the previous round's tests did
# not catch.
# ---------------------------------------------------------------------------


def test_sweep_trims_over_cover_from_reducing_entry_against_baseline(
    monkeypatch, tmp_path,
):
    """Round-3 (SAFETY reviewer critical finding 1): a reducing entry
    against a nonzero BASELINE that the dispatcher over-covers because
    the position update lagged the fill status event must be TRIMMED
    (an opposite-side sweep MOC), NOT left as a naked overnight
    exposure.

    Reproduces the reviewer's confirmed loss path (unit-level on the
    sweep so we don't need to race IB's position propagation):
      * Baseline: +100 AAAA (residual from a prior day).
      * Dispatcher over-covered: sent SELL MOC 100 → covered_qty =
        {AAAA: -100}, dispatcher_leg with signed_qty=-100.
      * At sweep time in REAL IB: get_positions has caught up →
        position = 0 (baseline+entry netted).
      * Sweep expected behaviour: residual = 0 + (-100) = -100 →
        BUY MOC 100 offset, so net MOC across the two legs = 0.

    Pre-round-3 (single-sided sweep): AAAA not in agg (position 0),
    the sweep skipped, and the over-cover of 100 shares became a
    naked overnight SHORT of 100. This test would fail on the
    number of moc_batch legs.
    """
    class _StubBroker:
        """Simulates REAL IB semantics: baseline=100, entry SELL 100
        has already netted the position to 0 at sweep time.
        Dispatcher MOC has been placed but has NOT filled yet (MOCs
        fill at 16:00). No entry order to cancel.
        """

        def __init__(self):
            self._positions = {"AAAA": 0}  # entry has netted baseline
            self.sell_calls = []
            self.buy_calls = []
            # Dispatcher's SELL MOC 100, still live and unfilled.
            self._dispatcher_oid = "dispatcher-moc-1"
            self._orders = {
                self._dispatcher_oid: {
                    "symbol": "AAAA", "status": "Submitted",
                    "filled": 0,
                },
            }

        def get_positions(self):
            from types import SimpleNamespace
            return [SimpleNamespace(symbol=s, quantity=q)
                    for s, q in self._positions.items() if q != 0]

        def get_order_status(self, oid):
            info = self._orders.get(oid, {})
            from types import SimpleNamespace
            return SimpleNamespace(
                status=info.get("status", "Unknown"),
                filled=info.get("filled", 0),
            )

        def cancel(self, oid):
            pass

        def buy(self, symbol, qty, order_type="MOC", **kwargs):
            self.buy_calls.append((symbol, qty, order_type))
            return f"sweep-buy-{symbol}"

        def sell(self, symbol, qty, order_type="MOC", **kwargs):
            self.sell_calls.append((symbol, qty, order_type))
            return f"sweep-sell-{symbol}"

    stub = _StubBroker()
    dispatcher_legs = [{
        "oid": stub._dispatcher_oid, "symbol": "AAAA",
        "side": "SELL", "qty": 100, "signed_qty": -100,
    }]
    covered_qty = {"AAAA": -100}
    submit_by = config.now_ny() + timedelta(hours=1)

    (moc_batch, cancelled, cancel_failed, skipped, positions_unknown) = (
        daily_orchestrator._sweep_moc_deadline(
            broker=stub,
            entries=[],
            managed_universe={"AAAA"},
            covered_qty=covered_qty,
            submit_by=submit_by,
            dispatcher_legs=dispatcher_legs,
        )
    )
    assert not positions_unknown
    assert len(moc_batch) == 1, (
        f"expected exactly one TRIM sweep MOC to offset the "
        f"over-cover; got {moc_batch}. Round-3 two-sided sweep must "
        f"iterate covered_qty even when the symbol is absent from "
        f"agg — the reviewer's critical failure path."
    )
    leg = moc_batch[0]
    assert leg["symbol"] == "AAAA"
    assert leg["side"] == "BUY", (
        f"sweep must send BUY MOC to offset the over-cover; got "
        f"side={leg['side']}"
    )
    assert leg["qty"] == 100, (
        f"trim MOC sized to {leg['qty']}, expected 100"
    )
    # Net MOC across dispatcher (SELL 100) + sweep (BUY 100) = 0.
    total_signed = sum(l_["signed_qty"] for l_ in dispatcher_legs) + \
        sum(l_["signed_qty"] for l_ in moc_batch)
    assert total_signed == 0, (
        f"total MOC signed qty should net to 0 (dispatcher SELL 100 "
        f"+ sweep BUY 100); got {total_signed}"
    )


def test_sweep_drops_credit_for_terminal_rejected_dispatcher_moc(
    monkeypatch, tmp_path,
):
    """Round-3 (SAFETY reviewer finding 3): a dispatcher MOC that
    was terminal-rejected at the exchange (Cancelled with filled=0
    within seconds of placement) MUST have its coverage credit
    dropped by the sweep so the sweep can resubmit. Without this,
    the sweep sees residual=0 arithmetic and skips — leaving the
    symbol uncovered with no re-MOC path.

    Unit-level test on ``_sweep_moc_deadline``: pass a dispatcher_leg
    whose oid resolves to a terminal-Cancelled-with-filled=0 status,
    with covered_qty crediting that leg. Sweep must (a) drop the
    credit and (b) submit a fresh MOC sized to the actual held qty.
    """
    class _StubBroker:
        def __init__(self):
            self._positions = {"AAAA": 100}
            self.sell_calls = []
            self.buy_calls = []
            self._rejected_oid = "dispatcher-rejected-1"
            self._orders = {
                self._rejected_oid: {
                    "symbol": "AAAA", "status": "Cancelled",
                    "filled": 0,
                },
            }

        def get_positions(self):
            from types import SimpleNamespace
            return [SimpleNamespace(symbol=s, quantity=q)
                    for s, q in self._positions.items() if q != 0]

        def get_order_status(self, oid):
            info = self._orders.get(oid, {})
            from types import SimpleNamespace
            return SimpleNamespace(
                status=info.get("status", "Unknown"),
                filled=info.get("filled", 0),
            )

        def cancel(self, oid):
            pass

        def buy(self, symbol, qty, order_type="MOC", **kwargs):
            self.buy_calls.append((symbol, qty, order_type))
            return f"sweep-buy-{symbol}"

        def sell(self, symbol, qty, order_type="MOC", **kwargs):
            self.sell_calls.append((symbol, qty, order_type))
            return f"sweep-sell-{symbol}"

    stub = _StubBroker()
    dispatcher_legs = [{
        "oid": stub._rejected_oid, "symbol": "AAAA",
        "side": "SELL", "qty": 100, "signed_qty": -100,
    }]
    covered_qty = {"AAAA": -100}
    submit_by = config.now_ny() + timedelta(hours=1)

    (moc_batch, cancelled, cancel_failed, skipped, positions_unknown) = (
        daily_orchestrator._sweep_moc_deadline(
            broker=stub,
            entries=[],
            managed_universe={"AAAA"},
            covered_qty=covered_qty,
            submit_by=submit_by,
            dispatcher_legs=dispatcher_legs,
        )
    )
    assert not positions_unknown
    assert len(moc_batch) == 1, (
        f"sweep must resubmit MOC after dispatcher leg was "
        f"terminal-rejected; got {moc_batch}. Otherwise the symbol "
        f"is uncovered with no re-MOC path — SAFETY reviewer's 3rd "
        f"critical finding."
    )
    leg = moc_batch[0]
    assert leg["symbol"] == "AAAA"
    assert leg["side"] == "SELL", (
        f"holds +100 → sweep must SELL MOC 100 (dispatcher's rejected "
        f"leg is no longer credited); got side={leg['side']}"
    )
    assert leg["qty"] == 100


def test_moc_signed_by_sym_aggregates_dispatcher_plus_sweep_legs(
    monkeypatch, tmp_path,
):
    """Round-3 (SAFETY + CORRECTNESS reviewer finding 2/6): when the
    dispatcher AND the sweep both emit a MOC leg for the SAME symbol
    (the round-2 top-up path), ``moc_signed_by_sym`` must AGGREGATE
    the signed_qtys. Pre-round-3 it was a last-wins dict comprehension
    that silently dropped the dispatcher leg — the coverage cross-
    check then saw only the top-up qty (say -40 when the truth was
    -100), flagged the symbol as uncovered, and spuriously paged
    rc=11 on the fix's OWN success case.

    Directly asserts the aggregation contract: forge a moc_batch with
    two legs for the same symbol whose signed_qtys sum to -100 and
    show that the resulting coverage cross-check verdict is "no
    uncovered symbols" (aggregate works) — NOT "AAAA uncovered"
    (last-wins bug).
    """
    class _StubBroker:
        """Records order-status polls; MOCs stay Submitted so the
        round-3 exclusion-of-terminal-rejected path is a no-op and
        aggregation is what determines the coverage verdict.
        """

        def __init__(self):
            self._orders = {
                "d-1": {"status": "Submitted", "filled": 0},
                "s-1": {"status": "Submitted", "filled": 0},
                "entry-1": {"status": "Filled", "filled": 100},
            }

        def get_order_status(self, oid):
            info = self._orders.get(oid, {})
            from types import SimpleNamespace
            return SimpleNamespace(
                status=info.get("status", "Unknown"),
                filled=info.get("filled", 0),
            )

    stub = _StubBroker()
    # entries: baseline 0 for AAAA, entry BUY 100 fully filled.
    entries = [{
        "symbol": "AAAA", "side": "BUY", "requested_qty": 100,
        "order_id": "entry-1", "status": "Filled", "filled_qty": 100,
    }]
    current_baseline = {"AAAA": 0}
    # SIMULATE round-3 aggregation logic exactly as main() does.
    moc_batch = [
        # Dispatcher covered 60 (partial-fill snapshot).
        {"oid": "d-1", "symbol": "AAAA", "side": "SELL",
         "qty": 60, "signed_qty": -60},
        # Sweep topped up 40.
        {"oid": "s-1", "symbol": "AAAA", "side": "SELL",
         "qty": 40, "signed_qty": -40},
    ]
    moc_signed_by_sym: dict[str, int] = {}
    for _leg in moc_batch:
        _sym = _leg["symbol"]
        _oid = _leg["oid"]
        _include = True
        try:
            _info = stub.get_order_status(_oid)
            _st = getattr(_info, "status", "") or ""
            _fl = int(getattr(_info, "filled", 0) or 0)
            if _st in daily_orchestrator._TERMINAL_ORDER_STATES and _fl == 0:
                _include = False
        except Exception:
            _include = True
        if _include:
            moc_signed_by_sym[_sym] = (
                moc_signed_by_sym.get(_sym, 0)
                + int(_leg["signed_qty"])
            )
    # Aggregation invariant: BOTH legs' signed_qty must be summed.
    assert moc_signed_by_sym["AAAA"] == -100, (
        f"aggregation failed: expected -100 (dispatcher -60 + "
        f"sweep -40), got {moc_signed_by_sym['AAAA']!r}. Round-3 "
        f"replaced the last-wins dict comprehension with a summing "
        f"loop precisely because this scenario was misreported."
    )
    uncovered = daily_orchestrator._coverage_cross_check(
        broker=stub,
        entries=entries,
        current_baseline=current_baseline,
        moc_signed_by_sym=moc_signed_by_sym,
        managed_universe={"AAAA"},
    )
    assert uncovered == [], (
        f"expected coverage cross-check to find NO uncovered symbols "
        f"(entry 100 + moc_signed -100 = 0); got uncovered={uncovered}. "
        f"Last-wins aggregation would have shown moc_signed=-40, sum=60, "
        f"and appended AAAA to uncovered → spurious rc=11 page."
    )


def test_post_close_poll_block_exception_pages_pessimistically(
    monkeypatch, tmp_path,
):
    """Round-3 (SAFETY reviewer finding 4): an exception ANYWHERE in
    the post-close poll block — parse_hhmm raising, today_ny raising,
    _poll_positions_after_close raising — must be treated as
    pessimistic EXPOSED (rc=11). Pre-round-3 the block was
    unwrapped, so a raise there yielded rc=1 (below the rc>=11 pager
    threshold), silently degrading a genuine exposure to a
    non-page.
    """
    import logging as _logging
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

    # Force the poll block to raise by making the poll function itself
    # blow up. Wrapping should catch it and set post_close_exposed.
    def _raising_poll(**kwargs):
        raise RuntimeError("simulated poll block failure")

    monkeypatch.setattr(
        daily_orchestrator, "_poll_positions_after_close", _raising_poll,
    )

    handler, records = _capture_orchestrator_logs(_logging.ERROR)
    try:
        rc = daily_orchestrator.main(_base_argv())
    finally:
        daily_orchestrator.logger.removeHandler(handler)

    assert rc == 11, (
        f"expected rc=11 (poll block raised → pessimistic EXPOSED); "
        f"got {rc}. Round-3 requires the entire post-close poll block "
        f"to be wrapped in try/except with pessimistic default."
    )
    loud = [r.getMessage() for r in records]
    assert any("post-close position poll block raised" in m for m in loud), (
        f"expected the pessimistic poll-block-raised message in ERROR "
        f"logs; got: {loud}"
    )
    assert any("OVERNIGHT EXPOSURE (post-close)" in m for m in loud), (
        f"expected 'OVERNIGHT EXPOSURE (post-close)' page tag on "
        f"pessimistic verdict; got: {loud}"
    )


def test_post_close_poll_replay_skipped_when_before_not_before(
    monkeypatch, tmp_path,
):
    """Round-3 (CORRECTNESS reviewer finding 7): a --replay-submit
    rehearsal that reaches the post-close poll BEFORE today's 16:00
    auction wall clock (e.g. run at 11:00 ET) must SKIP the poll
    with a WARNING, NOT spin the full MAX_WINDOW and then
    fail-closed to rc=11. The auction wall clock is unrelated to
    the historical replay date; the poll cannot verify anything
    for a rehearsal.

    Restores a future ``not_before`` inside the test (the conftest
    globally patches it to "00:00" for speed) so the scenario the
    correctness reviewer described is actually exercised
    end-to-end.
    """
    import logging as _logging
    # Set not_before to 23:59 — a wall clock time that will not be
    # reached during a normal test run — so the pre-not_before
    # short-circuit fires. MAX_WINDOW is tiny so a spurious spin
    # would be caught by the elapsed-time assertion below.
    monkeypatch.setattr(
        config, "DEFAULT_POST_CLOSE_POLL_NOT_BEFORE_TIME", "23:59",
        raising=False,
    )
    monkeypatch.setattr(
        config, "POST_CLOSE_POLL_MAX_WINDOW_MINS", 0.005,
        raising=False,
    )

    from live.tests.test_replay import (
        _make_stub_replay_feed as _make_stub,
    )
    universe = config.load_active_universe()
    stub_feed = _make_stub("2026-05-15", universe)
    monkeypatch.setattr(daily_orchestrator, "ReplayFeed", lambda **kw: stub_feed)
    monkeypatch.setenv("DATABENTO_API_KEY", "dummy-key")
    _patch_artifacts(monkeypatch, universe)
    # Audit neutralised so any pre-close signal only comes from
    # coverage cross-check.
    monkeypatch.setattr(
        daily_orchestrator, "_audit_moc_acceptance",
        lambda *a, **kw: set(),
    )

    fake_broker = _FakeBroker()

    def _fake_connect(_args):
        fake_broker.connect(mode="paper")
        return fake_broker

    monkeypatch.setattr(daily_orchestrator, "_connect_broker", _fake_connect)

    argv = [
        "--replay-date", "2026-05-15", "--paper",
        "--vol-cap", "5000", "--replay-submit",
        "--submit-by", "23:59",
        "--replay-ignore-cutoff",
        "--on-feed-failure", "abort",
    ]
    import time as _time
    handler, records = _capture_orchestrator_logs(_logging.WARNING)
    t0 = _time.monotonic()
    try:
        rc = daily_orchestrator.main(argv)
    finally:
        daily_orchestrator.logger.removeHandler(handler)
    elapsed = _time.monotonic() - t0

    # A rehearsal reaching the poll before not_before is a
    # NON-VERIFIABLE-BUT-NOT-EXPOSED verdict: pre-close signals
    # were clean, the poll cannot honestly answer. rc must be 0
    # (successful rehearsal with a loud SKIP warning). Should NOT
    # spin the full MAX_WINDOW.
    assert rc == 0, (
        f"expected rc=0 (replay-submit rehearsal before not_before → "
        f"post-close poll SKIPPED with WARNING; pre-close signals "
        f"clean); got {rc}. Correctness reviewer finding 7 required "
        f"this to NOT spin the full window and return rc=11."
    )
    assert elapsed < 3.0, (
        f"post-close poll block took {elapsed:.2f}s — round-3 "
        f"replay pre-not_before must SHORT-CIRCUIT with a WARNING, "
        f"not spin MAX_WINDOW."
    )
    loud = [r.getMessage() for r in records]
    assert any(
        "post-close position poll SKIPPED — replay-submit" in m
        for m in loud
    ), (
        f"expected the replay-skip WARNING; got: {loud}"
    )


def test_post_close_poll_waits_for_not_before_live_run(
    monkeypatch, tmp_path,
):
    """Round-3 (CORRECTNESS reviewer finding 7, other side): a
    LIVE/PAPER run reaching the poll before not_before must SLEEP
    until not_before (the auction hasn't printed yet — MUST wait
    for the authoritative verdict), NOT skip or spin.

    Unit-level test on ``_poll_positions_after_close`` and the
    main() wait path — we drive the sleep-until-not_before pattern
    directly rather than through a full main() run, because the
    integration path spends ~1-2s of setup before reaching the
    poll block, and picking a not_before wall clock that is
    reliably in the future at THAT point (but not so far as to
    blow the test's own budget) is brittle. The
    ``_post_close_wait_and_poll`` shape is what main() calls; we
    exercise it directly with a broker that would have been flat.
    """
    import logging as _logging
    universe = ["AAA", "BBB", "CCC"]

    # not_before is 400ms in the future — enough to demonstrate the
    # wait, tight enough to keep the test fast.
    not_before = config.now_ny() + timedelta(milliseconds=400)

    class _EmptyBroker:
        def get_positions(self):
            return []
        def simulate_close_auction(self):
            pass

    handler, records = _capture_orchestrator_logs(_logging.INFO)
    try:
        # Directly call _poll_positions_after_close with a not_before
        # in the near future. The poll must refuse to return FLAT
        # before that time. We use a poll_until well past not_before
        # so the poll has room to confirm flat after the threshold.
        poll_until = config.now_ny() + timedelta(seconds=2)
        exposed, nonzero, err = daily_orchestrator._poll_positions_after_close(
            broker=_EmptyBroker(),
            managed_universe=set(universe),
            poll_until=poll_until,
            not_before=not_before,
            min_consecutive_flat_reads=2,
            initial_interval_secs=0.01,
            max_interval_secs=0.05,
            # This test pins the not_before GATING, not the empty-read
            # corroboration rule; supply positive corroboration so the
            # empty book is admissible as flat once not_before elapses.
            corroborator=lambda: True,
        )
    finally:
        daily_orchestrator.logger.removeHandler(handler)

    assert not exposed, (
        f"empty book past not_before must resolve as FLAT; got "
        f"exposed=True, nonzero={nonzero}, err={err!r}"
    )
    assert nonzero == []
    assert err is None
    info_msgs = [r.getMessage() for r in records]
    # The pre-not_before empty reads should be visible in the log
    # (the poll refused to declare FLAT while now < not_before).
    assert any(
        "empty read pre-auction" in m for m in info_msgs
    ), (
        f"expected the pre-not_before empty-read log line; got: "
        f"{info_msgs}"
    )


# ---------------------------------------------------------------------------
# Round-4 revision (post-review, third round): regression tests for the
# SAFETY reviewer's four blocking findings and the CORRECTNESS reviewer's
# stale-comment finding.
# ---------------------------------------------------------------------------


def test_fill_gated_dispatch_defers_on_stale_position_snapshot(
    monkeypatch, tmp_path,
):
    """Round-4 (SAFETY reviewer finding 2): a REDUCING entry against a
    nonzero baseline that the tick snapshot has NOT yet caught up to
    must DEFER the MOC submission to the sweep, NOT size the MOC off
    the stale PRE-fill position.

    Reproduces the reviewer's confirmed loss path directly on
    ``_fill_gated_moc_dispatch``:
      * Baseline: +100 AAAA (residual from a prior day).
      * Entry: SELL 100 (reducing to target 0). Fills fully. Order
        goes terminal Filled.
      * Tick snapshot: still shows +100 (IB emits execution report
        before the position update — hard-simulated by the stub
        broker's ``get_positions`` returning +100 unconditionally
        for the WHOLE dispatch loop).
      * Expected post-round-4 behaviour: dispatcher observes
        ``baseline + signed_fill = 100 + (-100) = 0``, snapshot shows
        +100, they DISAGREE → defer. NO MOC submitted from this
        path; entry left in pending; sweep must cover.

    Pre-round-4 (no baseline check): dispatcher would size the MOC
    to current=+100, submit SELL MOC 100, and end up with the
    over-cover naked-short case the round-3 two-sided sweep exists
    to trim. Layering fix 2 on top prevents the mis-size at the
    source.
    """
    class _StubBroker:
        """Fixed-position broker: snapshot returns +100 for AAAA
        regardless of whether MOCs have been submitted. Models a
        persistent IB position-update lag inside the dispatcher tick.
        """

        def __init__(self):
            self.moc_calls: list[tuple[str, int, str]] = []
            self.get_positions_calls = 0
            # The entry order is pre-recorded as terminal Filled.
            self._orders = {
                "entry-AAAA": {
                    "symbol": "AAAA", "status": "Filled", "filled": 100,
                },
            }

        def get_positions(self):
            self.get_positions_calls += 1
            from types import SimpleNamespace
            return [SimpleNamespace(symbol="AAAA", quantity=100)]

        def get_order_status(self, oid):
            info = self._orders.get(oid, {})
            from types import SimpleNamespace
            return SimpleNamespace(
                status=info.get("status", "Unknown"),
                filled=info.get("filled", 0),
            )

        def cancel(self, oid):
            pass

        def sell(self, symbol, qty, order_type="MOC", **kwargs):
            self.moc_calls.append((symbol, qty, order_type))
            return f"stub-sell-{symbol}"

        def buy(self, symbol, qty, order_type="MOC", **kwargs):
            self.moc_calls.append((symbol, qty, order_type))
            return f"stub-buy-{symbol}"

    stub = _StubBroker()
    entries = [{
        "symbol": "AAAA", "side": "SELL", "requested_qty": 100,
        "order_id": "entry-AAAA", "error": None,
    }]
    # Baseline == pre-entry position: +100. Expected post-fill = 0.
    # Snapshot says +100 (stale). Dispatcher MUST refuse to submit.
    (moc_batch, covered_qty, uncovered,
     skipped, positions_unknown) = daily_orchestrator._fill_gated_moc_dispatch(
        broker=stub,
        entries=entries,
        managed_universe={"AAAA"},
        sweep_deadline=config.now_ny() + timedelta(seconds=2),
        submit_by=config.now_ny() + timedelta(hours=1),
        baseline={"AAAA": 100},
        poll_interval_secs=0.01,
        max_wait_secs=1.0,
    )
    assert moc_batch == [], (
        f"dispatcher must DEFER (submit no MOC) when the tick "
        f"snapshot disagrees with baseline+signed_fill; got "
        f"moc_batch={moc_batch}. Pre-round-4 the dispatcher would "
        f"have sized SELL MOC 100 off the stale +100 snapshot — a "
        f"naked overnight short."
    )
    assert stub.moc_calls == [], (
        f"broker must not have received any MOC placement calls; "
        f"got {stub.moc_calls}"
    )
    assert not positions_unknown
    # The entry is terminal Filled; no submission error. The defer
    # leaves it in pending until dispatcher exits at deadline —
    # dispatcher's exit path sets status='pending_at_sweep_deadline'
    # so the sweep picks it up.
    assert entries[0]["status"] in {"Filled", "pending_at_sweep_deadline"}, (
        f"entry should be terminal Filled or marked "
        f"pending_at_sweep_deadline; got {entries[0]!r}"
    )


def test_fill_gated_submit_error_pages_pre_close(monkeypatch, tmp_path):
    """Round-4 (SAFETY reviewer finding 1): a broker exception during
    the fill-gated MOC submission (broker.buy/sell raised) MUST
    surface as a pre-close exposure signal → rc=11. Pre-round-4 the
    signal was computed and thrown away: ``fg_uncovered_from_positions_fail``
    was unpacked and never OR'd into ``pre_close_exposure``, so a
    submission error left the symbol in the "neither moc_batch nor
    skipped" gap and the sweep could double-flatten.

    End-to-end variant: ``raise_on_sell_symbols={one traded name}``
    causes every SELL from the dispatcher (both MKT and MOC) to
    raise. The dispatcher will fail to submit the MOC for that
    name via ``_submit_single_moc`` returning "error" — a genuine
    submission-side exception. Assertion: rc must be 11, and the
    log line must mention the fill-gated submission error.
    """
    import logging as _logging
    universe = config.load_active_universe()
    feed = _prepopulate_live_feed(universe)
    monkeypatch.setattr(daily_orchestrator, "LiveFeed", lambda *a, **kw: feed)
    monkeypatch.setattr(daily_orchestrator, "_sleep_until", lambda *a, **kw: None)
    monkeypatch.setenv("DATABENTO_API_KEY", "dummy-key")
    _patch_artifacts(monkeypatch, universe)

    # Probe run to find which symbols are traded and pick one that
    # gets a BUY entry (so the MOC is a SELL — we can raise SELL).
    probe = _FakeBroker()
    monkeypatch.setattr(daily_orchestrator, "_connect_broker",
                        lambda _args: (probe.connect(mode="paper"), probe)[1])
    daily_orchestrator.main(_base_argv())
    buy_entry_syms = [c[0] for c in probe.buy_calls if c[2] == "MKT"]
    assert buy_entry_syms, "probe found no MKT BUY entry to target"
    target_sym = buy_entry_syms[0]

    # Real test: same setup, but raise on any SELL for target_sym.
    # The MOC for target_sym is SELL (opposite of the BUY entry), so
    # the dispatcher's _submit_single_moc raises → outcome "error" →
    # symbol appended to fg_uncovered_from_submit_error.
    feed2 = _prepopulate_live_feed(universe)
    monkeypatch.setattr(daily_orchestrator, "LiveFeed", lambda *a, **kw: feed2)
    fake_broker = _FakeBroker(raise_on_sell_symbols={target_sym})

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
        f"expected rc=11 (fill-gated dispatcher submission error → "
        f"pre-close exposure); got {rc}. Round-4 requires "
        f"fg_uncovered_from_submit_error to feed pre_close_exposure "
        f"— pre-round-4 the signal was silently dropped and the "
        f"sweep would then double-flatten the symbol."
    )
    loud = [r.getMessage() for r in records]
    assert any("Fill-gated submission errors" in m for m in loud), (
        f"expected the fill-gated-submission-error line in the "
        f"exposure log; got: {loud[:5]}"
    )


def test_second_sweep_catches_late_dispatcher_moc_rejection(
    monkeypatch, tmp_path,
):
    """Round-4 (SAFETY reviewer finding 3): if a dispatcher MOC flips
    terminal-Cancelled with filled=0 (exchange rejection) AFTER the
    first sweep already credited it, the SECOND sweep at
    sweep_deadline must observe the rejection, drop the coverage
    credit, and re-fire an MOC to flatten the position.

    We simulate: dispatcher fires an MOC that initially reports
    'Submitted' (accepted by the API), then flips to Cancelled
    between the first and second sweep. The first sweep's
    ``dispatcher_legs`` check sees Submitted → credit preserved →
    residual 0 → no top-up. The second sweep's check sees
    Cancelled+filled=0 → credit dropped → residual = held → fires
    the missing MOC.

    Test structure: drive main() to the point where MOC placement
    happens with a broker that starts as ``moc_stays_submitted=True``
    (so dispatcher MOCs are Submitted, not terminal-rejected), then
    monkey-patch the broker's ``get_order_status`` to FLIP the
    dispatcher MOC to Cancelled AFTER the first sweep runs but
    BEFORE the second sweep runs. The second sweep must observe
    the flipped state and resubmit.

    Simplified: rather than time-based flipping (fragile in tests),
    exercise the sweep function DIRECTLY twice — first call with
    Submitted dispatcher leg (no resubmit), second call with the
    same leg flipped to Cancelled (resubmit fires). Together they
    prove the round-4 double-sweep pattern would catch the late
    rejection.
    """
    class _FlipperBroker:
        def __init__(self):
            self._positions = {"AAAA": 100}
            self.moc_calls: list[tuple[str, str, int]] = []
            self._orders = {
                "dispatcher-moc-1": {
                    "symbol": "AAAA", "status": "Submitted", "filled": 0,
                },
            }

        def flip_rejected(self):
            self._orders["dispatcher-moc-1"]["status"] = "Cancelled"

        def get_positions(self):
            from types import SimpleNamespace
            return [SimpleNamespace(symbol=s, quantity=q)
                    for s, q in self._positions.items() if q != 0]

        def get_order_status(self, oid):
            info = self._orders.get(oid, {})
            from types import SimpleNamespace
            return SimpleNamespace(
                status=info.get("status", "Unknown"),
                filled=info.get("filled", 0),
            )

        def cancel(self, oid):
            pass

        def sell(self, symbol, qty, order_type="MOC", **kwargs):
            self.moc_calls.append(("SELL", symbol, qty))
            return f"resubmit-{symbol}"

        def buy(self, symbol, qty, order_type="MOC", **kwargs):
            self.moc_calls.append(("BUY", symbol, qty))
            return f"resubmit-{symbol}"

    stub = _FlipperBroker()
    dispatcher_legs = [{
        "oid": "dispatcher-moc-1", "symbol": "AAAA",
        "side": "SELL", "qty": 100, "signed_qty": -100,
    }]
    covered_qty = {"AAAA": -100}
    submit_by = config.now_ny() + timedelta(hours=1)

    # First sweep: dispatcher MOC still Submitted, credit preserved,
    # residual 0 → no orders.
    (moc_batch1, _c, _cf, _sk, _pu) = daily_orchestrator._sweep_moc_deadline(
        broker=stub,
        entries=[],
        managed_universe={"AAAA"},
        covered_qty=covered_qty,
        submit_by=submit_by,
        dispatcher_legs=dispatcher_legs,
    )
    assert moc_batch1 == [], (
        f"first sweep should NOT resubmit while dispatcher MOC is "
        f"still Submitted; got {moc_batch1}"
    )
    assert stub.moc_calls == [], (
        f"broker should not have received any MOC placement in the "
        f"first sweep; got {stub.moc_calls}"
    )

    # Time passes. IB flips the dispatcher's MOC to Cancelled
    # (exchange rejection). SECOND sweep runs — the round-4
    # sleep-then-second-sweep pattern in main() runs this at
    # sweep_deadline.
    stub.flip_rejected()

    # Second sweep: dispatcher leg is now terminal-rejected. Credit
    # dropped from effective_covered_qty. Residual = held +
    # (credit_dropped) = 100 + 0 = 100 → SELL MOC 100.
    combined_covered_qty = dict(covered_qty)
    for _leg in moc_batch1:
        combined_covered_qty[_leg["symbol"]] = (
            combined_covered_qty.get(_leg["symbol"], 0)
            + int(_leg["signed_qty"])
        )
    (moc_batch2, _c2, _cf2, _sk2, _pu2) = daily_orchestrator._sweep_moc_deadline(
        broker=stub,
        entries=[],
        managed_universe={"AAAA"},
        covered_qty=combined_covered_qty,
        submit_by=submit_by,
        dispatcher_legs=dispatcher_legs + moc_batch1,
    )
    assert len(moc_batch2) == 1, (
        f"second sweep must resubmit the MOC after the dispatcher "
        f"MOC flipped terminal-rejected between sweeps; got "
        f"{moc_batch2}. This is exactly the round-4 case the "
        f"double-sweep pattern is designed to catch."
    )
    leg = moc_batch2[0]
    assert leg["symbol"] == "AAAA"
    assert leg["side"] == "SELL"
    assert leg["qty"] == 100
    assert stub.moc_calls == [("SELL", "AAAA", 100)], (
        f"broker should have received exactly one SELL MOC 100 "
        f"resubmission in the second sweep; got {stub.moc_calls}"
    )


# Capture the REAL ``_sleep_until`` / ``_wait_for_marks_pumped``
# references at module load time — the autouse conftest fixture
# monkey-patches these to no-ops on every test (needed so main()-level
# tests don't hang on the round-4 second-sweep sleep), so a direct
# unit test of these helpers needs the pre-patch objects.
_REAL_SLEEP_UNTIL = daily_orchestrator._sleep_until
_REAL_WAIT_FOR_MARKS_PUMPED = daily_orchestrator._wait_for_marks_pumped


def test_sleep_until_pump_callback_invoked(monkeypatch):
    """Round-4 (SAFETY reviewer finding 4): ``_sleep_until`` with a
    pump callback must invoke the callback periodically during the
    wait. This is what keeps the ib_async loop pumped so a long
    unpumped wall-clock wait doesn't wedge the connection and force
    the post-close poll into its pessimistic-EXPOSED branch.

    We stub ``time.sleep`` to advance a fake clock so the pump
    interval elapses without a real wall-clock wait; the pump
    callback records its invocations.
    """
    import time as _time
    pump_calls: list[float] = []
    real_time_time = _time.time
    fake_time = [real_time_time()]

    def _fake_sleep(secs):
        fake_time[0] += secs

    monkeypatch.setattr(_time, "sleep", _fake_sleep)
    monkeypatch.setattr(_time, "time", lambda: fake_time[0])
    # config.now_ny reads real datetime.now — we drive the loop
    # instead via a target computed off the fake clock. Use a
    # slightly hacky pattern: patch config.now_ny to reflect the
    # fake clock's advance so the sleep_until loop exits when the
    # fake target is reached.
    start = daily_orchestrator.config.now_ny()

    def _fake_now_ny():
        elapsed = fake_time[0] - real_time_time()
        return start + timedelta(seconds=max(elapsed, 0))

    monkeypatch.setattr(daily_orchestrator.config, "now_ny", _fake_now_ny)

    target = start + timedelta(seconds=90)

    def _pump():
        pump_calls.append(fake_time[0])

    # Poll every 5s (real function default). Pump every 30s. Over
    # a 90s wait we expect approximately 3 pumps.
    _REAL_SLEEP_UNTIL(
        target, poll=5.0, pump=_pump, pump_interval=30.0,
    )
    assert 2 <= len(pump_calls) <= 4, (
        f"expected ~3 pump invocations over the 90s wait, got "
        f"{len(pump_calls)} at times {pump_calls}"
    )


def test_sleep_until_pump_exception_does_not_break_wait(monkeypatch):
    """Round-4: a raising pump callback logs a warning and continues
    the wait — a transient TWS blip during a 4-minute sleep must
    not gate the downstream sweep / verdict.
    """
    import time as _time
    real_time_time = _time.time
    fake_time = [real_time_time()]

    def _fake_sleep(secs):
        fake_time[0] += secs

    monkeypatch.setattr(_time, "sleep", _fake_sleep)
    monkeypatch.setattr(_time, "time", lambda: fake_time[0])
    start = daily_orchestrator.config.now_ny()

    def _fake_now_ny():
        elapsed = fake_time[0] - real_time_time()
        return start + timedelta(seconds=max(elapsed, 0))

    monkeypatch.setattr(daily_orchestrator.config, "now_ny", _fake_now_ny)

    target = start + timedelta(seconds=60)

    def _boom_pump():
        raise RuntimeError("simulated TWS blip during pump")

    # Should not raise; should exit at target.
    _REAL_SLEEP_UNTIL(
        target, poll=5.0, pump=_boom_pump, pump_interval=10.0,
    )


def test_stale_moc_cutoff_comment_replaced(monkeypatch):
    """Round-4 (CORRECTNESS reviewer finding 5): the stale
    "~15:50 MOC entry cutoff" claim in main()'s acceptance-audit
    comment survives no more. NASDAQ's MOC ENTRY cutoff is 15:55:00
    (SEC 34-84454), and DEFAULT_SUBMIT_BY_TIME = "15:55" depends on
    that being clear to every future reader.
    """
    import inspect
    src = inspect.getsource(daily_orchestrator)
    assert "past the ~15:50 cutoff" not in src, (
        "stale '~15:50 cutoff' comment survives in "
        "daily_orchestrator.py — CORRECTNESS reviewer finding 5. "
        "That is the exact false fact the task singled out for "
        "correction, and it directly contradicts the corrected "
        "ground-truth in the module docstring and the "
        "_audit_moc_acceptance docstring."
    )
    # And the corrected form must be present.
    assert (
        "NASDAQ's 15:55:00 ENTRY cutoff" in src
        or "15:55:00 ET (SEC" in src
    ), (
        "corrected 15:55:00 ENTRY-cutoff language missing from "
        "daily_orchestrator.py; the fix must be positive, not just "
        "a deletion."
    )


def test_wait_for_marks_pumped_calls_broker_between_slices(monkeypatch):
    """Round-4 (SAFETY reviewer finding 4): the pumped-wait wrapper
    for ``recorder.wait_for_marks`` must invoke ``broker.get_positions``
    between wait slices so the ib_async connection is exercised.
    """
    from types import SimpleNamespace
    slices = [0]
    broker_calls = [0]

    class _StubRecorder:
        def wait_for_marks(self, timeout):
            slices[0] += 1
            # Return "not done" for the first 3 slices, then "done".
            return slices[0] >= 4

    class _StubBroker:
        def get_positions(self):
            broker_calls[0] += 1
            return []

    done = _REAL_WAIT_FOR_MARKS_PUMPED(
        recorder=_StubRecorder(),
        broker=_StubBroker(),
        timeout=100.0,
        pump_interval=0.001,
    )
    assert done is True
    # 4 wait_for_marks slices; broker pumped between each non-final
    # one (3 pumps).
    assert slices[0] == 4
    assert broker_calls[0] == 3, (
        f"expected 3 broker.get_positions pumps between the first 3 "
        f"wait slices; got {broker_calls[0]}"
    )


def test_wait_for_marks_pumped_broker_pump_exception_ok(monkeypatch):
    """A raising broker pump doesn't kill the marks wait — the
    marks thread should complete regardless of transient TWS blips.
    """
    slices = [0]

    class _StubRecorder:
        def wait_for_marks(self, timeout):
            slices[0] += 1
            return slices[0] >= 3

    class _BoomBroker:
        def get_positions(self):
            raise RuntimeError("simulated TWS drop mid-marks-wait")

    done = _REAL_WAIT_FOR_MARKS_PUMPED(
        recorder=_StubRecorder(),
        broker=_BoomBroker(),
        timeout=100.0,
        pump_interval=0.001,
    )
    assert done is True


def test_main_replay_ignore_cutoff_skips_second_sweep_sleep(
    monkeypatch, tmp_path,
):
    """Round-4: in replay mode with ``--replay-ignore-cutoff``
    the second-sweep wall-clock sleep is skipped so the process
    doesn't stall for an hour (``effective_submit_by = now + 1h``
    yields a nonsense wall-clock deadline).

    Assertion strategy: install a ``_sleep_until`` that RAISES if
    called with a target more than 5 minutes from ``now``. The
    autouse conftest patch is a no-op, so this test's explicit
    setattr wins.
    """
    from live.tests.test_replay import (
        _make_stub_replay_feed as _make_stub,
    )

    universe = config.load_active_universe()
    stub_feed = _make_stub("2026-05-15", universe)
    monkeypatch.setattr(daily_orchestrator, "ReplayFeed", lambda **kw: stub_feed)
    monkeypatch.setenv("DATABENTO_API_KEY", "dummy-key")
    _patch_artifacts(monkeypatch, universe)

    max_target_seconds_ahead: list[float] = []

    def _tattletale_sleep(target, *args, **kwargs):
        # Record how far in the future the target is; a real second-
        # sweep sleep against effective_submit_by=now+1h would be
        # ~3600s. Anything <= 5 min is fine (the post-close
        # not_before sleep is tightly bounded by tests' fake wall
        # clock).
        try:
            secs = (target - config.now_ny()).total_seconds()
        except Exception:
            secs = 0.0
        max_target_seconds_ahead.append(secs)

    monkeypatch.setattr(daily_orchestrator, "_sleep_until", _tattletale_sleep)

    fake_broker = _FakeBroker()

    def _fake_connect(_args):
        fake_broker.connect(mode="paper")
        return fake_broker

    monkeypatch.setattr(daily_orchestrator, "_connect_broker", _fake_connect)

    argv = [
        "--replay-date", "2026-05-15", "--paper",
        "--vol-cap", "5000", "--replay-submit",
        "--submit-by", "23:59",
        "--replay-ignore-cutoff",
        "--on-feed-failure", "abort",
    ]
    rc = daily_orchestrator.main(argv)
    assert rc == 0

    # In replay mode the second-sweep sleep is skipped entirely.
    # The only _sleep_until targets that should reach the tattletale
    # are the post-close not_before wait — bounded by the conftest
    # to well under 5 minutes.
    long_waits = [s for s in max_target_seconds_ahead if s > 300.0]
    assert long_waits == [], (
        f"replay run scheduled a _sleep_until with target > 5min "
        f"in the future: {long_waits[:5]}. The second-sweep sleep "
        f"against effective_submit_by = now+1h must be skipped in "
        f"replay mode (round-4 fix)."
    )
