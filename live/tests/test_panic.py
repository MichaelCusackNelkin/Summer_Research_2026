"""Tests for the kill switch.

Small — covers the safety-critical branches only, but includes explicit
regression coverage for the H1 (per-leg flatten with per-leg failures
that must not skip AFTER + verdict), H2 (lingering non-panic orders
count as not-flat), and H3 (existing MOC past cutoff warns) findings.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from live import config
from live import panic


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeOpenOrder:
    def __init__(self, order_id, symbol, action, quantity, order_type,
                 status="Submitted", filled=0):
        self.order_id = order_id
        self.symbol = symbol
        self.action = action
        self.quantity = quantity
        self.order_type = order_type
        self.status = status
        self.filled = filled


class _FakePosition:
    def __init__(self, symbol, quantity, avg_cost=100.0):
        self.symbol = symbol
        self.quantity = quantity
        self.avg_cost = avg_cost


class _FakeBroker:
    """Ordered-event-log stand-in for ``ibkr_broker.broker.Broker``.

    ``events`` records the sequence of destructive calls so tests can
    assert the cancel-before-flatten invariant (M1). Positions and
    orders can be manipulated per-fixture to simulate stuck fills,
    lingering orders, and partial failures.
    """

    LIVE_CONFIRM_MSG = "YES_I_WANT_LIVE_TRADING"

    def __init__(
        self,
        initial_orders=(),
        initial_positions=(),
        residual_orders=(),
        residual_positions=None,
        close_position_side_effect=None,
    ):
        self._orders = list(initial_orders)
        self._positions = list(initial_positions)
        # After a MKT flatten the fake normally reports flat positions;
        # tests can override this to simulate stuck flatten.
        self._residual_positions_override = residual_positions
        # Orders that survive cancel_all (e.g. MOC past cutoff).
        self._residual_orders = list(residual_orders)
        # Optional side effect callable(symbol, order_type) -> str|None|raises.
        self._close_side_effect = close_position_side_effect

        self.connected = False
        self.disconnected = False
        self.events: list[str] = []
        self.cancel_all_calls = 0
        # (symbol, order_type, exchange) — exchange added so the
        # routing regression can assert every leg carries ACTIVE_ROUTING
        # (see live/nasdaq_routing.py).
        self.close_position_calls: list[tuple[str, str, str]] = []

    def connect(self, mode="paper", **kwargs):
        self.connected = True

    def disconnect(self):
        self.disconnected = True

    def sleep(self, seconds: float):
        return None

    def get_open_orders(self):
        return list(self._orders)

    def get_positions(self):
        return list(self._positions)

    def cancel_all(self):
        self.cancel_all_calls += 1
        self.events.append("cancel")
        # By default, cancel clears the order book EXCEPT for residuals
        # that survived (e.g. MOC past cutoff).
        self._orders = list(self._residual_orders)

    def close_position(self, symbol, order_type="MKT", tif="DAY",
                       exchange=None):
        # ``exchange`` defaults to ``None`` (not the real broker's
        # ``"SMART"``) so a call site that omits the kwarg is
        # OBSERVABLE in ``close_position_calls`` — a
        # ``(sym, order_type, None)`` tuple tells the routing
        # regression the caller relied on the implicit default.
        # Under the real broker's ``exchange="SMART"`` default and
        # the current ``ACTIVE_ROUTING == "SMART"``, an omission
        # would otherwise be indistinguishable from a correctly-
        # threaded constant and the guard would go dead.
        self.events.append(f"close:{symbol}:{order_type}")
        # Record the exchange too so the routing regression can assert
        # every panic-side flatten leg carries the ACTIVE_ROUTING code
        # (see live/nasdaq_routing.py).
        self.close_position_calls.append((symbol, order_type, exchange))
        if self._close_side_effect is not None:
            return self._close_side_effect(symbol, order_type)
        # Default: place a synthetic order id and, for MKT, zero the position.
        oid = f"panic-oid-{len(self.close_position_calls)}"
        self._orders.append(_FakeOpenOrder(oid, symbol, "SELL", 1, order_type))
        if order_type == "MKT":
            self._positions = [
                _FakePosition(p.symbol, 0, p.avg_cost) if p.symbol == symbol else p
                for p in self._positions
            ]
        return oid

    def apply_residuals(self):
        """Called by the test to swap in the post-flatten state override."""
        if self._residual_positions_override is not None:
            self._positions = list(self._residual_positions_override)

    def __repr__(self):
        return "FakeBroker(connected)"


def _install_broker(monkeypatch, broker):
    monkeypatch.setattr(panic, "_connect_broker", lambda args: broker)


# ---------------------------------------------------------------------------
# Behaviour tests
# ---------------------------------------------------------------------------


def test_default_run_cancels_and_flattens(monkeypatch):
    broker = _FakeBroker(
        initial_orders=[_FakeOpenOrder("1", "AAPL", "BUY", 100, "LMT")],
        initial_positions=[_FakePosition("MSFT", 50), _FakePosition("TSLA", -25)],
    )
    _install_broker(monkeypatch, broker)
    rc = panic.main(["--paper", "--yes"])
    assert rc == 0, f"expected rc=0, got {rc}"
    assert broker.cancel_all_calls == 1
    # M1: cancel must come first, then close_position per name.
    assert broker.events[0] == "cancel", broker.events
    close_events = [e for e in broker.events if e.startswith("close:")]
    assert len(close_events) == 2, close_events
    assert broker.disconnected


def test_cancel_precedes_flatten_ordering(monkeypatch):
    """M1: nothing about the flatten loop happens before cancel_all."""
    broker = _FakeBroker(
        initial_positions=[_FakePosition("MSFT", 10)],
    )
    _install_broker(monkeypatch, broker)
    panic.main(["--paper", "--yes"])
    # Find the position of "cancel" and every "close:*" event; cancel
    # must come strictly before all closes.
    cancel_idx = broker.events.index("cancel")
    close_idxs = [i for i, e in enumerate(broker.events) if e.startswith("close:")]
    assert all(i > cancel_idx for i in close_idxs), broker.events


def test_dry_run_places_nothing(monkeypatch):
    broker = _FakeBroker(
        initial_orders=[_FakeOpenOrder("1", "AAPL", "BUY", 100, "LMT")],
        initial_positions=[_FakePosition("MSFT", 50)],
    )
    _install_broker(monkeypatch, broker)
    rc = panic.main(["--paper", "--dry-run"])
    assert rc == 0
    assert broker.cancel_all_calls == 0
    assert broker.close_position_calls == []
    assert broker.events == []
    assert broker.disconnected


def test_cancel_only_skips_flatten(monkeypatch):
    broker = _FakeBroker(
        initial_orders=[_FakeOpenOrder("1", "AAPL", "BUY", 100, "LMT")],
        initial_positions=[_FakePosition("MSFT", 50)],
    )
    _install_broker(monkeypatch, broker)
    rc = panic.main(["--paper", "--cancel-only", "--yes"])
    assert rc == 0
    assert broker.cancel_all_calls == 1
    assert broker.close_position_calls == []


def test_per_leg_flatten_survives_partial_failure(monkeypatch, capsys):
    """H1: one close_position() raise must NOT abort the batch; the AFTER
    snapshot and exit-code verdict must ALWAYS run.
    """
    # First name raises, second succeeds and clears.
    def _side_effect(symbol, order_type):
        if symbol == "BADSYM":
            raise ValueError("simulated qualification failure")
        # Default MKT behaviour: return oid; we mutate positions after.
        return f"oid-for-{symbol}"

    broker = _FakeBroker(
        initial_positions=[_FakePosition("BADSYM", 10), _FakePosition("GOOD", 20)],
        close_position_side_effect=_side_effect,
        # Simulate post-flatten world: BADSYM remains, GOOD cleared.
        residual_positions=[_FakePosition("BADSYM", 10), _FakePosition("GOOD", 0)],
    )
    # We need the residual to be applied after the flatten loop runs.
    # Wire it via a monkeypatched sleep that triggers the swap.
    orig_sleep = broker.sleep
    swap_calls = [0]

    def _sleep_and_swap(seconds):
        # Second sleep call happens right after the flatten loop.
        swap_calls[0] += 1
        if swap_calls[0] == 2:
            broker.apply_residuals()
        return orig_sleep(seconds)

    broker.sleep = _sleep_and_swap  # type: ignore[method-assign]

    _install_broker(monkeypatch, broker)
    rc = panic.main(["--paper", "--yes"])
    out = capsys.readouterr().out
    # AFTER snapshot ran despite the partial failure.
    assert "=== AFTER ===" in out
    # Failure was reported.
    assert "BADSYM" in out
    # Non-zero rc because BADSYM position remains.
    assert rc == 1, f"expected rc=1 (partial failure), got {rc}"
    # Both legs were attempted (the good one wasn't skipped).
    # Panic threads ACTIVE_ROUTING ("SMART") through close_position, so
    # the recorded tuples include the exchange code.
    assert broker.close_position_calls == [
        ("BADSYM", "MKT", "SMART"),
        ("GOOD", "MKT", "SMART"),
    ]


def test_lingering_non_panic_order_marks_not_flat(monkeypatch, capsys):
    """H2: an order that survives cancel_all (e.g. MOC past cutoff) must
    make the MKT flatten verdict rc=1 — panic's own placed legs must NOT
    trip the check.
    """
    # A resting MOC order that cancel_all can't pull.
    lingering = _FakeOpenOrder("lingering-1", "MSFT", "BUY", 100, "MOC")
    broker = _FakeBroker(
        initial_orders=[lingering],
        initial_positions=[_FakePosition("AAPL", 10)],  # will be MKT-flattened
        residual_orders=[lingering],  # survives cancel_all
    )
    _install_broker(monkeypatch, broker)
    rc = panic.main(["--paper", "--yes"])
    out = capsys.readouterr().out
    assert "survived cancel_all" in out
    assert "MSFT" in out
    assert rc == 1, f"expected rc=1 (lingering order), got {rc}"


def test_lingering_only_panic_own_orders_still_flat(monkeypatch, capsys):
    """H2 caveat: panic's own newly-placed flatten legs (still working in
    the AFTER snapshot) must NOT count as lingering.
    """
    broker = _FakeBroker(
        initial_positions=[_FakePosition("AAPL", 10)],
    )
    # No residual_orders — the only orders in the AFTER snapshot are panic's
    # own synthetic legs.
    _install_broker(monkeypatch, broker)
    rc = panic.main(["--paper", "--yes"])
    out = capsys.readouterr().out
    # Panic's own oid should be in the AFTER snapshot's order list.
    assert "panic-oid" in out
    # But rc=0 because those are the flatten legs, not something we failed
    # to cancel.
    assert rc == 0, f"expected rc=0 (only panic's own legs open), got {rc}"


def test_existing_moc_past_cutoff_warning_regardless_of_flatten_flag(
    monkeypatch, capsys,
):
    """H3: warning is driven by the BEFORE snapshot + clock, NOT --flatten-with.
    Even with the default MKT flatten, an existing MOC past NASDAQ's
    15:50 CANCEL/MODIFY cutoff (round-2 fix — this is the relevant
    cutoff here, not the 15:55 entry cutoff) must surface a loud
    warning naming the count.
    """
    late = datetime(2026, 6, 15, 15, 55, tzinfo=config.NY_TZ)
    monkeypatch.setattr(config, "now_ny", lambda: late)
    broker = _FakeBroker(
        initial_orders=[
            _FakeOpenOrder("ex-1", "AAPL", "BUY", 100, "MOC"),
            _FakeOpenOrder("ex-2", "MSFT", "SELL", 50, "MOC"),
        ],
        # These MOC orders survive cancel_all.
        residual_orders=[
            _FakeOpenOrder("ex-1", "AAPL", "BUY", 100, "MOC"),
            _FakeOpenOrder("ex-2", "MSFT", "SELL", 50, "MOC"),
        ],
    )
    _install_broker(monkeypatch, broker)
    panic.main(["--paper", "--dry-run"])  # dry-run so we just inspect
    out = capsys.readouterr().out
    assert "2 MOC order(s) present in the BEFORE snapshot" in out
    assert "15:50" in out and "CANCEL" in out.upper(), (
        f"expected 15:50 CANCEL/MODIFY cutoff mention in output; got:\n{out}"
    )
    assert "cannot" in out.lower()


def test_moc_flatten_past_cutoff_returns_nonzero(monkeypatch, capsys):
    """M4: --flatten-with MOC past NASDAQ's 15:55 ENTRY cutoff (SEC
    34-84454, 2018 rule) must NOT claim success. Round-2 fix: the
    previous test used 15:55 which is exactly AT the cutoff (>= in
    _past_moc_entry_cutoff), so pick 15:56 to be unambiguously past."""
    late = datetime(2026, 6, 15, 15, 56, tzinfo=config.NY_TZ)
    monkeypatch.setattr(config, "now_ny", lambda: late)
    broker = _FakeBroker(
        initial_positions=[_FakePosition("MSFT", 50)],
        # MOC path doesn't zero positions intraday — they stay non-flat.
    )
    _install_broker(monkeypatch, broker)
    rc = panic.main(["--paper", "--flatten-with", "MOC", "--yes"])
    out = capsys.readouterr().out
    assert "NOT FLAT" in out
    assert rc == 1, f"expected rc=1 (MOC past cutoff), got {rc}"


def test_moc_flatten_before_cutoff_returns_zero(monkeypatch):
    """MOC before the cutoff still returns rc=0 with an auction note."""
    early = datetime(2026, 6, 15, 14, 30, tzinfo=config.NY_TZ)
    monkeypatch.setattr(config, "now_ny", lambda: early)
    broker = _FakeBroker(
        initial_positions=[_FakePosition("MSFT", 50)],
    )
    _install_broker(monkeypatch, broker)
    rc = panic.main(["--paper", "--flatten-with", "MOC", "--yes"])
    assert rc == 0


def test_mkt_flatten_returns_nonzero_when_positions_remain(monkeypatch):
    """MKT should fill in the sleep window; if positions still show,
    something is wrong and we exit non-zero.
    """
    # Simulate stuck ticker by returning a residual position via side effect.
    def _side_effect(symbol, order_type):
        return "oid-stuck"  # don't zero the position

    broker = _FakeBroker(
        initial_positions=[_FakePosition("MSFT", 50)],
        close_position_side_effect=_side_effect,
    )
    _install_broker(monkeypatch, broker)
    rc = panic.main(["--paper", "--flatten-with", "MKT", "--yes"])
    assert rc == 1


def test_live_without_confirm_string_refused(monkeypatch):
    """Isolated: the confirm-string check is in main() before any Broker
    import, so the test does not depend on ib_async being installed.
    Distinct rc=9 (from the generic rc=8 connect failure).
    """
    def _blowup(_args):
        raise AssertionError("connect must not be called on missing --live-confirm")

    monkeypatch.setattr(panic, "_connect_broker", _blowup)
    rc = panic.main(["--live", "--yes"])
    assert rc == 9, f"expected rc=9 (missing --live-confirm), got {rc}"


def test_broker_connect_generic_failure_returns_rc_8(monkeypatch):
    """A real connect failure (TWS down, live-port guard, bad client id)
    returns rc=8 — distinct from the confirm-string case."""
    def _boom(_args):
        raise RuntimeError("simulated TWS refuse")

    monkeypatch.setattr(panic, "_connect_broker", _boom)
    rc = panic.main(["--paper", "--yes"])
    assert rc == 8, f"expected rc=8 (connect failure), got {rc}"


def test_live_prompt_abort_places_nothing(monkeypatch):
    broker = _FakeBroker(initial_positions=[_FakePosition("MSFT", 50)])
    _install_broker(monkeypatch, broker)
    monkeypatch.setattr("builtins.input", lambda _prompt="": "no")
    rc = panic.main(["--live", "--live-confirm", panic._LIVE_CONFIRM_MSG])
    assert rc == 2
    assert broker.cancel_all_calls == 0
    assert broker.close_position_calls == []


def test_live_prompt_accepts_panic(monkeypatch):
    broker = _FakeBroker(initial_positions=[_FakePosition("MSFT", 50)])
    _install_broker(monkeypatch, broker)
    monkeypatch.setattr("builtins.input", lambda _prompt="": "PANIC")
    rc = panic.main(["--live", "--live-confirm", panic._LIVE_CONFIRM_MSG])
    assert rc == 0
    assert broker.cancel_all_calls == 1
    assert broker.close_position_calls == [("MSFT", "MKT", "SMART")]


def test_moc_startup_warning_after_cutoff(monkeypatch, capsys):
    """Pre-connect warning that new MOC submissions past NASDAQ's
    15:55 entry cutoff (SEC 34-84454, 2018 rule) will likely be
    rejected. Round-2 fix: the panic script used to encode 15:50 as
    the entry cutoff, which is FALSE for NASDAQ — 15:50 is the
    cancel/modify cutoff. The startup warning is now at 15:55."""
    # 15:56 is past NASDAQ's 15:55 entry cutoff.
    late = datetime(2026, 6, 15, 15, 56, tzinfo=config.NY_TZ)
    monkeypatch.setattr(config, "now_ny", lambda: late)
    broker = _FakeBroker()
    _install_broker(monkeypatch, broker)
    panic.main(["--paper", "--flatten-with", "MOC", "--dry-run"])
    out = capsys.readouterr().out
    assert "after NASDAQ's 15:55 ET entry cutoff" in out


def test_mkt_startup_warning_outside_hours(monkeypatch, capsys):
    weekend = datetime(2026, 6, 14, 12, 0, tzinfo=config.NY_TZ)  # Sunday
    monkeypatch.setattr(config, "now_ny", lambda: weekend)
    broker = _FakeBroker()
    _install_broker(monkeypatch, broker)
    panic.main(["--paper", "--flatten-with", "MKT", "--dry-run"])
    out = capsys.readouterr().out
    assert "MKT outside regular hours" in out


def test_moc_flatten_before_cutoff_with_lingering_order_returns_nonzero(
    monkeypatch, capsys,
):
    """R1: MOC pre-cutoff branch must respect ``lingering_orders``, matching
    the MKT branch's H2 rule. If cancel_all() failed to pull a pre-existing
    non-panic order, --flatten-with MOC before 15:50 must NOT return rc=0 —
    otherwise a non-panic order rides into the close silently.
    """
    early = datetime(2026, 6, 15, 14, 30, tzinfo=config.NY_TZ)
    monkeypatch.setattr(config, "now_ny", lambda: early)
    # A resting non-panic LMT order that cancel_all can't pull for whatever
    # reason (stuck PendingCancel, etc.).
    lingering = _FakeOpenOrder("lingering-lmt", "AAPL", "BUY", 100, "LMT")
    broker = _FakeBroker(
        initial_orders=[lingering],
        initial_positions=[_FakePosition("MSFT", 50)],
        residual_orders=[lingering],  # survives cancel_all
    )
    _install_broker(monkeypatch, broker)
    rc = panic.main(["--paper", "--flatten-with", "MOC", "--yes"])
    out = capsys.readouterr().out
    # The MOC pre-cutoff success line still fires...
    assert "MOC flatten submitted" in out
    # ...but the lingering order defeats the clean exit.
    assert "survived cancel_all" in out
    assert "AAPL" in out
    assert rc == 1, f"expected rc=1 (lingering non-panic order), got {rc}"


def test_after_snapshot_tws_dropout_marks_could_not_confirm(monkeypatch, capsys):
    """R2: if the AFTER snapshot's broker calls raise (TWS dropped mid-run),
    print a loud COULD-NOT-CONFIRM line and return rc=1 — do NOT fall through
    to the verdict, which would compare against stale/empty data.
    """
    broker = _FakeBroker(
        initial_positions=[_FakePosition("MSFT", 50)],
    )
    # Fail only the AFTER-snapshot calls: BEFORE snapshot + flatten loop's
    # get_positions() succeed, then the next two calls raise.
    call_count = [0]
    orig_get_positions = broker.get_positions
    orig_get_open_orders = broker.get_open_orders

    def _get_positions_flaky():
        call_count[0] += 1
        # calls 1 (BEFORE) and 2 (flatten iterator) succeed, call 3 (AFTER) raises.
        if call_count[0] >= 3:
            raise ConnectionError("simulated TWS drop")
        return orig_get_positions()

    open_call_count = [0]

    def _get_open_orders_flaky():
        open_call_count[0] += 1
        # call 1 is BEFORE, call 2 is AFTER — fail AFTER.
        if open_call_count[0] >= 2:
            raise ConnectionError("simulated TWS drop")
        return orig_get_open_orders()

    broker.get_positions = _get_positions_flaky  # type: ignore[method-assign]
    broker.get_open_orders = _get_open_orders_flaky  # type: ignore[method-assign]

    _install_broker(monkeypatch, broker)
    rc = panic.main(["--paper", "--yes"])
    out = capsys.readouterr().out
    assert "COULD NOT CONFIRM FLAT" in out
    assert "TWS unreachable" in out
    assert "INVESTIGATE" in out
    # AFTER snapshot never printed — the fail-loud path bypassed it.
    assert "=== AFTER ===" not in out
    assert rc == 1, f"expected rc=1 (AFTER snapshot dropout), got {rc}"
    # Disconnect still runs in the finally.
    assert broker.disconnected


def test_cancel_all_tws_dropout_marks_orders_may_still_be_live(monkeypatch, capsys):
    """N1: if ``broker.cancel_all()`` raises (TWS drops exactly at cancel time),
    print a loud "cancel_all FAILED" line and return rc=1 — do NOT fall through
    to the flatten loop (a TWS drop that killed cancel_all will kill
    close_position too). Disconnect must still run in the outer ``finally``,
    and no bare traceback may escape ``main()``.
    """
    broker = _FakeBroker(
        initial_orders=[_FakeOpenOrder("1", "AAPL", "BUY", 100, "LMT")],
        initial_positions=[_FakePosition("MSFT", 50)],
    )

    def _cancel_all_fails():
        broker.cancel_all_calls += 1
        broker.events.append("cancel-fail")
        raise ConnectionError("simulated TWS drop at cancel_all")

    broker.cancel_all = _cancel_all_fails  # type: ignore[method-assign]

    _install_broker(monkeypatch, broker)
    # main() must not let the exception escape.
    rc = panic.main(["--paper", "--yes"])
    out = capsys.readouterr().out
    assert "cancel_all FAILED" in out
    assert "orders may still be LIVE" in out
    assert "INVESTIGATE" in out
    # cancel_all was reached (call counted) but flatten never was.
    assert broker.cancel_all_calls == 1
    assert broker.close_position_calls == []
    # AFTER snapshot never printed — we bailed before it.
    assert "=== AFTER ===" not in out
    assert rc == 1, f"expected rc=1 (cancel_all dropout), got {rc}"
    # Disconnect still runs in the outer finally.
    assert broker.disconnected


def test_flatten_loop_get_positions_dropout_continues_to_after(
    monkeypatch, capsys,
):
    """R2 sibling: if the flatten loop's own ``get_positions()`` raises
    (TWS drops after cancel_all but before we can enumerate positions),
    we log an ERROR, skip the loop, and continue to the AFTER snapshot.
    The verdict then reflects the actual state we can still read.
    """
    broker = _FakeBroker(
        initial_positions=[_FakePosition("MSFT", 50)],
    )
    # BEFORE snapshot succeeds; flatten-loop get_positions() (call 2) raises;
    # AFTER get_positions() (call 3) succeeds and still shows MSFT.
    call_count = [0]
    orig_get_positions = broker.get_positions

    def _get_positions_flaky():
        call_count[0] += 1
        if call_count[0] == 2:
            raise ConnectionError("simulated TWS drop mid-flatten")
        return orig_get_positions()

    broker.get_positions = _get_positions_flaky  # type: ignore[method-assign]

    _install_broker(monkeypatch, broker)
    rc = panic.main(["--paper", "--yes"])
    out = capsys.readouterr().out
    # Cancelled ran before the drop.
    assert broker.cancel_all_calls == 1
    # ERROR was logged; no leg was attempted.
    assert "could not fetch positions to flatten" in out
    assert broker.close_position_calls == []
    # AFTER snapshot still printed.
    assert "=== AFTER ===" in out
    # MSFT is still non-flat, so rc=1 from the normal MKT verdict.
    assert rc == 1, f"expected rc=1 (positions remain after failed flatten), got {rc}"


def test_panic_flatten_routes_every_leg_via_active_routing(monkeypatch):
    """Routing regression: every close_position call panic issues during
    a flatten MUST carry ``exchange=ACTIVE_ROUTING`` — the same routing
    code the daily orchestrator uses (see ``live/nasdaq_routing.py``).
    Currently ``ACTIVE_ROUTING == "SMART"``. The kill switch is subject
    to the same routing contract as the daily; if a future flip changes
    ``ACTIVE_ROUTING`` (back to ISLAND, or forward to something else),
    the panic path must move with it.
    """
    from live.nasdaq_routing import ACTIVE_ROUTING

    broker = _FakeBroker(
        initial_positions=[
            _FakePosition("AAPL", 100),
            _FakePosition("MSFT", -50),
            _FakePosition("NVDA", 25),
        ],
    )
    _install_broker(monkeypatch, broker)
    rc = panic.main(["--paper", "--yes"])
    assert rc == 0
    # Every recorded flatten leg pinned to the currently-active route.
    assert broker.close_position_calls, "no flatten legs were attempted"
    for sym, order_type, exchange in broker.close_position_calls:
        assert exchange == ACTIVE_ROUTING == "SMART", (
            f"panic flatten leg for {sym} ({order_type}) routed to "
            f"exchange={exchange!r}, expected {ACTIVE_ROUTING!r}"
        )


def test_panic_flatten_moc_also_uses_active_routing(monkeypatch):
    """Routing regression, MOC branch: even the ``--flatten-with MOC``
    end-of-day flatten must carry ``exchange=ACTIVE_ROUTING`` so the
    kill switch stays consistent with the daily orchestrator's routing
    contract (see ``live/nasdaq_routing.py``).
    """
    from datetime import datetime as _dt
    from live.nasdaq_routing import ACTIVE_ROUTING

    early = _dt(2026, 6, 15, 14, 30, tzinfo=config.NY_TZ)
    monkeypatch.setattr(config, "now_ny", lambda: early)
    broker = _FakeBroker(
        initial_positions=[_FakePosition("MSFT", 50), _FakePosition("AMZN", -10)],
    )
    _install_broker(monkeypatch, broker)
    rc = panic.main(["--paper", "--flatten-with", "MOC", "--yes"])
    assert rc == 0
    assert broker.close_position_calls
    for sym, order_type, exchange in broker.close_position_calls:
        assert order_type == "MOC"
        assert exchange == ACTIVE_ROUTING == "SMART", (
            f"panic MOC leg for {sym} routed to exchange={exchange!r}, "
            f"expected {ACTIVE_ROUTING!r}"
        )


def test_panic_flatten_never_relies_on_broker_implicit_default(monkeypatch):
    """Companion pin for the panic path: no ``close_position`` call may
    omit the ``exchange`` kwarg and rely on ``Broker.close_position``'s
    ``exchange="SMART"`` default. The fake broker defaults ``exchange``
    to ``None`` (unlike the real broker), so an omission surfaces as a
    ``None`` in the recorded tuple. If a future edit drops the
    ``exchange=ACTIVE_ROUTING`` kwarg from ``panic.py::main``, this
    fires — same purpose as
    ``test_no_implicit_default_routing_leaks_through_live_paths`` for
    the orchestrator.
    """
    broker = _FakeBroker(
        initial_positions=[
            _FakePosition("AAPL", 100),
            _FakePosition("MSFT", -50),
            _FakePosition("NVDA", 25),
        ],
    )
    _install_broker(monkeypatch, broker)
    rc = panic.main(["--paper", "--yes"])
    assert rc == 0
    assert broker.close_position_calls, "no flatten legs were attempted"
    implicit = [
        (sym, order_type) for sym, order_type, exchange
        in broker.close_position_calls if exchange is None
    ]
    assert not implicit, (
        f"panic path relied on the broker's implicit exchange default "
        f"on: {implicit}"
    )


def test_confirm_msg_stays_in_sync_with_broker():
    """The module-level constant must match ``Broker.LIVE_CONFIRM_MSG``.

    If ibkr_broker is not importable in this environment we skip; the
    check runs whenever the broker package is available.
    """
    try:
        from ibkr_broker.broker import Broker
    except Exception:
        pytest.skip("ibkr_broker not importable")
    assert panic._LIVE_CONFIRM_MSG == Broker.LIVE_CONFIRM_MSG
