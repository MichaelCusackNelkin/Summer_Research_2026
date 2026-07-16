"""
Integration tests for broker.py against REAL TWS Paper Trading.

██████████████████████████████████████████████████████████████████
██  PAPER TRADING ONLY — NEVER RUN THESE AGAINST A LIVE ACCOUNT ██
██████████████████████████████████████████████████████████████████

Prerequisites:
    1. TWS (or IB Gateway) running with paper trading enabled
    2. API connections enabled in TWS: Configure > API > Settings
       - "Enable ActiveX and Socket Clients" checked
       - Socket port = 7497 (paper default)
       - "Download open orders on connection" checked
    3. pip install ib_async (or ib_insync) pytest

Usage:
    pytest test_broker_live.py -v                   # all tests
    pytest test_broker_live.py -v -k "test_buy"     # one test
    pytest test_broker_live.py -v -m market_hours    # only market-hours tests
    pytest test_broker_live.py -v -m "not market_hours"  # skip market-hours tests

    Override connection settings via env vars:
        TWS_HOST=127.0.0.1 TWS_PORT=7497 TWS_CLIENT_ID=99 pytest ...

Notes:
    - Tests use 1-2 shares of liquid stocks (AAPL, MSFT) to minimize impact
    - Limit orders are placed far from the market so they won't fill
    - Tests tagged @market_hours need the US equity market to be open
    - The test session starts by cancelling all orders and flattening
      positions, so don't run while you have paper trades you care about
"""

from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta

import pytest

from ibkr_broker import Broker, OrderInfo, Position


# ---------------------------------------------------------------------------
# Configuration — override with env vars if your setup differs
# ---------------------------------------------------------------------------

TWS_HOST = os.environ.get("TWS_HOST", "127.0.0.1")
TWS_PORT = int(os.environ.get("TWS_PORT", "7497"))       # 7497 = paper ONLY
TWS_CLIENT_ID = int(os.environ.get("TWS_CLIENT_ID", "99"))  # unique to avoid conflicts

# Stocks used in tests — very liquid US equities
SYMBOL_A = "AAPL"
SYMBOL_B = "MSFT"

# Small size so paper account isn't blown up
TEST_QTY = 1

# Custom pytest marker
market_hours = pytest.mark.market_hours


# ---------------------------------------------------------------------------
# Safety: hard-fail if someone points this at a live port
# ---------------------------------------------------------------------------

LIVE_PORTS = {7496, 4001}


def _assert_paper_only():
    if TWS_PORT in LIVE_PORTS:
        pytest.exit(
            f"\n\n"
            f"  ██ DANGER: TWS_PORT={TWS_PORT} is a LIVE trading port. ██\n"
            f"  ██ These tests are for PAPER TRADING ONLY.              ██\n"
            f"  ██ Aborting immediately to protect your account.        ██\n"
            f"\n",
            returncode=1,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_us_market_open() -> bool:
    """Rough check: US equities trade 9:30-16:00 ET on weekdays."""
    et = timezone(timedelta(hours=-5))  # EST (close enough; DST shifts ±1h)
    now = datetime.now(et)
    if now.weekday() >= 5:  # Sat/Sun
        return False
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= now <= market_close


# Sane per-symbol fallback prices used only when broker.get_price() returns
# None (no market data subscription, market closed, etc.). Rough mid-2026
# quotes; the exact number doesn't matter — it only has to be > 0 so the
# far-from-market math produces a genuinely unfillable price.
_FALLBACK_PRICES: dict[str, float] = {
    "AAPL": 200.0,
    "MSFT": 400.0,
}

# Cache one lookup per symbol per session. broker.get_price() sleeps ~2s
# internally, so we don't want to pay that for every LMT test.
_REFERENCE_PRICE_CACHE: dict[str, float] = {}


def _reference_price(broker, symbol: str) -> float:
    """
    Return a positive reference price for *symbol*, cached per session.

    Falls back to the per-symbol fallback in ``_FALLBACK_PRICES`` if the
    live snapshot is unavailable, then to $100 for anything unknown. The
    guarantee to callers is only that the result is > 0 — enough for
    ``_far_away_limit_price`` to produce a non-zero far-from-market price.
    """
    if symbol in _REFERENCE_PRICE_CACHE:
        return _REFERENCE_PRICE_CACHE[symbol]
    px = broker.get_price(symbol)
    if px is None or px <= 0:
        px = _FALLBACK_PRICES.get(symbol, 100.0)
    _REFERENCE_PRICE_CACHE[symbol] = float(px)
    return _REFERENCE_PRICE_CACHE[symbol]


def _far_away_limit_price(side: str, reference: float) -> float:
    """
    Return a limit price far from *reference* that will not fill.

    BUY: 70% below reference.  SELL: 200% above reference.

    A caller who forgets to pass a real reference used to silently get
    $0.00 back (the old default was ``reference=0.0``), which is
    immediately marketable — a SELL LMT @ $0 fills at the bid and a BUY
    STP @ $0 triggers instantly. We now reject non-positive references
    outright so the bug can't recur.
    """
    if reference is None or reference <= 0:
        raise ValueError(
            f"reference price must be > 0 (got {reference!r}). Fetch a "
            f"real market price via _reference_price(broker, symbol) so "
            f"far-from-market is genuinely far from market."
        )
    if side == "BUY":
        return round(reference * 0.30, 2)   # 70% below
    else:
        return round(reference * 3.0, 2)    # 200% above


def _far_from_market(broker, symbol: str, side: str) -> float:
    """Convenience: reference-price lookup + far-from-market in one call."""
    return _far_away_limit_price(side, _reference_price(broker, symbol))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def broker():
    """
    Session-wide Broker connected to TWS paper trading.
    Connects once, cleans slate, yields, then disconnects.
    """
    _assert_paper_only()

    b = Broker()
    try:
        b.connect(
            mode="paper",
            host=TWS_HOST,
            port=TWS_PORT,
            client_id=TWS_CLIENT_ID,
        )
    except Exception as e:
        pytest.skip(
            f"Could not connect to TWS at {TWS_HOST}:{TWS_PORT} — "
            f"is paper TWS running with API enabled? Error: {e}"
        )

    assert b.is_connected(), "Broker reports not connected after connect()"
    assert b.mode == "paper", f"Expected paper mode, got {b.mode}"

    # Clean slate: cancel everything and flatten
    b.cancel_all()
    b.sleep(2)
    b.flatten_all()
    b.sleep(3)

    yield b

    # Teardown: clean up again and disconnect
    try:
        b.cancel_all()
        b.sleep(1)
        b.flatten_all()
        b.sleep(2)
    finally:
        b.disconnect()


@pytest.fixture()
def clean_orders(broker):
    """Cancel all open orders before and after a test."""
    broker.cancel_all()
    broker.sleep(1)
    yield
    broker.cancel_all()
    broker.sleep(1)


@pytest.fixture()
def clean_positions(broker):
    """Flatten all positions before and after a test."""
    broker.cancel_all()
    broker.sleep(1)
    broker.flatten_all()
    broker.sleep(3)
    yield
    broker.cancel_all()
    broker.sleep(1)
    broker.flatten_all()
    broker.sleep(3)


# =========================================================================
# 1. CONNECTION
# =========================================================================

class TestConnection:

    def test_is_connected(self, broker):
        assert broker.is_connected() is True

    def test_mode_is_paper(self, broker):
        assert broker.mode == "paper"

    def test_repr(self, broker):
        r = repr(broker)
        assert "paper" in r
        assert "connected" in r

    def test_ib_property_returns_raw_object(self, broker):
        ib = broker.ib
        assert ib is not None
        assert ib.isConnected()

    def test_sleep_works(self, broker):
        broker.sleep(0.5)  # just shouldn't crash

    def test_live_connect_rejected_without_confirmation(self):
        b = Broker()
        with pytest.raises(RuntimeError, match="live_confirm"):
            b.connect(mode="live")

    def test_invalid_mode_rejected(self):
        b = Broker()
        with pytest.raises(ValueError, match="mode must be"):
            b.connect(mode="yolo")


# =========================================================================
# 2. ACCOUNT & PORTFOLIO (read-only, always safe)
# =========================================================================

class TestAccountInfo:

    def test_get_account_summary_has_data(self, broker):
        summary = broker.get_account_summary()
        assert isinstance(summary, dict)
        assert len(summary) > 0
        assert "NetLiquidation" in summary
        nlv = float(summary["NetLiquidation"])
        assert nlv > 0, "Paper account NLV should be positive"

    def test_get_account_summary_has_key_fields(self, broker):
        summary = broker.get_account_summary()
        for key in ["NetLiquidation", "BuyingPower", "AvailableFunds"]:
            assert key in summary, f"Missing key: {key}"

    def test_get_pnl_returns_floats(self, broker):
        pnl = broker.get_pnl()
        assert isinstance(pnl["unrealized"], float)
        assert isinstance(pnl["realized"], float)

    def test_get_margin_returns_positive_available(self, broker):
        margin = broker.get_margin()
        assert margin["available"] > 0

    def test_get_positions_returns_list_of_position(self, broker):
        positions = broker.get_positions()
        assert isinstance(positions, list)
        for p in positions:
            assert isinstance(p, Position)

    def test_show_account(self, broker, capsys):
        broker.show_account()
        assert "Account Summary" in capsys.readouterr().out

    def test_show_positions(self, broker):
        broker.show_positions()
        # No crash is the test; output depends on state

    def test_show_orders(self, broker):
        broker.show_orders()


# =========================================================================
# 3. INPUT VALIDATION (no orders placed)
# =========================================================================

class TestValidation:

    def test_zero_quantity(self, broker):
        with pytest.raises(ValueError, match="positive"):
            broker.buy(SYMBOL_A, 0)

    def test_negative_quantity(self, broker):
        with pytest.raises(ValueError, match="positive"):
            broker.sell(SYMBOL_A, -5)

    def test_exceeds_max_qty(self, broker):
        with pytest.raises(ValueError, match="MAX_ORDER_QTY"):
            broker.buy(SYMBOL_A, 999_999)

    def test_limit_without_price(self, broker):
        with pytest.raises(ValueError, match="limit_price"):
            broker.buy(SYMBOL_A, 1, "LMT")

    def test_stop_without_price(self, broker):
        with pytest.raises(ValueError, match="stop_price"):
            broker.sell(SYMBOL_A, 1, "STP")

    def test_bad_order_type(self, broker):
        with pytest.raises(ValueError, match="Unsupported"):
            broker.buy(SYMBOL_A, 1, "VWAP")

    def test_bad_symbol(self, broker):
        with pytest.raises(ValueError, match="Could not qualify"):
            broker.buy("ZZZNOTREAL123", 1)

    def test_cancel_unknown_order(self, broker):
        with pytest.raises(ValueError, match="Unknown order_id"):
            broker.cancel("999999999")

    def test_order_status_unknown(self, broker):
        with pytest.raises(ValueError, match="Unknown order_id"):
            broker.get_order_status("999999999")

    def test_wait_for_fill_unknown(self, broker):
        with pytest.raises(ValueError, match="Unknown order_id"):
            broker.wait_for_fill("999999999")

    def test_rebalance_rejects_lmt(self, broker):
        with pytest.raises(ValueError, match="only supports MKT"):
            broker.rebalance({"AAPL": 1}, order_type="LMT")

    def test_rebalance_accepts_moc(self, broker):
        # MOC is a supported order_type for rebalance() — the call still
        # needs a connection to actually place orders, but no ValueError
        # should be raised on the order_type check. To test the check in
        # isolation (without placing real orders) we call with an empty
        # target dict when the account is already flat/near-flat; either
        # way, the assertion is that MOC does NOT raise "only supports".
        try:
            broker.rebalance({}, order_type="MOC", close_unlisted=False)
        except ValueError as e:
            assert "only supports" not in str(e), (
                f"rebalance() should accept MOC, but raised: {e}"
            )


# =========================================================================
# 3b. MOC (Market-on-Close) — OFFLINE UNIT TESTS
# These do NOT require a live TWS.  They construct fresh Broker() objects
# and monkey-patch the underlying IB client so we can inspect the exact
# Order object built by ``_place()`` without hitting the network.
# =========================================================================

from unittest.mock import MagicMock  # noqa: E402  (kept local to this block)

from ibkr_broker.broker import Broker as _Broker  # noqa: E402


def _make_stubbed_broker() -> tuple["_Broker", list]:
    """
    Build a Broker whose ``_ib`` is a stub, so ``_place()`` executes end
    to end without a real TWS.  Returns (broker, captured_orders) where
    captured_orders is a list of (contract, order) tuples appended in
    the placeOrder shim.  Position-checking methods (``get_positions``)
    are stubbed to return an empty list so ``short_moc()``'s safety
    check finds no long position by default.
    """
    b = _Broker()
    captured: list = []

    fake_contract = MagicMock(name="fake_contract")
    fake_contract.symbol = "AAPL"

    def fake_place_order(contract, order):
        captured.append((contract, order))
        trade = MagicMock(name="fake_trade")
        trade.order = order
        trade.order.orderId = 4242
        return trade

    b._ib = MagicMock(name="fake_ib")
    b._ib.isConnected.return_value = True
    b._ib.qualifyContracts.return_value = [fake_contract]
    b._ib.placeOrder.side_effect = fake_place_order
    b._ib.positions.return_value = []
    # In ib_async 2.x, reqPositions() is blocking and returns the same
    # list positions() would return. Have the stub forward so tests that
    # only override positions.return_value still see the expected data
    # via get_positions() → reqPositions().
    b._ib.reqPositions.side_effect = lambda: b._ib.positions()

    return b, captured


class TestMOCOffline:
    """MOC behaviour that can be tested without a live gateway."""

    def test_place_builds_moc_order_with_day_tif(self):
        b, captured = _make_stubbed_broker()
        oid = b._place(
            "AAPL", 100, "BUY",
            order_type="MOC", limit_price=None, stop_price=None, tif="DAY",
        )
        assert oid == "4242"
        assert len(captured) == 1
        _, order = captured[0]
        assert order.orderType == "MOC"
        assert order.action == "BUY"
        assert order.totalQuantity == 100
        assert order.tif == "DAY"

    def test_place_coerces_non_day_tif_for_moc(self):
        """MOC + tif='GTC' must be silently coerced to 'DAY'."""
        b, captured = _make_stubbed_broker()
        b._place(
            "AAPL", 10, "SELL",
            order_type="MOC", limit_price=None, stop_price=None, tif="GTC",
        )
        _, order = captured[0]
        assert order.orderType == "MOC"
        assert order.tif == "DAY", (
            f"tif should be forced to DAY for MOC, got {order.tif!r}"
        )

    def test_place_moc_ignores_limit_and_stop_prices(self):
        """A MOC order should never carry the caller's limit/stop price."""
        b, captured = _make_stubbed_broker()
        # Even if the caller passes rogue prices, the MOC Order shouldn't
        # get them attached (MOC uses IB's default sentinel prices).
        b._place(
            "AAPL", 5, "BUY",
            order_type="MOC", limit_price=999.0, stop_price=888.0, tif="DAY",
        )
        _, order = captured[0]
        assert order.orderType == "MOC"
        # Order constructed via generic Order() has ib's default sentinel
        # for lmtPrice/auxPrice (sys.float_info.max), NOT the values we
        # passed in. The important guarantee: MOC didn't route through
        # LimitOrder/StopOrder.
        assert order.lmtPrice != 999.0
        assert order.auxPrice != 888.0

    def test_bad_order_type_error_message_mentions_moc(self):
        """A confused caller passing e.g. 'VWAP' should see MOC in the hint."""
        b, _ = _make_stubbed_broker()
        with pytest.raises(ValueError, match="MOC"):
            b._place(
                "AAPL", 1, "BUY",
                order_type="VWAP", limit_price=None, stop_price=None, tif="DAY",
            )

    def test_buy_moc_places_moc_order(self):
        b, captured = _make_stubbed_broker()
        oid = b.buy_moc("AAPL", 50)
        assert oid == "4242"
        _, order = captured[0]
        assert order.orderType == "MOC"
        assert order.action == "BUY"
        assert order.totalQuantity == 50
        assert order.tif == "DAY"

    def test_buy_moc_rejects_zero_quantity(self):
        b, _ = _make_stubbed_broker()
        with pytest.raises(ValueError, match="positive"):
            b.buy_moc("AAPL", 0)

    def test_buy_moc_rejects_negative_quantity(self):
        b, _ = _make_stubbed_broker()
        with pytest.raises(ValueError, match="positive"):
            b.buy_moc("AAPL", -5)

    def test_buy_moc_enforces_max_order_qty(self):
        b, _ = _make_stubbed_broker()
        with pytest.raises(ValueError, match="MAX_ORDER_QTY"):
            b.buy_moc("AAPL", 999_999)

    def test_buy_moc_rejects_when_disconnected(self):
        # Fresh, un-stubbed broker — is_connected() is really False.
        b = _Broker()
        with pytest.raises(RuntimeError, match="not connected"):
            b.buy_moc("AAPL", 100)

    def test_short_moc_places_sell_moc_order(self):
        b, captured = _make_stubbed_broker()
        # By default the stubbed IB reports no positions, so short_moc
        # sees no existing long and proceeds.
        oid = b.short_moc("AAPL", 25)
        assert oid == "4242"
        _, order = captured[0]
        assert order.orderType == "MOC"
        assert order.action == "SELL"
        assert order.totalQuantity == 25
        assert order.tif == "DAY"

    def test_short_moc_blocked_when_long(self):
        """short_moc must inherit short()'s no-long-position safety."""
        b, _ = _make_stubbed_broker()

        # Fake a long position of 10 shares in AAPL.
        fake_pos = MagicMock(name="fake_position")
        fake_pos.contract.symbol = "AAPL"
        fake_pos.position = 10
        fake_pos.avgCost = 150.0
        b._ib.positions.return_value = [fake_pos]

        with pytest.raises(RuntimeError, match="LONG position"):
            b.short_moc("AAPL", 5)

    def test_short_moc_rejects_when_disconnected(self):
        # Fresh, un-stubbed broker. short_moc calls get_positions() first
        # (which needs a real IB connection); is_connected() is False, so
        # the safety of raising a clear RuntimeError happens inside
        # _place(). But get_positions() runs against the real _ib and
        # would error obscurely — so we call it with a broker that has
        # a stubbed positions() returning [] and stubbed isConnected
        # returning False.
        b = _Broker()
        b._ib.isConnected = MagicMock(return_value=False)
        b._ib.reqPositions = MagicMock(return_value=[])
        b._ib.positions = MagicMock(return_value=[])
        b._ib.sleep = MagicMock()
        with pytest.raises(RuntimeError, match="not connected"):
            b.short_moc("AAPL", 100)

    def test_short_moc_rejects_zero_quantity(self):
        # Quantity is validated inside _place() after the long-position
        # check passes — a stubbed broker with no positions is enough.
        b, _ = _make_stubbed_broker()
        with pytest.raises(ValueError, match="positive"):
            b.short_moc("AAPL", 0)

    def test_short_moc_enforces_max_order_qty(self):
        b, _ = _make_stubbed_broker()
        with pytest.raises(ValueError, match="MAX_ORDER_QTY"):
            b.short_moc("AAPL", 999_999)


class TestFarAwayLimitPrice:
    """The far-from-market helper must never return a marketable price."""

    def test_rejects_zero_reference(self):
        with pytest.raises(ValueError, match="reference price must be > 0"):
            _far_away_limit_price("BUY", 0.0)

    def test_rejects_negative_reference(self):
        with pytest.raises(ValueError, match="reference price must be > 0"):
            _far_away_limit_price("SELL", -10.0)

    def test_rejects_none_reference(self):
        with pytest.raises(ValueError, match="reference price must be > 0"):
            _far_away_limit_price("BUY", None)  # type: ignore[arg-type]

    def test_buy_far_below_market(self):
        # BUY LMT should sit well below reference so it never fills.
        px = _far_away_limit_price("BUY", 200.0)
        assert px == 60.0
        assert px < 200.0

    def test_sell_far_above_market(self):
        # SELL LMT should sit well above reference so it never fills.
        px = _far_away_limit_price("SELL", 200.0)
        assert px == 600.0
        assert px > 200.0


class TestTickRounding:
    """
    ``_place()`` must round to min tick before submission — code 110
    otherwise kills the order.

    Rounding is directional: it must never make a limit price closer to
    the touch or a stop price closer to its trigger. Convention:

    * BUY  LMT → floor (never higher than requested)
    * SELL LMT → ceil  (never lower than requested)
    * BUY  STP → ceil  (never lower than requested; triggers later)
    * SELL STP → floor (never higher than requested; triggers later)
    """

    def test_round_to_tick_defaults_to_nearest_without_side(self):
        # Legacy signature (no action/order_type) — nearest-tick.
        b, _ = _make_stubbed_broker()
        assert b._round_to_tick(150.037, 0.01) == 150.04
        assert b._round_to_tick(150.032, 0.01) == 150.03

    def test_round_to_tick_zero_min_tick_falls_back_to_two_decimals(self):
        b, _ = _make_stubbed_broker()
        # A zero (or negative) min tick should not divide-by-zero — it
        # falls back to a 2-decimal round.
        assert b._round_to_tick(150.037, 0.0) == 150.04

    def test_buy_lmt_rounds_down(self):
        b, _ = _make_stubbed_broker()
        # A BUY limit is never rounded UP toward the ask — floor it.
        assert b._round_to_tick(
            150.03, 0.05, action="BUY", order_type="LMT",
        ) == 150.00
        assert b._round_to_tick(
            150.037, 0.01, action="BUY", order_type="LMT",
        ) == 150.03
        # Already on a tick: no change.
        assert b._round_to_tick(
            150.00, 0.05, action="BUY", order_type="LMT",
        ) == 150.00

    def test_sell_lmt_rounds_up(self):
        b, _ = _make_stubbed_broker()
        # A SELL limit is never rounded DOWN toward the bid — ceil it.
        assert b._round_to_tick(
            150.03, 0.05, action="SELL", order_type="LMT",
        ) == 150.05
        assert b._round_to_tick(
            150.032, 0.01, action="SELL", order_type="LMT",
        ) == 150.04
        assert b._round_to_tick(
            150.00, 0.05, action="SELL", order_type="LMT",
        ) == 150.00

    def test_buy_stp_rounds_up(self):
        b, _ = _make_stubbed_broker()
        # A BUY stop sits above the market — rounding UP delays the
        # trigger, never fires it earlier than intended.
        assert b._round_to_tick(
            150.03, 0.05, action="BUY", order_type="STP",
        ) == 150.05
        assert b._round_to_tick(
            150.032, 0.01, action="BUY", order_type="STP",
        ) == 150.04

    def test_sell_stp_rounds_down(self):
        b, _ = _make_stubbed_broker()
        # A SELL stop sits below the market — rounding DOWN delays the
        # trigger, never fires it earlier than intended.
        assert b._round_to_tick(
            150.03, 0.05, action="SELL", order_type="STP",
        ) == 150.00
        assert b._round_to_tick(
            150.037, 0.01, action="SELL", order_type="STP",
        ) == 150.03

    def test_min_tick_falls_back_on_missing_details(self):
        b, _ = _make_stubbed_broker()
        # Missing / falsy details should return the $0.01 fallback,
        # not crash.
        b._ib.reqContractDetails.return_value = []
        assert b._min_tick(MagicMock()) == 0.01

    def test_min_tick_falls_back_on_exception(self):
        b, _ = _make_stubbed_broker()
        b._ib.reqContractDetails.side_effect = RuntimeError("boom")
        assert b._min_tick(MagicMock()) == 0.01

    def test_min_tick_cached_across_calls(self):
        """
        Repeated ``_min_tick`` calls for the same contract must hit the
        cache, not the network. Otherwise every LMT/STP placement burns
        a round-trip on reqContractDetails.
        """
        b, _ = _make_stubbed_broker()
        details = MagicMock()
        details.minTick = 0.05
        b._ib.reqContractDetails.return_value = [details]

        contract = MagicMock()
        contract.conId = 12345
        contract.symbol = "AAPL"

        assert b._min_tick(contract) == 0.05
        assert b._min_tick(contract) == 0.05
        assert b._min_tick(contract) == 0.05
        assert b._ib.reqContractDetails.call_count == 1

    def test_min_tick_cache_distinguishes_contracts(self):
        """Distinct contracts must not collide in the cache."""
        b, _ = _make_stubbed_broker()

        det_penny = MagicMock()
        det_penny.minTick = 0.01
        det_nickel = MagicMock()
        det_nickel.minTick = 0.05

        # Return different minTicks based on the first arg to
        # reqContractDetails, so the cache-vs-network behaviour is
        # observable.
        def _details_for(contract):
            return [det_penny if contract.conId == 1 else det_nickel]

        b._ib.reqContractDetails.side_effect = _details_for

        c1 = MagicMock()
        c1.conId = 1
        c2 = MagicMock()
        c2.conId = 2

        assert b._min_tick(c1) == 0.01
        assert b._min_tick(c2) == 0.05
        # A second lookup on either should be cache-served.
        assert b._min_tick(c1) == 0.01
        assert b._min_tick(c2) == 0.05
        assert b._ib.reqContractDetails.call_count == 2

    def test_min_tick_cache_survives_place_calls(self):
        """
        Two consecutive LMT placements on the same symbol must only
        trigger one reqContractDetails round-trip. Regression guard for
        the "network round-trip per placement" review finding.
        """
        b, _ = _make_stubbed_broker()
        details = MagicMock()
        details.minTick = 0.05
        b._ib.reqContractDetails.return_value = [details]

        # Both placements go through _place → _min_tick(contract). The
        # stubbed qualifyContracts always returns the same fake_contract,
        # so cache-key equality is guaranteed.
        b._place(
            "AAPL", 1, "BUY",
            order_type="LMT", limit_price=150.00, stop_price=None, tif="DAY",
        )
        b._place(
            "AAPL", 1, "SELL",
            order_type="LMT", limit_price=200.00, stop_price=None, tif="DAY",
        )
        assert b._ib.reqContractDetails.call_count == 1

    def test_place_buy_lmt_floors_price_before_submission(self):
        """BUY LMT with a nickel tick — 150.03 must FLOOR to 150.00."""
        b, captured = _make_stubbed_broker()
        details = MagicMock()
        details.minTick = 0.05
        b._ib.reqContractDetails.return_value = [details]

        b._place(
            "AAPL", 1, "BUY",
            order_type="LMT", limit_price=150.03, stop_price=None, tif="DAY",
        )
        _, order = captured[0]
        assert order.lmtPrice == 150.00, (
            f"BUY LMT must FLOOR to nickel tick (safer), got {order.lmtPrice}"
        )

    def test_place_sell_lmt_ceils_price_before_submission(self):
        """SELL LMT with a nickel tick — 150.03 must CEIL to 150.05."""
        b, captured = _make_stubbed_broker()
        details = MagicMock()
        details.minTick = 0.05
        b._ib.reqContractDetails.return_value = [details]

        b._place(
            "AAPL", 1, "SELL",
            order_type="LMT", limit_price=150.03, stop_price=None, tif="DAY",
        )
        _, order = captured[0]
        assert order.lmtPrice == 150.05, (
            f"SELL LMT must CEIL to nickel tick (safer), got {order.lmtPrice}"
        )

    def test_place_buy_stp_ceils_price_before_submission(self):
        """BUY STP with a nickel tick — 150.03 must CEIL to 150.05."""
        b, captured = _make_stubbed_broker()
        details = MagicMock()
        details.minTick = 0.05
        b._ib.reqContractDetails.return_value = [details]

        b._place(
            "AAPL", 1, "BUY",
            order_type="STP", limit_price=None, stop_price=150.03, tif="DAY",
        )
        _, order = captured[0]
        assert order.auxPrice == 150.05, (
            f"BUY STP must CEIL to nickel tick (safer), got {order.auxPrice}"
        )

    def test_place_sell_stp_floors_price_before_submission(self):
        """SELL STP with a nickel tick — 150.03 must FLOOR to 150.00."""
        b, captured = _make_stubbed_broker()
        details = MagicMock()
        details.minTick = 0.05
        b._ib.reqContractDetails.return_value = [details]

        b._place(
            "AAPL", 1, "SELL",
            order_type="STP", limit_price=None, stop_price=150.03, tif="DAY",
        )
        _, order = captured[0]
        assert order.auxPrice == 150.00, (
            f"SELL STP must FLOOR to nickel tick (safer), got {order.auxPrice}"
        )


class TestGetPositionsAggregation:
    """
    ``reqPositions()`` returns rows per account; get_positions() must sum
    per symbol instead of silently overwriting a prior account's row.
    """

    def _fake_pos(self, symbol: str, qty: float, cost: float, account: str = "DU1"):
        p = MagicMock(name=f"pos_{symbol}_{account}")
        p.contract.symbol = symbol
        p.position = qty
        p.avgCost = cost
        p.account = account
        return p

    def test_aggregates_same_symbol_across_accounts(self):
        b, _ = _make_stubbed_broker()
        b._ib.positions.return_value = [
            self._fake_pos("AAPL", 100, 150.0, account="DU1"),
            self._fake_pos("AAPL",  50, 200.0, account="DU2"),
        ]
        positions = b.get_positions()
        assert len(positions) == 1
        p = positions[0]
        assert p.symbol == "AAPL"
        assert p.quantity == 150
        # Quantity-weighted average cost: (100*150 + 50*200) / 150 = 166.66...
        assert abs(p.avg_cost - (100 * 150.0 + 50 * 200.0) / 150) < 1e-6

    def test_long_short_offset_across_accounts_nets(self):
        b, _ = _make_stubbed_broker()
        b._ib.positions.return_value = [
            self._fake_pos("MSFT", 100, 300.0, account="DU1"),
            self._fake_pos("MSFT", -30, 320.0, account="DU2"),
        ]
        positions = b.get_positions()
        assert len(positions) == 1
        assert positions[0].symbol == "MSFT"
        assert positions[0].quantity == 70   # signed sum

    def test_distinct_symbols_stay_distinct(self):
        b, _ = _make_stubbed_broker()
        b._ib.positions.return_value = [
            self._fake_pos("AAPL", 10, 150.0),
            self._fake_pos("MSFT", 20, 300.0),
        ]
        positions = b.get_positions()
        by_sym = {p.symbol: p for p in positions}
        assert by_sym["AAPL"].quantity == 10
        assert by_sym["MSFT"].quantity == 20


class TestConnectAccountVerification:
    """H2: paper mode must reject a live account even on a paper port."""

    def _fresh(self):
        b = _Broker()
        b._ib = MagicMock(name="fake_ib")
        # connect() itself is a no-op — we're only testing the post-
        # connect account-type check.
        b._ib.connect.return_value = None
        b._ib.sleep = MagicMock()
        return b

    def test_paper_mode_rejects_live_account(self):
        b = self._fresh()
        b._ib.managedAccounts.return_value = ["U1234567"]  # live prefix
        with pytest.raises(RuntimeError, match="LIVE account"):
            b.connect(mode="paper", port=7497)
        # And we must have disconnected on the way out.
        assert b._ib.disconnect.called

    def test_paper_mode_accepts_paper_account(self):
        b = self._fresh()
        b._ib.managedAccounts.return_value = ["DU1234567"]
        b.connect(mode="paper", port=7497)
        assert b.mode == "paper"

    def test_paper_mode_accepts_advisor_paper_prefix(self):
        b = self._fresh()
        b._ib.managedAccounts.return_value = ["DF999"]
        b.connect(mode="paper", port=7497)
        assert b.mode == "paper"

    def test_empty_managed_accounts_raises(self):
        b = self._fresh()
        b._ib.managedAccounts.return_value = []  # always empty
        with pytest.raises(RuntimeError, match="managedAccounts"):
            b.connect(mode="paper", port=7497)
        assert b._ib.disconnect.called

    def test_live_mode_warns_but_proceeds_on_paper_account(self, caplog):
        b = self._fresh()
        b._ib.managedAccounts.return_value = ["DU1234567"]
        import logging
        with caplog.at_level(logging.WARNING, logger="ibkr_broker.broker"):
            b.connect(
                mode="live",
                port=7496,
                live_confirm=_Broker.LIVE_CONFIRM_MSG,
            )
        assert b.mode == "live"
        assert any(
            "paper" in rec.getMessage().lower() for rec in caplog.records
        ), f"expected a paper-account warning, got {caplog.records}"


class TestWaitForFillTerminatesOnInactive:
    """M1: a rejected order (status=Inactive) must terminate wait_for_fill."""

    def test_rejected_order_returns_instead_of_timing_out(self):
        b, _ = _make_stubbed_broker()

        trade = MagicMock(name="rejected_trade")
        trade.contract.symbol = "AAPL"
        trade.order.orderId = 4242
        trade.order.action = "BUY"
        trade.order.totalQuantity = 100
        trade.order.orderType = "LMT"
        trade.order.lmtPrice = 100.0
        trade.order.auxPrice = 0.0
        trade.orderStatus.status = "Inactive"
        trade.orderStatus.filled = 0
        trade.orderStatus.avgFillPrice = 0.0
        # In ib_async 2.x, Trade.isDone() returns True for Inactive
        # because Inactive is in OrderStatus.DoneStates. Mirror that here.
        trade.isDone.return_value = True

        b._trades["4242"] = trade

        info = b.wait_for_fill("4242", timeout=1.0)
        assert info.status == "Inactive"
        assert info.filled == 0


class TestRebalanceSurfacesRejectedLegs:
    """
    A rebalance() with one leg TWS turns to ``"Inactive"`` (codes 110 /
    201) must:

    * still return every placed order_id (backward-compat list shape)
    * expose the rejected id via ``result.rejected``
    * emit an ERROR log naming the rejected symbol.
    """

    def _rebalance_with_leg_status(
        self,
        leg_statuses: dict[str, str],
    ) -> tuple:
        """
        Set up a stubbed broker whose placeOrder assigns increasing
        order IDs and whose per-symbol trade objects report the given
        ``orderStatus.status`` values. Returns (broker, captured_orders,
        result).

        ``leg_statuses`` maps symbol -> desired status string
        (e.g. ``{"AAPL": "Inactive", "MSFT": "Submitted"}``).
        """
        from ibkr_broker.broker import RebalanceResult  # local import

        b = _Broker()
        captured: list = []

        # One qualified contract per symbol.
        contracts: dict[str, MagicMock] = {}
        for sym in leg_statuses:
            c = MagicMock(name=f"contract_{sym}")
            c.symbol = sym
            c.conId = hash(sym) & 0xFFFF
            contracts[sym] = c

        def _qualify(contract):
            return [contracts[contract.symbol]]

        next_id = {"n": 5000}

        def _place_order(contract, order):
            oid = next_id["n"]
            next_id["n"] += 1
            trade = MagicMock(name=f"trade_{contract.symbol}_{oid}")
            trade.order = order
            trade.order.orderId = oid
            trade.contract = contract
            status = leg_statuses.get(contract.symbol, "Submitted")
            trade.orderStatus.status = status
            trade.orderStatus.filled = 0
            trade.orderStatus.avgFillPrice = 0.0
            trade.isDone.return_value = status in {
                "Filled", "Cancelled", "ApiCancelled", "Inactive",
            }
            captured.append((contract, order, trade))
            return trade

        b._ib = MagicMock(name="fake_ib")
        b._ib.isConnected.return_value = True
        b._ib.qualifyContracts.side_effect = _qualify
        b._ib.placeOrder.side_effect = _place_order
        # No pre-existing positions — every leg is a fresh buy.
        b._ib.positions.return_value = []
        b._ib.reqPositions.side_effect = lambda: b._ib.positions()
        # Bounded-wait sleep is a no-op in tests. Shrink the deadline
        # so a stable "Submitted" (never-Inactive) leg exits the poll
        # quickly instead of spinning 2s per test.
        b._ib.sleep = MagicMock()
        b._REBALANCE_REJECT_POLL_SECONDS = 0.05

        targets = {sym: 10 for sym in leg_statuses}
        result = b.rebalance(targets, close_unlisted=False)
        return b, captured, result, RebalanceResult

    def test_rejected_leg_appears_in_rejected_field(self, caplog):
        import logging
        with caplog.at_level(logging.ERROR, logger="ibkr_broker.broker"):
            b, captured, result, RebalanceResult = self._rebalance_with_leg_status(
                {"AAPL": "Inactive", "MSFT": "Submitted"}
            )

        assert isinstance(result, RebalanceResult)
        # Backward compat: still list-like with two order IDs
        assert len(result) == 2
        for oid in result:
            assert isinstance(oid, str)

        # Rejected list surfaces exactly the AAPL leg. Find the oid by
        # matching the captured order's contract.symbol.
        aapl_oid = next(
            str(trade.order.orderId)
            for contract, _, trade in captured
            if contract.symbol == "AAPL"
        )
        msft_oid = next(
            str(trade.order.orderId)
            for contract, _, trade in captured
            if contract.symbol == "MSFT"
        )
        assert result.rejected == [aapl_oid]
        assert msft_oid not in result.rejected

        # And a per-symbol ERROR log names AAPL (the rejected symbol),
        # not MSFT.
        rejection_logs = [
            rec for rec in caplog.records
            if rec.levelno == logging.ERROR and "rejected" in rec.getMessage()
        ]
        assert rejection_logs, "expected an ERROR log for the rejected leg"
        assert any("AAPL" in rec.getMessage() for rec in rejection_logs)
        assert not any("MSFT" in rec.getMessage() for rec in rejection_logs)

    def test_all_accepted_returns_empty_rejected(self):
        b, captured, result, RebalanceResult = self._rebalance_with_leg_status(
            {"AAPL": "Submitted", "MSFT": "Submitted"}
        )
        assert isinstance(result, RebalanceResult)
        assert len(result) == 2
        assert result.rejected == []

    def test_result_is_still_list_for_legacy_callers(self):
        """Callers doing ``oids = broker.rebalance(...)`` must not break."""
        b, captured, result, _ = self._rebalance_with_leg_status(
            {"AAPL": "Submitted"}
        )
        # Every legacy usage exercised on real code.
        assert isinstance(result, list)
        assert len(result) == 1
        first = result[0]
        assert isinstance(first, str)
        collected = [oid for oid in result]
        assert collected == [first]


# =========================================================================
# 4. LIMIT ORDERS (far from market, won't fill — safe anytime)
# =========================================================================

class TestLimitOrders:

    def test_place_buy_limit(self, broker, clean_orders):
        far_price = _far_from_market(broker, SYMBOL_A, "BUY")
        oid = broker.buy(SYMBOL_A, TEST_QTY, "LMT", limit_price=far_price)

        assert isinstance(oid, str) and len(oid) > 0
        broker.sleep(1)

        status = broker.get_order_status(oid)
        assert isinstance(status, OrderInfo)
        assert status.symbol == SYMBOL_A
        assert status.action == "BUY"
        assert status.quantity == TEST_QTY
        assert status.order_type == "LMT"

    def test_place_sell_limit(self, broker, clean_orders):
        far_price = _far_from_market(broker, SYMBOL_A, "SELL")
        oid = broker.sell(SYMBOL_A, TEST_QTY, "LMT", limit_price=far_price)

        broker.sleep(1)
        status = broker.get_order_status(oid)
        assert status.action == "SELL"

    def test_get_open_orders_shows_placed_orders(self, broker, clean_orders):
        oid1 = broker.buy(SYMBOL_A, TEST_QTY, "LMT",
                          limit_price=_far_from_market(broker, SYMBOL_A, "BUY"))
        oid2 = broker.sell(SYMBOL_B, TEST_QTY, "LMT",
                           limit_price=_far_from_market(broker, SYMBOL_B, "SELL"))
        broker.sleep(2)

        orders = broker.get_open_orders()
        open_ids = {o.order_id for o in orders}
        assert oid1 in open_ids, f"{oid1} missing from open orders"
        assert oid2 in open_ids, f"{oid2} missing from open orders"

    def test_cancel_single_order(self, broker, clean_orders):
        oid = broker.buy(SYMBOL_A, TEST_QTY, "LMT",
                         limit_price=_far_from_market(broker, SYMBOL_A, "BUY"))
        broker.sleep(1)

        broker.cancel(oid)
        broker.sleep(2)

        open_ids = {o.order_id for o in broker.get_open_orders()}
        assert oid not in open_ids

    def test_cancel_all_clears_everything(self, broker):
        broker.buy(SYMBOL_A, TEST_QTY, "LMT",
                   limit_price=_far_from_market(broker, SYMBOL_A, "BUY"))
        broker.buy(SYMBOL_B, TEST_QTY, "LMT",
                   limit_price=_far_from_market(broker, SYMBOL_B, "BUY"))
        broker.sleep(1)

        broker.cancel_all()
        broker.sleep(3)

        orders = broker.get_open_orders()
        assert len(orders) == 0, f"Expected 0 open orders, got {len(orders)}"

    def test_stop_order(self, broker, clean_orders):
        # Stop BUY way above market — won't trigger. The "SELL" side of
        # _far_from_market gives us the 3x-market price we want here.
        oid = broker.buy(SYMBOL_A, TEST_QTY, "STP",
                         stop_price=_far_from_market(broker, SYMBOL_A, "SELL"))
        broker.sleep(1)

        status = broker.get_order_status(oid)
        assert status.order_type == "STP"

    def test_show_orders_with_data(self, broker, capsys, clean_orders):
        broker.buy(SYMBOL_A, TEST_QTY, "LMT",
                   limit_price=_far_from_market(broker, SYMBOL_A, "BUY"))
        broker.sleep(1)

        broker.show_orders()
        output = capsys.readouterr().out
        assert SYMBOL_A in output

    def test_wait_for_fill_times_out_on_limit(self, broker, clean_orders):
        oid = broker.buy(SYMBOL_A, TEST_QTY, "LMT",
                         limit_price=_far_from_market(broker, SYMBOL_A, "BUY"))
        broker.sleep(1)

        with pytest.raises(TimeoutError, match="not filled"):
            broker.wait_for_fill(oid, timeout=3)

    def test_place_moc_order_via_buy_moc(self, broker, clean_orders):
        # Placing a MOC order needs to happen inside the MOC submission
        # window (US equities: before ~15:50 ET). If the market is
        # closed OR we're past the cutoff, TWS rejects the order and it
        # shows up as Cancelled/Inactive rather than Submitted — so we
        # skip in that case rather than fail the whole build.
        if not _is_us_market_open():
            pytest.skip("US equity market is closed — MOC submission window closed")

        oid = broker.buy_moc(SYMBOL_A, TEST_QTY)
        assert isinstance(oid, str) and len(oid) > 0
        broker.sleep(2)

        status = broker.get_order_status(oid)
        # Whatever state TWS puts it in, order_type must round-trip as MOC
        # and neither a limit nor stop price should be attached.
        assert status.order_type == "MOC"
        assert status.limit_price is None
        assert status.stop_price is None
        assert status.symbol == SYMBOL_A
        assert status.action == "BUY"
        assert status.quantity == TEST_QTY


# =========================================================================
# 5. MARKET ORDERS — fills, positions, close (needs market open)
# =========================================================================

@market_hours
class TestMarketOrders:
    """
    Place real MARKET orders that fill immediately.
    Uses 1 share at a time to minimize cost.
    """

    @pytest.fixture(autouse=True)
    def _skip_if_closed(self):
        if not _is_us_market_open():
            pytest.skip("US equity market is closed")

    def test_buy_market_fills(self, broker, clean_positions):
        oid = broker.buy(SYMBOL_A, TEST_QTY)
        info = broker.wait_for_fill(oid, timeout=15)

        assert info.status == "Filled"
        assert info.filled == TEST_QTY
        assert info.avg_fill_price > 0

    def test_sell_market_fills(self, broker, clean_positions):
        buy_oid = broker.buy(SYMBOL_A, TEST_QTY)
        broker.wait_for_fill(buy_oid, timeout=15)

        sell_oid = broker.sell(SYMBOL_A, TEST_QTY)
        info = broker.wait_for_fill(sell_oid, timeout=15)
        assert info.status == "Filled"

    def test_position_appears_after_buy(self, broker, clean_positions):
        oid = broker.buy(SYMBOL_A, TEST_QTY)
        broker.wait_for_fill(oid, timeout=15)

        pos = broker.get_position(SYMBOL_A)
        assert pos is not None
        assert pos.quantity >= TEST_QTY
        assert pos.symbol == SYMBOL_A
        assert pos.avg_cost > 0

    def test_is_flat_after_round_trip(self, broker, clean_positions):
        oid1 = broker.buy(SYMBOL_A, TEST_QTY)
        broker.wait_for_fill(oid1, timeout=15)
        assert broker.is_flat(SYMBOL_A) is False

        oid2 = broker.sell(SYMBOL_A, TEST_QTY)
        broker.wait_for_fill(oid2, timeout=15)
        assert broker.is_flat(SYMBOL_A) is True

    def test_close_position_long(self, broker, clean_positions):
        oid = broker.buy(SYMBOL_A, TEST_QTY)
        broker.wait_for_fill(oid, timeout=15)

        close_oid = broker.close_position(SYMBOL_A)
        assert close_oid is not None
        broker.wait_for_fill(close_oid, timeout=15)
        assert broker.is_flat(SYMBOL_A) is True

    def test_close_position_returns_none_when_flat(self, broker, clean_positions):
        assert broker.close_position(SYMBOL_A) is None

    def test_flatten_all_closes_everything(self, broker, clean_positions):
        oid1 = broker.buy(SYMBOL_A, TEST_QTY)
        oid2 = broker.buy(SYMBOL_B, TEST_QTY)
        broker.wait_for_fill(oid1, timeout=15)
        broker.wait_for_fill(oid2, timeout=15)

        oids = broker.flatten_all()
        assert len(oids) >= 2
        for oid in oids:
            broker.wait_for_fill(oid, timeout=15)

        assert broker.is_flat() is True

    def test_show_positions_after_buy(self, broker, capsys, clean_positions):
        oid = broker.buy(SYMBOL_A, TEST_QTY)
        broker.wait_for_fill(oid, timeout=15)

        broker.show_positions()
        output = capsys.readouterr().out
        assert SYMBOL_A in output
        assert "LONG" in output


# =========================================================================
# 6. SHORT / COVER (needs market open)
# =========================================================================

@market_hours
class TestShortCover:

    @pytest.fixture(autouse=True)
    def _skip_if_closed(self):
        if not _is_us_market_open():
            pytest.skip("US equity market is closed")

    def test_short_creates_negative_position(self, broker, clean_positions):
        oid = broker.short(SYMBOL_A, TEST_QTY)
        broker.wait_for_fill(oid, timeout=15)

        pos = broker.get_position(SYMBOL_A)
        assert pos is not None
        assert pos.quantity < 0

    def test_cover_after_short_goes_flat(self, broker, clean_positions):
        oid = broker.short(SYMBOL_A, TEST_QTY)
        broker.wait_for_fill(oid, timeout=15)

        oid2 = broker.cover(SYMBOL_A, TEST_QTY)
        broker.wait_for_fill(oid2, timeout=15)
        assert broker.is_flat(SYMBOL_A) is True

    def test_short_blocked_when_long(self, broker, clean_positions):
        oid = broker.buy(SYMBOL_A, TEST_QTY)
        broker.wait_for_fill(oid, timeout=15)

        with pytest.raises(RuntimeError, match="LONG position"):
            broker.short(SYMBOL_A, TEST_QTY)

    def test_cover_blocked_when_flat(self, broker, clean_positions):
        with pytest.raises(RuntimeError, match="No short position"):
            broker.cover(SYMBOL_A, TEST_QTY)

    def test_cover_blocked_when_long(self, broker, clean_positions):
        oid = broker.buy(SYMBOL_A, TEST_QTY)
        broker.wait_for_fill(oid, timeout=15)

        with pytest.raises(RuntimeError, match="No short position"):
            broker.cover(SYMBOL_A, TEST_QTY)

    def test_close_position_covers_short(self, broker, clean_positions):
        oid = broker.short(SYMBOL_A, TEST_QTY)
        broker.wait_for_fill(oid, timeout=15)

        close_oid = broker.close_position(SYMBOL_A)
        assert close_oid is not None
        broker.wait_for_fill(close_oid, timeout=15)
        assert broker.is_flat(SYMBOL_A) is True


# =========================================================================
# 7. REBALANCE (needs market open)
# =========================================================================

@market_hours
class TestRebalance:

    @pytest.fixture(autouse=True)
    def _skip_if_closed(self):
        if not _is_us_market_open():
            pytest.skip("US equity market is closed")

    def test_rebalance_from_flat(self, broker, clean_positions):
        oids = broker.rebalance({SYMBOL_A: TEST_QTY})
        assert len(oids) == 1
        broker.wait_for_fill(oids[0], timeout=15)

        pos = broker.get_position(SYMBOL_A)
        assert pos is not None
        assert pos.quantity == TEST_QTY

    def test_rebalance_no_op_at_target(self, broker, clean_positions):
        oid = broker.buy(SYMBOL_A, TEST_QTY)
        broker.wait_for_fill(oid, timeout=15)

        oids = broker.rebalance({SYMBOL_A: TEST_QTY}, close_unlisted=False)
        assert len(oids) == 0

    def test_rebalance_closes_unlisted(self, broker, clean_positions):
        oid = broker.buy(SYMBOL_A, TEST_QTY)
        broker.wait_for_fill(oid, timeout=15)

        # Target only SYMBOL_B → SYMBOL_A should be closed
        oids = broker.rebalance({SYMBOL_B: TEST_QTY}, close_unlisted=True)
        assert len(oids) >= 2  # buy B + sell A
        for o in oids:
            broker.wait_for_fill(o, timeout=15)

        assert broker.is_flat(SYMBOL_A) is True
        assert broker.get_position(SYMBOL_B).quantity == TEST_QTY


# =========================================================================
# 8. MARKET DATA
# =========================================================================

class TestMarketData:

    def test_get_price(self, broker):
        price = broker.get_price(SYMBOL_A)
        if price is not None:
            assert isinstance(price, float)
            assert price > 0
        else:
            pytest.skip(
                "get_price returned None — may need market data subscription"
            )


# =========================================================================
# 9. ERROR CALLBACK
# =========================================================================

class TestErrorCallback:

    def test_on_error_registers(self, broker):
        def handler(reqId, errorCode, errorString, contract):
            pass

        # Should not crash
        broker.on_error(handler)
        broker.sleep(1)


# =========================================================================
# 10. CONTEXT MANAGER
# =========================================================================

class TestContextManager:

    def test_fresh_broker_context_manager(self):
        _assert_paper_only()

        b = Broker()
        try:
            b.connect(
                mode="paper",
                host=TWS_HOST,
                port=TWS_PORT,
                client_id=TWS_CLIENT_ID + 1,
            )
        except Exception as e:
            pytest.skip(
                f"Could not connect to TWS at {TWS_HOST}:{TWS_PORT} — "
                f"is paper TWS running with API enabled? Error: {e}"
            )

        # Now use the actual context manager — that's what the class name
        # advertises, and __exit__ is what we're supposed to be testing.
        with b:
            assert b.is_connected()
        assert not b.is_connected()
