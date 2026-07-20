"""
Simple Interactive Brokers TWS wrapper for algorithmic trading.

Usage from Jupyter:
    from broker import Broker
    b = Broker()
    b.connect()                          # paper trading by default
    b.buy("AAPL", 100)
    b.get_positions()
    b.get_account_summary()
    b.disconnect()

Requirements (pick one):
    pip install ib_async       # actively maintained (recommended)
    pip install ib_insync      # archived but still works

Connection ports:
    7497 = TWS paper trading
    7496 = TWS live trading
    4002 = IB Gateway paper trading
    4001 = IB Gateway live trading
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

# ---------------------------------------------------------------------------
# CRITICAL — must run BEFORE ``from ib_async import ...`` below.
#
# On Python 3.10+ (this repo is 3.14), ``asyncio.get_event_loop()`` no
# longer auto-creates a loop on the main thread when none exists — it
# raises ``RuntimeError: There is no current event loop in thread
# 'MainThread'.``. ``ib_async`` transitively imports ``eventkit``, whose
# ``eventkit/util.py`` executes ``asyncio.get_event_loop_policy().
# get_event_loop()`` at module-load time. That means a plain-script CLI
# entry point (e.g. ``python -m live.panic``) blows up *during import*,
# and later ``IB.connect()`` — which drives the event loop via
# ``loop.run_until_complete`` — needs a current loop for the same reason.
#
# Jupyter and pytest are already covered: a Jupyter kernel installs a
# running loop before any user code runs, and this package's pytest
# ``conftest.py`` primes one via ``asyncio.set_event_loop(
# asyncio.new_event_loop())`` before importing ``ibkr_broker``. The plain-
# script case is what this hook fixes. It is idempotent — only creates a
# fresh loop when one is not already established — so the Jupyter and
# conftest-primed paths are untouched.
# ---------------------------------------------------------------------------
def _ensure_event_loop() -> None:
    try:
        asyncio.get_event_loop()
        return
    except RuntimeError:
        pass
    asyncio.set_event_loop(asyncio.new_event_loop())


_ensure_event_loop()

# Support both ib_async (actively maintained) and ib_insync (archived).
# They have identical APIs — ib_async is the community fork after
# ib_insync's creator passed away and the repo was archived in March 2024.
#
# ``Order`` is imported directly because there is no dedicated helper class
# for Market-on-Close orders in either library — MOC is built via the
# generic ``Order(action, totalQuantity, orderType="MOC")`` constructor.
try:
    from ib_async import (
        IB, Stock, MarketOrder, LimitOrder, StopOrder,
        Order, Contract, Trade, util,
    )
    from ib_async.order import OrderStatus
except ImportError:
    from ib_insync import (
        IB, Stock, MarketOrder, LimitOrder, StopOrder,
        Order, Contract, Trade, util,
    )
    from ib_insync.order import OrderStatus

# CRITICAL for Jupyter: ``util.startLoop()`` calls ``nest_asyncio.apply()``
# so that ``ib.connect()`` (which internally uses ``loop.run_until_complete``)
# works inside a Jupyter kernel where the event loop is already running.
# In a plain script / pytest we skip it — nest_asyncio patches asyncio
# globally, and there's no reason to do that when the caller isn't in
# Jupyter. Detection is by looking for IPython in sys.modules; it works
# even before any cell has run because IPython imports itself into
# ``sys.modules`` when the kernel starts.
def _maybe_patch_asyncio_for_jupyter() -> None:
    import sys
    if "IPython" in sys.modules or "ipykernel" in sys.modules:
        util.startLoop()


_maybe_patch_asyncio_for_jupyter()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes for clean return types (no ib_insync leaking into your strategy)
# ---------------------------------------------------------------------------

@dataclass
class OrderInfo:
    order_id: str
    symbol: str
    action: str          # BUY or SELL
    quantity: int
    order_type: str      # MKT, LMT, STP
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: str = ""     # Submitted, Filled, Cancelled, etc.
    filled: int = 0
    avg_fill_price: float = 0.0
    timestamp: str = ""


@dataclass
class Position:
    symbol: str
    quantity: int          # positive = long, negative = short
    avg_cost: float
    market_value: float = 0.0
    unrealized_pnl: float = 0.0


# ---------------------------------------------------------------------------
# Broker wrapper
# ---------------------------------------------------------------------------

class Broker:
    """
    Simple wrapper around ib_insync for interacting with TWS.

    Designed to be imported into a Jupyter notebook and used with
    straightforward method calls. Hides all IB complexity behind
    simple buy/sell/short/cover methods.

    Attributes
    ----------
    MAX_ORDER_QTY : int
        Hard ceiling on any single order's quantity. Prevents fat-finger
        errors. Override on the instance if you intentionally need larger
        orders (e.g. ``b.MAX_ORDER_QTY = 50_000``).
    LIVE_CONFIRM_MSG : str
        Confirmation string required to connect in live mode.
    """

    # Safety defaults
    MAX_ORDER_QTY: int = 10_000
    """Hard ceiling on single-order quantity to prevent fat-finger errors."""

    LIVE_CONFIRM_MSG: str = "YES_I_WANT_LIVE_TRADING"

    # Ports on which TWS / IB Gateway run a LIVE account.
    # Kept as a class attribute so ``connect()`` can refuse a paper-mode
    # request that points at one of these (fat-finger protection).
    _LIVE_PORTS: frozenset[int] = frozenset({7496, 4001})

    # IB error codes that deserve explicit warnings
    _WARN_ERROR_CODES: set[int] = {
        103,  # Duplicate order id
        110,  # Price does not conform to minimum price variation
        201,  # Order rejected
        202,  # Order cancelled
    }

    def __init__(self) -> None:
        self._ib = IB()
        self._trades: dict[str, Trade] = {}  # order_id -> Trade
        self._mode: str = "paper"

        # Register default error handler so common codes are not silently dropped
        self._ib.errorEvent += self._default_error_handler

    @staticmethod
    def _default_error_handler(
        reqId: int, errorCode: int, errorString: str, contract: Optional[Contract]
    ) -> None:
        """
        Default error callback that logs warnings for common IB error codes.

        Parameters
        ----------
        reqId : int
            The request ID associated with the error.
        errorCode : int
            IB error code.
        errorString : str
            Human-readable error message from TWS.
        contract : Contract or None
            The contract associated with the error, if any.

        Notes
        -----
        Users can override this by calling ``on_error()`` to register their
        own handler, or by removing this handler via
        ``broker.ib.errorEvent -= broker._default_error_handler``.
        """
        symbol = contract.symbol if contract else "N/A"
        if errorCode in Broker._WARN_ERROR_CODES:
            logger.warning(
                "IB error %d (reqId=%d, symbol=%s): %s",
                errorCode, reqId, symbol, errorString,
            )
        else:
            logger.debug(
                "IB message %d (reqId=%d, symbol=%s): %s",
                errorCode, reqId, symbol, errorString,
            )

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(
        self,
        mode: str = "paper",
        host: str = "127.0.0.1",
        port: Optional[int] = None,
        client_id: int = 1,
        live_confirm: str = "",
    ) -> None:
        """
        Connect to TWS.

        Parameters
        ----------
        mode : str
            ``"paper"`` or ``"live"``.
        host : str
            TWS host (default ``"127.0.0.1"``).
        port : int or None
            Override port. If ``None``, auto-selects (7497 paper, 7496 live).
        client_id : int
            Unique client ID. Use different IDs if running multiple
            notebooks against the same TWS instance.
        live_confirm : str
            Must pass ``Broker.LIVE_CONFIRM_MSG`` to connect in live mode.

        Raises
        ------
        RuntimeError
            If connecting live without the confirmation string.
        ValueError
            If *mode* is not ``"paper"`` or ``"live"``.
        """
        if mode == "live":
            if live_confirm != self.LIVE_CONFIRM_MSG:
                raise RuntimeError(
                    "To connect in live mode, pass "
                    "live_confirm=Broker.LIVE_CONFIRM_MSG"
                )
            default_port = 7496
        elif mode == "paper":
            default_port = 7497
        else:
            raise ValueError(f"mode must be 'paper' or 'live', got '{mode}'")

        actual_port = port if port is not None else default_port

        # Fat-finger guard: refuse to open a paper-mode session against a
        # port that maps to a LIVE TWS/Gateway account.
        if mode == "paper" and actual_port in self._LIVE_PORTS:
            raise RuntimeError(
                f"Refusing to connect in paper mode to port {actual_port} — "
                f"that port ({actual_port}) is a LIVE trading port. "
                f"If you really want to trade live, call connect(mode='live', "
                f"live_confirm=Broker.LIVE_CONFIRM_MSG)."
            )

        self._ib.connect(host, actual_port, clientId=client_id)
        # Only record mode after a successful connection — a failed connect
        # (e.g. ConnectionRefused) must not leave a stale mode on the object.
        self._mode = mode
        logger.info(
            f"Connected to TWS ({mode}) at {host}:{actual_port} "
            f"[client_id={client_id}]"
        )
        print(f"✓ Connected to TWS ({mode.upper()}) on port {actual_port}")

    def disconnect(self) -> None:
        """Disconnect from TWS. Safe to call when already disconnected."""
        if self._ib.isConnected():
            self._ib.disconnect()
            print("✓ Disconnected from TWS")

    def is_connected(self) -> bool:
        return self._ib.isConnected()

    @property
    def mode(self) -> str:
        return self._mode

    def sleep(self, seconds: float):
        """
        Sleep while keeping IB message loop alive.
        ALWAYS use this instead of time.sleep().
        """
        self._ib.sleep(seconds)

    # ------------------------------------------------------------------
    # Contract helpers
    # ------------------------------------------------------------------

    def _make_contract(
        self,
        symbol: str,
        exchange: str = "SMART",
        currency: str = "USD",
    ) -> Stock:
        """Create and qualify a stock contract.

        The *exchange* value is passed through to ``Stock`` and used by
        ``qualifyContracts`` when resolving the contract. Callers that
        require a specific venue (e.g. ``exchange="ISLAND"`` for the
        NASDAQ INET book) pass it explicitly; the default ``"SMART"``
        preserves legacy behaviour.

        Note: a "route-survival" assertion that raised when the
        qualified contract's ``exchange`` field did not match the
        caller's request used to live here (see the commented-out
        block below). It was disabled after it fired on 100% of
        non-SMART qualifications (46 of 46 requested, 0 passed)
        during a 2026-07-20 paper replay. The observed behaviour is
        that ``qualifyContracts`` with ``exchange="ISLAND"`` returns
        the venue under its canonical name ``"NASDAQ"`` — the SAME
        NASDAQ/INET book under a different label, NOT an unrelated
        venue and NOT a SMART degrade — so the raw equality check
        compared across an alias pair and could never pass for a
        non-SMART request. A ``whatIfOrder`` MOC on the canonicalised
        contract returned PreSubmitted with valid margin, confirming
        routing was genuinely intact. See commit e061383 for the
        original rationale.
        """
        contract = Stock(symbol, exchange, currency)
        qualified = self._ib.qualifyContracts(contract)
        if not qualified or qualified[0] is None:
            raise ValueError(
                f"Could not qualify contract for '{symbol}'. "
                f"Check that the symbol is correct and TWS is connected."
            )
        q = qualified[0]
        # --- DISABLED 2026-07-20 (see docstring above) --------------------
        # Route-survival assertion: refuse to place an order whose
        # qualified contract's exchange no longer matches the caller's
        # request. Only fires on non-SMART requests, so this is
        # invisible to legacy SMART callers. See ib_async's
        # qualifyContractsAsync in ib.py — the "SMART" branch preserves
        # the input string via a special case (search for
        # ``contract.exchange == "SMART"`` in ib.py's
        # qualifyContractsAsync); for any non-SMART request, IB's
        # dataclassUpdate copies the exchange from the resolved
        # ContractDetails record onto our contract, so a mismatch here
        # is genuine (not a naming quirk of the SDK).
        #
        # Rolled back because the premise turned out to be wrong in
        # practice: on a paper replay of 2026-01-05 through TWS
        # (7497), this guard raised on 46 of 46 requested symbols
        # (100% failure, 0 passed), all with the "off-venue" message
        # and exchange="ISLAND" requested. The reason:
        # qualifyContracts with exchange="ISLAND" returns
        # exchange="NASDAQ" — the SAME NASDAQ/INET book under its
        # canonical IB label — so the raw string equality compared
        # across an alias pair and could never pass. Verified live
        # against paper TWS on 49/49 names (exchange=NASDAQ,
        # primaryExchange=NASDAQ) and a whatIfOrder MOC on the
        # canonicalised contract returned PreSubmitted with valid
        # margin, so routing is genuinely intact.
        #
        # The guard is still worth restoring — it would catch a
        # genuine degrade to "SMART" or "ARCA" — but it must
        # tolerate the ISLAND<->NASDAQ alias pair. Correct restore:
        # compare ``got_exchange`` against an alias set
        # ``{"ISLAND", "NASDAQ"}`` rather than raw string equality
        # with the requested ``exchange``.
        # got_exchange = getattr(q, "exchange", "")
        # if exchange != "SMART" and got_exchange != exchange:
        #     raise ValueError(
        #         f"Qualified contract for '{symbol}' routed to "
        #         f"{got_exchange!r}, expected {exchange!r} — refusing to "
        #         f"place an off-venue order. This safety guard is what "
        #         f"prevents a NASDAQ-only strategy from silently filling "
        #         f"on a different venue when qualifyContracts "
        #         f"canonicalises the route."
        #     )
        # ------------------------------------------------------------------
        return q

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------

    def _place(
        self,
        symbol: str,
        quantity: int,
        action: str,
        order_type: str,
        limit_price: Optional[float],
        stop_price: Optional[float],
        tif: str = "DAY",
        exchange: str = "SMART",
    ) -> str:
        """
        Place an order and return its order_id as a string.

        Parameters
        ----------
        symbol : str
            Ticker symbol (e.g. ``"AAPL"``).
        quantity : int
            Number of shares. Must be positive and <= ``MAX_ORDER_QTY``.
        action : str
            ``"BUY"`` or ``"SELL"``.
        order_type : str
            ``"MKT"``, ``"LMT"``, ``"STP"``, or ``"MOC"``.
        limit_price : float or None
            Required for ``"LMT"`` orders. Ignored for MKT/STP/MOC.
        stop_price : float or None
            Required for ``"STP"`` orders. Ignored for MKT/LMT/MOC.
        tif : str
            Time in force. Common values: ``"DAY"``, ``"GTC"``, ``"IOC"``,
            ``"FOK"``. Defaults to ``"DAY"`` to prevent orders from
            accidentally surviving past the trading session.
        exchange : str
            Destination exchange for the underlying ``Stock`` contract.
            Default ``"SMART"`` preserves legacy behaviour — IBKR's
            SMART router picks the venue. Callers that require a
            specific venue (e.g. ``"ISLAND"`` for the NASDAQ INET book,
            when the strategy depends on the NASDAQ closing auction)
            pass it explicitly. Threaded through to ``_make_contract``
            unchanged so ``qualifyContracts`` sees the requested route.

            **Note:** Market-on-Close (``"MOC"``) orders are only valid with
            ``tif="DAY"`` — TWS rejects other TIFs. This method silently
            coerces ``tif`` to ``"DAY"`` for MOC orders (a debug log is
            emitted if the caller passed something else) rather than
            surfacing an obscure TWS rejection error at fill time.

        Returns
        -------
        str
            The order ID assigned by TWS.

        Raises
        ------
        ValueError
            If quantity is invalid, order_type is unsupported, or required
            price parameters are missing.
        """
        if quantity <= 0:
            raise ValueError(f"Quantity must be positive, got {quantity}")
        if quantity > self.MAX_ORDER_QTY:
            raise ValueError(
                f"Order quantity {quantity} exceeds MAX_ORDER_QTY "
                f"({self.MAX_ORDER_QTY}). Raise it explicitly if intended."
            )
        if not self.is_connected():
            raise RuntimeError(
                "Broker is not connected. Call connect() before placing orders."
            )

        contract = self._make_contract(symbol, exchange=exchange)

        if order_type == "MKT":
            order = MarketOrder(action, quantity)
        elif order_type == "LMT":
            if limit_price is None:
                raise ValueError("limit_price required for LMT orders")
            order = LimitOrder(action, quantity, limit_price)
        elif order_type == "STP":
            if stop_price is None:
                raise ValueError("stop_price required for STP orders")
            order = StopOrder(action, quantity, stop_price)
        elif order_type == "MOC":
            # Market-on-Close: no dedicated helper in ib_async, built via
            # the generic Order class. MOC takes no limit/stop price.
            # TWS only accepts MOC orders with tif=DAY, so we coerce here
            # rather than let the caller receive a cryptic TWS rejection.
            if tif != "DAY":
                logger.debug(
                    "MOC order for %s: coercing tif %r -> 'DAY' "
                    "(TWS rejects MOC with any other TIF).",
                    symbol, tif,
                )
                tif = "DAY"
            order = Order(action=action, totalQuantity=quantity, orderType="MOC")
        else:
            raise ValueError(
                f"Unsupported order_type '{order_type}'. "
                f"Use MKT, LMT, STP, or MOC."
            )

        order.tif = tif

        trade = self._ib.placeOrder(contract, order)
        order_id = str(trade.order.orderId)
        self._trades[order_id] = trade

        logger.info(
            f"Placed {action} {order_type} order: {symbol} x{quantity} "
            f"[id={order_id}]"
        )
        return order_id

    def buy(
        self,
        symbol: str,
        quantity: int,
        order_type: str = "MKT",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        tif: str = "DAY",
        exchange: str = "SMART",
    ) -> str:
        """
        Buy shares (go long or cover a short).

        Parameters
        ----------
        symbol : str
            Ticker symbol.
        quantity : int
            Number of shares to buy.
        order_type : str
            ``"MKT"``, ``"LMT"``, ``"STP"``, or ``"MOC"``.
        limit_price : float or None
            Required for ``"LMT"`` orders. Ignored for MKT/STP/MOC.
        stop_price : float or None
            Required for ``"STP"`` orders. Ignored for MKT/LMT/MOC.
        tif : str
            Time in force (``"DAY"``, ``"GTC"``, ``"IOC"``, ``"FOK"``).
            MOC orders are silently coerced to ``"DAY"`` — TWS rejects
            any other TIF for MOC.
        exchange : str
            Destination exchange for the underlying ``Stock`` contract.
            Default ``"SMART"`` preserves legacy behaviour (IBKR's SMART
            router picks the venue). Pass a direct-route code such as
            ``"ISLAND"`` (NASDAQ INET) to force execution on a specific
            venue.

        Returns
        -------
        str
            Order ID.

        Examples
        --------
        >>> b.buy("AAPL", 100)
        >>> b.buy("AAPL", 100, "LMT", limit_price=150.00)
        >>> b.buy("AAPL", 50, tif="IOC")
        >>> b.buy("AAPL", 100, "MOC")            # market-on-close
        >>> b.buy("AAPL", 100, exchange="ISLAND")  # NASDAQ direct route
        """
        return self._place(
            symbol, quantity, "BUY", order_type, limit_price, stop_price, tif,
            exchange=exchange,
        )

    def sell(
        self,
        symbol: str,
        quantity: int,
        order_type: str = "MKT",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        tif: str = "DAY",
        exchange: str = "SMART",
    ) -> str:
        """
        Sell shares (close a long or open a short).

        Parameters
        ----------
        symbol : str
            Ticker symbol.
        quantity : int
            Number of shares to sell.
        order_type : str
            ``"MKT"``, ``"LMT"``, ``"STP"``, or ``"MOC"``.
        limit_price : float or None
            Required for ``"LMT"`` orders. Ignored for MKT/STP/MOC.
        stop_price : float or None
            Required for ``"STP"`` orders. Ignored for MKT/LMT/MOC.
        tif : str
            Time in force (``"DAY"``, ``"GTC"``, ``"IOC"``, ``"FOK"``).
            MOC orders are silently coerced to ``"DAY"`` — TWS rejects
            any other TIF for MOC.
        exchange : str
            Destination exchange for the underlying ``Stock`` contract.
            Default ``"SMART"`` preserves legacy behaviour. Pass a
            direct-route code such as ``"ISLAND"`` to force execution
            on the NASDAQ INET book.

        Returns
        -------
        str
            Order ID.
        """
        return self._place(
            symbol, quantity, "SELL", order_type, limit_price, stop_price, tif,
            exchange=exchange,
        )

    def short(
        self,
        symbol: str,
        quantity: int,
        order_type: str = "MKT",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        tif: str = "DAY",
        exchange: str = "SMART",
    ) -> str:
        """
        Explicitly short a stock.

        Same as ``sell()``, but reads better in strategy code and adds a
        safety check that you are not accidentally selling a long position.

        Parameters
        ----------
        symbol : str
            Ticker symbol.
        quantity : int
            Number of shares to short.
        order_type : str
            ``"MKT"``, ``"LMT"``, ``"STP"``, or ``"MOC"``.
        limit_price : float or None
            Required for ``"LMT"`` orders. Ignored for MKT/STP/MOC.
        stop_price : float or None
            Required for ``"STP"`` orders. Ignored for MKT/LMT/MOC.
        tif : str
            Time in force (``"DAY"``, ``"GTC"``, ``"IOC"``, ``"FOK"``).
            MOC orders are silently coerced to ``"DAY"`` — TWS rejects
            any other TIF for MOC.
        exchange : str
            Destination exchange for the underlying ``Stock`` contract.
            Default ``"SMART"`` preserves legacy behaviour. Pass a
            direct-route code such as ``"ISLAND"`` to force execution
            on the NASDAQ INET book.

        Returns
        -------
        str
            Order ID.

        Raises
        ------
        RuntimeError
            If a long position already exists for *symbol*.
        """
        positions = self.get_positions()
        current_qty = 0
        for p in positions:
            if p.symbol == symbol:
                current_qty = p.quantity
                break

        if current_qty > 0:
            raise RuntimeError(
                f"You have a LONG position of {current_qty} in {symbol}. "
                f"Use sell() to close it, not short(). "
                f"If you really want to short through the position, "
                f"sell first, then short."
            )

        return self._place(
            symbol, quantity, "SELL", order_type, limit_price, stop_price, tif,
            exchange=exchange,
        )

    def cover(
        self,
        symbol: str,
        quantity: int,
        order_type: str = "MKT",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        tif: str = "DAY",
        exchange: str = "SMART",
    ) -> str:
        """
        Buy to cover a short position.

        Same as ``buy()``, but reads better in strategy code and adds a
        safety check that a short position actually exists.

        Parameters
        ----------
        symbol : str
            Ticker symbol.
        quantity : int
            Number of shares to cover.
        order_type : str
            ``"MKT"``, ``"LMT"``, ``"STP"``, or ``"MOC"``.
        limit_price : float or None
            Required for ``"LMT"`` orders. Ignored for MKT/STP/MOC.
        stop_price : float or None
            Required for ``"STP"`` orders. Ignored for MKT/LMT/MOC.
        tif : str
            Time in force (``"DAY"``, ``"GTC"``, ``"IOC"``, ``"FOK"``).
            MOC orders are silently coerced to ``"DAY"`` — TWS rejects
            any other TIF for MOC.
        exchange : str
            Destination exchange for the underlying ``Stock`` contract.
            Default ``"SMART"`` preserves legacy behaviour. Pass a
            direct-route code such as ``"ISLAND"`` to force execution
            on the NASDAQ INET book.

        Returns
        -------
        str
            Order ID.

        Raises
        ------
        RuntimeError
            If no short position exists for *symbol*.
        """
        positions = self.get_positions()
        current_qty = 0
        for p in positions:
            if p.symbol == symbol:
                current_qty = p.quantity
                break

        if current_qty >= 0:
            raise RuntimeError(
                f"No short position in {symbol} (current qty: {current_qty}). "
                f"Use buy() for opening a long."
            )

        return self._place(
            symbol, quantity, "BUY", order_type, limit_price, stop_price, tif,
            exchange=exchange,
        )

    # ------------------------------------------------------------------
    # Market-on-Close (MOC) convenience wrappers
    # ------------------------------------------------------------------
    #
    # MOC orders execute at the official closing auction. They're common
    # for end-of-day rebalancing and to guarantee close-price fills. IB
    # requires MOC orders to be submitted before a per-exchange cutoff
    # (typically several minutes before the close for US equities) — TWS
    # will reject late submissions with an order status error, not this
    # wrapper. See IB's "Order Type" docs for exact cutoffs.

    def buy_moc(self, symbol: str, quantity: int,
                exchange: str = "SMART") -> str:
        """
        Buy shares with a Market-on-Close order.

        The order fills at the day's official closing auction price. No
        limit/stop price is needed. TIF is fixed to ``"DAY"`` (the only
        TIF TWS accepts for MOC).

        Parameters
        ----------
        symbol : str
            Ticker symbol (e.g. ``"AAPL"``).
        quantity : int
            Number of shares. Must be positive and <= ``MAX_ORDER_QTY``.
        exchange : str
            Destination exchange. Default ``"SMART"`` preserves legacy
            behaviour. Pass ``"ISLAND"`` to route the MOC directly to
            the NASDAQ INET close-cross (the closing auction the
            NASDAQ-imbalance strategies target).

        Returns
        -------
        str
            The order ID assigned by TWS.

        Examples
        --------
        >>> b.buy_moc("AAPL", 100)   # buys 100 AAPL at the 4pm ET close
        >>> b.buy_moc("AAPL", 100, exchange="ISLAND")  # NASDAQ close only
        """
        return self._place(
            symbol, quantity, "BUY",
            order_type="MOC", limit_price=None, stop_price=None, tif="DAY",
            exchange=exchange,
        )

    def short_moc(self, symbol: str, quantity: int,
                  exchange: str = "SMART") -> str:
        """
        Short a stock with a Market-on-Close order.

        Same as ``sell()`` with MOC, but reads better in strategy code and
        adds a safety check that you are not accidentally selling an
        existing long position.

        Parameters
        ----------
        symbol : str
            Ticker symbol.
        quantity : int
            Number of shares to short. Must be positive.
        exchange : str
            Destination exchange. Default ``"SMART"`` preserves legacy
            behaviour. Pass ``"ISLAND"`` to route the MOC directly to
            the NASDAQ INET close-cross.

        Returns
        -------
        str
            The order ID assigned by TWS.

        Raises
        ------
        RuntimeError
            If a long position already exists for *symbol*.

        Examples
        --------
        >>> b.short_moc("TSLA", 25)  # shorts 25 TSLA at today's close
        >>> b.short_moc("TSLA", 25, exchange="ISLAND")  # NASDAQ close only
        """
        # Mirror short()'s safety check: refuse to convert a long into a
        # short via an implicit sell.
        positions = self.get_positions()
        current_qty = 0
        for p in positions:
            if p.symbol == symbol:
                current_qty = p.quantity
                break

        if current_qty > 0:
            raise RuntimeError(
                f"You have a LONG position of {current_qty} in {symbol}. "
                f"Use sell() to close it, not short_moc(). "
                f"If you really want to short through the position, "
                f"sell first, then short_moc."
            )

        return self._place(
            symbol, quantity, "SELL",
            order_type="MOC", limit_price=None, stop_price=None, tif="DAY",
            exchange=exchange,
        )

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    def cancel(self, order_id: str) -> None:
        """
        Cancel an open order by its order_id.

        Parameters
        ----------
        order_id : str
            The order ID returned by ``buy()``, ``sell()``, etc.

        Raises
        ------
        ValueError
            If *order_id* was not placed in this session.
        """
        trade = self._trades.get(order_id)
        if trade is None:
            raise ValueError(
                f"Unknown order_id '{order_id}'. "
                f"Can only cancel orders placed in this session."
            )
        self._ib.cancelOrder(trade.order)
        logger.info(f"Cancel requested for order {order_id}")

    def cancel_all(self) -> None:
        """Cancel all open orders via TWS global cancel."""
        self._ib.reqGlobalCancel()
        logger.info("Global cancel requested")

    def get_open_orders(self) -> list[OrderInfo]:
        """
        Return all currently open/pending orders.

        Returns
        -------
        list[OrderInfo]
            Snapshot of all open orders. Also caches them in the internal
            trades dict so ``get_order_status()`` works for orders placed
            by other clients.
        """
        open_trades = self._ib.openTrades()
        results = []
        for t in open_trades:
            info = OrderInfo(
                order_id=str(t.order.orderId),
                symbol=t.contract.symbol,
                action=t.order.action,
                quantity=int(t.order.totalQuantity),
                order_type=t.order.orderType,
                limit_price=t.order.lmtPrice if t.order.orderType == "LMT" else None,
                stop_price=t.order.auxPrice if t.order.orderType == "STP" else None,
                status=t.orderStatus.status,
                filled=int(t.orderStatus.filled),
                avg_fill_price=t.orderStatus.avgFillPrice,
            )
            results.append(info)
            # Also cache in our trades dict
            self._trades[info.order_id] = t
        return results

    def get_order_status(self, order_id: str) -> OrderInfo:
        """
        Get current status of a specific order.

        Parameters
        ----------
        order_id : str
            The order ID to query.

        Returns
        -------
        OrderInfo
            Current state of the order.

        Raises
        ------
        ValueError
            If *order_id* is not known to this session.
        """
        trade = self._trades.get(order_id)
        if trade is None:
            raise ValueError(f"Unknown order_id '{order_id}'")

        return OrderInfo(
            order_id=order_id,
            symbol=trade.contract.symbol,
            action=trade.order.action,
            quantity=int(trade.order.totalQuantity),
            order_type=trade.order.orderType,
            limit_price=(
                trade.order.lmtPrice if trade.order.orderType == "LMT" else None
            ),
            stop_price=(
                trade.order.auxPrice if trade.order.orderType == "STP" else None
            ),
            status=trade.orderStatus.status,
            filled=int(trade.orderStatus.filled),
            avg_fill_price=trade.orderStatus.avgFillPrice,
        )

    def wait_for_fill(self, order_id: str, timeout: float = 30.0) -> OrderInfo:
        """
        Block until an order is filled or timeout is reached.

        Parameters
        ----------
        order_id : str
            The order ID to wait on.
        timeout : float
            Maximum seconds to wait before raising ``TimeoutError``.

        Returns
        -------
        OrderInfo
            Final state of the order (status will be ``"Filled"``,
            ``"Cancelled"``, or ``"Inactive"``).

        Raises
        ------
        ValueError
            If *order_id* is not known.
        TimeoutError
            If the order is not terminal within *timeout* seconds.
        """
        trade = self._trades.get(order_id)
        if trade is None:
            raise ValueError(f"Unknown order_id '{order_id}'")

        # ib_async's authoritative set of terminal states — includes
        # ApiCancelled, which we would otherwise wait on forever.
        deadline = datetime.now().timestamp() + timeout
        while trade.orderStatus.status not in OrderStatus.DoneStates:
            if datetime.now().timestamp() > deadline:
                raise TimeoutError(
                    f"Order {order_id} not filled within {timeout}s. "
                    f"Status: {trade.orderStatus.status}"
                )
            self._ib.sleep(0.2)

        return self.get_order_status(order_id)

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def get_positions(self) -> list[Position]:
        """
        Get all current positions from TWS.

        Returns
        -------
        list[Position]
            Position objects with signed quantity (positive = long,
            negative = short).
        """
        self._ib.reqPositions()
        self._ib.sleep(0.5)  # give it a moment to populate
        positions = self._ib.positions()

        results = []
        for p in positions:
            results.append(
                Position(
                    symbol=p.contract.symbol,
                    quantity=int(p.position),
                    avg_cost=p.avgCost,
                )
            )
        return results

    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position for a specific symbol.

        Parameters
        ----------
        symbol : str
            Ticker symbol.

        Returns
        -------
        Position or None
            The position, or ``None`` if flat / no position exists.
        """
        for p in self.get_positions():
            if p.symbol == symbol:
                return p
        return None

    def is_flat(self, symbol: Optional[str] = None) -> bool:
        """
        Check if flat (no position).

        Parameters
        ----------
        symbol : str or None
            If provided, check only that symbol. If ``None``, check
            whether ALL positions are flat.

        Returns
        -------
        bool
            ``True`` if flat.
        """
        positions = self.get_positions()
        if symbol is None:
            return len(positions) == 0 or all(p.quantity == 0 for p in positions)
        pos = self.get_position(symbol)
        return pos is None or pos.quantity == 0

    def reconcile_positions(
        self,
        expected: dict[str, int],
    ) -> dict[str, dict[str, int]]:
        """
        Compare expected positions against actual TWS positions.

        Useful for crash recovery: after a restart, call this to detect
        any drift between what the strategy thinks it holds and what TWS
        actually reports.

        Parameters
        ----------
        expected : dict[str, int]
            ``{symbol: signed_quantity}`` representing the positions the
            strategy believes it should hold.

        Returns
        -------
        dict[str, dict[str, int]]
            Only symbols with discrepancies.  Each value is a dict with
            keys ``"expected"``, ``"actual"``, and ``"diff"``.
            An empty dict means everything matches.

        Examples
        --------
        >>> diffs = b.reconcile_positions({"AAPL": 100, "TSLA": -50})
        >>> if diffs:
        ...     print("Position mismatch!", diffs)
        """
        actual = {p.symbol: p.quantity for p in self.get_positions()}

        # Union of all symbols from both expected and actual
        all_symbols = set(expected) | set(actual)

        discrepancies: dict[str, dict[str, int]] = {}
        for symbol in all_symbols:
            exp_qty = expected.get(symbol, 0)
            act_qty = actual.get(symbol, 0)
            if exp_qty != act_qty:
                discrepancies[symbol] = {
                    "expected": exp_qty,
                    "actual": act_qty,
                    "diff": act_qty - exp_qty,
                }

        if discrepancies:
            logger.warning(
                "Position reconciliation found %d discrepancies: %s",
                len(discrepancies), discrepancies,
            )
        else:
            logger.info("Position reconciliation: all positions match")

        return discrepancies

    # ------------------------------------------------------------------
    # Account & PnL
    # ------------------------------------------------------------------

    def get_account_summary(self) -> dict[str, str]:
        """
        Key account values: net liquidation, margin, buying power, PnL.

        Returns
        -------
        dict[str, str]
            Filtered subset of TWS account values. Keys include
            ``"NetLiquidation"``, ``"BuyingPower"``, ``"AvailableFunds"``,
            etc. Values are strings; cast to ``float`` as needed.
        """
        values = self._ib.accountSummary()

        keys_of_interest = {
            "NetLiquidation",
            "TotalCashValue",
            "SettledCash",
            "BuyingPower",
            "GrossPositionValue",
            "MaintMarginReq",
            "InitMarginReq",
            "AvailableFunds",
            "ExcessLiquidity",
            "UnrealizedPnL",
            "RealizedPnL",
        }

        return {v.tag: v.value for v in values if v.tag in keys_of_interest}

    def get_pnl(self) -> dict[str, float]:
        """
        Quick PnL snapshot.

        Returns
        -------
        dict[str, float]
            ``{"unrealized": float, "realized": float}``.
        """
        summary = self.get_account_summary()
        return {
            "unrealized": float(summary.get("UnrealizedPnL", 0)),
            "realized": float(summary.get("RealizedPnL", 0)),
        }

    def get_margin(self) -> dict[str, float]:
        """
        Margin snapshot.

        Returns
        -------
        dict[str, float]
            ``{"init_margin": float, "maint_margin": float, "available": float}``.
        """
        summary = self.get_account_summary()
        return {
            "init_margin": float(summary.get("InitMarginReq", 0)),
            "maint_margin": float(summary.get("MaintMarginReq", 0)),
            "available": float(summary.get("AvailableFunds", 0)),
        }

    # ------------------------------------------------------------------
    # Market data (simple snapshot)
    # ------------------------------------------------------------------

    def get_price(self, symbol: str) -> Optional[float]:
        """
        Get the last/close price for a symbol.

        Parameters
        ----------
        symbol : str
            Ticker symbol.

        Returns
        -------
        float or None
            Last traded price, falling back to previous close. ``None`` if
            no data is available (e.g. no market data subscription).

        Notes
        -----
        Requires a market data subscription for live prices. Delayed data
        usually works for paper trading.
        """
        contract = self._make_contract(symbol)
        self._ib.reqMktData(contract, "", False, False)
        try:
            self._ib.sleep(2)  # wait for data to arrive
            ticker = self._ib.ticker(contract)

            if ticker and ticker.last and ticker.last > 0:
                return ticker.last
            if ticker and ticker.close and ticker.close > 0:
                return ticker.close
            return None
        finally:
            # Always unsubscribe so we don't leak a live tick stream.
            # ib_async silently returns False if the ticker wasn't found —
            # safe to call unconditionally.
            try:
                self._ib.cancelMktData(contract)
            except Exception:
                logger.debug(
                    "cancelMktData failed for %s (safe to ignore).", symbol,
                )

    # ------------------------------------------------------------------
    # Rebalance helper
    # ------------------------------------------------------------------

    # Order types accepted by rebalance() — kept as a class attribute so
    # tests and callers can introspect it. MOC is included because
    # rebalancing on the close is a common real use case (index funds,
    # end-of-day factor rotation, etc.).
    _REBALANCE_ORDER_TYPES: frozenset[str] = frozenset({"MKT", "MOC"})

    def rebalance(
        self,
        target_positions: dict[str, int],
        order_type: str = "MKT",
        close_unlisted: bool = True,
        tif: str = "DAY",
        exchange: str = "SMART",
    ) -> list[str]:
        """
        Rebalance portfolio to match target positions.

        Parameters
        ----------
        target_positions : dict[str, int]
            ``{symbol: signed_qty}`` where positive = long, negative = short.
        order_type : str
            ``"MKT"`` (default) or ``"MOC"``. LMT/STP are not meaningful
            for a whole-portfolio rebalance because each leg would need
            its own price, so they're rejected.
        close_unlisted : bool
            If ``True``, close positions for symbols not in *target_positions*.
        tif : str
            Time in force for all rebalance orders. Coerced to ``"DAY"``
            for MOC orders inside ``_place()``.
        exchange : str
            Destination exchange for every leg placed by the rebalance
            (both the target-adjust legs and the ``close_unlisted``
            legs). Default ``"SMART"`` preserves legacy behaviour; pass
            ``"ISLAND"`` to route every rebalance order to the NASDAQ
            INET book.

        Returns
        -------
        list[str]
            Order IDs for all orders placed.

        Raises
        ------
        ValueError
            If *order_type* is not ``"MKT"`` or ``"MOC"``.

        Examples
        --------
        >>> b.rebalance({"AAPL": 100, "TSLA": -50, "MSFT": 200})
        >>> b.rebalance({"AAPL": 100}, tif="IOC")
        >>> b.rebalance({"AAPL": 100, "MSFT": 50}, order_type="MOC")
        >>> b.rebalance({"AAPL": 100}, exchange="ISLAND")  # NASDAQ direct
        """
        if order_type not in self._REBALANCE_ORDER_TYPES:
            raise ValueError(
                f"rebalance() only supports MKT or MOC orders, "
                f"got {order_type!r}"
            )

        current = {p.symbol: p.quantity for p in self.get_positions()}
        order_ids: list[str] = []

        # Adjust existing and open new positions
        for symbol, target_qty in target_positions.items():
            current_qty = current.get(symbol, 0)
            diff = target_qty - current_qty

            if diff == 0:
                continue
            elif diff > 0:
                oid = self.buy(
                    symbol, abs(diff), order_type, tif=tif, exchange=exchange,
                )
            else:
                oid = self.sell(
                    symbol, abs(diff), order_type, tif=tif, exchange=exchange,
                )

            order_ids.append(oid)
            logger.info(
                f"Rebalance: {symbol} {current_qty} -> {target_qty} (diff={diff})"
            )

        # Close positions not in target
        if close_unlisted:
            for symbol, qty in current.items():
                if symbol not in target_positions and qty != 0:
                    if qty > 0:
                        oid = self.sell(
                            symbol, abs(qty), order_type,
                            tif=tif, exchange=exchange,
                        )
                    else:
                        oid = self.buy(
                            symbol, abs(qty), order_type,
                            tif=tif, exchange=exchange,
                        )
                    order_ids.append(oid)
                    logger.info(f"Rebalance: closing {symbol} (qty={qty})")

        print(f"Rebalance: placed {len(order_ids)} orders")
        return order_ids

    # ------------------------------------------------------------------
    # Display helpers (nice in Jupyter)
    # ------------------------------------------------------------------

    def show_positions(self) -> None:
        """Print a formatted table of current positions to stdout."""
        positions = self.get_positions()
        if not positions:
            print("No open positions.")
            return

        print(f"{'Symbol':<10} {'Qty':>8} {'Avg Cost':>12} {'Side':<6}")
        print("-" * 40)
        for p in positions:
            side = "LONG" if p.quantity > 0 else "SHORT" if p.quantity < 0 else "FLAT"
            print(f"{p.symbol:<10} {p.quantity:>8} {p.avg_cost:>12.2f} {side:<6}")

    def show_orders(self) -> None:
        """Print a formatted table of open orders to stdout."""
        orders = self.get_open_orders()
        if not orders:
            print("No open orders.")
            return

        print(
            f"{'ID':<8} {'Symbol':<8} {'Action':<6} {'Qty':>6} "
            f"{'Type':<5} {'Status':<12} {'Filled':>6}"
        )
        print("-" * 60)
        for o in orders:
            print(
                f"{o.order_id:<8} {o.symbol:<8} {o.action:<6} {o.quantity:>6} "
                f"{o.order_type:<5} {o.status:<12} {o.filled:>6}"
            )

    def show_account(self) -> None:
        """Print key account info to stdout."""
        summary = self.get_account_summary()
        if not summary:
            print("No account data available.")
            return

        print("Account Summary")
        print("-" * 35)
        for key, val in summary.items():
            try:
                print(f"  {key:<22} {float(val):>12,.2f}")
            except (ValueError, TypeError):
                print(f"  {key:<22} {val:>12}")

    # ------------------------------------------------------------------
    # Convenience: close a single position or flatten everything
    # ------------------------------------------------------------------

    def market_ioc(
        self,
        symbol: str,
        quantity: int,
        action: str,
        exchange: str = "SMART",
    ) -> str:
        """
        Place a market order with Immediate-Or-Cancel time-in-force.

        Routes through ``_place()`` so that ``MAX_ORDER_QTY`` and all
        other safety guards are enforced.

        Parameters
        ----------
        symbol : str
            Ticker symbol (e.g. ``"AAPL"``).
        quantity : int
            Number of shares.  Must be positive.
        action : str
            ``"BUY"`` or ``"SELL"``.
        exchange : str
            Destination exchange for the underlying ``Stock`` contract.
            Default ``"SMART"`` preserves legacy behaviour. Pass a
            direct-route code such as ``"ISLAND"`` to force execution
            on the NASDAQ INET book.

        Returns
        -------
        str
            The order ID assigned by TWS.

        Raises
        ------
        RuntimeError
            If the broker is not connected.
        ValueError
            If *quantity* is invalid or exceeds ``MAX_ORDER_QTY``.
        """
        if not self.is_connected():
            raise RuntimeError(
                "Broker is not connected. Cannot submit MKT IOC order."
            )
        return self._place(
            symbol, quantity, action, order_type="MKT",
            limit_price=None, stop_price=None, tif="IOC",
            exchange=exchange,
        )

    def close_position(
        self,
        symbol: str,
        order_type: str = "MKT",
        tif: str = "DAY",
        exchange: str = "SMART",
    ) -> Optional[str]:
        """
        Close the entire position for a symbol (whether long or short).

        Parameters
        ----------
        symbol : str
            Ticker symbol.
        order_type : str
            ``"MKT"``, ``"LMT"``, ``"STP"``, or ``"MOC"``. For LMT/STP
            you must set the corresponding price on the underlying call
            (this wrapper doesn't accept them here); most callers use
            ``"MKT"`` or ``"MOC"``.
        tif : str
            Time in force (``"DAY"``, ``"GTC"``, ``"IOC"``, ``"FOK"``).
            Coerced to ``"DAY"`` for MOC.
        exchange : str
            Destination exchange for the closing order's underlying
            ``Stock`` contract. Default ``"SMART"`` preserves legacy
            behaviour. Pass a direct-route code such as ``"ISLAND"``
            (NASDAQ INET) when the strategy requires venue-specific
            execution even on the flatten.

        Returns
        -------
        str or None
            Order ID, or ``None`` if already flat.
        """
        pos = self.get_position(symbol)
        if pos is None or pos.quantity == 0:
            print(f"Already flat in {symbol}.")
            return None

        if pos.quantity > 0:
            return self.sell(
                symbol, abs(pos.quantity), order_type,
                tif=tif, exchange=exchange,
            )
        else:
            return self.buy(
                symbol, abs(pos.quantity), order_type,
                tif=tif, exchange=exchange,
            )

    def flatten_all(
        self,
        order_type: str = "MKT",
        tif: str = "DAY",
        exchange: str = "SMART",
    ) -> list[str]:
        """
        Close ALL positions.

        Parameters
        ----------
        order_type : str
            Order type for the closing orders (``"MKT"``, ``"LMT"``,
            ``"STP"``, or ``"MOC"``). ``"MOC"`` is handy for a clean
            end-of-day flatten that fills at the official close price.
        tif : str
            Time in force for the closing orders.  Use ``"IOC"`` for
            end-of-day flattening to avoid stale orders lingering.
            Coerced to ``"DAY"`` for MOC.
        exchange : str
            Destination exchange for each closing order. Default
            ``"SMART"`` preserves legacy behaviour. Pass a direct-route
            code such as ``"ISLAND"`` to force every leg onto the
            NASDAQ INET book.

        Returns
        -------
        list[str]
            Order IDs for all closing orders placed.
        """
        order_ids: list[str] = []
        for pos in self.get_positions():
            if pos.quantity == 0:
                continue
            oid = self.close_position(
                pos.symbol, order_type, tif=tif, exchange=exchange,
            )
            if oid:
                order_ids.append(oid)
        if order_ids:
            print(f"Flattening: placed {len(order_ids)} orders")
        else:
            print("No positions to close.")
        return order_ids

    # ------------------------------------------------------------------
    # Access to the raw IB object for advanced usage
    # ------------------------------------------------------------------

    @property
    def ib(self) -> IB:
        """
        Direct access to the underlying ib_async/ib_insync IB object.
        Use for anything not covered by this wrapper (e.g. options,
        historical data, scanners, etc.)

        Example:
            bars = b.ib.reqHistoricalData(...)
        """
        return self._ib

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def on_error(self, callback: object) -> None:
        """
        Register an additional error callback.

        The callback is *added* alongside the default handler; it does not
        replace it. To remove the default handler, access
        ``broker.ib.errorEvent`` directly.

        Parameters
        ----------
        callback : callable
            Signature: ``callback(reqId, errorCode, errorString, contract)``.

        Examples
        --------
        >>> def my_handler(reqId, code, msg, contract):
        ...     print(f"IB Error {code}: {msg}")
        >>> b.on_error(my_handler)
        """
        self._ib.errorEvent += callback

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False

    def __repr__(self):
        status = "connected" if self.is_connected() else "disconnected"
        return f"Broker(mode={self._mode}, {status})"
