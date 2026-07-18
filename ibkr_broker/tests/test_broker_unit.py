"""Offline unit tests for ``Broker._make_contract``.

These tests do NOT require TWS — they mock ``Broker._ib.qualifyContracts``
directly. They exist to cover the route-survival safety assert added
alongside the NASDAQ-only routing work in ``live/`` and pin two
independent properties of ``_make_contract``:

1. Backwards compatibility for SMART callers: when a caller passes
   ``exchange="SMART"`` (the default), the guard does NOT fire — the
   qualified contract is returned as-is regardless of what ``exchange``
   field IB's response carries. This preserves legacy behaviour for
   any non-live user of the broker.

2. Fail-closed on non-SMART: when a caller passes a specific
   direct-route (``"ISLAND"`` in the tests below, representing the
   ``live/`` NASDAQ-only pathway), and the qualified contract's
   ``exchange`` field does NOT match, ``_make_contract`` RAISES rather
   than returning an off-venue contract. This is the safety backstop
   that guarantees a NASDAQ-only strategy can never silently ship an
   order to a different venue.

The tests use ``types.SimpleNamespace`` to stand in for
``ib_async.Stock`` — ``_make_contract`` only reads the qualified
contract's ``exchange`` attribute for the guard, so anything with an
``exchange`` attribute suffices.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from ibkr_broker import Broker


def _broker_with_qualify(qualified) -> Broker:
    """Return a ``Broker`` whose ``qualifyContracts`` returns ``qualified``.

    ``qualified`` is the list returned by ``self._ib.qualifyContracts``.
    ``__init__`` is called normally so any side effects (error handler
    registration on the ``IB`` instance) still run — we then swap the
    ``_ib`` attribute for a stub before ``_make_contract`` is called.
    """
    b = Broker()

    def _qc(*contracts):
        return list(qualified)

    b._ib = SimpleNamespace(qualifyContracts=_qc)  # type: ignore[assignment]
    return b


def test_make_contract_island_preserved_returns_contract():
    """Happy path for the non-SMART safety guard: caller asks for
    ``"ISLAND"`` and IB's ``qualifyContracts`` returns a contract
    that kept ``exchange="ISLAND"``. ``_make_contract`` must return
    it unchanged and NOT raise.
    """
    qualified_contract = SimpleNamespace(
        symbol="AAPL", exchange="ISLAND", primaryExchange="NASDAQ",
    )
    b = _broker_with_qualify([qualified_contract])
    got = b._make_contract("AAPL", exchange="ISLAND")
    assert got is qualified_contract
    assert got.exchange == "ISLAND"


def test_make_contract_route_clobbered_to_arca_raises():
    """Fail-closed: caller asked for ``"ISLAND"`` but IB canonicalised
    the qualified contract's ``exchange`` field to ``"ARCA"`` (or any
    other non-matching value). ``_make_contract`` must RAISE a
    ``ValueError`` rather than return an off-venue contract, and the
    message must name both the requested and actual exchanges so
    diagnosis is trivial.
    """
    qualified_contract = SimpleNamespace(
        symbol="BADSYM", exchange="ARCA", primaryExchange="ARCA",
    )
    b = _broker_with_qualify([qualified_contract])
    with pytest.raises(ValueError) as exc_info:
        b._make_contract("BADSYM", exchange="ISLAND")
    msg = str(exc_info.value)
    assert "BADSYM" in msg
    assert "ARCA" in msg
    assert "ISLAND" in msg
    assert "off-venue" in msg.lower() or "refusing" in msg.lower()


def test_make_contract_route_clobbered_to_smart_raises():
    """Fail-closed variant: qualifyContracts stripped the direct route
    back to ``"SMART"`` (e.g. IB's ambiguity fallback). Still an
    off-venue outcome for a caller who explicitly asked for ISLAND —
    must raise.
    """
    qualified_contract = SimpleNamespace(
        symbol="AAPL", exchange="SMART", primaryExchange="NASDAQ",
    )
    b = _broker_with_qualify([qualified_contract])
    with pytest.raises(ValueError):
        b._make_contract("AAPL", exchange="ISLAND")


def test_make_contract_smart_caller_bypasses_guard():
    """Backwards compatibility: when the caller explicitly passes
    ``exchange="SMART"`` (the broker default preserved for every non-
    ``live/`` caller), the guard must NOT fire — even if the qualified
    contract's ``exchange`` field is a specific route. This preserves
    the historical behaviour where ib_async's ``qualifyContractsAsync``
    special-cases SMART to keep the input string, and no non-live user
    of the broker relies on the exchange field matching post-qualify.
    """
    # Regardless of what shape the qualified contract carries, a SMART
    # call must succeed and return the contract unchanged.
    for qualified_exchange in ("SMART", "ISLAND", "ARCA", "NYSE", ""):
        qualified_contract = SimpleNamespace(
            symbol="AAPL",
            exchange=qualified_exchange,
            primaryExchange="NASDAQ",
        )
        b = _broker_with_qualify([qualified_contract])
        got = b._make_contract("AAPL")  # default exchange="SMART"
        assert got is qualified_contract, (
            f"SMART caller should get the qualified contract unchanged "
            f"even when qualified.exchange={qualified_exchange!r}"
        )


def test_make_contract_empty_qualify_raises_unqualified():
    """The pre-existing ``ValueError('Could not qualify contract ...')``
    still fires on an empty qualifyContracts result. This is a
    regression pin for the original path (guarantees the new guard
    was inserted AFTER the unqualified check, not before it, so the
    error message the caller sees still names the qualification
    failure rather than a spurious route mismatch).
    """
    b = _broker_with_qualify([])
    with pytest.raises(ValueError, match="Could not qualify contract"):
        b._make_contract("ZZZZZ", exchange="ISLAND")


def test_make_contract_qualify_none_element_raises_unqualified():
    """ib_async can return a list with a ``None`` in it when the
    contract is ambiguous / not found. Same handling as the empty
    list case: ``ValueError('Could not qualify contract ...')``.
    """
    b = _broker_with_qualify([None])
    with pytest.raises(ValueError, match="Could not qualify contract"):
        b._make_contract("AMBIG", exchange="ISLAND")
