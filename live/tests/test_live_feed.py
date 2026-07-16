"""Regression tests for the ``LiveFeed`` symbol-resolution + record buffer.

The C1 review flagged the previous ``_on_record`` for storing the numeric
instrument_id string (Databento's live gateway always returns
``stype_out=instrument_id``, so ``stype_out_symbol`` is digits, not the
ticker). These tests use a synthetic ``SymbolMappingMsg``-shaped stub to
prove the fix.
"""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from live.live_feed import LiveFeed, _decode_side


class _FakeSymbolMappingMsg:
    """Duck-typed stand-in for ``databento_dbn.SymbolMappingMsg``.

    We subclass the real class via ``isinstance`` faking: LiveFeed imports
    the real ``SymbolMappingMsg`` inside ``_on_record``, so the test uses
    a real construction path via monkeypatching.
    """

    def __init__(self, instrument_id: int, stype_in_symbol: str, stype_out_symbol: str):
        self.instrument_id = instrument_id
        self.stype_in_symbol = stype_in_symbol
        self.stype_out_symbol = stype_out_symbol


def _patch_symbol_mapping_msg(monkeypatch):
    """Force LiveFeed's ``isinstance(record, SymbolMappingMsg)`` check to
    accept our fake class.
    """
    import databento_dbn
    monkeypatch.setattr(databento_dbn, "SymbolMappingMsg", _FakeSymbolMappingMsg)


def test_symbol_mapping_prefers_stype_in_when_out_is_digits(monkeypatch):
    """C1 regression: when ``stype_out_symbol`` is all digits (the normal
    live case), the buffered ticker must come from ``stype_in_symbol``.
    """
    _patch_symbol_mapping_msg(monkeypatch)
    feed = LiveFeed(api_key="dummy", universe=["AAPL"], reconnect=False)
    feed._on_record(_FakeSymbolMappingMsg(
        instrument_id=12345, stype_in_symbol="AAPL", stype_out_symbol="12345",
    ))
    assert feed._id_to_symbol[12345] == "AAPL"


def test_symbol_mapping_uses_stype_out_when_it_is_a_real_ticker(monkeypatch):
    """The non-live-gateway case (e.g. historical replay) can have a real
    ticker in ``stype_out_symbol``. Prefer it.
    """
    _patch_symbol_mapping_msg(monkeypatch)
    feed = LiveFeed(api_key="dummy", universe=["MSFT"], reconnect=False)
    feed._on_record(_FakeSymbolMappingMsg(
        instrument_id=99, stype_in_symbol="MSFT.OQ", stype_out_symbol="MSFT",
    ))
    assert feed._id_to_symbol[99] == "MSFT"


def test_decode_side_variants():
    assert _decode_side("B") == "B"
    assert _decode_side("a") == "A"
    assert _decode_side(None) == "N"
    assert _decode_side("X") == "N"
    # ord('B') = 66
    assert _decode_side(66) == "B"
