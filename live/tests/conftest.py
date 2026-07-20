"""Shared pytest configuration for the live/ test suite.

The autouse fixture below redirects every orchestrator output directory
(``_TARGETS_DIR``, ``_FILLS_DIR``, ``_TRADES_DIR``) to a per-test
``tmp_path`` — for the WHOLE suite, not just the new recorder tests.
Historically many tests here ran ``daily_orchestrator.main()`` end to
end and let it write real ``live/logs/targets/*.json`` and
``live/logs/fills/*.jsonl`` next to real production logs; this fixture
stops that. The redirect is applied only when the test hasn't already
patched the variable itself (monkeypatch stacks correctly — an
in-test ``monkeypatch.setattr`` wins over the autouse one until it
undoes).

This is a test-only change; production behavior is unaffected.

**Do not delete or rewrite existing files under ``live/logs/``** —
some may be real records from actual paper/live runs. Cleanup is out
of scope for this file.
"""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_live_output_dirs(monkeypatch, tmp_path):
    """Redirect ``_TARGETS_DIR`` / ``_FILLS_DIR`` / ``_TRADES_DIR`` to
    ``tmp_path`` for every test in ``live/tests/``.
    """
    # Import inside the fixture so a top-level pytest collection error
    # in daily_orchestrator can't take the fixture out with it.
    try:
        from live import daily_orchestrator
        from live import trade_recorder
    except Exception:
        yield
        return
    targets = tmp_path / "targets"
    fills = tmp_path / "fills"
    trades = tmp_path / "trades"
    monkeypatch.setattr(daily_orchestrator, "_TARGETS_DIR", targets, raising=False)
    monkeypatch.setattr(daily_orchestrator, "_FILLS_DIR", fills, raising=False)
    monkeypatch.setattr(daily_orchestrator, "_TRADES_DIR", trades, raising=False)
    # Also patch the recorder's own module-level default so a stray
    # ``TradeRecorder(...)`` built WITHOUT ``base_dir=...`` falls back
    # to tmp_path, not to the real ``live/logs/trades``. All current
    # constructions pass ``base_dir`` explicitly, so this is a
    # defence-in-depth catch for a future regression.
    monkeypatch.setattr(trade_recorder, "TRADES_DIR", trades, raising=False)
    yield


@pytest.fixture
def live_output_dirs(tmp_path) -> tuple[Path, Path, Path]:
    """Return the per-test ``(targets, fills, trades)`` triple that the
    autouse fixture is redirecting orchestrator output to. Handy for
    tests that want to assert something was written.
    """
    return (tmp_path / "targets", tmp_path / "fills", tmp_path / "trades")
