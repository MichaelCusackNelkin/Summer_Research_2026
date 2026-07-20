"""Shared pytest configuration for the live/ test suite.

The autouse fixture below redirects every orchestrator output surface
— ``_TARGETS_DIR``, ``_FILLS_DIR``, ``_TRADES_DIR``, AND the
``config.LIVE_LOG_DIR`` root that ``config.setup_logging`` roots its
``TimedRotatingFileHandler`` at — to a per-test ``tmp_path`` for the
WHOLE suite, not just the recorder tests.

Historically many tests here ran ``daily_orchestrator.main()`` end to
end and let it write real ``live/logs/targets/*.json`` and
``live/logs/fills/*.jsonl`` next to real production logs; the original
fixture stopped THAT. The log-file leak (Round 4 finding #8) survived
because the autouse fixture never patched ``LIVE_LOG_DIR`` itself: an
integration run was still appending ~316 KB to the genuine
``live/logs/live.daily_orchestrator.log``. That's append-only so
nothing is destroyed, but test-run noise interleaves with real
operator logs — exactly what this fixture exists to prevent.

The redirect is applied only when the test hasn't already patched the
variable itself (monkeypatch stacks correctly — an in-test
``monkeypatch.setattr`` wins over the autouse one until it undoes).

This is a test-only change; production behavior is unaffected.

**Do not delete or rewrite existing files under ``live/logs/``** —
some may be real records from actual paper/live runs. Cleanup is out
of scope for this file.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_live_output_dirs(monkeypatch, tmp_path):
    """Redirect every filesystem-writing surface used by the orchestrator
    to ``tmp_path`` for every test in ``live/tests/``.
    """
    # Import inside the fixture so a top-level pytest collection error
    # in daily_orchestrator can't take the fixture out with it.
    try:
        from live import config
        from live import daily_orchestrator
        from live import trade_recorder
    except Exception:
        yield
        return
    targets = tmp_path / "targets"
    fills = tmp_path / "fills"
    trades = tmp_path / "trades"
    logs_root = tmp_path / "logs"
    monkeypatch.setattr(daily_orchestrator, "_TARGETS_DIR", targets, raising=False)
    monkeypatch.setattr(daily_orchestrator, "_FILLS_DIR", fills, raising=False)
    monkeypatch.setattr(daily_orchestrator, "_TRADES_DIR", trades, raising=False)
    # Also patch the recorder's own module-level default so a stray
    # ``TradeRecorder(...)`` built WITHOUT ``base_dir=...`` falls back
    # to tmp_path, not to the real ``live/logs/trades``. All current
    # constructions pass ``base_dir`` explicitly, so this is a
    # defence-in-depth catch for a future regression.
    monkeypatch.setattr(trade_recorder, "TRADES_DIR", trades, raising=False)
    # Redirect the LIVE_LOG_DIR root used by ``config.setup_logging``.
    # ``setup_logging`` is idempotent per logger name (caches in
    # ``_JSON_FORMAT_LOGGERS``), so an earlier test that ran
    # ``daily_orchestrator.main()`` already bound its file handler to
    # the REAL ``live/logs/`` — clearing that cache AND removing any
    # existing FileHandler on the module loggers is what forces the
    # next ``setup_logging`` call to rebuild the handler at
    # ``tmp_path/logs/`` instead.
    monkeypatch.setattr(config, "LIVE_LOG_DIR", logs_root, raising=False)
    _saved_cache = set(config._JSON_FORMAT_LOGGERS)
    config._JSON_FORMAT_LOGGERS.clear()
    _detached_handlers: list[tuple[logging.Logger, logging.Handler]] = []
    for _name in _saved_cache:
        _lg = logging.getLogger(_name)
        for _h in list(_lg.handlers):
            # Rotating file handlers baseFilename lives under the real
            # LIVE_LOG_DIR; a StreamHandler's stream is stdout/stderr
            # (harmless) — only detach the file handlers.
            if isinstance(_h, logging.FileHandler):
                _lg.removeHandler(_h)
                _detached_handlers.append((_lg, _h))
    try:
        yield
    finally:
        # Handlers ADDED by the test (a ``main()``-calling test builds
        # a fresh TimedRotatingFileHandler + StreamHandler on each
        # logger the test's setup_logging call touched) must be
        # cleaned up too, else across a 14-test integration run one
        # logger accumulates ~28 handlers pointing at deleted
        # tmp_path dirs. Nuke every handler on every logger whose
        # name appears in either the saved cache OR the post-test
        # cache, then restore the ORIGINAL handlers this fixture
        # detached at setup. The ``finally`` runs on test failure
        # too, so no logger is left in a half-restored state.
        for _name in set(config._JSON_FORMAT_LOGGERS) | _saved_cache:
            _lg = logging.getLogger(_name)
            for _h in list(_lg.handlers):
                _lg.removeHandler(_h)
                try:
                    _h.close()
                except Exception:
                    pass
        config._JSON_FORMAT_LOGGERS.clear()
        config._JSON_FORMAT_LOGGERS.update(_saved_cache)
        for _lg, _h in _detached_handlers:
            _lg.addHandler(_h)


@pytest.fixture
def live_output_dirs(tmp_path) -> tuple[Path, Path, Path]:
    """Return the per-test ``(targets, fills, trades)`` triple that the
    autouse fixture is redirecting orchestrator output to. Handy for
    tests that want to assert something was written.
    """
    return (tmp_path / "targets", tmp_path / "fills", tmp_path / "trades")
