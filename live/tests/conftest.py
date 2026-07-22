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
    # Speed knobs for the round-2 fill-gated MOC dispatcher + post-close
    # position poll. Production defaults are geared to the 15:50-16:05
    # wall-clock window; a test suite has no such window and would take
    # 30+ minutes with the real values. Both surfaces sit behind kwargs
    # so tests can push these limits as needed.
    #
    # DEFAULT_ENTRY_FILL_TIMEOUT_SECS bounds ``_fill_gated_moc_dispatch``'s
    # wall-clock wait for entries to reach terminal state (30s in prod).
    # Push it to 0.5s here so tests with stuck / never-filling entries
    # exit the dispatcher promptly and let the sweep run.
    monkeypatch.setattr(
        config, "DEFAULT_ENTRY_FILL_TIMEOUT_SECS", 0.5, raising=False,
    )
    # DEFAULT_POST_CLOSE_POLL_* bound ``_poll_positions_after_close``'s
    # backoff loop. Cap them so a post-close verdict lands in
    # milliseconds regardless of wall clock.
    #
    # * INTERVAL / MAX_INTERVAL shrink the backoff sleeps to a few ms.
    # * ``POST_CLOSE_POLL_MAX_WINDOW_MINS`` is the round-2 hard upper
    #   cap on wall-clock time spent in the poll: production picks
    #   ``min(today_ny(16, 5), now + POST_CLOSE_POLL_MAX_WINDOW_MINS)``,
    #   so patching this small caps the whole poll window. 0.005 min
    #   = 0.3s, which fits ~6 polls at 0.02s interval — enough for
    #   the 2-consecutive-flat-reads guard while keeping the suite
    #   fast. Also prevents the wall-clock hang the correctness
    #   reviewer reproduced (81s vs 1.3s when the deadline was 16:05
    #   and the test ran at 09:00 ET).
    # * ``DEFAULT_POST_CLOSE_POLL_UNTIL_TIME`` is left near-future
    #   ("23:59") so the MAX_WINDOW clamp above is the binding term.
    # * ``DEFAULT_POST_CLOSE_POLL_NOT_BEFORE_TIME`` is the round-2
    #   LOWER bound — the poll refuses to declare FLAT before this
    #   wall-clock time. In prod 16:00 + settle; here we set to
    #   "00:00" + settle=0 so tests running at any wall clock never
    #   have to wait for a future not_before threshold.
    # * ``DEFAULT_POST_CLOSE_MIN_CONSECUTIVE_FLAT_READS`` — production
    #   requires 2 consecutive flats to survive a transient empty
    #   response; keep the prod default here so tests exercise the
    #   same behaviour, and rely on the MAX_WINDOW cap for speed.
    monkeypatch.setattr(
        config, "DEFAULT_POST_CLOSE_POLL_INITIAL_INTERVAL_SECS", 0.02,
        raising=False,
    )
    monkeypatch.setattr(
        config, "DEFAULT_POST_CLOSE_POLL_MAX_INTERVAL_SECS", 0.05,
        raising=False,
    )
    monkeypatch.setattr(
        config, "POST_CLOSE_POLL_MAX_WINDOW_MINS", 0.005,
        raising=False,
    )
    monkeypatch.setattr(
        config, "DEFAULT_POST_CLOSE_POLL_UNTIL_TIME", "23:59",
        raising=False,
    )
    monkeypatch.setattr(
        config, "DEFAULT_POST_CLOSE_POLL_NOT_BEFORE_TIME", "00:00",
        raising=False,
    )
    monkeypatch.setattr(
        config, "DEFAULT_POST_CLOSE_POLL_SETTLE_SECS", 0.0,
        raising=False,
    )
    # ROUND-4 SAFETY FIX (SAFETY reviewer finding 3). The daily
    # orchestrator now runs a SECOND SWEEP at ``sweep_deadline`` with
    # a real ``_sleep_until(sweep_deadline)`` between the first and
    # second sweeps. That deadline can be minutes to an hour in the
    # future for many tests (especially ``--replay-ignore-cutoff``
    # tests where ``effective_submit_by`` is ``now + 1h``). Without
    # a no-op patch here, every such test would actually sleep until
    # that deadline and stall the suite. Tests that WANT the real
    # sleep behaviour (unit tests of ``_sleep_until`` itself) override
    # this patch with their own ``monkeypatch.setattr``.
    #
    # NOTE: many individual tests still call
    # ``monkeypatch.setattr(daily_orchestrator, "_sleep_until", ...)``
    # explicitly — those overrides remain load-bearing (they document
    # WHICH tests rely on the no-op sleep). This autouse patch is a
    # defence-in-depth catch for tests written before the round-4
    # second-sweep landed.
    from live import daily_orchestrator as _do_sleep
    monkeypatch.setattr(
        _do_sleep, "_sleep_until", lambda *a, **kw: None, raising=False,
    )
    # And the pumped marks-wait wrapper — same reasoning. On the rc=0
    # path main() calls this with a 350s timeout, which would stall
    # any test that lets that codepath run without a stub.
    monkeypatch.setattr(
        _do_sleep, "_wait_for_marks_pumped",
        lambda recorder, broker, timeout=0, pump_interval=30.0: False,
        raising=False,
    )
    # Round-3 MOC-semantics test wiring: real IB does not fold MOC
    # fills into ``get_positions()`` until after the 16:00 crossing
    # prints. The test-only ``_FakeBroker.simulate_close_auction`` is
    # what triggers that fold in the fake. To keep happy-path
    # end-to-end tests concise (they should not have to know about
    # this fake-broker plumbing), wrap ``_poll_positions_after_close``
    # to invoke ``simulate_close_auction()`` on the broker BEFORE the
    # first poll. Tests that want the pre-auction position book
    # verbatim (moc_reject_all rejection tests) still work — a
    # rejected MOC never enters ``_pending_moc_fills``, so the fold
    # is a no-op for them. Tests that DELIBERATELY want to see the
    # pre-fold state can either use a broker without
    # ``simulate_close_auction`` (real class) or monkey-patch the
    # method to a no-op.
    from live import daily_orchestrator as _do  # local: avoid the
    # top-level import cycle risk during collection failure paths.
    _real_poll = _do._poll_positions_after_close

    def _poll_with_auction_sim(broker, **kwargs):
        _sim = getattr(broker, "simulate_close_auction", None)
        if callable(_sim):
            try:
                _sim()
            except Exception:
                # Fake's simulate_close_auction is best-effort. Don't
                # let a test-only exception mask a real poll failure.
                pass
        return _real_poll(broker=broker, **kwargs)

    monkeypatch.setattr(
        _do, "_poll_positions_after_close", _poll_with_auction_sim,
        raising=True,
    )
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
