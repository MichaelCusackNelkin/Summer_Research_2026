"""Pytest fixtures for the ibkr_broker package.

Sole responsibility (for now): prime an asyncio event loop BEFORE any
test module imports ``ibkr_broker`` (which transitively imports
``eventkit``, which calls ``asyncio.get_event_loop_policy().get_event_loop()``
at module load time). On Python 3.14 that call raises
``RuntimeError: There is no current event loop in thread 'MainThread'.``
if the loop policy has not yet been given a running loop — a
long-standing quirk between eventkit and the 3.10+ event-loop-
deprecation path that surfaces during pytest's collection phase.

conftest.py runs before any test module is imported by pytest, so
setting a default loop here is enough to satisfy eventkit's
module-level call downstream. No test uses this loop directly.
"""
from __future__ import annotations

import asyncio


def _prime_event_loop() -> None:
    try:
        asyncio.get_event_loop()
        return
    except RuntimeError:
        pass
    asyncio.set_event_loop(asyncio.new_event_loop())


_prime_event_loop()
