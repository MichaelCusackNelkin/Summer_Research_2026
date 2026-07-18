"""Central definition of the NASDAQ direct-route exchange code used by
every order the ``live/`` pipeline submits.

Why this exists
---------------

The strategy is the NASDAQ closing-auction imbalance play: we forecast
the 15:50 -> 16:00 close-auction move and exit at the NASDAQ MOC print.
A fill on any other venue (BATS, EDGX, IEX, NYSE, ARCA, ...) is nearly
useless — the model is calibrated to the NASDAQ auction print, so an
off-venue fill is uncorrelated with the signal we sized.

IBKR defaults every ``Stock`` contract's ``exchange`` field to
``"SMART"``, which lets IBKR's SMART router pick whichever venue offers
the best NBBO fill in that instant. That routinely means non-NASDAQ
execution — unacceptable for this book. Setting
``primaryExchange="NASDAQ"`` while leaving ``exchange="SMART"`` is NOT
sufficient: ``primaryExchange`` is only used for SMART's symbol-
disambiguation lookup; the routing directive itself is the ``exchange``
field.

Choice of code
--------------

We use ``"ISLAND"`` — IB's internal code for the NASDAQ INET book.
Evidence that this value routes the order rather than merely labelling
the primary listing:

* In ib_async's ``qualifyContractsAsync`` (search for
  ``contract.exchange == "SMART"`` in the installed
  ``ib_async/ib.py``), ``SMART`` is the ONLY exchange the qualifier
  special-cases — on a SMART request, the qualifier explicitly
  preserves the input string via ``c.exchange = contract.exchange``
  before the ``dataclassUpdate`` merges IB's response fields onto the
  contract. Every other exchange value (including ``"ISLAND"``) is
  treated as an authoritative routing directive: the qualified
  contract's ``exchange`` field is whatever IB's ``reqContractDetails``
  came back with, which for a valid direct-route request is the same
  code we asked for. That is exactly the semantics we want — a
  non-SMART code IS a routing choice, not a disambiguator.

* ib_async's ``Contract`` docstring uses ``ISLAND`` in its
  ``primaryExchange`` example ("e.g. ISLAND for MSFT"), showing that
  ``ISLAND`` is the string IB accepts for NASDAQ INET. That specific
  line is inside the ``primaryExchange`` documentation, not the
  ``exchange`` documentation — so it is evidence that IB recognises
  the token, not evidence that setting it on ``exchange`` routes the
  order. The routing evidence is the qualify-preservation logic above.

Together: setting ``exchange="ISLAND"`` requests the NASDAQ INET book
as the destination, and the qualifier preserves that request (SMART
is the sole exception). ``"NASDAQ"`` is accepted by IB as an alias
for the same book, but ``"ISLAND"`` is the canonical token exposed by
the SDK and the one we standardise on. A post-qualification
route-survival assert in ``Broker._make_contract`` (see broker.py)
turns "IB is expected to preserve non-SMART routes" into an enforced
invariant — a mismatch RAISES rather than placing an off-venue order.

Note on the Databento/MIC-style ``"XNAS"``: that is the ISO-10383 MIC
identifier for NASDAQ used in market-data feeds. IB's ``exchange`` field
does NOT accept ``"XNAS"`` as a routing directive — use ``"ISLAND"``.

Coverage
--------

Every order path in ``live/`` — ``daily_orchestrator._submit_mkt_entries``
(entry MKT), ``daily_orchestrator._submit_moc_flatten`` (exit MOC),
``daily_orchestrator._prequalify_symbols`` (contract qualification +
NASDAQ-listed eligibility filter), and ``panic.main`` (kill-switch
flatten) — threads this constant to the ``exchange`` kwarg of
``Broker.buy`` / ``Broker.sell`` / ``Broker.close_position`` /
``Broker._make_contract``. The broker in turn passes it to
``Stock(symbol, exchange, currency)`` before ``qualifyContracts``
runs, so IB sees the direct-route request at both qualification and
order-submission time.

The broker's default (``exchange="SMART"``) is intentionally preserved
for any non-``live/`` caller — this module and the ``live/`` code are
what commit to NASDAQ; other users of ``ibkr_broker`` retain the old
SMART default. If a future ``live/`` file adds a new broker call site,
it MUST import ``NASDAQ_EXCHANGE`` and pass it through. There is no
runtime guard for that inside the broker (adding one would break
existing non-``live/`` users) — code review is the enforcement.
``_prequalify_symbols`` additionally requires
``primaryExchange == "NASDAQ"`` on every qualified contract and DROPS
NYSE/ARCA-primary names, since ISLAND direct-routing does NOT deliver
the NASDAQ close-cross for names whose primary listing is elsewhere
(their MOC prints on their home exchange).
"""
from __future__ import annotations

# IB's direct-route identifier for NASDAQ (INET). Do NOT change this
# to ``"NASDAQ"`` without verifying the whole pipeline still qualifies
# and routes correctly — ib_async's own examples use ``"ISLAND"``.
NASDAQ_EXCHANGE: str = "ISLAND"
