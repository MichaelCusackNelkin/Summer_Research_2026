"""Central definition of the routing code used by every order the
``live/`` pipeline submits.

Why this file exists
--------------------

Routing is a strategy-critical parameter, not an incidental broker kwarg,
and one setting must apply everywhere in ``live/`` — pre-qualify, entry,
exit, kill switch, post-trade recorder — because a mixed regime would
leak the strategy into venues we can't reason about. Concentrating the
choice here (and forcing every call site to import a named constant, not
type a bare literal or lean on the broker's implicit default) is what
makes an ``ibkr_broker`` behaviour change a code-review issue instead of
a silent execution drift.

Current active routing: SMART
-----------------------------

``ACTIVE_ROUTING = "SMART"`` — IBKR's SMART router picks the venue that
offers the best NBBO fill at the moment the order arrives.

Rationale for the SMART choice on this book:

* **Entry leg (15:50 MKT) benefits from NBBO.** The entry trades in
  continuous session, where SMART routing gets the national best
  bid/offer rather than being confined to the NASDAQ book. On a
  liquid tape this is strictly better than direct-routing to ISLAND
  and paying whatever NASDAQ's local top-of-book happens to be.

* **Exit leg (MOC): two distinct premises — one enforced, one still
  unverified.** The argument is that because the active universe
  (``top_49_universe.txt`` as of 2026-07-21; the plan is to scale
  toward 500 later) is effectively all NASDAQ-primary listings, a
  SMART-routed MOC would land on each name's primary close cross —
  which for these names IS the NASDAQ close auction the model is
  calibrated to. Separate the argument into its two premises:

  1. **"Every name we send an MOC for is NASDAQ-primary"** — now
     enforced. The ``primaryExchange == "NASDAQ"`` filter in
     ``daily_orchestrator._prequalify_symbols`` was re-enabled on
     2026-07-21 alongside this routing change; every qualified
     contract whose ``primaryExchange`` is not ``"NASDAQ"`` is
     dropped and logged before it can reach any order. Universe
     drift (the broad ~2000-name ``universe.txt`` pool that
     ``live/retrain.py`` scans DOES include NYSE / ARCA-primary
     names) can no longer silently introduce a non-NASDAQ-primary
     name into the target book.

  2. **"IBKR's SMART router actually routes an MOC for a
     NASDAQ-primary name to the NASDAQ close cross specifically"**
     — still unverified on this system. We have no paper-session
     evidence that SMART lands the MOC on the NASDAQ close cross
     rather than a different NASDAQ-eligible venue's post-close
     mechanism. It is a plausible-sounding assumption, not an
     observation, and the failure mode is asymmetric — a SMART MOC
     that lands in a thin non-primary closing cross and does not
     fill leaves the position unflattened at 16:00 and carries
     overnight.

  **Pre-live gate.** Before flipping the mode from paper to live under
  this routing, run one paper session and confirm on that session:

  * every ``leg="moc"`` row in the trade recorder parquet has its
    ``venue`` field on the NASDAQ close cross (i.e., ``venue in
    {"ISLAND", "NASDAQ"}``; see the ISLAND alias note below).
    ``venue`` is populated from ``execution.exchange`` and is the
    realized-fill venue, not the requested route. Any row with
    ``venue`` on another auction, or with no fill at all on the MOC
    leg, disproves premise (2) for that name.
  * the ``contracts pre-qualified`` log line reports
    ``n_ok == n_requested`` (or ``n_ok == n_requested - 1`` if ITCI
    is still in the universe and still fails IB qualification). Any
    additional drop is the prequalify filter reacting to a real IB
    ``primaryExchange`` that isn't ``"NASDAQ"`` — either a universe-
    drift name reaching the active file, or (worth checking before
    concluding drift) IB responding with a different-but-equivalent
    spelling of NASDAQ that the filter's exact-string check doesn't
    recognise. Same session, same log, closes both the runtime-no-op
    claim below AND the ``primaryExchange`` spelling on the real
    active universe. The orchestrator escalates that log line to
    WARNING when any drop occurs so the aggregate is visible.

  Re-enabling the prequalify filter above closes premise (1); it
  does NOT close premise (2), and this gate must still pass before
  a live book runs under SMART.

Alternative: ISLAND (NASDAQ INET direct route)
----------------------------------------------

``NASDAQ_EXCHANGE = "ISLAND"`` is kept defined below as the alternative
routing choice. Setting ``exchange="ISLAND"`` requests the NASDAQ INET
book as the destination — bypassing the SMART router entirely. When
this was the active route, evidence that it actually routed the order
rather than merely labelling the primary listing was:

* In ib_async's ``qualifyContractsAsync`` (search for
  ``contract.exchange == "SMART"`` in the installed ``ib_async/ib.py``),
  ``SMART`` is the ONLY exchange the qualifier special-cases — on a
  SMART request, the qualifier explicitly preserves the input string
  via ``c.exchange = contract.exchange`` before the ``dataclassUpdate``
  merges IB's response fields onto the contract. Every other exchange
  value (including ``"ISLAND"``) is treated as an authoritative routing
  directive: the qualified contract's ``exchange`` field is whatever IB's
  ``reqContractDetails`` came back with, which for a valid direct-route
  request routes to the venue we asked for.

  **Caveat: IB canonicalises ISLAND to NASDAQ post-qualification.**
  Empirically verified in ``ibkr_broker/src/ibkr_broker/broker.py``
  (see the note at ``broker.py:366-368`` — a
  ``qualifyContracts(exchange="ISLAND")`` call returns
  ``exchange="NASDAQ"``). "NASDAQ" and "ISLAND" are the same NASDAQ
  INET book; IB just prefers the "NASDAQ" spelling on its response.
  This mismatched-alias pair is precisely why the earlier
  route-survival assertion fired on 46 of 46 requested symbols on
  2026-07-20 before being rolled back — the raw-equality check
  ``qualified.exchange == "ISLAND"`` failed on every symbol IB
  responded to. If that assertion is ever restored, it MUST compare
  against ``{"ISLAND", "NASDAQ"}`` (broker.py:377-380 records the
  same finding).

* ib_async's ``Contract`` docstring uses ``ISLAND`` in its
  ``primaryExchange`` example ("e.g. ISLAND for MSFT"), showing that
  ``ISLAND`` is the string IB accepts for NASDAQ INET.

``"NASDAQ"`` is accepted by IB as an alias for the same book — that is
the same canonicalisation observed above — but ``"ISLAND"`` is the
canonical token exposed by the SDK and the one we standardise on when
using direct routing.

There is no live route-survival assertion inside
``Broker._make_contract`` — the post-qualification check that raised
on a route change was rolled back on 2026-07-20 for the ISLAND→NASDAQ
alias reason described above. If a direct route is ever re-enabled,
restoring a corrected assertion (accepting either alias) is the way
to turn "IB is expected to preserve non-SMART routes" into an
enforced invariant.

Note on the Databento/MIC-style ``"XNAS"``: that is the ISO-10383 MIC
identifier for NASDAQ used in market-data feeds. IB's ``exchange`` field
does NOT accept ``"XNAS"`` as a routing directive — use ``"ISLAND"``.

Coverage
--------

Every order path in ``live/`` — ``daily_orchestrator._submit_mkt_entries``
(entry MKT), ``daily_orchestrator._submit_single_moc`` (per-symbol
exit MOC used by both ``_fill_gated_moc_dispatch`` and
``_sweep_moc_deadline``), ``daily_orchestrator._prequalify_symbols``
(contract qualification; its ``primaryExchange == "NASDAQ"``
eligibility filter is ACTIVE — see the note below), and
``panic.main`` (kill-switch flatten) —
threads ``ACTIVE_ROUTING`` to the ``exchange`` kwarg of
``Broker.buy`` / ``Broker.sell`` / ``Broker.close_position`` /
``Broker._make_contract``. The broker in turn passes it to
``Stock(symbol, exchange, currency)`` before ``qualifyContracts``
runs, so IB sees the routing choice at both qualification and
order-submission time.

The broker's default (``exchange="SMART"``) is intentionally left in
place for any non-``live/`` caller — ``ACTIVE_ROUTING`` currently equals
``"SMART"`` too, but that is a coincidence of the current setting, not
a shared default: every ``live/`` call site MUST still thread
``ACTIVE_ROUTING`` explicitly so switching the active route back to
ISLAND (or to any future venue) is a one-line change here, not a
grep-and-pray across the codebase. If a future ``live/`` file adds a
new broker call site, it MUST import ``ACTIVE_ROUTING`` and pass it
through. There is no runtime guard for that inside the broker (adding
one would break existing non-``live/`` users) — code review is the
enforcement, plus the ``test_no_implicit_default_routing_leaks_through_live_paths``
regression in ``live/tests/test_end_to_end.py`` (and its panic
counterpart) which pins that no call site omits the kwarg.

``_prequalify_symbols`` additionally requires ``primaryExchange ==
"NASDAQ"`` on every qualified contract, dropping NYSE / ARCA-primary
names on the reasoning that direct-routing to NASDAQ does NOT
deliver the NASDAQ close-cross for names whose primary listing is
elsewhere (their MOC prints on their home exchange). The filter is
currently ACTIVE, re-enabled 2026-07-21 alongside the SMART routing
switch — see the block in ``daily_orchestrator._prequalify_symbols``
and its "NASDAQ-primary eligibility filter (ACTIVE)" docstring
section. It had been rolled back 2026-07-20 alongside the broker's
route-survival assertion after a paper replay in which the broker
guard rejected 46 of 46 requested symbols upstream of this filter,
so the drop path was never exercised end-to-end on that run. Under
SMART routing the same concern applies as under ISLAND — a SMART
MOC on a non-NASDAQ-primary name would print on its home exchange's
close cross, not NASDAQ's, breaking the same auction premise the
model is calibrated to — so the filter is now required to make the
NASDAQ-primary premise a code-enforced invariant rather than a
universe-file assumption.

Expected filter behaviour on the current tiered universes:

* ``top_49_universe.txt`` — **TWS-observed**: 49/49 qualify with
  ``primaryExchange == "NASDAQ"``. The observation is recorded in
  ``ibkr_broker/src/ibkr_broker/broker.py:370-372`` from a real
  paper-TWS session.
* ``top_100_universe.txt`` (the previously active universe;
     ``top_49_universe.txt`` is the currently active universe as of
     2026-07-21) —
  **expected but not yet observed on this system**. The expectation
  is that 99 of the 100 qualify (ITCI fails IB qualification
  outright and is dropped by the still-active try/except upstream
  of this filter, never reaching it) with all 99 being NASDAQ-
  primary. No paper-TWS observation exists in this repo for the
  100-name set; the one paper run that would have exercised it
  (2026-07-20) rejected 46 of 46 requested symbols at the broker
  guard **upstream** of this filter, so the drop path was never
  exercised end-to-end on that run. The pre-live gate above is the
  check that settles this on the real active universe.

Its real value is guarding against future universe drift, since
``universe.txt`` (the broad ~2000-name pool ``retrain`` scans)
does include NYSE / ARCA-primary names.

The filter enforces only premise (1) in the "Exit leg (MOC)"
discussion above ("every name we send an MOC for is NASDAQ-
primary"). It does NOT close premise (2) ("a SMART MOC actually
reaches the NASDAQ close cross"); the paper-session ``venue``
gate above is still required before this routing goes to a live
book.
"""
from __future__ import annotations

# IB's direct-route identifier for NASDAQ (INET). Kept defined as the
# alternative routing option — flip ``ACTIVE_ROUTING`` below to this
# constant to force every ``live/`` order back to the NASDAQ INET book.
# Do NOT change this to ``"NASDAQ"`` without verifying the whole
# pipeline still qualifies and routes correctly — ib_async's own
# examples use ``"ISLAND"``. Note IB canonicalises "ISLAND" to
# "NASDAQ" on the qualified contract's response — see the ISLAND
# section of the module docstring.
NASDAQ_EXCHANGE: str = "ISLAND"

# The routing code every ``live/`` order call site actually uses.
# SMART: IBKR picks the best NBBO venue for the entry MKT. The MOC
# behaviour under SMART for our NASDAQ-primary universe is not yet
# empirically verified on this system — see the "pre-live gate" note
# in the module docstring for the venue check that must pass before
# this routing goes to a live book.
#
# Round-2 open concern (SAFETY reviewer finding 7): the round-2
# design widens the effective sweep window toward 15:55 (NASDAQ's
# MOC entry cutoff). Because SMART's MOC venue behaviour is not
# empirically verified for this book, an MOC that SMART routes to a
# non-NASDAQ venue is bounded by that venue's own cutoff (potentially
# 15:50 for NYSE). This is a KNOWN RISK that MUST be closed by the
# pre-live gate above before flipping to live. If the paper-session
# venue check reveals SMART is not always routing MOC to the NASDAQ
# close cross, the mitigation is to flip MOC's exchange kwarg in
# ``_submit_single_moc`` from ``ACTIVE_ROUTING`` to
# ``NASDAQ_EXCHANGE`` (i.e. direct-route MOC to INET) while leaving
# the entry MKT on SMART. Not shipped in round-2 because it changes
# the tested-in-paper routing surface; but the code path is
# structurally ready.
ACTIVE_ROUTING: str = "SMART"
