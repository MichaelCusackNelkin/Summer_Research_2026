"""Kill switch.

Deliberately small and obvious — this is what an operator runs when
something is on fire.

**Scope: ACCOUNT-WIDE.** ``cancel_all()`` is a global cancel and the
per-leg flatten iterates every position IB reports, including manual
trades and other-strategy books. Unlike the daily orchestrator (which
is universe-scoped and pins out-of-universe positions to their current
qty), panic will close whatever it sees. Operators of a shared account
must know this before hitting the button.

Flow:

  1. Connect to TWS (paper default; live only with the confirm string).
  2. BEFORE snapshot: open orders + non-flat positions.
  3. Cancel all open orders (unless --dry-run).
  4. Per-leg flatten via ``broker.close_position(sym, order_type=...)``
     for each non-flat position (unless --cancel-only or --dry-run).
     Failures are collected and printed; the rest of the batch continues.
  5. AFTER snapshot — ALWAYS runs, even after per-leg failures.
  6. Exit code reflects both lingering positions AND lingering orders
     (excluding panic's own newly placed flatten legs).

No new broker logic. Reuses ``ibkr_broker.Broker`` primitives:
``cancel_all``, ``close_position``, ``get_positions``, ``get_open_orders``.

Gotchas the script surfaces:

- MOC orders CANNOT be cancelled after the exchange's ~15:50 ET entry
  cutoff. If any BEFORE-snapshot order has order_type=MOC and it is past
  15:50, we warn: those orders will still print at the close.
- ``close_position`` with MKT outside regular hours queues MKT orders
  until the next open — use ``--flatten-with MOC`` or wait.
- The market-hours and MOC-cutoff checks are weekday+clock only. They
  DO NOT know about exchange holidays: a weekday holiday won't trip
  the warning but the exit code (position count + lingering orders)
  still catches the "not flat" outcome.
- The AFTER snapshot depends on ``broker.get_positions()`` returning a
  complete set; on a slow TWS this can under-report — the printed
  positions/orders lists are the source of truth, not the exit code alone.

Usage:
    .venv/bin/python -m live.panic                       # paper, prompt on live
    .venv/bin/python -m live.panic --dry-run             # inspect only
    .venv/bin/python -m live.panic --cancel-only         # pull orders, keep positions
    .venv/bin/python -m live.panic --flatten-with MOC    # clean end-of-day flatten
    .venv/bin/python -m live.panic --live --live-confirm YES_I_WANT_LIVE_TRADING --yes
"""
from __future__ import annotations

import argparse
import sys
from datetime import time as _time
from typing import Optional

from live import config
from live.nasdaq_routing import NASDAQ_EXCHANGE


# Distinct default client-id from daily_orchestrator (1). If the daily is
# running and holds client 1, panic can still connect with 99.
_PANIC_CLIENT_ID_DEFAULT = 99

# Kept as a module constant so panic can check the confirm string BEFORE
# importing ib_async through the broker package. Must match
# ``ibkr_broker.broker.Broker.LIVE_CONFIRM_MSG`` — the test suite
# verifies the two stay in sync.
_LIVE_CONFIRM_MSG = "YES_I_WANT_LIVE_TRADING"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Kill switch: cancel open orders and flatten all positions.",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--paper", action="store_true",
        help="Paper-trading mode (default).",
    )
    mode.add_argument(
        "--live", action="store_true",
        help=(
            "Live-trading mode. Requires --live-confirm to be the exact "
            f"broker confirmation string ({_LIVE_CONFIRM_MSG})."
        ),
    )
    p.add_argument(
        "--live-confirm", default="",
        help="Broker confirmation string required with --live.",
    )
    p.add_argument(
        "--flatten-with", choices=("MKT", "MOC"), default="MKT",
        help=(
            "Order type for the flatten. MKT (default) fills immediately "
            "during regular trading hours. MOC fills at the closing "
            "auction — useful for a clean end-of-day flatten."
        ),
    )
    p.add_argument(
        "--cancel-only", action="store_true",
        help="Cancel open orders but do NOT flatten positions.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Connect and print BEFORE state; place nothing.",
    )
    p.add_argument(
        "--yes", action="store_true",
        help="Skip the interactive confirmation prompt (for scripted use).",
    )
    p.add_argument(
        "--broker-host", default="127.0.0.1",
        help="TWS host. Default: 127.0.0.1.",
    )
    p.add_argument(
        "--broker-port", type=int, default=None,
        help="TWS port. Default: auto (7497 paper, 7496 live).",
    )
    p.add_argument(
        "--broker-client-id", type=int, default=_PANIC_CLIENT_ID_DEFAULT,
        help=(
            f"TWS client id. Default: {_PANIC_CLIENT_ID_DEFAULT} — distinct "
            "from the daily orchestrator's default (1) so panic can connect "
            "while the daily is holding its own client id."
        ),
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Timing helpers — simple checks, weekday+clock, NOT exchange-holiday-aware.
# ---------------------------------------------------------------------------

def _is_us_market_open_now() -> bool:
    """Weekday + 09:30-16:00 America/New_York. Ignores holidays.

    Mirrors the rough check pattern in
    ``ibkr_broker/tests/test_broker_live.py``.
    """
    now = config.now_ny()
    if now.weekday() >= 5:  # Sat, Sun
        return False
    t = now.time()
    return _time(9, 30) <= t < _time(16, 0)


def _past_moc_entry_cutoff() -> bool:
    """The exchange stops accepting new MOC orders around 15:50 ET.

    After this, ``cancel_all`` cannot pull already-accepted MOC orders
    and NEW MOC submissions get rejected. Weekday + clock only; a
    weekday exchange holiday will look like a normal trading day here.
    """
    now = config.now_ny()
    if now.weekday() >= 5:
        return False
    return now.time() >= _time(15, 50)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_snapshot(orders, positions, title: str) -> None:
    """Show open orders + non-flat positions."""
    print()
    print(f"=== {title} ===")
    print(f"Open orders: {len(orders)}")
    for o in orders:
        print(
            f"  id={o.order_id} {o.symbol} {o.action} qty={o.quantity} "
            f"{o.order_type} status={o.status} filled={o.filled}"
        )
    nonflat = [p for p in positions if p.quantity != 0]
    print(f"Positions: {len(nonflat)} non-flat")
    for p in nonflat:
        side = "LONG" if p.quantity > 0 else "SHORT"
        print(f"  {p.symbol}: {p.quantity} ({side}) avg_cost={p.avg_cost:.2f}")


# ---------------------------------------------------------------------------
# Confirmation prompt
# ---------------------------------------------------------------------------

def _confirm_prompt(args: argparse.Namespace) -> bool:
    """Return True if we may proceed with a destructive action.

    Non-interactive scripts pass ``--yes``. Live mode without ``--yes``
    always prompts (mistakes here are expensive). Paper mode without
    ``--yes`` proceeds silently — this is the "quick oh-no" ergonomics.
    """
    if args.yes:
        return True
    if not args.live:
        return True
    print()
    print("!!! LIVE MODE. This will cancel orders and (optionally) flatten positions.")
    print("!!! Type 'PANIC' and press enter to proceed, anything else to abort.")
    try:
        response = input("> ").strip()
    except EOFError:
        return False
    return response == "PANIC"


# ---------------------------------------------------------------------------
# Broker connect (imports ib_async lazily so tests can monkeypatch)
# ---------------------------------------------------------------------------

def _connect_broker(args: argparse.Namespace):
    from ibkr_broker.broker import Broker
    # Assertion, not runtime check: if this ever drifts the tests catch it.
    assert Broker.LIVE_CONFIRM_MSG == _LIVE_CONFIRM_MSG, (
        "panic._LIVE_CONFIRM_MSG drifted from Broker.LIVE_CONFIRM_MSG"
    )
    broker = Broker()
    kwargs = {"host": args.broker_host, "client_id": args.broker_client_id}
    if args.broker_port is not None:
        kwargs["port"] = args.broker_port
    if args.live:
        broker.connect(mode="live", live_confirm=args.live_confirm, **kwargs)
    else:
        broker.connect(mode="paper", **kwargs)
    return broker


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    # Missing --live-confirm gets its own coded exit so a wrapper /
    # operator can tell "forgot confirm string" apart from "TWS down".
    if args.live and args.live_confirm != _LIVE_CONFIRM_MSG:
        print(f"ERROR: --live requires --live-confirm='{_LIVE_CONFIRM_MSG}'")
        return 9

    # Startup warnings — before connect so the operator sees them first.
    # These are the "what you're ABOUT to do" warnings; the "what's ALREADY
    # in flight" warning fires after the BEFORE snapshot.
    if args.flatten_with == "MOC" and _past_moc_entry_cutoff() and not args.cancel_only:
        print(
            "WARNING: --flatten-with MOC after ~15:50 ET — the exchange "
            "typically rejects new MOC submissions past this cutoff. The "
            "flatten may be a no-op."
        )
    if args.flatten_with == "MKT" and not _is_us_market_open_now() and not args.cancel_only:
        print(
            "WARNING: --flatten-with MKT outside regular hours (09:30-16:00 ET, "
            "weekday). MKT orders will queue until the next open — they will "
            "not flatten immediately. Consider --flatten-with MOC or waiting "
            "for the open."
        )

    try:
        broker = _connect_broker(args)
    except Exception as e:
        print(f"ERROR: broker connect failed: {e}")
        return 8

    print(f"Connected: {broker!r}")

    try:
        # BEFORE snapshot
        orders_before = broker.get_open_orders()
        positions_before = broker.get_positions()
        _print_snapshot(orders_before, positions_before, "BEFORE")

        # H3: warn about EXISTING MOC orders past the cutoff. Driven by the
        # BEFORE snapshot + clock, NOT by --flatten-with — the default MKT
        # panic won't help against MOC orders the daily already queued.
        moc_before = [o for o in orders_before if o.order_type == "MOC"]
        if moc_before and _past_moc_entry_cutoff():
            print(
                f"\nWARNING: {len(moc_before)} MOC order(s) present in the "
                "BEFORE snapshot AND we are past the ~15:50 ET entry cutoff. "
                "cancel_all() CANNOT pull them — they WILL execute at the "
                "16:00 close. This kill switch cannot stop them."
            )

        if args.dry_run:
            action_summary = (
                "skip flatten (--cancel-only)" if args.cancel_only
                else f"per-leg close_position(order_type={args.flatten_with!r})"
            )
            print(f"\nDRY RUN — would cancel_all() and {action_summary}. Nothing placed.")
            return 0

        if not _confirm_prompt(args):
            print("Aborted.")
            return 2

        print()
        print("Cancelling all open orders...")
        # N1: symmetric with the R2 wraps below. If TWS drops right at cancel
        # time we don't want a bare traceback — and we cannot fall through to
        # a flatten attempt, because a TWS drop that killed cancel_all will
        # also kill close_position. End the run loudly. `finally` still
        # disconnects.
        try:
            broker.cancel_all()
        except Exception as e:
            print(
                f"\ncancel_all FAILED — orders may still be LIVE, INVESTIGATE. "
                f"TWS error: {e}"
            )
            return 1
        try:
            broker.sleep(1.0)
        except Exception:
            pass

        # H1: per-leg flatten with per-leg try/except. Every position is
        # attempted even if some fail; failures collected for reporting.
        # broker.flatten_all is intentionally NOT used — its all-or-nothing
        # behaviour would abort at the first unqualifiable name and skip
        # the rest of the book AND the AFTER snapshot.
        placed_ids: list[str] = []
        failures: list[tuple[str, str]] = []
        if not args.cancel_only:
            print(f"Flattening all positions with {args.flatten_with}...")
            # F5: TWS can drop mid-run. Wrap the position enumeration so the
            # loop's own broker call doesn't escape to `finally` and bypass
            # the AFTER snapshot / verdict. cancel_all() already ran, so we
            # cannot claim the book is either flat or unchanged.
            try:
                positions_to_flatten = list(broker.get_positions())
            except Exception as e:
                print(
                    f"ERROR: could not fetch positions to flatten: {e}. "
                    "cancel_all() already ran; legs may or may not have been "
                    "flattened. Continuing to AFTER snapshot."
                )
                positions_to_flatten = []
            for pos in positions_to_flatten:
                if pos.quantity == 0:
                    continue
                try:
                    # NASDAQ-only routing (see live/nasdaq_routing.py):
                    # the strategy is calibrated to the NASDAQ close
                    # auction, so even the kill-switch flatten routes
                    # to the NASDAQ INET book — an off-venue fill on
                    # a panic exit would leak the same off-strategy
                    # cost as an off-venue entry. Same rule everywhere.
                    oid = broker.close_position(
                        pos.symbol,
                        order_type=args.flatten_with,
                        exchange=NASDAQ_EXCHANGE,
                    )
                    if oid:
                        placed_ids.append(oid)
                except Exception as e:
                    failures.append((pos.symbol, str(e)))
                    print(f"WARNING: could not flatten {pos.symbol}: {e}")
            try:
                broker.sleep(3.0)
            except Exception:
                pass

        # AFTER snapshot — normally always runs (per-leg errors don't escape).
        # F5: if the snapshot itself blows up (TWS gone), we cannot compute a
        # trustworthy verdict from missing data. Print a loud line and exit 1
        # explicitly rather than fall through to the verdict, which would
        # compare stale/empty `positions_after` and could falsely report flat.
        try:
            orders_after = broker.get_open_orders()
            positions_after = broker.get_positions()
        except Exception as e:
            print(
                f"\nCOULD NOT CONFIRM FLAT — TWS unreachable ({e}); "
                "assume positions/orders LIVE, INVESTIGATE."
            )
            return 1
        _print_snapshot(orders_after, positions_after, "AFTER")

    finally:
        try:
            broker.disconnect()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Exit-code verdict.
    # ------------------------------------------------------------------
    nonflat_positions = [p for p in positions_after if p.quantity != 0]
    placed_set = set(placed_ids)
    # H2: exclude panic's OWN just-placed flatten legs from the "lingering
    # orders" count. Any order not in placed_set is something cancel_all
    # failed to pull (most importantly a pre-existing MOC past the cutoff).
    # Order-id uniqueness note: IB's `nextValidId` keeps order ids unique
    # across clients on a single running TWS but not across TWS restarts.
    # A theoretical id collision could hide a genuine pre-existing order
    # inside our placed_set — there's no better handle available here.
    lingering_orders = [o for o in orders_after if o.order_id not in placed_set]

    if args.cancel_only:
        if lingering_orders:
            print(f"\nNOT CLEAN: {len(lingering_orders)} orders survived cancel_all:")
            for o in lingering_orders:
                print(f"  id={o.order_id} {o.symbol} {o.order_type} status={o.status}")
            return 1
        print("\nCancelled cleanly. Positions untouched.")
        return 0

    if failures:
        print(f"\n{len(failures)} leg(s) failed to submit:")
        for sym, err in failures:
            print(f"  {sym}: {err}")

    if lingering_orders:
        print(f"\n{len(lingering_orders)} order(s) survived cancel_all:")
        for o in lingering_orders:
            print(f"  id={o.order_id} {o.symbol} {o.action} qty={o.quantity} {o.order_type}")

    if args.flatten_with == "MOC":
        if _past_moc_entry_cutoff():
            # M4: past the cutoff, new MOC submissions are typically rejected.
            # Do NOT claim success — the AFTER snapshot will still show the
            # positions and the operator needs to investigate.
            print(
                f"\nNOT FLAT: MOC flatten submitted past the ~15:50 ET entry "
                f"cutoff — the exchange likely rejected the new MOC orders. "
                f"{len(nonflat_positions)} position(s) remain, "
                f"{len(lingering_orders)} order(s) lingering, "
                f"{len(failures)} failure(s). Verify, and consider MKT or the "
                "next open."
            )
            return 1
        print(
            f"\nMOC flatten submitted. {len(nonflat_positions)} position(s) "
            "will clear at the 16:00 auction (verify at close)."
        )
        # H2 parity: lingering non-panic orders and failed legs both defeat
        # a clean exit — same rule the MKT branch enforces.
        return 0 if not (failures or lingering_orders) else 1

    # MKT branch: fills happen intraday, so anything remaining is stuck.
    if nonflat_positions or lingering_orders or failures:
        print(
            f"\nNOT FLAT: {len(nonflat_positions)} position(s) remain, "
            f"{len(lingering_orders)} order(s) survived cancel, "
            f"{len(failures)} failure(s). Investigate."
        )
        return 1
    print("\nFlat.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
