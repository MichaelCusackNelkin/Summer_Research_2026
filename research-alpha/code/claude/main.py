"""CLI entry point for the close-auction imbalance backtest.

Thin wrapper around ``strategy_close_imbalance.main``: parses flags, then
hands off to the async backtest. The implementation module remains importable
on its own and exposes the same defaults.

Usage:
    .venv/bin/python research-alpha/code/main.py --help
    .venv/bin/python research-alpha/code/main.py --smoke
    .venv/bin/python research-alpha/code/main.py --commission 0.0035 --capital 1000000
"""

from __future__ import annotations

import argparse
import asyncio

from strategy_close_imbalance import (
    DEFAULT_COMMISSION_PER_SHARE,
    STARTING_CAPITAL,
    main as run_backtest,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="Run the close-auction imbalance Databento backtest.",
    )
    p.add_argument(
        "--commission",
        type=float,
        default=DEFAULT_COMMISSION_PER_SHARE,
        help="Per-share commission in USD (capped at 1%% of leg notional per leg). "
             f"Default: {DEFAULT_COMMISSION_PER_SHARE}",
    )
    p.add_argument(
        "--capital",
        type=float,
        default=STARTING_CAPITAL,
        help=f"Starting capital used for equal-weight sizing. Default: {STARTING_CAPITAL}",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Run the smoke path: synthesize bid/ask from OHLCV-1s; skip the "
             "bbo-1s batch fetch. Equivalent to setting SMOKE_TEST=1.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    # If --smoke is omitted, leave smoke=None so the env var SMOKE_TEST still applies.
    smoke_arg: bool | None = True if args.smoke else None
    asyncio.run(run_backtest(
        commission=args.commission,
        capital=args.capital,
        smoke=smoke_arg,
    ))


if __name__ == "__main__":
    main()
