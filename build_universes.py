"""
Build PIT universes, estimate Databento cost, optionally download.

Usage
-----
Universes + cost estimate only (NO downloads):

    .venv/bin/python build_universes.py --strategy-start 2020-01-01

Also execute the deduped download plan (live Databento batch jobs):

    .venv/bin/python build_universes.py --strategy-start 2020-01-01 --download

Outputs (default ``--out-dir`` is ``<repo>/universes/``):

* ``universes.parquet``      -- long-format PIT universes (one row per
  (rebalance_date, symbol)).
* ``universes.json``         -- human-readable rebalance_date -> ticker list.
* ``cost_estimate.json``     -- per-schema + grand-total Databento cost (USD).
* ``download_ledger.json``   -- coverage ledger; resumable across runs.
* ``consolidated/<schema>.parquet`` -- one big parquet per schema, deduped
  on the natural key (only written under ``--download``).

Outputs are placed under ``--out-dir`` (default
``<repo_root>/universes``); the adapter cache lives at
``<repo_root>/data/dbn_cache/generic`` (matches ``data.py``).

Concurrency
-----------
Cost estimation and downloads fan out via ``asyncio.gather`` bounded by an
``asyncio.Semaphore`` (``--max-concurrency``, default 6). The full plan is
computed deterministically (single-threaded) before any I/O, then the
independent (schema, gap-group) requests are executed concurrently; the
coverage ledger is updated only as each gather result returns successfully so
crashes mid-batch leave the ledger consistent and the next run resumes from
the un-marked gaps.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Optional

import polars as pl
from dotenv import load_dotenv

# Project imports.
from data_collection.data_adapters import DatabentoGenericAdapter
from data_collection.universe_pipeline import (
    ADV_AGG,
    DATA_FLOOR,
    DATASET,
    DOWNLOAD_SCHEMAS,
    LOOKBACK,
    MIN_PRICE,
    MIN_TRADING_FRAC,
    OHLCV_SCHEMA,
    REBALANCE,
    TOP_N,
    UNIVERSE_ALL,
    CoverageLedger,
    DownloadRequest,
    PITUniverse,
    _iso,
    _to_date,
    build_pit_universes,
    plan_all_downloads,
)


# --------------------------------------------------------------------------- #
# Repo / cache layout (matches data.py)
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "dbn_cache" / "generic"
DEFAULT_OUT_DIR = REPO_ROOT / "universes"


# --------------------------------------------------------------------------- #
# Adapter helpers
# --------------------------------------------------------------------------- #

def _resolve_api_key() -> str:
    """Read the Databento API key from the environment.

    The notebooks reference ``DATABENTO-API-KEY`` (with hyphens), but the
    actual ``.env`` exposes ``DATABENTO_API_KEY``. We honor the underscore
    form (the project CLAUDE.md spec) and fall back to the hyphen form if a
    user has the legacy variable set.
    """
    key = os.getenv("DATABENTO_API_KEY") or os.getenv("DATABENTO-API-KEY")
    if not key:
        raise RuntimeError(
            "DATABENTO_API_KEY is not set. Add it to .env (the notebooks "
            "reference 'DATABENTO-API-KEY' with hyphens, but the project's "
            ".env standard is 'DATABENTO_API_KEY' with underscores)."
        )
    return key


async def _ensure_parquet_async(
    adapter: DatabentoGenericAdapter,
    tickers: Iterable[str] | str,
    start: str,
    end: str,
    schema: str,
    dataset: str = DATASET,
) -> Path:
    """
    Make sure the parquet cache exists for these args and return its path.

    Mirrors the ``_ensure_parquet`` try/except FileNotFoundError pattern from
    ``strategy_close_imbalance.py``: scan first, materialize on miss.
    """
    try:
        adapter.scan_data(
            tickers=tickers,
            start_date=start,
            end_date=end,
            dataset=dataset,
            schema=schema,
        )
    except FileNotFoundError:
        await adapter.get_data(
            tickers=tickers,
            start_date=start,
            end_date=end,
            dataset=dataset,
            schema=schema,
            materialize_only=True,
        )

    # Resolve the parquet path via the same cache-key contract.
    ticker_list = adapter._normalize_tickers(tickers)
    cache_key = adapter._generate_cache_key(
        ticker_list, start, end, dataset, schema
    )
    return adapter._get_cache_path(cache_key)


# --------------------------------------------------------------------------- #
# STEP 1 -- universes
# --------------------------------------------------------------------------- #

async def load_all_symbols_ohlcv1d(
    adapter: DatabentoGenericAdapter,
    start: date,
    end: date,
) -> pl.DataFrame:
    """
    Load the master OHLCV-1d frame for ALL_SYMBOLS over [start, end).

    Single adapter call -- if uncached, this submits one batch job; otherwise
    it scans the parquet layer.
    """
    s, e = _iso(start), _iso(end)
    try:
        lf = adapter.scan_data(
            tickers=UNIVERSE_ALL,
            start_date=s,
            end_date=e,
            dataset=DATASET,
            schema=OHLCV_SCHEMA,
        )
        return lf.select(
            ["ts_event", "symbol", "close", "volume"]
        ).collect(engine="streaming")
    except FileNotFoundError:
        df = await adapter.get_data(
            tickers=UNIVERSE_ALL,
            start_date=s,
            end_date=e,
            dataset=DATASET,
            schema=OHLCV_SCHEMA,
        )
        return df.select(["ts_event", "symbol", "close", "volume"])


def write_universe_outputs(
    universes: list[PITUniverse],
    long_frame: pl.DataFrame,
    out_dir: Path,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "universes.parquet"
    json_path = out_dir / "universes.json"

    long_frame.write_parquet(parquet_path)

    payload = {
        _iso(u.rebalance_date): {
            "active_start": _iso(u.active_start),
            "active_end": _iso(u.active_end),
            "train_start": _iso(u.train_start),
            "train_end": _iso(u.train_end),
            "n_constituents": len(u.constituents),
            "constituents": u.constituents,
        }
        for u in universes
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return parquet_path, json_path


# --------------------------------------------------------------------------- #
# STEP 2 -- cost estimation
# --------------------------------------------------------------------------- #

async def estimate_cost(
    plan: list[DownloadRequest],
    api_key: str,
    dataset: str = DATASET,
    max_concurrency: int = 6,
    verbose: bool = True,
) -> dict:
    """
    Sum Databento ``metadata.get_cost`` across the plan, in parallel.

    The blocking SDK call is wrapped in ``asyncio.to_thread`` and fanned out
    with ``asyncio.gather`` bounded by an ``asyncio.Semaphore`` so we don't
    hammer the metadata endpoint or trip rate limits. Result ordering matches
    ``plan`` order regardless of completion order.

    Mode note
    ---------
    The SDK's ``mode`` parameter is deprecated (see
    ``databento/historical/api/metadata.py``); the only available pricing path
    is the historical one, which is also what the adapter's batch downloads
    are billed against. We therefore call ``get_cost`` without ``mode`` to
    avoid a ``DeprecationWarning`` while still using the live pricing path
    that matches the real batch cost we'd incur in Step 3.

    Returns
    -------
    dict
        ``{schema: {requests: int, cost_usd: float}, total_usd: float,
        per_request: [...]}.``
    """
    import databento as db
    client = db.Historical(api_key)

    sem = asyncio.Semaphore(max(1, max_concurrency))
    total_calls = len(plan)
    done_counter = {"n": 0}  # mutable so the inner closure can update.

    def _blocking_get_cost(req: DownloadRequest) -> float:
        return float(client.metadata.get_cost(
            dataset=dataset,
            symbols=list(req.tickers),
            schema=req.schema,
            start=_iso(req.start_date),
            end=_iso(req.end_date),
        ))

    async def _one(i: int, req: DownloadRequest) -> tuple[int, DownloadRequest, float]:
        async with sem:
            cost = await asyncio.to_thread(_blocking_get_cost, req)
            done_counter["n"] += 1
            if verbose:
                print(
                    f"[cost] {done_counter['n']}/{total_calls} {req.schema:<11} "
                    f"{_iso(req.start_date)}..{_iso(req.end_date)} "
                    f"n={len(req.tickers):<5} -> ${cost:,.2f}"
                )
            return i, req, cost

    results = await asyncio.gather(
        *(_one(i, req) for i, req in enumerate(plan))
    )
    # Preserve deterministic ordering (by original plan index).
    results.sort(key=lambda x: x[0])

    per_schema_total: dict[str, float] = {}
    per_schema_count: dict[str, int] = {}
    per_request: list[dict] = []
    for i, req, cost in results:
        per_schema_total[req.schema] = per_schema_total.get(req.schema, 0.0) + cost
        per_schema_count[req.schema] = per_schema_count.get(req.schema, 0) + 1
        per_request.append({
            "i": i,
            "schema": req.schema,
            "rebalance_date": _iso(req.rebalance_date),
            "n_tickers": len(req.tickers),
            "start": _iso(req.start_date),
            "end": _iso(req.end_date),
            "cost_usd": cost,
        })

    by_schema = {
        s: {"requests": per_schema_count[s], "cost_usd": per_schema_total[s]}
        for s in per_schema_total
    }
    total = sum(per_schema_total.values())
    return {
        "by_schema": by_schema,
        "total_usd": total,
        "n_requests": len(plan),
        "per_request": per_request,
    }


def print_cost_summary(report: dict) -> None:
    print()
    print("=" * 60)
    print("Databento cost estimate")
    print("=" * 60)
    for schema, data in sorted(report["by_schema"].items()):
        print(
            f"  {schema:<12}  {data['requests']:>4} requests   "
            f"${data['cost_usd']:>14,.2f}"
        )
    print("-" * 60)
    print(f"  {'TOTAL':<12}  {report['n_requests']:>4} requests   "
          f"${report['total_usd']:>14,.2f}")
    print("=" * 60)


# --------------------------------------------------------------------------- #
# STEP 3 -- downloads + consolidation
# --------------------------------------------------------------------------- #

# Natural keys per schema for the consolidation dedup step. Each column must
# exist in the actual Databento output for that schema.
NATURAL_KEYS: dict[str, list[str]] = {
    "tbbo":       ["ts_recv", "ts_event", "symbol", "instrument_id"],
    "imbalance":  ["ts_recv", "ts_event", "symbol", "instrument_id"],
    "statistics": ["ts_recv", "ts_event", "symbol", "instrument_id", "stat_type"],
}


async def execute_downloads(
    plan: list[DownloadRequest],
    adapter: DatabentoGenericAdapter,
    ledger: CoverageLedger,
    ledger_path: Path,
    dataset: str = DATASET,
    max_concurrency: int = 6,
) -> list[Path]:
    """
    Execute the plan via the adapter, bounded by an ``asyncio.Semaphore``,
    persisting the ledger after each successful request so re-runs are
    resumable.

    The plan was already built deterministically (single-threaded) by
    ``plan_all_downloads``; here we just fan out the independent get-data
    calls. A failing request raises and aborts the gather without marking
    its gap covered, so a re-run will refetch that exact gap (idempotent).

    Returns the list of materialized parquet paths, in plan order.
    """
    sem = asyncio.Semaphore(max(1, max_concurrency))
    ledger_lock = asyncio.Lock()
    total_calls = len(plan)
    done_counter = {"n": 0}

    async def _one(i: int, req: DownloadRequest) -> tuple[int, Path]:
        async with sem:
            p = await _ensure_parquet_async(
                adapter,
                tickers=list(req.tickers),
                start=_iso(req.start_date),
                end=_iso(req.end_date),
                schema=req.schema,
                dataset=dataset,
            )
            # Mark the ledger only after a *successful* materialization, and
            # serialize the write so concurrent downloads can't race on the
            # JSON dump.
            async with ledger_lock:
                ledger.mark(req.schema, req.tickers, (req.start_date, req.end_date))
                ledger.save(ledger_path)
                done_counter["n"] += 1
                print(
                    f"[download] {done_counter['n']}/{total_calls} {req.schema} "
                    f"{_iso(req.start_date)}..{_iso(req.end_date)} "
                    f"n={len(req.tickers)}"
                )
            return i, p

    results = await asyncio.gather(
        *(_one(i, req) for i, req in enumerate(plan))
    )
    results.sort(key=lambda x: x[0])
    return [p for _, p in results]


def consolidate_schema(
    schema: str,
    plan: list[DownloadRequest],
    adapter: DatabentoGenericAdapter,
    out_dir: Path,
    dataset: str = DATASET,
) -> Optional[Path]:
    """
    Lazily scan every parquet produced by the plan for ``schema``, concat, dedup
    on the natural key, and ``sink_parquet`` to ``<out_dir>/<schema>.parquet``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{schema}.parquet"

    schema_reqs = [r for r in plan if r.schema == schema]
    if not schema_reqs:
        return None

    lfs: list[pl.LazyFrame] = []
    for req in schema_reqs:
        try:
            lf = adapter.scan_data(
                tickers=list(req.tickers),
                start_date=_iso(req.start_date),
                end_date=_iso(req.end_date),
                dataset=dataset,
                schema=schema,
            )
        except FileNotFoundError:
            # Should not happen if execute_downloads ran first -- but be loud.
            print(
                f"[consolidate] WARNING: missing parquet for "
                f"{schema} {_iso(req.start_date)}..{_iso(req.end_date)}",
                file=sys.stderr,
            )
            continue
        lfs.append(lf)

    if not lfs:
        return None

    combined = pl.concat(lfs, how="vertical_relaxed")

    # Use as much of the natural key as actually exists in the schema.
    available = combined.collect_schema().names()
    key_cols = [c for c in NATURAL_KEYS.get(schema, []) if c in available]
    if key_cols:
        combined = combined.unique(subset=key_cols, keep="first", maintain_order=False)

    combined.sink_parquet(out_path)
    return out_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build PIT universes, estimate Databento cost, "
                    "and optionally execute the deduped download plan.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--strategy-start",
        required=True,
        help="Hard start of all downloaded data (YYYY-MM-DD): the training-window "
             "floor. No data is fetched before this. The first rebalance is "
             "strategy-start + lookback, so the first 18mo window sits inside the range.",
    )
    p.add_argument(
        "--today",
        default=None,
        help="End of the rebalance schedule. Defaults to today.",
    )
    p.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="Where universes / cost / ledger / consolidated parquets go.",
    )
    p.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help="Databento adapter cache root.",
    )
    p.add_argument("--top-n", type=int, default=TOP_N)
    p.add_argument(
        "--min-price",
        type=float,
        default=MIN_PRICE,
        help="Drop names whose last close in window is below this.",
    )
    p.add_argument(
        "--min-trading-frac",
        type=float,
        default=MIN_TRADING_FRAC,
        help="Required fraction of distinct session dates traded.",
    )
    p.add_argument(
        "--lookback-months",
        type=int,
        default=LOOKBACK.months + 12 * (LOOKBACK.years or 0),
        help="Trailing window for ADV ranking, in months.",
    )
    p.add_argument(
        "--rebalance-months",
        type=int,
        default=REBALANCE.months + 12 * (REBALANCE.years or 0),
        help="Spacing between rebalances / active window length, in months.",
    )
    p.add_argument(
        "--adv-agg",
        choices=("median", "mean"),
        default=ADV_AGG,
    )
    p.add_argument(
        "--schemas",
        nargs="+",
        default=list(DOWNLOAD_SCHEMAS),
        help="Schemas to plan downloads for.",
    )
    p.add_argument(
        "--skip-cost",
        action="store_true",
        help="Skip the Databento cost estimate (live API call).",
    )
    p.add_argument(
        "--download",
        action="store_true",
        help="Execute the deduped download plan (live Databento batch jobs).",
    )
    p.add_argument(
        "--max-concurrency",
        type=int,
        default=6,
        help="Max concurrent adapter calls (cost estimates + downloads).",
    )
    return p.parse_args(argv)


async def amain(args: argparse.Namespace) -> int:
    from dateutil.relativedelta import relativedelta

    load_dotenv()
    pl.enable_string_cache()

    out_dir = Path(args.out_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    api_key = _resolve_api_key()
    adapter = DatabentoGenericAdapter(api_key, cache_dir)

    strategy_start = _to_date(args.strategy_start)
    today = _to_date(args.today) if args.today else date.today()

    lookback = relativedelta(months=args.lookback_months)
    step = relativedelta(months=args.rebalance_months)

    # -- STEP 1: PIT universes --
    # strategy_start is the hard left edge of all data; never precede the
    # dataset's absolute floor. Step 1 fetches only [data_start, today) so no
    # data before strategy_start is ever downloaded.
    data_start = max(strategy_start, DATA_FLOOR)
    print(f"[step1] Loading ALL_SYMBOLS OHLCV-1d [{_iso(data_start)}..{_iso(today)})")
    bars = await load_all_symbols_ohlcv1d(adapter, data_start, today)
    print(f"[step1] bars: {bars.height} rows, "
          f"{bars['symbol'].n_unique()} unique symbols")

    print(f"[step1] Building universes (strategy_start={_iso(strategy_start)}, "
          f"lookback={args.lookback_months}mo, rebalance={args.rebalance_months}mo, "
          f"top_n={args.top_n})")
    universes, long_frame = build_pit_universes(
        bars,
        strategy_start=strategy_start,
        today=today,
        lookback=lookback,
        step=step,
        top_n=args.top_n,
        min_trading_frac=args.min_trading_frac,
        min_price=args.min_price,
        adv_agg=args.adv_agg,
    )
    parquet_path, json_path = write_universe_outputs(universes, long_frame, out_dir)
    print(f"[step1] Wrote {parquet_path} ({long_frame.height} rows) and {json_path}")
    for u in universes:
        print(f"  {_iso(u.rebalance_date)}  active=[{_iso(u.active_start)},"
              f"{_iso(u.active_end)})  N={len(u.constituents)}")

    # -- Build download plan (used by both step 2 and step 3) --
    ledger_path = out_dir / "download_ledger.json"
    initial_ledger = CoverageLedger.load(ledger_path)
    plan, new_ledger = plan_all_downloads(
        universes, schemas=args.schemas, initial_ledger=initial_ledger,
        available_end=today,
    )
    print(f"[plan] {len(plan)} adapter calls planned across schemas "
          f"{sorted(set(r.schema for r in plan))}")

    # -- STEP 2: cost estimate --
    if not args.skip_cost:
        print(
            f"[step2] Estimating cost via Databento metadata.get_cost "
            f"(concurrency={args.max_concurrency}) ..."
        )
        report = await estimate_cost(
            plan, api_key, max_concurrency=args.max_concurrency,
        )
        (out_dir / "cost_estimate.json").write_text(json.dumps(report, indent=2))
        print_cost_summary(report)
    else:
        print("[step2] --skip-cost set; not calling metadata.get_cost.")

    # -- STEP 3: downloads --
    if args.download:
        print(
            f"[step3] Executing download plan "
            f"(concurrency={args.max_concurrency}) ..."
        )
        await execute_downloads(
            plan, adapter, new_ledger, ledger_path,
            max_concurrency=args.max_concurrency,
        )
        new_ledger.save(ledger_path)

        consolidated_dir = out_dir / "consolidated"
        for schema in args.schemas:
            print(f"[step3] Consolidating {schema} ...")
            out = consolidate_schema(schema, plan, adapter, consolidated_dir)
            if out is not None:
                print(f"[step3] wrote {out}")
    else:
        # Even without --download, persist the planned ledger as a *dry-run*
        # alongside the current state, so callers can inspect what *would*
        # have been fetched without committing to it.
        (out_dir / "download_ledger.planned.json").write_text(new_ledger.to_json())
        print(f"[plan] Dry-run ledger written to "
              f"{out_dir / 'download_ledger.planned.json'} "
              f"(re-run with --download to execute).")

    return 0


def main() -> int:
    args = parse_args()
    return asyncio.run(amain(args))


if __name__ == "__main__":
    sys.exit(main())
