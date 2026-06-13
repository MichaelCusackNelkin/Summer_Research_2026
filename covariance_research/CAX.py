"""
Download corporate actions (splits + dividends) for every ticker in universe.txt
via yfinance's `.actions` API, restricted to [2023-06-01, 2026-06-11] inclusive,
and write a tidy CSV next to this script.

Run from anywhere; paths resolve relative to this file.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

# --- Config ---------------------------------------------------------------
START_DATE = date(2023, 6, 1)
END_DATE = date(2026, 6, 11)  # inclusive

SCRIPT_DIR = Path(__file__).resolve().parent
UNIVERSE_PATH = Path(__file__).resolve().parents[1] / "universe.txt"
OUTPUT_CSV = SCRIPT_DIR / "corporate_actions.csv"

# Concurrency / retry knobs. Yahoo throttles aggressively at high concurrency;
# 16 threads is a reasonable balance for ~2000 names.
MAX_WORKERS = 16
MAX_ATTEMPTS_PER_TICKER = 5
BASE_BACKOFF_SECONDS = 2.0  # backoff = BASE * 2**(attempt-1) +/- jitter
MAX_BACKOFF_SECONDS = 60.0

# Retry sweeps: after the initial pass, we resweep any tickers that errored
# (not the legitimately-empty ones). Each sweep also retries up to
# MAX_ATTEMPTS_PER_TICKER per ticker.
MAX_RETRY_SWEEPS = 3


# --- Data classes ---------------------------------------------------------
@dataclass
class FetchResult:
    symbol: str
    status: str  # "ok_events", "ok_empty", "error"
    rows: list[dict]  # tidy rows (may be empty)
    error: Optional[str] = None


# --- Core fetch -----------------------------------------------------------
def _normalize_actions_frame(actions: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with columns ['date', 'dividend', 'split'] filtered
    to the configured window, with both ticker-empty and out-of-window cases
    handled. Empty DataFrame return is legitimate."""
    if actions is None or len(actions) == 0:
        return pd.DataFrame(columns=["date", "dividend", "split"])

    df = actions.copy()
    # Ensure expected columns exist.
    if "Dividends" not in df.columns:
        df["Dividends"] = 0.0
    if "Stock Splits" not in df.columns:
        df["Stock Splits"] = 0.0

    # Normalize tz-aware DatetimeIndex -> plain calendar date.
    idx = df.index
    if isinstance(idx, pd.DatetimeIndex):
        # Strip tz then take the date.
        try:
            idx_naive = idx.tz_localize(None) if idx.tz is not None else idx
        except TypeError:
            # Already tz-naive
            idx_naive = idx
        cal_dates = [ts.date() for ts in idx_naive]
    else:
        cal_dates = [pd.Timestamp(v).date() for v in idx]

    out = pd.DataFrame(
        {
            "date": cal_dates,
            "dividend": df["Dividends"].astype(float).values,
            "split": df["Stock Splits"].astype(float).values,
        }
    )

    # Filter to window (inclusive).
    mask = (out["date"] >= START_DATE) & (out["date"] <= END_DATE)
    out = out.loc[mask].reset_index(drop=True)

    # Drop rows where both are zero.
    nz = (out["dividend"] != 0.0) | (out["split"] != 0.0)
    out = out.loc[nz].reset_index(drop=True)

    # If a date repeats (rare), aggregate by summing dividends and taking max split.
    if not out.empty and out["date"].duplicated().any():
        out = (
            out.groupby("date", as_index=False)
            .agg(dividend=("dividend", "sum"), split=("split", "max"))
            .sort_values("date")
            .reset_index(drop=True)
        )

    return out


def _yahoo_symbol_variants(symbol: str) -> list[str]:
    """Yahoo-compatible alternate spellings to try for a given universe ticker.

    The universe uses Bloomberg/CRSP-style class delimiters (e.g. 'BRK.B', 'PBR.A')
    but Yahoo expects a hyphen ('BRK-B', 'PBR-A'). Try the literal form first,
    then the dot->hyphen variant.
    """
    variants = [symbol]
    if "." in symbol:
        alt = symbol.replace(".", "-")
        if alt not in variants:
            variants.append(alt)
    return variants


def _try_one_variant(
    yahoo_symbol: str,
) -> tuple[str, Optional[pd.DataFrame], Optional[str]]:
    """Try to obtain an actions-like DataFrame for one Yahoo symbol.

    Returns (kind, frame, error_str):
      - ("actions", df, None): an actions-like DataFrame is available. The frame
        may itself be empty of events — but it is only returned when we have
        positive evidence Yahoo served real data (either `.actions` succeeded,
        or the `.history()` fallback returned price rows). An empty frame here
        therefore means a *trustworthy* "no corporate actions in window".
      - ("error", None, "<msg>"): a retryable failure. This includes the throttle
        signature where `.actions` raises the yfinance '_dividends'/'_splits'
        AttributeError AND the `.history()` fallback returns ZERO rows — Yahoo
        commonly throttles real, listed names this way (404 "Quote not found" /
        empty history), so we must NOT treat zero rows as "no data". Only a
        genuinely delisted symbol also looks like this; we'd rather report it as
        an unresolved failure than silently record it as having no actions.
    """
    try:
        ticker = yf.Ticker(yahoo_symbol)
        actions = ticker.actions  # property; may raise
        return "actions", actions, None
    except AttributeError as e:
        # yfinance bug: if `.history()` returned no rows, the PriceHistory
        # object is missing `_dividends`/`_splits` and `.actions` crashes.
        # Fall back to a date-scoped history call to tell throttle (0 rows ->
        # retryable) apart from a real "has price rows but no events" empty.
        if "_dividends" in str(e) or "_splits" in str(e):
            try:
                end_inclusive = (END_DATE + pd.Timedelta(days=1)).isoformat()
                hist = yf.Ticker(yahoo_symbol).history(
                    start=START_DATE.isoformat(),
                    end=end_inclusive,
                    auto_adjust=False,
                    actions=True,
                )
                if len(hist) == 0:
                    # Zero rows == throttled/blocked (or truly delisted). Retryable;
                    # never recorded as a trustworthy empty.
                    return "error", None, "empty history (rows=0; throttled or delisted)"
                # Price rows exist even though `.actions` crashed — pull actions
                # directly from the history frame (same column names).
                return "actions", hist, None
            except Exception as e2:  # noqa: BLE001
                return "error", None, f"history-fallback: {type(e2).__name__}: {e2}"
        return "error", None, f"{type(e).__name__}: {e}"
    except Exception as e:  # noqa: BLE001
        return "error", None, f"{type(e).__name__}: {e}"


def _fetch_one(symbol: str, attempt: int) -> FetchResult:
    """Single attempt for one ticker. Returns FetchResult.

    Tries each Yahoo-style symbol variant in order. Distinguishes:
      - ok_events: any variant produced >=1 in-range nonzero row
      - ok_empty: a variant returned a trustworthy actions frame (Yahoo served
        real data) that had no in-window events
      - error: every variant raised a retryable failure (incl. zero-row history,
        the throttle signature)
    """
    last_error: Optional[str] = None
    saw_definitive_empty = False
    for yahoo_sym in _yahoo_symbol_variants(symbol):
        kind, frame, err = _try_one_variant(yahoo_sym)
        if kind == "error":
            last_error = err
            continue
        # kind == "actions"
        normalized = _normalize_actions_frame(frame)
        if normalized.empty:
            saw_definitive_empty = True
            continue
        rows = [
            {
                "symbol": symbol,
                "date": r["date"].isoformat(),
                "dividend": float(r["dividend"]),
                "split": float(r["split"]),
            }
            for _, r in normalized.iterrows()
        ]
        return FetchResult(symbol=symbol, status="ok_events", rows=rows)

    if saw_definitive_empty:
        return FetchResult(symbol=symbol, status="ok_empty", rows=[])
    return FetchResult(
        symbol=symbol, status="error", rows=[], error=last_error or "unknown"
    )


def _fetch_with_retries(symbol: str) -> FetchResult:
    """Fetch one ticker with exponential backoff over up to MAX_ATTEMPTS_PER_TICKER tries."""
    last: Optional[FetchResult] = None
    for attempt in range(1, MAX_ATTEMPTS_PER_TICKER + 1):
        result = _fetch_one(symbol, attempt)
        if result.status != "error":
            return result
        last = result
        # Backoff before the next attempt (unless this was the last).
        if attempt < MAX_ATTEMPTS_PER_TICKER:
            backoff = min(
                BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)),
                MAX_BACKOFF_SECONDS,
            )
            jitter = random.uniform(0, backoff * 0.25)
            time.sleep(backoff + jitter)
    assert last is not None
    return last


# --- Driver ---------------------------------------------------------------
def _load_universe() -> list[str]:
    raw = UNIVERSE_PATH.read_text().splitlines()
    tickers: list[str] = []
    seen: set[str] = set()
    for line in raw:
        t = line.strip()
        if not t or t.startswith("#"):
            continue
        if t not in seen:
            seen.add(t)
            tickers.append(t)
    return tickers


def _run_pass(symbols: list[str], label: str, workers: int = MAX_WORKERS) -> dict[str, FetchResult]:
    """Run a single concurrent pass over `symbols`. Returns symbol -> result."""
    results: dict[str, FetchResult] = {}
    total = len(symbols)
    done = 0
    print(f"[{label}] starting pass over {total} tickers (workers={workers})", flush=True)
    start = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_fetch_with_retries, s): s for s in symbols}
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                results[sym] = fut.result()
            except Exception as e:  # noqa: BLE001
                results[sym] = FetchResult(
                    symbol=sym, status="error", rows=[], error=f"future-raised: {e}"
                )
            done += 1
            if done % 100 == 0 or done == total:
                elapsed = time.time() - start
                rate = done / max(elapsed, 1e-6)
                eta = (total - done) / max(rate, 1e-6)
                ok_ev = sum(1 for r in results.values() if r.status == "ok_events")
                ok_em = sum(1 for r in results.values() if r.status == "ok_empty")
                er = sum(1 for r in results.values() if r.status == "error")
                print(
                    f"[{label}] {done}/{total} | events={ok_ev} empty={ok_em} err={er} "
                    f"| {elapsed:.1f}s elapsed, ~{eta:.1f}s eta",
                    flush=True,
                )
    return results


def _resolve(symbols: list[str], workers: int, cooldown: float) -> dict[str, FetchResult]:
    """Run an initial pass over `symbols` then resweep errored tickers."""
    final: dict[str, FetchResult] = {}
    final.update(_run_pass(symbols, label="pass-1", workers=workers))
    for sweep in range(1, MAX_RETRY_SWEEPS + 1):
        errored = [s for s, r in final.items() if r.status == "error"]
        if not errored:
            break
        cool = cooldown * sweep
        print(
            f"[sweep-{sweep}] {len(errored)} tickers still erroring; cooling {cool:.0f}s "
            f"then retrying",
            flush=True,
        )
        time.sleep(cool)
        final.update(_run_pass(errored, label=f"sweep-{sweep}", workers=workers))
    return final


def _rows_from_results(final: dict[str, FetchResult]) -> list[dict]:
    rows: list[dict] = []
    for r in final.values():
        rows.extend(r.rows)
    return rows


def _write_csv(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["symbol", "date", "dividend", "split"])
    if not df.empty:
        df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    df.to_csv(OUTPUT_CSV, index=False)
    return df


def _print_summary(final: dict[str, FetchResult], df: pd.DataFrame) -> int:
    n_total = len(final)
    n_events = sum(1 for r in final.values() if r.status == "ok_events")
    n_empty = sum(1 for r in final.values() if r.status == "ok_empty")
    failed = sorted(s for s, r in final.items() if r.status == "error")
    n_failed = len(failed)

    print("", flush=True)
    print("=" * 72, flush=True)
    print("CORPORATE ACTIONS DOWNLOAD SUMMARY", flush=True)
    print("=" * 72, flush=True)
    print(f"Tickers in this run:     {n_total}", flush=True)
    print(f"  with events in window: {n_events}", flush=True)
    print(f"  empty (no actions):    {n_empty}", flush=True)
    print(f"  still failed:          {n_failed}", flush=True)
    print(f"Output CSV: {OUTPUT_CSV}", flush=True)
    print(f"Rows in CSV: {len(df)}", flush=True)

    if n_failed > 0:
        print("", flush=True)
        print("!!! INCOMPLETE COVERAGE !!!", flush=True)
        print(f"The following {n_failed} tickers failed after all retries:", flush=True)
        for s in failed:
            print(f"  {s}  ({final[s].error})", flush=True)

    # NVDA spot-check (only meaningful when NVDA was part of the run/CSV).
    if not df.empty:
        nvda = df[(df["symbol"] == "NVDA") & (df["date"] == "2024-06-10")]
        if not nvda.empty:
            print("", flush=True)
            print("NVDA 2024-06-10 spot-check:", flush=True)
            for _, r in nvda.iterrows():
                print(
                    f"  symbol={r['symbol']} date={r['date']} "
                    f"dividend={r['dividend']} split={r['split']}",
                    flush=True,
                )

    if n_failed == 0:
        print("", flush=True)
        print("Coverage complete: zero unresolved failures.", flush=True)
        return 0
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--symbols",
        help="Comma-separated subset to fetch (recovery mode). Default: full universe.txt.",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge results into the existing CSV (replace only the fetched symbols' "
        "rows) instead of overwriting the whole file. Use with --symbols.",
    )
    parser.add_argument(
        "--workers", type=int, default=MAX_WORKERS,
        help=f"Concurrent fetch workers (default {MAX_WORKERS}; lower this for recovery "
        "runs to avoid re-triggering Yahoo throttling).",
    )
    parser.add_argument(
        "--cooldown", type=float, default=10.0,
        help="Base inter-sweep cool-off seconds (scaled by sweep number; default 10).",
    )
    args = parser.parse_args()

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
        print(f"Recovery mode: {len(symbols)} symbols", flush=True)
    else:
        if not UNIVERSE_PATH.exists():
            print(f"universe.txt not found at {UNIVERSE_PATH}", file=sys.stderr)
            return 2
        symbols = _load_universe()
        print(f"Loaded {len(symbols)} unique tickers from {UNIVERSE_PATH}", flush=True)
    print(f"Window: {START_DATE.isoformat()} .. {END_DATE.isoformat()} (inclusive)", flush=True)

    final = _resolve(symbols, workers=args.workers, cooldown=args.cooldown)
    new_rows = _rows_from_results(final)

    if args.merge and OUTPUT_CSV.exists():
        existing = pd.read_csv(OUTPUT_CSV)
        fetched = set(symbols)
        kept = existing[~existing["symbol"].isin(fetched)]
        print(
            f"Merge: kept {len(kept)} rows for {existing['symbol'].nunique() - len(fetched & set(existing['symbol']))} "
            f"untouched symbols; replacing rows for {len(fetched)} fetched symbols",
            flush=True,
        )
        combined = kept.to_dict("records") + new_rows
        df = _write_csv(combined)
    else:
        df = _write_csv(new_rows)

    return _print_summary(final, df)


if __name__ == "__main__":
    raise SystemExit(main())
