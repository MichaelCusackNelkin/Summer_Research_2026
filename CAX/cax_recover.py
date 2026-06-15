"""
Targeted recovery for tickers that CAX.py could not fetch from Yahoo.

Strategy (decided with the user):
  1. CLASSIFY (fast, serial, one attempt each) the failed tickers into:
       - live    : Yahoo served real data (price rows exist; events captured)
       - nodata  : Yahoo has no data for the symbol (404 "Quote not found" /
                   zero-row history) -- almost always a closed-M&A delisting
                   within the window (e.g. DFS, HES, K, JNPR, ANSS). Retrying
                   cannot fix these; free yfinance simply doesn't carry them.
       - transient: a retryable failure (curl timeout, connection reset, etc.)
  2. PATIENT RETRY (serial, long delays, several attempts) only the `transient`
     bucket. Anything that resolves to nodata along the way is reclassified.
  3. MERGE recovered rows into corporate_actions.csv (replacing only the
     recovered symbols' rows; the ~1919 already-good symbols are untouched).
  4. Write a sidecar `missing_corporate_actions.txt` listing the symbols Yahoo
     cannot supply, so the covariance pipeline can treat them as a known gap
     (and recover their *splits* from the Databento raw close series).

Run from anywhere:  .venv/bin/python covariance_research/cax_recover.py
Reads the failed list from `recovery_failed.txt` next to this script.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

import CAX.CAX as CAX  # same directory; reuses its fetch primitives + window/paths

SCRIPT_DIR = Path(__file__).resolve().parent
FAILED_LIST = SCRIPT_DIR / "recovery_failed.txt"
MISSING_SIDECAR = SCRIPT_DIR / "missing_corporate_actions.txt"

# Classification pass: serial pacing, single attempt per symbol. Yahoo throttles
# on request *volume per window*, so the only reliable recovery is a long initial
# cooldown (reset the budget) followed by a slow, well-spaced pass. Defaults below
# are gentle; override via --cooldown / --classify-delay for a cold-start slow run.
CLASSIFY_DELAY = 1.5
# Patient retry pass for the `transient` bucket.
PATIENT_ATTEMPTS = 5
PATIENT_BASE_DELAY = 8.0  # seconds; scales linearly with attempt number


def _load_failed() -> list[str]:
    return [
        ln.strip()
        for ln in FAILED_LIST.read_text().splitlines()
        if ln.strip() and not ln.startswith("#")
    ]


def _bucket_error(msg: str | None) -> str:
    """Map a FetchResult.error string to 'nodata' (Yahoo has nothing) or
    'transient' (retryable). Zero-row history is the delisted/404 signature."""
    if msg and "rows=0" in msg:
        return "nodata"
    return "transient"


def _classify(symbols: list[str], delay: float = CLASSIFY_DELAY) -> tuple[list[dict], list[str], dict[str, str], list[str]]:
    """One serial attempt per symbol. Returns
    (recovered_rows, covered_no_events, nodata{sym:reason}, transient[sym])."""
    recovered_rows: list[dict] = []
    covered: list[str] = []
    nodata: dict[str, str] = {}
    transient: list[str] = []

    total = len(symbols)
    print(f"[classify] {total} symbols, serial, one attempt each (delay={delay}s)", flush=True)
    for i, sym in enumerate(symbols, 1):
        if i > 1:
            time.sleep(delay)
        r = CAX._fetch_one(sym, attempt=1)
        if r.status == "ok_events":
            recovered_rows.extend(r.rows)
            tag = f"live ({len(r.rows)} events)"
        elif r.status == "ok_empty":
            covered.append(sym)
            tag = "live (no events)"
        else:
            kind = _bucket_error(r.error)
            if kind == "nodata":
                nodata[sym] = r.error or "no data"
                tag = "NODATA"
            else:
                transient.append(sym)
                tag = "transient"
        print(f"[classify] {i}/{total} {sym}: {tag}", flush=True)

    print(
        f"[classify] done -> live_with_events={len({r['symbol'] for r in recovered_rows})} "
        f"live_no_events={len(covered)} nodata={len(nodata)} transient={len(transient)}",
        flush=True,
    )
    return recovered_rows, covered, nodata, transient


def _patient_retry(
    symbols: list[str],
) -> tuple[list[dict], list[str], dict[str, str]]:
    """Patiently retry transient symbols. Returns
    (recovered_rows, covered_no_events, still_failed{sym:reason})."""
    recovered_rows: list[dict] = []
    covered: list[str] = []
    still_failed: dict[str, str] = {}

    if not symbols:
        return recovered_rows, covered, still_failed

    print(f"[patient] retrying {len(symbols)} transient symbols", flush=True)
    for sym in symbols:
        resolved = False
        last_err = None
        for attempt in range(1, PATIENT_ATTEMPTS + 1):
            r = CAX._fetch_one(sym, attempt=attempt)
            if r.status == "ok_events":
                recovered_rows.extend(r.rows)
                resolved = True
                print(f"[patient] {sym}: recovered {len(r.rows)} events "
                      f"(attempt {attempt})", flush=True)
                break
            if r.status == "ok_empty":
                covered.append(sym)
                resolved = True
                print(f"[patient] {sym}: live, no events (attempt {attempt})", flush=True)
                break
            last_err = r.error
            # If it has firmed up into a nodata signature, stop early.
            if _bucket_error(last_err) == "nodata":
                print(f"[patient] {sym}: now reads as NODATA, stopping retries", flush=True)
                break
            if attempt < PATIENT_ATTEMPTS:
                time.sleep(PATIENT_BASE_DELAY * attempt)
        if not resolved:
            still_failed[sym] = last_err or "unknown"
    return recovered_rows, covered, still_failed


def _merge_into_csv(symbols_fetched: list[str], new_rows: list[dict]) -> pd.DataFrame:
    """Replace any existing rows for `symbols_fetched` with `new_rows` in the CSV."""
    fetched = set(symbols_fetched)
    if CAX.OUTPUT_CSV.exists():
        existing = pd.read_csv(CAX.OUTPUT_CSV)
        kept = existing[~existing["symbol"].isin(fetched)]
        combined = kept.to_dict("records") + new_rows
    else:
        combined = new_rows
    df = pd.DataFrame(combined, columns=["symbol", "date", "dividend", "split"])
    if not df.empty:
        df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    df.to_csv(CAX.OUTPUT_CSV, index=False)
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cooldown", type=float, default=0.0,
        help="Seconds to sleep BEFORE any requests, to let Yahoo's volume throttle "
        "reset. Recommended ~1800 (30 min) after heavy use.",
    )
    parser.add_argument(
        "--classify-delay", type=float, default=CLASSIFY_DELAY,
        help="Serial spacing between requests in the classify pass. Use ~12 for a "
        "cold-start slow pass to avoid re-tripping the throttle.",
    )
    args = parser.parse_args()

    symbols = _load_failed()
    print(f"Loaded {len(symbols)} failed symbols from {FAILED_LIST}", flush=True)
    print(f"Window: {CAX.START_DATE} .. {CAX.END_DATE} (inclusive)", flush=True)

    if args.cooldown > 0:
        print(f"Cooling down {args.cooldown:.0f}s before any requests "
              f"(letting Yahoo throttle reset)...", flush=True)
        time.sleep(args.cooldown)

    rows_a, covered_a, nodata, transient = _classify(symbols, delay=args.classify_delay)
    rows_b, covered_b, still_failed = _patient_retry(transient)

    # still_failed splits: nodata-signature -> unavailable; else -> genuinely
    # unresolved transient (network) which we also report as not-yet-covered.
    for sym, err in still_failed.items():
        nodata.setdefault(sym, err)

    recovered_rows = rows_a + rows_b
    covered_no_events = sorted(set(covered_a + covered_b))
    recovered_syms = sorted({r["symbol"] for r in recovered_rows})

    # Merge: replace rows for every symbol we positively resolved (events OR
    # confirmed-no-events). Leave nodata symbols out of the CSV entirely.
    resolved_syms = sorted(set(recovered_syms) | set(covered_no_events))
    df = _merge_into_csv(resolved_syms, recovered_rows)

    # Sidecar for the unrecoverable names.
    unavailable = sorted(nodata)
    header = (
        "# Symbols whose corporate actions Yahoo/yfinance could not supply.\n"
        "# Almost all are closed-M&A delistings within the data window; their\n"
        "# tickers are purged from Yahoo. Recover SPLITS for these from the\n"
        "# Databento raw close series (overnight gaps at round ratios); dividends\n"
        "# are not available from free sources for these names.\n"
        "# Format: <symbol>\\t<reason>\n"
    )
    with MISSING_SIDECAR.open("w") as fh:
        fh.write(header)
        for sym in unavailable:
            fh.write(f"{sym}\t{nodata[sym]}\n")

    # Summary.
    print("", flush=True)
    print("=" * 72, flush=True)
    print("RECOVERY SUMMARY", flush=True)
    print("=" * 72, flush=True)
    print(f"Failed symbols processed:   {len(symbols)}", flush=True)
    print(f"  recovered with events:    {len(recovered_syms)}", flush=True)
    print(f"  recovered, no events:     {len(covered_no_events)}", flush=True)
    print(f"  unavailable on Yahoo:     {len(unavailable)}", flush=True)
    print(f"Rows now in CSV:            {len(df)}  ({CAX.OUTPUT_CSV})", flush=True)
    print(f"Unavailable list written to {MISSING_SIDECAR}", flush=True)
    if unavailable:
        print("", flush=True)
        print(f"Unavailable ({len(unavailable)}) -> handle via Databento split detection:",
              flush=True)
        print("  " + ", ".join(unavailable), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
