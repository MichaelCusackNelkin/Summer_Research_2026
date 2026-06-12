"""
Point-in-time (PIT) universe selection + Databento download orchestration.

This module is the pure-logic backbone for a CLI script that:

    1. Builds a sequence of PIT universes (top-N by trailing dollar-volume) at
       rebalance dates spaced ``REBALANCE`` apart starting from a user-supplied
       ``strategy_start``.
    2. Builds a deduped download plan -- per ``(schema, ticker)`` it tracks the
       date-intervals already covered and only requests the missing gaps for
       each new rebalance. Tickers sharing an identical gap interval are
       grouped into one adapter call so we issue O(few) batch requests per
       rebalance per schema.
    3. (Driven by the CLI) executes the plan via ``DatabentoGenericAdapter``
       and consolidates per-schema parquets.

The aggregations and filters mirror ``notebook_studies/universe_selection.ipynb``
but exposed as pure functions of in-memory frames so they unit-test cleanly
against synthetic data.

Constants
---------
* ``DATA_FLOOR``        -- earliest date any cached data exists (clamp lookbacks).
* ``LOOKBACK``          -- trailing window used to rank liquidity.
* ``REBALANCE``         -- spacing between rebalance dates / active-window length.
* ``TOP_N``             -- universe size per rebalance.
* ``MIN_TRADING_FRAC``  -- must have traded on >= this fraction of sessions in window.
* ``MIN_PRICE``         -- drop names whose last close is below this.
* ``ADV_AGG``           -- how to aggregate daily dollar volume into ADV. Swap
                           ``"median"`` <-> ``"mean"`` here; the notebook uses median.
* ``DATASET``           -- Databento dataset (XNAS.ITCH).
* ``OHLCV_SCHEMA``      -- daily bars schema used for universe construction.
* ``DOWNLOAD_SCHEMAS``  -- schemas downloaded for each PIT universe.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Literal, Optional

import polars as pl
from dateutil.relativedelta import relativedelta


# --------------------------------------------------------------------------- #
# Module-level constants -- spec defaults. CLI flags can override these.
# --------------------------------------------------------------------------- #

DATA_FLOOR: date = date(2018, 5, 1)
LOOKBACK: relativedelta = relativedelta(months=18)
REBALANCE: relativedelta = relativedelta(months=6)
TOP_N: int = 3000
MIN_TRADING_FRAC: float = 0.90
MIN_PRICE: float = 1.0
ADV_AGG: Literal["median", "mean"] = "median"

DATASET: str = "XNAS.ITCH"
OHLCV_SCHEMA: str = "ohlcv-1d"
DOWNLOAD_SCHEMAS: tuple[str, ...] = ("tbbo", "imbalance", "statistics")

UNIVERSE_ALL: str = "ALL_SYMBOLS"


# --------------------------------------------------------------------------- #
# Date helpers
# --------------------------------------------------------------------------- #

def _to_date(x: date | datetime | str) -> date:
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    return datetime.strptime(x, "%Y-%m-%d").date()


def _iso(d: date) -> str:
    return d.strftime("%Y-%m-%d")


def clamp_floor(d: date, floor: date = DATA_FLOOR) -> date:
    """Clamp ``d`` to the data floor."""
    return d if d >= floor else floor


def rebalance_dates(
    strategy_start: date | str,
    today: date | str,
    step: relativedelta = REBALANCE,
) -> list[date]:
    """
    Return rebalance anchor dates: ``strategy_start, +step, +2*step, ...``
    up to and including the last one that is <= ``today``.
    """
    start = _to_date(strategy_start)
    end = _to_date(today)
    out: list[date] = []
    cur = start
    while cur <= end:
        out.append(cur)
        cur = cur + step
    return out


# --------------------------------------------------------------------------- #
# Universe selection (pure)
# --------------------------------------------------------------------------- #

def select_universe(
    bars: pl.DataFrame,
    window_start: date,
    window_end: date,
    *,
    top_n: int = TOP_N,
    min_trading_frac: float = MIN_TRADING_FRAC,
    min_price: float = MIN_PRICE,
    adv_agg: Literal["median", "mean"] = ADV_AGG,
) -> pl.DataFrame:
    """
    Compute the top-N universe over ``[window_start, window_end)``.

    Parameters
    ----------
    bars
        OHLCV-1d frame with at least columns ``ts_event``, ``symbol``,
        ``close``, ``volume``. ``ts_event`` may be a date or datetime.
    window_start, window_end
        Half-open window applied to ``ts_event`` before aggregation.
    top_n
        Number of names to keep after filters + sort by ADV.
    min_trading_frac
        Required fraction of distinct session dates within the window the
        symbol must have traded on (drops names with sparse history).
    min_price
        Minimum last close in the window. Drops penny names.
    adv_agg
        ``"median"`` (notebook default; robust to one-off volume spikes) or
        ``"mean"``.

    Returns
    -------
    A frame sorted by ``adv`` descending, with columns
    ``[symbol, adv, n_days, last_px, rank]``. ``rank`` is 1-indexed.
    """
    if adv_agg not in ("median", "mean"):
        raise ValueError(f"adv_agg must be 'median' or 'mean', got {adv_agg!r}")

    ts = pl.col("ts_event")
    # Cast to date for half-open comparison regardless of whether ts_event is a
    # date or a datetime.
    ts_date = ts.cast(pl.Date)

    win = bars.filter(
        (ts_date >= pl.lit(window_start)) & (ts_date < pl.lit(window_end))
    )
    # Edge case K (data hygiene): drop null / non-positive close or volume rows
    # before any aggregation. This keeps bad ticks from inflating ADV and makes
    # ``n_days`` (and the ``n_sessions`` denominator below) count only sessions
    # the name actually traded.
    win = win.filter(
        pl.col("close").is_not_null()
        & pl.col("volume").is_not_null()
        & (pl.col("close") > 0)
        & (pl.col("volume") > 0)
    )
    if win.is_empty():
        return pl.DataFrame(
            schema={
                "symbol": pl.Utf8,
                "adv": pl.Float64,
                "n_days": pl.UInt32,
                "last_px": pl.Float64,
                "rank": pl.UInt32,
            }
        )

    # Distinct session dates actually present in this window -- the denominator
    # for the trading-day filter.
    n_sessions = win.select(pl.col("ts_event").cast(pl.Date).unique().count()).item()
    min_days = min_trading_frac * n_sessions

    dollar_vol = (pl.col("close") * pl.col("volume")).alias("dollar_vol")
    adv_expr = (
        pl.col("dollar_vol").median().alias("adv")
        if adv_agg == "median"
        else pl.col("dollar_vol").mean().alias("adv")
    )

    liq = (
        win.with_columns(dollar_vol)
        .group_by("symbol")
        .agg(
            adv_expr,
            pl.len().alias("n_days"),
            pl.col("close").last().alias("last_px"),
        )
        .filter(
            (pl.col("n_days") >= min_days)
            & (pl.col("last_px") >= min_price)
        )
        .sort("adv", descending=True)
        .head(top_n)
        .with_columns(
            pl.col("symbol").cast(pl.Utf8),
            pl.int_range(1, pl.len() + 1, dtype=pl.UInt32).alias("rank"),
            pl.col("n_days").cast(pl.UInt32),
        )
        .select(["symbol", "adv", "n_days", "last_px", "rank"])
    )
    return liq


@dataclass(frozen=True)
class PITUniverse:
    """One PIT universe = a rebalance date, its active window, and tickers."""

    rebalance_date: date
    active_start: date            # == rebalance_date
    active_end: date              # exclusive: rebalance_date + REBALANCE
    train_start: date             # max(strategy_start, rebalance_date - LOOKBACK)
    train_end: date               # == rebalance_date (exclusive)
    constituents: list[str]       # sorted, deduped, uppercase

    def coverage_window(self) -> tuple[date, date]:
        """Full data window needed for this universe: [train_start, active_end)."""
        return self.train_start, self.active_end


def build_pit_universes(
    bars: pl.DataFrame,
    strategy_start: date | str,
    today: date | str,
    *,
    lookback: relativedelta = LOOKBACK,
    step: relativedelta = REBALANCE,
    data_floor: date = DATA_FLOOR,
    top_n: int = TOP_N,
    min_trading_frac: float = MIN_TRADING_FRAC,
    min_price: float = MIN_PRICE,
    adv_agg: Literal["median", "mean"] = ADV_AGG,
) -> tuple[list[PITUniverse], pl.DataFrame]:
    """
    Build all PIT universes for the rebalance schedule.

    ``strategy_start`` is the hard left edge of all data: no training window (and
    therefore no download) reaches before it. Because the first universe needs a
    full ``lookback`` of history, the earliest rebalance is
    ``max(strategy_start, data_floor) + lookback``; every training window then
    sits wholly inside ``[strategy_start, today)``.

    Returns
    -------
    universes
        List of ``PITUniverse``, one per rebalance date.
    long_frame
        Long-format frame ready to write as ``universes.parquet`` with columns
        ``[rebalance_date, active_start, active_end, train_start, train_end,
        symbol, adv, n_days, last_px, rank]``.
    """
    strategy_start = _to_date(strategy_start)
    today = _to_date(today)
    # `strategy_start` is the HARD left edge of all data: nothing is trained on
    # (or downloaded) before it. Guard against preceding the dataset's absolute
    # floor.
    floor = clamp_floor(strategy_start, floor=data_floor)
    # The first universe needs a full `lookback` of training data sitting wholly
    # inside [floor, today), so the earliest rebalance is floor + lookback.
    first_rebalance = floor + lookback
    rebals = rebalance_dates(first_rebalance, today, step=step)
    universes: list[PITUniverse] = []
    long_rows: list[pl.DataFrame] = []

    for r in rebals:
        train_start = clamp_floor(r - lookback, floor=floor)
        train_end = r
        active_start = r
        active_end = r + step

        liq = select_universe(
            bars,
            window_start=train_start,
            window_end=train_end,
            top_n=top_n,
            min_trading_frac=min_trading_frac,
            min_price=min_price,
            adv_agg=adv_agg,
        )
        constituents = sorted({s.upper() for s in liq["symbol"].to_list()})
        universes.append(
            PITUniverse(
                rebalance_date=r,
                active_start=active_start,
                active_end=active_end,
                train_start=train_start,
                train_end=train_end,
                constituents=constituents,
            )
        )
        if liq.height > 0:
            long_rows.append(
                liq.with_columns(
                    pl.lit(r).alias("rebalance_date"),
                    pl.lit(active_start).alias("active_start"),
                    pl.lit(active_end).alias("active_end"),
                    pl.lit(train_start).alias("train_start"),
                    pl.lit(train_end).alias("train_end"),
                ).select(
                    [
                        "rebalance_date",
                        "active_start",
                        "active_end",
                        "train_start",
                        "train_end",
                        "symbol",
                        "adv",
                        "n_days",
                        "last_px",
                        "rank",
                    ]
                )
            )

    if long_rows:
        long_frame = pl.concat(long_rows, how="vertical")
    else:
        long_frame = pl.DataFrame(
            schema={
                "rebalance_date": pl.Date,
                "active_start": pl.Date,
                "active_end": pl.Date,
                "train_start": pl.Date,
                "train_end": pl.Date,
                "symbol": pl.Utf8,
                "adv": pl.Float64,
                "n_days": pl.UInt32,
                "last_px": pl.Float64,
                "rank": pl.UInt32,
            }
        )
    return universes, long_frame


# --------------------------------------------------------------------------- #
# Interval algebra (closed-open: [start, end))
# --------------------------------------------------------------------------- #

Interval = tuple[date, date]


def _normalize(intervals: Iterable[Interval]) -> list[Interval]:
    """Merge overlapping / touching [start, end) intervals, drop empties."""
    cleaned = [(s, e) for s, e in intervals if s < e]
    cleaned.sort()
    out: list[Interval] = []
    for s, e in cleaned:
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def subtract_intervals(
    required: Interval,
    covered: Iterable[Interval],
) -> list[Interval]:
    """
    Return the half-open subintervals of ``required`` not covered by any
    interval in ``covered``. Result is normalized.
    """
    req_s, req_e = required
    if req_s >= req_e:
        return []

    out: list[Interval] = []
    cursor = req_s
    for s, e in _normalize(covered):
        if e <= cursor:
            continue
        if s >= req_e:
            break
        if s > cursor:
            out.append((cursor, min(s, req_e)))
        cursor = max(cursor, e)
        if cursor >= req_e:
            break
    if cursor < req_e:
        out.append((cursor, req_e))
    return out


# --------------------------------------------------------------------------- #
# Coverage ledger
# --------------------------------------------------------------------------- #

@dataclass
class CoverageLedger:
    """
    Per-(schema, ticker) record of date-intervals already requested.

    Persists to / loads from a JSON file. The file format is::

        {
            "schema_name": {
                "TICKER": [["YYYY-MM-DD", "YYYY-MM-DD"], ...],
                ...
            },
            ...
        }
    """

    by_schema: dict[str, dict[str, list[Interval]]] = field(default_factory=dict)

    # --- core ops --- #

    def covered(self, schema: str, ticker: str) -> list[Interval]:
        return list(self.by_schema.get(schema, {}).get(ticker.upper(), []))

    def gap(
        self,
        schema: str,
        ticker: str,
        required: Interval,
    ) -> list[Interval]:
        return subtract_intervals(required, self.covered(schema, ticker))

    def mark(
        self,
        schema: str,
        tickers: Iterable[str],
        interval: Interval,
    ) -> None:
        s, e = interval
        if s >= e:
            return
        bucket = self.by_schema.setdefault(schema, {})
        for t in tickers:
            t = t.upper()
            existing = bucket.get(t, [])
            existing.append((s, e))
            bucket[t] = _normalize(existing)

    # --- persistence --- #

    def to_json(self) -> str:
        out = {
            schema: {
                t: [[_iso(s), _iso(e)] for s, e in intervals]
                for t, intervals in by_t.items()
            }
            for schema, by_t in self.by_schema.items()
        }
        return json.dumps(out, indent=2, sort_keys=True)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(self.to_json())
        tmp.replace(path)

    @classmethod
    def from_json(cls, payload: str) -> "CoverageLedger":
        raw = json.loads(payload) if payload else {}
        out: dict[str, dict[str, list[Interval]]] = {}
        for schema, by_t in raw.items():
            out[schema] = {
                t.upper(): _normalize(
                    (_to_date(s), _to_date(e)) for s, e in intervals
                )
                for t, intervals in by_t.items()
            }
        return cls(by_schema=out)

    @classmethod
    def load(cls, path: Path) -> "CoverageLedger":
        path = Path(path)
        if not path.exists():
            return cls()
        return cls.from_json(path.read_text())


# --------------------------------------------------------------------------- #
# Download plan
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class DownloadRequest:
    """One ``get_data`` invocation: tickers sharing the same gap interval."""

    schema: str
    rebalance_date: date
    tickers: tuple[str, ...]   # sorted, deduped, uppercase
    start_date: date           # inclusive
    end_date: date             # exclusive (matches adapter convention)


def plan_downloads_for_rebalance(
    ledger: CoverageLedger,
    universe: PITUniverse,
    schemas: Iterable[str] = DOWNLOAD_SCHEMAS,
    *,
    mark: bool = False,
    available_end: Optional[date] = None,
) -> list[DownloadRequest]:
    """
    Build the deduped download plan for one PIT rebalance.

    For each (schema, ticker) compute ``gap = required - already_covered``,
    then group tickers by identical gap interval into one ``DownloadRequest``.
    With the usual 18-month-trailing + 6-month-forward layout, this collapses
    to ~2 requests per schema: one for brand-new names (full ~24mo span) and
    one for persisting names (just the new 6mo forward slice).

    If ``mark=True`` the ledger is updated in place with the planned gaps.

    ``available_end`` (exclusive) clamps the coverage window so we never request
    data past what the dataset actually has -- the last universe's active window
    extends to ``rebalance + 6mo``, which is in the future near the present edge.
    Clamping here (rather than at the request) keeps the ledger honest: it only
    ever records coverage up to available data, so a later re-run with fresh data
    correctly fetches the newly-available tail.
    """
    required: Interval = universe.coverage_window()
    if available_end is not None and required[1] > available_end:
        required = (required[0], available_end)
    plan: list[DownloadRequest] = []

    # Entire window is beyond available data (e.g. a future-dated rebalance):
    # nothing to fetch, nothing to mark.
    if required[0] >= required[1]:
        return plan

    for schema in schemas:
        # gap -> sorted tuple of tickers needing that gap.
        groups: dict[tuple[Interval, ...], list[str]] = {}
        for ticker in universe.constituents:
            gaps = ledger.gap(schema, ticker, required)
            if not gaps:
                continue
            key = tuple(gaps)
            groups.setdefault(key, []).append(ticker.upper())

        for gap_tuple, tickers in groups.items():
            tickers_sorted = tuple(sorted(set(tickers)))
            for s, e in gap_tuple:
                plan.append(
                    DownloadRequest(
                        schema=schema,
                        rebalance_date=universe.rebalance_date,
                        tickers=tickers_sorted,
                        start_date=s,
                        end_date=e,
                    )
                )
            if mark:
                for s, e in gap_tuple:
                    ledger.mark(schema, tickers_sorted, (s, e))

    # Stable, predictable ordering for tests/logging.
    plan.sort(key=lambda r: (r.schema, r.start_date, r.end_date, r.tickers))
    return plan


# Databento caps a single metadata.get_cost / batch.submit_job request at this
# many explicit symbols (use ALL_SYMBOLS above it). A brand-new universe needs
# all top_n names over one window, so a gap-group can exceed this and must be
# chunked.
MAX_SYMBOLS_PER_REQUEST = 2000


def split_oversized_requests(
    plan: list[DownloadRequest],
    max_symbols: int = MAX_SYMBOLS_PER_REQUEST,
) -> list[DownloadRequest]:
    """
    Split any request whose ticker count exceeds Databento's per-call symbol
    cap into consecutive ``<= max_symbols`` chunks.

    Tickers are already sorted within a request, so chunking is deterministic.
    The union of the chunks equals the original request, and the coverage
    ledger tracks coverage per ticker (not per request), so splitting the
    fetch unit leaves coverage, cost totals, and consolidation unchanged --
    it only turns one too-big call into several valid ones. Relative plan
    order is preserved (chunks land where the oversized request was).
    """
    out: list[DownloadRequest] = []
    for req in plan:
        if len(req.tickers) <= max_symbols:
            out.append(req)
            continue
        for i in range(0, len(req.tickers), max_symbols):
            out.append(
                DownloadRequest(
                    schema=req.schema,
                    rebalance_date=req.rebalance_date,
                    tickers=req.tickers[i:i + max_symbols],
                    start_date=req.start_date,
                    end_date=req.end_date,
                )
            )
    return out


def plan_all_downloads(
    universes: Iterable[PITUniverse],
    schemas: Iterable[str] = DOWNLOAD_SCHEMAS,
    initial_ledger: Optional[CoverageLedger] = None,
    available_end: Optional[date] = None,
) -> tuple[list[DownloadRequest], CoverageLedger]:
    """
    Walk the rebalance schedule chronologically, building the full plan and
    leaving a populated coverage ledger. Returns the plan in execution order
    (rebalance-chronological, then by schema), and the *new* ledger reflecting
    everything the plan would fetch (so callers can persist it before / after
    actual execution).

    ``available_end`` (exclusive) clamps every coverage window to the dataset's
    available data edge so no request asks for future data; pass the run's
    ``today``.
    """
    ledger = (
        CoverageLedger(by_schema={
            s: {t: list(intervals) for t, intervals in by_t.items()}
            for s, by_t in initial_ledger.by_schema.items()
        })
        if initial_ledger is not None
        else CoverageLedger()
    )
    schemas = list(schemas)
    full_plan: list[DownloadRequest] = []
    for u in sorted(universes, key=lambda x: x.rebalance_date):
        per_r = plan_downloads_for_rebalance(
            ledger, u, schemas, mark=True, available_end=available_end
        )
        full_plan.extend(per_r)
    # Respect Databento's 2,000-symbol-per-request cap (applies to both
    # get_cost and batch.submit_job) so neither the cost estimate nor the
    # download issues an oversized call.
    full_plan = split_oversized_requests(full_plan)
    return full_plan, ledger
