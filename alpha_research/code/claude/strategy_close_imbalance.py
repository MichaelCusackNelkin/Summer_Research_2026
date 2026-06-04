"""
Close-auction imbalance strategy backtest (Databento-driven).

Mirrors the pandas/CSV homework in ``closing_imbalance_homework.ipynb`` (Q5: the
intraday "ride-with the published imbalance" strategy) but is driven off the
Databento Imbalance schema instead of NYSE IBF gz files.

Logic per trading day:

  * NOII dissemination begins at 15:50:00 NY. For EACH symbol, the decision
    snapshot is that symbol's FIRST imbalance message whose timestamp is
    at-or-after 15:50:00. The live bbo (latest bbo-1s tick known at that
    moment) is paired with the snapshot.
  * Two filters:
      - ``abs(imb_dollar_qty)            > $500,000``   (dollar imbalance)
      - ``abs(qty_signal = imb_dollar/sdv) > 0.01``     (size vs. session $vol)
    No top-N ADV universe pre-filter -- every symbol passing both filters is
    tradeable that day.
  * Sizing is EQUAL WEIGHT across the N symbols that pass the filters:
    ``target_notional = capital / N``. The dollar-neutrality constraint is
    DROPPED -- if every name has a buy-side imbalance we go net long, and
    that's fine. There is no rolling regression / alpha overlay.
  * Direction follows the homework convention: trade WITH the published
    imbalance -- long when the imbalance is on the buy side (``side == 'B'``),
    short when on the sell/ask side (``side == 'A'``). Equivalently
    ``sign(imb_dollar_qty)``.
  * Entries pay one half-spread against the live bbo (ask if long, bid if
    short); exits fill at the official close-auction print.

The per-day execution is still driven by DBNStore.replay so porting to a live
event-loop is mechanical: the same Strategy.on_event callback would consume
real-time records instead of replayed ones. Out-of-band ("batch") lookups are
limited to data that would genuinely be batch in production: today's session
$ volume for the qty_signal denominator and today's official close print.

Results are persisted to parquet (``trades.parquet`` and ``daily_pnl.parquet``
alongside the PNG plots) for post-hoc analysis.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path

import databento as db
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from dotenv import load_dotenv

from data_collection.data_adapters import DatabentoGenericAdapter


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "data" / "dbn_cache" / "generic"
PLOT_DIR = Path(__file__).resolve().parent / "strategy_close_imbalance_plots"
RESULTS_DIR = PLOT_DIR  # parquet results land alongside the PNGs

DATASET = "XNAS.ITCH"
START_DATE = "2026-04-26"
END_DATE = "2026-05-26"
# When SMOKE_TEST=1 we (a) skip the bbo-1s batch fetch and synthesize bid/ask
# from the 15:50:00 OHLCV-1s close, and (b) skip the ohlcv-1d batch fetch
# entirely (the day-list is derived from intraday bars). Both paths exist
# purely so the trade-booking / filtering / plotting path can be validated
# end-to-end without triggering a Databento batch download.
SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"
SMOKE_HALF_SPREAD_BPS = 5.0  # synthetic half-spread for the SMOKE bid/ask

# NOII (closing imbalance) dissemination begins at 15:50:00 NY. Per-symbol
# decision time is that symbol's FIRST imbalance message at-or-after this.
SNAP_TIME_NY = time(15, 50, 0)
SESSION_OPEN_NY = time(9, 30)
SESSION_PRE_CUTOFF_NY = time(15, 50)
# Once every known symbol has latched (or we hit this NY-time wall) we stop
# the per-day replay. 15:58 is the NYSE "imbalance freeze" -- no further OC
# interest can change, so the snapshot is fully formed by then.
REPLAY_STOP_TIME_NY = time(15, 58, 0)

MIN_ABS_QTY_SIGNAL = 0.01            # |imbalance / session $vol| > 1%
MIN_ABS_DOLLAR_IMBALANCE = 500_000.0  # |imbalance * ref_price| > $500k
STARTING_CAPITAL = 1_000_000.0
# IBKR-style per-share commission cap. Applied as min(per_share*qty, cap_frac*notional)
# PER LEG (entry fill AND exit fill), since both are real executions.
DEFAULT_COMMISSION_PER_SHARE = 0.0035
COMMISSION_CAP_FRAC = 0.01

PX_SCALE = 1e-9          # Databento fixed-point int64 -> dollars on XNAS.ITCH
INT64_PX_UNDEF = np.iinfo(np.int64).max
UINT64_TS_UNDEF = (1 << 64) - 1   # DBN sentinel for "no timestamp set"
NY_TZ = "America/New_York"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _record_ts_ns(rec) -> int | None:
    """Best-available event-time clock for a DBN record.

    bbo-1s snapshots use ``ts_recv`` as the second-aligned sample tick and leave
    ``ts_event`` set to the DBN UINT64 sentinel whenever the underlying book
    didn't update in that second. Falling back to ``ts_recv`` keeps the replay
    clock monotone and prevents sentinel-keyed records from being interpreted
    as 'after the decision time'.
    """
    ts = getattr(rec, "ts_event", None)
    if ts is None or ts == UINT64_TS_UNDEF:
        ts = getattr(rec, "ts_recv", None)
    if ts is None or ts == UINT64_TS_UNDEF:
        return None
    return ts


def _ny_time_to_utc_ns(day: date, t: time) -> int:
    """Nanoseconds since epoch (UTC) for `day` at NY-time `t`."""
    dt = pl.datetime(day.year, day.month, day.day, t.hour, t.minute, t.second,
                     time_zone=NY_TZ)
    return int(pl.select(dt.dt.convert_time_zone("UTC").dt.timestamp("ns")).item())


def _build_iid_to_symbol(store: db.DBNStore) -> dict[int, str]:
    """Invert the symbology mapping to instrument_id -> raw symbol."""
    out: dict[int, str] = {}
    for sym, intervals in store.mappings.items():
        for iv in intervals:
            iid_str = iv.get("symbol")
            if iid_str and iid_str.isdigit():
                out[int(iid_str)] = sym
    return out


# ---------------------------------------------------------------------------
# Data pulls (batch / out-of-band)
# ---------------------------------------------------------------------------

async def _ensure_parquet(adapter: DatabentoGenericAdapter, schema: str) -> None:
    try:
        adapter.scan_data(
            tickers="ALL_SYMBOLS",
            start_date=START_DATE,
            end_date=END_DATE,
            dataset=DATASET,
            schema=schema,
        )
    except FileNotFoundError:
        await adapter.get_data(
            tickers="ALL_SYMBOLS",
            start_date=START_DATE,
            end_date=END_DATE,
            dataset=DATASET,
            schema=schema,
        )


async def load_imbalance(adapter: DatabentoGenericAdapter) -> pl.LazyFrame:
    await _ensure_parquet(adapter, "Imbalance")
    return (
        adapter.scan_data(
            tickers="ALL_SYMBOLS",
            start_date=START_DATE,
            end_date=END_DATE,
            dataset=DATASET,
            schema="Imbalance",
        )
        .with_columns(
            pl.col("ts_event").dt.convert_time_zone(NY_TZ),
            pl.col("symbol").cast(pl.Utf8),
        )
        .filter(pl.col("ts_event").dt.time().is_between(SESSION_OPEN_NY, time(16, 0)))
    )


async def load_ohlcv_1s(adapter: DatabentoGenericAdapter) -> pl.LazyFrame:
    await _ensure_parquet(adapter, "OHLCV-1s")
    return (
        adapter.scan_data(
            tickers="ALL_SYMBOLS",
            start_date=START_DATE,
            end_date=END_DATE,
            dataset=DATASET,
            schema="OHLCV-1s",
        )
        .with_columns(
            pl.col("ts_event").dt.convert_time_zone(NY_TZ),
            pl.col("symbol").cast(pl.Utf8),
        )
        .filter(pl.col("ts_event").dt.time().is_between(SESSION_OPEN_NY, time(16, 0)))
    )


async def load_statistics(adapter: DatabentoGenericAdapter) -> pl.LazyFrame:
    await _ensure_parquet(adapter, "statistics")
    return (
        adapter.scan_data(
            tickers="ALL_SYMBOLS",
            start_date=START_DATE,
            end_date=END_DATE,
            dataset=DATASET,
            schema="statistics",
        )
        .with_columns(
            pl.col("ts_event").dt.convert_time_zone(NY_TZ),
            pl.col("symbol").cast(pl.Utf8),
        )
        .filter(pl.col("stat_type") == 11)
    )


# ---------------------------------------------------------------------------
# Per-day replay
# ---------------------------------------------------------------------------

@dataclass
class SymbolState:
    bid_px: float = float("nan")
    ask_px: float = float("nan")
    # Latched snapshot fields (set on the first post-15:50 imbalance message).
    latched: bool = False
    snap_bid: float = float("nan")
    snap_ask: float = float("nan")
    snap_ref_price: float = float("nan")
    snap_total_imbalance_qty: int = 0
    snap_side: str = "N"


@dataclass
class SnapshotResult:
    bid_px: dict[str, float]            # bid at the per-symbol latch moment
    ask_px: dict[str, float]            # ask at the per-symbol latch moment
    imb_dollar_qty: dict[str, float]    # signed: +B (long imbalance), -A (sell imb)
    side: dict[str, str]                # 'B' or 'A'
    ref_price: dict[str, float]         # NOII reference price at the latch
    total_imbalance_qty: dict[str, int] # unsigned imbalance qty at the latch


class Strategy:
    """
    Maintains per-symbol bbo state from the replay stream and latches each
    symbol's closing-imbalance snapshot on its FIRST imbalance message whose
    timestamp is at-or-after the decision time (15:50:00 NY).

    Once a symbol is latched, subsequent imbalance updates for that symbol are
    ignored -- we use the first observable on-close state, which mirrors the
    homework's ``groupby('symbol').first()`` after filtering on source_time.

    bbo updates continue to flow into ``state[*].bid/ask`` for any symbol that
    hasn't latched yet, so each symbol's latch reflects the live bbo as of its
    own first post-15:50 imbalance message.
    """

    def __init__(self, iid_to_symbol: dict[int, str], decision_ts_ns: int,
                 stop_ts_ns: int, expected_symbols: set[str] | None = None) -> None:
        self.iid_to_symbol = iid_to_symbol
        self.decision_ts_ns = decision_ts_ns
        self.stop_ts_ns = stop_ts_ns
        self.expected_symbols = expected_symbols
        self.state: dict[str, SymbolState] = {}
        self.n_latched: int = 0

    def _get_state(self, sym: str) -> SymbolState:
        st = self.state.get(sym)
        if st is None:
            st = SymbolState()
            self.state[sym] = st
        return st

    def on_bbo(self, rec) -> None:
        sym = self.iid_to_symbol.get(rec.instrument_id)
        if sym is None:
            return
        st = self._get_state(sym)
        # Once latched, the BBO at the latch moment is frozen for that symbol;
        # further bbo updates don't affect its snapshot but we still keep the
        # running book for diagnostics.
        bid_px = getattr(rec, "bid_px_00", None)
        if bid_px is None:
            bid_px = getattr(rec, "bid_px", None)
        ask_px = getattr(rec, "ask_px_00", None)
        if ask_px is None:
            ask_px = getattr(rec, "ask_px", None)
        if bid_px is not None and bid_px != INT64_PX_UNDEF:
            st.bid_px = bid_px * PX_SCALE
        if ask_px is not None and ask_px != INT64_PX_UNDEF:
            st.ask_px = ask_px * PX_SCALE

    def on_imbalance(self, rec, ts: int) -> None:
        sym = self.iid_to_symbol.get(rec.instrument_id)
        if sym is None:
            return
        st = self._get_state(sym)
        if st.latched:
            return
        if ts < self.decision_ts_ns:
            return
        # First message at/after 15:50:00 NY for this symbol -- latch it.
        if not (st.bid_px > 0 and st.ask_px > 0):
            # No live bbo yet -- can't price an entry, so skip this name.
            # (Latch flag stays False so any later imbalance + bbo combo
            # could in principle latch -- but the homework convention is
            # "first" so we don't update side fields from later messages.
            # In practice the bbo lands well before 15:50 for any liquid
            # name, so this branch is the rare missing-data case.)
            return
        if rec.ref_price != INT64_PX_UNDEF:
            st.snap_ref_price = rec.ref_price * PX_SCALE
        st.snap_total_imbalance_qty = int(rec.total_imbalance_qty)
        st.snap_side = rec.side
        st.snap_bid = st.bid_px
        st.snap_ask = st.ask_px
        st.latched = True
        self.n_latched += 1

    def all_expected_latched(self) -> bool:
        if self.expected_symbols is None:
            return False
        for sym in self.expected_symbols:
            st = self.state.get(sym)
            if st is None or not st.latched:
                return False
        return True

    def on_event(self, rec) -> bool:
        """Returns True when the caller may stop the replay."""
        ts = _record_ts_ns(rec)
        # Wall-clock stop -- e.g. 15:58 NY 'imbalance freeze'.
        if ts is not None and ts >= self.stop_ts_ns:
            return True

        rtype = type(rec).__name__
        if rtype.startswith("BBO") or rtype in ("MBP1Msg", "CMBP1Msg", "CBBOMsg"):
            self.on_bbo(rec)
        elif rtype == "ImbalanceMsg":
            if ts is not None:
                self.on_imbalance(rec, ts)

        # If we know the expected universe and all of them have latched, stop.
        if self.expected_symbols and self.n_latched >= len(self.expected_symbols):
            if self.all_expected_latched():
                return True
        return False

    def snapshot(self) -> SnapshotResult:
        bid: dict[str, float] = {}
        ask: dict[str, float] = {}
        dq: dict[str, float] = {}
        side: dict[str, str] = {}
        refpx: dict[str, float] = {}
        tiq: dict[str, int] = {}
        for sym, st in self.state.items():
            if not st.latched:
                continue
            if st.snap_side not in ("A", "B"):
                continue
            if not (st.snap_bid > 0 and st.snap_ask > 0):
                continue
            bid[sym] = st.snap_bid
            ask[sym] = st.snap_ask
            side[sym] = st.snap_side
            sign = 1.0 if st.snap_side == "B" else -1.0
            dq[sym] = sign * st.snap_total_imbalance_qty * st.snap_ref_price
            refpx[sym] = st.snap_ref_price
            tiq[sym] = int(st.snap_total_imbalance_qty)
        return SnapshotResult(bid_px=bid, ask_px=ask, imb_dollar_qty=dq, side=side,
                              ref_price=refpx, total_imbalance_qty=tiq)


def replay_day_snapshot(
    bbo_files: list[Path],
    imbalance_files: list[Path],
    decision_ts_ns: int,
    stop_ts_ns: int,
    expected_symbols: set[str] | None = None,
    seed_bid_ask: dict[str, tuple[float, float]] | None = None,
) -> SnapshotResult:
    """Merge-replay the day's bbo-1s and imbalance DBN files in chronological
    order, dispatch records to a single Strategy, return the per-symbol latch
    snapshot.

    The replay runs until either (a) every symbol in ``expected_symbols`` has
    latched, or (b) the clock crosses ``stop_ts_ns`` (15:58 NY by default). In
    a live deployment the same Strategy.on_event callback would keep firing on
    every subsequent record.

    SMOKE-mode hook: ``seed_bid_ask`` lets the caller pre-populate each
    symbol's bid/ask before the replay starts (used when bbo-1s isn't on disk
    to avoid the batch download). When seeded, the replay only consumes the
    imbalance stream and we still exercise the strategy state machine."""

    iid_to_symbol: dict[int, str] = {}
    stores_bbo = [db.DBNStore.from_file(p) for p in bbo_files]
    stores_imb = [db.DBNStore.from_file(p) for p in imbalance_files]
    for store in stores_bbo + stores_imb:
        iid_to_symbol.update(_build_iid_to_symbol(store))

    strat = Strategy(iid_to_symbol, decision_ts_ns, stop_ts_ns,
                     expected_symbols=expected_symbols)

    if seed_bid_ask:
        for sym, (bid, ask) in seed_bid_ask.items():
            st = strat._get_state(sym)
            st.bid_px = bid
            st.ask_px = ask

    import heapq

    def keyed(store: db.DBNStore):
        for rec in store:
            ts = _record_ts_ns(rec)
            # Records without a valid clock (e.g. bbo-1s warmup snapshots with
            # ts_event == UINT64 sentinel and no real prior tick) are still
            # informational state -- emit them at t=0 so they update the book
            # before the decision-time check fires.
            yield (ts if ts is not None else 0, rec)

    streams = [keyed(s) for s in stores_bbo + stores_imb]
    merged = heapq.merge(*streams, key=lambda kv: kv[0])

    for _, rec in merged:
        if strat.on_event(rec):
            break

    return strat.snapshot()


def synthesize_bid_ask_from_bars(
    bars: pl.LazyFrame, day: date, half_spread_bps: float = SMOKE_HALF_SPREAD_BPS
) -> dict[str, tuple[float, float]]:
    """Build a synthetic {symbol: (bid_px, ask_px)} map at SNAP_TIME_NY from
    the OHLCV-1s close on the given day. SMOKE-mode only."""
    bps = half_spread_bps * 1e-4
    snap = (
        bars
        .with_columns(pl.col("ts_event").dt.date().alias("date"))
        .filter((pl.col("date") == day) & (pl.col("ts_event").dt.time() == SNAP_TIME_NY))
        .group_by("symbol")
        .agg(pl.col("close").last().alias("mid"))
        .filter(pl.col("mid") > 0)
        .collect(engine="streaming")
    )
    return {
        row["symbol"]: (row["mid"] * (1.0 - bps), row["mid"] * (1.0 + bps))
        for row in snap.iter_rows(named=True)
    }


# ---------------------------------------------------------------------------
# Trade booking
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    date: date
    symbol: str
    side_int: int       # +1 long, -1 short
    qty: int
    entry_px: float
    exit_px: float
    notional: float     # signed: +long, -short
    gross_pnl: float    # sign*qty*(exit_px - entry_px), before costs
    commission: float   # total round-trip commission (entry leg + exit leg)
    pnl: float          # net: gross_pnl - commission


@dataclass
class PanelRow:
    """One per (date, symbol) latched candidate -- includes rejects so the user
    can analyze the filters' selectivity post-hoc. Trade-side fields are NaN/None
    when the row was filtered out."""
    date: date
    symbol: str
    side: str                       # 'A' or 'B'
    bid_px: float
    ask_px: float
    ref_price: float
    total_imbalance_qty: int
    imb_dollar_qty: float
    session_dollar_volume: float
    qty_signal: float               # imb_dollar_qty / session_dollar_volume (nan if sdv<=0)
    auction_px: float               # nan if missing
    passed_dollar_filter: bool
    passed_qty_filter: bool
    traded: bool
    sign: int | None = None         # +1/-1 when traded, else None
    entry_px: float | None = None
    exit_px: float | None = None
    qty: int | None = None
    gross_pnl: float | None = None
    commission: float | None = None
    pnl: float | None = None


def _leg_commission(qty: int, leg_px: float, commission_per_share: float) -> float:
    """IBKR-style per-leg commission: per-share * qty, capped at 1% of leg notional."""
    leg_notional = qty * leg_px
    return min(commission_per_share * qty, COMMISSION_CAP_FRAC * leg_notional)


def book_trades(
    day: date,
    snapshot: SnapshotResult,
    session_dollar_volume: dict[str, float],
    auction_px: dict[str, float],
    capital: float,
    commission_per_share: float = DEFAULT_COMMISSION_PER_SHARE,
) -> tuple[list[Trade], list[PanelRow]]:
    """
    Filter the snapshot to symbols passing both thresholds
    (``|imb_dollar| > $500k`` and ``|imb_dollar / session_$vol| > 1%``),
    size EQUAL WEIGHT across the survivors, and book entry-at-half-spread /
    exit-at-auction trades.

    Direction follows the homework convention: trade WITH the imbalance.
    ``side == 'B'`` (buy imbalance, ``imb_dollar_qty > 0``) -> go LONG;
    ``side == 'A'`` (sell imbalance, ``imb_dollar_qty < 0``) -> go SHORT.

    Returns ``(trades, panel_rows)``. ``panel_rows`` has one entry per latched
    candidate -- including rejects -- so the caller can persist the full
    decision-time dataset for post-hoc analysis. Commission is applied per leg
    (entry fill + exit fill), IBKR-style: ``min(per_share * qty, 1% * leg_notional)``.
    """
    # Build raw per-symbol panel rows first (one per latched candidate, including
    # those that fail the side/bbo/sdv sanity checks).
    panel_rows: list[PanelRow] = []
    survivors: list[PanelRow] = []  # rows that pass BOTH filters and are tradeable

    for sym, dollar_qty in snapshot.imb_dollar_qty.items():
        side = snapshot.side.get(sym, "N")
        bid = snapshot.bid_px.get(sym, float("nan"))
        ask = snapshot.ask_px.get(sym, float("nan"))
        sdv = session_dollar_volume.get(sym, 0.0)
        ref_px = float(snapshot.ref_price.get(sym, float("nan")))
        total_imb_qty = int(snapshot.total_imbalance_qty.get(sym, 0))

        if side not in ("A", "B"):
            # Shouldn't happen because Strategy.snapshot already filters these
            # out, but guard anyway -- skip producing a panel row since we
            # don't have a valid side.
            continue
        if not (bid > 0 and ask > 0):
            continue

        qty_signal = (dollar_qty / sdv) if sdv > 0 else float("nan")
        auct = auction_px.get(sym, float("nan"))
        if auct is None:
            auct = float("nan")

        passed_dollar = abs(dollar_qty) > MIN_ABS_DOLLAR_IMBALANCE
        passed_qty = (sdv > 0) and (abs(qty_signal) > MIN_ABS_QTY_SIGNAL)

        row = PanelRow(
            date=day, symbol=sym, side=side,
            bid_px=float(bid), ask_px=float(ask),
            ref_price=ref_px,
            total_imbalance_qty=total_imb_qty,
            imb_dollar_qty=float(dollar_qty),
            session_dollar_volume=float(sdv),
            qty_signal=float(qty_signal) if np.isfinite(qty_signal) else float("nan"),
            auction_px=float(auct),
            passed_dollar_filter=bool(passed_dollar),
            passed_qty_filter=bool(passed_qty),
            traded=False,
        )
        panel_rows.append(row)

        # Tradeability gate: both filters AND a finite positive auction price.
        if passed_dollar and passed_qty and np.isfinite(auct) and auct > 0:
            survivors.append(row)

    trades: list[Trade] = []
    if not survivors:
        return trades, panel_rows

    # Equal weight (no dollar-neutrality constraint): every survivor gets the
    # same target gross notional. If every name is the same direction, the
    # book is net long or net short and that's OK -- the homework's Q5
    # strategy is explicitly unhedged.
    target_notional = capital / len(survivors)

    for row in survivors:
        sign = 1 if row.imb_dollar_qty > 0 else -1
        entry_px = row.ask_px if sign > 0 else row.bid_px   # pay the half-spread
        qty = int(np.floor(target_notional / entry_px))
        if qty <= 0:
            continue
        exit_px = row.auction_px
        gross_pnl = sign * qty * (exit_px - entry_px)
        commission = (
            _leg_commission(qty, entry_px, commission_per_share)
            + _leg_commission(qty, exit_px, commission_per_share)
        )
        net_pnl = gross_pnl - commission
        trades.append(Trade(
            date=day, symbol=row.symbol, side_int=sign, qty=qty,
            entry_px=entry_px, exit_px=exit_px,
            notional=sign * qty * entry_px,
            gross_pnl=gross_pnl, commission=commission, pnl=net_pnl,
        ))
        # Mutate the panel row in place with the booked trade fields.
        row.traded = True
        row.sign = sign
        row.entry_px = float(entry_px)
        row.exit_px = float(exit_px)
        row.qty = int(qty)
        row.gross_pnl = float(gross_pnl)
        row.commission = float(commission)
        row.pnl = float(net_pnl)

    return trades, panel_rows


# ---------------------------------------------------------------------------
# Reporting & plots
# ---------------------------------------------------------------------------

def summarize_and_plot(daily_pnl: pl.DataFrame, trades: pl.DataFrame, capital0: float) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    daily = daily_pnl.sort("date").with_columns(
        equity=capital0 + pl.col("pnl").cum_sum(),
    )
    equity = daily["equity"].to_numpy()
    pnl = daily["pnl"].to_numpy()
    dates = daily["date"].to_list()
    daily_ret = pnl / np.concatenate([[capital0], equity[:-1]])

    total_pnl = float(pnl.sum())
    total_return = equity[-1] / capital0 - 1.0
    n_days = len(daily)
    ann_return = (1.0 + np.mean(daily_ret)) ** 252 - 1.0
    sharpe = (np.mean(daily_ret) / np.std(daily_ret, ddof=1) * np.sqrt(252)
              if np.std(daily_ret, ddof=1) > 0 else float("nan"))
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1.0
    max_dd = float(drawdown.min())
    # CVaR-5%: mean of the worst 5% of daily returns
    k = max(1, int(np.ceil(0.05 * len(daily_ret))))
    cvar = float(np.mean(np.sort(daily_ret)[:k]))

    has_trades = not trades.is_empty()
    if has_trades:
        trade_pnl = trades["pnl"].to_numpy()
        trade_notional = np.abs(trades["notional"].to_numpy())
        trade_ret = np.where(trade_notional > 0, trade_pnl / trade_notional, 0.0)
        win_rate = float(np.mean(trade_pnl > 0))
        mean_abs_trade_ret = float(np.mean(np.abs(trade_ret)))
    else:
        trade_pnl = np.array([], dtype=float)
        trade_ret = np.array([], dtype=float)
        win_rate = float("nan")
        mean_abs_trade_ret = float("nan")

    print()
    print("=" * 70)
    print("Close-Auction Imbalance Strategy")
    print("=" * 70)
    print(f"Period:                 {dates[0]} -> {dates[-1]}")
    print(f"Starting capital:       ${capital0:,.2f}")
    print(f"Ending capital:         ${equity[-1]:,.2f}")
    print(f"Total PnL:              ${total_pnl:,.2f}")
    print(f"Total return:           {total_return:.4%}")
    print(f"Annualized return:      {ann_return:.4%}")
    print(f"Sharpe (daily, ann.):   {sharpe:.3f}")
    print(f"Max drawdown:           {max_dd:.4%}")
    print(f"CVaR-5% (daily):        {cvar:.4%}")
    print(f"N days:                 {n_days}")
    print(f"N trades:               {len(trades):,}")
    print(f"Per-trade win rate:     {win_rate:.4%}")
    print(f"Per-trade mean |ret|:   {mean_abs_trade_ret:.4%}  "
          f"({mean_abs_trade_ret * 1e4:.1f} bps)")
    print("=" * 70)

    # Equity curve
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, equity, marker="o", color="tab:blue")
    ax.axhline(capital0, color="black", linewidth=0.5, alpha=0.5)
    ax.set_title("Equity Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Capital ($)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "equity_curve.png", dpi=120)
    plt.close(fig)

    # Daily PnL bars
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["tab:green" if x >= 0 else "tab:red" for x in pnl]
    ax.bar(range(len(dates)), pnl, color=colors)
    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels([str(d) for d in dates], rotation=45, ha="right")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Daily PnL")
    ax.set_ylabel("PnL ($)")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "daily_pnl.png", dpi=120)
    plt.close(fig)

    # Drawdown curve
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(dates, drawdown, 0, color="tab:red", alpha=0.5)
    ax.set_title("Drawdown")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "drawdown.png", dpi=120)
    plt.close(fig)

    # Per-trade return histogram
    if has_trades:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(trade_ret * 1e4, bins=80, edgecolor="black", alpha=0.7)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_title("Per-Trade Returns")
        ax.set_xlabel("Return (bps)")
        ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "trade_returns.png", dpi=120)
        plt.close(fig)

    print(f"Plots written to: {PLOT_DIR}")


def save_results_parquet(
    daily_pnl: pl.DataFrame,
    trades: pl.DataFrame,
    panel: pl.DataFrame,
    out_dir: Path,
) -> tuple[Path, Path, Path]:
    """Persist the per-trade, per-day, and full per-(date,symbol) panel tables."""
    out_dir.mkdir(parents=True, exist_ok=True)
    trades_path = out_dir / "trades.parquet"
    daily_path = out_dir / "daily_pnl.parquet"
    panel_path = out_dir / "backtest_panel.parquet"

    trades_out = trades
    if not trades.is_empty():
        trades_out = trades.with_columns(
            pl.col("date").cast(pl.Date),
            pl.col("symbol").cast(pl.Utf8),
            pl.col("side").cast(pl.Int8),
            pl.col("qty").cast(pl.Int64),
            pl.col("entry_px").cast(pl.Float64),
            pl.col("exit_px").cast(pl.Float64),
            pl.col("notional").cast(pl.Float64),
            pl.col("gross_pnl").cast(pl.Float64),
            pl.col("commission").cast(pl.Float64),
            pl.col("pnl").cast(pl.Float64),
        )

    daily_out = daily_pnl.with_columns(
        pl.col("date").cast(pl.Date),
        pl.col("pnl").cast(pl.Float64),
        pl.col("n_trades").cast(pl.Int64),
        pl.col("capital_eod").cast(pl.Float64),
    )

    panel_out = panel
    if not panel.is_empty():
        panel_out = panel.with_columns(
            pl.col("date").cast(pl.Date),
            pl.col("symbol").cast(pl.Utf8),
            pl.col("side").cast(pl.Utf8),
            pl.col("bid_px").cast(pl.Float64),
            pl.col("ask_px").cast(pl.Float64),
            pl.col("ref_price").cast(pl.Float64),
            pl.col("total_imbalance_qty").cast(pl.Int64),
            pl.col("imb_dollar_qty").cast(pl.Float64),
            pl.col("session_dollar_volume").cast(pl.Float64),
            pl.col("qty_signal").cast(pl.Float64),
            pl.col("auction_px").cast(pl.Float64),
            pl.col("passed_dollar_filter").cast(pl.Boolean),
            pl.col("passed_qty_filter").cast(pl.Boolean),
            pl.col("traded").cast(pl.Boolean),
            pl.col("sign").cast(pl.Int8),
            pl.col("entry_px").cast(pl.Float64),
            pl.col("exit_px").cast(pl.Float64),
            pl.col("qty").cast(pl.Int64),
            pl.col("gross_pnl").cast(pl.Float64),
            pl.col("commission").cast(pl.Float64),
            pl.col("pnl").cast(pl.Float64),
        )

    trades_out.write_parquet(trades_path)
    daily_out.write_parquet(daily_path)
    panel_out.write_parquet(panel_path)
    return trades_path, daily_path, panel_path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

# Memoize the raw-file -> trading-date mapping per schema. Each raw bbo-1s file
# is ~2 GB, so opening every file once per (schema, day) to read its metadata
# would re-scan ~40 GB across the run. We build the date index once per schema
# and reuse it for every day.
_RAW_DAY_INDEX: dict[str, dict[date, list[Path]]] = {}


def _raw_files_by_day(adapter: DatabentoGenericAdapter, schema: str) -> dict[date, list[Path]]:
    """Build (and cache) a ``{trading_date: [raw DBN files]}`` index for a schema.

    The raw cache holds one file per trading day per cache key on XNAS.ITCH
    batch jobs. We read each file's metadata exactly once and bucket by date."""
    cached = _RAW_DAY_INDEX.get(schema)
    if cached is not None:
        return cached
    cache_key = adapter._generate_cache_key(
        adapter._normalize_tickers("ALL_SYMBOLS"),
        START_DATE, END_DATE, DATASET, schema,
    )
    index: dict[date, list[Path]] = {}
    for p in adapter._find_raw_path(cache_key):
        store = db.DBNStore.from_file(p)
        start_ns = store.metadata.start
        # Databento splits batch deliveries on UTC midnight. XNAS records all
        # land >= 07:25 NY (>= 11:25 UTC), so the file's UTC date equals the
        # NY trading date of every record inside -- we can compare directly.
        d = datetime.fromtimestamp(start_ns / 1e9, tz=timezone.utc).date()
        index.setdefault(d, []).append(p)
    _RAW_DAY_INDEX[schema] = index
    return index


def _files_for_day(adapter: DatabentoGenericAdapter, schema: str, day: date) -> list[Path]:
    """Return the raw DBN files for a given (schema, day) from the cached index."""
    return _raw_files_by_day(adapter, schema).get(day, [])


def _trading_days(bars: pl.LazyFrame) -> list[date]:
    """All NY trading dates present in the bars feed."""
    return sorted(
        bars
        .with_columns(pl.col("ts_event").dt.date().alias("date"))
        .select("date")
        .unique()
        .collect(engine="streaming")["date"]
        .to_list()
    )


def _session_dollar_volume_for_day(bars: pl.LazyFrame, day: date) -> dict[str, float]:
    """Sum of ``volume * close`` over today's session up to the snap cutoff."""
    df = (
        bars
        .with_columns(pl.col("ts_event").dt.date().alias("date"))
        .filter(
            pl.col("date") == day,
            pl.col("ts_event").dt.time().is_between(SESSION_OPEN_NY, SESSION_PRE_CUTOFF_NY),
        )
        .group_by("symbol")
        .agg((pl.col("volume").cast(pl.Float64) * pl.col("close")).sum().alias("sdv"))
        .collect(engine="streaming")
    )
    return {row["symbol"]: float(row["sdv"]) for row in df.iter_rows(named=True)}


def _auction_px_for_day(statistics: pl.LazyFrame, day: date) -> dict[str, float]:
    df = (
        statistics
        .with_columns(pl.col("ts_event").dt.date().alias("date"))
        .filter(pl.col("date") == day)
        .group_by("symbol")
        .agg(pl.col("price").last().alias("close_eod"))
        .collect(engine="streaming")
    )
    return {
        row["symbol"]: (float(row["close_eod"]) if row["close_eod"] is not None else float("nan"))
        for row in df.iter_rows(named=True)
    }


def _expected_symbols_for_day(
    imbalance: pl.LazyFrame, day: date
) -> set[str]:
    """Universe of symbols with at least one closing-auction imbalance message
    on ``day`` -- used as the 'all latched' early-stop signal in the replay."""
    df = (
        imbalance
        .with_columns(pl.col("ts_event").dt.date().alias("date"))
        .filter(pl.col("date") == day)
        .select("symbol")
        .unique()
        .collect(engine="streaming")
    )
    return set(df["symbol"].to_list())


async def main(
    commission: float = DEFAULT_COMMISSION_PER_SHARE,
    capital: float = STARTING_CAPITAL,
    smoke: bool | None = None,
) -> None:
    """Run the close-auction imbalance backtest.

    Parameters
    ----------
    commission : float
        Per-share commission (USD/share), capped at 1% of leg notional per leg.
    capital : float
        Starting capital used for equal-weight sizing.
    smoke : bool | None
        If True, force smoke mode (synthesize bid/ask from OHLCV-1s; skip bbo-1s
        fetch). If None, fall back to the ``SMOKE_TEST`` env var.
    """
    load_dotenv()
    api_key = os.getenv("DATABENTO-API-KEY")
    if not api_key:
        raise RuntimeError("DATABENTO-API-KEY not set in environment")

    smoke_mode = SMOKE_TEST if smoke is None else bool(smoke)

    adapter = DatabentoGenericAdapter(api_key, CACHE_DIR, verbose=True)

    imbalance_lf = await load_imbalance(adapter)
    bars_lf = await load_ohlcv_1s(adapter)
    statistics_lf = await load_statistics(adapter)

    if smoke_mode:
        print("[SMOKE] skipping bbo-1s batch fetch; bid/ask will be "
              "synthesized from the OHLCV-1s close")
    else:
        await adapter.get_data(
            tickers="ALL_SYMBOLS",
            start_date=START_DATE,
            end_date=END_DATE,
            dataset=DATASET,
            schema="bbo-1s",
            materialize_only=True,
        )

    all_days = _trading_days(bars_lf)

    capital0 = capital
    running_capital = capital
    all_trades: list[Trade] = []
    all_panel_rows: list[PanelRow] = []
    daily_pnl_rows: list[dict] = []

    for day in all_days:
        imb_files = _files_for_day(adapter, "Imbalance", day)
        if not imb_files:
            print(f"[{day}] missing raw imbalance DBN; skipping")
            continue
        if smoke_mode:
            bbo_files: list[Path] = []
            seed_bid_ask = synthesize_bid_ask_from_bars(bars_lf, day)
        else:
            bbo_files = _files_for_day(adapter, "bbo-1s", day)
            if not bbo_files:
                print(f"[{day}] missing raw bbo DBN; skipping")
                continue
            seed_bid_ask = None

        sdv_denom = _session_dollar_volume_for_day(bars_lf, day)
        auction_px = _auction_px_for_day(statistics_lf, day)
        expected = _expected_symbols_for_day(imbalance_lf, day)

        decision_ts_ns = _ny_time_to_utc_ns(day, SNAP_TIME_NY)
        stop_ts_ns = _ny_time_to_utc_ns(day, REPLAY_STOP_TIME_NY)
        snap = replay_day_snapshot(
            bbo_files, imb_files, decision_ts_ns, stop_ts_ns,
            expected_symbols=expected, seed_bid_ask=seed_bid_ask,
        )

        trades, panel_rows = book_trades(
            day=day,
            snapshot=snap,
            session_dollar_volume=sdv_denom,
            auction_px=auction_px,
            capital=running_capital,
            commission_per_share=commission,
        )
        day_pnl = sum(t.pnl for t in trades)
        running_capital += day_pnl
        all_trades.extend(trades)
        all_panel_rows.extend(panel_rows)
        daily_pnl_rows.append({"date": day, "pnl": day_pnl, "n_trades": len(trades),
                               "capital_eod": running_capital})
        print(f"[{day}] latched={len(snap.side):4d}  trades={len(trades):4d}  "
              f"day_pnl=${day_pnl:>12,.2f}  capital=${running_capital:>14,.2f}")

    if not daily_pnl_rows:
        print("No trades booked. Check filters / data alignment.")
        return

    daily_pnl_df = pl.DataFrame(daily_pnl_rows)
    trades_df = pl.DataFrame([{
        "date": t.date, "symbol": t.symbol, "side": t.side_int, "qty": t.qty,
        "entry_px": t.entry_px, "exit_px": t.exit_px,
        "notional": t.notional,
        "gross_pnl": t.gross_pnl, "commission": t.commission, "pnl": t.pnl,
    } for t in all_trades])

    # Panel: one row per latched (date, symbol). Use a schema so empty cols
    # (e.g. all-None sign/entry_px when nothing traded) keep correct dtypes.
    panel_schema = {
        "date": pl.Date, "symbol": pl.Utf8, "side": pl.Utf8,
        "bid_px": pl.Float64, "ask_px": pl.Float64, "ref_price": pl.Float64,
        "total_imbalance_qty": pl.Int64, "imb_dollar_qty": pl.Float64,
        "session_dollar_volume": pl.Float64, "qty_signal": pl.Float64,
        "auction_px": pl.Float64,
        "passed_dollar_filter": pl.Boolean, "passed_qty_filter": pl.Boolean,
        "traded": pl.Boolean,
        "sign": pl.Int8, "entry_px": pl.Float64, "exit_px": pl.Float64,
        "qty": pl.Int64, "gross_pnl": pl.Float64, "commission": pl.Float64,
        "pnl": pl.Float64,
    }
    panel_df = pl.DataFrame(
        [{
            "date": r.date, "symbol": r.symbol, "side": r.side,
            "bid_px": r.bid_px, "ask_px": r.ask_px, "ref_price": r.ref_price,
            "total_imbalance_qty": r.total_imbalance_qty,
            "imb_dollar_qty": r.imb_dollar_qty,
            "session_dollar_volume": r.session_dollar_volume,
            "qty_signal": r.qty_signal, "auction_px": r.auction_px,
            "passed_dollar_filter": r.passed_dollar_filter,
            "passed_qty_filter": r.passed_qty_filter,
            "traded": r.traded,
            "sign": r.sign, "entry_px": r.entry_px, "exit_px": r.exit_px,
            "qty": r.qty, "gross_pnl": r.gross_pnl,
            "commission": r.commission, "pnl": r.pnl,
        } for r in all_panel_rows],
        schema=panel_schema,
    )

    summarize_and_plot(daily_pnl_df, trades_df, capital0)
    trades_path, daily_path, panel_path = save_results_parquet(
        daily_pnl_df, trades_df, panel_df, RESULTS_DIR
    )
    print(f"Trades parquet:    {trades_path}")
    print(f"Daily PnL parquet: {daily_path}")
    print(f"Panel parquet:     {panel_path}")


if __name__ == "__main__":
    asyncio.run(main())
