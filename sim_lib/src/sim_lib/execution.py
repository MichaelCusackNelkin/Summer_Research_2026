"""
execution.py — Mid-price markout analysis for MBP-10 tick data.

Exposes two functions:
    - compute_markouts: sample ticks and measure forward mid-price moves
      at configurable horizons (nanoseconds, so μs → s is supported).
    - plot_markout_curves: plot mean markout (bps) vs horizon, binned by
      a signal column (``book_imbalance`` or ``obi_level0``).

All heavy calculations run in polars; matplotlib is used only to render
the final pre-aggregated curves.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from typing import Optional, Sequence

__all__ = [
    "compute_features",
    "compute_markouts",
    "markout_by_signal",
    "ExecutionSimulator",
    "compare_strategies",
    "plot_markout_curves",
    "plot_strategy_comparison",
    "plot_book_imbalance_vs_markout",
    "plot_impact_vs_alpha_decay",
    "plot_spread_and_imbalance_timeseries",
]


# One microsecond through ten seconds, log-spaced.
_DEFAULT_HORIZONS_NS: list[int] = [
    1_000, 10_000, 100_000,
    1_000_000, 10_000_000, 100_000_000,
    1_000_000_000, 10_000_000_000,
]

_LEVELS = range(10)


def compute_markouts(
    df: pl.DataFrame,
    horizons_ns: Optional[Sequence[int]] = None,
    sample_n: int = 50_000,
    seed: int = 42,
) -> pl.DataFrame:
    """Compute forward mid-price markouts at multiple horizons.

    For each sampled tick the markout at horizon *h* (ns) is the
    mid-price change from the current tick to the first tick whose
    timestamp is ≥ current_ts + *h*. If no such future tick exists
    within the instrument's data, the last observed mid is used.

    Parameters
    ----------
    df : pl.DataFrame
        MBP-10 polars frame. Must contain ``ts_event`` (datetime[ns]),
        ``bid_px_00``/``ask_px_00``, and size columns
        ``bid_sz_00``..``bid_sz_09`` / ``ask_sz_00``..``ask_sz_09``.
        If ``instrument_id`` is present, markouts are computed per
        instrument so the forward-join never crosses boundaries.
    horizons_ns : sequence of int, optional
        Forward horizons in nanoseconds. Defaults span 1 μs to 10 s.
    sample_n : int
        Total ticks to sample across all instruments.
    seed : int
        RNG seed.

    Returns
    -------
    pl.DataFrame
        One row per sampled tick with columns: ``ts_recv``, ``mid_price``,
        ``book_imbalance``, ``obi_level0``, ``instrument_id`` (if present),
        and ``markout_{h}ns`` / ``markout_bps_{h}ns`` for each horizon.
    """
    if horizons_ns is None:
        horizons_ns = _DEFAULT_HORIZONS_NS
    horizons_ns = [int(h) for h in horizons_ns]

    has_instrument = "instrument_id" in df.columns

    bid_sz_cols = [f"bid_sz_{lvl:02d}" for lvl in _LEVELS if f"bid_sz_{lvl:02d}" in df.columns]
    ask_sz_cols = [f"ask_sz_{lvl:02d}" for lvl in _LEVELS if f"ask_sz_{lvl:02d}" in df.columns]

    bid_full = pl.sum_horizontal([pl.col(c).cast(pl.Float64) for c in bid_sz_cols[:6]])
    ask_full = pl.sum_horizontal([pl.col(c).cast(pl.Float64) for c in ask_sz_cols[:6]])
    denom_full = bid_full + ask_full
    bid0 = pl.col("bid_sz_00").cast(pl.Float64)
    ask0 = pl.col("ask_sz_00").cast(pl.Float64)
    denom0 = bid0 + ask0

    enriched = df.with_columns(
        ((pl.col("bid_px_00") + pl.col("ask_px_00")) / 2.0).alias("mid_price"),
        pl.when(denom_full > 0)
          .then((bid_full - ask_full) / denom_full)
          .otherwise(None)
          .alias("book_imbalance"),
        pl.when(denom0 > 0)
          .then((bid0 - ask0) / denom0)
          .otherwise(None)
          .alias("obi_level0"),
    )

    sort_keys = ["instrument_id", "ts_event"] if has_instrument else ["ts_event"]
    enriched = enriched.sort(sort_keys)

    rng = np.random.default_rng(seed)
    if has_instrument:
        total_rows = enriched.height
        if total_rows == 0:
            return pl.DataFrame()
        parts = enriched.partition_by("instrument_id", as_dict=True, maintain_order=True)
        sampled_parts: list[pl.DataFrame] = []
        for _, gdf in parts.items():
            n = gdf.height
            if n == 0:
                continue
            share = n / total_rows
            k = min(n, max(1, int(round(sample_n * share))))
            group_seed = int(rng.integers(0, 2**31 - 1))
            sampled_parts.append(gdf.sample(n=k, seed=group_seed))
        if not sampled_parts:
            return pl.DataFrame()
        sampled = pl.concat(sampled_parts)
    else:
        n = enriched.height
        if n == 0:
            return pl.DataFrame()
        k = min(n, sample_n)
        sampled = enriched.sample(n=k, seed=seed)

    base_cols = ["ts_event", "mid_price", "book_imbalance", "obi_level0"]
    if has_instrument:
        base_cols = ["instrument_id"] + base_cols
    sampled = sampled.select(base_cols).sort(sort_keys).with_row_index("_rid")

    lookup_cols = (["instrument_id"] if has_instrument else []) + ["ts_event", "mid_price"]
    lookup = enriched.select(lookup_cols).rename({"mid_price": "_future_mid"})

    # Fallback when target_ts exceeds the last observed ts in a group:
    # join_asof(forward) returns null there, so we coalesce with the
    # instrument's final mid to match the original clip-to-last behavior.
    if has_instrument:
        last_mid = enriched.group_by("instrument_id").agg(
            pl.col("mid_price").last().alias("_last_mid")
        )
    else:
        last_mid_scalar = enriched["mid_price"].last()

    for h in horizons_ns:
        target = sampled.with_columns(
            (pl.col("ts_event") + pl.duration(nanoseconds=h)).alias("_target_ts")
        )
        target_sort = ["instrument_id", "_target_ts"] if has_instrument else ["_target_ts"]
        target_sorted = target.sort(target_sort)

        join_args = dict(
            left_on="_target_ts",
            right_on="ts_event",
            strategy="forward",
        )
        if has_instrument:
            join_args["by"] = "instrument_id"

        joined = target_sorted.join_asof(lookup, **join_args)

        if has_instrument:
            joined = (
                joined.join(last_mid, on="instrument_id", how="left")
                .with_columns(pl.col("_future_mid").fill_null(pl.col("_last_mid")))
                .drop("_last_mid")
            )
        else:
            joined = joined.with_columns(
                pl.col("_future_mid").fill_null(pl.lit(last_mid_scalar))
            )

        joined = joined.sort("_rid").select("_rid", "_future_mid")

        sampled = (
            sampled.join(joined, on="_rid", how="left")
            .with_columns((pl.col("_future_mid") - pl.col("mid_price")).alias(f"markout_{h}ns"))
            .with_columns(
                pl.when(pl.col("mid_price") > 0)
                  .then(pl.col(f"markout_{h}ns") / pl.col("mid_price") * 10_000)
                  .otherwise(None)
                  .alias(f"markout_bps_{h}ns")
            )
            .drop("_future_mid")
        )

    ts_tz = sampled.schema["ts_event"].time_zone
    if ts_tz is None:
        ts_expr = pl.col("ts_event").dt.replace_time_zone("UTC").dt.convert_time_zone("US/Eastern")
    else:
        ts_expr = pl.col("ts_event").dt.convert_time_zone("US/Eastern")

    result = sampled.with_columns(ts_expr.alias("ts_recv")).drop("_rid", "ts_event")

    front = ["ts_recv", "mid_price", "book_imbalance", "obi_level0"]
    if has_instrument:
        front.append("instrument_id")
    horizon_cols: list[str] = []
    for h in horizons_ns:
        horizon_cols.extend([f"markout_{h}ns", f"markout_bps_{h}ns"])
    return result.select(front + horizon_cols)


def plot_markout_curves(
    markouts_df: pl.DataFrame,
    signal_col: str = "book_imbalance",
    n_bins: int = 5,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot mean markout (bps) vs horizon, binned by a signal.

    Parameters
    ----------
    markouts_df : pl.DataFrame
        Output of :func:`compute_markouts`.
    signal_col : {"book_imbalance", "obi_level0"} or any numeric column
        Column to bin on. Defaults to ``"book_imbalance"``.
    n_bins : int
        Number of quantile bins.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If None, a new figure is created.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if signal_col not in markouts_df.columns:
        raise ValueError(
            f"signal_col={signal_col!r} not in markouts_df; "
            f"available: {markouts_df.columns}"
        )

    bps_cols = [c for c in markouts_df.columns if c.startswith("markout_bps_") and c.endswith("ns")]
    horizons = sorted(int(c[len("markout_bps_"):-2]) for c in bps_cols)
    ordered_cols = [f"markout_bps_{h}ns" for h in horizons]

    # Quantile breakpoints as absolute floats keeps bin labels numerically
    # sortable, and avoids relying on categorical ordering from qcut.
    qs = [i / n_bins for i in range(1, n_bins)]
    signal = markouts_df.get_column(signal_col).drop_nulls().drop_nans()
    breaks = sorted({float(signal.quantile(q)) for q in qs})

    binned = (
        markouts_df
        .with_columns(pl.col(signal_col).cut(breaks, include_breaks=False).alias("_bin"))
        .drop_nulls("_bin")
        .group_by("_bin", maintain_order=False)
        .agg(
            [pl.col(c).mean() for c in ordered_cols]
            + [pl.col(signal_col).min().alias("_lo")]
        )
        .sort("_lo")
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    cmap = plt.cm.RdYlGn
    n_rows = binned.height
    denom = max(n_rows - 1, 1)
    labels = binned.get_column("_bin").cast(pl.String).to_list()
    values = binned.select(ordered_cols).to_numpy()
    for i, label in enumerate(labels):
        ax.plot(
            horizons,
            values[i],
            linewidth=2,
            markersize=4,
            label=label,
            color=cmap(i / denom),
        )

    ax.set_xscale("symlog", linthresh=5_000)
    ax.set_xlim(left=0)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Markout (mils)")
    ax.set_title(f"Markout Curves by {signal_col}")
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.grid(True, which="major", alpha=0.5)
    ax.grid(True, which="minor", alpha=0.25, linestyle=":")
    ax.legend(title=signal_col, fontsize=8, loc="best")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 1. Data Ingestion / Feature Engineering
# ---------------------------------------------------------------------------

def compute_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add microstructure feature columns to the raw MBP-10 DataFrame.

    New columns added:
        mid_price, spread, spread_bps, micro_price, trade_flow,
        cumulative_trade_flow, obi_level0, volume_imbalance.

    Parameters
    ----------
    df : pl.DataFrame
        Raw MBP-10 DataFrame from DataBentoMBP10Adapter.

    Returns
    -------
    pl.DataFrame
        The same DataFrame with new columns appended (existing columns
        are preserved).
    """
    bid0 = pl.col("bid_px_00")
    ask0 = pl.col("ask_px_00")
    bid_sz0 = pl.col("bid_sz_00").cast(pl.Float64)
    ask_sz0 = pl.col("ask_sz_00").cast(pl.Float64)
    total_sz = bid_sz0 + ask_sz0

    mid_price = ((bid0 + ask0) / 2.0).alias("mid_price")
    spread = (ask0 - bid0).alias("spread")
    spread_bps = (pl.col("spread") / pl.col("mid_price") * 10_000).alias("spread_bps")
    micro_price = (
        pl.when(total_sz > 0)
          .then((bid0 * ask_sz0 + ask0 * bid_sz0) / total_sz)
          .otherwise(pl.col("mid_price"))
          .alias("micro_price")
    )

    # Signed trade flow (positive = buyer-initiated, negative = seller-initiated)
    is_trade = pl.col("action") == "T"
    size_f = pl.col("size").cast(pl.Float64)
    trade_flow = (
        pl.when(is_trade & (pl.col("price") >= ask0))
          .then(size_f)
          .when(is_trade & (pl.col("price") <= bid0))
          .then(-size_f)
          .otherwise(0.0)
          .alias("trade_flow")
    )

    obi_denom = bid_sz0 + ask_sz0
    obi_level0 = (
        pl.when(obi_denom > 0)
          .then((bid_sz0 - ask_sz0) / obi_denom)
          .otherwise(0.0)
          .alias("obi_level0")
    )

    # First pass: add mid_price (needed for spread_bps / micro_price fallback)
    out = df.with_columns(mid_price)
    # Second pass: add spread + dependent columns and trade_flow + obi
    out = out.with_columns(spread, micro_price, trade_flow, obi_level0)
    out = out.with_columns(spread_bps)

    # Cumulative + rolling features over the trade_flow column
    rolling_flow = pl.col("trade_flow").rolling_sum(window_size=1000, min_samples=1)
    rolling_abs = pl.col("trade_flow").abs().rolling_sum(window_size=1000, min_samples=1)
    out = out.with_columns(
        pl.col("trade_flow").cum_sum().alias("cumulative_trade_flow"),
        pl.when(rolling_abs > 0)
          .then(rolling_flow / rolling_abs)
          .otherwise(None)
          .alias("volume_imbalance"),
    )

    return out


# ---------------------------------------------------------------------------
# 2. Markout Analysis
# ---------------------------------------------------------------------------

_DEFAULT_HORIZONS_MS = [100, 500, 1000, 5000, 10_000, 30_000, 60_000]


def markout_by_signal(
    markouts_df: pl.DataFrame,
    signal_col: str,
    n_bins: int = 10,
) -> pl.DataFrame:
    """Bin rows by a signal column and compute mean markouts per bin.

    Parameters
    ----------
    markouts_df : pl.DataFrame
        Output of :func:`compute_markouts`.
    signal_col : str
        Column to bin on (e.g. ``"book_imbalance"``, ``"obi_level0"``).
    n_bins : int
        Number of quantile bins.

    Returns
    -------
    pl.DataFrame
        One row per bin with a ``bin`` label column and one column per
        horizon (bps markout).
    """
    markout_cols = [c for c in markouts_df.columns if c.startswith("markout_bps_")]
    binned = (
        markouts_df
        .with_columns(
            pl.col(signal_col).qcut(n_bins, allow_duplicates=True).alias("bin"),
            pl.col(signal_col).alias("_signal_sort"),
        )
        .drop_nulls("bin")
        .group_by("bin", maintain_order=False)
        .agg(
            [pl.col(c).mean() for c in markout_cols]
            + [pl.col("_signal_sort").min().alias("_lo")]
        )
        .sort("_lo")
        .drop("_lo")
    )
    return binned


# ---------------------------------------------------------------------------
# 3. Execution Strategies
# ---------------------------------------------------------------------------

class ExecutionSimulator:
    """Simulate execution of a parent buy order using different strategies.

    The simulator operates on a feature-enriched DataFrame and evaluates
    execution quality via arrival cost in basis points.

    Parameters
    ----------
    df : pl.DataFrame
        Feature-enriched DataFrame (output of :func:`compute_features`).
        Must contain a ``ts_event`` column of dtype ``datetime[ns]``.
    parent_qty : int
        Total shares to purchase (default 10 000).
    horizon_seconds : float
        Execution horizon in seconds (default 300 = 5 minutes).
    """

    def __init__(
        self,
        df: pl.DataFrame,
        parent_qty: int = 10_000,
        horizon_seconds: float = 300.0,
    ):
        self.df = df
        self.parent_qty = parent_qty
        self.horizon_seconds = horizon_seconds

        # Pre-compute convenience arrays
        self.ts_ns = df["ts_event"].cast(pl.Int64).to_numpy()
        self.horizon_ns = int(horizon_seconds * 1e9)

    # ---- internal helpers ------------------------------------------------

    def _window(self, start_idx: int) -> pl.DataFrame:
        """Return the sub-DataFrame within the execution horizon."""
        start_ns = self.ts_ns[start_idx]
        end_ns = start_ns + self.horizon_ns
        end_idx = int(np.searchsorted(self.ts_ns, end_ns, side="right"))
        return self.df.slice(start_idx, end_idx - start_idx)

    @staticmethod
    def _result(fills: list[dict], arrival_mid: float, parent_qty: int) -> dict:
        """Aggregate a list of fills into a result summary."""
        if not fills:
            return {
                "avg_price": np.nan,
                "arrival_cost_bps": np.nan,
                "total_shares_filled": 0,
                "fill_rate": 0.0,
                "fills": fills,
            }
        total_qty = sum(f["qty"] for f in fills)
        avg_px = sum(f["qty"] * f["price"] for f in fills) / total_qty if total_qty > 0 else np.nan
        cost_bps = (avg_px - arrival_mid) / arrival_mid * 10_000 if arrival_mid > 0 else np.nan
        return {
            "avg_price": avg_px,
            "arrival_cost_bps": cost_bps,
            "total_shares_filled": total_qty,
            "fill_rate": total_qty / parent_qty,
            "fills": fills,
        }

    # ---- strategies ------------------------------------------------------

    def twap(self, n_slices: int = 20, start_idx: int = 0) -> dict:
        """Time-Weighted Average Price execution.

        Splits the horizon into *n_slices* equal intervals and executes
        ``parent_qty / n_slices`` shares at the prevailing ask at the
        start of each interval.

        Parameters
        ----------
        n_slices : int
            Number of time slices.
        start_idx : int
            Index into ``self.df`` at which execution begins.

        Returns
        -------
        dict
            Execution result with avg_price, arrival_cost_bps, etc.
        """
        window = self._window(start_idx)
        if window.height == 0:
            return self._result([], np.nan, self.parent_qty)

        ts_col = window["ts_event"]
        arrival_mid = window["mid_price"][0]
        ts_ns_w = ts_col.cast(pl.Int64).to_numpy()
        start_ns = ts_ns_w[0]
        slice_ns = self.horizon_ns / n_slices
        qty_per_slice = self.parent_qty / n_slices

        fills: list[dict] = []
        remaining = self.parent_qty

        for i in range(n_slices):
            if remaining <= 0:
                break
            target_ns = start_ns + int(i * slice_ns)
            loc = int(np.searchsorted(ts_ns_w, target_ns, side="left"))
            if loc >= window.height:
                break
            row = window.row(loc, named=True)
            qty = min(qty_per_slice, remaining)
            fills.append({"ts": ts_col[loc], "price": row["ask_px_00"], "qty": qty})
            remaining -= qty

        return self._result(fills, arrival_mid, self.parent_qty)

    def vwap(self, n_slices: int = 20, start_idx: int = 0) -> dict:
        """Volume-Weighted Average Price execution.

        Splits the horizon into *n_slices* intervals and allocates
        shares proportionally to observed trade volume in each interval.

        Parameters
        ----------
        n_slices : int
            Number of time slices.
        start_idx : int
            Index into ``self.df`` at which execution begins.

        Returns
        -------
        dict
            Execution result.
        """
        window = self._window(start_idx)
        if window.height == 0:
            return self._result([], np.nan, self.parent_qty)

        ts_col = window["ts_event"]
        arrival_mid = window["mid_price"][0]
        ts_ns_w = ts_col.cast(pl.Int64).to_numpy()
        start_ns = ts_ns_w[0]
        slice_ns = self.horizon_ns / n_slices

        # Compute volume per slice
        trades = window.filter(pl.col("action") == "T")
        slice_volumes = np.zeros(n_slices)
        if trades.height > 0:
            trade_ts = trades["ts_event"].cast(pl.Int64).to_numpy()
            trade_sizes = trades["size"].cast(pl.Float64).to_numpy()
            slice_indices = np.clip(
                ((trade_ts - start_ns) / slice_ns).astype(int), 0, n_slices - 1
            )
            np.add.at(slice_volumes, slice_indices, trade_sizes)

        total_volume = slice_volumes.sum()
        if total_volume == 0:
            # Fall back to uniform
            weights = np.ones(n_slices) / n_slices
        else:
            weights = slice_volumes / total_volume

        fills: list[dict] = []
        remaining = self.parent_qty

        for i in range(n_slices):
            if remaining <= 0:
                break
            target_ns = start_ns + int(i * slice_ns)
            loc = int(np.searchsorted(ts_ns_w, target_ns, side="left"))
            if loc >= window.height:
                break
            row = window.row(loc, named=True)
            qty = min(round(weights[i] * self.parent_qty), remaining)
            if qty <= 0:
                continue
            fills.append({"ts": ts_col[loc], "price": row["ask_px_00"], "qty": qty})
            remaining -= qty

        # Allocate any rounding remainder to the last slice
        if remaining > 0 and fills:
            fills[-1]["qty"] += remaining

        return self._result(fills, arrival_mid, self.parent_qty)

    def obi_adaptive(
        self,
        n_slices: int = 20,
        aggression_threshold: float = 0.3,
        start_idx: int = 0,
    ) -> dict:
        """Order Book Imbalance adaptive execution.

        Uses ``book_imbalance`` to decide aggressiveness in each slice:
        - OBI > threshold  -> aggressive (buy at ask immediately)
        - OBI < -threshold -> passive (limit at bid; fills only if a
          trade prints at or below the bid in that interval)
        - otherwise        -> execute at mid

        Parameters
        ----------
        n_slices : int
            Number of time slices.
        aggression_threshold : float
            OBI threshold for switching between aggressive / passive.
        start_idx : int
            Index into ``self.df`` at which execution begins.

        Returns
        -------
        dict
            Execution result.
        """
        window = self._window(start_idx)
        if window.height == 0:
            return self._result([], np.nan, self.parent_qty)

        ts_col = window["ts_event"]
        arrival_mid = window["mid_price"][0]
        ts_ns_w = ts_col.cast(pl.Int64).to_numpy()
        start_ns = ts_ns_w[0]
        slice_ns = self.horizon_ns / n_slices
        qty_per_slice = self.parent_qty / n_slices

        fills: list[dict] = []
        remaining = self.parent_qty

        for i in range(n_slices):
            if remaining <= 0:
                break
            slice_start_ns = start_ns + int(i * slice_ns)
            slice_end_ns = start_ns + int((i + 1) * slice_ns)

            loc_start = int(np.searchsorted(ts_ns_w, slice_start_ns, side="left"))
            loc_end = int(np.searchsorted(ts_ns_w, slice_end_ns, side="left"))
            if loc_start >= window.height:
                break

            row = window.row(loc_start, named=True)
            obi = row.get("book_imbalance", 0.0) or 0.0
            qty = min(qty_per_slice, remaining)

            if obi > aggression_threshold:
                # Aggressive: buy at ask
                fills.append({
                    "ts": ts_col[loc_start],
                    "price": row["ask_px_00"],
                    "qty": qty,
                })
                remaining -= qty

            elif obi < -aggression_threshold:
                # Passive: limit at bid — check if any trade in the
                # interval printed at or below our bid price
                bid_px = row["bid_px_00"]
                slice_df = window.slice(loc_start, loc_end - loc_start)
                slice_trades = slice_df.filter(
                    (pl.col("action") == "T") & (pl.col("price") <= bid_px)
                )
                if slice_trades.height > 0:
                    fills.append({
                        "ts": slice_trades["ts_event"][0],
                        "price": bid_px,
                        "qty": qty,
                    })
                    remaining -= qty
                # else: no fill this slice (passive order not executed)

            else:
                # Neutral: execute at mid
                fills.append({
                    "ts": ts_col[loc_start],
                    "price": row["mid_price"],
                    "qty": qty,
                })
                remaining -= qty

        return self._result(fills, arrival_mid, self.parent_qty)

    def immediate(self, start_idx: int = 0) -> dict:
        """Execute the entire order immediately at the ask.

        Maximum market impact, zero alpha decay. Serves as a baseline.

        Parameters
        ----------
        start_idx : int
            Index into ``self.df`` at which execution begins.

        Returns
        -------
        dict
            Execution result.
        """
        if start_idx >= self.df.height:
            return self._result([], np.nan, self.parent_qty)

        row = self.df.row(start_idx, named=True)
        arrival_mid = row["mid_price"]
        fills = [{
            "ts": self.df["ts_event"][start_idx],
            "price": row["ask_px_00"],
            "qty": self.parent_qty,
        }]
        return self._result(fills, arrival_mid, self.parent_qty)


def compare_strategies(
    simulator: ExecutionSimulator,
    n_simulations: int = 50,
    seed: int = 42,
) -> pl.DataFrame:
    """Run each execution strategy from random starting points.

    Parameters
    ----------
    simulator : ExecutionSimulator
        A configured simulator instance.
    n_simulations : int
        Number of random starting points.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pl.DataFrame
        Columns: strategy, arrival_cost_bps, fill_rate.
        One row per simulation per strategy.
    """
    rng = np.random.default_rng(seed)
    n = simulator.df.height
    # Ensure there is enough room for the full execution horizon
    max_start = int(np.searchsorted(
        simulator.ts_ns,
        simulator.ts_ns[-1] - simulator.horizon_ns,
        side="right",
    ))
    max_start = max(1, min(max_start, n - 1))
    start_indices = rng.choice(max_start, size=n_simulations, replace=True)

    strategies = {
        "immediate": lambda idx: simulator.immediate(start_idx=idx),
        "twap": lambda idx: simulator.twap(start_idx=idx),
        "vwap": lambda idx: simulator.vwap(start_idx=idx),
        "obi_adaptive": lambda idx: simulator.obi_adaptive(start_idx=idx),
    }

    rows: list[dict] = []
    for idx in start_indices:
        for name, fn in strategies.items():
            res = fn(int(idx))
            rows.append({
                "strategy": name,
                "arrival_cost_bps": res["arrival_cost_bps"],
                "fill_rate": res["fill_rate"],
            })

    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Plotting Functions
# ---------------------------------------------------------------------------

def plot_strategy_comparison(results_df: pl.DataFrame) -> plt.Figure:
    """Box plot of arrival cost (bps) for each execution strategy.

    Parameters
    ----------
    results_df : pl.DataFrame
        Output of :func:`compare_strategies`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    strategies = results_df["strategy"].unique().to_list()
    data = [
        results_df.filter(pl.col("strategy") == s)["arrival_cost_bps"]
        .drop_nulls()
        .drop_nans()
        .to_numpy()
        for s in strategies
    ]
    bp = ax.boxplot(data, labels=strategies, patch_artist=True, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="red", markersize=7))
    colors = plt.cm.Set2(np.linspace(0, 1, len(strategies)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    ax.set_ylabel("Arrival Cost (bps)")
    ax.set_title("Execution Strategy Cost Comparison")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def plot_book_imbalance_vs_markout(
    markouts_df: pl.DataFrame,
    horizon_ns: int = 100_000,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Scatter / binned-line plot of OBI level 0 vs markout.

    Parameters
    ----------
    markouts_df : pl.DataFrame
        Output of :func:`compute_markouts`.
    horizon_ns : int
        Horizon (nanoseconds) to use for the markout column.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If *None*, a new figure is created.

    Returns
    -------
    matplotlib.figure.Figure
    """
    horizon_ns = int(horizon_ns)
    col = f"markout_bps_{horizon_ns}ns"
    if col not in markouts_df.columns:
        available = []
        for c in markouts_df.columns:
            if c.startswith("markout_bps_") and c.endswith("ns"):
                try:
                    available.append(int(c[len("markout_bps_"):-len("ns")]))
                except ValueError:
                    continue
        if not available:
            raise ValueError("No markout_bps_*ns columns found in markouts_df")
        nearest = min(available, key=lambda n: abs(n - horizon_ns))
        print(
            f"Warning: column '{col}' not found; using nearest available "
            f"horizon {nearest}ns instead."
        )
        horizon_ns = nearest
        col = f"markout_bps_{horizon_ns}ns"
    tmp = (
        markouts_df
        .select("obi_level0", col)
        .drop_nulls()
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    # Binned means for cleaner visualization
    binned = (
        tmp
        .with_columns(pl.col("obi_level0").qcut(50, allow_duplicates=True).alias("bin"))
        .drop_nulls("bin")
        .group_by("bin", maintain_order=False)
        .agg(
            pl.col("obi_level0").mean().alias("x"),
            pl.col(col).mean().alias("y"),
        )
        .sort("x")
    )
    x = binned["x"].to_numpy()
    y = binned["y"].to_numpy()
    ax.scatter(x, y, alpha=0.7, s=30, zorder=3)

    # Linear fit
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() > 1:
        coeffs = np.polyfit(x[mask], y[mask], 1)
        fit_x = np.linspace(x[mask].min(), x[mask].max(), 100)
        ax.plot(fit_x, np.polyval(coeffs, fit_x), "r--", linewidth=2,
                label=f"fit: {coeffs[0]:.2f}x + {coeffs[1]:.2f}")
        ax.legend()

    ax.set_xlabel("OBI Level 0")
    ax.set_ylabel(f"Markout (bps) @ {horizon_ns / 1e3 }us")
    ax.set_title(f"OBI Level 0 vs {horizon_ns / 1e3}us Markout")
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_impact_vs_alpha_decay(results_df: pl.DataFrame) -> plt.Figure:
    """Plot fill rate vs arrival cost for each strategy.

    Visualizes the market-impact / alpha-decay tradeoff.

    Parameters
    ----------
    results_df : pl.DataFrame
        Output of :func:`compare_strategies`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    summary = (
        results_df
        .group_by("strategy", maintain_order=True)
        .agg(
            pl.col("fill_rate").mean().alias("fill_rate"),
            pl.col("arrival_cost_bps").mean().alias("arrival_cost_bps"),
        )
    )

    fig, ax = plt.subplots(figsize=(8,6))
    n = summary.height
    colors = plt.cm.Set1(np.linspace(0, 1, n))
    names = summary["strategy"].to_list()
    fill_rates = summary["fill_rate"].to_numpy()
    costs = summary["arrival_cost_bps"].to_numpy()
    for name, fr, cost, color in zip(names, fill_rates, costs, colors):
        ax.scatter(fr, cost, s=120, color=color, zorder=3)
        ax.annotate(
            name,
            (fr, cost),
            textcoords="offset points",
            xytext=(10, 5),
            fontsize=11,
            fontweight="bold",
        )

    ax.set_xlabel("Fill Rate")
    ax.set_ylabel("Arrival Cost (bps)")
    ax.set_title("Arrival Cost vs Fill Rate (means)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_spread_and_imbalance_timeseries(
    df: pl.DataFrame,
    start_time: Optional[str] = None,
    duration_minutes: float = 5.0,
) -> plt.Figure:
    """Two-panel plot of spread (bps) and book imbalance over a short window.

    Parameters
    ----------
    df : pl.DataFrame
        Feature-enriched DataFrame. Must contain ``ts_event``,
        ``spread_bps``, and ``book_imbalance`` columns.
    start_time : str, optional
        Starting timestamp (ISO 8601 string). If *None*, a random
        window is selected.
    duration_minutes : float
        Window length in minutes.

    Returns
    -------
    matplotlib.figure.Figure
    """
    duration_ns = int(duration_minutes * 60 * 1e9)
    ts_ns = df["ts_event"].cast(pl.Int64).to_numpy()

    if start_time is None:
        max_start = int(np.searchsorted(ts_ns, ts_ns[-1] - duration_ns, side="right"))
        max_start = max(1, max_start)
        rng = np.random.default_rng()
        start_idx = int(rng.integers(0, max_start))
    else:
        # Parse via polars to a single ns-resolution timestamp.
        parsed = (
            pl.Series([start_time])
            .str.to_datetime(time_unit="ns", strict=False)
        )
        ts_dtype = df.schema["ts_event"]
        target_tz = ts_dtype.time_zone if isinstance(ts_dtype, pl.Datetime) else None
        if parsed.dtype == pl.Datetime and parsed.dtype.time_zone is None:
            tz = target_tz or "US/Eastern"
            parsed = parsed.dt.replace_time_zone(tz)
        if target_tz is not None:
            parsed = parsed.dt.convert_time_zone(target_tz)
        start_ns_val = int(parsed.cast(pl.Int64)[0])
        start_idx = int(np.searchsorted(ts_ns, start_ns_val, side="left"))

    end_ns = ts_ns[start_idx] + duration_ns
    end_idx = int(np.searchsorted(ts_ns, end_ns, side="right"))
    window = df.slice(start_idx, end_idx - start_idx)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    x = window["ts_event"].to_list()
    ax1.plot(x, window["spread_bps"].to_list(), linewidth=0.5, alpha=0.8)
    ax1.set_ylabel("Spread (bps)")
    ax1.set_title("Spread and Book Imbalance")
    ax1.grid(True, alpha=0.3)

    ax2.plot(x, window["book_imbalance"].to_list(), linewidth=0.5, alpha=0.8, color="tab:orange")
    ax2.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax2.set_ylabel("Book Imbalance")
    ax2.set_xlabel("Time")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig
