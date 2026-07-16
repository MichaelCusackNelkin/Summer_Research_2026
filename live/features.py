"""Feature engineering shared between the live orchestrator and the offline
retrain script.

The forecast model was trained on features engineered from three Databento
schemas (``imbalance``, ``tbbo``, ``statistics``) over a research window in
``forecast/forecasting.ipynb``. The live pipeline needs to compute the SAME
features from the live tape at 15:50 so the persisted sklearn pipeline
produces meaningful predictions. Rather than duplicating the polars pipeline
across two scripts, we express the core computations as pure functions that
take polars ``DataFrame``s / ``LazyFrame``s and return frames.

The optimizer also needs a handful of PER-DAY inputs beyond the alpha
features — ``entry_price``, ``spread_return``, ``ADV``, ``vol`` — so those
are computed here too and joined into the same live feature frame.

Key design decision: the feature frame column names EXACTLY match what the
sklearn Pipeline expects (the notebook's ``features`` list). Renaming would
silently break predictions.
"""
from __future__ import annotations

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Column contract — importing these instead of stringly-typed constants
# forces call sites to stay in sync with the model artifact.
# ---------------------------------------------------------------------------

ALPHA_FEATURES: list[str] = [
    "norm_rank_net_imb",
    "recent_return_xm",
    "auction_dollar_volume_ratio_xm",
    "norm_rank_net_x_recent_xm",
    "norm_rank_net_x_auction",
]
"""Column names the persisted forecast pipeline consumes, in order."""

TARGET_COLUMN = "close_return_xm"
"""Regression target — only present at retrain time."""

OPTIMIZER_INPUT_COLUMNS: list[str] = [
    "entry_price",   # $ — used to convert dollar positions -> shares
    "spread_return", # return-space half-spread proxy for the cost model
    "ADV",           # causal $-volume — impact denominator
    "vol",           # causal daily return vol — impact numerator
]
"""Non-feature columns the ``optimizer.py`` classes read off each day-frame."""


# ---------------------------------------------------------------------------
# Aggregators
# ---------------------------------------------------------------------------

def compute_close_trades(
    tbbo: pl.LazyFrame,
    entry_window_start: str = "15:49:55",
    entry_window_end: str = "15:50:00",
) -> pl.LazyFrame:
    """Per-(date, symbol) entry price and half-spread from the last few
    seconds before the 15:50 print.

    Matches ``optimizer/optimal_portfolio.ipynb`` verbatim (last mid in the
    window, mean spread as spread_return). We take the LAST mid because a
    single blip can knock the mean around; the auction cross is done off the
    last-touch anyway.
    """
    from datetime import time as _time
    hs, ms, ss = [int(x) for x in entry_window_start.split(":")]
    he, me, se = [int(x) for x in entry_window_end.split(":")]
    return (
        tbbo
        .with_columns(
            pl.col("ts_event").dt.date().alias("date"),
            ((pl.col("ask_px_00") + pl.col("bid_px_00")) / 2).alias("mid"),
        )
        .with_columns(
            ((pl.col("ask_px_00") - pl.col("bid_px_00")) / pl.col("mid"))
            .alias("spread_return")
        )
        .filter(
            pl.col("ts_event").dt.time().is_between(
                pl.time(hs, ms, ss), pl.time(he, me, se)
            )
        )
        .group_by(["date", "symbol"], maintain_order=True)
        .agg(
            pl.col("mid").last(),
            pl.col("spread_return").mean(),
        )
        .rename({"mid": "entry_price"})
    )


def compute_recent_return(
    tbbo: pl.LazyFrame,
    start: str = "15:40:00",
    end: str = "15:49:59",
) -> pl.LazyFrame:
    """Per-(date, symbol) mid-price return over the pre-print window."""
    hs, ms, ss = [int(x) for x in start.split(":")]
    he, me, se = [int(x) for x in end.split(":")]
    return (
        tbbo
        .with_columns(
            pl.col("ts_event").dt.date().alias("date"),
            ((pl.col("ask_px_00") + pl.col("bid_px_00")) / 2).alias("mid"),
        )
        .filter(
            pl.col("ts_event").dt.time().is_between(
                pl.time(hs, ms, ss), pl.time(he, me, se)
            )
        )
        .group_by(["date", "symbol"], maintain_order=True)
        .agg(
            ((pl.col("mid").last() - pl.col("mid").first()) / pl.col("mid").first())
            .alias("recent_return"),
        )
    )


def compute_avg_daily_dollar_volume(
    tbbo: pl.LazyFrame,
    ewm_span: int = 10,
) -> pl.LazyFrame:
    """Causal per-(date, symbol) EWMA $-volume — shifted 1 day so today's
    value only sees data through yesterday. Same recipe as the notebook.
    """
    return (
        tbbo
        .with_columns(pl.col("ts_event").dt.date().alias("date"))
        .filter(pl.col("ts_event").dt.time().is_between(pl.time(9, 30), pl.time(16, 0)))
        .group_by(["date", "symbol"])
        .agg((pl.col("size") * pl.col("price")).sum().alias("day_volume"))
        .sort(["date", "symbol"])
        .with_columns(
            pl.col("day_volume")
            .ewm_mean(span=ewm_span, adjust=True)
            .shift(1)
            .over("symbol")
            .alias("ADV")
        )
        .drop("day_volume")
    )


def compute_daily_vol(
    tbbo: pl.LazyFrame,
    ewm_span: int = 10,
) -> pl.LazyFrame:
    """Causal per-(date, symbol) daily return volatility. Same recipe as the
    optimizer notebook — EWMA(day_ret^2) shifted 1 day, then sqrt.
    """
    return (
        tbbo
        .with_columns(
            pl.col("ts_event").dt.date().alias("date"),
            ((pl.col("ask_px_00") + pl.col("bid_px_00")) / 2).alias("mid"),
        )
        .filter(pl.col("ts_event").dt.time().is_between(pl.time(9, 30), pl.time(16, 0)))
        .group_by(["date", "symbol"], maintain_order=True)
        .agg(pl.col("mid").last().alias("close"))
        .sort(["date", "symbol"])
        .with_columns(pl.col("close").pct_change().over("symbol").alias("day_ret"))
        .with_columns(
            pl.col("day_ret")
            .pow(2)
            .ewm_mean(span=ewm_span, adjust=True)
            .shift(1)
            .over("symbol")
            .sqrt()
            .alias("vol")
        )
        .select(["date", "symbol", "vol"])
    )


def compute_auction_prices(df_stats: pl.LazyFrame) -> pl.LazyFrame:
    """Per-(date, symbol) official closing-auction price from statistics.

    Only present at retrain time — the auction price is the regression
    target, so the live orchestrator does not need this.
    """
    return (
        df_stats
        .with_columns(pl.col("ts_event").dt.date().alias("date"))
        .select("date", "symbol", "price")
        .rename({"price": "auction_price"})
    )


def compute_imbalance_print(
    df_imbalance: pl.LazyFrame,
    window_start: str = "15:50:00",
    window_end: str = "15:50:05",
) -> pl.LazyFrame:
    """Per-(date, symbol) close-imbalance print — the first imbalance record
    at 15:50. The window is 5 seconds wide so we tolerate a slightly delayed
    print without falling through to the next revision.
    """
    hs, ms, ss = [int(x) for x in window_start.split(":")]
    he, me, se = [int(x) for x in window_end.split(":")]
    return (
        df_imbalance
        .filter(
            pl.col("ts_event").dt.time().is_between(
                pl.time(hs, ms, ss), pl.time(he, me, se)
            )
        )
        .with_columns(pl.col("ts_event").dt.date().alias("date"))
        .group_by(["date", "symbol"], maintain_order=True)
        .agg(
            pl.col(["side", "total_imbalance_qty", "paired_qty", "ref_price"]).first()
        )
    )


# ---------------------------------------------------------------------------
# Cross-sectional feature builder
# ---------------------------------------------------------------------------

# Columns whose null-ness is a real reason to drop a row: they are the
# per-name observations the alpha model can't be predicted without.
#
# ADV and vol ARE required inputs. In the retrain path they're already
# non-null past the causal-EWMA warm-up, so the drop cleans out the
# warm-up rows. In the LIVE path the single-day buffer produces null
# ADV/vol for every row (the causal EWMA needs prior days), so callers
# MUST inject reference stats via ``merge_reference_stats`` — either by
# passing ``reference_stats`` to ``build_feature_frame`` or by calling
# ``merge_reference_stats`` before ``derive_features``. Otherwise every
# live row is dropped and the daily orchestrator's coverage assertion
# aborts the run (a deliberately fatal failure mode — better an aborted
# day than mistakenly flattening the book on empty alpha).
_REQUIRED_INPUTS: tuple[str, ...] = (
    "date",
    "symbol",
    "side",
    "paired_qty",
    "ref_price",
    "entry_price",
    "spread_return",
    "recent_return",
    "ADV",
    "vol",
)


def join_raw_pieces(
    imb_print: pl.DataFrame,
    close_trades: pl.DataFrame,
    avg_daily_dollar_volume: pl.DataFrame,
    daily_vol: pl.DataFrame,
    recent_return: pl.DataFrame,
    auction: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Left-join per-(date, symbol) aggregates without dropping any rows.

    ADV/vol are left-joined so a single-day live snapshot (where causal EWMA
    is null everywhere) still produces one row per (date, symbol). Callers
    then either backfill ADV/vol from reference stats or run offline where
    the full window has non-null values.
    """
    frame = (
        imb_print
        .join(close_trades, on=["date", "symbol"], how="inner")
        .join(recent_return, on=["date", "symbol"], how="inner")
        .join(avg_daily_dollar_volume, on=["date", "symbol"], how="left")
        .join(daily_vol, on=["date", "symbol"], how="left")
    )
    if auction is not None:
        frame = frame.join(auction, on=["date", "symbol"], how="inner")
    return frame.sort(["date", "symbol"])


def derive_features(
    frame: pl.DataFrame,
    auction_present: bool,
) -> pl.DataFrame:
    """Given a joined frame with ADV/vol already populated (either from the
    tape or from reference stats), compute the ratio + rank features the
    alpha model consumes.

    The caller is responsible for injecting ADV/vol before this step —
    otherwise ``auction_dollar_volume_ratio`` divides by null.
    """
    frame = (
        frame
        .with_columns(
            (pl.col("total_imbalance_qty") * pl.col("ref_price")).alias("dollar_imbalance"),
            (pl.col("paired_qty") * pl.col("ref_price")).alias("paired_dollars"),
            (
                pl.when(pl.col("side") == "B").then(1)
                .when(pl.col("side") == "A").then(-1)
                .when(pl.col("side") == "N").then(0)
            ).alias("sign"),
        )
        .with_columns(
            (pl.col("sign") * pl.col("dollar_imbalance")).alias("signed_dollar_imbalance"),
        )
        .drop("total_imbalance_qty")
    )

    # Scoped drop: rows missing a REQUIRED input are unusable. ADV/vol nulls
    # are handled by the ratio's null propagation + a downstream cross-section
    # mean that ignores nulls; a name that ends up with null features is
    # excluded from the sklearn ``predict`` step because ForecastModel drops it.
    keep_cols = [c for c in _REQUIRED_INPUTS if c in frame.columns]
    frame = frame.drop_nulls(subset=keep_cols)

    frame = frame.with_columns(
        (pl.col("paired_dollars") / pl.col("ADV")).alias("auction_dollar_volume_ratio"),
        (
            pl.when(pl.col("paired_dollars") != 0)
            .then(pl.col("signed_dollar_imbalance") / pl.col("paired_dollars"))
            .otherwise(0)
        ).alias("net_signed_imb"),
    )

    if auction_present:
        frame = frame.with_columns(
            (
                (pl.col("auction_price") - pl.col("entry_price")) / pl.col("entry_price")
            ).alias("return_to_close"),
        )
        frame = frame.with_columns(
            (
                pl.col("return_to_close") - pl.col("return_to_close").mean().over("date")
            ).alias(TARGET_COLUMN),
        )

    frame = frame.with_columns(
        (
            pl.col("recent_return") - pl.col("recent_return").mean().over("date")
        ).alias("recent_return_xm"),
        (
            pl.col("auction_dollar_volume_ratio")
            - pl.col("auction_dollar_volume_ratio").mean().over("date")
        ).alias("auction_dollar_volume_ratio_xm"),
        _norm_rank_expr("net_signed_imb").alias("norm_rank_net_imb"),
    )
    frame = frame.with_columns(
        (pl.col("norm_rank_net_imb") * pl.col("recent_return_xm"))
        .alias("norm_rank_net_x_recent_xm"),
        (pl.col("auction_dollar_volume_ratio_xm") * pl.col("norm_rank_net_imb"))
        .alias("norm_rank_net_x_auction"),
        (pl.col("auction_dollar_volume_ratio_xm") * pl.col("recent_return_xm"))
        .alias("auction_x_recent_xm"),
    )
    return frame


def build_feature_frame(
    imb_print: pl.DataFrame,
    close_trades: pl.DataFrame,
    avg_daily_dollar_volume: pl.DataFrame,
    daily_vol: pl.DataFrame,
    recent_return: pl.DataFrame,
    auction: pl.DataFrame | None = None,
    reference_stats: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Join per-(date, symbol) aggregates and derive the alpha features.

    Mirrors ``forecast/forecasting.ipynb`` and ``optimizer/optimal_portfolio.ipynb``.

    ``reference_stats`` is an optional per-symbol snapshot (``symbol, ADV, vol``)
    used to backfill null ADV/vol BEFORE ratios are computed. Retrain uses this
    only for the final derived reference-stats output; the live path passes it
    at every call so ``auction_dollar_volume_ratio`` is finite on a one-day
    snapshot.

    When ``auction`` is None (live), the ``return_to_close``/``close_return_xm``
    target columns are simply absent.
    """
    frame = join_raw_pieces(
        imb_print=imb_print,
        close_trades=close_trades,
        avg_daily_dollar_volume=avg_daily_dollar_volume,
        daily_vol=daily_vol,
        recent_return=recent_return,
        auction=auction,
    )
    if reference_stats is not None:
        frame = merge_reference_stats(frame, reference_stats)
    return derive_features(frame, auction_present=auction is not None)


def _norm_rank_expr(col: str) -> pl.Expr:
    """Normalised rank in ``[-1, 1]``, cross-sectional per date.

    Matches the notebook's ``(rank - rank.min) / (rank.max - rank.min) * 2 - 1``
    formulation. A day with a single name would produce ``nan``, so callers
    must ensure per-day counts are >= 2 or handle nulls.
    """
    r = pl.col(col).rank().over("date")
    return (r - r.min().over("date")) / (r.max().over("date") - r.min().over("date")) * 2 - 1


# ---------------------------------------------------------------------------
# Live-side glue: convert buffered live records to the join-ready frames.
#
# The live_feed hands us a list of dicts (buffered from the Databento
# callback). We compact them into DataFrames with the same column names/dtypes
# the historical parquets have, then run them through the same aggregators.
# ---------------------------------------------------------------------------

def imbalance_records_to_frame(records: list[dict]) -> pl.DataFrame:
    """Build a polars frame from live ``ImbalanceMsg`` dicts.

    Expected keys (see ``live_feed.LiveFeed._on_record``):
      ``ts_event`` (datetime UTC), ``symbol`` (str), ``side`` (str),
      ``total_imbalance_qty`` (int), ``paired_qty`` (int), ``ref_price`` (float).
    """
    if not records:
        return pl.DataFrame(
            schema={
                "ts_event": pl.Datetime("ns", "UTC"),
                "symbol": pl.Categorical,
                "side": pl.Utf8,
                "total_imbalance_qty": pl.UInt64,
                "paired_qty": pl.UInt64,
                "ref_price": pl.Float64,
            }
        )
    return pl.DataFrame(records).with_columns(
        pl.col("ts_event").dt.convert_time_zone("America/New_York"),
        pl.col("symbol").cast(pl.Categorical),
    )


def tbbo_records_to_frame(records: list[dict]) -> pl.DataFrame:
    """Build a polars frame from live tbbo/MBP1 dicts.

    Expected keys: ``ts_event`` (datetime UTC), ``symbol`` (str),
    ``price`` (float), ``size`` (int), ``ask_px_00`` (float),
    ``bid_px_00`` (float).
    """
    if not records:
        return pl.DataFrame(
            schema={
                "ts_event": pl.Datetime("ns", "UTC"),
                "symbol": pl.Categorical,
                "price": pl.Float64,
                "size": pl.UInt64,
                "ask_px_00": pl.Float64,
                "bid_px_00": pl.Float64,
            }
        )
    return pl.DataFrame(records).with_columns(
        pl.col("ts_event").dt.convert_time_zone("America/New_York"),
        pl.col("symbol").cast(pl.Categorical),
    )


def merge_reference_stats(
    features_df: pl.DataFrame,
    reference_stats: pl.DataFrame,
) -> pl.DataFrame:
    """Overwrite ``ADV`` and ``vol`` with retrain-time reference stats.

    The live pipeline computes per-day ADV/vol from live tape data, but in
    practice there is only ONE day of data at 15:50 so the causal EWMA is
    null-only. The retrain script produces a per-symbol snapshot of the same
    causal ADV/vol as of the retrain date; this helper joins it in so the
    optimizer's impact and cost models get non-null inputs.

    ``reference_stats`` must have columns ``symbol``, ``ADV``, ``vol``.
    """
    stats = reference_stats.select(["symbol", "ADV", "vol"]).rename(
        {"ADV": "ADV_ref", "vol": "vol_ref"}
    ).with_columns(pl.col("symbol").cast(pl.Categorical))
    joined = features_df.join(stats, on="symbol", how="left")
    return joined.with_columns(
        pl.coalesce(["ADV", "ADV_ref"]).alias("ADV"),
        pl.coalesce(["vol", "vol_ref"]).alias("vol"),
    ).drop(["ADV_ref", "vol_ref"])


# ---------------------------------------------------------------------------
# Covariance builder — used by retrain.py
# ---------------------------------------------------------------------------

def apply_corporate_actions(
    daily_bars: pl.DataFrame,
    cax: pl.DataFrame,
) -> pl.DataFrame:
    """Fold splits/dividends into a per-symbol cumulative adjustment factor
    and return an ``adj_close`` frame — exactly like ``covariance.ipynb``.

    The output has one row per (symbol, date) with columns
    ``ts_event, symbol, adj_close``.
    """
    return (
        daily_bars
        .join(cax, left_on=["ts_event", "symbol"], right_on=["date", "symbol"], how="left")
        .sort(["symbol", "ts_event"])
        .with_columns(
            pl.col("dividend").fill_null(0.0),
            pl.when(pl.col("split").fill_null(0.0) == 0.0)
            .then(1.0)
            .otherwise(pl.col("split"))
            .alias("split"),
        )
        .with_columns(
            (pl.col("split") / (1 - pl.col("dividend") / pl.col("close").shift().over("symbol")))
            .alias("CAX")
        )
        .with_columns(pl.col("CAX").fill_null(1.0))
        .with_columns(pl.col("CAX").cum_prod().over("symbol"))
        .with_columns((pl.col("close") * pl.col("CAX")).alias("adj_close"))
        .select(["ts_event", "symbol", "adj_close"])
    )


def build_factor_covariance(
    adj_px: pl.DataFrame,
    active_set: list[str],
    n_factors: int = 10,
    causal_vol_floor: float = 1e-3,
    eps_floor: float = 0.1,
    name_vol_percentiles: tuple[float, float] = (2.5, 97.5),
) -> dict:
    """Fit a factor covariance model from the notebook's recipe.

    Returns a dict of artifacts:
      ``name_vol`` (p,), ``loadings_return`` (k, p), ``factor_returns`` (T, k),
      ``factor_cov_matrix`` (k, k), ``idiosyncratic_diag_return`` (p,),
      ``cov_matrix_return`` (p, p), ``eps_var`` (p,), and the diagnostic
      scalar ``condition_number``.
    """
    daily_ret = (
        adj_px
        .with_columns(pl.col("adj_close").pct_change().over("symbol").alias("ret"))
        .pivot(on="symbol", index="ts_event", values="ret")
        .with_columns(pl.all().exclude("ts_event").fill_null(pl.all().exclude("ts_event").mean()))
        .with_columns((pl.all().exclude("ts_event") - pl.all().exclude("ts_event").mean()).round(6))
        .sort("ts_event")
    )

    # active_set order MUST be preserved end-to-end so downstream artifacts
    # index the right symbols. Any symbols missing from the pivot get inserted
    # as zero columns — the optimizer will treat them as flat regardless.
    sym_cols = [c for c in daily_ret.columns if c != "ts_event"]
    for s in active_set:
        if s not in sym_cols:
            daily_ret = daily_ret.with_columns(pl.lit(0.0).alias(s))
    daily_ret = daily_ret.select(["ts_event"] + list(active_set))
    sym_cols = list(active_set)

    causal_vol = daily_ret.with_columns(
        pl.all().exclude("ts_event").pow(2).ewm_mean(alpha=0.06, adjust=True).sqrt().shift(1)
    )
    st_ret = (
        daily_ret.select(sym_cols)
        / causal_vol.select(sym_cols).select(pl.all().clip(lower_bound=causal_vol_floor))
    ).select(pl.all().clip(-10, 10))

    X = (
        st_ret
        .slice(1)
        .with_columns(pl.all().fill_null(0.0))
        .to_numpy()
    )
    R = daily_ret.select(sym_cols).slice(1).to_numpy()

    name_vol = R.std(axis=0, ddof=1)
    name_vol = np.maximum(name_vol, causal_vol_floor)
    lo, hi = np.percentile(name_vol, list(name_vol_percentiles))
    name_vol = np.clip(name_vol, lo, hi)

    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    k = n_factors
    U_k, S_k, Vt_k = U[:, :k], np.diag(s[:k]), Vt[:k, :]
    F = U_k
    L = S_k @ Vt_k

    eps = X - F @ L
    eps_var = np.var(eps, axis=0)
    eps_var = np.maximum(eps_var, eps_floor)

    Sigma_f = np.cov(F, rowvar=False)
    D = np.diag(eps_var)
    Sigma_r = L.T @ Sigma_f @ L + D  # standardised space

    # return-unit rescale
    S_diag = np.diag(name_vol)
    L_ret = L @ S_diag
    Dr_diag = name_vol**2 * eps_var
    Sigma_r_ret = L_ret.T @ Sigma_f @ L_ret + np.diag(Dr_diag)

    condition_number = float(np.linalg.cond(Sigma_r_ret))

    return {
        "name_vol": name_vol,
        "loadings_return": L_ret,
        "factor_returns": F,
        "factor_cov_matrix": Sigma_f,
        "idiosyncratic_diag_return": Dr_diag,
        "cov_matrix_return": Sigma_r_ret,
        "eps_var": eps_var,
        "cov_matrix_standardised": Sigma_r,
        "loadings_standardised": L,
        "condition_number": condition_number,
    }


# ---------------------------------------------------------------------------
# Impact calibration — used by retrain.py
# ---------------------------------------------------------------------------

def build_impact_frame(
    tbbo: pl.LazyFrame,
    active_set: list[str],
    ewm_span: int = 10,
) -> pl.DataFrame:
    """Assemble the per-(symbol, date) fit frame used by the impact
    regression. Mirrors ``impact/impact_estimation.ipynb``.
    """
    from datetime import time as _time
    window = (
        tbbo
        .with_columns(
            pl.col("ts_event").dt.date().alias("date"),
            ((pl.col("ask_px_00") + pl.col("bid_px_00")) / 2).alias("mid"),
        )
        .filter(pl.col("symbol").is_in(active_set))
        .filter(pl.col("ts_event").dt.time().is_between(pl.time(15, 50), pl.time(16, 0)))
        .sort("symbol", "ts_event")
        .with_columns(
            pl.when(pl.col("price") > pl.col("mid")).then(1)
            .when(pl.col("price") < pl.col("mid")).then(-1)
            .otherwise(None).alias("quote_sign"),
            pl.when(pl.col("price").diff().over("symbol") > 0).then(1)
            .when(pl.col("price").diff().over("symbol") < 0).then(-1)
            .otherwise(None).alias("tick_sign"),
        )
        .with_columns(pl.col("tick_sign").forward_fill().over("symbol"))
        .with_columns(
            pl.when(pl.col("side") == "B").then(1)
            .when(pl.col("side") == "A").then(-1)
            .otherwise(pl.coalesce("quote_sign", "tick_sign"))
            .alias("trade_sign")
        )
    )

    per_day_window = (
        window
        .group_by(["symbol", "date"])
        .agg(
            pl.col("mid").first().alias("p0"),
            pl.col("mid").last().alias("p1"),
            (
                pl.col("trade_sign").fill_null(0).cast(pl.Float64)
                * pl.col("size").cast(pl.Float64) * pl.col("price")
            ).sum().alias("Q"),
            pl.len().alias("n_trades"),
        )
        .collect(engine="streaming")
    )

    daily_dollar_volume = (
        tbbo
        .with_columns(pl.col("ts_event").dt.date().alias("date"))
        .filter(pl.col("symbol").is_in(active_set))
        .group_by(["symbol", "date"])
        .agg((pl.col("size").cast(pl.Float64) * pl.col("price")).sum().alias("day_volume"))
        .sort(["symbol", "date"])
        .with_columns(
            pl.col("day_volume")
            .ewm_mean(span=ewm_span, adjust=True)
            .shift(1)
            .over("symbol")
            .alias("ADV")
        )
        .drop("day_volume")
        .collect(engine="streaming")
    )

    daily_close = (
        tbbo
        .with_columns(
            pl.col("ts_event").dt.date().alias("date"),
            ((pl.col("ask_px_00") + pl.col("bid_px_00")) / 2).alias("mid"),
        )
        .filter(pl.col("symbol").is_in(active_set))
        .filter(pl.col("ts_event").dt.time().is_between(pl.time(15, 50), pl.time(16, 0)))
        .group_by(["symbol", "date"])
        .agg(pl.col("mid").last().alias("close_mid"))
        .sort(["symbol", "date"])
        .with_columns(pl.col("close_mid").pct_change().over("symbol").alias("daily_ret"))
        .collect(engine="streaming")
    )
    sigma_df = (
        daily_close.drop_nulls("daily_ret")
        .group_by("symbol")
        .agg(pl.col("daily_ret").std(ddof=1).alias("sigma"), pl.len().alias("n_days"))
    )

    fit = (
        per_day_window
        .join(daily_dollar_volume, on=["symbol", "date"], how="inner")
        .join(sigma_df.select("symbol", "sigma"), on="symbol", how="inner")
        .filter(
            pl.col("p0").is_not_null() & pl.col("p1").is_not_null()
            & (pl.col("p0") > 0)
            & pl.col("ADV").is_not_null() & (pl.col("ADV") > 0)
            & pl.col("sigma").is_not_null() & (pl.col("sigma") > 0)
        )
        .with_columns(
            (pl.col("p1") / pl.col("p0") - 1.0).alias("r"),
            (
                pl.col("Q").sign() * (pl.col("Q").abs() / pl.col("ADV")).sqrt()
            ).alias("x_sqrt"),
            (pl.col("Q") / pl.col("ADV")).alias("x_lin"),
        )
        .with_columns((pl.col("r") - pl.col("r").mean().over("date")).alias("r_idio"))
        .with_columns((pl.col("r_idio") / pl.col("sigma")).alias("y"))
        .drop_nulls(["x_sqrt", "y"])
    )
    return fit


def calibrate_impact_gamma(fit_frame: pl.DataFrame, active_set: list[str]) -> tuple[np.ndarray, dict]:
    """Fit per-name square-root impact ``gamma_i`` with pooled + partial-pooling
    shrinkage, aligned to ``active_set`` order. Verbatim from
    ``impact/impact_estimation.ipynb``.
    """
    from sklearn.linear_model import HuberRegressor
    from scipy.stats import theilslopes

    def winsorize(arr: np.ndarray, lo_q: float = 0.01, hi_q: float = 0.99) -> np.ndarray:
        lo = np.quantile(arr, lo_q)
        hi = np.quantile(arr, hi_q)
        return np.clip(arr, lo, hi)

    x_raw = fit_frame["x_sqrt"].to_numpy()
    y_raw = fit_frame["y"].to_numpy()
    # L3: winsorize x/y POOLED (1st/99th percentile over all name-days),
    # not per-symbol. Notebook does the pool-level clip too — per-group
    # clipping shrinks the tails within each name and biases the slopes.
    x_w = winsorize(x_raw)
    y_w = winsorize(y_raw)
    fit_frame = fit_frame.with_columns(
        pl.Series("x_w", x_w),
        pl.Series("y_w", y_w),
    )

    def fit_slope(x: np.ndarray, y: np.ndarray, min_n: int = 20) -> tuple[float, int]:
        m = np.isfinite(x) & np.isfinite(y)
        x = x[m]; y = y[m]
        n = int(x.size)
        if n < min_n:
            return float("nan"), n
        try:
            hr = HuberRegressor(fit_intercept=False, max_iter=200)
            hr.fit(x.reshape(-1, 1), y)
            slope = float(hr.coef_[0])
            if not np.isfinite(slope):
                raise ValueError("non-finite")
            return slope, n
        except Exception:
            nz = x != 0
            if int(nz.sum()) < min_n:
                return float("nan"), n
            try:
                slope, *_ = theilslopes(y[nz], x[nz])
                return float(slope), n
            except Exception:
                return float("nan"), n

    gamma_pool, n_pool = fit_slope(x_w, y_w)

    raw_g: list[float] = []
    raw_n: list[int] = []
    fit_by_sym: dict[str, tuple[float, int]] = {}
    for sym, grp in fit_frame.group_by("symbol", maintain_order=True):
        sym_str = str(sym[0])
        gi, ni = fit_slope(grp["x_w"].to_numpy(), grp["y_w"].to_numpy())
        fit_by_sym[sym_str] = (gi, ni)
        raw_g.append(gi)
        raw_n.append(ni)

    raw_gammas = np.array(raw_g)
    raw_ns = np.array(raw_n)
    mask_ok = np.isfinite(raw_gammas)
    mask_pos = mask_ok & (raw_gammas > 0)

    pos_mask = mask_pos & np.isfinite(raw_gammas)
    between_var = float(np.nanvar(raw_gammas[pos_mask], ddof=1)) if int(pos_mask.sum()) > 5 else float("nan")
    pooled_pred = gamma_pool * x_w
    resid_var = float(np.var(y_w - pooled_pred, ddof=1))
    if np.isfinite(between_var) and between_var > 0:
        n0 = resid_var / between_var
    else:
        n0 = float(np.median(raw_ns[raw_ns > 0])) if (raw_ns > 0).any() else 100.0
    n0 = float(np.clip(n0, 20.0, 5000.0))

    shrunk = np.empty_like(raw_gammas)
    for i, (gi, ni) in enumerate(zip(raw_gammas, raw_ns)):
        if not np.isfinite(gi) or gi <= 0 or ni < 20:
            shrunk[i] = gamma_pool
        else:
            w = ni / (ni + n0)
            shrunk[i] = w * gi + (1 - w) * gamma_pool
    shrunk = np.abs(shrunk)
    gamma_by_sym = {sym: g for sym, g in zip(fit_by_sym.keys(), shrunk)}

    gamma_vec = np.array(
        [gamma_by_sym.get(sym, gamma_pool) for sym in active_set],
        dtype=float,
    )
    gamma_vec = np.where(np.isfinite(gamma_vec) & (gamma_vec > 0), gamma_vec, gamma_pool)
    gamma_vec = np.abs(gamma_vec)

    return gamma_vec, {
        "gamma_pool": float(gamma_pool),
        "n_pool": int(n_pool),
        "n0": float(n0),
        "resid_var": float(resid_var),
        "between_var": float(between_var) if np.isfinite(between_var) else float("nan"),
        "n_fit_symbols": int(len(fit_by_sym)),
    }
