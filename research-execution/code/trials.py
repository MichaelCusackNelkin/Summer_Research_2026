from data_collection.data_adapters import DatabentoGenericAdapter
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm
import hashlib
import sys
import json
import polars as pl
import os
load_dotenv()

def _trial_cache_key(
    ticker: str,
    start: str,
    end: str,
    latency: float,
    replenishment: float,
    parent_qty: int,
    depth_lmt: str,
    OBI_THRESHOLD: float,
    horizon: float,
    min_ns: float,
    min_shares: int,
    max_size: int,
    wait_ns: float,
    n_chunks: int,
    max_overlap_ns: float,
    method: str,
    n_trials: int,
    seed: int,
) -> str:
    """Compute an MD5 hash over the trial parameters for cache keying.

    Parameters
    ----------
    ticker : str
        Ticker symbol used for the trial.
    latency : float
        Simulated latency in nanoseconds.
    replenishment : float
        Poisson rate of book replenishment (wall-clock).
    parent_qty : int
        Total parent order quantity.
    depth_lmt : str
        Zero-padded depth level string (e.g. "03").
    OBI_THRESHOLD : float
        OBI entry OBI_THRESHOLD (absolute value).
    horizon : float
        Trading horizon in nanoseconds.
    min_ns : float
        Minimum wait between fills for OBI filler, in nanoseconds.
    min_shares : int
        Minimum shares traded between fills for OBI filler.
    max_size : int
        Maximum child order size.
    wait_ns : float
        Constant filler inter-child wait, in nanoseconds.
    n_chunks : int
        Number of chunks used by the constant filler.
    method : str
        Filler method name (e.g. "OBI" or "constant").
    n_trials : int
        Number of trials in the run.

    Returns
    -------
    str
        Hex digest of the MD5 hash of the serialized parameter dict.
    """
    payload = {
        "ticker": ticker,
        "start": start,
        "end": end,
        "latency": latency,
        "replenishment": replenishment,
        "parent_qty": parent_qty,
        "depth_lmt": depth_lmt,
        "OBI_THRESHOLD": OBI_THRESHOLD,
        "horizon": horizon,
        "min_ns": min_ns,
        "min_shares": min_shares,
        "max_size": max_size,
        "wait_ns": wait_ns,
        "n_chunks": n_chunks,
        "max_overlap_ns": max_overlap_ns,
        "method": method,
        "n_trials": n_trials,
        "seed": seed,
    }
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.md5(serialized.encode("utf-8")).hexdigest()

def load_or_run(
    method: str,
    trial_args: list,
    n_trials: int,
    worker_args: tuple[str, int],
    cache_dir: Path,
    max_workers: int,
    **cache_kwargs,
) -> pl.DataFrame:
    """Load cached trial results or run the ProcessPoolExecutor trial loop.

    Parameters
    ----------
    method : str
        Filler method name (e.g. "OBI" or "constant"). Used for the cache
        key, the tqdm progress-bar description, and caching.
    trial_args : list
        List of argument tuples passed to ``run_trial`` (one per trial).
    n_trials : int
        Number of trials to run; also used as the ``tqdm`` total.
    worker_args : tuple[str, int]
        Two-tuple ``(parquet_path, instrument_id)`` broadcast to each
        worker process via ``init_worker``. Workers lazy-scan the
        parquet file per-trial rather than loading the full DataFrame
        into memory.
    cache_dir : pathlib.Path
        Directory where cached parquet results are stored.
    max_workers : int
        Number of worker processes for the ``ProcessPoolExecutor``.
    **cache_kwargs
        Scalar trial parameters forwarded to :func:`_trial_cache_key`
        (all keyword arguments except ``method`` and ``n_trials``).

    Returns
    -------
    pl.DataFrame
        DataFrame of trial results with a ``lifetime_ns`` column
        appended.
    """
    from concurrent.futures import ProcessPoolExecutor
    from sim_lib.simulation import init_worker, run_trial

    key = _trial_cache_key(method=method, n_trials=n_trials, **cache_kwargs)
    path = cache_dir / f"{key}.parquet"
    params_path = cache_dir / f"{key}.json"

    if path.exists():
        print(f"[cache hit] {method}: loading {path}")
        return pl.read_parquet(path)

    print(f"[cache miss] {method}: running {n_trials} trials -> {path}")
    with ProcessPoolExecutor(
        max_workers=max_workers, initializer=init_worker, initargs=(worker_args,)
    ) as ex:
        results = [
            c
            for c in tqdm(
                ex.map(run_trial, trial_args), total=n_trials, desc=method, file=sys.stderr
            )
            if c is not None
        ]

    out = pl.DataFrame(results)
    out = out.with_columns(
        pl.col("child_fill_times").list.sum().alias("lifetime_ns")
    )
    out.write_parquet(path)
    with open(params_path, "w") as f:
        json.dump(
            {"method": method, "n_trials": n_trials, **cache_kwargs},
            f,
            indent=2,
            sort_keys=True,
            default=str,
        )
    return out

async def get_data_paths(tickers, start_date, end_date, dataset, schema, cache_only: bool = True) -> tuple[Path, dict[str, int]]:
    """Resolve the adapter's parquet cache path and symbol mapping.

    Mirrors the three-layer cache used by
    :meth:`DatabentoGenericAdapter.get_data`:

    1. If the parquet cache exists, return its path immediately without
       loading any rows.
    2. Otherwise, if raw ``.dbn.zst`` files are on disk, build the
       filtered DataFrame, write it to the parquet cache, and return.
    3. Otherwise, when ``cache_only`` is ``False``, submit a Databento
       batch job, download the raw files, build and persist the parquet
       cache, and return.

    Parameters
    ----------
    cache_only : bool, default True
        If ``True``, never hit the paid Databento API. Raise
        :class:`FileNotFoundError` when neither the parquet cache nor
        raw ``.dbn.zst`` files are available.

    Returns
    -------
    tuple[pathlib.Path, dict[str, int]]
        The parquet cache path and a ``{ticker: instrument_id}``
        mapping. The mapping is empty when the parquet cache is present
        but the raw ``.dbn.zst`` files have been cleared.

    Raises
    ------
    FileNotFoundError
        If ``cache_only`` is ``True`` and no cached data exists.
    """

    cache_dir = Path(__file__).resolve().parents[2] / "data" / "dbn_cache" / "generic"
    adapter = DatabentoGenericAdapter(os.getenv("DATABENTO_API_KEY"), cache_dir)
    ticker_list = adapter._normalize_tickers(tickers)
    cache_key = adapter._generate_cache_key(
        ticker_list, start_date, end_date, dataset, schema
    )
    parquet_path = adapter._get_cache_path(cache_key)

    # Layer 1: parquet cache
    if parquet_path.exists():
        raw_paths = adapter._find_raw_path(cache_key)
        mappings = adapter._merge_mappings(raw_paths) if raw_paths else {}
        ticker_to_id = {v: k for k, v in mappings.items()}
        return parquet_path, ticker_to_id

    # Layer 2: raw dbn.zst cache
    raw_paths = adapter._find_raw_path(cache_key)
    if raw_paths:
        df, mappings = adapter._load_and_filter(raw_paths)
        df.write_parquet(parquet_path)
        ticker_to_id = {v: k for k, v in mappings.items()}
        return parquet_path, ticker_to_id

    # Layer 3: Databento batch API
    if cache_only:
        raise FileNotFoundError(
            f"No cached data for {ticker_list} ({start_date} to {end_date}), "
            f"dataset={dataset}, schema={schema}. "
            "Call get_data_paths(cache_only=False) to download from Databento."
        )

    raw_paths = await adapter._batch_download(
        ticker_list, start_date, end_date, dataset, schema, cache_key
    )
    df, mappings = adapter._load_and_filter(raw_paths)
    df.write_parquet(parquet_path)
    ticker_to_id = {v: k for k, v in mappings.items()}
    return parquet_path, ticker_to_id

if __name__ == "__main__":
    import asyncio
    import numpy as np
    import polars as pl
    import matplotlib.pyplot as plt
    from sim_lib.simulation import *

    # -- Data Params --
    TICKER_LIST = ["AMZN", "CSCO", "DRH"]
    # TICKER_LIST = ["CSCO"]
    START = "2026-03-18"
    END = "2026-04-18"
    DATASET = "XNAS.ITCH"
    SCHEMA = "MBP-10"
    CACHE_ONLY = False

    # -- General/Data Params --
    LATENCY = 100 * 1e3                 #Child order latency ns
    REPLENISHMENT = 200.0               #poisson rate of replenishment using wall-clock
    PARENT_QTY = 10_000
    MAX_WORKERS = 10
    N_TRIALS = 5_000
    SEED = 12345

    # -- Window Params --
    HORIZON = 300 * 1e9                 # Window of data to pull for each trial
    MAX_OVERLAP_NS = HORIZON - 1        # Overlap between trial windows. Strictly less than horizon.

    # # -- OBI Params --
    WAIT_NS = 0                         #1ms, waiting time between fills for OBI filler, ns
    MIN_SHARES = 0                      # waiting time in shares traded between fills for OBI filler
    OBI_THRESHOLD = 0.85               # OBI threshold for entry, +- for buys and sells
    MAX_SIZE = 350                      # Maximum child order sizes

    # -- Constant / Hybrid Params --
    DEPTH_LMT = "09"                    # None or "0X" for the Xth level deep into the book at parent order time.
    WAIT_NS = 200 * 1e6                 # 200 wait between fills / min interval for OBI
    N_CHUNKS = 50
    OBI_THRESHOLD = 0.10                # OBI threshold for entry, +- for buys and sells
    MIN_SHARES = 0                      # unused by constant/hybrid; kept for cache-key consistency
    MAX_SIZE = 0                        # unused by constant/hybrid; kept for cache-key consistency

    # -- Cache Dir --
    CACHE_DIR = Path("trial_cache")
    CACHE_DIR.mkdir(exist_ok=True)

    # Resolve parquet path + ticker mapping without loading any rows.
    parquet_path, ticker_to_id = asyncio.run(get_data_paths(TICKER_LIST, START, END, DATASET, SCHEMA, CACHE_ONLY))
    # ticker_id = ticker_to_id[TICKER]
    TICKER_LIST.reverse()

    # Single-pass streaming write for any missing ticker parquets.
    # One scan of the source file regardless of how many tickers are missing.
    import pyarrow.parquet as pq
    import pyarrow.compute as pc

    missing = {
        ticker: (ticker_to_id[ticker], CACHE_DIR / f"{ticker}_{parquet_path.stem}.parquet")
        for ticker in TICKER_LIST
        if not (CACHE_DIR / f"{ticker}_{parquet_path.stem}.parquet").exists()
    }
    if missing:
        print(f"[ticker parquet] single-pass write for: {list(missing)}")
        writers: dict[str, pq.ParquetWriter] = {}
        src = pq.ParquetFile(str(parquet_path))
        for batch in src.iter_batches(batch_size=100_000):
            for ticker, (inst_id, out_path) in missing.items():
                mask = pc.equal(batch.column("instrument_id"), inst_id)
                filtered = batch.filter(mask)
                if filtered.num_rows == 0:
                    continue
                df = pl.from_arrow(filtered).with_columns(
                    pl.col("ts_event").dt.timestamp("ns").alias("ts_nano"),
                    ((pl.col("ask_px_00") + pl.col("bid_px_00")) / 2).alias("mid"),
                )
                arrow_batch = df.to_arrow()
                if ticker not in writers:
                    writers[ticker] = pq.ParquetWriter(str(out_path), arrow_batch.schema)
                writers[ticker].write_table(arrow_batch)
        for w in writers.values():
            w.close()
        print("[ticker parquet] done.")

    for ticker in TICKER_LIST:
        TICKER = ticker
        ticker_id = ticker_to_id[TICKER]
        ticker_parquet_path = CACHE_DIR / f"{TICKER}_{parquet_path.stem}.parquet"
        print(f"[ticker parquet hit] using {ticker_parquet_path}")

        # Lazy-scan only the columns needed for window generation.
        df_meta = (
            pl.scan_parquet(str(ticker_parquet_path))
            .select(["ts_nano", "ts_event"])
            .collect()
        )

        # Pre-generate intraday start timestamps with bounded overlap between
        # trials. Restrict starts to 09:30 ET through 16:00 ET - HORIZON so
        # every window ends before the close. Adjacent windows may overlap by
        # up to MAX_OVERLAP_NS (30s); the cursor advances by
        # (HORIZON + LATENCY) - MAX_OVERLAP_NS between starts.
        _window_ns = int(HORIZON + LATENCY)
        _step_ns = int(HORIZON + LATENCY) - MAX_OVERLAP_NS
        _horizon_ns = int(HORIZON)

        _df_valid = (
            df_meta.with_columns(
                pl.col("ts_event").dt.convert_time_zone("US/Eastern").alias("ts_et")
            )
            .with_columns(pl.col("ts_et").dt.date().alias("date"))
            .filter(
                (
                    (pl.col("ts_et").dt.hour() > 9)
                    | (
                        (pl.col("ts_et").dt.hour() == 9)
                        & (pl.col("ts_et").dt.minute() >= 30)
                    )
                )
                & (
                    (pl.col("ts_et").dt.hour() < 15)
                    | (
                        (pl.col("ts_et").dt.hour() == 15)
                        & (
                            pl.col("ts_et").dt.minute() * 60
                            + pl.col("ts_et").dt.second()
                            <= 3600 - _horizon_ns // int(1e9)
                        )
                    )
                )
            )
            .sort(["date", "ts_nano"])
        )

        trial_starts: list[int] = []
        for (_,), _group in _df_valid.group_by(["date"], maintain_order=True):
            _ts_day = _group["ts_nano"].to_numpy()
            if _ts_day.size == 0:
                continue
            # Greedy tiling: take the first available ts, then skip forward by
            # (HORIZON + LATENCY) - MAX_OVERLAP_NS (allowing up to 30s of
            # overlap between adjacent windows) until the trading day is
            # exhausted.
            _cursor = int(_ts_day[0])
            _day_end = int(_ts_day[-1])
            _idx = 0
            while _cursor + _window_ns <= _day_end + 1:
                # Advance _idx to the first row with ts_nano >= _cursor.
                while _idx < _ts_day.size and _ts_day[_idx] < _cursor:
                    _idx += 1
                if _idx >= _ts_day.size:
                    break
                _start = int(_ts_day[_idx])
                if _start + _window_ns > _day_end + 1:
                    break
                trial_starts.append(_start)
                _cursor = _start + _step_ns

        rng = np.random.default_rng(SEED)
        rng.shuffle(trial_starts)
        if len(trial_starts) < N_TRIALS:
            print(
                f"[WARNING] Only {len(trial_starts)} {HORIZON/1e9:.0f}s "
                f"intraday windows (with at most {MAX_OVERLAP_NS / 1e9}s overlap) fit in the "
                f"dataset ({N_TRIALS} requested). Using {len(trial_starts)} "
                f"trials."
            )
        else:
            trial_starts = trial_starts[:N_TRIALS]
        print(f"Generated {len(trial_starts)} non-overlapping trial windows.")

        """Parallelized Trials"""
        shared_kwargs = dict(
            ticker=TICKER,
            start=START,
            end=END,
            latency=LATENCY,
            replenishment=REPLENISHMENT,
            parent_qty=PARENT_QTY,
            depth_lmt=DEPTH_LMT,
            OBI_THRESHOLD=OBI_THRESHOLD,
            horizon=HORIZON,
            min_ns=WAIT_NS,
            min_shares=MIN_SHARES,
            max_size=MAX_SIZE,
            wait_ns=WAIT_NS,
            n_chunks=N_CHUNKS,
            max_overlap_ns=MAX_OVERLAP_NS,
            seed=SEED,
        )


        #================
        # Constant Trial
        #================
        METHOD = "constant"
        trial_args = [(LATENCY, REPLENISHMENT, PARENT_QTY, DEPTH_LMT,
                    OBI_THRESHOLD, HORIZON, WAIT_NS, MIN_SHARES, MAX_SIZE, WAIT_NS, N_CHUNKS, METHOD, ts)
                    for ts in trial_starts]
        cost_constant = load_or_run(
            method=METHOD,
            trial_args=trial_args,
            n_trials=len(trial_starts),
            worker_args=(str(ticker_parquet_path), ticker_id),
            cache_dir=CACHE_DIR,
            max_workers=MAX_WORKERS,
            **shared_kwargs,
        )

        # #================
        # # OBI Trial
        # #================
        # METHOD = "obi"
        # trial_args = [(LATENCY, REPLENISHMENT, PARENT_QTY, DEPTH_LMT,
        #             OBI_THRESHOLD, HORIZON, WAIT_NS, MIN_SHARES, MAX_SIZE, WAIT_NS, N_CHUNKS, METHOD, ts)
        #             for ts in trial_starts]
        # cost_constant = load_or_run(
        #     method=METHOD,
        #     trial_args=trial_args,
        #     n_trials=len(trial_starts),
        #     worker_args=(str(ticker_parquet_path), ticker_id),
        #     cache_dir=CACHE_DIR,
        #     max_workers=MAX_WORKERS,
        #     **shared_kwargs,
        # )

        #================
        # Hybrid Trial
        #================
        METHOD = "hybrid"
        trial_args = [(LATENCY, REPLENISHMENT, PARENT_QTY, DEPTH_LMT,
                    OBI_THRESHOLD, HORIZON, WAIT_NS, MIN_SHARES, MAX_SIZE, WAIT_NS, N_CHUNKS, METHOD, ts)
                    for ts in trial_starts]
        cost_hybrid = load_or_run(
            method=METHOD,
            trial_args=trial_args,
            n_trials=len(trial_starts),
            worker_args=(str(ticker_parquet_path), ticker_id),
            cache_dir=CACHE_DIR,
            max_workers=MAX_WORKERS,
            **shared_kwargs,
        )

        # -- Descriptive Stats --
        print("\nHybrid Fill Costs")
        print(cost_hybrid.with_columns((pl.col("lifetime_ns") / 1e9).alias("lifetime_s"))
            .select(
                ["parent_filled", "parent_arrival_cost", "mean_child_arrival_cost", "median_child_arrival_cost", "lifetime_s"]
                ).describe()
            )
        print("\nConstant Fill Costs")
        print(cost_constant.with_columns((pl.col("lifetime_ns") / 1e9).alias("lifetime_s"))
            .select(
                ["parent_filled", "parent_arrival_cost", "mean_child_arrival_cost", "median_child_arrival_cost", "lifetime_s"]
                ).describe()
            )

        #===========
        # Plotting
        #===========
        plot_hybrid = cost_hybrid.with_columns(pl.from_epoch("ts_event", time_unit="ns").alias("ts_event").dt.convert_time_zone('US/Eastern'))
        plot_constant = cost_constant.with_columns(pl.from_epoch("ts_event", time_unit="ns", ).alias("ts_event").dt.convert_time_zone('US/Eastern'))
        plot_pre = f"temp_plot_folder/"
        plot_end = f"mid.{TICKER}"
        PLOT_DIR = Path("temp_plot_folder")
        PLOT_DIR.mkdir(exist_ok=True)

        # -- Parent Costs --
        log = False
        plt.hist(plot_hybrid["parent_arrival_cost"], bins=25, edgecolor="black", color="tab:orange", alpha=0.7, label="Hybrid", log=log)
        plt.hist(plot_constant["parent_arrival_cost"], bins=25, edgecolor="black", color="tab:blue", alpha=0.7, label="Naive", log=log)
        plt.axvline(plot_hybrid['parent_arrival_cost'].mean(), color='tab:orange', linestyle='--', linewidth=1, label='hybrid mean')
        plt.axvline(plot_constant['parent_arrival_cost'].mean(), color='tab:blue', linestyle='--', linewidth=1, label='naive mean')
        plt.xlabel("Cost per Share (bps)")
        plt.ylabel("Count")
        plt.title("Parent Cost Per Share Comparison")
        plt.legend()
        plt.savefig(f"{plot_pre}parent_cost_{plot_end}.png", dpi=300, bbox_inches='tight')
        plt.close()

        # -- Child Costs --
        fig, ax = plt.subplots(1,2,figsize=(8,4))
        # -- Means --
        ax[0].hist(plot_hybrid["mean_child_arrival_cost"], bins=25, edgecolor="black", color="tab:orange", alpha=0.7, label="Hybrid")
        ax[0].hist(plot_constant["mean_child_arrival_cost"], bins=25, edgecolor="black", color="tab:blue", alpha=0.7, label="Naive")
        ax[0].set_xlabel("Cost per Share (bps)")
        ax[0].set_ylabel("Count")
        ax[0].set_title("Mean Cost")
        ax[0].legend()

        # -- Medians --
        ax[1].hist(plot_hybrid["median_child_arrival_cost"], bins=25, edgecolor="black", color="tab:orange", alpha=0.7, label="Hybrid")
        ax[1].hist(plot_constant["median_child_arrival_cost"], bins=25, edgecolor="black", color="tab:blue", alpha=0.7, label="Naive")
        ax[1].set_xlabel("Cost per Share (bps)")
        ax[1].set_ylabel("Count")
        ax[1].set_title("Median Cost")
        ax[1].legend()

        plt.suptitle("Child Arrival Cost Per Share Comparison")
        plt.tight_layout()
        plt.savefig(f"{plot_pre}child_cost_{plot_end}.png", dpi=300, bbox_inches='tight')
        plt.close()

        # -- Fills --
        from matplotlib.ticker import PercentFormatter
        plt.hist(
            plot_hybrid["parent_filled"],
            weights=np.ones(len(plot_hybrid["parent_filled"])) / len(plot_hybrid["parent_filled"]),
            bins=25, edgecolor="black", color="tab:orange", alpha=0.7, label="Hybrid"
            )
        plt.hist(
            plot_constant["parent_filled"],
            weights=np.ones(len(plot_constant["parent_filled"])) / len(plot_constant["parent_filled"]),
            bins=25, edgecolor="black", color="tab:blue", alpha=0.7, label="Naive"
            )
        plt.xlabel("Parent Quantity Filled")
        plt.ylabel("Percent (%)")
        plt.title("Fill Reliability Comparison")
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.legend()
        plt.savefig(f"{plot_pre}fill_reliability_{plot_end}.png", dpi=300, bbox_inches='tight')
        plt.close()

        # -- Parent Lifetimes --
        plt.hist(
            plot_hybrid["lifetime_ns"] / 1e9,
            weights=np.ones(len(plot_hybrid["lifetime_ns"])) / len(plot_hybrid["lifetime_ns"]),
            bins=25, edgecolor="black", color="tab:orange", alpha=0.7, label="Hybrid"
            )
        plt.hist(
            plot_constant["lifetime_ns"] / 1e9,
            weights=np.ones(len(plot_constant["lifetime_ns"])) / len(plot_constant["lifetime_ns"]),
            bins=25, edgecolor="black", color="tab:blue", alpha=0.7, label="Naive"
            )
        plt.xlabel("Lifetime (s)")
        plt.ylabel("Percent (%)")
        plt.title("Parent Order Lifetime Comparison")
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.legend()
        plt.savefig(f"{plot_pre}parent_lifetimes_{plot_end}.png", dpi=300, bbox_inches='tight')
        plt.close()

        # -- Scatter --
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
        axes[0].scatter(
            plot_hybrid["parent_filled"], plot_hybrid["parent_arrival_cost"],
            color="tab:orange", alpha=0.4, s=8,
        )
        axes[0].set_title("Hybrid")
        axes[0].set_xlabel("parent_filled")
        axes[0].set_ylabel("parent_arrival_cost (bps)")

        axes[1].scatter(
            plot_constant["parent_filled"], plot_constant["parent_arrival_cost"],
            color="tab:blue", alpha=0.4, s=8,
        )
        axes[1].set_title("Naive")
        axes[1].set_xlabel("parent_filled")

        fig.suptitle(
            f"{TICKER}: parent_filled vs parent_arrival_cost"
        )
        fig.tight_layout()
        fig.savefig(f"{plot_pre}scatter_{plot_end}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # -- By Time of Day --
        plot_hybrid = (plot_hybrid.sort("ts_event")
             .group_by_dynamic(
                 pl.col("ts_event"), 
                 every="1m",
                 closed="right"
                 )
             .agg(
                 pl.col("parent_arrival_cost").mean()
                 )
        )

        plot_constant = (plot_constant.sort("ts_event")
             .group_by_dynamic(
                 pl.col("ts_event"), 
                 every="1m",
                 closed="right"
                 )
             .agg(
                 pl.col("parent_arrival_cost").mean()
                 )
)
        plot_hybrid = plot_hybrid.group_by(
            pl.col("ts_event").dt.time().alias("time"),
            maintain_order=True).agg(
                pl.col("parent_arrival_cost").median().alias("median cost"),
                pl.col("parent_arrival_cost").quantile(0.75).alias("upper qrt"),
                pl.col("parent_arrival_cost").quantile(0.25).alias("lower qrt")
                ).sort("time")
        
        plot_constant = plot_constant.group_by(
            pl.col("ts_event").dt.time().alias("time"),
            maintain_order=True).agg(
                pl.col("parent_arrival_cost").median().alias("median cost"),
                pl.col("parent_arrival_cost").quantile(0.75).alias("upper qrt"),
                pl.col("parent_arrival_cost").quantile(0.25).alias("lower qrt")
                ).sort("time")
        from datetime import datetime, date
        import matplotlib.dates as mdates
        import matplotlib.ticker as ticker
        def to_dt(series):
            return [datetime.combine(date.today(), t) for t in series]

        fig, ax = plt.subplots(figsize=(10, 4))
        x_hybrid = to_dt(plot_hybrid["time"])
        ax.plot(x_hybrid, plot_hybrid["median cost"],
                color="tab:orange", label="Hybrid")
        ax.fill_between(x_hybrid,
                        plot_hybrid["upper qrt"],
                        plot_hybrid["lower qrt"],
                        color="tab:orange", alpha=0.2, linewidth=0)

        x_const = to_dt(plot_constant["time"])
        ax.plot(x_const, plot_constant["median cost"],
                color="tab:blue", label="Naive")
        ax.fill_between(x_const,
                        plot_constant["upper qrt"],
                        plot_constant["lower qrt"],
                        color="tab:blue", alpha=0.2, linewidth=0)

        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

        ax.legend()
        ax.set_xlabel("Time of Day")
        ax.set_ylabel("Arrival Cost (bps)")
        ax.set_title("Median Arrival Cost by Time")
        ax.grid(True)
        fig.savefig(f"{plot_pre}time_{plot_end}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)