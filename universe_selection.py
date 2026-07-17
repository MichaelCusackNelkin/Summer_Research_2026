from data_collection.data_adapters import DatabentoGenericAdapter
from dotenv import load_dotenv
from pathlib import Path
import os
import polars as pl
load_dotenv()
pl.enable_string_cache()

CACHE_DIR = Path.cwd().resolve() / "data" / "dbn_cache" / "generic"
adapter = DatabentoGenericAdapter(os.getenv("DATABENTO-API-KEY"), CACHE_DIR)

print(f"CACHE_DIR: {CACHE_DIR}")

START = "2025-01-01"
END = "2025-06-10"
DATASET = "XNAS.ITCH"

K = 500

day_bars = (
    adapter.scan_data(
        tickers="ALL_SYMBOLS",
        start_date=START,
        end_date=END,
        dataset=DATASET,
        schema="ohlcv-1d"
    )
)

defs = (
    adapter.scan_data(
        tickers="ALL_SYMBOLS",
        start_date=START,
        end_date=END,
        dataset=DATASET,
        schema="definition"
    )
)

nasdaq_stocks = (
    defs
    .group_by("symbol")
    .agg(
        pl.col("security_type").last().alias("sec_type"),
        (pl.col("exchange") == "XNAS").all().alias("xnas_full_window"),
        pl.col("exchange").n_unique().alias("n_exch"),
    )
    .filter(
        (pl.col("sec_type") == "C") & pl.col("xnas_full_window"),
        ~pl.col("symbol").str.contains(r"ZZT$|^Z[A-Z]YZ|^ZBZX|^ZSPC|^ZWZ")
    )
    .select("symbol")
    .sort("symbol")
    .collect(engine="streaming")
    .to_series()
    .to_list()
)

top_k = (
    day_bars
    .select(["ts_event", "symbol", "close", "volume"])
    .group_by(
        by="symbol"
    )
    .agg(
        (pl.col("close")*pl.col("volume")).mean().alias("ADV")
    )
    .rename({"by":"symbol"})
    .filter(pl.col("symbol").is_in(nasdaq_stocks))
    .top_k(K, by="ADV")
    .select("symbol")
    .collect(engine="streaming")
    .to_series()
    .to_list()
)

if Path.exists(f"top_{K}_universe.txt"):
    with open(f"top_{K}_universe.txt", "w") as f:
        f.write("\n".join(top_k))
else:
    with open(f"top_{K}_universe.txt", "x") as f:
        f.write("\n".join(top_k))