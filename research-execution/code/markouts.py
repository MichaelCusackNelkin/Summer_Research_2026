import os
from pathlib import Path
from data_collection.data_adapters import DatabentoGenericAdapter
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from saf_lib.execution import compute_markouts, plot_markout_curves, plot_book_imbalance_vs_markout
import polars as pl
import numpy as np
import asyncio
load_dotenv()

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "dbn_cache" / "generic"

async def get_data() -> pl.DataFrame:
    adapter = DatabentoGenericAdapter(os.getenv("DATABENTO_API_KEY"), CACHE_DIR)
    df, map = await adapter.get_data(
        tickers=["CSCO", "AMZN", "DRH"],
        start_date="2026-03-25",
        end_date="2026-03-31",
        dataset="XNAS.ITCH",
        schema="MBP-10",
        cache_only=True
    )
    map = {v: k for k,v in map.items()}
    return df, map


if __name__ == "__main__":
    SEED = 1324

    # Get Data adn select to relavent columns
    df, map = asyncio.run(get_data())
    cols = ["ts_event", "instrument_id"] + [
        col for n in range(0, 10)
        for col in [f"bid_px_{str(n).zfill(2)}", f"ask_px_{str(n).zfill(2)}", f"bid_sz_{str(n).zfill(2)}", f"ask_sz_{str(n).zfill(2)}"]
    ]
    df = df.select(cols).filter(pl.col('instrument_id') == map["CSCO"])
    print(f"File size (gb): {df.estimated_size('gb'):.2f}")

    # Compute markouts at multiple horizons (sampled for efficiency)
    horizon_ns = np.arange(0, 10_000_000, 10_000) # 0 to 10ms in steps of 1us
    markouts = compute_markouts(df, horizons_ns=horizon_ns, sample_n=100_000, seed=SEED)
    print(f"Computed markouts for {len(markouts):,} sampled ticks")

    # Markout curves conditioned on full-book imbalance
    plot_markout_curves(markouts, signal_col='book_imbalance', n_bins=5)
    plt.savefig(f'plots/full_book_markout_{SEED}', dpi=300, bbox_inches='tight')
    plt.show()

    # # Markout curves conditioned on top level imbalance only
    # plot_markout_curves(markouts, signal_col='obi_level0', n_bins=5)
    # plt.savefig(f'plots/top_level_markout_{SEED}', dpi=300, bbox_inches='tight')
    # plt.show()

    # fig, ax = plt.subplots(2,1)
    # plot_book_imbalance_vs_markout(markouts, horizon_ns=100*1e3, ax=ax[0])
    # plot_book_imbalance_vs_markout(markouts, horizon_ns=1500*1e3, ax=ax[1])
    # plt.tight_layout()
    # plt.show()