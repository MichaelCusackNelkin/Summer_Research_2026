from data_collection.data_adapters import DatabentoGenericAdapter
import polars as pl
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
import asyncio
import os
load_dotenv()
pl.enable_string_cache()

CACHE_DIR = Path.cwd().resolve() / "data" / "dbn_cache" / "generic"
adapter = DatabentoGenericAdapter(os.getenv("DATABENTO_API_KEY"), CACHE_DIR)

# Clear the cache if needed
# adapter.clear_cache()

# Reading universe.txt file
with open("universe.txt", "r") as f:
    UNIVERSE = [line.strip() for line in f]
START = "2025-06-02"
END = "2026-06-02"
DATASET = "XNAS.ITCH"

async def main():
      schemas = ["imbalance", "tbbo", "statistics"]

      async def fetch(schema):
          return await adapter.get_data(
              tickers=UNIVERSE,
              start_date=START,
              end_date=END,
              dataset=DATASET,
              schema=schema,
              materialize_only=True,
          )

      results = await asyncio.gather(*(fetch(s) for s in schemas))

      for schema, indicator in zip(schemas, results):
          print(f"{schema}: {not indicator}")


if __name__ == "__main__":
    asyncio.run(main())