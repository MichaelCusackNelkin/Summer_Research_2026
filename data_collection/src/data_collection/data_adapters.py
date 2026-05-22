import asyncio
import hashlib
import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Union

import databento as db
import polars as pl
from tqdm import tqdm


class DataAdapter(ABC):
    """Cached data source. Subclasses define their own get_data signature and return type."""

    def __init__(self, cache_dir: Union[str, Path]) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    async def get_data(self, *args, **kwargs):
        ...

    @abstractmethod
    def clear_cache(self) -> None:
        ...


class DatabentoGenericAdapter(DataAdapter):
    """
    Generic adapter for any Databento schema, returning polars DataFrames.

    Uses a three-layer cache (parquet -> raw dbn.zst -> Databento batch API)
    and returns time columns exactly as Databento delivers them -- all
    hours, no timezone conversion, no market-hours filtering.
    """

    def __init__(self, api_key: str, cache_dir: Union[str, Path],
                 verbose: bool = True) -> None:
        super().__init__(cache_dir)
        self.api_key = api_key
        self.verbose = verbose
        self.raw_data_dir = self.cache_dir / "raw_dbn"
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

    def _normalize_tickers(self, tickers: Union[str, Iterable[str]]) -> list[str]:
        """Return a sorted, deduplicated list of uppercase ticker strings."""
        if isinstance(tickers, str):
            ticker_list = [tickers]
        else:
            ticker_list = list(tickers)
        return sorted({t.upper() for t in ticker_list})

    def _generate_cache_key(self, tickers: Union[str, Iterable[str]],
                            start_date: str, end_date: str,
                            dataset: str = "XNAS.ITCH",
                            schema: str = "ohlcv-1m") -> str:
        """Deterministic MD5 hash over tickers, dates, dataset, and schema."""
        if isinstance(tickers, str):
            ticker_list = [tickers]
        else:
            ticker_list = sorted(str(t) for t in tickers)

        payload = json.dumps({
            "tickers": ticker_list,
            "start": start_date,
            "end": end_date,
            "dataset": dataset,
            "schema": schema,
        }, sort_keys=True)
        return hashlib.md5(payload.encode("utf-8")).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.parquet"

    def _find_raw_path(self, cache_key: str) -> list[Path]:
        """Return all raw .dbn.zst files matching this cache key (any job_id prefix)."""
        return sorted(self.raw_data_dir.glob(f"*_{cache_key}*.dbn.zst"))

    def clear_cache(self) -> None:
        """Remove all processed parquet files. Raw .dbn.zst files are preserved."""
        for file in self.cache_dir.glob("*.parquet"):
            file.unlink()

    def _merge_mappings(self, raw_paths: list[Path]) -> dict[int, str]:
        """
        Read symbol mappings from .dbn.zst files into a flat
        ``{instrument_id: raw_symbol}`` dict.

        Raises
        ------
        ValueError
            If the same instrument_id resolves to two different raw symbols
            (violates the 1:1 symbol-to-instrument_id guarantee).
        """
        merged: dict[int, str] = {}
        for p in raw_paths:
            data = db.DBNStore.from_file(p)
            for raw_symbol, intervals in data.metadata.mappings.items():
                for interval in intervals:
                    resolved = (
                        interval["symbol"] if isinstance(interval, dict)
                        else interval.symbol
                    )
                    if not resolved:
                        continue
                    iid = int(resolved)
                    existing = merged.get(iid)
                    if existing is not None and existing != raw_symbol:
                        raise ValueError(
                            f"instrument_id {iid} maps to both "
                            f"{existing!r} and {raw_symbol!r}"
                        )
                    merged[iid] = raw_symbol
        return merged

    def _load(self, raw_paths: list[Path]) -> tuple[pl.DataFrame, dict[int, str]]:
        """
        Load one or more .dbn.zst files into a single polars DataFrame.

        Time columns are returned exactly as Databento delivers them -- all
        hours, no timezone conversion, no market-hours filtering.

        Raises
        ------
        ValueError
            If the same instrument_id resolves to two different raw symbols.
        """
        mappings = self._merge_mappings(raw_paths)
        dfs = []
        for p in raw_paths:
            data = db.DBNStore.from_file(p)
            pdf = data.to_df()
            dfs.append(pl.from_pandas(pdf.reset_index()))
        df = pl.concat(dfs)
        return df, mappings

    async def _batch_download(self, tickers: list[str], start_date: str,
                              end_date: str, dataset: str, schema: str,
                              cache_key: str) -> list[Path]:
        """
        Submit a Databento batch job and download results.

        Runs the blocking SDK calls in a thread so Jupyter event loops
        are not blocked.
        """
        def _submit_and_download() -> list[Path]:
            client = db.Historical(self.api_key)
            job_marker = self.raw_data_dir / f"{cache_key}.job"
            pbar = tqdm(
                total=3,
                desc=f"Batch {schema} {','.join(tickers)}",
                unit="step",
                disable=not self.verbose,
            )

            # Resume an in-flight job instead of submitting a duplicate
            if job_marker.exists():
                job_id = job_marker.read_text().strip()
                pbar.set_postfix_str("resuming")
                pbar.update(1)
            else:
                pbar.set_postfix_str("submitting")
                job = client.batch.submit_job(
                    dataset=dataset,
                    symbols=tickers,
                    schema=schema,
                    start=start_date,
                    end=end_date,
                    encoding="dbn",
                    compression="zstd",
                )
                job_id = job["id"] if isinstance(job, dict) else job.id
                job_marker.write_text(job_id)
                pbar.update(1)

            # Poll until the job is done
            import time
            pbar.set_postfix_str("waiting")
            while True:
                jobs = client.batch.list_jobs()
                status = None
                matched_job = None
                for j in jobs:
                    jid = j["id"] if isinstance(j, dict) else j.id
                    if jid == job_id:
                        matched_job = j
                        status = j["state"] if isinstance(j, dict) else j.state
                        break
                if status and status in ("done", "expired"):
                    break
                # Show progress percentage when processing, if available
                pct = None
                if matched_job is not None:
                    for key in ("percent_complete", "progress"):
                        val = (matched_job.get(key) if isinstance(matched_job, dict)
                               else getattr(matched_job, key, None))
                        if val is not None:
                            pct = val
                            break
                if status == "processing" and pct is not None:
                    pbar.set_postfix_str(f"processing {pct}%")
                else:
                    pbar.set_postfix_str(f"waiting ({status})")
                time.sleep(10)

            pbar.set_postfix_str("downloading")
            pbar.update(1)
            files = client.batch.download(
                job_id=job_id,
                output_dir=str(self.raw_data_dir),
            )
            pbar.set_postfix_str("done")
            pbar.update(1)
            pbar.close()

            dbn_files = [Path(f) for f in files if str(f).endswith(".dbn.zst")]
            if not dbn_files:
                raise FileNotFoundError(
                    f"Batch job {job_id} produced no .dbn.zst files."
                )

            # Rename each file to include job_id, cache_key, and index
            traced_paths: list[Path] = []
            for i, downloaded in enumerate(dbn_files):
                traced_path = self.raw_data_dir / f"{job_id}_{cache_key}_{i}.dbn.zst"
                downloaded.rename(traced_path)
                traced_paths.append(traced_path)

            # Clean up the job marker now that the data is on disk
            job_marker.unlink(missing_ok=True)

            return traced_paths

        return await asyncio.to_thread(_submit_and_download)

    async def get_data(
        self,
        tickers: Union[str, Iterable[str]],
        start_date: str,
        end_date: Optional[str] = None,
        dataset: str = "XNAS.ITCH",
        schema: str = "ohlcv-1m",
        cache_only: bool = False,
    ) -> tuple[pl.DataFrame, dict[int, str]]:
        """
        Fetch data through a three-layer cache.

        ``end_date`` is exclusive and defaults to today. Returns
        ``(df, mappings)`` where ``df`` has time columns exactly as Databento
        delivers them (all hours, no timezone conversion, no market-hours
        filtering) and ``mappings`` is ``{instrument_id: raw_symbol}`` -- empty
        when the parquet cache is hit but the raw ``.dbn.zst`` files have been
        cleared.

        Raises
        ------
        FileNotFoundError
            If ``cache_only`` is True and no cached data exists.
        """
        if end_date is None:
            end_date = datetime.today().strftime("%Y-%m-%d")

        ticker_list = self._normalize_tickers(tickers)
        cache_key = self._generate_cache_key(
            ticker_list, start_date, end_date, dataset, schema
        )

        # Layer 1: parquet cache
        parquet_path = self._get_cache_path(cache_key)
        if parquet_path.exists():
            if self.verbose:
                print(f"[GenericAdapter] Parquet cache hit: {parquet_path.name}")
            df = pl.read_parquet(parquet_path)
            raw_paths = self._find_raw_path(cache_key)
            mappings = self._merge_mappings(raw_paths) if raw_paths else {}
            return df, mappings

        # Layer 2: raw dbn.zst cache
        raw_paths = self._find_raw_path(cache_key)
        if raw_paths:
            if self.verbose:
                print(f"[GenericAdapter] Raw DBN cache hit: {len(raw_paths)} file(s)")
            df, mappings = self._load(raw_paths)
            df.write_parquet(parquet_path)
            return df, mappings

        # Layer 3: Databento batch API
        if cache_only:
            raise FileNotFoundError(
                f"No cached data for {ticker_list} ({start_date} to {end_date}), "
                f"dataset={dataset}, schema={schema}. "
                "Set cache_only=False to download from Databento."
            )

        raw_paths = await self._batch_download(
            ticker_list, start_date, end_date, dataset, schema, cache_key
        )
        df, mappings = self._load(raw_paths)
        df.write_parquet(parquet_path)
        return df, mappings
