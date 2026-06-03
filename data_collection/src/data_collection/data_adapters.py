import asyncio
import gc
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

    @staticmethod
    def _decode_one_to_parquet(raw_path: Path, out_path: Path) -> None:
        """
        Decode a single .dbn.zst file and write it to a parquet at
        ``out_path``.

        Streams through ``DBNStore.to_parquet`` so peak memory is one
        ``PARQUET_CHUNK_SIZE`` chunk (65,536 records as of databento
        0.78), not the whole decoded file. Internally the SDK pulls a
        ``DataFrameIterator`` from ``to_df(count=PARQUET_CHUNK_SIZE)``
        and writes each chunk through a ``pyarrow.parquet.ParquetWriter``.

        Schema parity with the previous ``to_df().reset_index()`` path:

        - ``price_type="float"``, ``pretty_ts=True``, ``map_symbols=True``
          all match the ``to_df`` defaults that the old path used. Prices
          are floats (dollars). Timestamps are tz-aware UTC nanoseconds.
        - ``to_df`` puts ``ts_recv`` (else ``ts_event``) on the index;
          ``reset_index()`` moves it to the *first* column. ``to_parquet``
          via ``pa.Schema.from_pandas`` instead writes the index column
          *last*. To preserve the historical column order we re-sink the
          parquet with the index column moved to the front. The re-sink is
          a streaming lazy scan, so memory stays bounded.
        """
        staged = out_path.with_suffix(".staged.parquet")
        try:
            data = db.DBNStore.from_file(raw_path)
            data.to_parquet(staged)

            cols = pl.scan_parquet(staged).collect_schema().names()
            # `to_df` indexes on ts_recv when present, otherwise ts_event.
            if "ts_recv" in cols:
                index_col = "ts_recv"
            elif "ts_event" in cols:
                index_col = "ts_event"
            else:
                index_col = None

            if index_col is not None and cols[0] != index_col:
                reordered = [index_col] + [c for c in cols if c != index_col]
                pl.scan_parquet(staged).select(reordered).sink_parquet(out_path)
                staged.unlink(missing_ok=True)
            else:
                staged.replace(out_path)
        except BaseException:
            staged.unlink(missing_ok=True)
            raise

    def _load(self, raw_paths: list[Path], parquet_path: Path) -> None:
        """
        Decode one or more .dbn.zst files and write a single parquet at
        ``parquet_path``.

        Each raw file is streamed through ``_decode_one_to_parquet``
        (``DBNStore.to_parquet`` under the hood), which writes one chunk
        at a time so peak memory is one decoded chunk -- not one whole
        file and not the full concatenated frame. Per-file temp parquets
        live next to the destination and are then lazy-concat-sunk into
        ``parquet_path`` (also chunked). Temp files are removed on
        success and on failure.

        Time columns are written exactly as Databento delivers them -- all
        hours, no timezone conversion, no market-hours filtering.
        """
        tmp_paths: list[Path] = []
        try:
            for i, p in enumerate(raw_paths):
                tmp = parquet_path.with_suffix(f".part{i}.parquet")
                self._decode_one_to_parquet(p, tmp)
                tmp_paths.append(tmp)
                # Reclaim the decoded frames before the next file is read.
                gc.collect()

            if len(tmp_paths) == 1:
                # Single-file case: rename the temp into place.
                tmp_paths[0].replace(parquet_path)
                tmp_paths = []
            else:
                lf = pl.concat(
                    [pl.scan_parquet(tp) for tp in tmp_paths],
                    how="vertical",
                )
                lf.sink_parquet(parquet_path)
        finally:
            for tp in tmp_paths:
                tp.unlink(missing_ok=True)

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
                desc=f"Batch {schema}",
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
        materialize_only: bool = False,
    ) -> Optional[pl.DataFrame]:
        """
        Fetch data through a three-layer cache.

        ``end_date`` is exclusive and defaults to today. Returns a DataFrame
        with time columns exactly as Databento delivers them (all hours, no
        timezone conversion, no market-hours filtering).

        ``materialize_only=True`` skips the final eager ``pl.read_parquet`` --
        ensures the parquet exists on disk and returns ``None``. Use this when
        the caller only needs the cache populated and will read it back via
        ``scan_data`` or ``DBNStore.from_file`` later.

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
            return None if materialize_only else pl.read_parquet(parquet_path)

        # Layer 2: raw dbn.zst cache
        raw_paths = self._find_raw_path(cache_key)
        if raw_paths:
            if self.verbose:
                print(f"[GenericAdapter] Raw DBN cache hit: {len(raw_paths)} file(s)")
            self._load(raw_paths, parquet_path)
            return None if materialize_only else pl.read_parquet(parquet_path)

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
        self._load(raw_paths, parquet_path)
        return None if materialize_only else pl.read_parquet(parquet_path)

    def scan_data(
        self,
        tickers: Union[str, Iterable[str]],
        start_date: str,
        end_date: Optional[str] = None,
        dataset: str = "XNAS.ITCH",
        schema: str = "ohlcv-1m",
    ) -> pl.LazyFrame:
        """
        Lazy sibling of :meth:`get_data` for the parquet-cache layer only.

        Returns a ``pl.LazyFrame`` backed by ``pl.scan_parquet`` so callers can
        chain column pruning and predicate filters that polars pushes down
        into the parquet read -- avoiding the memory blowup of
        ``pl.read_parquet`` on multi-GB caches.

        Unlike :meth:`get_data`, this method is synchronous: it is a pure
        local-file lookup and never decodes raw DBN or hits the Databento
        API. If the parquet cache is missing, it raises ``FileNotFoundError``
        and tells the caller to materialize the cache via ``get_data`` first.

        Raises
        ------
        FileNotFoundError
            If no parquet cache exists for the given arguments.
        """
        if end_date is None:
            end_date = datetime.today().strftime("%Y-%m-%d")

        ticker_list = self._normalize_tickers(tickers)
        cache_key = self._generate_cache_key(
            ticker_list, start_date, end_date, dataset, schema
        )

        parquet_path = self._get_cache_path(cache_key)
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"No parquet cache for {ticker_list} ({start_date} to {end_date}), "
                f"dataset={dataset}, schema={schema}. "
                "scan_data is lazy-only and will not download; "
                "call get_data(...) once first to materialize the parquet cache."
            )

        if self.verbose:
            print(f"[GenericAdapter] scan_parquet: {parquet_path.name}")
        return pl.scan_parquet(parquet_path)
