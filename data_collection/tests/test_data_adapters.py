from pathlib import Path
from unittest.mock import patch

import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_collection.data_adapters import DataAdapter, DatabentoGenericAdapter


# ---------------------------------------------------------------------------
# DataAdapter ABC
# ---------------------------------------------------------------------------

class _MinimalAdapter(DataAdapter):
    """Concrete subclass implementing the required abstracts."""

    async def get_data(self, *args, **kwargs):
        return None

    def clear_cache(self) -> None:
        return None


class _MissingGetData(DataAdapter):
    def clear_cache(self) -> None:
        return None


class _MissingClearCache(DataAdapter):
    async def get_data(self, *args, **kwargs):
        return None


def test_data_adapter_is_abstract():
    with pytest.raises(TypeError):
        DataAdapter("/tmp/whatever")  # type: ignore[abstract]


def test_minimal_subclass_constructs(tmp_path):
    adapter = _MinimalAdapter(tmp_path)
    assert adapter.cache_dir == tmp_path


def test_subclass_missing_get_data_cannot_construct(tmp_path):
    with pytest.raises(TypeError):
        _MissingGetData(tmp_path)  # type: ignore[abstract]


def test_subclass_missing_clear_cache_cannot_construct(tmp_path):
    with pytest.raises(TypeError):
        _MissingClearCache(tmp_path)  # type: ignore[abstract]


def test_cache_dir_is_created(tmp_path):
    new_dir = tmp_path / "new" / "nested"
    assert not new_dir.exists()
    adapter = _MinimalAdapter(new_dir)
    assert adapter.cache_dir.exists()
    assert adapter.cache_dir.is_dir()


# ---------------------------------------------------------------------------
# DatabentoGenericAdapter -- construction
# ---------------------------------------------------------------------------

def test_databento_adapter_creates_raw_subdir(tmp_path):
    adapter = DatabentoGenericAdapter("fake-key", tmp_path / "cache")
    assert adapter.raw_data_dir == tmp_path / "cache" / "raw_dbn"
    assert adapter.raw_data_dir.exists()
    assert adapter.raw_data_dir.is_dir()


# ---------------------------------------------------------------------------
# _normalize_tickers
# ---------------------------------------------------------------------------

def _adapter(tmp_path):
    return DatabentoGenericAdapter("fake-key", tmp_path, verbose=False)


def test_normalize_tickers_single_str(tmp_path):
    a = _adapter(tmp_path)
    assert a._normalize_tickers("aapl") == ["AAPL"]


def test_normalize_tickers_list(tmp_path):
    a = _adapter(tmp_path)
    assert a._normalize_tickers(["msft", "AAPL", "googl"]) == ["AAPL", "GOOGL", "MSFT"]


def test_normalize_tickers_mixed_case_dedup(tmp_path):
    a = _adapter(tmp_path)
    assert a._normalize_tickers(["aapl", "AAPL", "Aapl", "MSFT"]) == ["AAPL", "MSFT"]


def test_normalize_tickers_sorted(tmp_path):
    a = _adapter(tmp_path)
    assert a._normalize_tickers(["ZZZZ", "AAAA", "MMMM"]) == ["AAAA", "MMMM", "ZZZZ"]


# ---------------------------------------------------------------------------
# _generate_cache_key
# ---------------------------------------------------------------------------

def test_cache_key_deterministic(tmp_path):
    a = _adapter(tmp_path)
    k1 = a._generate_cache_key(["AAPL", "MSFT"], "2026-01-01", "2026-01-05")
    k2 = a._generate_cache_key(["AAPL", "MSFT"], "2026-01-01", "2026-01-05")
    assert k1 == k2


def test_cache_key_order_independent(tmp_path):
    a = _adapter(tmp_path)
    k1 = a._generate_cache_key(["AAPL", "MSFT"], "2026-01-01", "2026-01-05")
    k2 = a._generate_cache_key(["MSFT", "AAPL"], "2026-01-01", "2026-01-05")
    assert k1 == k2


def test_cache_key_changes_with_dataset(tmp_path):
    a = _adapter(tmp_path)
    k1 = a._generate_cache_key(["AAPL"], "2026-01-01", "2026-01-05", dataset="XNAS.ITCH")
    k2 = a._generate_cache_key(["AAPL"], "2026-01-01", "2026-01-05", dataset="DBEQ.BASIC")
    assert k1 != k2


def test_cache_key_changes_with_schema(tmp_path):
    a = _adapter(tmp_path)
    k1 = a._generate_cache_key(["AAPL"], "2026-01-01", "2026-01-05", schema="ohlcv-1m")
    k2 = a._generate_cache_key(["AAPL"], "2026-01-01", "2026-01-05", schema="mbp-10")
    assert k1 != k2


def test_cache_key_changes_with_dates(tmp_path):
    a = _adapter(tmp_path)
    k1 = a._generate_cache_key(["AAPL"], "2026-01-01", "2026-01-05")
    k2 = a._generate_cache_key(["AAPL"], "2026-01-02", "2026-01-05")
    k3 = a._generate_cache_key(["AAPL"], "2026-01-01", "2026-01-06")
    assert k1 != k2
    assert k1 != k3
    assert k2 != k3


# ---------------------------------------------------------------------------
# _get_cache_path
# ---------------------------------------------------------------------------

def test_get_cache_path_returns_parquet_under_cache_dir(tmp_path):
    a = _adapter(tmp_path)
    path = a._get_cache_path("abc123")
    assert path == tmp_path / "abc123.parquet"
    assert path.suffix == ".parquet"


# ---------------------------------------------------------------------------
# clear_cache
# ---------------------------------------------------------------------------

def test_clear_cache_removes_parquet_preserves_dbn(tmp_path):
    a = _adapter(tmp_path)

    parquet_file = tmp_path / "key1.parquet"
    parquet_file.write_bytes(b"fake-parquet")

    other_parquet = tmp_path / "key2.parquet"
    other_parquet.write_bytes(b"fake-parquet-2")

    dbn_file = tmp_path / "raw_dbn" / "job_key1_0.dbn.zst"
    dbn_file.write_bytes(b"fake-dbn")

    a.clear_cache()

    assert not parquet_file.exists()
    assert not other_parquet.exists()
    assert dbn_file.exists()
    assert dbn_file.read_bytes() == b"fake-dbn"


# ---------------------------------------------------------------------------
# get_data -- cache_only behavior (no network)
# ---------------------------------------------------------------------------

async def test_get_data_cache_only_miss_raises(tmp_path):
    a = _adapter(tmp_path)
    with pytest.raises(FileNotFoundError):
        await a.get_data(
            tickers=["AAPL"],
            start_date="2026-01-01",
            end_date="2026-01-05",
            cache_only=True,
        )


async def test_get_data_parquet_cache_hit(tmp_path):
    a = _adapter(tmp_path)
    ticker_list = a._normalize_tickers(["AAPL"])
    cache_key = a._generate_cache_key(
        ticker_list, "2026-01-01", "2026-01-05",
        dataset="XNAS.ITCH", schema="ohlcv-1m",
    )
    expected = pl.DataFrame({"ts_event": [1, 2, 3], "price": [10.0, 11.0, 12.0]})
    parquet_path = a._get_cache_path(cache_key)
    expected.write_parquet(parquet_path)

    df = await a.get_data(
        tickers=["AAPL"],
        start_date="2026-01-01",
        end_date="2026-01-05",
        cache_only=True,
    )
    assert df.equals(expected)


# ---------------------------------------------------------------------------
# scan_data -- lazy parquet-only layer
# ---------------------------------------------------------------------------

def test_scan_data_cache_hit_returns_lazyframe(tmp_path):
    a = _adapter(tmp_path)
    ticker_list = a._normalize_tickers(["AAPL"])
    cache_key = a._generate_cache_key(
        ticker_list, "2026-01-01", "2026-01-05",
        dataset="XNAS.ITCH", schema="ohlcv-1m",
    )
    expected = pl.DataFrame({"ts_event": [1, 2, 3], "price": [10.0, 11.0, 12.0]})
    expected.write_parquet(a._get_cache_path(cache_key))

    lf = a.scan_data(
        tickers=["AAPL"],
        start_date="2026-01-01",
        end_date="2026-01-05",
    )
    assert isinstance(lf, pl.LazyFrame)
    assert lf.collect().equals(expected)


async def test_scan_data_matches_get_data_for_same_args(tmp_path):
    a = _adapter(tmp_path)
    ticker_list = a._normalize_tickers(["AAPL"])
    cache_key = a._generate_cache_key(
        ticker_list, "2026-01-01", "2026-01-05",
        dataset="XNAS.ITCH", schema="ohlcv-1m",
    )
    expected = pl.DataFrame({"ts_event": [1, 2, 3], "price": [10.0, 11.0, 12.0]})
    expected.write_parquet(a._get_cache_path(cache_key))

    eager = await a.get_data(
        tickers=["AAPL"],
        start_date="2026-01-01",
        end_date="2026-01-05",
        cache_only=True,
    )
    lazy = a.scan_data(
        tickers=["AAPL"],
        start_date="2026-01-01",
        end_date="2026-01-05",
    ).collect()
    assert lazy.equals(eager)


def test_scan_data_missing_cache_raises_filenotfounderror(tmp_path):
    a = _adapter(tmp_path)
    with pytest.raises(FileNotFoundError) as excinfo:
        a.scan_data(
            tickers=["AAPL"],
            start_date="2026-01-01",
            end_date="2026-01-05",
        )
    msg = str(excinfo.value)
    assert "get_data" in msg
    assert "parquet" in msg.lower()


def test_scan_data_verbose_prints_filename(tmp_path, capsys):
    a = DatabentoGenericAdapter("fake-key", tmp_path, verbose=True)
    ticker_list = a._normalize_tickers(["AAPL"])
    cache_key = a._generate_cache_key(
        ticker_list, "2026-01-01", "2026-01-05",
        dataset="XNAS.ITCH", schema="ohlcv-1m",
    )
    pl.DataFrame({"x": [1]}).write_parquet(a._get_cache_path(cache_key))

    a.scan_data(
        tickers=["AAPL"],
        start_date="2026-01-01",
        end_date="2026-01-05",
    )
    out = capsys.readouterr().out
    assert "[GenericAdapter] scan_parquet:" in out
    assert f"{cache_key}.parquet" in out


def test_scan_data_quiet_when_not_verbose(tmp_path, capsys):
    a = _adapter(tmp_path)  # verbose=False
    ticker_list = a._normalize_tickers(["AAPL"])
    cache_key = a._generate_cache_key(
        ticker_list, "2026-01-01", "2026-01-05",
        dataset="XNAS.ITCH", schema="ohlcv-1m",
    )
    pl.DataFrame({"x": [1]}).write_parquet(a._get_cache_path(cache_key))

    a.scan_data(
        tickers=["AAPL"],
        start_date="2026-01-01",
        end_date="2026-01-05",
    )
    out = capsys.readouterr().out
    assert out == ""


# ---------------------------------------------------------------------------
# _decode_one_to_parquet / _load -- streaming decode, schema parity with the
# legacy ``DBNStore.to_df().reset_index()`` path
# ---------------------------------------------------------------------------

class _FakeDBNStore:
    """
    Minimal stand-in for ``databento.DBNStore`` for decode-path tests.

    Mirrors the parts of the SDK contract that ``_decode_one_to_parquet``
    depends on:

    - ``to_df()`` returns a single ``pd.DataFrame`` (index is whatever the
      caller set on the source frames).
    - ``to_df(count=N)`` returns an iterator yielding ``pd.DataFrame``
      chunks -- this is the streaming hook the real SDK exposes.
    - ``to_parquet(path)`` writes the data using the same per-chunk
      ``pyarrow.parquet.ParquetWriter`` pattern as the real SDK, so we
      exercise the ``pa.Schema.from_pandas`` index-column promotion that
      the new decode path has to reorder to match the legacy layout.
    """

    def __init__(self, frames: list[pd.DataFrame]) -> None:
        self._frames = frames

    def to_df(self, **kwargs):
        if kwargs.get("count") is None:
            return pd.concat(self._frames)
        return iter(self._frames)

    def to_parquet(self, path, **kwargs):
        writer = None
        try:
            for frame in self.to_df(count=65536):
                if writer is None:
                    schema = pa.Schema.from_pandas(frame)
                    writer = pq.ParquetWriter(str(path), schema=schema)
                writer.write_table(pa.Table.from_pandas(frame, schema=schema))
        finally:
            if writer is not None:
                writer.close()


def _ohlcv_frame_ts_event_index() -> pd.DataFrame:
    """OHLCV-shaped frame indexed on ``ts_event`` (e.g., ohlcv-1m / ohlcv-1s)."""
    idx = pd.DatetimeIndex(
        ["2026-01-01 14:30", "2026-01-01 14:31", "2026-01-01 14:32"],
        name="ts_event",
        tz="UTC",
    )
    return pd.DataFrame(
        {
            "open":   [10.0, 10.1, 10.2],
            "high":   [10.5, 10.6, 10.7],
            "low":    [9.9,  10.0, 10.05],
            "close":  [10.1, 10.2, 10.6],
            "volume": [1000, 1100, 1200],
            "symbol": ["AAPL", "AAPL", "AAPL"],
        },
        index=idx,
    )


def _tbbo_frame_ts_recv_index() -> pd.DataFrame:
    """tbbo/mbp-1-shaped frame indexed on ``ts_recv`` with ``ts_event`` as a column."""
    idx = pd.DatetimeIndex(
        ["2026-01-01 14:30", "2026-01-01 14:31"],
        name="ts_recv",
        tz="UTC",
    )
    return pd.DataFrame(
        {
            "ts_event": pd.DatetimeIndex(
                ["2026-01-01 14:30", "2026-01-01 14:31"], tz="UTC"
            ),
            "bid_px": [9.99, 10.00],
            "ask_px": [10.01, 10.02],
            "symbol": ["AAPL", "AAPL"],
        },
        index=idx,
    )


def test_decode_one_to_parquet_matches_legacy_layout_ts_event(tmp_path):
    """ohlcv schemas index on ts_event -- decoded parquet must put it first."""
    df = _ohlcv_frame_ts_event_index()
    store = _FakeDBNStore([df.iloc[:2], df.iloc[2:]])

    raw = tmp_path / "fake.dbn.zst"
    raw.touch()
    out = tmp_path / "out.parquet"

    with patch(
        "data_collection.data_adapters.db.DBNStore.from_file",
        return_value=store,
    ):
        DatabentoGenericAdapter._decode_one_to_parquet(raw, out)

    got = pl.read_parquet(out)
    expected = pl.from_pandas(df.reset_index())

    assert got.columns == expected.columns
    assert got.schema == expected.schema
    assert got.height == expected.height
    assert got.equals(expected)


def test_decode_one_to_parquet_matches_legacy_layout_ts_recv(tmp_path):
    """tbbo/mbp schemas index on ts_recv -- decoded parquet must put it first."""
    df = _tbbo_frame_ts_recv_index()
    store = _FakeDBNStore([df])

    raw = tmp_path / "fake.dbn.zst"
    raw.touch()
    out = tmp_path / "out.parquet"

    with patch(
        "data_collection.data_adapters.db.DBNStore.from_file",
        return_value=store,
    ):
        DatabentoGenericAdapter._decode_one_to_parquet(raw, out)

    got = pl.read_parquet(out)
    expected = pl.from_pandas(df.reset_index())

    assert got.columns == expected.columns
    assert got.schema == expected.schema
    assert got.equals(expected)


def test_decode_one_to_parquet_no_staged_file_leftover(tmp_path):
    """The ``.staged.parquet`` temp must be cleaned up on success."""
    df = _ohlcv_frame_ts_event_index()
    store = _FakeDBNStore([df])

    raw = tmp_path / "fake.dbn.zst"
    raw.touch()
    out = tmp_path / "out.parquet"

    with patch(
        "data_collection.data_adapters.db.DBNStore.from_file",
        return_value=store,
    ):
        DatabentoGenericAdapter._decode_one_to_parquet(raw, out)

    assert out.exists()
    assert not (tmp_path / "out.staged.parquet").exists()


def test_decode_one_to_parquet_cleans_staged_on_failure(tmp_path):
    """A broken store must leave no staged temp behind."""

    class _BrokenStore:
        def to_parquet(self, path, **kwargs):
            # Write something, then fail -- prove cleanup runs.
            Path(path).write_bytes(b"partial")
            raise RuntimeError("boom")

    raw = tmp_path / "fake.dbn.zst"
    raw.touch()
    out = tmp_path / "out.parquet"

    with patch(
        "data_collection.data_adapters.db.DBNStore.from_file",
        return_value=_BrokenStore(),
    ):
        with pytest.raises(RuntimeError, match="boom"):
            DatabentoGenericAdapter._decode_one_to_parquet(raw, out)

    assert not out.exists()
    assert not (tmp_path / "out.staged.parquet").exists()


def test_load_concats_multiple_files_in_order(tmp_path):
    """``_load`` must produce the concat of per-file decodes, in order."""
    df = _ohlcv_frame_ts_event_index()
    df_a = df.iloc[:2]
    df_b = df.iloc[2:]
    store_a = _FakeDBNStore([df_a])
    store_b = _FakeDBNStore([df_b])

    raw_a = tmp_path / "a.dbn.zst"; raw_a.touch()
    raw_b = tmp_path / "b.dbn.zst"; raw_b.touch()
    out = tmp_path / "out.parquet"
    adapter = DatabentoGenericAdapter("k", tmp_path / "cache", verbose=False)

    def from_file(p):
        return store_a if Path(p).name == "a.dbn.zst" else store_b

    with patch(
        "data_collection.data_adapters.db.DBNStore.from_file",
        side_effect=from_file,
    ):
        adapter._load([raw_a, raw_b], out)

    got = pl.read_parquet(out)
    expected = pl.from_pandas(df.reset_index())
    assert got.equals(expected)
    # No temp parts left behind.
    assert sorted(p.name for p in tmp_path.glob("out.*")) == ["out.parquet"]
