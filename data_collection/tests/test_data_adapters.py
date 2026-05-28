import polars as pl
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
