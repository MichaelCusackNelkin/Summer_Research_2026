"""Historical replay feed — a ``LiveFeed`` stand-in for cached Databento data.

Lets the user dress-rehearse the daily pipeline against a specific past
trading day without paying for the live imbalance subscription:

    .venv/bin/python -m live.daily_orchestrator \
        --replay-date 2026-05-15 --paper

The rest of the pipeline (feature engineering, alpha, MVO, share sizing,
broker submission) runs UNCHANGED — that is the point. ``ReplayFeed``
is API-compatible with ``live_feed.LiveFeed`` for the three methods
``_build_live_day_frame`` calls: ``start()``, ``stop()``, ``snapshot()``,
plus ``stats()`` for the log line.

Data source: ``DatabentoGenericAdapter`` — same three-layer cache the
retrain script uses. Cache key is ``{tickers, [date, date+1), dataset,
schema}``. A single-day historical pull is CHEAP (historical, not live).
On the first replay of a new date the parquet layer misses and a
historical batch job materialises it; subsequent runs hit the parquet
cache.

Snapshot output MUST match the shapes ``imbalance_records_to_frame`` and
``tbbo_records_to_frame`` produce so ``_build_live_day_frame`` cannot
tell the difference:

- imbalance: ``ts_event`` (Datetime[ns, America/New_York]), ``symbol``
  (Categorical), ``side`` (Utf8: 'B'/'A'/'N'), ``total_imbalance_qty``,
  ``paired_qty``, ``ref_price``.
- tbbo:      ``ts_event`` (same tz), ``symbol`` (Categorical),
  ``price``, ``size``, ``ask_px_00``, ``bid_px_00``.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import date, timedelta
from typing import Iterable, Optional, Union

import polars as pl

from data_collection.data_adapters import DatabentoGenericAdapter
from live.cache import CoverageResolver, format_coverage_decision

# Reuse — do not duplicate — the B/A/N normalisation. LiveFeed already
# owns the canonical mapping.
from live.live_feed import _decode_side

logger = logging.getLogger(__name__)


class ReplayFeed:
    """Adapter-backed historical replay of one trading day.

    Parameters
    ----------
    adapter : DatabentoGenericAdapter
        Same instance the retrain script uses. Cache is shared.
    universe : list[str]
        Active universe. Passed verbatim to the adapter so the cache key
        matches any prior materialisation.
    replay_date : datetime.date | str
        The trading day to replay. Databento's ``end_date`` is exclusive,
        so we always fetch ``[replay_date, replay_date + 1 day)``.
    dataset : str, default ``"XNAS.ITCH"``
        Databento dataset.
    schemas : Iterable[str], default ``("imbalance", "tbbo")``
        Schemas to serve. Must be a subset of the schemas the LIVE feed
        served (``_build_live_day_frame`` only reads these two).
    """

    _SUPPORTED_SCHEMAS = frozenset({"imbalance", "tbbo"})

    _IMBALANCE_COLUMNS: tuple[str, ...] = (
        "ts_event", "symbol", "side",
        "total_imbalance_qty", "paired_qty", "ref_price",
    )
    _TBBO_COLUMNS: tuple[str, ...] = (
        "ts_event", "symbol", "price", "size", "ask_px_00", "bid_px_00",
    )

    def __init__(
        self,
        adapter: DatabentoGenericAdapter,
        universe: list[str],
        replay_date: Union[date, str],
        dataset: str = "XNAS.ITCH",
        schemas: Iterable[str] = ("imbalance", "tbbo"),
        resolver: Optional[CoverageResolver] = None,
        allow_fetch: bool = False,
    ) -> None:
        self.adapter = adapter
        self.universe = list(universe)
        if isinstance(replay_date, str):
            replay_date = date.fromisoformat(replay_date)
        self.replay_date: date = replay_date
        self.dataset = dataset
        self.schemas = tuple(schemas)
        for s in self.schemas:
            if s not in self._SUPPORTED_SCHEMAS:
                raise ValueError(
                    f"Unsupported replay schema {s!r}. "
                    f"Supported: {sorted(self._SUPPORTED_SCHEMAS)}"
                )
        # Route every historical read through the coverage resolver so a
        # replay day that lives inside an existing (differently-keyed)
        # parquet is served in place, not re-fetched from Databento. The
        # resolver refuses to fetch on MISS unless the operator opts in
        # via ``allow_fetch=True`` (wired from the orchestrator's
        # ``--allow-fetch`` flag). Construction is deferred to first use
        # so callers passing lightweight test stubs for ``adapter``
        # (that never actually reach ``_ensure_materialised`` /
        # ``_build_frame``) don't need to satisfy the resolver's contract.
        self._resolver_override = resolver
        self._resolver_cached: Optional[CoverageResolver] = resolver
        self.allow_fetch = allow_fetch
        self._started = False
        self._stopped = False
        self._materialised: dict[str, bool] = {s: False for s in self.schemas}

    # ------------------------------------------------------------------
    # Public API — mirrors LiveFeed
    # ------------------------------------------------------------------

    @property
    def start_iso(self) -> str:
        return self.replay_date.isoformat()

    @property
    def end_iso(self) -> str:
        # Databento's end_date is exclusive; a single-day pull covers
        # exactly ``[replay_date, replay_date + 1 day)``.
        return (self.replay_date + timedelta(days=1)).isoformat()

    def start(self) -> None:
        """Ensure both schemas are materialised into the parquet cache.

        Idempotent — the adapter short-circuits on a parquet cache hit,
        so re-running does no work. This is where the very first replay
        of a new date pays a one-off Databento historical-batch cost.
        """
        if self._started:
            return
        for schema in self.schemas:
            self._ensure_materialised(schema)
        self._started = True
        logger.info(
            "replay feed ready",
            extra={
                "replay_date": self.start_iso,
                "dataset": self.dataset,
                "schemas": list(self.schemas),
                "n_symbols": len(self.universe),
            },
        )

    def stop(self, timeout: float = 0.0) -> None:  # noqa: D401  API parity
        """No-op — no background thread, no socket to close."""
        self._stopped = True

    def snapshot(self) -> dict[str, pl.DataFrame]:
        """Return one polars frame per subscribed schema.

        Column layout matches the live path (see module docstring). The
        frames are the FULL historical day for the requested universe —
        downstream ``compute_*`` helpers apply the time-window filters
        (15:40-15:50, 15:50:00-15:50:05, etc.) the same way they do on
        the buffered live records.
        """
        out: dict[str, pl.DataFrame] = {}
        for schema in self.schemas:
            if not self._materialised.get(schema):
                # Called snapshot() without start() — recover gracefully by
                # materialising on demand. Costs one historical pull if the
                # parquet cache is cold.
                self._ensure_materialised(schema)
            out[schema] = self._build_frame(schema)
        return out

    def stats(self) -> dict[str, object]:
        """Structured stats for the ``snapshotted live feed`` log entry."""
        return {
            "mode": "replay",
            "replay_date": self.start_iso,
            "dataset": self.dataset,
            "schemas": list(self.schemas),
            "universe_size": len(self.universe),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @property
    def resolver(self) -> CoverageResolver:
        """Lazily construct the coverage resolver on first use."""
        if self._resolver_cached is None:
            self._resolver_cached = CoverageResolver(self.adapter)
        return self._resolver_cached

    def _ensure_materialised(self, schema: str) -> None:
        """Route the request through the coverage resolver.

        The resolver first checks the adapter's exact-hash cache, then
        falls to the coverage manifest (any existing parquet whose
        contents cover the replay day). Only if neither can serve the
        request AND ``allow_fetch=True`` does it call Databento — the
        default is to refuse and raise ``CoverageError`` so a replay
        cannot silently trigger a historical batch charge.
        """
        # Plan first so the operator sees a per-schema HIT/MISS line at
        # INFO level BEFORE materialize() either succeeds or raises. On a
        # MISS this line names the closest parquet and the shortfall so
        # the failure isn't opaque. Costs one extra manifest lookup, which
        # is trivial vs the historical fetch itself.
        decision = self.resolver.plan(
            tickers=self.universe,
            start_date=self.start_iso,
            end_date=self.end_iso,
            dataset=self.dataset,
            schema=schema,
            required_symbols=self.universe,
        )
        logger.info(
            "coverage plan — %s",
            format_coverage_decision(schema, decision),
        )
        asyncio.run(
            self.resolver.materialize(
                tickers=self.universe,
                start_date=self.start_iso,
                end_date=self.end_iso,
                dataset=self.dataset,
                schema=schema,
                required_symbols=self.universe,
                allow_fetch=self.allow_fetch,
            )
        )
        self._materialised[schema] = True

    def _build_frame(self, schema: str) -> pl.DataFrame:
        """Read the cached parquet lazily, filter to the universe and the
        replay date, tz-convert to America/New_York, and normalise
        ``side`` (imbalance only) via ``_decode_side``.
        """
        lf = self.resolver.resolve_lazy(
            tickers=self.universe,
            start_date=self.start_iso,
            end_date=self.end_iso,
            dataset=self.dataset,
            schema=schema,
            required_symbols=self.universe,
            allow_fetch=False,
        )
        if schema == "imbalance":
            return self._collect_imbalance(lf)
        if schema == "tbbo":
            return self._collect_tbbo(lf)
        raise ValueError(f"unsupported schema for replay: {schema!r}")

    def _collect_imbalance(self, lf: pl.LazyFrame) -> pl.DataFrame:
        df = (
            lf
            .select(list(self._IMBALANCE_COLUMNS))
            .with_columns(
                pl.col("ts_event").dt.convert_time_zone("America/New_York"),
                pl.col("symbol").cast(pl.Categorical),
            )
            .filter(
                pl.col("symbol").is_in(self.universe),
                # Belt-and-braces window: even if the adapter's cache spans
                # more than one day (some parquets are broader than the
                # exact ``[date, date+1)`` key implies), pin to the replay
                # date so features/alpha only see one day.
                pl.col("ts_event").dt.date() == self.replay_date,
            )
            .collect(engine="streaming")
        )
        # Normalise side to the B/A/N scheme downstream expects. Reuse
        # LiveFeed's helper — do not duplicate the mapping.
        if df.is_empty():
            return df
        df = df.with_columns(
            pl.col("side")
            .map_elements(_decode_side, return_dtype=pl.Utf8)
            .alias("side")
        )
        return df

    def _collect_tbbo(self, lf: pl.LazyFrame) -> pl.DataFrame:
        return (
            lf
            .select(list(self._TBBO_COLUMNS))
            .with_columns(
                pl.col("ts_event").dt.convert_time_zone("America/New_York"),
                pl.col("symbol").cast(pl.Categorical),
            )
            .filter(
                pl.col("symbol").is_in(self.universe),
                pl.col("ts_event").dt.date() == self.replay_date,
            )
            .collect(engine="streaming")
        )
