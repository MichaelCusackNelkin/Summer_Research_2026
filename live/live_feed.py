"""Databento live client wrapper for the daily orchestrator.

Provides a small threaded façade over ``databento.Live`` that:

- Subscribes to ``imbalance`` + ``tbbo`` (or configurable schemas) for the
  active universe, resolved via ``stype_in="raw_symbol"``.
- Runs the client's callback in a background thread so the main thread can
  block until 15:50 without freezing the read side.
- Buffers records into per-schema lists of dicts (converted from the pyo3
  message structs) with thread-safe append.
- Resolves ``instrument_id -> raw_symbol`` from ``SymbolMappingMsg`` records
  emitted at session start; live records carry ``instrument_id`` only.
- Exposes a ``snapshot()`` that returns a shallow copy of the buffers as
  polars ``DataFrame``s.
- Cleanly stops the underlying client on ``stop()``.

Reconnection: the Live SDK offers ``ReconnectPolicy.RECONNECT`` — we set it
so a transient disconnect during the trading day recovers without operator
intervention. If the reconnect exhausts, the orchestrator's ``--on-feed-failure``
flag decides whether to abort or fall back to the last snapshot.
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Iterable, Optional

import databento as db
import polars as pl

logger = logging.getLogger(__name__)


class LiveFeed:
    """Background-threaded Databento live subscriber.

    Parameters
    ----------
    api_key : str
        Databento API key (``DATABENTO_API_KEY``).
    universe : list[str]
        Ticker symbols to subscribe to. Resolved with ``stype_in='raw_symbol'``.
    dataset : str
        Databento dataset. Defaults to ``XNAS.ITCH`` — matches the historical
        pull used everywhere else in the repo.
    schemas : Iterable[str]
        Live schemas to subscribe to. Defaults to ``('imbalance', 'tbbo')``,
        the two the daily feature pipeline needs.
    reconnect : bool
        If True, tell the Live client to reconnect on transient drops.
    """

    _KNOWN_SCHEMAS = frozenset({"imbalance", "tbbo", "mbp-1", "trades"})

    def __init__(
        self,
        api_key: str,
        universe: list[str],
        dataset: str = "XNAS.ITCH",
        schemas: Iterable[str] = ("imbalance", "tbbo"),
        reconnect: bool = True,
    ) -> None:
        self.api_key = api_key
        self.universe = list(universe)
        self.dataset = dataset
        self.schemas = tuple(schemas)
        for s in self.schemas:
            if s not in self._KNOWN_SCHEMAS:
                raise ValueError(
                    f"Unknown schema {s!r}. Known: {sorted(self._KNOWN_SCHEMAS)}"
                )
        self.reconnect = reconnect

        # Thread synchronisation. The Databento callback fires on the client's
        # internal thread; ``snapshot()`` reads on the main thread.
        self._lock = threading.Lock()
        self._id_to_symbol: dict[int, str] = {}
        self._buffers: dict[str, list[dict]] = {s: [] for s in self.schemas}
        self._record_count = 0
        self._error_count = 0
        self._started = False
        self._stopped = False

        # Constructed lazily so tests can instantiate LiveFeed without
        # hitting the network.
        self._client: Optional[db.Live] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the connection, submit the subscriptions, and begin
        streaming into the buffers.

        Returns immediately — records arrive on the Live client's internal
        thread via ``add_callback``.
        """
        if self._started:
            return
        reconnect_policy = (
            db.common.enums.ReconnectPolicy.RECONNECT
            if self.reconnect
            else db.common.enums.ReconnectPolicy.NONE
        )
        self._client = db.Live(key=self.api_key, reconnect_policy=reconnect_policy)
        for schema in self.schemas:
            logger.info(
                "subscribing to Databento live",
                extra={"schema": schema, "n_symbols": len(self.universe)},
            )
            self._client.subscribe(
                dataset=self.dataset,
                schema=schema,
                symbols=self.universe,
                stype_in="raw_symbol",
            )
        self._client.add_callback(self._on_record, exception_callback=self._on_error)
        self._client.start()
        self._started = True
        logger.info("live feed started", extra={"dataset": self.dataset})

    def stop(self, timeout: float = 5.0) -> None:
        """Close the connection cleanly. Idempotent."""
        if not self._started or self._stopped:
            return
        try:
            self._client.stop()
        except Exception as e:
            logger.warning("live stop() failed", extra={"error": str(e)})
        try:
            # block_for_close waits for the internal reader to drain
            self._client.block_for_close(timeout=timeout)
        except Exception as e:
            logger.warning("block_for_close failed", extra={"error": str(e)})
        self._stopped = True
        logger.info("live feed stopped")

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, pl.DataFrame]:
        """Return a dict of ``{schema: polars.DataFrame}``.

        The internal buffers are copied (not moved) so a slow downstream
        does not race a still-arriving stream. For the daily orchestrator
        we call ``stop()`` before ``snapshot()`` so nothing else is
        appending, but it is safe either way.
        """
        from live.features import imbalance_records_to_frame, tbbo_records_to_frame
        with self._lock:
            buffers = {s: list(v) for s, v in self._buffers.items()}
        frames: dict[str, pl.DataFrame] = {}
        for schema, recs in buffers.items():
            if schema == "imbalance":
                frames[schema] = imbalance_records_to_frame(recs)
            elif schema in ("tbbo", "mbp-1"):
                frames[schema] = tbbo_records_to_frame(recs)
            else:
                # Fall back to raw dict->frame (best-effort)
                frames[schema] = pl.DataFrame(recs) if recs else pl.DataFrame()
        return frames

    def stats(self) -> dict[str, object]:
        with self._lock:
            per_schema = {s: len(v) for s, v in self._buffers.items()}
        return {
            "records_processed": self._record_count,
            "errors": self._error_count,
            "per_schema_counts": per_schema,
            "symbols_mapped": len(self._id_to_symbol),
        }

    # ------------------------------------------------------------------
    # Internal callbacks
    # ------------------------------------------------------------------

    def _on_record(self, record) -> None:
        """Route each Databento record to its buffer.

        Runs on the Live client's internal thread. Keep this fast — no I/O,
        no logging on the hot path (only on the rare error branch).
        """
        try:
            # SymbolMappingMsg establishes instrument_id -> raw_symbol
            # bindings at session start (and again on re-subscription).
            #
            # Databento's live gateway always emits records with
            # ``stype_out=instrument_id``, so the OUTPUT symbol on the mapping
            # message is the numeric instrument-id-as-string (e.g. "12345")
            # and the human ticker lives in ``stype_in_symbol``. Mirror the
            # SDK's own selection rule in
            # ``databento/common/symbology.py:409-412``: prefer the OUT symbol,
            # but fall back to the IN symbol when OUT is all-digits.
            from databento_dbn import SymbolMappingMsg
            if isinstance(record, SymbolMappingMsg):
                sym = record.stype_out_symbol
                if sym is None or sym.isdigit():
                    sym = record.stype_in_symbol
                with self._lock:
                    self._id_to_symbol[record.instrument_id] = sym
                return

            record_dict = self._to_dict(record)
            if record_dict is None:
                return
            schema = record_dict.pop("_schema")
            with self._lock:
                if schema in self._buffers:
                    self._buffers[schema].append(record_dict)
                self._record_count += 1
        except Exception as e:
            self._error_count += 1
            # Only log once per class of error, or the bad record throughput
            # will drown the log. First few we log verbosely, then rate-limit.
            if self._error_count <= 5:
                logger.exception("live record handling failed", extra={"error": str(e)})

    def _on_error(self, exc: Exception) -> None:
        self._error_count += 1
        logger.error("live client exception", extra={"error": repr(exc)})

    # ------------------------------------------------------------------
    # Record -> dict conversion
    # ------------------------------------------------------------------

    def _to_dict(self, record) -> Optional[dict]:
        """Convert one live record into the DataFrame-ready shape.

        Prices come off ``ImbalanceMsg`` / ``MBP1Msg`` / ``TradeMsg`` as
        fixed-point int64 (scale 1e-9). We use the ``pretty_*`` accessors
        (they return Python floats already scaled) to avoid re-implementing
        the fixed-point conversion.

        Returns ``None`` when the record has no instrument_id mapping yet —
        the SymbolMappingMsg has not landed for this instrument, so we
        cannot attach a raw_symbol. Losing a handful of records before the
        first mapping is expected; the buffer grows quickly after.
        """
        from databento_dbn import ImbalanceMsg, MBP1Msg, TradeMsg
        instrument_id = getattr(record, "instrument_id", None)
        if instrument_id is None:
            return None
        with self._lock:
            symbol = self._id_to_symbol.get(instrument_id)
        if symbol is None:
            return None
        ts_event = self._to_datetime(record.ts_event)
        if ts_event is None:
            return None

        if isinstance(record, ImbalanceMsg):
            return {
                "_schema": "imbalance",
                "ts_event": ts_event,
                "symbol": symbol,
                "side": _decode_side(record.side),
                "total_imbalance_qty": int(record.total_imbalance_qty or 0),
                "paired_qty": int(record.paired_qty or 0),
                "ref_price": float(record.pretty_ref_price)
                if record.ref_price is not None else float("nan"),
            }
        if isinstance(record, MBP1Msg):
            # tbbo lands as MBP1Msg with a trade in it
            return {
                "_schema": "tbbo",
                "ts_event": ts_event,
                "symbol": symbol,
                "price": float(record.pretty_price),
                "size": int(record.size),
                "ask_px_00": float(record.pretty_ask_px_00),
                "bid_px_00": float(record.pretty_bid_px_00),
            }
        if isinstance(record, TradeMsg):
            return {
                "_schema": "trades",
                "ts_event": ts_event,
                "symbol": symbol,
                "price": float(record.pretty_price),
                "size": int(record.size),
            }
        return None

    @staticmethod
    def _to_datetime(ts_ns: Optional[int]) -> Optional[datetime]:
        if ts_ns is None or ts_ns == 0:
            return None
        # ts_event is int64 nanoseconds since epoch UTC.
        return datetime.fromtimestamp(ts_ns / 1_000_000_000, tz=timezone.utc)


def _decode_side(side) -> str:
    """Coerce imbalance side to the ``'B' / 'A' / 'N'`` scheme the historical
    feature pipeline expects.

    The imbalance ``side`` field is a single-char code that Databento delivers
    as a Python ``str`` in the pyo3 binding. We normalise defensively — the
    ``ImbalanceMsg.side`` doc string in this SDK version is empty, so accept
    both the literal chars and integer variants.
    """
    if side is None:
        return "N"
    if isinstance(side, str):
        s = side.upper().strip()
        if s in ("B", "A", "N"):
            return s
        return "N"
    if isinstance(side, int):
        # 66=B, 65=A, 78=N in ASCII
        try:
            return chr(int(side)).upper()
        except ValueError:
            return "N"
    return "N"
