"""
Order book simulator
"""

import numpy as np
import pandas as pd
import databento as db
import polars as pl

class ExecutionSimulator:
    """
    Replay-driven execution simulator that walks a time-sorted L2 orderbook
    DataFrame and simulates market-order fills against the book state.

    Parameters
    ----------
    df : pl.DataFrame
        Polars DataFrame with columns ``ts_nano``, ``instrument_id``,
        and 10 levels of ``bid_px_XX``, ``ask_px_XX``, ``bid_sz_XX``,
        ``ask_sz_XX`` (XX = 00..09).
    replenishment_rate : float
        Lambda parameter for the Poisson process that draws new liquidity
        sizes when replenishing swept price levels after an execution.
        A value of 0 disables replenishment (default).
    latency_ns : int
        Simulated latency in nanoseconds added to every order's timestamp
        before execution (default 0).
    """

    _N_LEVELS: int = 10

    def __init__(self, df: pl.DataFrame, replenishment_rate: float = 0.0, latency_ns: int = 0) -> None:
        df = df.sort("ts_nano")

        # Extract numpy arrays for fast row-level access (writable copies).
        self.ts_ns: np.ndarray = df["ts_nano"].to_numpy().copy()
        self.ts_event: np.ndarray = df["ts_event"].to_numpy().copy() if "ts_event" in df.columns else np.zeros(len(df), dtype=np.int64)
        self.instrument_id: np.ndarray = df["instrument_id"].to_numpy().copy()
        self.action: np.ndarray = df["action"].to_numpy().copy() if "action" in df.columns else np.full(len(df), None, dtype=object)
        self.size: np.ndarray = df["size"].to_numpy().copy().astype(np.float64) if "size" in df.columns else np.zeros(len(df), dtype=np.float64)

        lvls = [f"{i:02d}" for i in range(self._N_LEVELS)]
        self.bid_px: np.ndarray = df.select([f"bid_px_{l}" for l in lvls]).to_numpy().copy().astype(np.float64)
        self.ask_px: np.ndarray = df.select([f"ask_px_{l}" for l in lvls]).to_numpy().copy().astype(np.float64)
        self.bid_sz: np.ndarray = df.select([f"bid_sz_{l}" for l in lvls]).to_numpy().copy().astype(np.float64)
        self.ask_sz: np.ndarray = df.select([f"ask_sz_{l}" for l in lvls]).to_numpy().copy().astype(np.float64)

        self._n_rows: int = len(self.ts_ns)
        self._cur_idx: int = 0
        self.replenishment_rate: float = replenishment_rate
        self.latency_ns: int = latency_ns

        # Internal book state — mutable lists of (price, size) tuples.
        self.bids: list[tuple[float, int]]
        self.asks: list[tuple[float, int]]
        self.bids, self.asks = self._build_book_from_row(0)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _build_book_from_row(self, idx: int) -> tuple[list[tuple[float, int]], list[tuple[float, int]]]:
        """
        Read row *idx* from the numpy arrays and return ``(bids, asks)``.

        Parameters
        ----------
        idx : int
            Row index into the numpy arrays.

        Returns
        -------
        tuple[list[tuple[float, int]], list[tuple[float, int]]]
            Bid and ask sides, each a list of ``(price, size)`` tuples with
            zero-price or zero-size levels filtered out.
        """
        bids = [
            (float(self.bid_px[idx, l]), int(self.bid_sz[idx, l]))
            for l in range(self._N_LEVELS)
            if self.bid_px[idx, l] > 0 and self.bid_sz[idx, l] > 0
        ]
        asks = [
            (float(self.ask_px[idx, l]), int(self.ask_sz[idx, l]))
            for l in range(self._N_LEVELS)
            if self.ask_px[idx, l] > 0 and self.ask_sz[idx, l] > 0
        ]
        return bids, asks
    
    def _compute_replenishment(self, side: str) -> list[tuple[float, int]]:
        """
        Draw Poisson-distributed liquidity for swept gap levels after execution.

        This method only *computes* which levels to add and their sizes — it
        does **not** modify ``self.bids`` / ``self.asks``.  The replenished
        liquidity is written exclusively to future rows (starting at
        ``_cur_idx + 1``) by ``_apply_replenishment_to_future``, so the fill
        row itself reflects the depleted book state.

        For a BUY order (asks were consumed): replenish ask levels from the
        current best ask down to ``best_bid + 0.01``.

        For a SELL order (bids were consumed): replenish bid levels from the
        current best bid up to ``best_ask - 0.01``.

        Parameters
        ----------
        side : str
            ``"BUY"`` or ``"SELL"`` — the side of the *order* that executed.

        Returns
        -------
        list[tuple[float, int]]
            Levels to be inserted, as ``(price, size)`` tuples.
        """
        if self.replenishment_rate <= 0:
            return []

        added: list[tuple[float, int]] = []

        if side == "BUY":
            if not self.bids or not self.asks:
                return []
            best_bid = self.bids[0][0]
            best_ask = self.asks[0][0]
            existing_ask_prices = {round(px, 2) for px, _ in self.asks}
            level_px = round(best_ask - 0.01, 2)
            floor_px = round(best_bid + 0.01, 2)
            while level_px >= floor_px:
                if level_px not in existing_ask_prices:
                    sz = int(np.random.poisson(self.replenishment_rate))
                    if sz > 0:
                        added.append((level_px, sz))
                level_px = round(level_px - 0.01, 2)
        else:
            if not self.bids or not self.asks:
                return []
            best_ask = self.asks[0][0]
            best_bid = self.bids[0][0]
            existing_bid_prices = {round(px, 2) for px, _ in self.bids}
            level_px = round(best_bid + 0.01, 2)
            ceiling_px = round(best_ask - 0.01, 2)
            while level_px <= ceiling_px:
                if level_px not in existing_bid_prices:
                    sz = int(np.random.poisson(self.replenishment_rate))
                    if sz > 0:
                        added.append((level_px, sz))
                level_px = round(level_px + 0.01, 2)

        return added

    def _apply_replenishment_to_future(self, side: str, added: list[tuple[float, int]]) -> None:
        """
        Propagate replenished liquidity into future rows starting at
        ``_cur_idx + 1`` (the next timestamp after the fill).

        Parameters
        ----------
        side : str
            ``"BUY"`` or ``"SELL"`` — the side of the *order* that executed.
            A BUY replenishes asks; a SELL replenishes bids.
        added : list[tuple[float, int]]
            List of ``(price, size)`` tuples to add.
        """
        start = self._cur_idx + 1
        if start >= self._n_rows or not added:
            return

        if side == "BUY":
            px_arr = self.ask_px
            sz_arr = self.ask_sz
        else:
            px_arr = self.bid_px
            sz_arr = self.bid_sz

        future_px = px_arr[start:]
        future_sz = sz_arr[start:]

        for add_px, add_qty in added:
            mask = future_px == add_px
            future_sz[mask] += add_qty

    def _sync_book_to_arrays(self) -> None:
        """Write ``self.bids`` / ``self.asks`` back into the numpy arrays at
        ``self._cur_idx``, shifting remaining levels to the front and
        back-filling any trailing empty slots with synthetic liquidity
        (one tick worse per level, random size 100-150)."""
        idx = self._cur_idx

        # --- bids ---
        n_bids = len(self.bids)
        for l in range(self._N_LEVELS):
            if l < n_bids:
                self.bid_px[idx, l] = self.bids[l][0]
                self.bid_sz[idx, l] = self.bids[l][1]
            else:
                prev_px = self.bid_px[idx, l - 1]
                syn_px = round(prev_px - 0.01, 2)
                self.bid_px[idx, l] = syn_px
                self.bid_sz[idx, l] = 100
                self.bids.append((syn_px, 100))

        # --- asks ---
        n_asks = len(self.asks)
        for l in range(self._N_LEVELS):
            if l < n_asks:
                self.ask_px[idx, l] = self.asks[l][0]
                self.ask_sz[idx, l] = self.asks[l][1]
            else:
                prev_px = self.ask_px[idx, l - 1]
                syn_px = round(prev_px + 0.01, 2)
                self.ask_px[idx, l] = syn_px
                self.ask_sz[idx, l] = 100
                self.asks.append((syn_px, 100))

    def _apply_impact_to_future(self, side: str, fills: list[tuple[float, int]]) -> None:
        """
        Propagate liquidity removal from *fills* into all future rows.

        For each filled price level, every future row whose corresponding
        price array matches that price has its size reduced by the filled
        quantity (clamped to 0).

        Parameters
        ----------
        side : str
            ``"BUY"`` or ``"SELL"`` — determines which arrays to modify.
        fills : list[tuple[float, int]]
            List of ``(fill_price, fill_qty)`` produced by the execution.
        """
        start = self._cur_idx + 1
        if start >= self._n_rows or not fills:
            return

        if side == "BUY":
            px_arr = self.ask_px
            sz_arr = self.ask_sz
        else:
            px_arr = self.bid_px
            sz_arr = self.bid_sz

        future_px = px_arr[start:]
        future_sz = sz_arr[start:]

        for fill_px, fill_qty in fills:
            mask = future_px == fill_px
            future_sz[mask] -= fill_qty
        np.maximum(future_sz, 0, out=future_sz)

        # Compact: remove zero-size levels (where price is still set) and
        # shift deeper levels forward so there are no gaps.
        # A level needs removal when size==0 but price is not NaN (genuinely emptied).
        has_gap = np.any((future_sz == 0) & ~np.isnan(future_px), axis=1)
        for row in np.where(has_gap)[0]:
            keep = (future_sz[row] > 0) | np.isnan(future_px[row])
            n_keep = int(np.sum(keep))
            future_px[row, :n_keep] = future_px[row][keep]
            future_sz[row, :n_keep] = future_sz[row][keep]
            for l in range(n_keep, self._N_LEVELS):
                prev = future_px[row, l - 1]
                if side == "BUY":
                    future_px[row, l] = round(prev + 0.01, 2)
                else:
                    future_px[row, l] = round(prev - 0.01, 2)
                future_sz[row, l] = 100

    # ------------------------------------------------------------------
    # time control
    # ------------------------------------------------------------------

    def advance_to(self, ts_nano: int) -> None:
        """
        Advance the simulator to the last snapshot with ``ts_nano <= ts_nano``.

        The internal book state is replaced by the market snapshot at that
        point in time, discarding any prior simulated modifications.

        Parameters
        ----------
        ts_nano : int
            Target nanosecond timestamp.
        """
        # searchsorted(..., "right") gives the first index *after* ts_nano;
        # subtract 1 to land on the last row <= ts_nano.
        target = int(np.searchsorted(self.ts_ns, ts_nano, side="right")) - 1
        if target < 0:
            target = 0
        if target >= self._n_rows:
            target = self._n_rows - 1

        self._cur_idx = target
        self.bids, self.asks = self._build_book_from_row(target)

    # ------------------------------------------------------------------
    # order execution
    # ------------------------------------------------------------------

    def accept_mkt_order(self, side: str, order_size: int, ts_nano: int) -> dict:
        """
        Simulate a market order at a given timestamp.

        Parameters
        ----------
        side : str
            ``"BUY"`` (walks asks) or ``"SELL"`` (walks bids).
        order_size : int
            Number of shares to fill.
        ts_nano : int
            Nanosecond timestamp at which the order is sent.

        Returns
        -------
        dict
            ``vwap`` — volume-weighted average fill price.
            ``fills`` — list of ``(price, qty_filled)`` tuples.
            ``total_filled`` — total shares filled (may be < *order_size*).
            ``levels_swept`` — number of levels fully consumed.
            ``executed_at_ns`` — ``ts_nano`` of the book row the order
            executed against.
            ``executed_at_idx`` — integer index of that row in the internal
            arrays (useful for slicing).
            ``arrival_cost`` — difference between VWAP and midprice at
            execution time (positive means paid more than mid for BUY,
            received less than mid for SELL).
        """
        self.advance_to(ts_nano + self.latency_ns)
        mid = self.get_midprice()
        book = self.asks if side == "BUY" else self.bids

        fills: list[tuple[float, int]] = []
        remaining = order_size
        levels_swept = 0

        for i, (px, sz) in enumerate(book):
            if remaining <= 0:
                break
            filled = min(remaining, sz)
            fills.append((px, filled))
            remaining -= filled
            if filled == sz:
                levels_swept += 1
                book[i] = (px, 0)
            else:
                book[i] = (px, sz - filled)

        # Remove fully swept levels from the front.
        del book[:levels_swept]

        total_filled = order_size - remaining
        vwap = (
            sum(px * qty for px, qty in fills) / total_filled
            if total_filled > 0
            else 0.0
        )

        if mid is not None and total_filled > 0:
            arrival_cost = (vwap - mid) / mid if side == "BUY" else (mid - vwap) / mid
        else:
            arrival_cost = None

        replenished = self._compute_replenishment(side)
        self._sync_book_to_arrays()
        self._apply_impact_to_future(side, fills)
        self._apply_replenishment_to_future(side, replenished)

        return {
            "vwap": vwap,
            "fills": fills,
            "total_filled": total_filled,
            "levels_swept": levels_swept,
            "executed_at_ns": int(self.ts_ns[self._cur_idx]),
            "executed_at_idx": self._cur_idx,
            "arrival_cost": arrival_cost,
        }

    def accept_ioc_lmt_order(self, side: str, limit_price: float, order_size: int, ts_nano: int) -> dict:
        """
        Simulate an IOC Limit order execution at a given timestamp.

        Parameters
        ----------
        side : str
            ``"BUY"`` (walks asks) or ``"SELL"`` (walks bids).
        limit_price: float
            Limit price.
        order_size : int
            Number of shares to fill.
        ts_nano : int
            Nanosecond timestamp at which the order is sent.

        Returns
        -------
        dict
            ``vwap`` — volume-weighted average fill price.
            ``fills`` — list of ``(price, qty_filled)`` tuples.
            ``total_filled`` — total shares filled (may be < *order_size*).
            ``levels_swept`` — number of levels fully consumed.
            ``executed_at_ns`` — ``ts_nano`` of the book row the order
            executed against.
            ``executed_at_idx`` — integer index of that row in the internal
            arrays (useful for slicing).
            ``arrival_cost`` — difference between VWAP and midprice at
            execution time (positive means paid more than mid for BUY,
            received less than mid for SELL).
        """
        self.advance_to(ts_nano + self.latency_ns)
        mid = self.get_midprice()
        book = self.asks if side == "BUY" else self.bids

        fills: list[tuple[float, int]] = []
        remaining = order_size
        levels_swept = 0

        for i, (px, sz) in enumerate(book):
            if remaining <= 0:
                break
            if px > limit_price and side == "BUY":
                break
            if px < limit_price and side == "SELL":
                break
            filled = min(remaining, sz)
            fills.append((px, filled))
            remaining -= filled
            if filled == sz:
                levels_swept += 1
                book[i] = (px, 0)
            else:
                book[i] = (px, sz - filled)

        # Remove fully swept levels from the front.
        del book[:levels_swept]

        total_filled = order_size - remaining
        vwap = (
            sum(px * qty for px, qty in fills) / total_filled
            if total_filled > 0
            else 0.0
        )

        if mid is not None and total_filled > 0:
            arrival_cost = (vwap - mid) / mid if side == "BUY" else (mid - vwap) / mid
        else:
            arrival_cost = None

        replenished = self._compute_replenishment(side)
        self._sync_book_to_arrays()
        self._apply_impact_to_future(side, fills)
        self._apply_replenishment_to_future(side, replenished)

        return {
            "vwap": vwap,
            "fills": fills,
            "total_filled": total_filled,
            "levels_swept": levels_swept,
            "executed_at_ns": int(self.ts_ns[self._cur_idx]),
            "executed_at_idx": self._cur_idx,
            "arrival_cost": arrival_cost,
        }

    def accept_resting_lmt_order(
        self,
        side: str,
        limit_price: float,
        order_size: int,
        ts_nano: int,
        deadline_ns: int | None = None,
    ) -> dict:
        """
        Simulate a resting (passive) limit order at a given timestamp.

        The order is posted at ``limit_price`` and waits for the market to
        cross it.  A BUY resting on the bid fills when the best ask drops
        to or below the limit; a SELL resting on the ask fills when the
        best bid rises to or above the limit.  Each eligible row contributes
        up to the top-of-book opposite-side size as a proxy for available
        fill volume.

        Parameters
        ----------
        side : str
            ``"BUY"`` (resting on the bid) or ``"SELL"`` (resting on the ask).
        limit_price : float
            Resting limit price.
        order_size : int
            Number of shares to fill.
        ts_nano : int
            Nanosecond timestamp at which the order is sent.
        deadline_ns : int or None, optional
            Nanosecond timestamp after which the order is cancelled.  When
            ``None`` (default) the scan runs to the end of the data.

        Returns
        -------
        dict
            ``vwap`` — volume-weighted average fill price.
            ``fills`` — list of ``(price, qty_filled)`` tuples.
            ``total_filled`` — total shares filled (may be < *order_size*).
            ``levels_swept`` — always ``0`` for resting orders.
            ``executed_at_ns`` — ``ts_nano`` of the last row that produced
            a fill (or the placement row if no fill occurred).
            ``executed_at_idx`` — integer index of that row in the internal
            arrays (useful for slicing).
            ``arrival_cost`` — difference between VWAP and midprice at
            placement time (positive means paid more than mid for BUY,
            received less than mid for SELL).
        """
        self.advance_to(ts_nano + self.latency_ns)
        mid = self.get_midprice()

        fills: list[tuple[float, int]] = []
        remaining = order_size
        last_fill_idx = self._cur_idx

        for idx in range(self._cur_idx, self._n_rows):
            if remaining <= 0:
                break
            if deadline_ns is not None and int(self.ts_ns[idx]) > deadline_ns:
                break

            if side == "BUY":
                top_px = self.ask_px[idx, 0]
                top_sz = self.ask_sz[idx, 0]
                crossed = top_px > 0 and top_sz > 0 and top_px <= limit_price
            else:
                top_px = self.bid_px[idx, 0]
                top_sz = self.bid_sz[idx, 0]
                crossed = top_px > 0 and top_sz > 0 and top_px >= limit_price

            if not crossed:
                continue

            filled = min(remaining, int(top_sz))
            if filled <= 0:
                continue
            fills.append((limit_price, filled))
            remaining -= filled
            last_fill_idx = idx

        self._cur_idx = last_fill_idx
        self.bids, self.asks = self._build_book_from_row(last_fill_idx)

        total_filled = order_size - remaining
        vwap = (
            sum(px * qty for px, qty in fills) / total_filled
            if total_filled > 0
            else 0.0
        )

        if mid is not None and total_filled > 0:
            arrival_cost = (vwap - mid) / mid if side == "BUY" else (mid - vwap) / mid
        else:
            arrival_cost = None

        replenished = self._compute_replenishment(side)
        self._sync_book_to_arrays()
        self._apply_impact_to_future(side, fills)
        self._apply_replenishment_to_future(side, replenished)

        return {
            "vwap": vwap,
            "fills": fills,
            "total_filled": total_filled,
            "levels_swept": 0,
            "executed_at_ns": int(self.ts_ns[self._cur_idx]),
            "executed_at_idx": self._cur_idx,
            "arrival_cost": arrival_cost,
        }

    # ------------------------------------------------------------------
    # filling strategies
    # ------------------------------------------------------------------

    def naive_fill(
        self, side: str, parent_size: int, ts_nano: int
    ) -> tuple[dict, pl.DataFrame]:
        """
        Execute a single market order for the full parent size.

        Parameters
        ----------
        side : str
            ``"BUY"`` or ``"SELL"``.
        parent_size : int
            Total number of shares to fill.
        ts_nano : int
            Nanosecond timestamp at which the order is sent.

        Returns
        -------
        tuple[dict, pl.DataFrame]
            The execution result dict and the post-execution book DataFrame.
        """
        result = self.accept_mkt_order(side, parent_size, ts_nano)
        return result, self.get_book_df()

    def constant_fill(
        self,
        side: str,
        parent_size: int,
        start_ts_ns: int,
        wait_ns: int,
        n_chunks: int,
    ) -> tuple[list[dict], pl.DataFrame]:
        """
        Split a parent order into equal-sized market-order chunks.

        Parameters
        ----------
        side : str
            ``"BUY"`` or ``"SELL"``.
        parent_size : int
            Total number of shares to fill.
        start_ts_ns : int
            Nanosecond timestamp of the first child order.
        wait_ns : int
            Nanosecond interval between child orders.
        n_chunks : int
            Number of child orders.

        Returns
        -------
        tuple[list[dict], pl.DataFrame]
            List of execution result dicts and the post-execution book
            DataFrame.
        """
        chunk_size = int(parent_size / n_chunks)
        timestamps = np.arange(start_ts_ns, start_ts_ns + n_chunks * int(wait_ns), int(wait_ns))
        results: list[dict] = []
        for t in timestamps:
            r = self.accept_mkt_order(side, chunk_size, int(t))
            results.append(r)
        return results, self.get_book_df()

    def constant_ioc_fill(
        self,
        side: str,
        limit_px: float | None,
        parent_size: int,
        start_ts_ns: int,
        wait_ns: int,
        n_chunks: int,
    ) -> tuple[list[dict], pl.DataFrame]:
        """
        Split a parent order into equal-sized IOC limit-order chunks.

        Parameters
        ----------
        side : str
            ``"BUY"`` or ``"SELL"``.
        limit_px : float or None
            Limit price for every child IOC order.  When ``None``, the limit
            is resolved dynamically from each book snapshot using
            ``ask_px_02`` (BUY) or ``bid_px_02`` (SELL).
        parent_size : int
            Total number of shares to fill.
        start_ts_ns : int
            Nanosecond timestamp of the first child order.
        wait_ns : int
            Nanosecond interval between child orders.
        n_chunks : int
            Number of child orders.

        Returns
        -------
        tuple[list[dict], pl.DataFrame]
            List of execution result dicts and the post-execution book
            DataFrame.
        """
        chunk_size = parent_size // n_chunks
        timestamps = np.arange(start_ts_ns, start_ts_ns + n_chunks * int(wait_ns), int(wait_ns))
        results: list[dict] = []

        for i, t in enumerate(timestamps):
            size = parent_size - (n_chunks - 1) * chunk_size if i == n_chunks - 1 else chunk_size
            if limit_px is None:
                r = self.accept_mkt_order(side, size, int(t))
            else:
                r = self.accept_ioc_lmt_order(side, limit_px, size, int(t))
            results.append(r)
        return results, self.get_book_df()

    def constant_resting_fill(
        self,
        side: str,
        limit_price: float,
        parent_size: int,
        start_ts_ns: int,
        wait_ns: int,
        n_chunks: int,
        deadline_ns: int | None = None,
    ) -> tuple[list[dict], pl.DataFrame]:
        """
        Split a parent order into equal-sized resting limit-order chunks.

        Parameters
        ----------
        side : str
            ``"BUY"`` or ``"SELL"``.
        limit_price : float
            Resting limit price for every child order.
        parent_size : int
            Total number of shares to fill.
        start_ts_ns : int
            Nanosecond timestamp of the first child order.
        wait_ns : int
            Nanosecond interval between child orders.
        n_chunks : int
            Number of child orders.
        deadline_ns : int or None, optional
            Nanosecond cancellation deadline passed to each child order.

        Returns
        -------
        tuple[list[dict], pl.DataFrame]
            List of execution result dicts and the post-execution book
            DataFrame.
        """
        chunk_size = parent_size // n_chunks
        timestamps = np.arange(start_ts_ns, start_ts_ns + n_chunks * int(wait_ns), int(wait_ns))
        results: list[dict] = []

        for i, t in enumerate(timestamps):
            size = parent_size - (n_chunks - 1) * chunk_size if i == n_chunks - 1 else chunk_size
            r = self.accept_resting_lmt_order(side, limit_price, size, int(t), deadline_ns=deadline_ns)
            results.append(r)
        return results, self.get_book_df()

    def hybrid_fill(
        self,
        side: str,
        parent_size: int,
        limit: float,
        start_ts_ns: int,
        obi_threshold: float,
        n_slices: int,
        wait_ns: int,
    ) -> tuple[list[dict], pl.DataFrame]:
        """
        Fill a parent order by switching tactics based on top-of-book OBI.

        The horizon is carved into ``n_slices`` equal child orders spaced by
        ``wait_ns``.  At each decision point the top-of-book OBI is compared
        to ``obi_threshold`` and the tactic is chosen accordingly: an
        aggressive market order when the imbalance points against the trade,
        a fan of resting limit orders across the top three passive levels
        when it points with the trade, and a plain market order otherwise.
        Any residual quantity after the horizon is swept with a final
        mop-up market order.

        Parameters
        ----------
        side : str
            ``"BUY"`` or ``"SELL"``.
        parent_size : int
            Total number of shares to fill.
        start_ts_ns : int
            Nanosecond timestamp of the first child order.
        obi_threshold : float
            Absolute OBI threshold separating the adverse, neutral, and
            favorable regimes.
        n_slices : int
            Number of scheduled child orders within the horizon.
        wait_ns : int
            Nanosecond interval between decision points.

        Returns
        -------
        tuple[list[dict], pl.DataFrame]
            List of execution result dicts and the post-execution book
            DataFrame.
        """
        qty_per_slice = parent_size // n_slices
        remaining = parent_size
        current_ts = start_ts_ns
        results: list[dict] = []

        while remaining > 0:
            if current_ts > int(self.ts_ns[-1]):
                break
            idx = max(0, int(np.searchsorted(self.ts_ns, current_ts, side="right")) - 1)
            bid_sz0 = np.sum(self.bid_sz[idx, 0:6])
            ask_sz0 = np.sum(self.ask_sz[idx, 0:6])
            denom = bid_sz0 + ask_sz0
            obi = (bid_sz0 - ask_sz0) / denom if denom > 0 else 0.0 # OBI using the top 5 levels
            size = remaining if remaining <= qty_per_slice else qty_per_slice

            adverse = (side == "BUY" and obi > obi_threshold) or (
                side == "SELL" and obi < -obi_threshold
            )
            favorable = (side == "BUY" and obi < -obi_threshold) or (
                side == "SELL" and obi > obi_threshold
            )

            if adverse:
                r = self.accept_ioc_lmt_order(side, limit, size, current_ts)
                results.append(r)
                remaining -= r["total_filled"]
            elif favorable:
                # expiry = current_ts + wait_ns
                expiry = current_ts + (300 * 1e9)
                for level in range(6):
                    if remaining <= 0:
                        break
                    px = (
                        float(self.bid_px[idx, level])
                        if side == "BUY"
                        else float(self.ask_px[idx, level])
                    )
                    if px <= 0:
                        continue
                    child_size = remaining if remaining <= qty_per_slice else qty_per_slice
                    r = self.accept_resting_lmt_order(
                        side,
                        px,
                        child_size,
                        current_ts + level,
                        deadline_ns=expiry,
                    )
                    results.append(r)
                    remaining -= r["total_filled"]
            else:
                r = self.accept_ioc_lmt_order(side, limit, size, current_ts)
                results.append(r)
                remaining -= r["total_filled"]

            current_ts += wait_ns

        while remaining > 0:
            mop_ts = min(current_ts, int(self.ts_ns[-1]))
            r = self.accept_mkt_order(side, remaining, mop_ts)
            remaining -= r["total_filled"]
            results.append(r)
            if r["total_filled"] == 0 or mop_ts >= int(self.ts_ns[-1]):
                break
            current_ts += int(wait_ns)

        return results, self.get_book_df()

    def obi_adaptive_fill(
        self,
        side: str,
        parent_size: int,
        start_ts_ns: int,
        max_size: int,
        threshold: float,
        horizon: int,
        limit_price: float | None = None,
        min_interval_ns: int = 0,
        min_shares_between: int = 0,
    ) -> tuple[list[dict], pl.DataFrame, pl.DataFrame]:
        """
        Adaptively fill a parent order using order-book imbalance (OBI).

        At each step the method scans the book for timestamps within the
        rolling *horizon* window, computes OBI, derives order sizes
        proportional to ``|OBI| * max_size``, and sends IOC limit orders when
        the imbalance exceeds *threshold* in the favourable direction.

        Parameters
        ----------
        side : str
            ``"BUY"`` or ``"SELL"``.
        parent_size : int
            Total number of shares to fill.
        max_size : int
            Maximum child-order size (scaled by ``|OBI|``).
        threshold : float
            OBI threshold that must be exceeded before a child order fires.
        start_ts_ns : int
            Nanosecond timestamp at which to begin scanning.
        horizon : int
            Nanosecond look-ahead window for candidate selection.
        limit_price : float or None, optional
            Fixed limit price for all child orders.  When ``None`` (default),
            the limit is computed dynamically from each candidate row using
            ``ask_px_02`` (BUY) or ``bid_px_02`` (SELL).

        Returns
        -------
        tuple[list[dict], pl.DataFrame, pl.DataFrame]
            ``(results, final_book_df, last_candidates)`` — the list of
            execution result dicts, the post-execution book DataFrame, and the
            candidates DataFrame from the last iteration.
        """
        init_ts_ns = start_ts_ns
        deadline = init_ts_ns + horizon
        results: list[dict] = []
        remaining = parent_size
        candidates = pl.DataFrame()

        while remaining > 0 and start_ts_ns < deadline:
            candidates = (
                self._get_window_df(start_ts_ns, deadline, side)
                .with_columns(
                    pl.col("bid_sz_00").fill_null(0).fill_nan(0).alias("bid_sz_00"),
                    pl.col("ask_sz_00").fill_null(0).fill_nan(0).alias("ask_sz_00"),
                )
                .with_columns(
                    (
                        (pl.col("bid_sz_00").cast(pl.Int32) - pl.col("ask_sz_00").cast(pl.Int32))
                        / (pl.col("bid_sz_00").cast(pl.Int32) + pl.col("ask_sz_00").cast(pl.Int32))
                    ).fill_nan(0).clip(-1, 1).alias("obi")
                )
                .with_columns(
                    #(pl.col("obi").abs() * max_size)
                    pl.lit(max_size)
                    #.clip(1, None)
                    #.cast(pl.Int64)
                    .alias("order_size")
                )
            )

            if side == "BUY":
                candidates = candidates.filter(pl.col("obi") > threshold)
            else:
                candidates = candidates.filter(pl.col("obi") < threshold)

            if candidates.is_empty():
                break

            row = candidates.row(0, named=True)
            size = min(row["order_size"], remaining)

            if size <= 0:
                break

            if limit_price is None:
                r = self.accept_mkt_order(side, size, row["ts_nano"])
            else:
                r = self.accept_ioc_lmt_order(side, limit_price, size, row["ts_nano"])
            remaining -= r["total_filled"]
            results.append(r)

            last_order_ts = row["ts_nano"]
            time_gate_ts = last_order_ts + max(1, min_interval_ns)

            if min_shares_between > 0:
                lo = int(np.searchsorted(self.ts_ns, last_order_ts, side="right"))
                future = pl.DataFrame({
                    "ts_nano": self.ts_ns[lo:],
                    "action": self.action[lo:],
                    "size": self.size[lo:],
                })
                cum_trade_vol = (
                    future
                    .select(
                        pl.when(pl.col("action") == "T")
                        .then(pl.col("size").fill_null(0))
                        .otherwise(0)
                        .cum_sum()
                        .alias("cum_vol")
                    )
                    .to_series()
                )
                gate_rows = (cum_trade_vol >= min_shares_between).arg_true()
                shares_gate_ts = (
                    int(future["ts_nano"][gate_rows[0]]) if len(gate_rows) > 0
                    else time_gate_ts
                )
                start_ts_ns = min(time_gate_ts, shares_gate_ts)
            else:
                start_ts_ns = time_gate_ts

        return results, self.get_book_df(), candidates

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------

    def get_book_state(self) -> dict:
        """
        Return the current internal book state.

        Returns
        -------
        dict
            ``bids`` — list of ``(price, size)`` tuples.
            ``asks`` — list of ``(price, size)`` tuples.
            ``ts_nano`` — nanosecond timestamp of the current snapshot.
        """
        return {
            "bids": list(self.bids),
            "asks": list(self.asks),
            "ts_nano": int(self.ts_ns[self._cur_idx]),
        }

    def get_midprice(self) -> float | None:
        """
        Return the midprice (average of best bid and best ask).

        Returns
        -------
        float or None
            ``None`` if either side of the book is empty.
        """
        if not self.bids or not self.asks:
            return None
        return (self.bids[0][0] + self.asks[0][0]) / 2.0

    def _get_window_df(self, start_ts_ns: int, end_ts_ns: int, side: str) -> pl.DataFrame:
        lo = int(np.searchsorted(self.ts_ns, start_ts_ns, side="left"))
        hi = int(np.searchsorted(self.ts_ns, end_ts_ns, side="right"))
        px_col = "ask_px_02" if side == "BUY" else "bid_px_02"
        px_level = 2
        if lo >= hi:
            return pl.DataFrame({
                "ts_nano": np.empty(0, dtype=self.ts_ns.dtype),
                "bid_sz_00": np.empty(0, dtype=self.bid_sz.dtype),
                "ask_sz_00": np.empty(0, dtype=self.ask_sz.dtype),
                px_col: np.empty(0, dtype=self.ask_px.dtype),
            })
        sl = slice(lo, hi)
        return pl.DataFrame({
            "ts_nano": self.ts_ns[sl],
            "bid_sz_00": self.bid_sz[sl, 0].copy(),
            "ask_sz_00": self.ask_sz[sl, 0].copy(),
            px_col: (self.ask_px[sl, px_level].copy() if side == "BUY" else self.bid_px[sl, px_level].copy()),
        })

    def get_book_df(self) -> pl.DataFrame:
        """
        Return the full orderbook across all timestamps as a Polars DataFrame.

        Values are read from the internal numpy arrays, so they reflect any
        post-execution modifications propagated to future rows.

        Returns
        -------
        pl.DataFrame
            DataFrame with ``ts_nano``, ``instrument_id``, and 10 levels
            of ``bid_px_XX``, ``ask_px_XX``, ``bid_sz_XX``, ``ask_sz_XX``.
        """
        data: dict[str, np.ndarray] = {
            "ts_nano": self.ts_ns,
            "ts_event": self.ts_event,
            "instrument_id": self.instrument_id,
            "action": self.action,
            "size": self.size,
        }
        for l in range(self._N_LEVELS):
            tag = f"{l:02d}"
            data[f"bid_px_{tag}"] = self.bid_px[:, l]
            data[f"ask_px_{tag}"] = self.ask_px[:, l]
            data[f"bid_sz_{tag}"] = self.bid_sz[:, l]
            data[f"ask_sz_{tag}"] = self.ask_sz[:, l]
        return pl.DataFrame(data)



import random

_worker_parquet_path: str | None = None


def init_worker(worker_args: tuple[str, int]) -> None:
    """Store the ticker-filtered parquet path in a module global for workers.

    Parameters
    ----------
    worker_args : tuple[str, int]
        Two-tuple ``(parquet_path, instrument_id)`` broadcast to each worker
        by the ``ProcessPoolExecutor`` initializer. Only the path is used;
        the parquet is already pre-filtered to a single instrument. No data
        is loaded here; workers lazy-scan the parquet file per-trial.
    """
    global _worker_parquet_path
    _worker_parquet_path, _ = worker_args


def _scan_window(ts_start: int, ts_end: int) -> pl.DataFrame:
    """Lazy-scan the worker parquet for a single [ts_start, ts_end] window.

    The worker parquet is pre-filtered to a single instrument and
    pre-sorted by ``ts_nano``, with ``ts_nano`` and ``mid`` already
    materialised, so this scan is a pure row-group-pruned range read.

    Parameters
    ----------
    ts_start : int
        Inclusive lower bound on ``ts_nano``.
    ts_end : int
        Inclusive upper bound on ``ts_nano``.

    Returns
    -------
    pl.DataFrame
        Collected window.
    """
    return (
        pl.scan_parquet(_worker_parquet_path)
        .filter((pl.col("ts_nano") >= ts_start) & (pl.col("ts_nano") <= ts_end))
        .collect()
    )


def run_trial(args):
    latency, replenishment, parent_qty, depth_lmt, threshold, horizon, min_ns, min_shares, max_size, wait_ns, n_chunks, method, ts = args
    side = random.choice(["BUY", "SELL"])
    df_window = _scan_window(ts, ts + horizon + latency)
    if df_window.is_empty():
        return None
    sim = ExecutionSimulator(df_window, latency_ns=latency, replenishment_rate=replenishment)
    if depth_lmt != None:
        df_at_ts = df_window.filter(pl.col("ts_nano") == ts)
        if df_at_ts.is_empty():
            return None
        lmt = df_at_ts[f"ask_px_{depth_lmt}" if side == "BUY" else f"bid_px_{depth_lmt}"][0]
    else:
        lmt = depth_lmt
    if method=="obi":
        results, _, _ = sim.obi_adaptive_fill(
            side, parent_qty, ts, limit_price=lmt,
            max_size=max_size, threshold=threshold, horizon=horizon,
            min_interval_ns=min_ns, min_shares_between=min_shares,
        )
    elif method=="constant":
        results, _ = sim.constant_ioc_fill(side, lmt, parent_qty, ts, wait_ns=wait_ns, n_chunks=n_chunks)
    elif method=="hybrid":
        results, _ = sim.hybrid_fill(
            side, parent_qty, lmt, ts,
            obi_threshold=threshold,
            n_slices=n_chunks,
            wait_ns=int(wait_ns),
        )
    filled_results = [r for r in results if r["vwap"] != 0.0]
    if not filled_results:
        return None

    # -- Return dict --
    df_at_ts = df_window.filter(pl.col("ts_nano") == ts)
    if df_at_ts.is_empty():
        return None
    
    p0 = df_at_ts["ask_px_00"][0] if side == "BUY" else df_at_ts["bid_px_00"][0]

    mean_child_arrival_cost = np.mean([r["arrival_cost"] for r in filled_results]) * 10_000
    median_child_arrival_cost = np.median([r["arrival_cost"] for r in filled_results]) * 10_000
    child_fill_times = np.diff([r["executed_at_ns"] for r in filled_results])

    parent_filled = np.sum([r["total_filled"] for r in filled_results])
    parent_vwap = np.sum([r["vwap"] * r["total_filled"] for r in filled_results]) / parent_filled
    parent_arrival_cost = (parent_vwap - p0) / p0 * 10_000 if side == "BUY" else (p0 - parent_vwap) / p0 * 10_000

    ts_event = [r["executed_at_ns"] for r in filled_results][0]

    return {
        "ts_event": ts_event,
        "side": side,
        "parent_filled": parent_filled,
        "child_fill_times": child_fill_times,
        "parent_arrival_cost": parent_arrival_cost,
        "mean_child_arrival_cost": mean_child_arrival_cost,
        "median_child_arrival_cost": median_child_arrival_cost,
    }

