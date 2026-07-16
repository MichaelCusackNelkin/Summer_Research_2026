"""Portfolio construction models, lifted out of ``optimizer/optimal_portfolio.ipynb``.

Three plug-in models fed to the ``Optimizer``:

- ``ForecastModel`` — wraps the persisted sklearn Pipeline; returns a dense
  alpha vector indexed by the active universe.
- ``CostModel`` — half-spread + commission return-space cost per name.
- ``ImpactModel`` — per-day ``sigma_i / sqrt(ADV$_i)`` coefficient used by
  the square-root impact term inside the QP objective and by the ex-post PnL
  accounting.

``Optimizer`` compiles a parametrized cvxpy QP once and re-solves per day by
plugging alpha / cost / impact_coeff into its ``cp.Parameter``s. The daily
orchestrator only ever calls ``set_problem`` once and ``solve`` once per run,
but the same interface allows retrain-time backtests to warm-start through
many days on one compiled problem.

The only meaningful deviation from the notebook is that ``ImpactModel``
accepts an optional per-name ``gamma`` VECTOR (not the scalar the notebook
sweeps). If ``gamma`` is a vector, the objective multiplies it entrywise into
the 3/2-power impact sum — matching the follow-up outlined at the end of
``impact/impact_estimation.ipynb``. When ``gamma`` is a scalar the code path
is bit-exact with the notebook.
"""
from __future__ import annotations

from typing import Optional, Union

import cvxpy as cp
import numpy as np
import polars as pl
import scipy


class ForecastModel:
    """Alpha vector from the persisted sklearn Pipeline.

    The Pipeline was trained on features listed in ``features.ALPHA_FEATURES``
    — pass in the same list so the DataFrame -> numpy conversion picks the
    right columns in the right order.
    """

    def __init__(self, universe: list[str], model, features: list[str]) -> None:
        self.universe = universe
        self.sym_idx_map = {s: i for i, s in enumerate(universe)}
        self.model = model
        self.features = features

    def get_alpha(self, day_df: pl.DataFrame) -> np.ndarray:
        alpha = np.zeros(len(self.universe))
        if day_df.is_empty():
            return alpha
        preds = self.model.predict(day_df[self.features].to_numpy()).ravel()
        for symbol, p in zip(day_df["symbol"].to_list(), preds):
            idx = self.sym_idx_map.get(symbol)
            if idx is not None:
                alpha[idx] = p
        return alpha


class CostModel:
    """Half-spread + commission per name, in return space."""

    def __init__(self, universe: list[str]) -> None:
        self.universe = universe
        self.sym_idx_map = {s: i for i, s in enumerate(universe)}

    def get_cost(self, day_df: pl.DataFrame, commission_per_share: float) -> np.ndarray:
        cost = np.zeros(len(self.universe))
        if day_df.is_empty():
            return cost
        for symbol, spread, entry_price in zip(
            day_df["symbol"].to_list(),
            day_df["spread_return"].to_list(),
            day_df["entry_price"].to_list(),
        ):
            idx = self.sym_idx_map.get(symbol)
            if idx is None or spread is None or entry_price is None or entry_price <= 0:
                continue
            comm_rate = commission_per_share / entry_price
            cost[idx] = (spread * 0.5) + (2 * comm_rate)
        return cost


class ImpactModel:
    """Per-day, per-name impact coefficient ``sigma_i / sqrt(ADV$_i)``.

    Names missing vol or ADV that day get charged like the highest-coefficient
    present name — see the notebook rationale ("never free to trade").
    """

    def __init__(self, universe: list[str]) -> None:
        self.universe = universe
        self.sym_idx_map = {s: i for i, s in enumerate(universe)}

    def get_impact_coeff(self, day_df: pl.DataFrame) -> np.ndarray:
        v = np.zeros(len(self.universe))
        if day_df.is_empty():
            return v
        for symbol, vol, adv in zip(
            day_df["symbol"].to_list(),
            day_df["vol"].to_list(),
            day_df["ADV"].to_list(),
        ):
            idx = self.sym_idx_map.get(symbol)
            if idx is None:
                continue
            if vol is not None and adv is not None and vol > 0 and adv > 0:
                v[idx] = vol / np.sqrt(adv)
        if (v == 0).any() and (v > 0).any():
            v[v == 0] = v[v > 0].max()
        return v


class Optimizer:
    """Single-period daily MVO — matches ``optimal_portfolio.ipynb``.

    Objective (in per-scale units, ``Q$ = scale * x``):

        maximise  alpha @ x
                  - (lambda * scale / 2) * ( (Lx)ᵀ Σ_f (Lx) + xᵀ D x )
                  - (cost / 2) @ |x|
                  - gamma * sqrt(scale) * sum_i g_i * impact_coeff_i * |x_i|^{3/2}

    where ``g_i`` is the per-name impact coefficient (a vector if
    ``impact_gamma`` was supplied as a vector; ones otherwise), and the risk
    term is enforced ALSO as a quadratic constraint
    ``sqrt(xᵀ Σ x) <= vol_cap / scale`` for numerical stability.

    Two-factor neutrality (first two loadings). Solved with CLARABEL.
    """

    def __init__(self, universe: list[str]) -> None:
        self.universe = universe
        self.N = len(universe)

    def set_problem(
        self,
        Sigma_r: np.ndarray,
        Sigma_f: np.ndarray,
        Loadings: np.ndarray,
        eps_diag: np.ndarray,
        lam: float,
        gam: Union[float, np.ndarray],
        vol_cap: float,
        scale: Optional[float] = None,
    ) -> None:
        self.x = cp.Variable(self.N)
        self.alpha = cp.Parameter(self.N)
        self.cost = cp.Parameter(self.N, nonneg=True)
        self.impact_coeff = cp.Parameter(self.N, nonneg=True)

        if np.ndim(gam) == 0:
            self.gamma_scalar = float(gam)
            self.gamma_vec = None
            gam_expr = self.gamma_scalar * cp.multiply(
                self.impact_coeff, cp.power(cp.abs(self.x), 1.5)
            )
        else:
            gam_arr = np.asarray(gam, dtype=float).ravel()
            if gam_arr.shape != (self.N,):
                raise ValueError(
                    f"gamma vector shape {gam_arr.shape} != universe size {self.N}"
                )
            self.gamma_scalar = None
            self.gamma_vec = gam_arr
            gam_expr = cp.multiply(
                cp.Constant(gam_arr),
                cp.multiply(self.impact_coeff, cp.power(cp.abs(self.x), 1.5)),
            )

        self.Sigma_r = cp.Constant(Sigma_r)
        self.L = cp.Constant(Loadings)
        self.eps_diag = np.asarray(eps_diag, dtype=float).ravel()
        self.D = np.diag(self.eps_diag)
        self.scale = float(vol_cap if scale is None else scale)
        self.lam = cp.Constant(lam)

        self.objective = cp.Maximize(
            self.alpha @ self.x
            - (self.lam * self.scale / 2)
            * (cp.quad_form(self.L @ self.x, Sigma_f) + cp.quad_form(self.x, self.D))
            - (self.cost / 2) @ cp.abs(self.x)
            - (self.scale ** 0.5) * cp.sum(gam_expr)
        )

        Sigma_f_sqrt = scipy.linalg.sqrtm(Sigma_f).real
        D_sqrt = np.sqrt(self.eps_diag)
        quad_risk = cp.norm(
            cp.hstack([Sigma_f_sqrt @ self.L @ self.x, cp.multiply(D_sqrt, self.x)]), 2
        )
        self.constraints = [quad_risk <= vol_cap / self.scale]

        # First two factors of the covariance model. Prevents the optimizer
        # from loading up on the market and second-largest style factors.
        # ``Loadings`` shape is (n_factors, n_symbols).
        n_neutral = min(2, Loadings.shape[0])
        self.constraints += [
            self.x @ Loadings[i, :] == 0 for i in range(n_neutral)
        ]

        self.problem = cp.Problem(self.objective, self.constraints)

    def solve(
        self,
        alpha: np.ndarray,
        cost: np.ndarray,
        impact_coeff: np.ndarray,
        solver: str = cp.CLARABEL,
    ) -> np.ndarray:
        """Return a dollar-position vector (aligned with ``universe``)."""
        self.alpha.value = alpha
        self.cost.value = cost
        self.impact_coeff.value = impact_coeff
        self.problem.solve(solver=solver)
        if self.problem.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"solver status: {self.problem.status}")
        return self.x.value * self.scale


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def dollar_positions_to_shares(
    dollar_positions: np.ndarray,
    prices: np.ndarray,
    universe: list[str],
    min_price: float = 0.01,
) -> dict[str, int]:
    """Turn a dollar-position vector into signed integer share counts.

    Rounds half-to-even. Names with a non-positive or NaN price are dropped
    (impossible to size) and logged out by the caller — we return them
    silently absent from the mapping so the broker's ``rebalance()`` does not
    try to close a position it can't price.
    """
    if len(dollar_positions) != len(universe) or len(prices) != len(universe):
        raise ValueError("dollar_positions, prices, universe must be same length")
    shares: dict[str, int] = {}
    for sym, d, px in zip(universe, dollar_positions, prices):
        if not np.isfinite(d) or not np.isfinite(px) or px < min_price:
            continue
        s = int(np.rint(d / px))
        if s == 0:
            continue
        shares[sym] = s
    return shares
