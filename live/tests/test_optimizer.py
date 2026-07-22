"""Tests for the optimizer glue classes and end-to-end MVO solve.

Uses the on-disk covariance / impact artifacts as the risk model to keep
the test realistic, but with a synthetic alpha vector and a hand-built
day-frame — no network, no forecast pipeline load.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from live import config
from live.optimizer import (
    CostModel,
    ForecastModel,
    ImpactModel,
    Optimizer,
    dollar_positions_to_shares,
)


def _make_day_frame(universe: list[str]) -> pl.DataFrame:
    """One-day cross section with all optimizer input columns populated."""
    return pl.DataFrame({
        "date": [date(2026, 6, 15)] * len(universe),
        "symbol": universe,
        # Alpha feature columns (populated with plausible values)
        "norm_rank_net_imb": np.linspace(-1, 1, len(universe)),
        "recent_return_xm": np.linspace(-0.001, 0.001, len(universe)),
        "auction_dollar_volume_ratio_xm": np.zeros(len(universe)),
        "norm_rank_net_x_recent_xm": np.zeros(len(universe)),
        "norm_rank_net_x_auction": np.zeros(len(universe)),
        # Optimizer inputs
        "entry_price": np.linspace(20.0, 500.0, len(universe)),
        "spread_return": np.full(len(universe), 0.0001),
        "ADV": np.full(len(universe), 1e9),
        "vol": np.full(len(universe), 0.02),
    })


def test_forecast_model_returns_zero_for_out_of_universe_symbols():
    universe = ["AAA", "BBB", "CCC"]

    class DummyModel:
        def predict(self, X):
            return np.array([0.1, 0.2, 0.3][: X.shape[0]])

    features = ["norm_rank_net_imb"]
    fm = ForecastModel(universe, DummyModel(), features)
    day = pl.DataFrame({"symbol": ["BBB", "ZZZ"], "norm_rank_net_imb": [0.5, 0.7]})
    alpha = fm.get_alpha(day)
    assert alpha.shape == (3,)
    assert alpha[0] == 0.0  # AAA not in day
    assert alpha[1] == 0.1  # BBB predicted
    assert alpha[2] == 0.0  # CCC not in day
    # ZZZ silently dropped (out of universe)


def test_cost_model_ignores_missing_prices():
    universe = ["AAA", "BBB"]
    cm = CostModel(universe)
    day = pl.DataFrame({
        "symbol": ["AAA"],
        "spread_return": [0.001],
        "entry_price": [100.0],
    })
    cost = cm.get_cost(day, commission_per_share=0.0035)
    # AAA: 0.001 * 0.5 + 2 * 0.0035/100 = 5e-4 + 7e-5
    assert cost[0] == pytest.approx(0.001 * 0.5 + 2 * (0.0035 / 100.0))
    assert cost[1] == 0.0  # BBB missing -> zero cost (optimizer will still charge risk)


def test_impact_model_fills_missing_with_max():
    universe = ["AAA", "BBB", "CCC"]
    im = ImpactModel(universe)
    day = pl.DataFrame({
        "symbol": ["AAA", "BBB"],
        "vol": [0.02, 0.05],
        "ADV": [1e9, 1e6],
    })
    v = im.get_impact_coeff(day)
    # BBB should have the higher coefficient (0.05/sqrt(1e6) > 0.02/sqrt(1e9))
    assert v[0] < v[1]
    # CCC missing -> filled with max present (=BBB's)
    assert v[2] == v[1]


def test_dollar_to_shares_drops_bad_prices():
    universe = ["AAA", "BBB", "CCC"]
    dollars = np.array([1000.0, -500.0, 200.0])
    prices = np.array([100.0, 50.0, float("nan")])
    shares = dollar_positions_to_shares(dollars, prices, universe)
    assert shares == {"AAA": 10, "BBB": -10}


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_optimizer_end_to_end_with_real_artifacts():
    """Load real cov + impact artifacts, feed a synthetic alpha, solve.

    N is derived from the on-disk covariance loadings so the test tracks
    whatever universe the artifacts were built for (top-49 today; the
    plan is to scale toward 500 later). Hardcoding N was a regression
    trap — a mismatch means the covariance was built against a different
    universe file than the test expected, not that the optimizer is
    broken.

    Round-2 fix (correctness review): the earlier "if drift, pad" branch
    made a real invariant undetectable — a future retrain writing
    loadings at 500 names against a top_49 active universe would be
    silently padded with 451 fake tickers and pass. Replaced with a
    hard cross-check that the covariance artifact and the active
    universe agree; a mismatch fails the test with a message pointing
    at the fix (rerun ``live.retrain --refit covariance``).
    """
    L = np.load(config.COV_DIR / config.COV_LOADINGS_NAME)
    Sigma_f = np.load(config.COV_DIR / config.COV_FACTOR_COV_NAME)
    Sigma_r = np.load(config.COV_DIR / config.COV_MATRIX_NAME)
    eps_diag = np.load(config.COV_DIR / config.COV_EPS_DIAG_NAME)
    n_factors, N = L.shape
    universe = config.load_active_universe()
    assert len(universe) == N, (
        f"covariance artifact has N={N} names but active universe "
        f"({config.DEFAULT_ACTIVE_UNIVERSE_FILE}) has {len(universe)} — "
        f"rerun ``live.retrain --refit covariance`` to regenerate the "
        f"loadings against the current active universe."
    )

    day = _make_day_frame(universe)
    cost = CostModel(universe).get_cost(day, config.COMMISSION_PER_SHARE)
    impact_coeff = ImpactModel(universe).get_impact_coeff(day)

    # Synthetic alpha: rank-based signal in return space (~O(1e-3)).
    alpha = np.linspace(-2e-3, 2e-3, N)

    opt = Optimizer(universe)
    opt.set_problem(
        Sigma_r=Sigma_r, Sigma_f=Sigma_f, Loadings=L,
        eps_diag=eps_diag, lam=1e-4, gam=0.5, vol_cap=1e5, scale=1e5,
    )
    dollar_positions = opt.solve(alpha, cost, impact_coeff)
    assert dollar_positions.shape == (N,)
    # Should produce a book with GMV > 0 but bounded by the vol cap.
    assert float(np.abs(dollar_positions).sum()) > 0.0
    # Two-factor neutrality
    factor_exposures = L[:2] @ dollar_positions
    assert np.allclose(factor_exposures, 0.0, atol=1e-3)
    # Sanity: n_factors matches the covariance shape convention.
    assert n_factors == 10
