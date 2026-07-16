"""Live daily trading + retrain pipeline.

Two entry points:

- ``live.daily_orchestrator`` runs every trading day a few minutes before the
  15:50 America/New_York rebalance clock. It ingests Databento live data
  (imbalance + tbbo) from market open, engineers the same 15:50 close-imbalance
  features the offline forecast was trained on, predicts alpha with the
  persisted sklearn pipeline, solves the daily MVO (same objective as
  ``optimizer/optimal_portfolio.ipynb``), then submits share targets as
  Market-on-Close orders through ``ibkr_broker``.
- ``live.retrain`` runs on a slower cadence (weekly / monthly). It refreshes
  the historical Databento cache on a rolling window and rebuilds the
  covariance ``.npy`` artifacts, the impact ``.npy`` vectors, and the forecast
  ``.joblib`` pipeline in place. The daily orchestrator picks up the fresh
  artifacts on its next run without any changes to its code.
"""

__all__ = ["config", "features", "live_feed", "optimizer"]
