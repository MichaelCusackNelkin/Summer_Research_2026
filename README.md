# Summer_Research_2026
Repo containing work on a summer research project for Statistical Aspects of Finance Spring 2026 at the Columbia MAFN: combining a forecast, covariance, adn spread costs into an optimal portoflio problem, and backtesting it on a point in time universe, covariance estimate, and forecasting model.

### Notes / Research Direction
1. Universe Selection and Point-in-Time:
    - We need a redesign that uses a point in time universe, trailing 12-24months rebalanced every quarter/6months.
    - We will need multiple databento calls. One full time-range ohlcv-1d to select point-in-time universes, then each trailing window for the tickers that aren't contained in past windows to avoid redundant spending.
    - Covariance and forecast need to be refit on point in time data at each rebalance.
    - Need lots of new plumbing to make this happen.
2. Covariance matrix is crap at the moment.
    - We need way more daily data
    - We need to use a factor-structure (and add a market-neutrality constraint to the optimizer)
2. The forecast is ok but also probably crap.
    - We need to think creatively about using more features. They may be weaker individually, but powerful in interaction with the stronger imbalance signal.
    - This may allow us to use lasso, or maybe a random forest.
3. Need to look over what Claude wrote (parallelization, accountant, plotting) to understand adn make changes.
4. Once these things are worked out:
    - We need to plot GMV (gross exposure above) against different values of the risk-aversion parameter lambda.
    - Try continuous trading version from 15:50-15:55
    - Try shorter horizon from 15:55(ish)-16:00
    - Try multi-period from 15:50-15:55(ish), 15:55(ish)-16:00, or minutely from 15:00-16:00
    - Try combining the imbalance strategy with the overshoot into the morning auction.