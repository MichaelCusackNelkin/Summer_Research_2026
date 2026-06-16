# Summer_Research_2026
Repo containing work on a summer research project for Statistical Aspects of Finance Spring 2026 at the Columbia MAFN: combining a forecast, covariance, adn spread costs into an optimal portoflio problem, and backtesting it on a point in time universe, covariance estimate, and forecasting model.

### Notes / Research Direction
1. Covariance matrix condition number is ~11,000 and epsilon is a floored simple error. We want closer to 10 and time varying epsilon (idiosyncratic) wiht GARCH(1,1)
    - We use a factor-structure (and add a market-neutrality constraint to the opt
2. The forecast is ok but also probably crap.
    - We need to think creatively about using more features. They may be weaker individually, but powerful in interaction with the stronger imbalance signal.
    - This may allow us to use lasso, or maybe a random forest.
3. Once these things are worked out:
    - maybe shrink the universe active set.
    - We need to plot GMV against different values of the risk-aversion parameter lambda.
    - Try continuous trading version from 15:50-15:55
    - Try shorter horizon from 15:55(ish)-16:00
    - Try multi-period from 15:50-15:55(ish), 15:55(ish)-16:00, or minutely from 15:00-16:00
    - Try combining the imbalance strategy with the overshoot into the morning auction.