# Tripwire

**Which Value-at-Risk model can you actually trust? A backtesting lab that grades risk models the way a bank's model-validation team would.**

Value-at-Risk (VaR) says "on 99% of days, we won't lose more than X." If the model is right, losses should exceed X on about 1% of days, and those days should be scattered, not bunched together in a crisis. Tripwire tests exactly that.

## Models compared
Historical simulation, Normal (variance-covariance), EWMA (RiskMetrics, lambda 0.94), and Filtered Historical Simulation (EWMA volatility plus the empirical tails). Each forecast for day *t* uses only returns before *t* (a test enforces this).

## Statistical tests
- **Kupiec POF:** is the breach rate consistent with the promised 1%?
- **Christoffersen independence:** do breaches cluster, meaning the model reacts too slowly when volatility jumps?
- **Conditional coverage:** both together.
- **Basel traffic light:** green, yellow, or red zone for 99% VaR over the last 250 days.
- **Average VaR:** a model that is safe only because it is huge is expensive, so capital cost is shown too.

## Run it
```bash
pip install -r requirements.txt
pytest
python run.py --synthetic                          # simulated fat-tailed, volatility-clustered returns
python run.py --ticker SPY --start 2005-01-01      # real data from Yahoo Finance
```
Writes `results/backtest.csv` and `results/var_forecasts.png`.

## Limitations
- One asset, one-day horizon, VaR only (no Expected Shortfall yet).
- Passing a test means "no evidence of failure," not "correct." Breach counts in a 1% test are small, so p-values are noisy.
- Simulated results show how the tool behaves, not how markets behave. Report only what you measure on real data.
