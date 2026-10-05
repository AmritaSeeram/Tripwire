"""python run.py --ticker SPY --start 2005-01-01 [--alpha 0.01]      |      python run.py --synthetic"""
import argparse, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from tripwire.backtest import run
from tripwire.data import load_returns, simulate_garch

ap = argparse.ArgumentParser(); ap.add_argument("--ticker", default="SPY"); ap.add_argument("--start", default="2005-01-01")
ap.add_argument("--alpha", type=float, default=0.01); ap.add_argument("--synthetic", action="store_true"); a = ap.parse_args()
r = simulate_garch() if a.synthetic else load_returns(a.ticker, a.start)
df, f = run(r, a.alpha); os.makedirs("results", exist_ok=True); df.to_csv("results/backtest.csv", index=False)
print(df.round(4).to_string(index=False))
fig, ax = plt.subplots(figsize=(10, 5)); k = slice(-500, None); ax.plot(r[k], color="#bbb", lw=.8, label="daily return")
for name, v in f.items(): ax.plot(v[k], lw=1.4, label=name)
ax.set_title(f"{int((1 - a.alpha) * 100)}% one-day VaR forecasts, last 500 days" + (" (simulated data)" if a.synthetic else f" ({a.ticker})")); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig("results/var_forecasts.png", dpi=150)
