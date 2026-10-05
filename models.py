"""One-day-ahead Value-at-Risk forecasts. VaR is a return threshold (negative number): a 'breach'
is a day when the actual return falls below it. The forecast for day t uses returns before t only."""
import numpy as np
from scipy.stats import norm

LAMBDA = 0.94  # RiskMetrics decay


def ewma_var(r: np.ndarray, lam: float = LAMBDA) -> np.ndarray:
    """sig2[t] = variance forecast for day t, built only from r[:t]."""
    s2 = np.empty(len(r)); s2[0] = np.var(r[:50])
    for t in range(1, len(r)):
        s2[t] = lam * s2[t - 1] + (1 - lam) * r[t - 1] ** 2
    return s2


def forecasts(r: np.ndarray, alpha: float = 0.01, window: int = 250) -> dict:
    """Returns {model: array of VaR forecasts}, NaN before `window`."""
    n, out = len(r), {}
    s2 = ewma_var(r); z = r / np.sqrt(s2)
    for name in ("Historical", "Normal", "EWMA", "Filtered Hist. Sim."):
        out[name] = np.full(n, np.nan)
    for t in range(window, n):
        w = r[t - window:t]
        out["Historical"][t] = np.quantile(w, alpha)
        out["Normal"][t] = w.mean() + norm.ppf(alpha) * w.std(ddof=1)
        out["EWMA"][t] = norm.ppf(alpha) * np.sqrt(s2[t])
        out["Filtered Hist. Sim."][t] = np.quantile(z[t - window:t], alpha) * np.sqrt(s2[t])
    return out
