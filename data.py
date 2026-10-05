import numpy as np


def load_returns(ticker: str, start: str, end=None) -> np.ndarray:
    import yfinance as yf  # lazy import: tests and --synthetic need no network
    px = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)["Close"].squeeze()
    return np.log(px).diff().dropna().to_numpy()


def simulate_garch(n=4000, seed=0, omega=1e-6, a=0.10, b=0.88, df=5) -> np.ndarray:
    """Returns with volatility clustering and fat tails (GARCH(1,1) with Student-t shocks)."""
    rng = np.random.default_rng(seed); r = np.empty(n); s2 = omega / (1 - a - b)
    for t in range(n):
        e = rng.standard_t(df) / np.sqrt(df / (df - 2)); r[t] = np.sqrt(s2) * e
        s2 = omega + a * r[t] ** 2 + b * s2
    return r
