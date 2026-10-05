"""Formal backtests for VaR models: do breaches occur at the promised rate, and are they independent?"""
import numpy as np
from scipy.special import xlogy
from scipy.stats import chi2


def kupiec_pof(breaches, p):
    """Proportion-of-failures test: is the breach rate consistent with the promised rate p?"""
    b = np.asarray(breaches, int); T, x = len(b), int(b.sum())
    ll = lambda q: xlogy(T - x, 1 - q) + xlogy(x, q)
    lr = max(0.0, -2 * (ll(p) - ll(x / T)))
    return lr, float(chi2.sf(lr, 1))


def christoffersen_ind(breaches):
    """Independence test: do breaches cluster (a sign the model is slow to react to volatility)?"""
    b = np.asarray(breaches, int); a, c = b[:-1], b[1:]
    n00, n01, n10, n11 = [int(((a == i) & (c == j)).sum()) for i in (0, 1) for j in (0, 1)]
    p01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    p11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    p = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll0 = xlogy(n00 + n10, 1 - p) + xlogy(n01 + n11, p)
    ll1 = xlogy(n00, 1 - p01) + xlogy(n01, p01) + xlogy(n10, 1 - p11) + xlogy(n11, p11)
    lr = max(0.0, -2 * (ll0 - ll1))
    return lr, float(chi2.sf(lr, 1))


def traffic_light(breaches_last_250):
    """Basel zones for 99% VaR over 250 days."""
    x = int(np.sum(breaches_last_250))
    return "green" if x <= 4 else "yellow" if x <= 9 else "red"
