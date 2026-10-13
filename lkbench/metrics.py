"""Accuracy and calibration metrics for probabilistic predictions."""
import numpy as np
from scipy.stats import norm


def rmse(y, m):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(m)) ** 2)))


def q2(y, m):
    """Predictivity coefficient 1 - SSE/SST on a test set (1 is perfect)."""
    y, m = np.asarray(y), np.asarray(m)
    return float(1 - np.sum((y - m) ** 2) / np.sum((y - y.mean()) ** 2))


def nlpd(y, m, s):
    """Mean negative log predictive density under N(m, s^2)."""
    s = np.maximum(np.asarray(s), 1e-12)
    return float(np.mean(0.5 * np.log(2 * np.pi * s ** 2) + 0.5 * ((np.asarray(y) - m) / s) ** 2))


def coverage(y, m, s, level=0.95):
    """Fraction of test values inside the central ``level`` prediction interval."""
    z = norm.ppf(0.5 + level / 2)
    return float(np.mean(np.abs(np.asarray(y) - m) <= z * np.asarray(s)))


def crps(y, m, s):
    """Mean continuous ranked probability score of N(m, s^2) (lower is better)."""
    s = np.maximum(np.asarray(s), 1e-12)
    z = (np.asarray(y) - m) / s
    return float(np.mean(s * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))))


def all_metrics(y, m, s=None):
    out = {"rmse": rmse(y, m), "q2": q2(y, m)}
    if s is not None:
        out.update(nlpd=nlpd(y, m, s), coverage95=coverage(y, m, s), crps=crps(y, m, s))
    return out


def finite_diff_grad(f, x, h=1e-6):
    """Central finite-difference gradient of a scalar function, relative step."""
    x = np.asarray(x, float)
    g = np.zeros_like(x)
    for i in range(x.size):
        e = np.zeros_like(x)
        e[i] = h * max(1.0, abs(x[i]))
        g[i] = (f(x + e) - f(x - e)) / (2 * e[i])
    return g
