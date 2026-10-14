"""Seeded designs of experiments, all on [0,1]^d (scale with ``TestFunction.scale``)."""
import numpy as np
from scipy.stats import qmc


def lhs(n, d, seed=0, optimized=True):
    """Latin hypercube; ``optimized`` applies scipy's random-CD improvement (space filling)."""
    sampler = qmc.LatinHypercube(d=d, seed=seed, optimization="random-cd" if optimized and d > 1 else None)
    return sampler.random(n)


def sobol(n, d, seed=0):
    return qmc.Sobol(d=d, scramble=True, seed=seed).random(n)


def uniform(n, d, seed=0):
    return np.random.default_rng(seed).random((n, d))


def dataset(tf, n, n_test=1000, seed=0, noise_sd=0.0):
    """Training/test sets for test function ``tf`` on the unit hypercube.

    Returns (X, y, Xt, yt) with X in [0,1]^d (the GP sees the unit cube; the
    function is evaluated in its native domain). ``noise_sd`` adds Gaussian
    noise to y only (yt stays noise-free, it is the target)."""
    X = lhs(n, tf.d, seed=seed)
    Xt = uniform(n_test, tf.d, seed=10_000 + seed)
    y = tf.unit(X)
    if noise_sd > 0:
        y = y + np.random.default_rng(20_000 + seed).normal(0.0, noise_sd, size=n)
    return X, y, Xt, tf.unit(Xt)
