"""Helpers of the libKriging gallery notebooks (Python), in one file.

Loaded by the notebooks with ``import gallery`` (their first cell puts ``tools/python``
on ``sys.path``). The R, Octave and Julia notebooks load ``tools/R/gallery.R``,
``tools/octave/gallery.m`` and ``tools/julia/gallery.jl``, which provide the same
check / metric / summary / data functions under the same names (see each file),
and write the same CSV files, aggregated by ``tools/report.py``.

Sections:
  1. test functions (``TESTS``, ``TestFunction``)
  2. designs of experiments (``design.lhs``, ``design.dataset``, ...)
  3. metrics (``metrics.rmse``, ``metrics.q2``, ``metrics.nlpd``, ...)
  4. what the installed pylibkriging offers (``compat.has``, ``compat.version``)
  5. checks and results (``Suite``: ``check``, ``check_close``, ``metric``, ``summary``, ...)
  6. persistence checks (``persistence.check``)
  7. shared datasets (``read_data``)
"""
import copy
import csv
import functools
import importlib
import math
import os
import pickle
import tempfile
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.stats import norm, qmc


# ======================================================================================
# 1. Test functions
# ======================================================================================

# Test functions for surrogate modelling, after the Virtual Library of Simulation
# Experiments (Surjanovic & Bingham, https://www.sfu.ca/~ssurjano/).
#
# Every function takes ``X`` of shape (n, d) in its *native* domain and returns a
# length-n vector. ``TESTS`` gathers the metadata (dimension, domain, a reference
# point with its known value) used by ``test-functions-reference-values_no-model_python.ipynb``
# to check the implementations themselves before any model is fitted on them.

# --- 1D -------------------------------------------------------------------------

def forrester(X):
    x = X[:, 0]
    return (6 * x - 2) ** 2 * np.sin(12 * x - 4)


def gramacy_lee_2012(X):
    x = X[:, 0]
    return np.sin(10 * np.pi * x) / (2 * x) + (x - 1) ** 4


# --- 2D -------------------------------------------------------------------------

def branin(X):
    x1, x2 = X[:, 0], X[:, 1]
    a, b, c = 1.0, 5.1 / (4 * np.pi ** 2), 5 / np.pi
    r, s, t = 6.0, 10.0, 1 / (8 * np.pi)
    return a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s


def six_hump_camel(X):
    x1, x2 = X[:, 0], X[:, 1]
    return (4 - 2.1 * x1 ** 2 + x1 ** 4 / 3) * x1 ** 2 + x1 * x2 + (-4 + 4 * x2 ** 2) * x2 ** 2


def goldstein_price(X):
    x1, x2 = X[:, 0], X[:, 1]
    a = 1 + (x1 + x2 + 1) ** 2 * (19 - 14 * x1 + 3 * x1 ** 2 - 14 * x2 + 6 * x1 * x2 + 3 * x2 ** 2)
    b = 30 + (2 * x1 - 3 * x2) ** 2 * (18 - 32 * x1 + 12 * x1 ** 2 + 48 * x2 - 36 * x1 * x2 + 27 * x2 ** 2)
    return a * b


def log_goldstein_price(X):
    """Logarithmic Goldstein-Price (Picheny et al. 2013), on [0,1]^2."""
    Z = 4 * X - 2
    return (np.log(goldstein_price(Z)) - 8.693) / 2.427


def currin_exp(X):
    x1, x2 = X[:, 0], X[:, 1]
    return (1 - np.exp(-1 / (2 * x2))) * (2300 * x1 ** 3 + 1900 * x1 ** 2 + 2092 * x1 + 60) / (
        100 * x1 ** 3 + 500 * x1 ** 2 + 4 * x1 + 20)


def lim_polynomial(X):
    x1, x2 = X[:, 0], X[:, 1]
    return (9 + 5 / 2 * x1 - 35 / 2 * x2 + 5 / 2 * x1 * x2 + 19 * x2 ** 2 - 15 / 2 * x1 ** 3
            - 5 / 2 * x1 * x2 ** 2 - 11 / 2 * x2 ** 4 + x1 ** 3 * x2 ** 2)


def gramacy_lee_2008(X):
    """Non-stationary 2D function x1 exp(-x1^2 - x2^2) on [-2, 6]^2."""
    x1, x2 = X[:, 0], X[:, 1]
    return x1 * np.exp(-x1 ** 2 - x2 ** 2)


# --- 3D to 5D -------------------------------------------------------------------

def ishigami(X, a=7.0, b=0.1):
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    return np.sin(x1) + a * np.sin(x2) ** 2 + b * x3 ** 4 * np.sin(x1)


_H3_ALPHA = np.array([1.0, 1.2, 3.0, 3.2])
_H3_A = np.array([[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]], float)
_H3_P = 1e-4 * np.array([[3689, 1170, 2673], [4699, 4387, 7470],
                         [1091, 8732, 5547], [381, 5743, 8828]], float)


def hartmann3(X):
    diff2 = (X[:, None, :] - _H3_P[None, :, :]) ** 2
    return -np.exp(-np.sum(_H3_A[None, :, :] * diff2, axis=2)) @ _H3_ALPHA


def friedman(X):
    x1, x2, x3, x4, x5 = (X[:, i] for i in range(5))
    return 10 * np.sin(np.pi * x1 * x2) + 20 * (x3 - 0.5) ** 2 + 10 * x4 + 5 * x5


# --- 6D to 8D -------------------------------------------------------------------

_H6_ALPHA = np.array([1.0, 1.2, 3.0, 3.2])
_H6_A = np.array([[10, 3, 17, 3.5, 1.7, 8],
                  [0.05, 10, 17, 0.1, 8, 14],
                  [3, 3.5, 1.7, 10, 17, 8],
                  [17, 8, 0.05, 10, 0.1, 14]], float)
_H6_P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                         [2329, 4135, 8307, 3736, 1004, 9991],
                         [2348, 1451, 3522, 2883, 3047, 6650],
                         [4047, 8828, 8732, 5743, 1091, 381]], float)


def hartmann6(X):
    diff2 = (X[:, None, :] - _H6_P[None, :, :]) ** 2
    return -np.exp(-np.sum(_H6_A[None, :, :] * diff2, axis=2)) @ _H6_ALPHA


def otl_circuit(X):
    Rb1, Rb2, Rf, Rc1, Rc2, beta = (X[:, i] for i in range(6))
    Vb1 = 12 * Rb2 / (Rb1 + Rb2)
    den = beta * (Rc2 + 9) + Rf
    return ((Vb1 + 0.74) * beta * (Rc2 + 9) / den + 11.35 * Rf / den
            + 0.74 * Rf * beta * (Rc2 + 9) / (den * Rc1))


def piston(X):
    M, S, V0, k, P0, Ta, T0 = (X[:, i] for i in range(7))
    A = P0 * S + 19.62 * M - k * V0 / S
    V = S / (2 * k) * (np.sqrt(A ** 2 + 4 * k * P0 * V0 * Ta / T0) - A)
    return 2 * np.pi * np.sqrt(M / (k + S ** 2 * P0 * V0 * Ta / (T0 * V ** 2)))


def borehole(X):
    rw, r, Tu, Hu, Tl, Hl, L, Kw = (X[:, i] for i in range(8))
    lnr = np.log(r / rw)
    return 2 * np.pi * Tu * (Hu - Hl) / (lnr * (1 + 2 * L * Tu / (lnr * rw ** 2 * Kw) + Tu / Tl))


def robot_arm(X):
    theta, L = X[:, :4], X[:, 4:]
    angles = np.cumsum(theta, axis=1)
    u = np.sum(L * np.cos(angles), axis=1)
    v = np.sum(L * np.sin(angles), axis=1)
    return np.sqrt(u ** 2 + v ** 2)


def dette_pepelyshev(X):
    x = X
    s = 0.0
    for i in range(4, 9):  # i = 4..8, inner sum over j = 3..i (1-based)
        s = s + i * np.log(1 + np.sum(x[:, 2:i], axis=1))
    return (4 * (x[:, 0] - 2 + 8 * x[:, 1] - 8 * x[:, 1] ** 2) ** 2 + (3 - 4 * x[:, 1]) ** 2
            + 16 * np.sqrt(x[:, 2] + 1) * (2 * x[:, 2] - 1) ** 2 + s)


# --- 10D and more ---------------------------------------------------------------

def wing_weight(X):
    Sw, Wfw, A, Lam, q, lam, tc, Nz, Wdg, Wp = (X[:, i] for i in range(10))
    Lam = np.deg2rad(Lam)
    return (0.036 * Sw ** 0.758 * Wfw ** 0.0035 * (A / np.cos(Lam) ** 2) ** 0.6 * q ** 0.006
            * lam ** 0.04 * (100 * tc / np.cos(Lam)) ** -0.3 * (Nz * Wdg) ** 0.49 + Sw * Wp)


def welch(X):
    x = [None] + [X[:, i] for i in range(20)]  # 1-based, as in the reference
    return (5 * x[12] / (1 + x[1]) + 5 * (x[4] - x[20]) ** 2 + x[5] + 40 * x[19] ** 3 - 5 * x[19]
            + 0.05 * x[2] + 0.08 * x[3] - 0.03 * x[6] + 0.03 * x[7] - 0.09 * x[9] - 0.01 * x[10]
            - 0.07 * x[11] + 0.25 * x[13] ** 2 - 0.04 * x[14] + 0.06 * x[15] - 0.01 * x[17]
            - 0.03 * x[18])


def ackley(X, a=20.0, b=0.2, c=2 * np.pi):
    d = X.shape[1]
    return (-a * np.exp(-b * np.sqrt(np.sum(X ** 2, axis=1) / d))
            - np.exp(np.sum(np.cos(c * X), axis=1) / d) + a + np.e)


@dataclass
class TestFunction:
    name: str
    f: callable
    lower: np.ndarray
    upper: np.ndarray
    # (x, f(x)) pairs taken from the reference pages (global optima, nominal values)
    known: list = field(default_factory=list)
    tags: tuple = ()

    @property
    def d(self):
        return len(self.lower)

    def __call__(self, X):
        return self.f(np.atleast_2d(X))

    def scale(self, U):
        """Map points of [0,1]^d to the native domain."""
        return self.lower + np.asarray(U) * (self.upper - self.lower)

    def unit(self, U):
        """The function seen on [0,1]^d."""
        return self.f(self.scale(np.atleast_2d(U)))


def _box(*bounds):
    b = np.array(bounds, float)
    return b[:, 0], b[:, 1]


def _tf(name, f, bounds, known=(), tags=()):
    lo, up = _box(*bounds)
    return TestFunction(name, f, lo, up, [(np.array(x, float), v) for x, v in known], tuple(tags))


TESTS = {t.name: t for t in [
    _tf("forrester", forrester, [(0, 1)], [([0.757249], -6.02074)], ["1d"]),
    _tf("gramacy_lee_2012", gramacy_lee_2012, [(0.5, 2.5)], [([0.548563], -0.869011)], ["1d", "nonstationary"]),
    _tf("branin", branin, [(-5, 10), (0, 15)],
        [([-np.pi, 12.275], 0.397887), ([np.pi, 2.275], 0.397887), ([9.42478, 2.475], 0.397887)], ["2d"]),
    _tf("six_hump_camel", six_hump_camel, [(-3, 3), (-2, 2)],
        [([0.0898, -0.7126], -1.0316), ([-0.0898, 0.7126], -1.0316)], ["2d"]),
    _tf("goldstein_price", goldstein_price, [(-2, 2), (-2, 2)], [([0, -1], 3.0)], ["2d", "large-range"]),
    _tf("log_goldstein_price", log_goldstein_price, [(0, 1), (0, 1)],
        [([0.5, 0.25], (np.log(3.0) - 8.693) / 2.427)], ["2d"]),
    _tf("currin_exp", currin_exp, [(0, 1), (1e-6, 1)], [], ["2d"]),
    _tf("lim_polynomial", lim_polynomial, [(0, 1), (0, 1)], [([0, 0], 9.0)], ["2d", "polynomial"]),
    _tf("gramacy_lee_2008", gramacy_lee_2008, [(-2, 6), (-2, 6)],
        [([-1 / np.sqrt(2), 0], -np.exp(-0.5) / np.sqrt(2))], ["2d", "nonstationary"]),
    _tf("ishigami", ishigami, [(-np.pi, np.pi)] * 3, [([0, 0, 0], 0.0), ([0, np.pi / 2, 0], 7.0)], ["3d"]),
    _tf("hartmann3", hartmann3, [(0, 1)] * 3, [([0.114614, 0.555649, 0.852547], -3.86278)], ["3d"]),
    _tf("friedman", friedman, [(0, 1)] * 5, [([0, 0, 0.5, 0, 0], 0.0)], ["5d"]),
    _tf("hartmann6", hartmann6, [(0, 1)] * 6,
        [([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573], -3.32237)], ["6d"]),
    _tf("otl_circuit", otl_circuit, [(50, 150), (25, 70), (0.5, 3), (1.2, 2.5), (0.25, 1.2), (50, 300)],
        [], ["6d", "engineering"]),
    _tf("piston", piston, [(30, 60), (0.005, 0.020), (0.002, 0.010), (1000, 5000), (90000, 110000),
                           (290, 296), (340, 360)], [], ["7d", "engineering"]),
    _tf("borehole", borehole, [(0.05, 0.15), (100, 50000), (63070, 115600), (990, 1110),
                               (63.1, 116), (700, 820), (1120, 1680), (9855, 12045)], [], ["8d", "engineering"]),
    _tf("robot_arm", robot_arm, [(0, 2 * np.pi)] * 4 + [(0, 1)] * 4,
        [([0, 0, 0, 0, 1, 1, 1, 1], 4.0), ([0, np.pi, 0, 0, 1, 1, 1, 1], 2.0)], ["8d", "oscillating"]),
    _tf("dette_pepelyshev", dette_pepelyshev, [(0, 1)] * 8, [], ["8d", "curved"]),
    _tf("wing_weight", wing_weight, [(150, 200), (220, 300), (6, 10), (-10, 10), (16, 45), (0.5, 1),
                                     (0.08, 0.18), (2.5, 6), (1700, 2500), (0.025, 0.08)], [], ["10d", "engineering"]),
    _tf("welch", welch, [(-0.5, 0.5)] * 20, [([0] * 20, 0.0)], ["20d", "screening"]),
    _tf("ackley4", ackley, [(-2, 2)] * 4, [([0, 0, 0, 0], 0.0)], ["4d", "multimodal"]),
]}

# ======================================================================================
# 2. Designs of experiments
# ======================================================================================

# Seeded designs of experiments, all on [0,1]^d (scale with ``TestFunction.scale``).

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

# ======================================================================================
# 3. Metrics
# ======================================================================================

# Accuracy and calibration metrics for probabilistic predictions.

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

# ======================================================================================
# 4. What the installed pylibkriging offers
# ======================================================================================

# What the installed pylibkriging offers, so that one notebook runs against both
# the last release and the development head: a feature missing from the installed
# version is recorded as SKIP (with the reason), never as a failure.

def version():
    import pylibkriging as lk
    v = getattr(lk, "__version__", None)
    if v is None:
        try:
            from importlib.metadata import version as _v
            v = _v("pylibkriging")
        except Exception:  # noqa: BLE001
            v = "unknown"
    return v


@functools.lru_cache(None)
def has(feature):
    """Probe a feature on a tiny problem (cached)."""
    import pylibkriging as lk
    X = np.linspace(0, 1, 6).reshape(-1, 1)
    y = np.sin(3 * X[:, 0])
    probes = {
        "method": lambda: lk.Kriging(y, X, "matern5_2", method="vecchia(3)"),
        "noise": lambda: lk.Kriging(y, X, "matern5_2", noise="nugget"),
        "WarpKriging": lambda: lk.WarpKriging(y, X, ["none"], "matern5_2"),
        "MLPKriging": lambda: lk.MLPKriging(y, X, hidden_dims=[2], d_out=1, kernel="gauss"),
        "NestedKriging": lambda: lk.NestedKriging(y, X, "matern5_2", nb_groups=2),
        "subsetOfData": lambda: lk.Kriging.subsetOfData(X, n_max=3, method="kmeans", seed=1),
        "save": lambda: lk.Kriging(y, X, "matern5_2").save,
        "pickle": lambda: __import__("pickle").dumps(lk.Kriging(y, X, "matern5_2")),
        "NestedKriging.save": lambda: lk.NestedKriging(y, X, "matern5_2", nb_groups=2).save,
        "load(noise)": lambda: _roundtrip(lk, lk.Kriging(y, X, "matern5_2", noise="nugget")),
        "sklearn": lambda: importlib.import_module("pylibkriging.sklearn").KrigingRegressor,
    }
    try:
        probes[feature]()
        return True
    except Exception:  # noqa: BLE001
        return False


def _roundtrip(lk, k):
    import os
    import tempfile
    f = os.path.join(tempfile.mkdtemp(), "k.json")
    k.save(f)
    return lk.load(f)


def optional(module):
    """Import an optional comparison package, or return None."""
    try:
        return importlib.import_module(module)
    except Exception:  # noqa: BLE001 - broken installs count as absent
        return None

# ======================================================================================
# 5. Checks and results
# ======================================================================================

# Record checks and metrics from a notebook, and fail it at the end if needed.
#
# A notebook creates one ``Suite``, records any number of checks (assertions that
# do *not* stop the notebook, so every problem of a run is reported at once) and
# metrics, then calls ``suite.finalize()`` in its last cell: it writes
#
#   results/<suite>.checks.csv   suite,lang,check,severity,status,value,expected,detail
#   results/<suite>.metrics.csv  suite,lang,case,package,metric,value
#
# and raises ``AssertionError`` if any check of severity "error" failed, which makes
# the notebook (hence the CI job) fail. Checks of severity "warn" are reported but
# never fail the run: they flag performance drifts or comparisons with other
# packages, where a threshold is a judgement call rather than a correctness bound.
#
# Statuses: PASS, FAIL (error-level), WARN (warn-level), SKIP (feature missing in
# the installed version), KNOWN (a failing check, of either severity, listed in
# ``tools/known_issues.csv``: reported with its issue, but does not fail the run). A
# listed check that passes is reported as FIXED by ``tools/report.py``, so that the
# list does not rot.
#
# Same names in every language: ``suite.check_close(...)`` here is ``check_close(s, ...)``
# in R and Octave and ``check_close!(s, ...)`` in Julia; ``suite.metric``,
# ``suite.summary()`` and ``suite.finalize()`` are ``metric``, ``suite_summary`` and
# ``suite_finalize`` there.

def _find_root():
    """The repository root: $LKGALLERY_ROOT, or the one this file belongs to
    (tools/python/gallery.py)."""
    root = Path(os.environ.get("LKGALLERY_ROOT", Path(__file__).resolve().parents[2]))
    if not (root / "tools" / "known_issues.csv").is_file():
        raise RuntimeError(f"{root} is not the libKriging gallery repository (no tools/known_issues.csv)")
    return root


ROOT = _find_root()
RESULTS_DIR = Path(os.environ.get("LKGALLERY_RESULTS", ROOT / "results"))
KNOWN_ISSUES_FILE = ROOT / "tools" / "known_issues.csv"
DATA_DIR = ROOT / "data"


def known_issues():
    """{(suite, check): issue} from tools/known_issues.csv."""
    with open(KNOWN_ISSUES_FILE, newline="") as fh:
        return {(r["suite"], r["check"]): r["issue"] for r in csv.DictReader(fh)}


CHECK_FIELDS = ["suite", "lang", "check", "severity", "status", "value", "expected", "detail"]
METRIC_FIELDS = ["suite", "lang", "case", "package", "metric", "value"]


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, float, np.integer, np.floating)):
        return f"{float(v):.6g}"
    return str(v)


class Suite:
    def __init__(self, name, lang="python"):
        self.name, self.lang = name, lang
        self.checks, self.metrics = [], []
        self.t0 = time.time()
        self.known = known_issues()

    # -- checks ---------------------------------------------------------------

    def check(self, check, ok, value=None, expected=None, detail="", severity="error"):
        status = "PASS" if ok else ("FAIL" if severity == "error" else "WARN")
        if not ok and (self.name, check) in self.known:
            status, detail = "KNOWN", f"{self.known[self.name, check]} | {detail}".rstrip(" |")
        self.checks.append(dict(suite=self.name, lang=self.lang, check=check, severity=severity,
                                status=status, value=_fmt(value), expected=_fmt(expected), detail=detail))
        if not ok:
            print(f"[{status}] {check}: value={_fmt(value)} expected {_fmt(expected)} {detail}")
        return ok

    def check_true(self, check, cond, detail="", severity="error"):
        return self.check(check, bool(cond), cond, True, detail, severity)

    def check_less(self, check, value, bound, detail="", severity="error"):
        """value < bound (NaN fails)."""
        value = float(value)
        return self.check(check, math.isfinite(value) and value < bound, value, f"< {bound:g}", detail, severity)

    def check_greater(self, check, value, bound, detail="", severity="error"):
        value = float(value)
        return self.check(check, math.isfinite(value) and value > bound, value, f"> {bound:g}", detail, severity)

    def check_between(self, check, value, lo, hi, detail="", severity="error"):
        value = float(value)
        return self.check(check, math.isfinite(value) and lo <= value <= hi, value, f"in [{lo:g}, {hi:g}]",
                         detail, severity)

    def check_close(self, check, a, b, rtol=1e-6, atol=0.0, detail="", severity="error"):
        """All |a - b| <= atol + rtol |b| (element-wise); reports the worst relative gap."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        if a.shape != b.shape:
            return self.check(check, False, f"shape {a.shape}", f"shape {b.shape}", detail, severity)
        gap = np.abs(a - b)
        ok = bool(np.all(np.isfinite(a)) and np.all(gap <= atol + rtol * np.abs(b)))
        rel = float(np.max(gap / np.maximum(np.abs(b), 1e-300))) if a.size else 0.0
        return self.check(check, ok, rel, f"rtol {rtol:g}, atol {atol:g}", detail, severity)

    def check_finite(self, check, a, detail="", severity="error"):
        a = np.asarray(a, float)
        return self.check(check, bool(a.size and np.all(np.isfinite(a))), f"{np.sum(~np.isfinite(a))} non-finite",
                         "all finite", detail, severity)

    def check_skip(self, check, reason):
        self.checks.append(dict(suite=self.name, lang=self.lang, check=check, severity="info",
                                status="SKIP", value="", expected="", detail=reason))
        print(f"[SKIP] {check}: {reason}")

    @contextmanager
    def check_runs(self, check, severity="error"):
        """``with suite.check_runs("x"): ...`` records a FAIL (with the exception) instead of stopping."""
        try:
            yield
        except Exception as exc:  # noqa: BLE001 - every failure is a finding here
            self.check(check, False, type(exc).__name__, "no exception",
                      str(exc).splitlines()[0][:200] if str(exc) else traceback.format_exc(limit=1), severity)
        else:
            self.check(check, True, "ok", "no exception", "", severity)

    def check_raises(self, check, fn, *args, severity="error", **kwargs):
        """``fn(*args)`` must raise (invalid input must be rejected, not silently accepted)."""
        try:
            fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            return self.check(check, True, type(exc).__name__, "an exception", "", severity)
        return self.check(check, False, "no exception", "an exception", "", severity)

    # -- metrics --------------------------------------------------------------

    def metric(self, case, package, **values):
        for k, v in values.items():
            self.metrics.append(dict(suite=self.name, lang=self.lang, case=case, package=package,
                                     metric=k, value=_fmt(v)))

    def table(self):
        """Metrics as a wide pandas DataFrame (one row per case x package)."""
        import pandas as pd
        if not self.metrics:
            return pd.DataFrame()
        df = pd.DataFrame(self.metrics)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df.pivot_table(index=["case", "package"], columns="metric", values="value", sort=False)

    def summary(self):
        """End-of-notebook summary: the metrics table, the count per status, and every check
        that did not pass (with its value, threshold and, for KNOWN, its issue)."""
        import pandas as pd
        try:
            from IPython.display import display
        except ImportError:  # outside Jupyter
            display = print
        if self.metrics:
            display(self.table().round(4))
        df = pd.DataFrame(self.checks, columns=CHECK_FIELDS)
        counts = {s: int((df["status"] == s).sum()) for s in ("PASS", "FAIL", "WARN", "KNOWN", "SKIP")}
        print(f"{len(df)} checks: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
        others = df[df["status"] != "PASS"][["status", "severity", "check", "value", "expected", "detail"]]
        if len(others):
            display(others.reset_index(drop=True))

    # -- output ---------------------------------------------------------------

    def finalize(self):
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        self.metric("_suite", "_", elapsed_s=time.time() - self.t0)
        for rows, fields, ext in ((self.checks, CHECK_FIELDS, "checks"), (self.metrics, METRIC_FIELDS, "metrics")):
            with open(RESULTS_DIR / f"{self.name}.{ext}.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)
        counts = {s: sum(c["status"] == s for c in self.checks) for s in ("PASS", "FAIL", "WARN", "KNOWN", "SKIP")}
        print(f"{self.name}: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
        failed = [c for c in self.checks if c["status"] == "FAIL"]
        if failed:
            raise AssertionError(f"{len(failed)} check(s) failed in {self.name}: "
                                 + "; ".join(c["check"] for c in failed[:20]))
        return counts

# ======================================================================================
# 6. Persistence checks
# ======================================================================================

# Persistence checks shared by the notebooks of every model class: a model saved and
# loaded back (class-specific and generic ``pylibkriging.load``), pickled, or deep-copied
# must predict exactly as the original.

def check_persistence(suite, name, k, Xn, predict):
    """Record the persistence checks of model ``k`` (label ``name``) on the points ``Xn``.

    ``predict(model, Xn)`` returns (mean, stdev) as 1-D arrays; it carries the
    prediction method of an approximate fit (Vecchia / Nystrom)."""
    import pylibkriging as lk
    m0, s0 = predict(k, Xn)
    cls = type(k)
    path = Path(tempfile.mkdtemp()) / f"{name.replace(' ', '_')}.json"
    if not hasattr(k, "save"):
        suite.check_skip(f"{name}: save + load", "no save() in this pylibkriging")
    else:
        with suite.check_runs(f"{name}: save + load"):
            k.save(str(path))
            k1 = cls.load(str(path))
            m1, s1 = predict(k1, Xn)
            suite.check_close(f"{name}: load(save(k)) predicts the same mean", m1, m0, rtol=1e-12, atol=1e-12)
            suite.check_close(f"{name}: load(save(k)) predicts the same stdev", s1, s0, rtol=1e-10, atol=1e-12)
            if hasattr(k, "theta"):
                suite.check_close(f"{name}: same theta after load", np.ravel(k1.theta()), np.ravel(k.theta()),
                            rtol=1e-14, atol=0)
        if hasattr(lk, "load") and (has("load(noise)") or "noise" not in name and "nugget" not in name):
            with suite.check_runs(f"{name}: generic pylibkriging.load"):
                k2 = lk.load(str(path))
                suite.check_true(f"{name}: generic load returns {cls.__name__}", type(k2).__name__ == cls.__name__,
                           detail=type(k2).__name__)
                suite.check_close(f"{name}: generic load predicts the same", predict(k2, Xn)[0], m0, rtol=1e-12,
                            atol=1e-12)
    if not has("pickle"):
        suite.check_skip(f"{name}: pickle / deepcopy", "models not picklable in this pylibkriging")
        return
    with suite.check_runs(f"{name}: pickle round trip"):
        k3 = pickle.loads(pickle.dumps(k))
        suite.check_close(f"{name}: pickle predicts the same", predict(k3, Xn)[0], m0, rtol=1e-12, atol=1e-12)
    with suite.check_runs(f"{name}: deepcopy"):
        k4 = copy.deepcopy(k)
        suite.check_close(f"{name}: deepcopy predicts the same", predict(k4, Xn)[0], m0, rtol=1e-12, atol=1e-12)

# ======================================================================================
# 7. Shared datasets
# ======================================================================================

def read_data(name, stem):
    """data/<name>_<stem>.csv (no header) as a 2-D array, e.g. read_data("branin", "X")."""
    return np.loadtxt(DATA_DIR / f"{name}_{stem}.csv", delimiter=",", ndmin=2)


# the notebooks use these names: design.dataset(...), metrics.q2(...), compat.has(...), ...
design = SimpleNamespace(lhs=lhs, sobol=sobol, uniform=uniform, dataset=dataset)
metrics = SimpleNamespace(rmse=rmse, q2=q2, nlpd=nlpd, coverage=coverage, crps=crps, all_metrics=all_metrics,
                          finite_diff_grad=finite_diff_grad)
compat = SimpleNamespace(version=version, has=has, optional=optional)
persistence = SimpleNamespace(check=check_persistence)
