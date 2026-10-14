"""Test functions for surrogate modelling, after the Virtual Library of Simulation
Experiments (Surjanovic & Bingham, https://www.sfu.ca/~ssurjano/).

Every function takes ``X`` of shape (n, d) in its *native* domain and returns a
length-n vector. ``TESTS`` gathers the metadata (dimension, domain, a reference
point with its known value) used by ``python_no-model_test-functions-reference-values.ipynb``
to check the implementations themselves before anything is benchmarked on them.
"""
from dataclasses import dataclass, field

import numpy as np


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
