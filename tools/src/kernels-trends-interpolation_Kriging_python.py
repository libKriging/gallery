# %% [markdown]
# # Python · Kriging — kernels × trends, interpolation, predict (covariance, derivatives)
#
# **Idea.** The basic promise of noise-free Kriging, over the whole grid of kernels
# and trends: it interpolates the data, its predictive variance vanishes at the design
# points, its predicted derivatives are the derivatives of its prediction, and it is
# accurate on smooth 1D to 3D functions.
#
# **Reference.** Finite differences of the prediction, the data themselves, and per-
# function Q² floors.
#
# Plain noise-free `Kriging`, every kernel (`gauss`, `exp`, `matern3_2`,
# `matern5_2`) crossed with every trend (`constant`, `linear`, `interactive`,
# `quadratic`) on functions of different regularity:
#
# | function | d | character |
# |---|---|---|
# | Forrester | 1 | smooth, one oscillation |
# | Gramacy & Lee (2012) | 1 | fast oscillation then polynomial: non-stationary |
# | Branin | 2 | smooth, three global minima |
# | Six-hump camel | 2 | polynomial, steep borders |
# | log-Goldstein-Price | 2 | strongly varying |
# | Lim polynomial | 2 | low-degree polynomial: the trend should help |
# | Ishigami | 3 | strongly non-linear, interactions |
# | Hartmann 3 | 3 | multimodal, localized |
#
# Checks:
# * **interpolation**: a noise-free kriging mean reproduces the observations, with
#   a (near) zero predictive standard deviation at the design points. Error-level
#   for every kernel but `gauss`: its correlation matrices are notoriously
#   ill-conditioned, and the numerical jitter libKriging adds to factorize them
#   breaks exact interpolation (warn-level, with the size of the gap);
# * **accuracy**: Q² on a test set above a per-function floor for
#   `matern5_2` / `constant` with a multistart optimizer (`BFGS10`): a
#   regression shows up here first. The same floor for the default single-start
#   `BFGS` is warn-level: a single start can end in a poor local optimum of the
#   likelihood (see `objectives-LL-LOO-LMP-gradients_Kriging_python.ipynb`),
#   which is worth seeing in the report but is not a defect of the model;
# * **sanity of the fitted hyperparameters**: finite, positive θ and σ²;
# * **consistency**: `predict` on a batch equals `predict` point by point, the
#   returned covariance has the predicted variances on its diagonal, and is
#   symmetric positive semi-definite.

# %%
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylibkriging as lk

import gallery
from gallery import TESTS, design, metrics, compat

suite = gallery.Suite("kernels-trends-interpolation_Kriging_python")
print("pylibkriging", compat.version())

KERNELS = ["gauss", "exp", "matern3_2", "matern5_2"]
TRENDS = ["constant", "linear", "interactive", "quadratic"]
CASES = {  # function: (n, Q2 floor for matern5_2/constant)
    "forrester": (12, 0.95),
    "gramacy_lee_2012": (20, 0.60),
    "branin": (30, 0.97),
    "six_hump_camel": (30, 0.95),
    "log_goldstein_price": (40, 0.85),
    "lim_polynomial": (15, 0.99),
    "ishigami": (80, 0.75),
    "hartmann3": (40, 0.85),
}


def predict(model, X, **kw):
    m, s, cov, *_ = model.predict(X, return_stdev=True, **kw)
    return np.asarray(m).ravel(), np.asarray(s).ravel(), cov

# %% [markdown]
# ## Kernel × trend grid

# %%
for fname, (n, q2_floor) in CASES.items():
    tf = TESTS[fname]
    X, y, Xt, yt = design.dataset(tf, n, n_test=1000, seed=1)
    for kernel in KERNELS:
        for trend in TRENDS:
            n_beta = {"constant": 1, "linear": 1 + tf.d, "interactive": 1 + tf.d + tf.d * (tf.d - 1) // 2,
                      "quadratic": 1 + 2 * tf.d + tf.d * (tf.d - 1) // 2}[trend]
            case = f"{fname}/{kernel}/{trend}"
            if n_beta >= n - 1:
                suite.check_skip(case, f"{n_beta} trend coefficients for n={n}")
                continue
            t0 = time.time()
            try:
                k = lk.Kriging(y, X, kernel, regmodel=trend, optim="BFGS")
            except Exception as exc:  # noqa: BLE001
                suite.check_true(f"{case}: fit", False, detail=str(exc)[:200])
                continue
            fit_s = time.time() - t0
            m, s, _ = predict(k, Xt)
            res = metrics.all_metrics(yt, m, s)
            suite.metric(case, "pylibkriging", fit_s=fit_s, **res)
            theta, sigma2 = np.asarray(k.theta()).ravel(), float(k.sigma2())
            suite.check_true(f"{case}: theta, sigma2 finite and > 0",
                       np.all(np.isfinite(theta)) and np.all(theta > 0) and np.isfinite(sigma2) and sigma2 > 0,
                       detail=f"theta={theta}, sigma2={sigma2:.3g}")
            # interpolation at the design points (relative to the spread of y)
            md, sd, _ = predict(k, X)
            scale = np.std(y)
            sev = "warn" if kernel == "gauss" else "error"
            suite.check_less(f"{case}: interpolates y (max |m(X)-y| / sd(y))", np.max(np.abs(md - y)) / scale, 1e-4,
                       severity=sev)
            suite.check_less(f"{case}: stdev(X) ~ 0 (max sd / sd(y))", np.max(sd) / scale, 1e-2, severity=sev)
            if kernel == "matern5_2" and trend == "constant":
                suite.check_greater(f"{case}: Q2 (single-start BFGS)", res["q2"], q2_floor, severity="warn")
                k10 = lk.Kriging(y, X, kernel, regmodel=trend, optim="BFGS10")
                res10 = metrics.all_metrics(yt, *predict(k10, Xt)[:2])
                suite.metric(case, "pylibkriging BFGS10", **res10)
                suite.check_greater(f"{case}: Q2 (BFGS10)", res10["q2"], q2_floor)

# %%
tab = suite.table().reset_index()
tab = tab[tab["package"] == "pylibkriging"]
tab[["function", "kernel", "trend"]] = tab["case"].str.split("/", expand=True)
pivot = tab.pivot_table(index=["function", "trend"], columns="kernel", values="q2", sort=False)
pivot.style.format("{:.3f}").background_gradient(axis=None, vmin=0.5, vmax=1, cmap="RdYlGn")

# %% [markdown]
# Expectations that should hold whatever the version (warn-level: they are
# empirical, not mathematical, statements):
# * on the smooth Branin the `exp` kernel (non-differentiable paths) is the least accurate;
# * on the Lim polynomial a `quadratic` trend is at least as good as a `constant` one.

# %%
q = tab.set_index(["function", "kernel", "trend"])["q2"]
suite.check_true("branin: exp is the worst kernel (constant trend)",
           q["branin", "exp", "constant"] <= min(q["branin", k, "constant"] for k in KERNELS),
           severity="warn")
suite.check_greater("lim_polynomial/matern5_2: Q2(quadratic) - Q2(constant)",
              q["lim_polynomial", "matern5_2", "quadratic"] - q["lim_polynomial", "matern5_2", "constant"], -1e-3,
              severity="warn")

# %% [markdown]
# ## Consistency of `predict`

# %%
tf = TESTS["branin"]
X, y, Xt, yt = design.dataset(tf, 30, n_test=50, seed=3)
k = lk.Kriging(y, X, "matern5_2")
m, s, cov = predict(k, Xt, return_cov=True)
cov = np.asarray(cov)
suite.check_close("predict: diag(cov) == stdev^2", np.diag(cov), s ** 2, rtol=1e-6, atol=1e-10)
suite.check_close("predict: cov symmetric", cov, cov.T, rtol=1e-10, atol=1e-12)
suite.check_greater("predict: cov PSD (min eigenvalue / max)", np.linalg.eigvalsh(cov).min() / np.abs(cov).max(), -1e-8)
m1 = np.array([predict(k, Xt[i:i + 1])[0][0] for i in range(len(Xt))])
s1 = np.array([predict(k, Xt[i:i + 1])[1][0] for i in range(len(Xt))])
suite.check_close("predict: batch == point by point (mean)", m, m1, rtol=1e-8, atol=1e-10 * np.std(y))
suite.check_close("predict: batch == point by point (stdev)", s, s1, rtol=1e-6, atol=1e-10 * np.std(y))
# the mean is a function of the point only: duplicating/permuting the batch changes nothing
perm = np.random.default_rng(0).permutation(len(Xt))
suite.check_close("predict: invariant to the order of the batch", predict(k, Xt[perm])[0], m[perm], rtol=1e-10, atol=1e-10)

# %% [markdown]
# ### Derivatives of the predictor
#
# `return_deriv=True` gives the gradients of the mean and of the stdev w.r.t. x;
# compared with central finite differences.

# %%
with suite.check_runs("predict(return_deriv=True)"):
    out = k.predict(Xt[:5], True, False, True)
    dm, ds = np.asarray(out[3]), np.asarray(out[4])
    h = 1e-6
    fd_m, fd_s = np.zeros_like(dm), np.zeros_like(ds)
    for j in range(2):
        e = np.zeros(2); e[j] = h
        mp, sp, _ = predict(k, Xt[:5] + e)
        mm, sm, _ = predict(k, Xt[:5] - e)
        fd_m[:, j], fd_s[:, j] = (mp - mm) / (2 * h), (sp - sm) / (2 * h)
    suite.check_close("d mean / dx == finite differences", dm, fd_m, rtol=1e-4, atol=1e-4 * np.abs(fd_m).max())
    suite.check_close("d stdev / dx == finite differences", ds, fd_s, rtol=1e-3, atol=1e-3 * np.abs(fd_s).max())

# %% [markdown]
# ## A look at the 1D fits

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for ax, fname in zip(axes, ["forrester", "gramacy_lee_2012"]):
    tf = TESTS[fname]
    n = CASES[fname][0]
    X, y, _, _ = design.dataset(tf, n, seed=1)
    u = np.linspace(0, 1, 400)[:, None]
    ax.plot(u, tf.unit(u), "k--", label="truth")
    for kernel in ["gauss", "matern5_2", "exp"]:
        mm, ss, _ = predict(lk.Kriging(y, X, kernel), u)
        ax.plot(u, mm, label=kernel)
        if kernel == "matern5_2":
            ax.fill_between(u[:, 0], mm - 2 * ss, mm + 2 * ss, alpha=0.2)
    ax.plot(X, y, "ko")
    ax.set_title(fname)
    ax.legend()

# %% [markdown]
# ## Summary
#
# The metrics of this notebook, the number of checks per status, and every check that
# did not pass (FAIL: error-level, fails the run; WARN: warn-level; KNOWN: listed in
# `tools/known_issues.csv`, with its issue; SKIP: feature absent from the installed version).
# All of them are also written to `results/kernels-trends-interpolation_Kriging_python.checks.csv` and `.metrics.csv`.

# %%
suite.summary()

# %%
suite.finalize()
