# %% [markdown]
# # Python · Kriging — objectives (LL, LOO, LMP), their gradients and optima
#
# **Idea.** The three estimation criteria (likelihood, leave-one-out, marginal
# posterior) must be computed and differentiated correctly, and the fitted θ̂ must be
# an optimum of the criterion it maximises.
#
# **Reference.** Finite differences of each criterion, explicit leave-one-out refits,
# and the criterion profile around θ̂.
#
# The hyperparameters θ of `Kriging` are estimated by maximizing the
# log-likelihood (`LL`), maximizing the log marginal posterior (`LMP`), or
# minimizing the leave-one-out error (`LOO`). This notebook checks the machinery
# behind those fits rather than the predictions:
#
# * **analytic gradients** of `logLikelihoodFun`, `leaveOneOutFun`,
#   `logMargPostFun` against central finite differences, at random θ, for
#   every kernel;
# * **optimality** of the fitted θ: the objective at θ̂ is not improved by small
#   perturbations of θ̂ (a local optimum was reached). A component θ_j > 5 on
#   the unit cube is a nearly linear direction whose optimum sits at the upper
#   bound of θ: moving it further out is allowed to improve the objective;
# * **global vs local optimum**: the gap between the default single-start fit
#   and a 20-start fit (warn-level: it measures how often the default fit ends in
#   a poor local optimum);
# * **consistency** between `logLikelihood()` and `logLikelihoodFun(theta())`;
# * the closed-form **leave-one-out** predictions of `leaveOneOutVec` against an
#   explicit refit without each point (same θ, σ²);
# * multistart (`BFGS10`) never ends below a single `BFGS` run (warn-level: a
#   multistart that does worse points to a starting-point issue).
#
# Functions: Branin (2D), Hartmann 3 (3D), Ishigami (3D), Friedman (5D).

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylibkriging as lk

import lkgallery
from lkgallery import TESTS, design, metrics, compat

suite = lkgallery.Suite("objectives-LL-LOO-LMP-gradients_Kriging_python")
print("pylibkriging", compat.version())
CASES = {"branin": 25, "hartmann3": 40, "ishigami": 60, "friedman": 50}
KERNELS = ["gauss", "exp", "matern3_2", "matern5_2"]
rng = np.random.default_rng(0)


def objective_fun(k, objective):
    """(value, gradient) of the objective as a function of theta."""
    if objective == "LL":
        return lambda t, g=True: k.logLikelihoodFun(np.asarray(t, float), g)[:2]
    if objective == "LOO":
        return lambda t, g=True: k.leaveOneOutFun(np.asarray(t, float), g)
    return lambda t, g=True: k.logMargPostFun(np.asarray(t, float), g)

# %% [markdown]
# ## Analytic gradients vs finite differences

# %%
rows = []
for fname, n in CASES.items():
    tf = TESTS[fname]
    X, y, _, _ = design.dataset(tf, n, seed=2)
    for kernel in KERNELS:
        k = lk.Kriging(y, X, kernel)
        for objective in ["LL", "LOO", "LMP"]:
            f = objective_fun(k, objective)
            worst = 0.0
            for _ in range(3):
                theta = np.exp(rng.uniform(np.log(0.05), np.log(1.0), tf.d))
                val, grad = f(theta)
                fd = metrics.finite_diff_grad(lambda t: f(t, False)[0], theta, h=1e-5)
                grad = np.asarray(grad).ravel()
                err = np.max(np.abs(grad - fd)) / max(np.max(np.abs(fd)), 1e-8)
                worst = max(worst, err)
            rows.append((fname, kernel, objective, worst))
            suite.less(f"{fname}/{kernel}/{objective}: gradient vs finite diff (rel. err)", worst, 1e-3)
pd.DataFrame(rows, columns=["function", "kernel", "objective", "max rel. error"]).pivot_table(
    index=["function", "kernel"], columns="objective", values="max rel. error").style.format("{:.1e}")

# %% [markdown]
# ## Optimality of the fitted θ, objective by objective

# %%
SIGN = {"LL": 1, "LMP": 1, "LOO": -1}  # +1: maximized, -1: minimized
for fname, n in CASES.items():
    tf = TESTS[fname]
    X, y, Xt, yt = design.dataset(tf, n, n_test=1000, seed=2)
    for objective in ["LL", "LOO", "LMP"]:
        k = lk.Kriging(y, X, "matern5_2", objective=objective)
        theta = np.asarray(k.theta()).ravel()
        f = objective_fun(k, objective)
        best = SIGN[objective] * f(theta, False)[0]
        improved = []
        for j in range(tf.d):
            for fac in (0.9, 1.1):
                t = theta.copy(); t[j] *= fac
                at_bound = fac > 1 and theta[j] > 5
                # tolerance: the optimizer's stopping criterion (0.01 on a log-density is immaterial)
                tol = 1e-4 * abs(best) + (0.0 if objective == "LOO" else 1e-2)
                if SIGN[objective] * f(t, False)[0] > best + tol and not at_bound:
                    improved.append(f"theta{j + 1}*{fac}")
        suite.true(f"{fname}/{objective}: theta_hat not improved by +/-10% moves", not improved,
                   detail=f"theta={np.round(theta, 4)}; improving moves: {improved}")
        m, s, *_ = k.predict(Xt, return_stdev=True)
        suite.metric(f"{fname}/{objective}", "pylibkriging", **metrics.all_metrics(yt, m, s))

# %%
tab = suite.table().reset_index()
tab[["function", "objective"]] = tab["case"].str.split("/", expand=True)
tab.pivot_table(index="function", columns="objective", values=["q2", "coverage95"]).round(3)

# %% [markdown]
# ## In pictures: each criterion along θ = t · θ̂ (Branin)
#
# Each criterion fitted on its own, then evaluated along the ray t · θ̂ through its
# optimum (t = 1, dashed line): a maximum for LL and LMP, a minimum for LOO.
# The three optima are not the same θ, which is why the fitted models differ.

# %%
tf = TESTS["branin"]
X, y, _, _ = design.dataset(tf, CASES["branin"], seed=2)
ts = np.exp(np.linspace(np.log(0.2), np.log(5), 61))
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
for ax, objective in zip(axes, ["LL", "LOO", "LMP"]):
    k = lk.Kriging(y, X, "matern5_2", objective=objective)
    theta = np.asarray(k.theta()).ravel()
    f = objective_fun(k, objective)
    vals = [f(t * theta, False)[0] for t in ts]
    ax.semilogx(ts, vals); ax.axvline(1, color="k", ls="--", lw=.8)
    ax.set_title(f"{objective}, θ̂ = {np.round(theta, 3)}"); ax.set_xlabel("t (θ = t · θ̂)")
    i = int(np.argmax(vals) if SIGN[objective] > 0 else np.argmin(vals))
    suite.between(f"branin/{objective}: optimum of the profile along t * theta_hat (t)", ts[i], 0.8, 1.25,
                  severity="warn")
axes[0].set_ylabel("criterion"); plt.tight_layout(); plt.show()

# %% [markdown]
# ## `logLikelihood()` == `logLikelihoodFun(theta())` (and the same for LOO, LMP)

# %%
tf = TESTS["hartmann3"]
X, y, _, _ = design.dataset(tf, 40, seed=5)
for objective, getter in [("LL", "logLikelihood"), ("LOO", "leaveOneOut"), ("LMP", "logMargPost")]:
    k = lk.Kriging(y, X, "matern5_2", objective=objective)
    v_fun = objective_fun(k, objective)(np.asarray(k.theta()).ravel(), False)[0]
    suite.close(f"{getter}() == {objective} function at theta()", getattr(k, getter)(), v_fun, rtol=1e-8)

# %% [markdown]
# ## Closed-form leave-one-out vs explicit refits
#
# `leaveOneOutVec(θ)` gives the mean and stdev of the prediction of each `y_i`
# from the n−1 others, by the closed-form (Dubrule) formulas. An explicit refit
# without point i, with θ and σ² fixed (β re-estimated, as in the formula), must
# give the same mean.

# %%
for fname in ["branin", "hartmann3"]:
    tf = TESTS[fname]
    X, y, _, _ = design.dataset(tf, 20, seed=6)
    for trend in ["constant", "linear"]:
        k = lk.Kriging(y, X, "matern5_2", regmodel=trend)
        theta, sigma2 = np.asarray(k.theta()).ravel(), float(k.sigma2())
        m_loo, s_loo = (np.asarray(a).ravel() for a in k.leaveOneOutVec(theta))
        m_ref, s_ref = np.zeros(len(y)), np.zeros(len(y))
        for i in range(len(y)):
            keep = np.arange(len(y)) != i
            ki = lk.Kriging(y[keep], X[keep], "matern5_2", regmodel=trend, optim="none",
                            parameters={"theta": theta.reshape(1, -1), "is_theta_estim": False,
                                        "sigma2": sigma2, "is_sigma2_estim": False})
            mi, si, *_ = ki.predict(X[i:i + 1], return_stdev=True)
            m_ref[i], s_ref[i] = np.ravel(mi)[0], np.ravel(si)[0]
        suite.close(f"{fname}/{trend}: leaveOneOutVec mean == explicit refits", m_loo, m_ref,
                    rtol=1e-6, atol=1e-8 * np.std(y))
        suite.close(f"{fname}/{trend}: leaveOneOutVec stdev == explicit refits", s_loo, s_ref,
                    rtol=1e-4, atol=1e-8 * np.std(y), severity="warn")
        # leaveOneOut() is the mean squared LOO error at the fitted theta
        k_loo = lk.Kriging(y, X, "matern5_2", regmodel=trend, objective="LOO")
        m2, _ = k_loo.leaveOneOutVec(np.asarray(k_loo.theta()).ravel())
        suite.close(f"{fname}/{trend}: leaveOneOut() == mean squared LOO error", k_loo.leaveOneOut(),
                    np.mean((np.ravel(m2) - y) ** 2), rtol=1e-6, severity="warn")

# %% [markdown]
# ## Multistart
#
# Gap between the default single start and 10 / 20 starts. A large gap means the
# default fit is unreliable on this kind of problem.

# %%
for fname, n in CASES.items():
    tf = TESTS[fname]
    X, y, _, _ = design.dataset(tf, n, seed=7)
    ll1 = lk.Kriging(y, X, "matern5_2", optim="BFGS").logLikelihood()
    ll10 = lk.Kriging(y, X, "matern5_2", optim="BFGS10").logLikelihood()
    ll20 = lk.Kriging(y, X, "matern5_2", optim="BFGS20").logLikelihood()
    suite.metric(fname, "BFGS", ll=ll1)
    suite.metric(fname, "BFGS10", ll=ll10)
    suite.metric(fname, "BFGS20", ll=ll20)
    suite.greater(f"{fname}: LL(BFGS10) - LL(BFGS)", ll10 - ll1, -1e-3 * abs(ll1), severity="warn")
    suite.less(f"{fname}: LL(BFGS20) - LL(BFGS) (default fit far from the best optimum)", ll20 - ll1, 1.0,
               severity="warn")
suite.table().loc[list(CASES)].unstack("package")

# %% [markdown]
# ## Summary
#
# The metrics of this notebook, the number of checks per status, and every check that
# did not pass (FAIL: error-level, fails the run; WARN: warn-level; KNOWN: listed in
# `tools/known_issues.csv`, with its issue; SKIP: feature absent from the installed version).
# All of them are also written to `results/objectives-LL-LOO-LMP-gradients_Kriging_python.checks.csv` and `.metrics.csv`.

# %%
suite.summary()

# %%
suite.finalize()
