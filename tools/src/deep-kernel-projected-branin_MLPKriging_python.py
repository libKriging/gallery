# %% [markdown]
# # Python · MLPKriging — deep kernel on a 10D function with a 2D structure
#
# **Idea.** A deep kernel (an MLP feature map before the kernel) can find a low-
# dimensional structure that a per-axis kernel cannot: Branin of a 2D projection of 10
# inputs.
#
# **Reference.** Plain `Kriging` on the same data.
#
# f(x) = Branin(Wx) for x ∈ [0,1]¹⁰ and a fixed 2 × 10 projection W: the
# function only varies in a 2D subspace, which a joint feature map (an MLP from
# ℝ¹⁰ to ℝ², then a stationary kernel) can find, whereas a per-axis kernel
# cannot.
#
# Checks: finite predictions; with a multistart optimizer, a Q² floor
# (error-level); the single-start fits compared with plain `Kriging`
# (warn-level), for two activations (`tanh`, smooth; `selu`, the default, with
# a kink at 0).

# %%
import time

import matplotlib.pyplot as plt
import numpy as np
import pylibkriging as lk

import lkgallery
from lkgallery import TESTS, design, metrics, compat

suite = lkgallery.Suite("deep-kernel-projected-branin_MLPKriging_python")
print("pylibkriging", compat.version())


def pred(k, X):
    m, s, *_ = k.predict(X, return_stdev=True)
    return np.ravel(m), np.ravel(s)

# %%
if compat.has("MLPKriging"):
    rng = np.random.default_rng(43)
    W = rng.normal(size=(2, 10))
    W /= np.abs(W).sum(axis=1, keepdims=True)
    tfb = TESTS["branin"]

    def projected(X):
        Z = X @ W.T
        Z = (Z - (W.clip(max=0).sum(axis=1))) / np.abs(W).sum(axis=1)  # back to [0,1]^2
        return tfb.unit(Z)

    X = design.lhs(120, 10, seed=44); Xt = design.uniform(2000, 10, seed=45)
    y, yt = projected(X), projected(Xt)
    kp = lk.Kriging(y, X, "matern5_2")
    res_k = metrics.all_metrics(yt, *pred(kp, Xt))
    suite.metric("projected_branin_10d", "Kriging", **res_k)
    for act in ["tanh", "selu"]:
        for optim in ["BFGS", "BFGS10"]:
            case = f"MLPKriging({act}, {optim})"
            with suite.runs(f"{case}: fit"):
                t0 = time.time()
                km = lk.MLPKriging(y, X, hidden_dims=[16, 8], d_out=2, activation=act, kernel="matern5_2",
                                   optim=optim)
                fit_s = time.time() - t0
                m, s = pred(km, Xt)
                suite.finite(f"{case}: predictions finite", m)
                res = metrics.all_metrics(yt, m, s)
                suite.metric("projected_branin_10d", f"MLPKriging {act} {optim}", fit_s=fit_s,
                             ll=float(km.logLikelihood()), **res)
                if optim == "BFGS10":
                    suite.greater(f"{case}: Q2", res["q2"], 0.8)
                else:
                    suite.greater(f"{case}: Q2 - Q2(Kriging)", res["q2"] - res_k["q2"], -0.1, severity="warn")
    tab = suite.table().loc["projected_branin_10d"]
    tab["q2"].plot.barh(figsize=(6, 3)); plt.xlabel("Q² (test set)"); plt.xlim(min(0, tab["q2"].min()), 1)
    plt.title("Branin(Wx), x in [0,1]^10"); plt.show()
    display(tab.round(4))
else:
    suite.skip("MLPKriging", "not available in this pylibkriging")

# %% [markdown]
# ## save / load, pickle, copy
#
# On Hartmann 3 (n = 40): a model saved and loaded back (class-specific and
# generic `pylibkriging.load`), pickled, or deep-copied predicts exactly as the
# original.

# %%
if compat.has("MLPKriging"):
    X, y, Xt, yt = design.dataset(TESTS["hartmann3"], 40, n_test=100, seed=61)
    with suite.runs("MLPKriging: fit (persistence)"):
        lkgallery.persistence.check(suite, "MLPKriging", lk.MLPKriging(y, X, hidden_dims=[8], d_out=2, activation="tanh",
                                                       kernel="matern5_2"), Xt, pred)

# %% [markdown]
# ## Summary
#
# The metrics of this notebook, the number of checks per status, and every check that
# did not pass (FAIL: error-level, fails the run; WARN: warn-level; KNOWN: listed in
# `tools/known_issues.csv`, with its issue; SKIP: feature absent from the installed version).
# All of them are also written to `results/deep-kernel-projected-branin_MLPKriging_python.checks.csv` and `.metrics.csv`.

# %%
suite.summary()

# %%
suite.finalize()
