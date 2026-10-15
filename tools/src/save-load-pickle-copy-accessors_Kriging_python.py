# %% [markdown]
# # Python · Kriging — save/load, pickle, copy, accessors
#
# **Idea.** A model must survive a round trip (file, pickle, copy) unchanged, and
# expose its settings and hyperparameters: for every noise model, objective and fit
# method of `Kriging`.
#
# **Reference.** The original model: predictions and hyperparameters must be identical
# to the last bit (1e-12).
#
# Persistence and introspection of `Kriging`, for each of its variants (noise
# models, objectives, approximate fit methods):
#
# * `save` → `load` (class-specific and generic `pylibkriging.load`) gives a model
#   with **identical** predictions and hyperparameters;
# * `pickle` / `copy` round trips, and a copy is independent of the original
#   (updating one leaves the other untouched);
# * accessors mirror the constructor arguments (`kernel()`, `regmodel()`,
#   `optim()`, `objective()`, `X()`, `y()`, ...).
#
# The same persistence checks for the other classes are in their own notebooks
# (`*_WarpKriging_python`, `*_MLPKriging_python`, `*_NestedKriging_python`); the
# scikit-learn estimators in `sklearn-estimators_Kriging_python.ipynb`.
#
# An API notebook: its results are identities (to 1e-12), so they are given as
# the table of checks of the Summary rather than as plots.

# %%
import copy

import numpy as np
import pylibkriging as lk

import gallery
from gallery import TESTS, design, compat

suite = gallery.Suite("save-load-pickle-copy-accessors_Kriging_python")
print("pylibkriging", compat.version())

tf = TESTS["hartmann3"]
X, y, Xt, yt = design.dataset(tf, 40, n_test=100, seed=61)


def mean_sd(k, Xn):
    out = k.predict(Xn, return_stdev=True)
    return np.ravel(out[0]), np.ravel(out[1])


MODELS = {
    "Kriging": lambda: lk.Kriging(y, X, "matern5_2", regmodel="linear"),
    "Kriging nugget": lambda: lk.Kriging(y, X, "matern3_2", noise="nugget"),
    "Kriging noise vector": lambda: lk.Kriging(y, X, "gauss", noise=np.full(len(y), 1e-4)),
    "Kriging LOO": lambda: lk.Kriging(y, X, "matern5_2", objective="LOO"),
    "Kriging LMP": lambda: lk.Kriging(y, X, "matern5_2", objective="LMP"),
}
if compat.has("method"):
    MODELS["Kriging vecchia"] = lambda: lk.Kriging(y, X, "matern5_2", method="vecchia(10)")
    MODELS["Kriging nystrom"] = lambda: lk.Kriging(y, X, "matern5_2", method="nystrom(20)")


def predict_any(k, Xn):
    meth = k.method() if hasattr(k, "method") and isinstance(k, lk.Kriging) else None
    if meth and meth != "exact":
        out = k.predict(Xn, return_stdev=True, method=meth.split("(")[0])
        return np.ravel(out[0]), np.ravel(out[1])
    return mean_sd(k, Xn)

# %% [markdown]
# ## save / load, pickle, copy

# %%
for name, make in MODELS.items():
    with suite.check_runs(f"{name}: fit"):
        gallery.persistence.check(suite, name, make(), Xt, predict_any)

# %% [markdown]
# A copy is independent: updating it leaves the original model unchanged.

# %%
k = lk.Kriging(y[:30], X[:30], "matern5_2")
m_before = mean_sd(k, Xt)[0]
kc = k.copy() if hasattr(k, "copy") else copy.deepcopy(k)
kc.update(y[30:], X[30:], refit=False)
suite.check_close("update on a copy leaves the original unchanged", mean_sd(k, Xt)[0], m_before, rtol=0, atol=0)
suite.check_true("the updated copy did change", not np.allclose(mean_sd(kc, Xt)[0], m_before))

# %% [markdown]
# ## Accessors mirror the constructor

# %%
k = lk.Kriging(y, X, "matern3_2", regmodel="linear", normalize=False, optim="BFGS", objective="LMP")
for acc, expected in [("kernel", "matern3_2"), ("regmodel", "linear"), ("optim", "BFGS"), ("objective", "LMP"),
                      ("normalize", False)]:
    suite.check_true(f"Kriging.{acc}() == {expected!r}", getattr(k, acc)() == expected, detail=str(getattr(k, acc)()))
suite.check_close("Kriging.X() is the design", np.asarray(k.X()), X, rtol=0, atol=0)
suite.check_close("Kriging.y() is the response", np.ravel(k.y()), y, rtol=0, atol=0)
# with normalize=True: do the accessors return the data as given, or as rescaled internally?
kn = lk.Kriging(y, X * 10, "matern3_2", normalize=True)
suite.check_true("Kriging.normalize() == True", kn.normalize() is True or kn.normalize() == 1)
suite.check_close("normalize=True: X() is the design as given", np.asarray(kn.X()), X * 10, rtol=1e-12, atol=0,
            severity="warn")
suite.check_close("normalize=True: y() is the response as given", np.ravel(kn.y()), y, rtol=1e-12, atol=0,
            severity="warn")
suite.check_true("Kriging.beta() has 1 + d coefficients (linear trend)", np.ravel(k.beta()).size == 1 + X.shape[1])
suite.check_true("summary() is a non-empty string", isinstance(k.summary(), str) and len(k.summary()) > 20)
suite.check_true("is_theta_estim() after an optimized fit", bool(k.is_theta_estim()))

# %% [markdown]
# ## Summary
#
# The metrics of this notebook, the number of checks per status, and every check that
# did not pass (FAIL: error-level, fails the run; WARN: warn-level; KNOWN: listed in
# `tools/known_issues.csv`, with its issue; SKIP: feature absent from the installed version).
# All of them are also written to `results/save-load-pickle-copy-accessors_Kriging_python.checks.csv` and `.metrics.csv`.

# %%
suite.summary()

# %%
suite.finalize()
