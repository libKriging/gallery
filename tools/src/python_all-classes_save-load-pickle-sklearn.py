# %% [markdown]
# # Python · all classes — save/load, pickle, copy, accessors, scikit-learn estimators
#
# Persistence and introspection, for every model class:
#
# * `save` → `load` (class-specific and generic `pylibkriging.load`) gives a model
#   with **identical** predictions and hyperparameters;
# * `pickle` / `copy` round trips, and a copy is independent of the original
#   (updating one leaves the other untouched);
# * accessors mirror the constructor arguments (`kernel()`, `regmodel()`,
#   `optim()`, `objective()`, `X()`, `y()`, ...);
# * scikit-learn estimators (`pylibkriging.sklearn`): `fit`/`predict`,
#   `get_params`/`set_params`/`clone`, use inside `Pipeline` and `GridSearchCV`,
#   same predictions as the native class.

# %%
import copy
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pylibkriging as lk

import lkbench
from lkbench import TESTS, design, compat

suite = lkbench.Suite("python_all-classes_save-load-pickle-sklearn")
print("pylibkriging", compat.version())
tmp = Path(tempfile.mkdtemp())

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
if compat.has("WarpKriging"):
    MODELS["WarpKriging"] = lambda: lk.WarpKriging(y, X, ["kumaraswamy", "none", "affine"], "matern5_2")
if compat.has("MLPKriging"):
    MODELS["MLPKriging"] = lambda: lk.MLPKriging(y, X, hidden_dims=[8], d_out=2, activation="tanh",
                                                 kernel="matern5_2")
if compat.has("NestedKriging"):
    MODELS["NestedKriging"] = lambda: lk.NestedKriging(y, X, "matern5_2", nb_groups=4, seed=1)


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
    with suite.runs(f"{name}: fit"):
        k = make()
    m0, s0 = predict_any(k, Xt)
    cls = type(k)
    path = tmp / f"{name.replace(' ', '_')}.json"
    if not hasattr(k, "save"):
        suite.skip(f"{name}: save + load", "no save() in this pylibkriging")
        continue
    with suite.runs(f"{name}: save + load"):
        k.save(str(path))
        k1 = cls.load(str(path))
        m1, s1 = predict_any(k1, Xt)
        suite.close(f"{name}: load(save(k)) predicts the same mean", m1, m0, rtol=1e-12, atol=1e-12)
        suite.close(f"{name}: load(save(k)) predicts the same stdev", s1, s0, rtol=1e-10, atol=1e-12)
        if hasattr(k, "theta"):
            suite.close(f"{name}: same theta after load", np.ravel(k1.theta()), np.ravel(k.theta()),
                        rtol=1e-14, atol=0)
    if hasattr(lk, "load") and (compat.has("load(noise)") or "noise" not in name and "nugget" not in name):
        with suite.runs(f"{name}: generic pylibkriging.load"):
            k2 = lk.load(str(path))
            suite.true(f"{name}: generic load returns {cls.__name__}", type(k2).__name__ == cls.__name__,
                       detail=type(k2).__name__)
            suite.close(f"{name}: generic load predicts the same", predict_any(k2, Xt)[0], m0, rtol=1e-12,
                        atol=1e-12)
    if not compat.has("pickle"):
        suite.skip(f"{name}: pickle / deepcopy", "models not picklable in this pylibkriging")
        continue
    with suite.runs(f"{name}: pickle round trip"):
        k3 = pickle.loads(pickle.dumps(k))
        suite.close(f"{name}: pickle predicts the same", predict_any(k3, Xt)[0], m0, rtol=1e-12, atol=1e-12)
    with suite.runs(f"{name}: deepcopy"):
        k4 = copy.deepcopy(k)
        suite.close(f"{name}: deepcopy predicts the same", predict_any(k4, Xt)[0], m0, rtol=1e-12, atol=1e-12)

# %% [markdown]
# A copy is independent: updating it leaves the original model unchanged.

# %%
k = lk.Kriging(y[:30], X[:30], "matern5_2")
m_before = mean_sd(k, Xt)[0]
kc = k.copy() if hasattr(k, "copy") else copy.deepcopy(k)
kc.update(y[30:], X[30:], refit=False)
suite.close("update on a copy leaves the original unchanged", mean_sd(k, Xt)[0], m_before, rtol=0, atol=0)
suite.true("the updated copy did change", not np.allclose(mean_sd(kc, Xt)[0], m_before))

# %% [markdown]
# ## Accessors mirror the constructor

# %%
k = lk.Kriging(y, X, "matern3_2", regmodel="linear", normalize=False, optim="BFGS", objective="LMP")
for acc, expected in [("kernel", "matern3_2"), ("regmodel", "linear"), ("optim", "BFGS"), ("objective", "LMP"),
                      ("normalize", False)]:
    suite.true(f"Kriging.{acc}() == {expected!r}", getattr(k, acc)() == expected, detail=str(getattr(k, acc)()))
suite.close("Kriging.X() is the design", np.asarray(k.X()), X, rtol=0, atol=0)
suite.close("Kriging.y() is the response", np.ravel(k.y()), y, rtol=0, atol=0)
# with normalize=True: do the accessors return the data as given, or as rescaled internally?
kn = lk.Kriging(y, X * 10, "matern3_2", normalize=True)
suite.true("Kriging.normalize() == True", kn.normalize() is True or kn.normalize() == 1)
suite.close("normalize=True: X() is the design as given", np.asarray(kn.X()), X * 10, rtol=1e-12, atol=0,
            severity="warn")
suite.close("normalize=True: y() is the response as given", np.ravel(kn.y()), y, rtol=1e-12, atol=0,
            severity="warn")
suite.true("Kriging.beta() has 1 + d coefficients (linear trend)", np.ravel(k.beta()).size == 1 + X.shape[1])
suite.true("summary() is a non-empty string", isinstance(k.summary(), str) and len(k.summary()) > 20)
suite.true("is_theta_estim() after an optimized fit", bool(k.is_theta_estim()))

# %% [markdown]
# ## Installing scikit-learn
#
# Needed by the estimators of `pylibkriging.sklearn` below. With pip, into the
# kernel's environment, numpy pinned to its installed version (a package that
# would downgrade it is not installed: the libKriging extension is built
# against it). A package that cannot be installed is skipped, and
# reported.

# %%
import importlib
import subprocess
import sys

import numpy

for package in ["scikit-learn"]:
    r = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", package, f"numpy=={numpy.__version__}"],
                       capture_output=True, text=True)
    print(f"{package}: " + ("installed" if r.returncode == 0 else "NOT installed\n" + r.stderr.strip()[-800:]))
importlib.invalidate_caches()

# %% [markdown]
# ## scikit-learn estimators

# %%
if compat.has("sklearn"):
    from sklearn.base import clone
    from sklearn.model_selection import GridSearchCV, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import MinMaxScaler
    from pylibkriging.sklearn import KrigingRegressor

    est = KrigingRegressor(kernel="matern5_2").fit(X, y)
    m_sk, s_sk = est.predict(Xt, return_std=True)
    suite.true("KrigingRegressor.predict returns 1-D arrays (scikit-learn convention)",
               np.ndim(m_sk) == 1 and np.ndim(s_sk) == 1, detail=f"shapes {np.shape(m_sk)}, {np.shape(s_sk)}",
               severity="warn")
    m_sk, s_sk = np.ravel(m_sk), np.ravel(s_sk)
    m_nat, s_nat = mean_sd(lk.Kriging(y, X, "matern5_2"), Xt)
    suite.close("KrigingRegressor == Kriging (mean)", m_sk, m_nat, rtol=1e-8, atol=1e-10)
    suite.close("KrigingRegressor == Kriging (stdev)", s_sk, s_nat, rtol=1e-6, atol=1e-10)
    est2 = clone(est).set_params(kernel="gauss")
    suite.true("clone + set_params", est2.get_params()["kernel"] == "gauss" and est.get_params()["kernel"] ==
               "matern5_2")
    with suite.runs("Pipeline(MinMaxScaler, KrigingRegressor)"):
        pipe = make_pipeline(MinMaxScaler(), KrigingRegressor(kernel="matern5_2")).fit(tf.scale(X), y)
        suite.greater("pipeline Q2 on native inputs", pipe.score(tf.scale(Xt), yt), 0.8)
    with suite.runs("GridSearchCV over kernels"):
        gs = GridSearchCV(KrigingRegressor(), {"kernel": ["gauss", "matern3_2", "matern5_2"]}, cv=3).fit(X, y)
        suite.true("GridSearchCV picked a kernel", gs.best_params_["kernel"] in ("gauss", "matern3_2", "matern5_2"),
                   detail=str(gs.best_params_))
    with suite.runs("cross_val_score"):
        sc = cross_val_score(KrigingRegressor(kernel="matern5_2"), X, y, cv=4)
        suite.finite("cross_val_score finite", sc)
    with suite.runs("sample_y"):
        S = est.sample_y(Xt[:10], n_samples=5, random_state=1)
        suite.true("sample_y shape (q, n_samples)", np.shape(S) == (10, 5), detail=str(np.shape(S)))
else:
    suite.skip("sklearn estimators", "pylibkriging.sklearn or scikit-learn not available")

# %%
suite.finalize()
