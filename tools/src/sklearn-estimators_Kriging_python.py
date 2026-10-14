# %% [markdown]
# # Python · Kriging — scikit-learn estimators (`pylibkriging.sklearn`)
#
# `KrigingRegressor` (the scikit-learn face of `Kriging`) on Hartmann 3:
# `fit`/`predict` (scikit-learn conventions), the same predictions as the native
# class, `get_params`/`set_params`/`clone`, use inside `Pipeline`,
# `GridSearchCV`, `cross_val_score`, and `sample_y`.

# %%
import numpy as np
import pylibkriging as lk

import lkbench
from lkbench import TESTS, design, compat

suite = lkbench.Suite("sklearn-estimators_Kriging_python")
print("pylibkriging", compat.version())

tf = TESTS["hartmann3"]
X, y, Xt, yt = design.dataset(tf, 40, n_test=100, seed=61)


def mean_sd(k, Xn):
    out = k.predict(Xn, return_stdev=True)
    return np.ravel(out[0]), np.ravel(out[1])

# %% [markdown]
# ## Installing scikit-learn
#
# With pip, into the kernel's environment, numpy pinned to its installed version
# (a package that would downgrade it is not installed: the libKriging extension
# is built against it). A package that cannot be installed is skipped, and
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
