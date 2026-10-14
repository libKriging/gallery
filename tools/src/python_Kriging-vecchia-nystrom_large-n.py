# %% [markdown]
# # Python · Kriging with `method="vecchia"` / `"nystrom"` — large n
#
# The exact fit costs O(n³). `Kriging` offers:
#
# * `method="vecchia(m)"`: Vecchia likelihood with m conditioning neighbours
#   (local, recommended in low dimension);
# * `method="nystrom(k)"`: low-rank Nyström approximation with k landmarks
#   (global);
# * `Kriging.subsetOfData`: a representative subset before an exact fit.
#
# (`NestedKriging`, the partition-based alternative, on the same problems:
# `python_NestedKriging_aggregations-large-n.ipynb`.)
#
# Problems: Hartmann 3 (3D, local structure: Vecchia's ground) and the OTL
# circuit (6D, smooth: Nyström's ground), n = 1500 (exact fit still affordable,
# for reference) and n = 5000 (approximations only).
#
# Checks:
# * **limit cases**: Vecchia with m = n − 1 neighbours *is* the exact likelihood;
# * the method is explicit at prediction time: `predict(method="exact")` on a
#   Nyström/Vecchia fit raises, `method=` gives finite predictions;
# * accuracy of each approximation close to the exact fit (warn-level gaps,
#   error-level floors) and faster than the exact fit at n = 1500 (warn-level).

# %%
import time

import numpy as np
import pandas as pd
import pylibkriging as lk

import lkbench
from lkbench import TESTS, design, metrics, compat

suite = lkbench.Suite("python_Kriging-vecchia-nystrom_large-n")
print("pylibkriging", compat.version())
HAS_METHOD = compat.has("method")
if not HAS_METHOD:
    suite.skip("method=vecchia/nystrom", "pylibkriging without the method= argument")


def pred(k, X, **kw):
    m, s, *_ = k.predict(X, return_stdev=True, **kw)
    return np.ravel(m), np.ravel(s)


def timed(f):
    t0 = time.time()
    out = f()
    return out, time.time() - t0

# %% [markdown]
# ## Limit case: Vecchia with all the previous points as neighbours is exact

# %%
if HAS_METHOD:
    tf = TESTS["branin"]
    X, y, Xt, _ = design.dataset(tf, 40, n_test=100, seed=51)
    ke = lk.Kriging(y, X, "matern5_2")
    fixed = {"theta": np.asarray(ke.theta()).reshape(1, -1), "is_theta_estim": False}
    kv = lk.Kriging(y, X, "matern5_2", optim="none", parameters=fixed, method=f"vecchia({len(y) - 1})")
    suite.close("vecchia(n-1) LL == exact LL (same theta)", kv.logLikelihood(method="vecchia"),
                ke.logLikelihood(), rtol=1e-6)
    suite.close("vecchia(n-1) predict == exact predict", pred(kv, Xt, method="vecchia")[0], pred(ke, Xt)[0],
                rtol=1e-5, atol=1e-6 * np.std(y), severity="warn")
    kn = lk.Kriging(y, X, "matern5_2", optim="none", parameters=fixed, method=f"nystrom({len(y)})")
    suite.close("nystrom(n) LL == exact LL (same theta)", kn.logLikelihood(method="nystrom"), ke.logLikelihood(),
                rtol=1e-3, severity="warn")
    # the method must be explicit after an approximate fit
    suite.raises("nystrom fit: predict(method='exact') raises", lambda: kn.predict(Xt, return_stdev=True,
                                                                                    method="exact"))
    suite.true("method() reports the fit method", kv.method().startswith("vecchia") and
               kn.method().startswith("nystrom"), detail=f"{kv.method()}, {kn.method()}")

# %% [markdown]
# ## Accuracy and cost

# %%
CASES = {"hartmann3": (1500, 5000), "otl_circuit": (1500, 5000)}
FLOORS = {"hartmann3": 0.98, "otl_circuit": 0.99}
for fname, (n_mid, n_big) in CASES.items():
    tf = TESTS[fname]
    X, y, Xt, yt = design.dataset(tf, n_big, n_test=2000, seed=52)
    Xm, ym = X[:n_mid], y[:n_mid]
    ke, t_exact = timed(lambda: lk.Kriging(ym, Xm, "matern5_2"))
    res_e = metrics.all_metrics(yt, *pred(ke, Xt))
    suite.metric(f"{fname}/n={n_mid}", "exact", fit_s=t_exact, **res_e)
    suite.greater(f"{fname}/n={n_mid}: Q2 exact", res_e["q2"], FLOORS[fname])
    variants = {}
    if HAS_METHOD:
        variants.update({
            "vecchia(30)": lambda Xa, ya: (lk.Kriging(ya, Xa, "matern5_2", method="vecchia(30)"), "vecchia"),
            "nystrom(100)": lambda Xa, ya: (lk.Kriging(ya, Xa, "matern5_2", method="nystrom(100)"), "nystrom"),
        })
    if compat.has("subsetOfData"):
        def sod(Xa, ya):
            idx = np.ravel(lk.Kriging.subsetOfData(Xa, n_max=500, method="kmeans", seed=1)).astype(int)
            return lk.Kriging(ya[idx], Xa[idx], "matern5_2"), None
        variants["subset of data (500)"] = sod
    for label, make in variants.items():
        for n in (n_mid, n_big):
            case = f"{fname}/n={n}"
            with suite.runs(f"{case}/{label}: fit + predict"):
                (k, meth), t = timed(lambda: make(X[:n], y[:n]))
                m, s = pred(k, Xt, method=meth) if meth else (
                    lambda o: (np.ravel(o[0]), np.ravel(o[1])))(k.predict(Xt, return_stdev=True))
                suite.finite(f"{case}/{label}: finite predictions", m)
                res = metrics.all_metrics(yt, m, s)
                suite.metric(case, label, fit_s=t, **res)
                suite.greater(f"{case}/{label}: Q2", res["q2"], FLOORS[fname] - 0.05)
                if n == n_mid:
                    suite.greater(f"{case}/{label}: Q2 - Q2(exact)", res["q2"] - res_e["q2"], -0.02,
                                  severity="warn")
                    suite.less(f"{case}/{label}: fit time / exact fit time", t / t_exact, 1.0, severity="warn")
suite.table().round(4)

# %%
suite.finalize()
