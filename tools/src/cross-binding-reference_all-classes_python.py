# %% [markdown]
# # Python · all classes — cross-binding consistency: the Python reference
#
# **Idea.** All bindings wrap the same C++ library: on the same data with the same
# options, they must give the same numbers, for every model class and fit method. This
# notebook writes the Python values, the reference of the comparison.
#
# **Reference.** The R, Octave and Julia notebooks `cross-binding_all-classes_*`,
# compared by `tools/report.py`.
#
# All bindings (Python, R, Octave, Julia) wrap the same C++ library: on the
# same data with the same options they must return the same numbers. Each
# language runs the protocol below on the shared datasets of `data/` and writes
# `results/xbind_<lang>.csv` (`lang,dataset,quantity,index,value`);
# `tools/report.py` then compares every language to Python.
#
# Protocol, for each dataset (Branin n=25, Hartmann 3 n=40, Borehole n=80, on
# the unit cube), kernel `matern5_2`, trend `constant`, θ₀ = 0.3 in every
# dimension:
#
# | quantity | how | tolerance in the report |
# |---|---|---|
# | `fixed_sigma2`, `fixed_beta`, `fixed_ll`, `fixed_loo` | `optim="none"`, θ = θ₀ fixed | rel. 1e-8 (error) |
# | `fixed_mean`, `fixed_stdev` | predict at the first 50 test points | rel. 1e-8 (error) |
# | `llfun`, `llfun_grad` | `logLikelihoodFun(θ₀)` with gradient | rel. 1e-8 (error) |
# | `fit_theta`, `fit_ll`, `fit_mean` | default fit (`BFGS`) | rel. 1e-4 (warn: optimizer paths may differ) |
# | `sim` | `simulate(nsim=3, seed=123)` at the first 5 test points (5 × 3, column-major) | rel. 1e-8 (warn: RNG may be binding-specific) |
#
# Then, on Branin only, every other model class and fit method, so that each
# binding is checked on all of them (the class-specific notebooks are in Python
# only): with fixed hyperparameters the numbers must be identical (error); a
# fit is compared at warn level, as above.
#
# | quantity | how | tolerance in the report |
# |---|---|---|
# | `nugget_fixed_mean`, `nugget_fixed_stdev` | `noise="nugget"`, θ = θ₀, σ² = var(y), nugget = 0.01 var(y), all fixed | rel. 1e-8 (error) |
# | `noise_fixed_mean`, `noise_fixed_stdev` | known noise variances 0.01 var(y), θ = θ₀ and σ² fixed | rel. 1e-8 (error) |
# | `nugget_fit_ll`, `nugget_fit_mean` | `noise="nugget"`, default fit | rel. 1e-4 (warn) |
# | `vecchia_fixed_ll`, `vecchia_fixed_mean`, `nystrom_fixed_ll`, `nystrom_fixed_mean` | `method="vecchia(10)"` / `"nystrom(10)"`, θ = θ₀ fixed; likelihood and prediction with that method | rel. 1e-8 (error) |
# | `warp_fit_mean` | `WarpKriging`, `kumaraswamy` on both inputs, default fit | rel. 1e-4 (warn) |
# | `mlp_fit_mean` | `MLPKriging`, hidden layer of 8 (`tanh`), 2 features, `BFGS10` | rel. 1e-4 (warn) |
# | `nested_fit_mean` | `NestedKriging`, 2 groups (`kmeans`, seed 1), `NK` | rel. 1e-4 (warn) |

# %%
import csv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylibkriging as lk

import gallery
from gallery import compat
from gallery import DATA_DIR as DATA, RESULTS_DIR

suite = gallery.Suite("cross-binding-reference_all-classes_python")
print("pylibkriging", compat.version())
DATASETS = ["branin", "hartmann3", "borehole"]
rows = []


def put(dataset, quantity, values):
    for i, v in enumerate(np.ravel(values), start=1):
        rows.append(("python", dataset, quantity, i, f"{float(v):.17g}"))


def load(name, stem):
    return np.loadtxt(DATA / f"{name}_{stem}.csv", delimiter=",", ndmin=2)

# %%
for name in DATASETS:
    X, y, Xt = load(name, "X"), load(name, "y").ravel(), load(name, "Xt")[:50]
    d = X.shape[1]
    theta0 = np.full((1, d), 0.3)
    kf = lk.Kriging(y, X, "matern5_2", regmodel="constant", optim="none",
                    parameters={"theta": theta0, "is_theta_estim": False})
    put(name, "fixed_sigma2", kf.sigma2())
    put(name, "fixed_beta", kf.beta())
    put(name, "fixed_ll", kf.logLikelihood())
    put(name, "fixed_loo", kf.leaveOneOut())
    m, s, *_ = kf.predict(Xt, return_stdev=True)
    put(name, "fixed_mean", m)
    put(name, "fixed_stdev", s)
    ll, grad, _ = kf.logLikelihoodFun(theta0.ravel(), True)
    put(name, "llfun", ll)
    put(name, "llfun_grad", grad)
    suite.check_close(f"{name}: logLikelihood() == logLikelihoodFun(theta0)", kf.logLikelihood(), ll, rtol=1e-10)
    k = lk.Kriging(y, X, "matern5_2")
    put(name, "fit_theta", k.theta())
    put(name, "fit_ll", k.logLikelihood())
    put(name, "fit_mean", k.predict(Xt, return_stdev=False)[0])
    put(name, "sim", np.ravel(np.asarray(kf.simulate(3, 123, Xt[:5])), order="F"))  # column-major, as R/Octave/Julia

# %% [markdown]
# ## Other classes and fit methods (Branin)

# %%
name = "branin"
X, y, Xt = load(name, "X"), load(name, "y").ravel(), load(name, "Xt")[:50]
theta0 = np.full((1, 2), 0.3)
s2 = float(np.var(y)); nug = 0.01 * s2
fixed = {"theta": theta0, "is_theta_estim": False, "sigma2": s2, "is_sigma2_estim": False}
kn = lk.Kriging(y, X, "matern5_2", noise="nugget", optim="none",
                parameters={**fixed, "nugget": nug, "is_nugget_estim": False})
m, s, *_ = kn.predict(Xt, return_stdev=True)
put(name, "nugget_fixed_mean", m); put(name, "nugget_fixed_stdev", s)
kh = lk.Kriging(y, X, "matern5_2", noise=np.full(len(y), nug), optim="none", parameters=fixed)
m, s, *_ = kh.predict(Xt, return_stdev=True)
put(name, "noise_fixed_mean", m); put(name, "noise_fixed_stdev", s)
kf = lk.Kriging(y, X, "matern5_2", noise="nugget")
put(name, "nugget_fit_ll", kf.logLikelihood()); put(name, "nugget_fit_mean", kf.predict(Xt, return_stdev=False)[0])
for meth in ["vecchia", "nystrom"] if compat.has("method") else []:
    ka = lk.Kriging(y, X, "matern5_2", optim="none", parameters={"theta": theta0, "is_theta_estim": False},
                    method=f"{meth}(10)")
    put(name, f"{meth}_fixed_ll", ka.logLikelihood(method=meth))
    put(name, f"{meth}_fixed_mean", ka.predict(Xt, return_stdev=False, method=meth)[0])
if not compat.has("method"):
    suite.check_skip("vecchia / nystrom quantities", "pylibkriging without the method= argument")
kw = lk.WarpKriging(y, X, ["kumaraswamy", "kumaraswamy"], "matern5_2")
put(name, "warp_fit_mean", kw.predict(Xt, return_stdev=False)[0])
km = lk.MLPKriging(y, X, hidden_dims=[8], d_out=2, activation="tanh", kernel="matern5_2", optim="BFGS10")
put(name, "mlp_fit_mean", km.predict(Xt, return_stdev=False)[0])
kk = lk.NestedKriging(y, X, "matern5_2", nb_groups=2, aggregation="NK", seed=1)
put(name, "nested_fit_mean", kk.predict(Xt, return_stdev=False)[0])

# %%
RESULTS_DIR.mkdir(exist_ok=True)
with open(RESULTS_DIR / "xbind_python.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["lang", "dataset", "quantity", "index", "value"])
    w.writerows(rows)
print(len(rows), "values written")

# %% [markdown]
# ## The reference values
#
# Number of values per quantity and dataset (the report compares each of them
# with the other bindings), and the default fits they come from: predicted vs
# true values on the 50 test points.

# %%
ref = pd.DataFrame(rows, columns=["lang", "dataset", "quantity", "index", "value"])
ref["value"] = ref["value"].astype(float)
display(ref.groupby(["quantity", "dataset"])["value"].agg(["size", "first"]).unstack("dataset"))
fig, axes = plt.subplots(1, len(DATASETS), figsize=(12, 3.6))
for ax, name in zip(axes, DATASETS):
    yt = load(name, "yt").ravel()[:50]
    m = ref[(ref.dataset == name) & (ref.quantity == "fit_mean")]["value"].to_numpy()
    ax.plot(yt, m, "o", ms=4); ax.plot([yt.min(), yt.max()], [yt.min(), yt.max()], "k--", lw=.8)
    q2 = 1 - np.sum((yt - m) ** 2) / np.sum((yt - yt.mean()) ** 2)
    suite.metric(name, "pylibkriging default fit", q2=q2)
    ax.set_title(f"{name}: Q² = {q2:.3f}"); ax.set_xlabel("true"); ax.set_ylabel("predicted")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## Summary
#
# The metrics of this notebook, the number of checks per status, and every check that
# did not pass (FAIL: error-level, fails the run; WARN: warn-level; KNOWN: listed in
# `tools/known_issues.csv`, with its issue; SKIP: feature absent from the installed version).
# All of them are also written to `results/cross-binding-reference_all-classes_python.checks.csv` and `.metrics.csv`.

# %%
suite.summary()

# %%
suite.finalize()
