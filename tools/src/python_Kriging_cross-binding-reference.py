# %% [markdown]
# # Python · Kriging — cross-binding consistency: the Python reference
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

# %%
import csv

import numpy as np
import pylibkriging as lk

import lkbench
from lkbench import compat
from lkbench.checks import DATA_DIR as DATA, RESULTS_DIR

suite = lkbench.Suite("python_Kriging_cross-binding-reference")
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
    suite.close(f"{name}: logLikelihood() == logLikelihoodFun(theta0)", kf.logLikelihood(), ll, rtol=1e-10)
    k = lk.Kriging(y, X, "matern5_2")
    put(name, "fit_theta", k.theta())
    put(name, "fit_ll", k.logLikelihood())
    put(name, "fit_mean", k.predict(Xt, return_stdev=False)[0])
    put(name, "sim", np.ravel(np.asarray(kf.simulate(3, 123, Xt[:5])), order="F"))  # column-major, as R/Octave/Julia

RESULTS_DIR.mkdir(exist_ok=True)
with open(RESULTS_DIR / "xbind_python.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["lang", "dataset", "quantity", "index", "value"])
    w.writerows(rows)
print(len(rows), "values written")

# %%
suite.finalize()
