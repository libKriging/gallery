# %% [markdown]
# # Python · Kriging — moderate and high dimension: engineering functions, `normalize`, screening
#
# **Idea.** From 6 to 20 inputs on engineering functions: accuracy, the effect of
# `normalize` on inputs in physical units, and the ability of the fitted ranges θ to
# tell influential inputs from inactive ones.
#
# **Reference.** The same model on inputs rescaled by hand, and the known active
# inputs of the Welch function.
#
# | function | d | character |
# |---|---|---|
# | OTL circuit | 6 | smooth, inputs on very different scales |
# | Piston | 7 | smooth, strongly non-linear in a few inputs |
# | Borehole | 8 | smooth, 2–3 dominant inputs, raw scales from 0.05 to 10⁵ |
# | Robot arm | 8 | oscillating (angles): hard for a stationary GP |
# | Dette & Pepelyshev | 8 | curved, logarithmic terms |
# | Wing weight | 10 | smooth, nearly additive |
# | Welch et al. (1992) | 20 | screening problem: few active inputs, interactions |
#
# Checks:
# * **`normalize=True` on raw inputs** gives the same accuracy as an explicit
#   rescaling to [0, 1] (they are the same model up to the optimizer);
#   inputs in native units (Borehole spans 0.05 to 50 000) must not break a fit;
# * Q² floors per function (regression detection);
# * the fitted θ of Welch's **inactive** inputs are larger than those of the
#   active ones (screening ability, warn-level).
#
# `MLPKriging` on a high-dimensional function with a low-dimensional structure:
# `deep-kernel-projected-branin_MLPKriging_python.ipynb`.

# %%
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylibkriging as lk

import lkgallery
from lkgallery import TESTS, design, metrics, compat

suite = lkgallery.Suite("high-dimension-normalize-screening_Kriging_python")
print("pylibkriging", compat.version())


def pred(k, X):
    m, s, *_ = k.predict(X, return_stdev=True)
    return np.ravel(m), np.ravel(s)

# %% [markdown]
# ## Accuracy, unit-cube inputs vs native inputs with `normalize=True`

# %%
CASES = {  # function: (n, Q2 floor on the unit cube)
    "otl_circuit": (60, 0.999),
    "piston": (70, 0.98),
    "borehole": (80, 0.995),
    "robot_arm": (160, 0.45),
    "dette_pepelyshev": (80, 0.99),
    "wing_weight": (100, 0.999),
    "welch": (200, 0.98),
}
for fname, (n, floor) in CASES.items():
    tf = TESTS[fname]
    U, y, Ut, yt = design.dataset(tf, n, n_test=2000, seed=41)
    t0 = time.time()
    k = lk.Kriging(y, U, "matern5_2")
    fit_s = time.time() - t0
    res = metrics.all_metrics(yt, *pred(k, Ut))
    suite.metric(fname, "unit cube", fit_s=fit_s, **res)
    suite.greater(f"{fname}: Q2 (unit cube, matern5_2)", res["q2"], floor)
    # same problem in native units, rescaled internally
    Xn, Xtn = tf.scale(U), tf.scale(Ut)
    with suite.runs(f"{fname}: fit on native inputs with normalize=True"):
        kn = lk.Kriging(y, Xn, "matern5_2", normalize=True)
        res_n = metrics.all_metrics(yt, *pred(kn, Xtn))
        suite.metric(fname, "native + normalize", **res_n)
        suite.greater(f"{fname}: Q2(normalize) - Q2(unit cube)", res_n["q2"] - res["q2"], -0.02)
tab = suite.table()
tab.round(4)

# %% [markdown]
# ## Screening: Welch et al. (1992)
#
# Inputs 8 and 16 do not appear in the function; inputs 1, 4, 5, 12, 19, 20 are
# the most influential ones. A small θ_j (short correlation length) means an
# influential input, so the inactive inputs should get the largest θ.

# %%
tf = TESTS["welch"]
U, y, _, _ = design.dataset(tf, 200, seed=42)
k = lk.Kriging(y, U, "matern5_2")
theta = np.ravel(k.theta())
inactive = [7, 15]            # 0-based indices of x8, x16
active = [0, 3, 4, 11, 18, 19]  # x1, x4, x5, x12, x19, x20
suite.greater("welch: min theta(inactive) / max theta(active)", theta[inactive].min() / theta[active].max(), 1.0,
              severity="warn")
colors = ["tab:red" if i in active else "tab:gray" if i in inactive else "tab:blue" for i in range(20)]
plt.figure(figsize=(9, 3))
plt.bar([f"x{i + 1}" for i in range(20)], theta, color=colors)
plt.yscale("log"); plt.ylabel("fitted θ (log)")
plt.title("Welch (20D): red = most influential inputs, gray = inactive (x8, x16)"); plt.show()
pd.Series(theta, index=[f"x{i + 1}" for i in range(20)]).round(3).to_frame("theta").T

# %% [markdown]
# ## Summary
#
# The metrics of this notebook, the number of checks per status, and every check that
# did not pass (FAIL: error-level, fails the run; WARN: warn-level; KNOWN: listed in
# `tools/known_issues.csv`, with its issue; SKIP: feature absent from the installed version).
# All of them are also written to `results/high-dimension-normalize-screening_Kriging_python.checks.csv` and `.metrics.csv`.

# %%
suite.summary()

# %%
suite.finalize()
