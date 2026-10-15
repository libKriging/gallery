# %% [markdown]
# # Python · NestedKriging — aggregations, large n
#
# **Idea.** NestedKriging splits a large design into groups and aggregates their
# predictions; how far is it from the exact predictor as the number of groups grows,
# and which aggregation is closest?
#
# **Reference.** The exact predictor at the same θ (limit: 1 group), and the exact
# fit.
#
# `NestedKriging` partitions the design into groups, fits one `Kriging`
# sub-model per group (common hyperparameters), and aggregates the sub-model
# predictions: `NK` (nested kriging: kriging of the sub-predictors, a proper
# kriging predictor), or the products of experts `PoE`, `gPoE`, `BCM`, `rBCM`
# (cheaper, variances not fully consistent).
#
# Problems: as in `large-n_Kriging-vecchia-nystrom_python.ipynb`, Hartmann 3
# and the OTL circuit with n = 1500 (exact fit as reference) and n = 5000; then
# every aggregation on Branin.
#
# Checks:
# * **limit and asymptotic behaviour** at fixed θ (Hartmann 3, n = 300): with one
#   group, `NK` *is* the exact kriging predictor; the gap to the exact predictor
#   as the number of groups grows, against the fit and prediction cost;
# * accuracy close to the exact fit (warn-level gap, error-level floor), and
#   faster than the exact fit at n = 1500 (warn-level);
# * `NK` interpolates the data (it is a proper kriging predictor); the PoE family
#   does not need to; all stdev ≥ 0;
# * `NK` with a non-constant trend is rejected (simple-kriging theory).

# %%
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylibkriging as lk

import lkbench
from lkbench import TESTS, design, metrics, compat

suite = lkbench.Suite("aggregations-large-n_NestedKriging_python")
print("pylibkriging", compat.version())
if not compat.has("NestedKriging"):
    suite.skip("NestedKriging", "not available in this pylibkriging")


def pred(k, X, **kw):
    m, s, *_ = k.predict(X, return_stdev=True, **kw)
    return np.ravel(m), np.ravel(s)


def timed(f):
    t0 = time.time()
    out = f()
    return out, time.time() - t0

# %% [markdown]
# ## From one group (the exact predictor) to many, at fixed θ
#
# Hartmann 3, n = 300, θ of the exact fit kept fixed in every sub-model: the gap
# is that of the aggregation alone. RMS gap of the predicted mean to the exact
# predictor on 500 test points, relative to its standard deviation, and fit time.

# %%
if compat.has("NestedKriging"):
    tf = TESTS["hartmann3"]
    X, y, Xt, yt = design.dataset(tf, 300, n_test=500, seed=54)
    theta = np.asarray(lk.Kriging(y, X, "matern5_2").theta()).reshape(1, -1)
    fixed = {"theta": theta, "is_theta_estim": False}
    m_e = pred(lk.Kriging(y, X, "matern5_2", optim="none", parameters=fixed), Xt)[0]
    rows = []
    for agg in ["NK", "rBCM"]:
        for g in [1, 2, 4, 8, 16, 32]:
            with suite.runs(f"nested {agg}, {g} groups, fixed theta: fit + predict"):
                k, t = timed(lambda: lk.NestedKriging(y, X, "matern5_2", nb_groups=g, aggregation=agg, seed=1,
                                                      optim="none", parameters=fixed))
                m = pred(k, Xt)[0]
                rows.append(dict(aggregation=agg, groups=g, time_s=t,
                                 mean_gap=np.sqrt(np.mean((m - m_e) ** 2)) / np.std(m_e),
                                 q2=metrics.q2(yt, m)))
    conv = pd.DataFrame(rows)
    nk = conv[conv["aggregation"] == "NK"]
    suite.less("nested NK, 1 group == exact predictor (RMS gap / sd)", nk["mean_gap"].iloc[0], 1e-6)
    suite.less("nested NK, 32 groups: RMS gap to the exact predictor / sd", nk["mean_gap"].iloc[-1], 0.1,
               severity="warn")
    rb = conv[conv["aggregation"] == "rBCM"]
    suite.true("nested NK closer to the exact predictor than rBCM (2 to 32 groups)",
               bool(np.all(nk["mean_gap"].to_numpy()[1:] <= rb["mean_gap"].to_numpy()[1:])), severity="warn")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for agg, g in conv.groupby("aggregation"):
        axes[0].semilogy(g["groups"], np.maximum(g["mean_gap"], 1e-12), "o-", label=agg)
        axes[1].plot(g["groups"], g["time_s"], "o-", label=agg)
    axes[0].set_ylabel("RMS(mean − mean(exact)) / sd"); axes[1].set_ylabel("fit time (s)")
    for ax in axes:
        ax.set_xscale("log", base=2); ax.set_xlabel("number of groups"); ax.grid(True, which="both", alpha=.3)
    axes[0].legend(); fig.suptitle("Hartmann 3, n = 300, θ fixed: NestedKriging vs the exact predictor"); plt.show()
    display(conv.round(5))

# %% [markdown]
# ## Accuracy and cost against the exact fit

# %%
CASES = {"hartmann3": (1500, 5000), "otl_circuit": (1500, 5000)}
FLOORS = {"hartmann3": 0.98, "otl_circuit": 0.99}
if compat.has("NestedKriging"):
    for fname, (n_mid, n_big) in CASES.items():
        tf = TESTS[fname]
        X, y, Xt, yt = design.dataset(tf, n_big, n_test=2000, seed=52)
        ke, t_exact = timed(lambda: lk.Kriging(y[:n_mid], X[:n_mid], "matern5_2"))
        res_e = metrics.all_metrics(yt, *pred(ke, Xt))
        suite.metric(f"{fname}/n={n_mid}", "exact", fit_s=t_exact, **res_e)
        for agg in ["NK", "rBCM"]:
            label = f"nested {agg} (20 groups)"
            for n in (n_mid, n_big):
                case = f"{fname}/n={n}"
                with suite.runs(f"{case}/{label}: fit + predict"):
                    k, t = timed(lambda: lk.NestedKriging(y[:n], X[:n], "matern5_2", nb_groups=20, aggregation=agg,
                                                          seed=1))
                    m, s = pred(k, Xt)
                    suite.finite(f"{case}/{label}: finite predictions", m)
                    res = metrics.all_metrics(yt, m, s)
                    suite.metric(case, label, fit_s=t, **res)
                    suite.greater(f"{case}/{label}: Q2", res["q2"], FLOORS[fname] - 0.05)
                    if n == n_mid:
                        suite.greater(f"{case}/{label}: Q2 - Q2(exact)", res["q2"] - res_e["q2"], -0.02,
                                      severity="warn")
                        suite.less(f"{case}/{label}: fit time / exact fit time", t / t_exact, 1.0, severity="warn")
    display(suite.table().round(4))

# %% [markdown]
# ## `NestedKriging` aggregations: interpolation and variance

# %%
if compat.has("NestedKriging"):
    tf = TESTS["branin"]
    X, y, Xt, yt = design.dataset(tf, 200, n_test=500, seed=53)
    for agg in ["NK", "PoE", "gPoE", "BCM", "rBCM"]:
        with suite.runs(f"nested {agg}: fit"):
            k = lk.NestedKriging(y, X, "matern5_2", nb_groups=8, aggregation=agg, seed=2)
            m, s = (np.ravel(a) for a in k.predict(Xt, return_stdev=True)[:2])
            suite.true(f"nested {agg}: stdev >= 0", np.all(s >= 0))
            suite.metric("branin/nested", agg, **metrics.all_metrics(yt, m, s))
            if agg == "NK":
                mX = np.ravel(k.predict(X, return_stdev=True)[0])
                suite.less("nested NK: interpolates (max |m(X)-y| / sd(y))", np.max(np.abs(mX - y)) / np.std(y),
                           1e-4)
    display(suite.table().loc["branin/nested"].round(4))
    suite.raises("nested NK with a linear trend is rejected",
                 lambda: lk.NestedKriging(y, X, "matern5_2", nb_groups=4, aggregation="NK", regmodel="linear"),
                 severity="warn")

# %% [markdown]
# ## save / load, pickle, copy
#
# On Hartmann 3 (n = 40): a model saved and loaded back (class-specific and
# generic `pylibkriging.load`), pickled, or deep-copied predicts exactly as the
# original.

# %%
if compat.has("NestedKriging"):
    X, y, Xt, yt = design.dataset(TESTS["hartmann3"], 40, n_test=100, seed=61)
    with suite.runs("NestedKriging: fit (persistence)"):
        lkbench.persistence.check(suite, "NestedKriging", lk.NestedKriging(y, X, "matern5_2", nb_groups=4, seed=1), Xt, pred)

# %% [markdown]
# ## Summary
#
# The metrics of this notebook, the number of checks per status, and every check that
# did not pass (FAIL: error-level, fails the run; WARN: warn-level; KNOWN: listed in
# `tools/known_issues.csv`, with its issue; SKIP: feature absent from the installed version).
# All of them are also written to `results/aggregations-large-n_NestedKriging_python.checks.csv` and `.metrics.csv`.

# %%
suite.summary()

# %%
suite.finalize()
