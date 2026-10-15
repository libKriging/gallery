# %% [markdown]
# # Python · Kriging — robustness: degenerate and adversarial inputs
#
# **Idea.** What happens off the happy path: invalid, degenerate or ill-conditioned
# inputs must be rejected or handled, results must not depend on the units, and the
# prediction cost must grow linearly with the number of points.
#
# **Reference.** Equivariance identities (rescaled inputs / outputs give rescaled
# results) and the O(q) cost of `predict`.
#
# What a user eventually feeds a surrogate model, and what must happen:
#
# | input | expected behaviour |
# |---|---|
# | NaN / Inf in X or y | rejected with an exception (never a silent NaN model) |
# | X and y with inconsistent sizes | rejected |
# | a wrong number of warping specs | rejected |
# | duplicated design points, noise-free | either rejected or a finite model (no NaN, no crash) |
# | duplicated design points with `noise="nugget"` | a finite model |
# | constant response y | a finite model predicting the constant |
# | very few points (n = 2, 3) | a finite model |
# | very clustered points (1e-8 apart) with the `gauss` kernel | finite predictions (ill-conditioning) |
# | inputs scaled by 1e6 or 1e-6 with `normalize=True` | the same predictions as on [0,1] |
# | y scaled by 1e8 / shifted by 1e6 | predictions scaled / shifted accordingly (equivariance) |
# | predicting far outside the design | the mean reverts to the trend, the stdev to (slightly above) √σ² |
# | many prediction points | cost linear in their number (mean and stdev only need O(n·q), not O(q²)) |

# %%
import time

import matplotlib.pyplot as plt
import numpy as np
import pylibkriging as lk

import lkgallery
from lkgallery import TESTS, design, compat

suite = lkgallery.Suite("robustness-invalid-inputs_Kriging_python")
print("pylibkriging", compat.version())
tf = TESTS["branin"]
X, y, Xt, yt = design.dataset(tf, 20, n_test=200, seed=71)


def pred(k, Xn):
    m, s, *_ = k.predict(Xn, return_stdev=True)
    return np.ravel(m), np.ravel(s)


def finite_model(check, make, Xn=Xt, severity="error"):
    """The model either raises (acceptable) or predicts finite values."""
    try:
        k = make()
    except Exception as exc:  # noqa: BLE001
        suite.true(f"{check}: rejected or finite", True, detail=f"rejected: {type(exc).__name__}")
        return None
    m, s = pred(k, Xn)
    suite.true(f"{check}: rejected or finite", np.all(np.isfinite(m)) and np.all(np.isfinite(s)),
               detail=f"non-finite: {np.sum(~np.isfinite(m))} mean, {np.sum(~np.isfinite(s))} stdev",
               severity=severity)
    return k

# %% [markdown]
# ## Invalid inputs are rejected

# %%
Xn = X.copy(); Xn[3, 1] = np.nan
yn = y.copy(); yn[5] = np.nan
yi = y.copy(); yi[5] = np.inf
suite.raises("NaN in X is rejected", lambda: lk.Kriging(y, Xn, "matern5_2"))
suite.raises("NaN in y is rejected", lambda: lk.Kriging(yn, X, "matern5_2"))
suite.raises("Inf in y is rejected", lambda: lk.Kriging(yi, X, "matern5_2"))
suite.raises("len(y) != nrow(X) is rejected", lambda: lk.Kriging(y[:-1], X, "matern5_2"))
suite.raises("unknown kernel is rejected", lambda: lk.Kriging(y, X, "matern7_2"))
suite.raises("unknown trend is rejected", lambda: lk.Kriging(y, X, "matern5_2", regmodel="cubic"))
suite.raises("unknown objective is rejected", lambda: lk.Kriging(y, X, "matern5_2", objective="XYZ"))
suite.raises("noise vector of the wrong length is rejected",
             lambda: lk.Kriging(y, X, "matern5_2", noise=np.ones(len(y) - 1)))
suite.raises("negative noise variance is rejected",
             lambda: lk.Kriging(y, X, "matern5_2", noise=-np.ones(len(y))), severity="warn")
k = lk.Kriging(y, X, "matern5_2")
suite.raises("predict with the wrong number of columns is rejected", lambda: k.predict(np.ones((3, 3)), True))
if compat.has("WarpKriging"):
    suite.raises("WarpKriging with 1 spec for 2 columns is rejected", lambda: lk.WarpKriging(y, X, ["none"], "gauss"))
    suite.raises("WarpKriging with an unknown spec is rejected",
                 lambda: lk.WarpKriging(y, X, ["none", "spline"], "gauss"))

# %% [markdown]
# ## Degenerate designs and responses

# %%
Xd = np.vstack([X, X[:3]]); yd = np.concatenate([y, y[:3]])
finite_model("duplicated points, noise-free", lambda: lk.Kriging(yd, Xd, "matern5_2"))
if compat.has("noise"):
    k = finite_model("duplicated points, nugget", lambda: lk.Kriging(yd, Xd, "matern5_2", noise="nugget"))
    suite.true("duplicated points, nugget: a model is built", k is not None)
kc = finite_model("constant y", lambda: lk.Kriging(np.full(20, 3.7), X, "matern5_2"))
if kc is not None:
    suite.close("constant y: predicts the constant", pred(kc, Xt)[0], np.full(len(Xt), 3.7), rtol=1e-6, atol=1e-6)
for n in (2, 3):
    finite_model(f"n = {n}", lambda: lk.Kriging(y[:n], X[:n], "matern5_2"))
Xclose = np.vstack([X[:10], X[:10] + 1e-8])
yclose = tf.unit(Xclose)
for kernel in ("gauss", "matern5_2"):
    finite_model(f"points 1e-8 apart, {kernel}", lambda: lk.Kriging(yclose, Xclose, kernel))
# a single input dimension constant over the design
Xconst = np.column_stack([X[:, 0], np.full(20, 0.5)])
finite_model("one input constant over the design", lambda: lk.Kriging(tf.unit(Xconst), Xconst, "matern5_2"))

# %% [markdown]
# ## Scale invariance and equivariance

# %%
k0 = lk.Kriging(y, X, "matern5_2", normalize=True)
m0, s0 = pred(k0, Xt)
for scale in (1e-6, 1e6):
    with suite.runs(f"X * {scale:g} with normalize=True: fit"):
        ks = lk.Kriging(y, X * scale, "matern5_2", normalize=True)
        suite.close(f"X * {scale:g}, normalize=True: same mean as on [0,1]", pred(ks, Xt * scale)[0], m0,
                    rtol=1e-3, atol=1e-3 * np.std(y))
for a, b in [(1e8, 0.0), (1.0, 1e6), (1e-8, 0.0)]:
    with suite.runs(f"y * {a:g} + {b:g}: fit"):
        ka = lk.Kriging(a * y + b, X, "matern5_2", normalize=True)
        ma, sa = pred(ka, Xt)
        suite.close(f"y * {a:g} + {b:g}: mean equivariant", ma, a * m0 + b, rtol=1e-3, atol=1e-3 * a * np.std(y))
        suite.close(f"y * {a:g} + {b:g}: stdev scales by |a|", sa, abs(a) * s0, rtol=1e-2,
                    atol=1e-3 * a * np.std(y))
# the same without normalize: the optimizer starts from a scale-dependent point, warn-level only
for a, b in [(1e6, 0.0), (1.0, 1e6)]:
    ka = lk.Kriging(a * y + b, X, "matern5_2")
    suite.close(f"y * {a:g} + {b:g} (normalize=False): mean equivariant", pred(ka, Xt)[0],
                a * pred(lk.Kriging(y, X, "matern5_2"), Xt)[0] + b, rtol=1e-3, atol=1e-3 * a * np.std(y),
                severity="warn")

# %% [markdown]
# ## Far from the data

# %%
k = lk.Kriging(y, X, "matern5_2")
far = np.array([[50.0, 50.0], [-30.0, 80.0]])
m, s = pred(k, far)
suite.close("far away: mean -> beta (constant trend)", m, np.full(2, np.ravel(k.beta())[0]), rtol=1e-6,
            atol=1e-8)
# universal kriging: sigma2 * (1 + 1 / (1' R^-1 1)) with the trend uncertainty, slightly above sigma2
suite.between("far away: stdev / sqrt(sigma2)", np.max(s) / np.sqrt(k.sigma2()), 1.0 - 1e-6, 1.5)
suite.close("far away: same stdev at both points", s[0], s[1], rtol=1e-6)

# %% [markdown]
# ## Many prediction points
#
# Mean and stdev at q points need O(n·q) operations and memory; only the full
# covariance (`return_cov=True`) is O(q²). The time for q = 1000, 2000, 4000,
# 8000 points must grow linearly (log2 of the ratio per doubling ~ 1). An
# exponent close to 2 means a q × q matrix is formed: a million points then needs
# terabytes. (Not run at 10⁶ points directly: it would exhaust the runner.)

# %%
times = {}
for q in (1000, 2000, 4000, 8000):
    Xq = design.uniform(q, 2, seed=72)
    best = np.inf
    for _ in range(3):
        t0 = time.perf_counter()
        pred(k, Xq)
        best = min(best, time.perf_counter() - t0)
    times[q] = best
    suite.metric(f"predict_q={q}", "pylibkriging", seconds=best)
expo = np.log2(times[8000] / times[4000])
suite.less("predict(stdev) time growth exponent in q (8000 vs 4000)", expo, 1.5,
           detail=", ".join(f"q={q}: {t:.3f}s" for q, t in times.items()))
qs = np.array(list(times)); ts = np.array(list(times.values()))
plt.figure(figsize=(5, 3.6))
plt.loglog(qs, ts, "o-", label="predict(return_stdev=True)")
plt.loglog(qs, ts[0] * qs / qs[0], "k--", lw=.8, label="O(q) reference")
plt.loglog(qs, ts[0] * (qs / qs[0]) ** 2, "k:", lw=.8, label="O(q²) reference")
plt.xlabel("q (prediction points)"); plt.ylabel("time (s)"); plt.legend(); plt.grid(True, which="both", alpha=.3)
plt.show()

# %% [markdown]
# ## Summary
#
# The metrics of this notebook, the number of checks per status, and every check that
# did not pass (FAIL: error-level, fails the run; WARN: warn-level; KNOWN: listed in
# `tools/known_issues.csv`, with its issue; SKIP: feature absent from the installed version).
# All of them are also written to `results/robustness-invalid-inputs_Kriging_python.checks.csv` and `.metrics.csv`.

# %%
suite.summary()

# %%
suite.finalize()
