# %% [markdown]
# # Python · WarpKriging — non-stationarity, categorical and ordinal inputs
#
# `WarpKriging` learns a transform per input variable jointly with the GP
# hyperparameters. Problems where a plain stationary GP is the wrong model:
#
# | case | function | warp |
# |---|---|---|
# | 1D non-stationary | Gramacy & Lee (2012): fast oscillation on the left, flat on the right | `kumaraswamy`, `knots(5)`, `neural_mono(8)`, `boxcox`, `affine` |
# | 2D non-stationary | Gramacy & Lee (2008): a bump in a corner of [−2, 6]² | `kumaraswamy` × 2 |
# | mixed continuous / categorical | Branin + a level-dependent shift and scale | `categorical(5,2)`, `none` |
# | mixed continuous / ordinal | Hartmann 3 with x₁ observed on 6 ordered levels | `ordinal(6)` |
#
# Checks: all-`none` warping reproduces plain `Kriging`; a noise-free
# `WarpKriging` interpolates; predictions are finite; and (warn-level) the warps
# beat plain `Kriging` where they are meant to.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylibkriging as lk

import lkbench
from lkbench import TESTS, design, metrics, compat

suite = lkbench.Suite("python_WarpKriging_warpings-categorical-ordinal")
print("pylibkriging", compat.version())
if not compat.has("WarpKriging"):
    suite.skip("WarpKriging", "not available in this pylibkriging")


def pred(k, X):
    m, s, *_ = k.predict(X, return_stdev=True)
    return np.ravel(m), np.ravel(s)


BASE_NLPD = {}


def evaluate(case, label, k, X, y, Xt, yt):
    m, s = pred(k, Xt)
    res = metrics.all_metrics(yt, m, s)
    # calibration: a warp must not make the stdev collapse where the error is not small
    if label.startswith("Kriging"):
        BASE_NLPD[case] = res["nlpd"]
    else:
        suite.less(f"{case}/{label}: NLPD - NLPD(Kriging) (overconfident stdev?)", res["nlpd"] - BASE_NLPD[case],
                   2.0, severity="warn", detail=f"min stdev {np.min(s):.3g}, max |error|/stdev "
                   f"{np.max(np.abs(yt - m) / np.maximum(s, 1e-300)):.3g}")
    suite.metric(case, label, ll=float(k.logLikelihood()), **res)
    suite.finite(f"{case}/{label}: predictions finite", m)
    mX, _ = pred(k, X)
    suite.less(f"{case}/{label}: interpolates (max |m(X)-y| / sd(y))", np.max(np.abs(mX - y)) / np.std(y), 1e-3,
               severity="warn")
    return res

# %% [markdown]
# ## `none` warping == plain Kriging

# %%
if compat.has("WarpKriging"):
    tf = TESTS["branin"]
    X, y, Xt, yt = design.dataset(tf, 30, n_test=500, seed=31)
    for kernel in ["gauss", "matern5_2"]:
        kp = lk.Kriging(y, X, kernel)
        kw = lk.WarpKriging(y, X, ["none", "none"], kernel)
        suite.close(f"{kernel}: WarpKriging(none) LL == Kriging LL", kw.logLikelihood(), kp.logLikelihood(),
                    rtol=1e-3, severity="warn")
        suite.close(f"{kernel}: WarpKriging(none) mean == Kriging mean", pred(kw, Xt)[0], pred(kp, Xt)[0],
                    rtol=1e-2, atol=1e-2 * np.std(y), severity="warn")

# %% [markdown]
# ## 1D non-stationary: Gramacy & Lee (2012)

# %%
WARPS_1D = ["affine", "boxcox", "kumaraswamy", "knots(5)", "neural_mono(8)"]
if compat.has("WarpKriging"):
    tf = TESTS["gramacy_lee_2012"]
    X, y, Xt, yt = design.dataset(tf, 25, n_test=1000, seed=32)
    X = X * 0.98 + 0.01; Xt = Xt * 0.98 + 0.01  # strictly inside (0,1) for boxcox/kumaraswamy
    y, yt = tf.unit(X), tf.unit(Xt)
    base = evaluate("gl2012", "Kriging", lk.Kriging(y, X, "matern5_2"), X, y, Xt, yt)
    fits = {}
    for w in WARPS_1D:
        with suite.runs(f"gl2012/{w}: fit"):
            fits[w] = lk.WarpKriging(y, X, [w], "matern5_2")
            evaluate("gl2012", w, fits[w], X, y, Xt, yt)
    t = suite.table().loc["gl2012"]
    best = t.drop(index="Kriging")["rmse"].min()
    suite.less("gl2012: best warp RMSE / Kriging RMSE", best / base["rmse"], 1.0, severity="warn")
    u = np.linspace(0.01, 0.99, 400)[:, None]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(u, tf.unit(u), "k--", lw=2, label="truth")
    ax.plot(u, pred(lk.Kriging(y, X, "matern5_2"), u)[0], label="Kriging")
    for w in ["kumaraswamy", "knots(5)", "neural_mono(8)"]:
        if w in fits:
            ax.plot(u, pred(fits[w], u)[0], label=w)
    ax.plot(X, y, "ko"); ax.legend(); ax.set_title("Gramacy & Lee (2012), n=25")
    display(t.round(4))

# %% [markdown]
# ## 2D non-stationary: Gramacy & Lee (2008)

# %%
if compat.has("WarpKriging"):
    tf = TESTS["gramacy_lee_2008"]
    X, y, Xt, yt = design.dataset(tf, 40, n_test=2000, seed=33)
    X = X * 0.98 + 0.01; Xt = Xt * 0.98 + 0.01
    y, yt = tf.unit(X), tf.unit(Xt)
    base = evaluate("gl2008", "Kriging", lk.Kriging(y, X, "matern5_2"), X, y, Xt, yt)
    with suite.runs("gl2008/kumaraswamy: fit"):
        res = evaluate("gl2008", "kumaraswamy", lk.WarpKriging(y, X, ["kumaraswamy"] * 2, "matern5_2"),
                       X, y, Xt, yt)
        suite.less("gl2008: RMSE(kumaraswamy) / RMSE(Kriging)", res["rmse"] / base["rmse"], 1.0, severity="warn")

# %% [markdown]
# ## Mixed continuous / categorical
#
# y = a_c · Branin(x₁, x₂) + b_c for a level c ∈ {0..4} with level-specific
# (a_c, b_c): two pairs of levels are close, the fifth one is different. The
# categorical variable is passed as its level index 0..L−1.

# %%
if compat.has("WarpKriging"):
    tf = TESTS["branin"]
    L = 5
    a = np.array([1.0, 1.05, 0.6, 0.65, -0.5])
    b = np.array([0.0, 5.0, 40.0, 45.0, 100.0])

    def mixed(Xc):
        c = Xc[:, 0].astype(int)
        return a[c] * tf.unit(Xc[:, 1:]) + b[c]

    def mixed_design(n, seed):
        rng = np.random.default_rng(seed)
        Xc = np.column_stack([np.arange(n) % L, design.lhs(n, 2, seed=seed)])
        return Xc[rng.permutation(n)]

    X = mixed_design(60, 34); Xt = mixed_design(1000, 35)
    y, yt = mixed(X), mixed(Xt)
    base = evaluate("mixed", "Kriging (level as number)", lk.Kriging(y, X, "matern5_2"), X, y, Xt, yt)
    with suite.runs("mixed/categorical: fit"):
        kc = lk.WarpKriging(y, X, [f"categorical({L},2)", "none", "none"], "matern5_2")
        res = evaluate("mixed", "categorical(5,2)", kc, X, y, Xt, yt)
        suite.less("mixed: RMSE(categorical) / RMSE(level as number)", res["rmse"] / base["rmse"], 1.0,
                   severity="warn")
        suite.greater("mixed/categorical: Q2", res["q2"], 0.8, severity="warn")

# %% [markdown]
# ## Mixed continuous / ordinal
#
# Hartmann 3 with x₁ only observed on 6 non-equally spaced ordered levels; the
# model sees the level rank 0..5.

# %%
if compat.has("WarpKriging"):
    tf = TESTS["hartmann3"]
    levels = np.array([0.0, 0.05, 0.15, 0.4, 0.8, 1.0])

    def ordinal_design(n, seed):
        U = design.lhs(n, 3, seed=seed)
        r = np.minimum((U[:, 0] * 6).astype(int), 5)
        return np.column_stack([r, U[:, 1:]]).astype(float)

    def f_ord(Xo):
        return tf.unit(np.column_stack([levels[Xo[:, 0].astype(int)], Xo[:, 1:]]))

    X = ordinal_design(60, 36); Xt = ordinal_design(1000, 37)
    y, yt = f_ord(X), f_ord(Xt)
    base = evaluate("ordinal", "Kriging (rank as number)", lk.Kriging(y, X / [5, 1, 1], "matern5_2"),
                    X / [5, 1, 1], y, Xt / [5, 1, 1], yt)
    with suite.runs("ordinal/ordinal(6): fit"):
        ko = lk.WarpKriging(y, X, ["ordinal(6)", "none", "none"], "matern5_2")
        res = evaluate("ordinal", "ordinal(6)", ko, X, y, Xt, yt)
        suite.less("ordinal: RMSE(ordinal) / RMSE(rank as number)", res["rmse"] / base["rmse"], 1.1,
                   severity="warn")

# %%
suite.table().round(4)

# %%
suite.finalize()
