# %% [markdown]
# # Python · no model — the test functions against their published reference values
#
# **Idea.** Every example of this gallery is only as good as its test functions:
# before fitting anything, each of the 21 functions is evaluated at points where its
# value is published, and its range and shape are plotted.
#
# **Reference.** Published optima and nominal values (Virtual Library of Simulation
# Experiments, Surjanovic & Bingham).
#
# Every notebook of this gallery relies on the test functions of
# `gallery.functions` (after the [Virtual Library of Simulation Experiments](https://www.sfu.ca/~ssurjano/)).
# A typo in one of them would silently bias every result built on it, so they are
# checked first, against the reference values published on the library pages
# (global optima, nominal points) and against the known output ranges.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gallery
from gallery import TESTS, design

suite = gallery.Suite("test-functions-reference-values_no-model_python")

# %% [markdown]
# ## Reference values

# %%
rows = []
for name, tf in TESTS.items():
    for x, v in tf.known:
        fx = float(tf(x)[0])
        suite.check_close(f"{name}: f({np.round(x, 4).tolist()})", fx, v, rtol=1e-4, atol=1e-4)
        rows.append((name, tf.d, np.round(x, 4).tolist(), v, fx))
pd.DataFrame(rows, columns=["function", "d", "x", "reference", "computed"])

# %% [markdown]
# ## Output ranges
#
# For the engineering functions without a closed-form optimum, the range over the
# domain is compared with the one reported in the literature (loose bounds: a
# random sample does not reach the extremes).

# %%
EXPECTED_RANGES = {  # (lower bound of min, upper bound of max) over the domain
    "otl_circuit": (2.5, 10.0),
    "piston": (0.1, 1.3),
    "borehole": (2.0, 320.0),
    "wing_weight": (120.0, 480.0),
    "currin_exp": (1.0, 14.0),
    "dette_pepelyshev": (0.0, 100.0),
}
summary = []
for name, tf in TESTS.items():
    U = design.uniform(20_000, tf.d, seed=1)
    y = tf.unit(U)
    suite.check_finite(f"{name}: finite on the domain", y)
    if name in EXPECTED_RANGES:
        lo, hi = EXPECTED_RANGES[name]
        suite.check_true(f"{name}: range within [{lo}, {hi}]", lo <= y.min() and y.max() <= hi,
                   detail=f"sampled range [{y.min():.4g}, {y.max():.4g}]")
    summary.append((name, tf.d, ",".join(tf.tags), y.min(), y.max(), y.std()))
pd.DataFrame(summary, columns=["function", "d", "tags", "min", "max", "std"])

# %% [markdown]
# ## Designs
#
# The designs are seeded: the same seed gives the same points (reproducibility of
# every notebook), and the optimized LHS is a Latin hypercube (one point per
# stratum in each dimension).

# %%
A, B = design.lhs(20, 3, seed=7), design.lhs(20, 3, seed=7)
suite.check_close("lhs: reproducible with a seed", A, B, rtol=0, atol=0)
strata = np.floor(A * 20).astype(int)
suite.check_true("lhs: one point per stratum", all(len(set(strata[:, j])) == 20 for j in range(3)))

# %%
fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
for ax, name in zip(axes, ["forrester", "gramacy_lee_2012"]):
    tf = TESTS[name]
    u = np.linspace(0, 1, 500)[:, None]
    ax.plot(tf.scale(u)[:, 0], tf.unit(u))
    ax.set_title(name)
for ax, name in zip(axes[2:], ["branin", "gramacy_lee_2008"]):
    tf = TESTS[name]
    g = np.linspace(0, 1, 100)
    G = np.array(np.meshgrid(g, g)).reshape(2, -1).T
    ax.contourf(g, g, tf.unit(G).reshape(100, 100), 30)
    ax.set_title(name)
plt.tight_layout()

# %% [markdown]
# ## Summary
#
# The metrics of this notebook, the number of checks per status, and every check that
# did not pass (FAIL: error-level, fails the run; WARN: warn-level; KNOWN: listed in
# `tools/known_issues.csv`, with its issue; SKIP: feature absent from the installed version).
# All of them are also written to `results/test-functions-reference-values_no-model_python.checks.csv` and `.metrics.csv`.

# %%
suite.summary()

# %%
suite.finalize()
