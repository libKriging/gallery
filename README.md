# libKriging benchmarks

Test and benchmark notebooks for [libKriging](https://github.com/libKriging/libKriging),
kept outside the library so that they can follow both its development head and
its releases. Every notebook **checks** what it computes (error-level checks fail
the run, warn-level ones are reported), and CI executes all of them and
aggregates the findings of every language into one report.

## Executed notebooks (libKriging master)

CI executes every notebook against the **development head of libKriging and of
its bindings** (master channel, see [CI](#ci)) and saves it with its outputs,
failing checks included (a notebook that fails is kept up to its failing cell):

* **[latest executed notebooks (zip)](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-notebooks-master.zip)**:
  the `executed-notebooks-master` artifact of the last run on `main`
  (`python/`, `r/`, `octave/`, `julia/`);
* **[latest report, results and notebooks (zip)](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/report-master.zip)**:
  the `report-master` artifact (aggregated `report.md`, all CSV results, executed
  notebooks, refreshed baseline);
* **[all runs on `main`](https://github.com/libKriging/benchmarks/actions/workflows/notebooks.yml?query=branch%3Amain)**:
  each run keeps these artifacts for 90 days, and its job summary shows the report.

The same artifacts exist for the release channel (`executed-notebooks-release`,
`report-release`).

## What is covered

| notebook | problems (test functions) | libKriging features | compared with |
|---|---|---|---|
| [python/00](notebooks/python/00_test_functions.ipynb) | the 21 test functions themselves, against published optima / ranges | — | — |
| [python/01](notebooks/python/01_kriging_kernels_trends.ipynb) | Forrester, Gramacy & Lee 2012, Branin, six-hump camel, log-Goldstein-Price, Lim, Ishigami, Hartmann 3 | 4 kernels × 4 trends, interpolation, `predict` (cov, derivatives) | finite differences |
| [python/02](notebooks/python/02_objectives_gradients.ipynb) | Branin, Hartmann 3, Ishigami, Friedman | `LL` / `LOO` / `LMP` objectives and gradients, optimality of θ̂, `leaveOneOutVec`, multistart | finite differences, explicit refits |
| [python/03](notebooks/python/03_noise.ipynb) | Forrester, Friedman, Gramacy & Lee 2012, Branin (replicates) | `noise="nugget"`, heteroscedastic noise vector | true noise, replicate averaging |
| [python/04](notebooks/python/04_update_simulate.ipynb) | Branin, Hartmann 3, Hartmann 6 | `update`, `simulate`, `update_simulate` | fits from scratch, moments |
| [python/05](notebooks/python/05_warpkriging.ipynb) | Gramacy & Lee 2012 / 2008, mixed categorical Branin, ordinal Hartmann 3 | `WarpKriging`: affine, boxcox, kumaraswamy, knots, neural_mono, categorical, ordinal | plain `Kriging` |
| [python/06](notebooks/python/06_high_dimension_mlp.ipynb) | OTL circuit, piston, borehole, robot arm, Dette-Pepelyshev, wing weight, Welch (20D), projected Branin (10D) | `normalize`, screening, `MLPKriging` | plain `Kriging` |
| [python/07](notebooks/python/07_scalability.ipynb) | Hartmann 3, OTL circuit (n up to 5000), Branin | `method="vecchia(m)"`, `"nystrom(k)"`, `NestedKriging` (5 aggregations), `subsetOfData` | exact fit |
| [python/08](notebooks/python/08_api_persistence.ipynb) | Hartmann 3 | `save`/`load`, pickle, copy, accessors, scikit-learn estimators | — |
| [python/09](notebooks/python/09_robustness.ipynb) | Branin | invalid inputs, duplicates, constant y, ill-conditioning, scale equivariance, extrapolation, cost in q | — |
| [python/10](notebooks/python/10_compare_python_packages.ipynb) | 10 functions, 2D to 10D | default fits | scikit-learn, SMT, OpenTURNS, GPy (exact equality at fixed hyperparameters + accuracy/cost) |
| [python/11](notebooks/python/11_cross_binding.ipynb), [r/01](notebooks/r/01_cross_binding.ipynb), [octave/01](notebooks/octave/01_cross_binding.ipynb), [julia/01](notebooks/julia/01_cross_binding.ipynb) | shared datasets of `data/` | the same protocol in every binding | each other (report) |
| [r/02](notebooks/r/02_vs_dicekriging_robustgasp.ipynb) | Branin, Hartmann 3, Borehole | `rlibkriging` | DiceKriging (exact equality), RobustGaSP |
| [octave/02](notebooks/octave/02_vs_stk.ipynb) | Branin, Hartmann 3, Borehole | `mLibKriging` | STK (exact equality, Gaussian kernel) |
| [julia/01](notebooks/julia/01_cross_binding.ipynb) | Branin, Hartmann 3, Borehole | `JLibKriging` | GaussianProcesses.jl |

Test functions: [`lkbench/functions.py`](lkbench/functions.py), after the
[Virtual Library of Simulation Experiments](https://www.sfu.ca/~ssurjano/)
(Surjanovic & Bingham).

## Checks and statuses

Each notebook records its checks with `lkbench.Suite` (Python) or the small
equivalents in `notebooks/{r,octave,julia}/` and writes
`results/<suite>.checks.csv` and `results/<suite>.metrics.csv`.

| status | meaning | fails CI |
|---|---|---|
| PASS | | |
| FAIL | an error-level check failed: a correctness property (gradient vs finite differences, `update` vs fit from scratch, equality with another package at fixed hyperparameters, ...) or an accuracy floor | yes |
| WARN | a warn-level check failed: performance comparison, optimizer behaviour, judgement thresholds | no |
| KNOWN | an error-level failure listed in [`known_issues.csv`](known_issues.csv), with the issue | no |
| SKIP | feature absent from the installed version (release channel) | no |

`tools/report.py` aggregates all results: status per notebook, failures,
warnings, known issues still present, known issues that now pass (to be removed
from the list), cross-binding consistency against Python, and metric drifts
against [`baseline/`](baseline).

## Running locally

```sh
pip install -e ".[notebooks,compare]" jinja2 pylibkriging   # or a pylibkriging built from source
python tools/run_notebooks.py notebooks/python/*.ipynb
python tools/report.py
```

R needs `rlibkriging`, `DiceKriging`, `RobustGaSP` and `IRkernel`; Octave needs
mLibKriging on `OCTAVE_PATH`, STK on `STK_PATH` and `octave_kernel`; Julia needs
`JLibKriging`, `IJulia`, `DelimitedFiles` (and optionally `GaussianProcesses`,
`Optim`). See the CI workflow for exact commands.

## Editing notebooks

Notebooks are generated from plain-text sources (`notebooks/<lang>/src/`, cells
separated by `# %%` / `# %% [markdown]`, or `% %%` in Octave), so that reviews are
plain diffs; outputs are not committed (CI archives the executed notebooks as
artifacts, see [Executed notebooks](#executed-notebooks-libkriging-master)).

```sh
python tools/py2nb.py notebooks/*/src/*          # regenerate
python tools/py2nb.py --check notebooks/*/src/*  # CI: sources and notebooks in sync
```

Shared datasets (`data/`, used by the cross-binding notebooks of every language)
are written once by `tools/make_data.py` and committed: the files are the
reference.

## CI

[`.github/workflows/notebooks.yml`](.github/workflows/notebooks.yml), on push,
pull request, weekly, on demand (`libkriging_ref`, `libkriging_repo`,
`channels` inputs) and on a `repository_dispatch` of type `libkriging-updated`
(sent by libKriging after a merge, with the commit in `client_payload.ref`):

* **master channel**: libKriging built from source at the requested ref, in
  every language (Python wheel, Octave mex, JLibKriging.jl against the same
  checkout; R from the `libKriging/rlibkriging` repository, which follows
  libKriging master);
* **release channel**: pylibkriging from PyPI, rlibkriging from CRAN,
  mLibKriging built from the last release tag (the published binaries target
  another Octave version than Ubuntu's).

One job per Python notebook, one per language for the others, then one report
per channel (job summary + `report-<channel>` artifact with the report, all CSV
results, the executed notebooks and a refreshed baseline, + an
`executed-notebooks-<channel>` artifact with the executed notebooks only; see
[Executed notebooks](#executed-notebooks-libkriging-master)). The initial `baseline/` files come
from a local run (Python notebooks only, 4-core container); to refresh them,
commit the `new-baseline/` files of a trusted CI run.
