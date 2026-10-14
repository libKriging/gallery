# libKriging benchmarks

Test and benchmark notebooks for [libKriging](https://github.com/libKriging/libKriging),
kept outside the library so that they can follow both its development head and
its releases. Every notebook **checks** what it computes (error-level checks fail
the run, warn-level ones are reported), and CI executes each of them against the
**development head of libKriging and of its bindings**, saves it with its outputs,
and aggregates the findings of every language into one report.

The notebooks are at the root of the repository, named
`<theme>_<kriging type>_<binding>.ipynb`; everything else (notebook sources,
helpers, binding installation, CI scripts, known issues, baselines) is under
[`tools/`](tools), and the shared datasets under [`data/`](data).

## Notebooks

The **executed** column links to the notebook as executed by the last CI run on
`main` against libKriging master (outputs saved, failing checks included: a
notebook that fails is kept up to its failing cell), a zip of the
`executed-master-<notebook>` artifact (through [nightly.link](https://nightly.link),
which serves the last artifact of a branch at a stable URL). Every run keeps
these artifacts for 90 days: [all runs on `main`](https://github.com/libKriging/benchmarks/actions/workflows/notebooks.yml?query=branch%3Amain).
All the executed notebooks of a run, in one zip:
[executed-notebooks-master](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-notebooks-master.zip);
the aggregated report with all CSV results:
[report-master](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/report-master.zip).

### Python (`pylibkriging`)

| notebook | kriging type | theme: problems, features | compared with | executed |
|---|---|---|---|---|
| [test-functions-reference-values_no-model_python](test-functions-reference-values_no-model_python.ipynb) | — | the 21 test functions of [`tools/lkbench/functions.py`](tools/lkbench/functions.py) against published optima / ranges | — | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-test-functions-reference-values_no-model_python.zip) |
| [kernels-trends-interpolation_Kriging_python](kernels-trends-interpolation_Kriging_python.ipynb) | `Kriging` | 4 kernels × 4 trends on Forrester, Gramacy & Lee, Branin, six-hump camel, log-Goldstein-Price, Lim, Ishigami, Hartmann 3: interpolation, `predict` (covariance, derivatives) | finite differences | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-kernels-trends-interpolation_Kriging_python.zip) |
| [objectives-LL-LOO-LMP-gradients_Kriging_python](objectives-LL-LOO-LMP-gradients_Kriging_python.ipynb) | `Kriging` | `LL` / `LOO` / `LMP` objectives and gradients, optimality of θ̂, `leaveOneOutVec`, multistart (Branin, Hartmann 3, Ishigami, Friedman) | finite differences, explicit refits | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-objectives-LL-LOO-LMP-gradients_Kriging_python.zip) |
| [update-simulate_Kriging_python](update-simulate_Kriging_python.ipynb) | `Kriging` | `update`, `simulate`, `update_simulate` (Branin, Hartmann 3, Hartmann 6) | fits from scratch, moments | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-update-simulate_Kriging_python.zip) |
| [high-dimension-normalize-screening_Kriging_python](high-dimension-normalize-screening_Kriging_python.ipynb) | `Kriging` | `normalize`, screening, 6D to 20D engineering functions (OTL circuit, piston, borehole, robot arm, Dette-Pepelyshev, wing weight, Welch) | unit-cube fit | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-high-dimension-normalize-screening_Kriging_python.zip) |
| [robustness-invalid-inputs_Kriging_python](robustness-invalid-inputs_Kriging_python.ipynb) | `Kriging` | invalid inputs, duplicates, constant y, ill-conditioning, scale equivariance, extrapolation, cost in q (Branin) | — | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-robustness-invalid-inputs_Kriging_python.zip) |
| [save-load-pickle-copy-accessors_Kriging_python](save-load-pickle-copy-accessors_Kriging_python.ipynb) | `Kriging` | `save`/`load`, pickle, copy, accessors, for every noise model, objective and fit method (Hartmann 3) | — | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-save-load-pickle-copy-accessors_Kriging_python.zip) |
| [sklearn-estimators_Kriging_python](sklearn-estimators_Kriging_python.ipynb) | `Kriging` | `pylibkriging.sklearn.KrigingRegressor`: `Pipeline`, `GridSearchCV`, `clone`, `sample_y` (Hartmann 3) | the native class | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-sklearn-estimators_Kriging_python.zip) |
| [vs-sklearn-SMT-OpenTURNS-GPy_Kriging_python](vs-sklearn-SMT-OpenTURNS-GPy_Kriging_python.ipynb) | `Kriging` | default fits on 10 functions, 2D to 10D | scikit-learn, SMT, OpenTURNS, GPy (exact equality at fixed hyperparameters + accuracy/cost) | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-vs-sklearn-SMT-OpenTURNS-GPy_Kriging_python.zip) |
| [cross-binding-reference_Kriging_python](cross-binding-reference_Kriging_python.ipynb) | `Kriging` | the cross-binding protocol on the datasets of `data/`: the Python reference | the other bindings (report) | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-cross-binding-reference_Kriging_python.zip) |
| [noisy-data_Kriging-nugget-noise_python](noisy-data_Kriging-nugget-noise_python.ipynb) | `Kriging` with `noise="nugget"` or variances | estimated nugget, heteroscedastic noise, calibration (Forrester, Friedman, Gramacy & Lee, Branin with replicates) | true noise, replicate averaging | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-noisy-data_Kriging-nugget-noise_python.zip) |
| [large-n_Kriging-vecchia-nystrom_python](large-n_Kriging-vecchia-nystrom_python.ipynb) | `Kriging` with `method="vecchia(m)"` / `"nystrom(k)"` | large n (Hartmann 3, OTL circuit, n up to 5000), limit cases, `subsetOfData` | exact fit | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-large-n_Kriging-vecchia-nystrom_python.zip) |
| [warpings-categorical-ordinal_WarpKriging_python](warpings-categorical-ordinal_WarpKriging_python.ipynb) | `WarpKriging` | affine, boxcox, kumaraswamy, knots, neural_mono, categorical, ordinal warpings (Gramacy & Lee 2012 / 2008, mixed categorical Branin, ordinal Hartmann 3); save/load, pickle | plain `Kriging` | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-warpings-categorical-ordinal_WarpKriging_python.zip) |
| [deep-kernel-projected-branin_MLPKriging_python](deep-kernel-projected-branin_MLPKriging_python.ipynb) | `MLPKriging` | deep kernel on a 10D function with a 2D structure, activations, multistart; save/load, pickle | plain `Kriging` | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-deep-kernel-projected-branin_MLPKriging_python.zip) |
| [aggregations-large-n_NestedKriging_python](aggregations-large-n_NestedKriging_python.ipynb) | `NestedKriging` | `NK`, `PoE`, `gPoE`, `BCM`, `rBCM` aggregations, large n; save/load, pickle | exact fit | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-aggregations-large-n_NestedKriging_python.zip) |

### R (`rlibkriging`)

| notebook | kriging type | theme: problems, features | compared with | executed |
|---|---|---|---|---|
| [interpolation-simulate-update-save_Kriging_R](interpolation-simulate-update-save_Kriging_R.ipynb) | `Kriging` | interpolation, conditional simulation, `update`, `save`/`load` (Branin) | fit on all points | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-interpolation-simulate-update-save_Kriging_R.zip) |
| [cross-binding_Kriging_R](cross-binding_Kriging_R.ipynb) | `Kriging` | the cross-binding protocol, R side | Python (report) | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-cross-binding_Kriging_R.zip) |
| [vs-DiceKriging-RobustGaSP_Kriging_R](vs-DiceKriging-RobustGaSP_Kriging_R.ipynb) | `Kriging` | default fits (Branin, Hartmann 3, Borehole) | DiceKriging (exact equality), RobustGaSP | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-vs-DiceKriging-RobustGaSP_Kriging_R.zip) |
| [noisy-data_Kriging-nugget-noise_R](noisy-data_Kriging-nugget-noise_R.ipynb) | `Kriging` with nugget or known variances | estimated nugget (10 restarts vs single start), known noise variances (Branin + noise) | — | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-noisy-data_Kriging-nugget-noise_R.zip) |
| [large-n_Kriging-vecchia-nystrom_R](large-n_Kriging-vecchia-nystrom_R.ipynb) | `Kriging` with Vecchia / Nyström | approximate fits, `method()`, prediction with the fit method (Branin, n = 400) | — | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-large-n_Kriging-vecchia-nystrom_R.zip) |
| [warpings-categorical_WarpKriging_R](warpings-categorical_WarpKriging_R.ipynb) | `WarpKriging` | kumaraswamy warpings, a categorical input (Branin) | — | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-warpings-categorical_WarpKriging_R.zip) |
| [deep-kernel_MLPKriging_R](deep-kernel_MLPKriging_R.ipynb) | `MLPKriging` | joint feature map, multistart vs default optimizer (Branin) | — | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-deep-kernel_MLPKriging_R.zip) |
| [aggregations_NestedKriging_R](aggregations_NestedKriging_R.ipynb) | `NestedKriging` | `NK` and `PoE` aggregations, interpolation, `save`/`load` (Branin, n = 400) | — | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-aggregations_NestedKriging_R.zip) |

### Octave (`mLibKriging`)

| notebook | kriging type | theme: problems, features | compared with | executed |
|---|---|---|---|---|
| [interpolation-simulate-update-save_Kriging_octave](interpolation-simulate-update-save_Kriging_octave.ipynb) | `Kriging` | interpolation, conditional simulation, `update`, `save`/`load` (Branin) | fit on all points | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-interpolation-simulate-update-save_Kriging_octave.zip) |
| [cross-binding_Kriging_octave](cross-binding_Kriging_octave.ipynb) | `Kriging` | the cross-binding protocol, Octave side | Python (report) | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-cross-binding_Kriging_octave.zip) |
| [vs-STK_Kriging_octave](vs-STK_Kriging_octave.ipynb) | `Kriging` | default fits (Branin, Hartmann 3, Borehole) | STK (exact equality, Gaussian kernel) | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-vs-STK_Kriging_octave.zip) |
| [noisy-data_Kriging-nugget-noise_octave](noisy-data_Kriging-nugget-noise_octave.ipynb) | `Kriging` with nugget or known variances | estimated nugget (10 restarts vs single start), known noise variances (Branin + noise) | — | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-noisy-data_Kriging-nugget-noise_octave.zip) |
| [large-n_Kriging-vecchia-nystrom_octave](large-n_Kriging-vecchia-nystrom_octave.ipynb) | `Kriging` with Vecchia / Nyström | approximate fits, `method()`, prediction with the fit method (Branin, n = 400) | — | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-large-n_Kriging-vecchia-nystrom_octave.zip) |
| [warpings-categorical_WarpKriging_octave](warpings-categorical_WarpKriging_octave.ipynb) | `WarpKriging` | kumaraswamy warpings, a categorical input (Branin) | — | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-warpings-categorical_WarpKriging_octave.zip) |
| [deep-kernel_MLPKriging_octave](deep-kernel_MLPKriging_octave.ipynb) | `MLPKriging` | joint feature map, multistart vs default optimizer (Branin) | — | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-deep-kernel_MLPKriging_octave.zip) |
| [aggregations_NestedKriging_octave](aggregations_NestedKriging_octave.ipynb) | `NestedKriging` | `NK` and `PoE` aggregations, interpolation, `save`/`load` (Branin, n = 400) | — | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-aggregations_NestedKriging_octave.zip) |

### Julia (`JLibKriging`)

| notebook | kriging type | theme: problems, features | compared with | executed |
|---|---|---|---|---|
| [interpolation-simulate-update-save_Kriging_julia](interpolation-simulate-update-save_Kriging_julia.ipynb) | `Kriging` | interpolation, conditional simulation, `update`, `save`/`load` (Branin) | fit on all points | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-interpolation-simulate-update-save_Kriging_julia.zip) |
| [cross-binding_Kriging_julia](cross-binding_Kriging_julia.ipynb) | `Kriging` | the cross-binding protocol, Julia side | Python (report) | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-cross-binding_Kriging_julia.zip) |
| [vs-GaussianProcesses_Kriging_julia](vs-GaussianProcesses_Kriging_julia.ipynb) | `Kriging` | default fits (Branin, Hartmann 3, Borehole) | GaussianProcesses.jl | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-vs-GaussianProcesses_Kriging_julia.zip) |
| [noisy-data_Kriging-nugget-noise_julia](noisy-data_Kriging-nugget-noise_julia.ipynb) | `Kriging` with nugget or known variances | estimated nugget (10 restarts vs single start), known noise variances (Branin + noise) | — | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-noisy-data_Kriging-nugget-noise_julia.zip) |
| [large-n_Kriging-vecchia-nystrom_julia](large-n_Kriging-vecchia-nystrom_julia.ipynb) | `Kriging` with Vecchia / Nyström | approximate fits, `method()`, prediction with the fit method (Branin, n = 400) | — | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-large-n_Kriging-vecchia-nystrom_julia.zip) |
| [warpings-categorical_WarpKriging_julia](warpings-categorical_WarpKriging_julia.ipynb) | `WarpKriging` | kumaraswamy warpings, a categorical input (Branin) | — | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-warpings-categorical_WarpKriging_julia.zip) |
| [deep-kernel_MLPKriging_julia](deep-kernel_MLPKriging_julia.ipynb) | `MLPKriging` | joint feature map, multistart vs default optimizer (Branin) | — | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-deep-kernel_MLPKriging_julia.zip) |
| [aggregations_NestedKriging_julia](aggregations_NestedKriging_julia.ipynb) | `NestedKriging` | `NK` and `PoE` aggregations, interpolation, `save`/`load` (Branin, n = 400) | — | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-aggregations_NestedKriging_julia.zip) |

A notebook that compares libKriging with another package installs that package
itself, in its first cells (pip, CRAN, `Pkg.add`, or a clone of STK); a package
that cannot be installed is reported (warn-level) and its comparison skipped.

Test functions: [`tools/lkbench/functions.py`](tools/lkbench/functions.py), after the
[Virtual Library of Simulation Experiments](https://www.sfu.ca/~ssurjano/)
(Surjanovic & Bingham).

## Checks and statuses

Each notebook records its checks with `lkbench.Suite` (Python) or the small
equivalents of `tools/r`, `tools/octave` and `tools/julia`, under a suite named
after the notebook, and writes `results/<notebook>.checks.csv` and
`results/<notebook>.metrics.csv`.

| status | meaning | fails CI |
|---|---|---|
| PASS | | |
| FAIL | an error-level check failed: a correctness property (gradient vs finite differences, `update` vs fit from scratch, equality with another package at fixed hyperparameters, ...) or an accuracy floor | yes |
| WARN | a warn-level check failed: performance comparison, optimizer behaviour, judgement thresholds | no |
| KNOWN | a failing check (of either level) listed in [`tools/known_issues.csv`](tools/known_issues.csv), with the issue | no |
| SKIP | feature absent from the installed version (release channel) | no |

`tools/report.py` aggregates all results: status per notebook, failures,
warnings, known issues still present, known issues that now pass (to be removed
from the list), cross-binding consistency against Python, and metric drifts
against [`tools/baseline/`](tools/baseline).

## Running locally

From the repository root (the notebooks' paths are relative to it):

```sh
git clone --recurse-submodules https://github.com/libKriging/libKriging ../libKriging   # for a master build
tools/install/python.sh master ../libKriging      # or: tools/install/python.sh release
tools/install/r.sh master ~/Rlib ../libKriging    # then R_LIBS_USER=~/Rlib
tools/install/octave.sh ../libKriging ~/mlib      # then OCTAVE_PATH=~/mlib
tools/install/julia.sh ../libKriging
python tools/run_notebooks.py --inplace update-simulate_Kriging_python.ipynb
python tools/report.py --expect python
```

The scripts of [`tools/install/`](tools/install) are the ones CI runs; each says
what it needs (BLAS/LAPACK headers, Octave, R, Julia).

## Editing notebooks

Notebooks are generated from plain-text sources ([`tools/src/`](tools/src), cells
separated by `# %%` / `# %% [markdown]`, or `% %%` in Octave), so that reviews are
plain diffs; outputs are not committed (CI archives the executed notebooks as
artifacts, see [Notebooks](#notebooks)).

```sh
python tools/py2nb.py           # regenerate every notebook from tools/src/
python tools/py2nb.py --check   # CI: sources and notebooks in sync, no notebook without source
```

A new notebook is a new source in `tools/src/` named `<theme>_<kriging type>_<binding>`,
with its suite named the same, and a line in the tables above (CI checks that
every notebook is linked from this README); CI picks it up automatically.

Shared datasets (`data/`, used by the cross-binding notebooks of every language)
are written once by `tools/make_data.py` and committed: the files are the
reference.

## CI

[`.github/workflows/notebooks.yml`](.github/workflows/notebooks.yml), on push,
pull request, weekly, on demand (`libkriging_ref`, `libkriging_repo`,
`channels` inputs) and on a `repository_dispatch` of type `libkriging-updated`
(sent by libKriging after a merge, with the commit in `client_payload.ref`):

* **master channel** (every event): libKriging built from source at the
  requested ref, with every binding from the same checkout: Python wheel, R
  package (`bindings/R`), Octave mex, JLibKriging.jl;
* **release channel** (weekly, or on demand): pylibkriging from PyPI,
  rlibkriging from CRAN, mLibKriging built from the last release tag (the
  published binaries target another Octave version than Ubuntu's); no Julia.

One build job per binding and channel, then **one job per notebook**, each
uploading its executed notebook and its results as the artifact
`executed-<channel>-<notebook>`; then one report per channel (job summary,
`report-<channel>` artifact with the report, all CSV results and a refreshed
baseline, and `executed-notebooks-<channel>` with all the executed notebooks).
The initial `tools/baseline/` files come from a local run (Python notebooks only,
4-core container); to refresh them, commit the `new-baseline/` files of a
trusted CI run.
