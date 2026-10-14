# libKriging benchmarks

Test and benchmark notebooks for [libKriging](https://github.com/libKriging/libKriging),
kept outside the library so that they can follow both its development head and
its releases. Every notebook **checks** what it computes (error-level checks fail
the run, warn-level ones are reported), and CI executes each of them against the
**development head of libKriging and of its bindings**, saves it with its outputs,
and aggregates the findings of every language into one report.

The notebooks are at the root of the repository, named
`<binding>_<kriging type>_<theme>.ipynb`; everything else (notebook sources,
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
| [python_no-model_test-functions-reference-values](python_no-model_test-functions-reference-values.ipynb) | — | the 21 test functions of [`tools/lkbench/functions.py`](tools/lkbench/functions.py) against published optima / ranges | — | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-python_no-model_test-functions-reference-values.zip) |
| [python_Kriging_kernels-trends-interpolation](python_Kriging_kernels-trends-interpolation.ipynb) | `Kriging` | 4 kernels × 4 trends on Forrester, Gramacy & Lee, Branin, six-hump camel, log-Goldstein-Price, Lim, Ishigami, Hartmann 3: interpolation, `predict` (covariance, derivatives) | finite differences | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-python_Kriging_kernels-trends-interpolation.zip) |
| [python_Kriging_objectives-LL-LOO-LMP-gradients](python_Kriging_objectives-LL-LOO-LMP-gradients.ipynb) | `Kriging` | `LL` / `LOO` / `LMP` objectives and gradients, optimality of θ̂, `leaveOneOutVec`, multistart (Branin, Hartmann 3, Ishigami, Friedman) | finite differences, explicit refits | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-python_Kriging_objectives-LL-LOO-LMP-gradients.zip) |
| [python_Kriging-nugget-noise_noisy-data](python_Kriging-nugget-noise_noisy-data.ipynb) | `Kriging` with `noise="nugget"` or variances | estimated nugget, heteroscedastic noise, calibration (Forrester, Friedman, Gramacy & Lee, Branin with replicates) | true noise, replicate averaging | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-python_Kriging-nugget-noise_noisy-data.zip) |
| [python_Kriging_update-simulate](python_Kriging_update-simulate.ipynb) | `Kriging` | `update`, `simulate`, `update_simulate` (Branin, Hartmann 3, Hartmann 6) | fits from scratch, moments | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-python_Kriging_update-simulate.zip) |
| [python_WarpKriging_warpings-categorical-ordinal](python_WarpKriging_warpings-categorical-ordinal.ipynb) | `WarpKriging` | affine, boxcox, kumaraswamy, knots, neural_mono, categorical, ordinal warpings (Gramacy & Lee 2012 / 2008, mixed categorical Branin, ordinal Hartmann 3) | plain `Kriging` | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-python_WarpKriging_warpings-categorical-ordinal.zip) |
| [python_Kriging_high-dimension-normalize-screening](python_Kriging_high-dimension-normalize-screening.ipynb) | `Kriging` | `normalize`, screening, 6D to 20D engineering functions (OTL circuit, piston, borehole, robot arm, Dette-Pepelyshev, wing weight, Welch) | unit-cube fit | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-python_Kriging_high-dimension-normalize-screening.zip) |
| [python_MLPKriging_deep-kernel-projected-branin](python_MLPKriging_deep-kernel-projected-branin.ipynb) | `MLPKriging` | deep kernel on a 10D function with a 2D structure, activations, multistart | plain `Kriging` | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-python_MLPKriging_deep-kernel-projected-branin.zip) |
| [python_Kriging-vecchia-nystrom_large-n](python_Kriging-vecchia-nystrom_large-n.ipynb) | `Kriging` with `method="vecchia(m)"` / `"nystrom(k)"` | large n (Hartmann 3, OTL circuit, n up to 5000), limit cases, `subsetOfData` | exact fit | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-python_Kriging-vecchia-nystrom_large-n.zip) |
| [python_NestedKriging_aggregations-large-n](python_NestedKriging_aggregations-large-n.ipynb) | `NestedKriging` | `NK`, `PoE`, `gPoE`, `BCM`, `rBCM` aggregations, large n | exact fit | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-python_NestedKriging_aggregations-large-n.zip) |
| [python_all-classes_save-load-pickle-sklearn](python_all-classes_save-load-pickle-sklearn.ipynb) | all classes | `save`/`load`, pickle, copy, accessors, scikit-learn estimators (Hartmann 3) | — | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-python_all-classes_save-load-pickle-sklearn.zip) |
| [python_Kriging_robustness-invalid-inputs](python_Kriging_robustness-invalid-inputs.ipynb) | `Kriging` | invalid inputs, duplicates, constant y, ill-conditioning, scale equivariance, extrapolation, cost in q (Branin) | — | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-python_Kriging_robustness-invalid-inputs.zip) |
| [python_Kriging_vs-sklearn-SMT-OpenTURNS-GPy](python_Kriging_vs-sklearn-SMT-OpenTURNS-GPy.ipynb) | `Kriging` | default fits on 10 functions, 2D to 10D | scikit-learn, SMT, OpenTURNS, GPy (exact equality at fixed hyperparameters + accuracy/cost) | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-python_Kriging_vs-sklearn-SMT-OpenTURNS-GPy.zip) |
| [python_Kriging_cross-binding-reference](python_Kriging_cross-binding-reference.ipynb) | `Kriging` | the cross-binding protocol on the datasets of `data/`: the Python reference | the other bindings (report) | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-python_Kriging_cross-binding-reference.zip) |

### R (`rlibkriging`)

| notebook | kriging type | theme: problems, features | compared with | executed |
|---|---|---|---|---|
| [R_all-classes_api-tour](R_all-classes_api-tour.ipynb) | all classes | `Kriging` (noise-free, nugget, known noise, Vecchia, Nyström), `WarpKriging` (continuous, categorical), `MLPKriging`, `NestedKriging`, `update`, `simulate`, `save`/`load` (Branin) | — | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-R_all-classes_api-tour.zip) |
| [R_Kriging_cross-binding](R_Kriging_cross-binding.ipynb) | `Kriging` | the cross-binding protocol, R side | Python (report) | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-R_Kriging_cross-binding.zip) |
| [R_Kriging_vs-DiceKriging-RobustGaSP](R_Kriging_vs-DiceKriging-RobustGaSP.ipynb) | `Kriging` | default fits (Branin, Hartmann 3, Borehole) | DiceKriging (exact equality), RobustGaSP | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-R_Kriging_vs-DiceKriging-RobustGaSP.zip) |

### Octave (`mLibKriging`)

| notebook | kriging type | theme: problems, features | compared with | executed |
|---|---|---|---|---|
| [octave_all-classes_api-tour](octave_all-classes_api-tour.ipynb) | all classes | the R tour, with the positional arguments of the Octave binding | — | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-octave_all-classes_api-tour.zip) |
| [octave_Kriging_cross-binding](octave_Kriging_cross-binding.ipynb) | `Kriging` | the cross-binding protocol, Octave side | Python (report) | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-octave_Kriging_cross-binding.zip) |
| [octave_Kriging_vs-STK](octave_Kriging_vs-STK.ipynb) | `Kriging` | default fits (Branin, Hartmann 3, Borehole) | STK (exact equality, Gaussian kernel) | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-octave_Kriging_vs-STK.zip) |

### Julia (`JLibKriging`)

| notebook | kriging type | theme: problems, features | compared with | executed |
|---|---|---|---|---|
| [julia_all-classes_api-tour](julia_all-classes_api-tour.ipynb) | all classes | the R tour, with the Julia binding | — | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-julia_all-classes_api-tour.zip) |
| [julia_Kriging_cross-binding](julia_Kriging_cross-binding.ipynb) | `Kriging` | the cross-binding protocol, Julia side | Python (report) | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-julia_Kriging_cross-binding.zip) |
| [julia_Kriging_vs-GaussianProcesses](julia_Kriging_vs-GaussianProcesses.ipynb) | `Kriging` | default fits (Branin, Hartmann 3, Borehole) | GaussianProcesses.jl | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-julia_Kriging_vs-GaussianProcesses.zip) |

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
python tools/run_notebooks.py --inplace python_Kriging_update-simulate.ipynb
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

A new notebook is a new source in `tools/src/` named `<binding>_<kriging type>_<theme>`,
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
