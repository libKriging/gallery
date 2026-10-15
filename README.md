# libKriging gallery

Worked examples of [libKriging](https://github.com/libKriging/libKriging), one per
use case: every model class (`Kriging` with its noise models and approximate fit
methods, `NestedKriging`, `WarpKriging`, `MLPKriging`) on classical test problems,
explained, plotted and tabulated, next to the packages it is usually compared with.

Each example is also a **validation**: what it shows is checked against a reference
(finite differences, a fit from scratch, the exact model an approximation converges
to, another kriging package at the same hyperparameters, the other bindings).
CI runs every notebook against the **development head of libKriging and of its
bindings** (and weekly against the releases), saves it with its outputs, and turns
the checks into a report: the gallery is libKriging's end-to-end acceptance test.

Every notebook starts with its **idea** and its **reference**, and ends with a
**summary** of its checks; error-level checks fail the run, warn-level ones are
reported (see [Checks and statuses](#checks-and-statuses)).

## The examples

**executed**: the notebook as run by the last CI run on `main` against libKriging
master, with its outputs (a zip of the `executed-master-<notebook>` artifact through
[nightly.link](https://nightly.link), which serves the last artifact of a branch at a
stable URL; [all runs on `main`](https://github.com/libKriging/gallery/actions/workflows/notebooks.yml?query=branch%3Amain)
keep them 90 days; all of a run at once:
[executed-notebooks-master](https://nightly.link/libKriging/gallery/workflows/notebooks/main/executed-notebooks-master.zip),
and the aggregated report: [report-master](https://nightly.link/libKriging/gallery/workflows/notebooks/main/report-master.zip)).
**open in**: run it in a free hosted service (see [below](#running-a-notebook-in-colab-or-binder)).

One binding per use case: Python, the most complete binding, unless the reference
belongs to another language (DiceKriging and RobustGaSP in R, STK in Octave,
GaussianProcesses.jl in Julia); the cross-binding notebooks check that every
binding gives the same numbers as Python, for every class and fit method.

### Kriging: the basics

| notebook | binding | kriging type | what it shows | validated against | executed | open in |
|---|---|---|---|---|---|---|
| [update-simulate_Kriging_python](update-simulate_Kriging_python.ipynb) | python | `Kriging` | the workflow on 1D to 6D functions: fit, `predict`, conditional paths, `update` with new data, `update_simulate`, in pictures | fit from scratch, Monte Carlo moments | [zip](https://nightly.link/libKriging/gallery/workflows/notebooks/main/executed-master-update-simulate_Kriging_python.zip) | [Colab](https://colab.research.google.com/github/libKriging/gallery/blob/main/update-simulate_Kriging_python.ipynb) · [Binder](https://mybinder.org/v2/gh/libKriging/gallery/main?labpath=update-simulate_Kriging_python.ipynb) |
| [kernels-trends-interpolation_Kriging_python](kernels-trends-interpolation_Kriging_python.ipynb) | python | `Kriging` | choosing a kernel and a trend: 4 × 4 combinations on 8 functions, interpolation, predicted covariance and derivatives | finite differences, Q² floors | [zip](https://nightly.link/libKriging/gallery/workflows/notebooks/main/executed-master-kernels-trends-interpolation_Kriging_python.zip) | [Colab](https://colab.research.google.com/github/libKriging/gallery/blob/main/kernels-trends-interpolation_Kriging_python.ipynb) · [Binder](https://mybinder.org/v2/gh/libKriging/gallery/main?labpath=kernels-trends-interpolation_Kriging_python.ipynb) |
| [objectives-LL-LOO-LMP-gradients_Kriging_python](objectives-LL-LOO-LMP-gradients_Kriging_python.ipynb) | python | `Kriging` | choosing the estimation criterion (`LL`, `LOO`, `LMP`), its profile around θ̂, multistart | finite differences, explicit leave-one-out refits | [zip](https://nightly.link/libKriging/gallery/workflows/notebooks/main/executed-master-objectives-LL-LOO-LMP-gradients_Kriging_python.zip) | [Colab](https://colab.research.google.com/github/libKriging/gallery/blob/main/objectives-LL-LOO-LMP-gradients_Kriging_python.ipynb) · [Binder](https://mybinder.org/v2/gh/libKriging/gallery/main?labpath=objectives-LL-LOO-LMP-gradients_Kriging_python.ipynb) |
| [high-dimension-normalize-screening_Kriging_python](high-dimension-normalize-screening_Kriging_python.ipynb) | python | `Kriging` | 6 to 20 inputs in physical units: `normalize`, reading the fitted θ to screen inputs (Welch) | explicit rescaling, known active inputs | [zip](https://nightly.link/libKriging/gallery/workflows/notebooks/main/executed-master-high-dimension-normalize-screening_Kriging_python.zip) | [Colab](https://colab.research.google.com/github/libKriging/gallery/blob/main/high-dimension-normalize-screening_Kriging_python.ipynb) · [Binder](https://mybinder.org/v2/gh/libKriging/gallery/main?labpath=high-dimension-normalize-screening_Kriging_python.ipynb) |
| [save-load-pickle-copy-accessors_Kriging_python](save-load-pickle-copy-accessors_Kriging_python.ipynb) | python | `Kriging` | saving, loading, pickling, copying a model; reading its settings back | the original model (to 1e-12) | [zip](https://nightly.link/libKriging/gallery/workflows/notebooks/main/executed-master-save-load-pickle-copy-accessors_Kriging_python.zip) | [Colab](https://colab.research.google.com/github/libKriging/gallery/blob/main/save-load-pickle-copy-accessors_Kriging_python.ipynb) · [Binder](https://mybinder.org/v2/gh/libKriging/gallery/main?labpath=save-load-pickle-copy-accessors_Kriging_python.ipynb) |
| [sklearn-estimators_Kriging_python](sklearn-estimators_Kriging_python.ipynb) | python | `Kriging` | libKriging inside scikit-learn: `Pipeline`, `GridSearchCV` over kernels, `cross_val_score`, `sample_y` | the native class | [zip](https://nightly.link/libKriging/gallery/workflows/notebooks/main/executed-master-sklearn-estimators_Kriging_python.zip) | [Colab](https://colab.research.google.com/github/libKriging/gallery/blob/main/sklearn-estimators_Kriging_python.ipynb) · [Binder](https://mybinder.org/v2/gh/libKriging/gallery/main?labpath=sklearn-estimators_Kriging_python.ipynb) |

### Noisy observations

| notebook | binding | kriging type | what it shows | validated against | executed | open in |
|---|---|---|---|---|---|---|
| [noisy-data_Kriging-nugget-noise_python](noisy-data_Kriging-nugget-noise_python.ipynb) | python | `Kriging` with `noise="nugget"` or variances | estimated nugget vs known noise variances, replicates, calibrated intervals on the noise-free function | true noise, replicate averaging, interpolating fit | [zip](https://nightly.link/libKriging/gallery/workflows/notebooks/main/executed-master-noisy-data_Kriging-nugget-noise_python.zip) | [Colab](https://colab.research.google.com/github/libKriging/gallery/blob/main/noisy-data_Kriging-nugget-noise_python.ipynb) · [Binder](https://mybinder.org/v2/gh/libKriging/gallery/main?labpath=noisy-data_Kriging-nugget-noise_python.ipynb) |

### Large datasets

| notebook | binding | kriging type | what it shows | validated against | executed | open in |
|---|---|---|---|---|---|---|
| [large-n_Kriging-vecchia-nystrom_python](large-n_Kriging-vecchia-nystrom_python.ipynb) | python | `Kriging` with `method="vecchia(m)"` / `"nystrom(k)"` | n up to 5000: Vecchia and Nyström fits, their convergence to the exact model as m and k grow, `subsetOfData` | exact model (limits m = n − 1, k = n), exact fit | [zip](https://nightly.link/libKriging/gallery/workflows/notebooks/main/executed-master-large-n_Kriging-vecchia-nystrom_python.zip) | [Colab](https://colab.research.google.com/github/libKriging/gallery/blob/main/large-n_Kriging-vecchia-nystrom_python.ipynb) · [Binder](https://mybinder.org/v2/gh/libKriging/gallery/main?labpath=large-n_Kriging-vecchia-nystrom_python.ipynb) |
| [aggregations-large-n_NestedKriging_python](aggregations-large-n_NestedKriging_python.ipynb) | python | `NestedKriging` | splitting a design into groups: `NK`, `PoE`, `gPoE`, `BCM`, `rBCM`, from 1 group to 32 | exact predictor (limit: 1 group), exact fit | [zip](https://nightly.link/libKriging/gallery/workflows/notebooks/main/executed-master-aggregations-large-n_NestedKriging_python.zip) | [Colab](https://colab.research.google.com/github/libKriging/gallery/blob/main/aggregations-large-n_NestedKriging_python.ipynb) · [Binder](https://mybinder.org/v2/gh/libKriging/gallery/main?labpath=aggregations-large-n_NestedKriging_python.ipynb) |

### Non-stationary, categorical and ordinal inputs

| notebook | binding | kriging type | what it shows | validated against | executed | open in |
|---|---|---|---|---|---|---|
| [warpings-categorical-ordinal_WarpKriging_python](warpings-categorical-ordinal_WarpKriging_python.ipynb) | python | `WarpKriging` | input warpings (affine, boxcox, kumaraswamy, knots, neural_mono), categorical and ordinal inputs | plain `Kriging` (accuracy, NLPD), `none` warping = `Kriging` | [zip](https://nightly.link/libKriging/gallery/workflows/notebooks/main/executed-master-warpings-categorical-ordinal_WarpKriging_python.zip) | [Colab](https://colab.research.google.com/github/libKriging/gallery/blob/main/warpings-categorical-ordinal_WarpKriging_python.ipynb) · [Binder](https://mybinder.org/v2/gh/libKriging/gallery/main?labpath=warpings-categorical-ordinal_WarpKriging_python.ipynb) |

### Deep kernels

| notebook | binding | kriging type | what it shows | validated against | executed | open in |
|---|---|---|---|---|---|---|
| [deep-kernel-projected-branin_MLPKriging_python](deep-kernel-projected-branin_MLPKriging_python.ipynb) | python | `MLPKriging` | an MLP feature map finding the 2D structure of a 10D function; activations, multistart | plain `Kriging` | [zip](https://nightly.link/libKriging/gallery/workflows/notebooks/main/executed-master-deep-kernel-projected-branin_MLPKriging_python.zip) | [Colab](https://colab.research.google.com/github/libKriging/gallery/blob/main/deep-kernel-projected-branin_MLPKriging_python.ipynb) · [Binder](https://mybinder.org/v2/gh/libKriging/gallery/main?labpath=deep-kernel-projected-branin_MLPKriging_python.ipynb) |

### libKriging and other kriging packages

| notebook | binding | kriging type | what it shows | validated against | executed | open in |
|---|---|---|---|---|---|---|
| [vs-sklearn-SMT-OpenTURNS-GPy_Kriging_python](vs-sklearn-SMT-OpenTURNS-GPy_Kriging_python.ipynb) | python | `Kriging` | same hyperparameters → same numbers, then default fits on 10 functions (accuracy, calibration, cost) | scikit-learn, SMT, OpenTURNS, GPy | [zip](https://nightly.link/libKriging/gallery/workflows/notebooks/main/executed-master-vs-sklearn-SMT-OpenTURNS-GPy_Kriging_python.zip) | [Colab](https://colab.research.google.com/github/libKriging/gallery/blob/main/vs-sklearn-SMT-OpenTURNS-GPy_Kriging_python.ipynb) · [Binder](https://mybinder.org/v2/gh/libKriging/gallery/main?labpath=vs-sklearn-SMT-OpenTURNS-GPy_Kriging_python.ipynb) |
| [vs-DiceKriging-RobustGaSP_Kriging_R](vs-DiceKriging-RobustGaSP_Kriging_R.ipynb) | R | `Kriging` | rlibkriging next to the R references: equality at fixed hyperparameters, default fits | DiceKriging, RobustGaSP | [zip](https://nightly.link/libKriging/gallery/workflows/notebooks/main/executed-master-vs-DiceKriging-RobustGaSP_Kriging_R.zip) | — |
| [vs-STK_Kriging_octave](vs-STK_Kriging_octave.ipynb) | octave | `Kriging` | mLibKriging next to STK: equality at fixed hyperparameters (Gaussian kernel), default fits | STK | [zip](https://nightly.link/libKriging/gallery/workflows/notebooks/main/executed-master-vs-STK_Kriging_octave.zip) | — |
| [vs-GaussianProcesses_Kriging_julia](vs-GaussianProcesses_Kriging_julia.ipynb) | julia | `Kriging` | JLibKriging next to GaussianProcesses.jl: default fits | GaussianProcesses.jl | [zip](https://nightly.link/libKriging/gallery/workflows/notebooks/main/executed-master-vs-GaussianProcesses_Kriging_julia.zip) | — |

### Validation only

| notebook | binding | kriging type | what it shows | validated against | executed | open in |
|---|---|---|---|---|---|---|
| [test-functions-reference-values_no-model_python](test-functions-reference-values_no-model_python.ipynb) | python | — | the 21 test functions used throughout, plotted and checked | published optima / values | [zip](https://nightly.link/libKriging/gallery/workflows/notebooks/main/executed-master-test-functions-reference-values_no-model_python.zip) | [Colab](https://colab.research.google.com/github/libKriging/gallery/blob/main/test-functions-reference-values_no-model_python.ipynb) · [Binder](https://mybinder.org/v2/gh/libKriging/gallery/main?labpath=test-functions-reference-values_no-model_python.ipynb) |
| [robustness-invalid-inputs_Kriging_python](robustness-invalid-inputs_Kriging_python.ipynb) | python | `Kriging` | invalid and degenerate inputs, ill-conditioning, units, extrapolation, prediction cost in q | equivariance identities, O(q) cost | [zip](https://nightly.link/libKriging/gallery/workflows/notebooks/main/executed-master-robustness-invalid-inputs_Kriging_python.zip) | [Colab](https://colab.research.google.com/github/libKriging/gallery/blob/main/robustness-invalid-inputs_Kriging_python.ipynb) · [Binder](https://mybinder.org/v2/gh/libKriging/gallery/main?labpath=robustness-invalid-inputs_Kriging_python.ipynb) |
| [cross-binding-reference_all-classes_python](cross-binding-reference_all-classes_python.ipynb) | python | all classes | the same data and options in every binding, every class and fit method: the Python values | R, Octave, Julia (report) | [zip](https://nightly.link/libKriging/gallery/workflows/notebooks/main/executed-master-cross-binding-reference_all-classes_python.zip) | [Colab](https://colab.research.google.com/github/libKriging/gallery/blob/main/cross-binding-reference_all-classes_python.ipynb) · [Binder](https://mybinder.org/v2/gh/libKriging/gallery/main?labpath=cross-binding-reference_all-classes_python.ipynb) |
| [cross-binding_all-classes_R](cross-binding_all-classes_R.ipynb) | R | all classes | the R side of the cross-binding protocol | Python (report) | [zip](https://nightly.link/libKriging/gallery/workflows/notebooks/main/executed-master-cross-binding_all-classes_R.zip) | — |
| [cross-binding_all-classes_octave](cross-binding_all-classes_octave.ipynb) | octave | all classes | the Octave side | Python (report) | [zip](https://nightly.link/libKriging/gallery/workflows/notebooks/main/executed-master-cross-binding_all-classes_octave.zip) | — |
| [cross-binding_all-classes_julia](cross-binding_all-classes_julia.ipynb) | julia | all classes | the Julia side | Python (report) | [zip](https://nightly.link/libKriging/gallery/workflows/notebooks/main/executed-master-cross-binding_all-classes_julia.zip) | — |

Notebooks are named `<theme>_<kriging type>_<binding>.ipynb`. Those comparing
with another package install it themselves, in their first cells (pip, CRAN,
`Pkg.add`, or a clone of STK); a package that cannot be installed is reported
(warn-level) and its comparison skipped. Test functions:
[`tools/lkgallery/functions.py`](tools/lkgallery/functions.py), after the
[Virtual Library of Simulation Experiments](https://www.sfu.ca/~ssurjano/)
(Surjanovic & Bingham). Everything that is not a notebook is under
[`tools/`](tools) (sources of the notebooks, helpers, installation of the bindings,
CI scripts, known issues, baselines), the shared datasets under [`data/`](data),
the Binder environment under [`.binder/`](.binder).

## Running a notebook in Colab or Binder

The **open in** links run a Python notebook in a free hosted service, against the
**released** pylibkriging (from PyPI), not against libKriging master as CI does: a
feature not released yet is reported as SKIP. The first code cell of each Python
notebook ([`tools/notebook_setup.py`](tools/notebook_setup.py), a no-op in this
repository) clones the repository for its datasets and helpers and installs
pylibkriging when missing; the notebooks that compare with other packages install
them themselves.

* **Colab**: PyPI has pylibkriging wheels for CPython 3.10 to 3.12 only, and Colab's
  default runtime has used Python 3.13 since August-September 2026: choose
  *Runtime > Change runtime version > 2026.07* (Python 3.12) before running,
  otherwise the setup cell stops with this message.
* **Binder**: the environment of [`.binder/`](.binder) (Python 3.12, pylibkriging);
  the first start builds the image, which takes a few minutes.

Not offered for the R, Octave and Julia notebooks: Octave is not available on Colab,
and the installation of rlibkriging (compiled from CRAN sources) and of JLibKriging.jl
(which compiles libKriging) on these services has not been checked.

## Checks and statuses

Each notebook records its checks with `lkgallery.Suite` (Python) or the small
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
plain diffs, and get the setup cell of `tools/notebook_setup.py` (Python); outputs
are not committed (CI archives the executed notebooks as artifacts, see
[The examples](#the-examples)).

```sh
python tools/py2nb.py           # regenerate every notebook from tools/src/
python tools/py2nb.py --check   # CI: sources and notebooks in sync, no notebook without source
```

A new example is a new source in `tools/src/` named `<theme>_<kriging type>_<binding>`,
with its suite named the same, an **Idea** and a **Reference** paragraph after its
title, plots and tables of its results, its checks, a final `## Summary` cell, and a
line in the tables above (CI checks that
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
