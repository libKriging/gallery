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
[`tools/`](tools), the shared datasets under [`data/`](data), and the Binder
environment under [`.binder/`](.binder).

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

One binding per theme and Kriging type: Python, the most complete binding, unless the
reference is specific to another language (DiceKriging and RobustGaSP in R, STK in
Octave, GaussianProcesses.jl in Julia). The other bindings are checked against
Python on every model class and fit method by the cross-binding notebooks.

### Python (`pylibkriging`)

| notebook | kriging type | theme: problems, features | reference: compared with | executed | open in |
|---|---|---|---|---|---|
| [test-functions-reference-values_no-model_python](test-functions-reference-values_no-model_python.ipynb) | — | the 21 test functions of [`tools/lkbench/functions.py`](tools/lkbench/functions.py), plotted | published optima / values | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-test-functions-reference-values_no-model_python.zip) | [Colab](https://colab.research.google.com/github/libKriging/benchmarks/blob/main/test-functions-reference-values_no-model_python.ipynb) · [Binder](https://mybinder.org/v2/gh/libKriging/benchmarks/main?labpath=test-functions-reference-values_no-model_python.ipynb) |
| [kernels-trends-interpolation_Kriging_python](kernels-trends-interpolation_Kriging_python.ipynb) | `Kriging` | 4 kernels × 4 trends on 8 functions (1D to 3D): interpolation, `predict` with covariance and derivatives | finite differences, Q² floors | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-kernels-trends-interpolation_Kriging_python.zip) | [Colab](https://colab.research.google.com/github/libKriging/benchmarks/blob/main/kernels-trends-interpolation_Kriging_python.ipynb) · [Binder](https://mybinder.org/v2/gh/libKriging/benchmarks/main?labpath=kernels-trends-interpolation_Kriging_python.ipynb) |
| [objectives-LL-LOO-LMP-gradients_Kriging_python](objectives-LL-LOO-LMP-gradients_Kriging_python.ipynb) | `Kriging` | `LL` / `LOO` / `LMP` objectives and gradients, optimality of θ̂, `leaveOneOutVec`, multistart | finite differences, explicit refits | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-objectives-LL-LOO-LMP-gradients_Kriging_python.zip) | [Colab](https://colab.research.google.com/github/libKriging/benchmarks/blob/main/objectives-LL-LOO-LMP-gradients_Kriging_python.ipynb) · [Binder](https://mybinder.org/v2/gh/libKriging/benchmarks/main?labpath=objectives-LL-LOO-LMP-gradients_Kriging_python.ipynb) |
| [update-simulate_Kriging_python](update-simulate_Kriging_python.ipynb) | `Kriging` | `update`, `simulate`, `update_simulate`, conditional paths in pictures | fit from scratch, Monte Carlo moments | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-update-simulate_Kriging_python.zip) | [Colab](https://colab.research.google.com/github/libKriging/benchmarks/blob/main/update-simulate_Kriging_python.ipynb) · [Binder](https://mybinder.org/v2/gh/libKriging/benchmarks/main?labpath=update-simulate_Kriging_python.ipynb) |
| [high-dimension-normalize-screening_Kriging_python](high-dimension-normalize-screening_Kriging_python.ipynb) | `Kriging` | 6D to 20D engineering functions, `normalize`, screening (Welch) | explicit rescaling, known active inputs | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-high-dimension-normalize-screening_Kriging_python.zip) | [Colab](https://colab.research.google.com/github/libKriging/benchmarks/blob/main/high-dimension-normalize-screening_Kriging_python.ipynb) · [Binder](https://mybinder.org/v2/gh/libKriging/benchmarks/main?labpath=high-dimension-normalize-screening_Kriging_python.ipynb) |
| [robustness-invalid-inputs_Kriging_python](robustness-invalid-inputs_Kriging_python.ipynb) | `Kriging` | invalid and degenerate inputs, ill-conditioning, scale equivariance, extrapolation, cost in q | equivariance identities, O(q) cost | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-robustness-invalid-inputs_Kriging_python.zip) | [Colab](https://colab.research.google.com/github/libKriging/benchmarks/blob/main/robustness-invalid-inputs_Kriging_python.ipynb) · [Binder](https://mybinder.org/v2/gh/libKriging/benchmarks/main?labpath=robustness-invalid-inputs_Kriging_python.ipynb) |
| [save-load-pickle-copy-accessors_Kriging_python](save-load-pickle-copy-accessors_Kriging_python.ipynb) | `Kriging` | `save`/`load`, pickle, copy, accessors, for every noise model, objective and fit method | the original model | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-save-load-pickle-copy-accessors_Kriging_python.zip) | [Colab](https://colab.research.google.com/github/libKriging/benchmarks/blob/main/save-load-pickle-copy-accessors_Kriging_python.ipynb) · [Binder](https://mybinder.org/v2/gh/libKriging/benchmarks/main?labpath=save-load-pickle-copy-accessors_Kriging_python.ipynb) |
| [sklearn-estimators_Kriging_python](sklearn-estimators_Kriging_python.ipynb) | `Kriging` | `pylibkriging.sklearn`: `Pipeline`, `GridSearchCV`, `clone`, `sample_y` | the native class | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-sklearn-estimators_Kriging_python.zip) | [Colab](https://colab.research.google.com/github/libKriging/benchmarks/blob/main/sklearn-estimators_Kriging_python.ipynb) · [Binder](https://mybinder.org/v2/gh/libKriging/benchmarks/main?labpath=sklearn-estimators_Kriging_python.ipynb) |
| [vs-sklearn-SMT-OpenTURNS-GPy_Kriging_python](vs-sklearn-SMT-OpenTURNS-GPy_Kriging_python.ipynb) | `Kriging` | default fits on 10 functions, 2D to 10D | scikit-learn, SMT, OpenTURNS, GPy (equality at fixed hyperparameters, accuracy, cost) | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-vs-sklearn-SMT-OpenTURNS-GPy_Kriging_python.zip) | [Colab](https://colab.research.google.com/github/libKriging/benchmarks/blob/main/vs-sklearn-SMT-OpenTURNS-GPy_Kriging_python.ipynb) · [Binder](https://mybinder.org/v2/gh/libKriging/benchmarks/main?labpath=vs-sklearn-SMT-OpenTURNS-GPy_Kriging_python.ipynb) |
| [noisy-data_Kriging-nugget-noise_python](noisy-data_Kriging-nugget-noise_python.ipynb) | `Kriging` with `noise="nugget"` or variances | estimated nugget, heteroscedastic noise, calibration, single start vs multistart | true noise, replicate averaging, interpolating fit | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-noisy-data_Kriging-nugget-noise_python.zip) | [Colab](https://colab.research.google.com/github/libKriging/benchmarks/blob/main/noisy-data_Kriging-nugget-noise_python.ipynb) · [Binder](https://mybinder.org/v2/gh/libKriging/benchmarks/main?labpath=noisy-data_Kriging-nugget-noise_python.ipynb) |
| [large-n_Kriging-vecchia-nystrom_python](large-n_Kriging-vecchia-nystrom_python.ipynb) | `Kriging` with `method="vecchia(m)"` / `"nystrom(k)"` | large n (up to 5000), convergence to the exact model as m, k grow, `subsetOfData` | exact model (limit m = n − 1, k = n), exact fit | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-large-n_Kriging-vecchia-nystrom_python.zip) | [Colab](https://colab.research.google.com/github/libKriging/benchmarks/blob/main/large-n_Kriging-vecchia-nystrom_python.ipynb) · [Binder](https://mybinder.org/v2/gh/libKriging/benchmarks/main?labpath=large-n_Kriging-vecchia-nystrom_python.ipynb) |
| [aggregations-large-n_NestedKriging_python](aggregations-large-n_NestedKriging_python.ipynb) | `NestedKriging` | `NK`, `PoE`, `gPoE`, `BCM`, `rBCM`, from 1 group to 32, large n; save/load, pickle | exact predictor (limit: 1 group), exact fit | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-aggregations-large-n_NestedKriging_python.zip) | [Colab](https://colab.research.google.com/github/libKriging/benchmarks/blob/main/aggregations-large-n_NestedKriging_python.ipynb) · [Binder](https://mybinder.org/v2/gh/libKriging/benchmarks/main?labpath=aggregations-large-n_NestedKriging_python.ipynb) |
| [warpings-categorical-ordinal_WarpKriging_python](warpings-categorical-ordinal_WarpKriging_python.ipynb) | `WarpKriging` | affine, boxcox, kumaraswamy, knots, neural_mono, categorical, ordinal warpings; save/load, pickle | plain `Kriging` | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-warpings-categorical-ordinal_WarpKriging_python.zip) | [Colab](https://colab.research.google.com/github/libKriging/benchmarks/blob/main/warpings-categorical-ordinal_WarpKriging_python.ipynb) · [Binder](https://mybinder.org/v2/gh/libKriging/benchmarks/main?labpath=warpings-categorical-ordinal_WarpKriging_python.ipynb) |
| [deep-kernel-projected-branin_MLPKriging_python](deep-kernel-projected-branin_MLPKriging_python.ipynb) | `MLPKriging` | deep kernel on a 10D function with a 2D structure, activations, multistart; save/load, pickle | plain `Kriging` | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-deep-kernel-projected-branin_MLPKriging_python.zip) | [Colab](https://colab.research.google.com/github/libKriging/benchmarks/blob/main/deep-kernel-projected-branin_MLPKriging_python.ipynb) · [Binder](https://mybinder.org/v2/gh/libKriging/benchmarks/main?labpath=deep-kernel-projected-branin_MLPKriging_python.ipynb) |
| [cross-binding-reference_all-classes_python](cross-binding-reference_all-classes_python.ipynb) | all classes | the cross-binding protocol (all classes and fit methods): the Python reference | the other bindings (report) | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-cross-binding-reference_all-classes_python.zip) | [Colab](https://colab.research.google.com/github/libKriging/benchmarks/blob/main/cross-binding-reference_all-classes_python.ipynb) · [Binder](https://mybinder.org/v2/gh/libKriging/benchmarks/main?labpath=cross-binding-reference_all-classes_python.ipynb) |

### R, Octave, Julia (`rlibkriging`, `mLibKriging`, `JLibKriging`)

| notebook | kriging type | theme: problems, features | reference: compared with | executed | open in |
|---|---|---|---|---|---|
| [vs-DiceKriging-RobustGaSP_Kriging_R](vs-DiceKriging-RobustGaSP_Kriging_R.ipynb) | `Kriging` | rlibkriging: default fits (Branin, Hartmann 3, Borehole) | DiceKriging (equality at fixed hyperparameters), RobustGaSP | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-vs-DiceKriging-RobustGaSP_Kriging_R.zip) | — |
| [vs-STK_Kriging_octave](vs-STK_Kriging_octave.ipynb) | `Kriging` | mLibKriging: default fits (Branin, Hartmann 3, Borehole) | STK (equality at fixed hyperparameters, Gaussian kernel) | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-vs-STK_Kriging_octave.zip) | — |
| [vs-GaussianProcesses_Kriging_julia](vs-GaussianProcesses_Kriging_julia.ipynb) | `Kriging` | JLibKriging: default fits (Branin, Hartmann 3, Borehole) | GaussianProcesses.jl | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-vs-GaussianProcesses_Kriging_julia.zip) | — |
| [cross-binding_all-classes_R](cross-binding_all-classes_R.ipynb) | all classes | the cross-binding protocol, R side | Python (report) | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-cross-binding_all-classes_R.zip) | — |
| [cross-binding_all-classes_octave](cross-binding_all-classes_octave.ipynb) | all classes | the cross-binding protocol, Octave side | Python (report) | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-cross-binding_all-classes_octave.zip) | — |
| [cross-binding_all-classes_julia](cross-binding_all-classes_julia.ipynb) | all classes | the cross-binding protocol, Julia side | Python (report) | [zip](https://nightly.link/libKriging/benchmarks/workflows/notebooks/main/executed-master-cross-binding_all-classes_julia.zip) | — |

A notebook that compares libKriging with another package installs that package
itself, in its first cells (pip, CRAN, `Pkg.add`, or a clone of STK); a package
that cannot be installed is reported (warn-level) and its comparison skipped.

Test functions: [`tools/lkbench/functions.py`](tools/lkbench/functions.py), after the
[Virtual Library of Simulation Experiments](https://www.sfu.ca/~ssurjano/)
(Surjanovic & Bingham).

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
plain diffs, and get the setup cell of `tools/notebook_setup.py` (Python); outputs are not committed (CI archives the executed notebooks as
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
