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

Every notebook starts with its **idea**, its **reference** and its **tags**, and ends
with a **summary** of its checks; error-level checks fail the run, warn-level ones are
reported (see [Checks and statuses](#checks-and-statuses)).

**Gallery site: <https://libkriging.github.io/gallery/>**: one card per notebook
(first figure, title, idea, tags: language, Kriging class, kernel, objective, method,
noise/nugget, warping, aggregation, compared packages; status of its checks), tag
filters; a card opens its notebook, executed by the last CI run on `main`, in nbviewer
(Python cards also link to Colab and Binder). Built by [`tools/site.py`](tools/site.py),
published on the `gh-pages` branch.

## The examples

**executed**: the notebook as run by the last CI run on `main` against libKriging
master, with its outputs (a zip of the `executed-master-<notebook>` artifact through
[nightly.link](https://nightly.link), which serves the last artifact of a branch at a
stable URL; [all runs on `main`](https://github.com/libKriging/gallery/actions/workflows/notebooks.yml?query=branch%3Amain)
keep them 90 days; all of a run at once:
[executed-notebooks-master](https://nightly.link/libKriging/gallery/workflows/notebooks/main/executed-notebooks-master.zip),
and the aggregated report: [report-master](https://nightly.link/libKriging/gallery/workflows/notebooks/main/report-master.zip)).
**open in**: the executed notebook of the site, run in a free hosted service (see
[below](#running-a-notebook-in-colab-or-binder)), or read in nbviewer. The
**notebook** column links to its source in [`src/`](src): the notebooks themselves
are generated and executed by CI only, never committed.

One binding per use case: Python, the most complete binding, unless the reference
belongs to another language (DiceKriging and RobustGaSP in R, STK in Octave,
GaussianProcesses.jl in Julia); the cross-binding notebooks check that every
binding gives the same numbers as Python, for every class and fit method.

### Kriging: the basics

| notebook | binding | kriging type | what it shows | validated against | executed | open in |
|---|---|---|---|---|---|---|
| [update-simulate_Kriging_python](src/update-simulate_Kriging_python.py) | python | `Kriging` | the workflow on 1D to 6D functions: fit, `predict`, conditional paths, `update` with new data, `update_simulate`, in pictures | fit from scratch, Monte Carlo moments | [zip](https://nightly.link/libKriging/gallery/workflows/notebooks/main/executed-master-update-simulate_Kriging_python.zip) | [Colab](https://colab.research.google.com/github/libKriging/gallery/blob/gh-pages/notebooks/update-simulate_Kriging_python.ipynb) · [Binder](https://mybinder.org/v2/gh/libKriging/gallery/gh-pages?labpath=notebooks%2Fupdate-simulate_Kriging_python.ipynb) |
| [kernels-trends-interpolation_Kriging_python](src/kernels-trends-interpolation_Kriging_python.py) | python | `Kriging` | choosing a kernel and a trend: 4 × 4 combinations on 8 functions, interpolation, predicted covariance and derivatives | finite differences, Q² floors | [zip](https://nightly.link/libKriging/gallery/workflows/notebooks/main/executed-master-kernels-trends-interpolation_Kriging_python.zip) | [Colab](https://colab.research.google.com/github/libKriging/gallery/blob/gh-pages/notebooks/kernels-trends-interpolation_Kriging_python.ipynb) · [Binder](https://mybinder.org/v2/gh/libKriging/gallery/gh-pages?labpath=notebooks%2Fkernels-trends-interpolation_Kriging_python.ipynb) |
| [objectives-LL-LOO-LMP-gradients_Kriging_python](src/objectives-LL-LOO-LMP-gradients_Kriging_python.py) | python | `Kriging` | choosing the estimation criterion (`LL`, `LOO`, `LMP`), its profile around θ̂, multistart | finite differences, explicit leave-one-out refits | [zip](https://nightly.link/libKriging/gallery/workflows/notebooks/main/executed-master-objectives-LL-LOO-LMP-gradients_Kriging_python.zip) | [Colab](https://colab.research.google.com/github/libKriging/gallery/blob/gh-pages/notebooks/objectives-LL-LOO-LMP-gradients_Kriging_python.ipynb) · [Binder](https://mybinder.org/v2/gh/libKriging/gallery/gh-pages?labpath=notebooks%2Fobjectives-LL-LOO-LMP-gradients_Kriging_python.ipynb) |
| [high-dimension-normalize-screening_Kriging_python](src/high-dimension-normalize-screening_Kriging_python.py) | python | `Kriging` | 6 to 20 inputs in physical units: `normalize`, reading the fitted θ to screen inputs (Welch) | explicit rescaling, known active inputs | [zip](https://nightly.link/libKriging/gallery/workflows/notebooks/main/executed-master-high-dimension-normalize-screening_Kriging_python.zip) | [Colab](https://colab.research.google.com/github/libKriging/gallery/blob/gh-pages/notebooks/high-dimension-normalize-screening_Kriging_python.ipynb) · [Binder](https://mybinder.org/v2/gh/libKriging/gallery/gh-pages?labpath=notebooks%2Fhigh-dimension-normalize-screening_Kriging_python.ipynb) |
| [save-load-pickle-copy-accessors_Kriging_python](src/save-load-pickle-copy-accessors_Kriging_python.py) | python | `Kriging` | saving, loading, pickling, copying a model; reading its settings back | the original model (to 1e-12) | [zip](https://nightly.link/libKriging/gallery/workflows/notebooks/main/executed-master-save-load-pickle-copy-accessors_Kriging_python.zip) | [Colab](https://colab.research.google.com/github/libKriging/gallery/blob/gh-pages/notebooks/save-load-pickle-copy-accessors_Kriging_python.ipynb) · [Binder](https://mybinder.org/v2/gh/libKriging/gallery/gh-pages?labpath=notebooks%2Fsave-load-pickle-copy-accessors_Kriging_python.ipynb) |
| [sklearn-estimators_Kriging_python](src/sklearn-estimators_Kriging_python.py) | python | `Kriging` | libKriging inside scikit-learn: `Pipeline`, `GridSearchCV` over kernels, `cross_val_score`, `sample_y` | the native class | [zip](https://nightly.link/libKriging/gallery/workflows/notebooks/main/executed-master-sklearn-estimators_Kriging_python.zip) | [Colab](https://colab.research.google.com/github/libKriging/gallery/blob/gh-pages/notebooks/sklearn-estimators_Kriging_python.ipynb) · [Binder](https://mybinder.org/v2/gh/libKriging/gallery/gh-pages?labpath=notebooks%2Fsklearn-estimators_Kriging_python.ipynb) |

### Noisy observations

| notebook | binding | kriging type | what it shows | validated against | executed | open in |
|---|---|---|---|---|---|---|
| [noisy-data_Kriging-nugget-noise_python](src/noisy-data_Kriging-nugget-noise_python.py) | python | `Kriging` with `noise="nugget"` or variances | estimated nugget vs known noise variances, replicates, calibrated intervals on the noise-free function | true noise, replicate averaging, interpolating fit | [zip](https://nightly.link/libKriging/gallery/workflows/notebooks/main/executed-master-noisy-data_Kriging-nugget-noise_python.zip) | [Colab](https://colab.research.google.com/github/libKriging/gallery/blob/gh-pages/notebooks/noisy-data_Kriging-nugget-noise_python.ipynb) · [Binder](https://mybinder.org/v2/gh/libKriging/gallery/gh-pages?labpath=notebooks%2Fnoisy-data_Kriging-nugget-noise_python.ipynb) |

### Large datasets

| notebook | binding | kriging type | what it shows | validated against | executed | open in |
|---|---|---|---|---|---|---|
| [large-n_Kriging-vecchia-nystrom_python](src/large-n_Kriging-vecchia-nystrom_python.py) | python | `Kriging` with `method="vecchia(m)"` / `"nystrom(k)"` | n up to 5000: Vecchia and Nyström fits, their convergence to the exact model as m and k grow, `subsetOfData` | exact model (limits m = n − 1, k = n), exact fit | [zip](https://nightly.link/libKriging/gallery/workflows/notebooks/main/executed-master-large-n_Kriging-vecchia-nystrom_python.zip) | [Colab](https://colab.research.google.com/github/libKriging/gallery/blob/gh-pages/notebooks/large-n_Kriging-vecchia-nystrom_python.ipynb) · [Binder](https://mybinder.org/v2/gh/libKriging/gallery/gh-pages?labpath=notebooks%2Flarge-n_Kriging-vecchia-nystrom_python.ipynb) |
| [aggregations-large-n_NestedKriging_python](src/aggregations-large-n_NestedKriging_python.py) | python | `NestedKriging` | splitting a design into groups: `NK`, `PoE`, `gPoE`, `BCM`, `rBCM`, from 1 group to 32 | exact predictor (limit: 1 group), exact fit | [zip](https://nightly.link/libKriging/gallery/workflows/notebooks/main/executed-master-aggregations-large-n_NestedKriging_python.zip) | [Colab](https://colab.research.google.com/github/libKriging/gallery/blob/gh-pages/notebooks/aggregations-large-n_NestedKriging_python.ipynb) · [Binder](https://mybinder.org/v2/gh/libKriging/gallery/gh-pages?labpath=notebooks%2Faggregations-large-n_NestedKriging_python.ipynb) |

### Non-stationary, categorical and ordinal inputs

| notebook | binding | kriging type | what it shows | validated against | executed | open in |
|---|---|---|---|---|---|---|
| [warpings-categorical-ordinal_WarpKriging_python](src/warpings-categorical-ordinal_WarpKriging_python.py) | python | `WarpKriging` | input warpings (affine, boxcox, kumaraswamy, knots, neural_mono), categorical and ordinal inputs | plain `Kriging` (accuracy, NLPD), `none` warping = `Kriging` | [zip](https://nightly.link/libKriging/gallery/workflows/notebooks/main/executed-master-warpings-categorical-ordinal_WarpKriging_python.zip) | [Colab](https://colab.research.google.com/github/libKriging/gallery/blob/gh-pages/notebooks/warpings-categorical-ordinal_WarpKriging_python.ipynb) · [Binder](https://mybinder.org/v2/gh/libKriging/gallery/gh-pages?labpath=notebooks%2Fwarpings-categorical-ordinal_WarpKriging_python.ipynb) |

### Deep kernels

| notebook | binding | kriging type | what it shows | validated against | executed | open in |
|---|---|---|---|---|---|---|
| [deep-kernel-projected-branin_MLPKriging_python](src/deep-kernel-projected-branin_MLPKriging_python.py) | python | `MLPKriging` | an MLP feature map finding the 2D structure of a 10D function; activations, multistart | plain `Kriging` | [zip](https://nightly.link/libKriging/gallery/workflows/notebooks/main/executed-master-deep-kernel-projected-branin_MLPKriging_python.zip) | [Colab](https://colab.research.google.com/github/libKriging/gallery/blob/gh-pages/notebooks/deep-kernel-projected-branin_MLPKriging_python.ipynb) · [Binder](https://mybinder.org/v2/gh/libKriging/gallery/gh-pages?labpath=notebooks%2Fdeep-kernel-projected-branin_MLPKriging_python.ipynb) |

### libKriging and other kriging packages

| notebook | binding | kriging type | what it shows | validated against | executed | open in |
|---|---|---|---|---|---|---|
| [vs-sklearn-SMT-OpenTURNS-GPy_Kriging_python](src/vs-sklearn-SMT-OpenTURNS-GPy_Kriging_python.py) | python | `Kriging` | same hyperparameters → same numbers, then default fits on 10 functions (accuracy, calibration, cost) | scikit-learn, SMT, OpenTURNS, GPy | [zip](https://nightly.link/libKriging/gallery/workflows/notebooks/main/executed-master-vs-sklearn-SMT-OpenTURNS-GPy_Kriging_python.zip) | [Colab](https://colab.research.google.com/github/libKriging/gallery/blob/gh-pages/notebooks/vs-sklearn-SMT-OpenTURNS-GPy_Kriging_python.ipynb) · [Binder](https://mybinder.org/v2/gh/libKriging/gallery/gh-pages?labpath=notebooks%2Fvs-sklearn-SMT-OpenTURNS-GPy_Kriging_python.ipynb) |
| [vs-DiceKriging-RobustGaSP_Kriging_R](src/vs-DiceKriging-RobustGaSP_Kriging_R.R) | R | `Kriging` | rlibkriging next to the R references: equality at fixed hyperparameters, default fits | DiceKriging, RobustGaSP | [zip](https://nightly.link/libKriging/gallery/workflows/notebooks/main/executed-master-vs-DiceKriging-RobustGaSP_Kriging_R.zip) | [nbviewer](https://nbviewer.org/github/libKriging/gallery/blob/gh-pages/notebooks/vs-DiceKriging-RobustGaSP_Kriging_R.ipynb) |
| [vs-STK_Kriging_octave](src/vs-STK_Kriging_octave.m) | octave | `Kriging` | mLibKriging next to STK: equality at fixed hyperparameters (Gaussian kernel), default fits | STK | [zip](https://nightly.link/libKriging/gallery/workflows/notebooks/main/executed-master-vs-STK_Kriging_octave.zip) | [nbviewer](https://nbviewer.org/github/libKriging/gallery/blob/gh-pages/notebooks/vs-STK_Kriging_octave.ipynb) |
| [vs-GaussianProcesses_Kriging_julia](src/vs-GaussianProcesses_Kriging_julia.jl) | julia | `Kriging` | JLibKriging next to GaussianProcesses.jl: default fits | GaussianProcesses.jl | [zip](https://nightly.link/libKriging/gallery/workflows/notebooks/main/executed-master-vs-GaussianProcesses_Kriging_julia.zip) | [nbviewer](https://nbviewer.org/github/libKriging/gallery/blob/gh-pages/notebooks/vs-GaussianProcesses_Kriging_julia.ipynb) |

### Validation only

| notebook | binding | kriging type | what it shows | validated against | executed | open in |
|---|---|---|---|---|---|---|
| [test-functions-reference-values_no-model_python](src/test-functions-reference-values_no-model_python.py) | python | — | the 21 test functions used throughout, plotted and checked | published optima / values | [zip](https://nightly.link/libKriging/gallery/workflows/notebooks/main/executed-master-test-functions-reference-values_no-model_python.zip) | [Colab](https://colab.research.google.com/github/libKriging/gallery/blob/gh-pages/notebooks/test-functions-reference-values_no-model_python.ipynb) · [Binder](https://mybinder.org/v2/gh/libKriging/gallery/gh-pages?labpath=notebooks%2Ftest-functions-reference-values_no-model_python.ipynb) |
| [robustness-invalid-inputs_Kriging_python](src/robustness-invalid-inputs_Kriging_python.py) | python | `Kriging` | invalid and degenerate inputs, ill-conditioning, units, extrapolation, prediction cost in q | equivariance identities, O(q) cost | [zip](https://nightly.link/libKriging/gallery/workflows/notebooks/main/executed-master-robustness-invalid-inputs_Kriging_python.zip) | [Colab](https://colab.research.google.com/github/libKriging/gallery/blob/gh-pages/notebooks/robustness-invalid-inputs_Kriging_python.ipynb) · [Binder](https://mybinder.org/v2/gh/libKriging/gallery/gh-pages?labpath=notebooks%2Frobustness-invalid-inputs_Kriging_python.ipynb) |
| [cross-binding-reference_all-classes_python](src/cross-binding-reference_all-classes_python.py) | python | all classes | the same data and options in every binding, every class and fit method: the Python values | R, Octave, Julia (report) | [zip](https://nightly.link/libKriging/gallery/workflows/notebooks/main/executed-master-cross-binding-reference_all-classes_python.zip) | [Colab](https://colab.research.google.com/github/libKriging/gallery/blob/gh-pages/notebooks/cross-binding-reference_all-classes_python.ipynb) · [Binder](https://mybinder.org/v2/gh/libKriging/gallery/gh-pages?labpath=notebooks%2Fcross-binding-reference_all-classes_python.ipynb) |
| [cross-binding_all-classes_R](src/cross-binding_all-classes_R.R) | R | all classes | the R side of the cross-binding protocol | Python (report) | [zip](https://nightly.link/libKriging/gallery/workflows/notebooks/main/executed-master-cross-binding_all-classes_R.zip) | [nbviewer](https://nbviewer.org/github/libKriging/gallery/blob/gh-pages/notebooks/cross-binding_all-classes_R.ipynb) |
| [cross-binding_all-classes_octave](src/cross-binding_all-classes_octave.m) | octave | all classes | the Octave side | Python (report) | [zip](https://nightly.link/libKriging/gallery/workflows/notebooks/main/executed-master-cross-binding_all-classes_octave.zip) | [nbviewer](https://nbviewer.org/github/libKriging/gallery/blob/gh-pages/notebooks/cross-binding_all-classes_octave.ipynb) |
| [cross-binding_all-classes_julia](src/cross-binding_all-classes_julia.jl) | julia | all classes | the Julia side | Python (report) | [zip](https://nightly.link/libKriging/gallery/workflows/notebooks/main/executed-master-cross-binding_all-classes_julia.zip) | [nbviewer](https://nbviewer.org/github/libKriging/gallery/blob/gh-pages/notebooks/cross-binding_all-classes_julia.ipynb) |

Notebooks are named `<theme>_<kriging type>_<binding>`. Those comparing
with another package install it themselves, in their first cells (pip, CRAN,
`Pkg.add`, or a clone of STK); a package that cannot be installed is reported
(warn-level) and its comparison skipped. Test functions:
[`tools/python/gallery.py`](tools/python/gallery.py) (section 1), after the
[Virtual Library of Simulation Experiments](https://www.sfu.ca/~ssurjano/)
(Surjanovic & Bingham). The notebook sources are in [`src/`](src); everything else is under
[`tools/`](tools) (helpers, notebook generation, installation of the bindings,
CI scripts, site generator, known issues, baselines), the shared datasets under
[`data/`](data), the Binder environment under [`.binder/`](.binder).

## Running a notebook in Colab or Binder

The **open in** links (and the cards of the site) open the executed notebook of the
`gh-pages` branch, outputs of the last CI run included; re-running it in a free
hosted service uses the **released** pylibkriging (from PyPI), not libKriging master
as CI does: a feature not released yet is reported as SKIP. The first code cell of each Python
notebook ([`tools/notebook_setup.py`](tools/notebook_setup.py), a no-op in this
repository) clones the repository for its datasets and helpers and installs
pylibkriging when missing; the notebooks that compare with other packages install
them themselves.

* **Colab**: PyPI has pylibkriging wheels for CPython 3.10 to 3.12 only, and Colab's
  default runtime has used Python 3.13 since August-September 2026: choose
  *Runtime > Change runtime version > 2026.07* (Python 3.12) before running,
  otherwise the setup cell stops with this message.
* **Binder**: the environment of [`.binder/`](.binder) (Python 3.12, pylibkriging),
  copied to `gh-pages` by the site generator; the first start builds the image, which
  takes a few minutes.

Not offered for the R, Octave and Julia notebooks (nbviewer instead, read only):
Octave is not available on Colab,
and the installation of rlibkriging (compiled from CRAN sources) and of JLibKriging.jl
(which compiles libKriging) on these services has not been checked.

## Checks and statuses

Each notebook records its checks in a suite named after the notebook, and writes
`results/<notebook>.checks.csv` and `results/<notebook>.metrics.csv`, with the helpers
of one file per language, loaded by the notebook (not installed):

| | Python | R | Octave | Julia |
|---|---|---|---|---|
| file | [`tools/python/gallery.py`](tools/python/gallery.py) | [`tools/R/gallery.R`](tools/R/gallery.R) | [`tools/octave/gallery.m`](tools/octave/gallery.m) | [`tools/julia/gallery.jl`](tools/julia/gallery.jl) |
| loaded by | `import gallery` (path set by the setup cell) | `source("tools/R/gallery.R")` | `source("tools/octave/gallery.m")` | `include("tools/julia/gallery.jl")` |
| a suite | `suite = gallery.Suite(name)` | `s <- suite(name)` | `s = suite(name)` | `s = Suite(name)` |
| a check | `suite.check(name, ok, ...)` | `check(s, name, ok, ...)` | `s = check(s, name, ok, ...)` | `check!(s, name, ok; ...)` |
| usual checks | `suite.check_close`, `check_less`, `check_greater`, `check_between`, `check_true`, `check_finite`, `check_runs`, `check_skip`, `check_raises` | `check_close(s, ...)`, ... | `s = check_close(s, ...)`, ... (no `check_runs`: `try`) | `check_close!(s, ...)`, ..., `check_runs!(s, name) do ... end` |
| metrics | `suite.metric(case, package, q2=...)` | `metric(s, case, package, q2 = ...)` | `s = metric(s, case, package, "q2", v)` | `metric!(s, case, package; q2=...)` |
| end of notebook | `suite.summary()`, `suite.finalize()` | `suite_summary(s)`, `suite_finalize(s)` | `suite_summary(s)`, `suite_finalize(s)` | `suite_summary(s)`, `suite_finalize!(s)` |
| data, errors | `gallery.read_data`, `metrics.rmse`, `metrics.q2`, `metrics.nlpd`, ... | `read_data`, `rmse`, `q2`, `nlpd`, `coverage` | `read_data`, `rmse`, `q2` | `read_data`, `rmse`, `q2` |

The Python file also holds the test functions (`TESTS`), the designs (`design.lhs`,
`design.dataset`), what the installed pylibkriging offers (`compat.has`) and the
persistence checks (`persistence.check`), used by the Python notebooks only.

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

From the repository root (the notebooks' paths are relative to it; generated
notebooks land there, ignored by git):

```sh
git clone --recurse-submodules https://github.com/libKriging/libKriging ../libKriging   # for a master build
tools/install/python.sh master ../libKriging      # or: tools/install/python.sh release
tools/install/r.sh master ~/Rlib ../libKriging    # then R_LIBS_USER=~/Rlib
tools/install/octave.sh ../libKriging ~/mlib      # then OCTAVE_PATH=~/mlib
tools/install/julia.sh ../libKriging
python tools/py2nb.py src/update-simulate_Kriging_python.py   # or no argument: every source
python tools/run_notebooks.py --inplace update-simulate_Kriging_python.ipynb
python tools/report.py --expect python
mkdir -p executed && cp *.ipynb executed/
python tools/site.py --notebooks executed --results results --out _site   # then open _site/index.html
```

The scripts of [`tools/install/`](tools/install) are the ones CI runs; each says
what it needs (BLAS/LAPACK headers, Octave, R, Julia).

## Editing notebooks

A notebook exists only as a plain-text source in [`src/`](src) (cells separated by
`# %%` / `# %% [markdown]`, or `% %%` in Octave), so that reviews are plain diffs.
[`tools/py2nb.py`](tools/py2nb.py) turns it into a notebook (Python ones get the setup
cell of `tools/notebook_setup.py`); CI does so in each notebook job, executes it, and
archives and publishes the executed notebook (see [The examples](#the-examples)).

```sh
python tools/py2nb.py               # every notebook of src/, at the repository root
python tools/py2nb.py --check       # CI: every source valid (name, title, idea, reference, tags, suite)
```

A new example is a new source in `src/` named `<theme>_<kriging type>_<binding>`,
with its suite named the same, a first markdown cell holding its title
(`# <Language> · <class> — <theme>`) and an **Idea**, a **Reference** and a **Tags**
paragraph, then plots and tables of its results, its checks, a final `## Summary`
cell, and a line in the tables above (CI checks that every source is linked from this
README); CI picks it up automatically. Tags are `key: value, value` groups separated
by ` · `, with the keys `class` (required), `kernel`, `trend`, `objective`, `method`,
`noise`, `warping`, `aggregation`, `compared to`, `topic`; the language comes from the
file extension. They become the tags and filters of the site.

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
baseline, and `executed-notebooks-<channel>` with all the executed notebooks); then
the **site** job builds the gallery site from the executed notebooks of the master
channel (artifact `site`, for a preview, on every run) and, on `main` with
libKriging/libKriging as source, force-pushes it as the single commit of the
`gh-pages` branch (`contents: write` for this job only).
The initial `tools/baseline/` files come from a local run (Python notebooks only,
4-core container); to refresh them, commit the `new-baseline/` files of a
trusted CI run.
