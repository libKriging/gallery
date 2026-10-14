"""Record checks and metrics from a notebook, and fail it at the end if needed.

A notebook creates one ``Suite``, records any number of checks (assertions that
do *not* stop the notebook, so every problem of a run is reported at once) and
metrics, then calls ``suite.finalize()`` in its last cell: it writes

  results/<suite>.checks.csv   suite,lang,check,severity,status,value,expected,detail
  results/<suite>.metrics.csv  suite,lang,case,package,metric,value

and raises ``AssertionError`` if any check of severity "error" failed, which makes
the notebook (hence the CI job) fail. Checks of severity "warn" are reported but
never fail the run: they flag performance drifts or comparisons with other
packages, where a threshold is a judgement call rather than a correctness bound.

Statuses: PASS, FAIL (error-level), WARN (warn-level), SKIP (feature missing in
the installed version), KNOWN (a failing check, of either severity, listed in
``tools/known_issues.csv``: reported with its issue, but does not fail the run). A listed check that passes is
reported as FIXED by ``tools/report.py``, so that the list does not rot.

The R, Octave and Julia notebooks write the same two CSV formats with a few lines
of their own code (tools/r/checks.R, tools/octave/lkb_*.m, tools/julia/checks.jl), so that ``tools/report.py`` can
aggregate every language in one report.
"""
import csv
import math
import os
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import numpy as np


def _find_root():
    """The repository root: $LKBENCH_ROOT, or the first directory, from the working
    directory up, that holds tools/known_issues.csv (notebooks run from the root).
    Not derived from this file's location, which is site-packages for a non-editable
    install: the known issues would then be silently ignored."""
    if os.environ.get("LKBENCH_ROOT"):
        return Path(os.environ["LKBENCH_ROOT"]).resolve()
    for d in [Path.cwd(), *Path.cwd().parents]:
        if (d / "tools" / "known_issues.csv").is_file():
            return d
    raise RuntimeError("libKriging benchmarks repository not found from the working directory "
                       f"{Path.cwd()}: run the notebooks from the repository root, or set LKBENCH_ROOT")


ROOT = _find_root()
RESULTS_DIR = Path(os.environ.get("LKBENCH_RESULTS", ROOT / "results"))
KNOWN_ISSUES_FILE = ROOT / "tools" / "known_issues.csv"
DATA_DIR = ROOT / "data"


def known_issues():
    """{(suite, check): issue} from tools/known_issues.csv."""
    with open(KNOWN_ISSUES_FILE, newline="") as fh:
        return {(r["suite"], r["check"]): r["issue"] for r in csv.DictReader(fh)}


CHECK_FIELDS = ["suite", "lang", "check", "severity", "status", "value", "expected", "detail"]
METRIC_FIELDS = ["suite", "lang", "case", "package", "metric", "value"]


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, float, np.integer, np.floating)):
        return f"{float(v):.6g}"
    return str(v)


class Suite:
    def __init__(self, name, lang="python"):
        self.name, self.lang = name, lang
        self.checks, self.metrics = [], []
        self.t0 = time.time()
        self.known = known_issues()

    # -- checks ---------------------------------------------------------------

    def _add(self, check, ok, value=None, expected=None, detail="", severity="error"):
        status = "PASS" if ok else ("FAIL" if severity == "error" else "WARN")
        if not ok and (self.name, check) in self.known:
            status, detail = "KNOWN", f"{self.known[self.name, check]} | {detail}".rstrip(" |")
        self.checks.append(dict(suite=self.name, lang=self.lang, check=check, severity=severity,
                                status=status, value=_fmt(value), expected=_fmt(expected), detail=detail))
        if not ok:
            print(f"[{status}] {check}: value={_fmt(value)} expected {_fmt(expected)} {detail}")
        return ok

    def true(self, check, cond, detail="", severity="error"):
        return self._add(check, bool(cond), cond, True, detail, severity)

    def less(self, check, value, bound, detail="", severity="error"):
        """value < bound (NaN fails)."""
        value = float(value)
        return self._add(check, math.isfinite(value) and value < bound, value, f"< {bound:g}", detail, severity)

    def greater(self, check, value, bound, detail="", severity="error"):
        value = float(value)
        return self._add(check, math.isfinite(value) and value > bound, value, f"> {bound:g}", detail, severity)

    def between(self, check, value, lo, hi, detail="", severity="error"):
        value = float(value)
        return self._add(check, math.isfinite(value) and lo <= value <= hi, value, f"in [{lo:g}, {hi:g}]",
                         detail, severity)

    def close(self, check, a, b, rtol=1e-6, atol=0.0, detail="", severity="error"):
        """All |a - b| <= atol + rtol |b| (element-wise); reports the worst relative gap."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        if a.shape != b.shape:
            return self._add(check, False, f"shape {a.shape}", f"shape {b.shape}", detail, severity)
        gap = np.abs(a - b)
        ok = bool(np.all(np.isfinite(a)) and np.all(gap <= atol + rtol * np.abs(b)))
        rel = float(np.max(gap / np.maximum(np.abs(b), 1e-300))) if a.size else 0.0
        return self._add(check, ok, rel, f"rtol {rtol:g}, atol {atol:g}", detail, severity)

    def finite(self, check, a, detail="", severity="error"):
        a = np.asarray(a, float)
        return self._add(check, bool(a.size and np.all(np.isfinite(a))), f"{np.sum(~np.isfinite(a))} non-finite",
                         "all finite", detail, severity)

    def skip(self, check, reason):
        self.checks.append(dict(suite=self.name, lang=self.lang, check=check, severity="info",
                                status="SKIP", value="", expected="", detail=reason))
        print(f"[SKIP] {check}: {reason}")

    @contextmanager
    def runs(self, check, severity="error"):
        """``with suite.runs("x"): ...`` records a FAIL (with the exception) instead of stopping."""
        try:
            yield
        except Exception as exc:  # noqa: BLE001 - every failure is a finding here
            self._add(check, False, type(exc).__name__, "no exception",
                      str(exc).splitlines()[0][:200] if str(exc) else traceback.format_exc(limit=1), severity)
        else:
            self._add(check, True, "ok", "no exception", "", severity)

    def raises(self, check, fn, *args, severity="error", **kwargs):
        """``fn(*args)`` must raise (invalid input must be rejected, not silently accepted)."""
        try:
            fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            return self._add(check, True, type(exc).__name__, "an exception", "", severity)
        return self._add(check, False, "no exception", "an exception", "", severity)

    # -- metrics --------------------------------------------------------------

    def metric(self, case, package, **values):
        for k, v in values.items():
            self.metrics.append(dict(suite=self.name, lang=self.lang, case=case, package=package,
                                     metric=k, value=_fmt(v)))

    def table(self):
        """Metrics as a wide pandas DataFrame (one row per case x package)."""
        import pandas as pd
        if not self.metrics:
            return pd.DataFrame()
        df = pd.DataFrame(self.metrics)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df.pivot_table(index=["case", "package"], columns="metric", values="value", sort=False)

    # -- output ---------------------------------------------------------------

    def finalize(self):
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        self.metric("_suite", "_", elapsed_s=time.time() - self.t0)
        for rows, fields, ext in ((self.checks, CHECK_FIELDS, "checks"), (self.metrics, METRIC_FIELDS, "metrics")):
            with open(RESULTS_DIR / f"{self.name}.{ext}.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)
        counts = {s: sum(c["status"] == s for c in self.checks) for s in ("PASS", "FAIL", "WARN", "KNOWN", "SKIP")}
        print(f"{self.name}: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
        failed = [c for c in self.checks if c["status"] == "FAIL"]
        if failed:
            raise AssertionError(f"{len(failed)} check(s) failed in {self.name}: "
                                 + "; ".join(c["check"] for c in failed[:20]))
        return counts
