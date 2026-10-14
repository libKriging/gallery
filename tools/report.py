#!/usr/bin/env python3
"""Aggregate the outputs of every notebook (all languages) into one Markdown report.

Reads, from a results directory (default: results/; in CI, the merged artifacts
of every job):

  *.checks.csv    suite,lang,check,severity,status,value,expected,detail
  *.metrics.csv   suite,lang,case,package,metric,value
  xbind_*.csv     lang,dataset,quantity,index,value   (cross-binding protocol)

and reports:

  1. a status table per notebook (PASS / FAIL / WARN / KNOWN / SKIP);
  2. every FAIL, WARN and KNOWN check, with its value and threshold;
  3. known issues whose check now passes ("FIXED?": remove them from
     tools/known_issues.csv);
  4. cross-binding consistency: each language against Python, per quantity, with
     the tolerances of cross-binding-reference_Kriging_python.ipynb;
  5. drifts of the accuracy metrics (q2, rmse, nlpd) and of the fit times against a
     baseline (tools/baseline/metrics-<channel>.csv), when one is given.

Exit status 1 if any check FAILed, if a cross-binding quantity of error level
differs, or if --expect lists a suite that produced no output (crashed before
finalize). The Markdown goes to stdout and, with --summary, is appended to a file
(e.g. $GITHUB_STEP_SUMMARY). The suite of a notebook is its file name without
extension; `--expect python,R` expects one suite per `*_python.ipynb` and
`*_R.ipynb` notebook of the repository root (names are also accepted).

Usage:
  python tools/report.py [--results results] [--baseline tools/baseline/metrics-master.csv]
                         [--write-baseline new-baseline/metrics-master.csv] [--summary FILE]
                         [--expect python,R,octave,julia]
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STATUSES = ["PASS", "FAIL", "WARN", "KNOWN", "SKIP"]
XBIND_TOL = {  # quantity: (rtol, atol relative to the scale of the quantity, severity)
    "fixed_sigma2": (1e-8, 0, "error"), "fixed_beta": (1e-8, 1e-10, "error"), "fixed_ll": (1e-8, 0, "error"),
    "fixed_loo": (1e-8, 0, "error"), "fixed_mean": (1e-8, 1e-10, "error"), "fixed_stdev": (1e-6, 1e-8, "error"),
    "llfun": (1e-8, 0, "error"), "llfun_grad": (1e-6, 1e-8, "error"),
    "fit_theta": (1e-4, 1e-6, "warn"), "fit_ll": (1e-4, 0, "warn"), "fit_mean": (1e-4, 1e-6, "warn"),
    "sim": (1e-8, 1e-10, "warn"),
}
TRACKED = {"q2": "higher", "rmse": "lower", "nlpd": "lower", "fit_s": "time"}


def read_all(results, pattern):
    frames = []
    for p in sorted(Path(results).rglob(pattern)):
        try:
            frames.append(pd.read_csv(p, dtype=str, keep_default_na=False))
        except Exception as exc:  # noqa: BLE001
            print(f"warning: cannot read {p}: {exc}", file=sys.stderr)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def md_table(df, max_rows=200):
    if df.empty:
        return "_none_\n"
    df = df.head(max_rows)
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]).replace("|", "\\|").replace("\n", " ") for c in cols) + " |")
    return "\n".join(out) + "\n"


def xbind_report(results):
    df = read_all(results, "xbind_*.csv")
    if df.empty:
        return "_no cross-binding output_\n", 0
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    ref = df[df["lang"] == "python"].set_index(["dataset", "quantity", "index"])["value"]
    if ref.empty:
        return "_no Python reference (xbind_python.csv) to compare with_\n", 0
    rows, n_err = [], 0
    for lang, sub in df[df["lang"] != "python"].groupby("lang"):
        sub = sub.set_index(["dataset", "quantity", "index"])["value"]
        for (dataset, quantity), grp in sub.groupby(level=[0, 1]):
            rtol, atol, sev = XBIND_TOL.get(quantity, (1e-6, 0, "warn"))
            try:
                r = ref.loc[grp.index]
            except KeyError:
                rows.append((lang, dataset, quantity, "missing in Python", "", sev, "FAIL" if sev == "error" else "WARN"))
                continue
            a, b = grp.to_numpy(float), r.to_numpy(float)
            scale = max(np.max(np.abs(b)), 1e-300)
            gap = np.abs(a - b)
            ok = bool(np.all(np.isfinite(a)) and np.all(gap <= rtol * np.abs(b) + atol * scale))
            rel = float(np.max(gap / np.maximum(np.abs(b), 1e-300)))
            status = "PASS" if ok else ("FAIL" if sev == "error" else "WARN")
            n_err += status == "FAIL"
            rows.append((lang, dataset, quantity, f"{rel:.2e}", f"rtol {rtol:g}", sev, status))
        missing = set(ref.index.droplevel(2).unique()) - set(sub.index.droplevel(2).unique())
        for dataset, quantity in sorted(missing):
            rows.append((lang, dataset, quantity, "missing", "", "warn", "WARN"))
    tab = pd.DataFrame(rows, columns=["lang", "dataset", "quantity", "max rel. gap", "tolerance", "severity", "status"])
    summary = tab.groupby(["lang", "status"]).size().unstack(fill_value=0).reset_index()
    text = md_table(summary) + "\n" + md_table(tab[tab["status"] != "PASS"])
    return text, n_err


def drift_report(metrics, baseline):
    base = pd.read_csv(baseline, dtype=str, keep_default_na=False)
    key = ["suite", "lang", "case", "package", "metric"]
    cur = metrics[metrics["metric"].isin(TRACKED)].copy()
    m = cur.merge(base, on=key, suffixes=("", "_base"))
    m["value"] = pd.to_numeric(m["value"], errors="coerce")
    m["value_base"] = pd.to_numeric(m["value_base"], errors="coerce")
    rows = []
    for _, r in m.iterrows():
        kind, v, b = TRACKED[r["metric"]], r["value"], r["value_base"]
        if not np.isfinite(b):
            continue
        if not np.isfinite(v):
            bad = True
        elif kind == "higher":
            bad = v < b - 0.02
        elif kind == "lower":
            bad = v > b + 0.1 * abs(b) + 1e-12
        else:  # time: CI runners are noisy, only a 3x slowdown on a fit above 0.5 s is reported
            bad = v > 3 * b and v > 0.5
        if bad:
            rows.append(dict(suite=r["suite"], lang=r["lang"], case=r["case"], package=r["package"],
                             metric=r["metric"], baseline=f"{b:.4g}", now=f"{v:.4g}"))
    return pd.DataFrame(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=str(ROOT / "results"))
    p.add_argument("--baseline", help="metrics CSV of a reference run (drift report)")
    p.add_argument("--write-baseline", help="write the current metrics as a new baseline")
    p.add_argument("--summary", help="also append the report to this file")
    p.add_argument("--expect", default="",
                   help="comma-separated suites, or bindings (python, R, octave, julia), that must have output")
    args = p.parse_args(argv)

    checks = read_all(args.results, "*.checks.csv")
    metrics = read_all(args.results, "*.metrics.csv")
    lines = ["# libKriging benchmark report\n"]
    exit_code = 0

    # 1. status per notebook
    if checks.empty:
        lines.append("_no check output found_\n")
        exit_code = 1
    else:
        tab = pd.crosstab([checks["suite"], checks["lang"]], checks["status"])
        tab = tab.reindex(columns=STATUSES, fill_value=0).reset_index()
        tab.columns.name = None
        tab.insert(2, "result", np.where(tab["FAIL"] > 0, "❌", np.where(tab["WARN"] + tab["KNOWN"] > 0, "⚠️", "✅")))
        lines += ["## Notebooks\n", md_table(tab)]
        exit_code |= int((checks["status"] == "FAIL").any())

    expected = []
    for item in filter(None, args.expect.split(",")):
        stems = sorted(nb.stem for nb in ROOT.glob(f"*_{item}.ipynb"))
        expected += stems or [item]
    missing = sorted(set(expected) - set(checks.get("suite", pd.Series(dtype=str))))
    if missing:
        lines += ["## Notebooks without output (crashed before `finalize`)\n", md_table(pd.DataFrame({"suite": missing}))]
        exit_code = 1

    # 2. problems
    if not checks.empty:
        cols = ["suite", "lang", "check", "value", "expected", "detail"]
        for status, title in [("FAIL", "Failures"), ("WARN", "Warnings"), ("KNOWN", "Known issues (still present)")]:
            sub = checks[checks["status"] == status][cols]
            lines += [f"## {title} ({len(sub)})\n", md_table(sub)]

        # 3. known issues that now pass
        known_file = ROOT / "tools" / "known_issues.csv"
        if known_file.exists():
            known = pd.read_csv(known_file, dtype=str, keep_default_na=False)
            passing = known.merge(checks[checks["status"] == "PASS"][["suite", "check"]], on=["suite", "check"])
            if not passing.empty:
                lines += ["## Known issues that now pass — FIXED? (remove them from tools/known_issues.csv)\n",
                          md_table(passing)]

    # 4. cross-binding
    text, n_err = xbind_report(args.results)
    lines += ["## Cross-binding consistency (vs Python)\n", text]
    exit_code |= int(n_err > 0)

    # 5. drifts
    if args.baseline and Path(args.baseline).exists() and not metrics.empty:
        drift = drift_report(metrics, args.baseline)
        lines += [f"## Metric drifts vs baseline ({len(drift)})\n", md_table(drift)]
    if args.write_baseline and not metrics.empty:
        keep = metrics[metrics["metric"].isin(TRACKED)]
        Path(args.write_baseline).parent.mkdir(parents=True, exist_ok=True)
        keep.to_csv(args.write_baseline, index=False, quoting=csv.QUOTE_MINIMAL)

    report = "\n".join(lines)
    print(report)
    if args.summary:
        with open(args.summary, "a") as fh:
            fh.write(report + "\n")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
