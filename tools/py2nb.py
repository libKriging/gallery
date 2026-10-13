#!/usr/bin/env python3
"""Convert a "percent" script (cells separated by ``# %%`` / ``# %% [markdown]``) into a
notebook without outputs. The notebooks of this repository are written this way
(sources in ``notebooks/**/src/``) so that they review as plain diffs; the
``.ipynb`` files are generated, and CI checks that they are in sync.

Usage: python tools/py2nb.py SRC... [--check]
  each ``notebooks/<lang>/src/<name>.<ext>`` gives ``notebooks/<lang>/<name>.ipynb``.
"""
import argparse
import re
import sys
from pathlib import Path

import nbformat

KERNELS = {
    ".py": ({"name": "python3", "display_name": "Python 3", "language": "python"},
            {"name": "python"}, "#"),
    ".R": ({"name": "ir", "display_name": "R", "language": "R"},
           {"name": "R"}, "#"),
    ".jl": ({"name": "julia-1.11", "display_name": "Julia 1.11", "language": "julia"},
            {"name": "julia"}, "#"),
    ".m": ({"name": "octave", "display_name": "Octave", "language": "octave"},
           {"name": "octave"}, "%"),
}


def convert(src):
    kernelspec, lang_info, comment = KERNELS[src.suffix]
    marker = re.compile(rf"^{re.escape(comment)} %%( \[markdown\])?\s*$")
    cells, kind, buf = [], None, []

    def flush():
        text = "\n".join(buf).strip("\n")
        if kind == "markdown":
            text = "\n".join(re.sub(rf"^{re.escape(comment)} ?", "", ln) for ln in text.splitlines())
            cells.append(nbformat.v4.new_markdown_cell(text))
        elif kind == "code" and text:
            cells.append(nbformat.v4.new_code_cell(text))

    for line in src.read_text().splitlines():
        m = marker.match(line)
        if m:
            flush()
            kind, buf = ("markdown" if m.group(1) else "code"), []
        elif kind is not None:
            buf.append(line)
    flush()
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = kernelspec
    nb.metadata["language_info"] = lang_info
    for i, c in enumerate(nb.cells):  # stable ids: no spurious diffs on regeneration
        c["id"] = f"c{i:03d}"
    return nb


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sources", nargs="+", type=Path)
    p.add_argument("--check", action="store_true", help="fail if a notebook differs from its source")
    args = p.parse_args(argv)
    stale = []
    for src in args.sources:
        out = src.parent.parent / (src.stem + ".ipynb")
        text = nbformat.writes(convert(src)) + "\n"
        if args.check:
            if not out.exists() or out.read_text() != text:
                stale.append(out)
        else:
            out.write_text(text)
    if stale:
        print("out of date (run tools/py2nb.py):", *map(str, stale), sep="\n  ")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
