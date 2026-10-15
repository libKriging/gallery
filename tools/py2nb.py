#!/usr/bin/env python3
"""Convert a "percent" script (cells separated by ``# %%`` / ``# %% [markdown]``, or
``% %%`` in Octave) into a notebook without outputs. The notebooks of this gallery are
written this way, in ``src/``, so that they review as plain diffs; the ``.ipynb`` files
are not committed: CI generates and executes them, and publishes the executed ones (see
``tools/site.py``). Every Python notebook gets the cell of ``tools/notebook_setup.py``
after its title, to run on Colab or Binder too.

Every source is named ``<theme>_<kriging type>_<binding>.<ext>`` and starts with a
markdown cell holding its title (``# <Language> · <class> — <theme>``), an ``**Idea.**``
paragraph, a ``**Reference.**`` paragraph and a ``**Tags.**`` paragraph
(``key: value, value · key: value``, keys from ``TAG_KEYS``); the suite it writes
(``Suite("<name>")``) is named after the file. ``--check`` validates all that; the title,
idea and tags also go to the notebook metadata (``metadata.gallery``), for the site.

Usage: python tools/py2nb.py [--check] [--out DIR] [SRC...]   (default: every src/* file)
  each ``src/<name>.<ext>`` gives ``<DIR>/<name>.ipynb`` (default DIR: the repository root).
"""
import argparse
import re
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
# code cell inserted after the title cell of every Python notebook, so that it also runs
# outside the repository (Colab, Binder): a no-op in the repository itself
SETUP = {".py": ROOT / "tools" / "notebook_setup.py"}

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
BINDINGS = {".py": "python", ".R": "R", ".jl": "julia", ".m": "octave"}
LANGUAGES = {".py": "Python", ".R": "R", ".jl": "Julia", ".m": "Octave"}
# keys of the **Tags.** paragraph (all optional but class), in display order
TAG_KEYS = ("class", "kernel", "trend", "objective", "method", "noise", "warping",
            "aggregation", "compared to", "topic")
SUITE = re.compile(r'\b(?:Suite|suite)\("([^"]+)"\)')


def sources():
    return sorted(f for f in SRC.iterdir() if f.suffix in KERNELS)


def paragraph(text, label):
    """The paragraph starting with ``**<label>.**`` in markdown ``text``, unwrapped."""
    m = re.search(rf"^\*\*{label}\.\*\*(.*?)(?:\n\s*\n|\Z)", text, re.S | re.M)
    if not m:
        return None
    return re.sub(r"\s*\n\s*", " ", re.sub(r"(?<=\w-)\n\s*", "", m.group(1))).strip()


def parse_tags(text):
    tags = {}
    for part in text.split("·"):
        key, sep, values = part.partition(":")
        if not sep:
            raise ValueError(f"tag without 'key:' in {part.strip()!r}")
        tags[key.strip()] = [v.strip() for v in values.split(",") if v.strip()]
    return tags


def describe(src, cells=None):
    """Title, idea, reference and tags of a source (from its first markdown cell)."""
    cells = cells if cells is not None else convert(src).cells
    head = cells[0].source if cells and cells[0].cell_type == "markdown" else ""
    title = re.match(r"#\s+(.*)", head)
    tags = paragraph(head, "Tags")
    return {
        "stem": src.stem,
        "language": LANGUAGES[src.suffix],
        "binding": BINDINGS[src.suffix],
        "title": title.group(1).strip() if title else None,
        "idea": paragraph(head, "Idea"),
        "reference": paragraph(head, "Reference"),
        "tags": parse_tags(tags) if tags else None,
    }


def validate(src):
    """Problems of a source, as a list of messages (empty: valid)."""
    errors = []
    parts = src.stem.split("_")
    if len(parts) != 3 or not all(parts):
        errors.append("name is not <theme>_<kriging type>_<binding>")
    elif parts[2] != BINDINGS[src.suffix]:
        errors.append(f"binding {parts[2]!r} of the name does not match {src.suffix} ({BINDINGS[src.suffix]})")
    try:
        d = describe(src)
    except ValueError as e:
        return errors + [f"Tags: {e}"]
    if not d["title"] or not d["title"].startswith(d["language"] + " · "):
        errors.append(f"first cell does not start with a title '# {d['language']} · ...'")
    for key in ("idea", "reference", "tags"):
        if not d[key]:
            errors.append(f"no **{key.capitalize()}.** paragraph in the first cell")
    if d["tags"]:
        unknown = set(d["tags"]) - set(TAG_KEYS)
        if unknown:
            errors.append(f"unknown tag keys {sorted(unknown)} (allowed: {', '.join(TAG_KEYS)})")
        if not d["tags"].get("class"):
            errors.append("no 'class:' tag")
        if any(not v for v in d["tags"].values()):
            errors.append("a tag key without value")
    suites = set(SUITE.findall(src.read_text()))
    if suites != {src.stem}:
        errors.append(f"suite names {sorted(suites)} != [{src.stem!r}]")
    return errors


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
    meta = describe(src, cells)
    if src.suffix in SETUP and cells and cells[0].cell_type == "markdown":
        cells.insert(1, nbformat.v4.new_code_cell(SETUP[src.suffix].read_text().strip("\n")))
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = kernelspec
    nb.metadata["language_info"] = lang_info
    nb.metadata["gallery"] = meta
    for i, c in enumerate(nb.cells):  # stable ids: no spurious diffs on regeneration
        c["id"] = f"c{i:03d}"
    return nb


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sources", nargs="*", type=Path)
    p.add_argument("--check", action="store_true", help="validate the sources, write nothing")
    p.add_argument("--out", type=Path, default=ROOT, help="directory of the notebooks (default: repository root)")
    args = p.parse_args(argv)
    srcs = args.sources or sources()
    if args.check:
        bad = {s: e for s in srcs if (e := validate(s))}
        for s, errors in bad.items():
            print(f"{s.name}:", *errors, sep="\n  ")
        return 1 if bad else 0
    args.out.mkdir(parents=True, exist_ok=True)
    for src in srcs:
        (args.out / (src.stem + ".ipynb")).write_text(nbformat.writes(convert(src)) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
