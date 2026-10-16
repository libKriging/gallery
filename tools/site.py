#!/usr/bin/env python3
"""Build the gallery web site: one card per notebook, from the executed notebooks.

For every source of ``src/``, the card shows a thumbnail (the first figure of the executed
notebook, else a placeholder), the title, the idea, the tags (language, Kriging class,
kernel, objective, method, noise, warping, aggregation, compared packages, topic: the
``**Tags.**`` paragraph of the source) and the status of its checks
(``<results>/<stem>.checks.csv``). Clicking the card opens the executed notebook in
nbviewer; Python cards also link to Colab and Binder (which do not run libKriging for the
other languages). Tags filter the cards.

Output (``--out``, published as the ``gh-pages`` branch by CI):

  index.html                 the gallery (self-contained: no external resource)
  gallery.json               the same, as data
  notebooks/<stem>.ipynb     the executed notebooks (outputs included); a notebook whose
                             execution left nothing is generated from its source
  thumbs/<stem>.png|svg      the thumbnails
  .binder/                   the Binder environment (for the Binder links on this branch)
  .nojekyll

Usage: python tools/site.py --notebooks executed --results results --out _site
                            [--repo libKriging/gallery] [--branch gh-pages] [--libkriging TEXT]
"""
import argparse
import base64
import csv
import html
import json
import re
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import nbformat

sys.path.insert(0, str(Path(__file__).resolve().parent))
import py2nb  # noqa: E402

ROOT = py2nb.ROOT
LANG_COLORS = {"Python": "#3572a5", "R": "#198ce7", "Octave": "#0790c0", "Julia": "#9558b2"}
STATUSES = ("FAIL", "WARN", "KNOWN", "SKIP", "PASS")


def first_image(nb):
    """(extension, bytes) of the first figure among the outputs of ``nb``, or None."""
    for cell in nb.cells:
        for out in cell.get("outputs", []):
            data = out.get("data", {})
            if "image/png" in data:
                return "png", base64.b64decode(data["image/png"])
            if "image/svg+xml" in data:
                svg = data["image/svg+xml"]
                return "svg", ("".join(svg) if isinstance(svg, list) else svg).encode()
    return None


def placeholder(language, title):
    color = LANG_COLORS.get(language, "#555")
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 250">'
            f'<rect width="400" height="250" fill="{color}"/>'
            f'<text x="200" y="135" font-family="sans-serif" font-size="40" fill="#fff" '
            f'text-anchor="middle">{html.escape(language)}</text>'
            f'<text x="200" y="175" font-family="sans-serif" font-size="14" fill="#fff" '
            f'text-anchor="middle" opacity=".8">{html.escape(title[:48])}</text></svg>').encode()


def statuses(results, stem):
    f = results / f"{stem}.checks.csv" if results else None
    if not f or not f.is_file():
        return None
    with f.open(newline="") as fh:
        return dict(Counter(row["status"] for row in csv.DictReader(fh)))


def verdict(counts, executed):
    if counts is None:
        return ("not run", "none") if not executed else ("no checks", "none")
    if counts.get("FAIL"):
        return f"{counts['FAIL']} failed", "fail"
    n = sum(counts.values())
    if counts.get("WARN") or counts.get("KNOWN"):
        return f"{counts.get('PASS', 0)}/{n} passed", "warn"
    return f"{counts.get('PASS', 0)}/{n} passed", "pass"


def links(meta, repo, branch):
    path = f"notebooks/{meta['stem']}.ipynb"
    view = f"https://nbviewer.org/github/{repo}/blob/{branch}/{path}"
    src = f"https://github.com/{repo}/blob/main/src/{meta['stem']}{meta['ext']}"
    if meta["language"] == "Python":
        return {"open": view,
                "Colab": f"https://colab.research.google.com/github/{repo}/blob/{branch}/{path}",
                "Binder": f"https://mybinder.org/v2/gh/{repo}/{branch}?labpath={quote(path, safe='')}",
                "view": view, "source": src}
    return {"open": view, "view": view, "source": src}


def card_html(c):
    tags = [("language", c["language"])] + [(k, v) for k in py2nb.TAG_KEYS for v in c["tags"].get(k, [])]
    chips = "".join(
        f'<span class="tag k-{html.escape(k.replace(" ", "-"))}" title="{html.escape(k)}">'
        f'{"" if k in ("language", "class") else html.escape(k) + ": "}{html.escape(v)}</span>'
        for k, v in tags)
    data = html.escape(json.dumps([f"{k}: {v}" for k, v in tags]))
    others = " ".join(f'<a href="{html.escape(u)}" target="_blank" rel="noopener">{n}</a>'
                      for n, u in c["links"].items() if n != "open")
    status, cls = c["status"]
    lang = html.escape(c["language"])
    return f"""<article class="card" data-tags="{data}" data-text="{html.escape((c['title'] + ' ' + c['idea']).lower())}">
  <a class="main" href="{html.escape(c['links']['open'])}" target="_blank" rel="noopener"
     title="view the executed notebook">
    <img src="{html.escape(c['thumb'])}" alt="" loading="lazy">
  </a>
  <div class="body">
    <div class="head"><span class="lang" style="background:{LANG_COLORS.get(c['language'], '#555')}">{lang}</span>
      <span class="status s-{cls}">{html.escape(status)}</span></div>
    <h2>{html.escape(c['title'])}</h2>
    <p class="idea">{html.escape(c['idea'])}</p>
    <p class="ref"><b>Reference.</b> {html.escape(c['reference'])}</p>
    <div class="tags">{chips}</div>
    <div class="links">{others}</div>
  </div>
</article>"""


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>libKriging gallery</title>
<style>
:root { --bg:#f6f7f9; --fg:#1d2330; --muted:#5d6676; --card:#fff; --line:#dde1e7; --chip:#eef1f5; --on:#1d2330; --onfg:#fff; }
@media (prefers-color-scheme: dark) { :root { --bg:#14171c; --fg:#e4e7ec; --muted:#9aa3b2; --card:#1d2129; --line:#2d333d; --chip:#272c35; --on:#e4e7ec; --onfg:#14171c; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
header, main, footer { max-width:1280px; margin:0 auto; padding:16px; }
header h1 { margin:8px 0 4px; font-size:28px; }
header p { margin:4px 0; color:var(--muted); }
a { color:inherit; }
#filters { margin-top:12px; display:flex; flex-direction:column; gap:6px; }
#tagbox summary { cursor:pointer; color:var(--muted); font-size:13px; margin:4px 0; }
#tagbox[open] { display:flex; flex-direction:column; gap:6px; }
#filters .row { display:flex; flex-wrap:wrap; gap:4px; align-items:center; }
#filters .key { color:var(--muted); font-size:13px; min-width:92px; }
#filters button { border:1px solid var(--line); background:var(--chip); color:var(--fg); border-radius:12px; padding:2px 9px; font-size:13px; cursor:pointer; }
#filters button.on { background:var(--on); color:var(--onfg); border-color:var(--on); }
#search { width:100%; max-width:360px; padding:6px 10px; border:1px solid var(--line); border-radius:6px; background:var(--card); color:var(--fg); font-size:14px; }
#count { color:var(--muted); font-size:13px; margin-left:8px; }
.grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(300px, 1fr)); gap:16px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; overflow:hidden; display:flex; flex-direction:column; }
.card.hidden { display:none; }
.card .main { display:block; aspect-ratio:16/10; background:#fff; border-bottom:1px solid var(--line); }
.card img { width:100%; height:100%; object-fit:contain; display:block; }
.card .main:hover img { opacity:.85; }
.body { padding:10px 12px 12px; display:flex; flex-direction:column; gap:6px; flex:1; }
.head { display:flex; justify-content:space-between; align-items:center; }
.lang { color:#fff; border-radius:4px; padding:1px 7px; font-size:12px; font-weight:600; }
.status { font-size:12px; border-radius:4px; padding:1px 7px; }
.s-pass { background:#d9f2df; color:#14532d; } .s-warn { background:#fdf0cf; color:#713f12; }
.s-fail { background:#fbd9d9; color:#7f1d1d; } .s-none { background:var(--chip); color:var(--muted); }
h2 { font-size:16px; margin:2px 0; }
.idea { margin:0; }
.ref { margin:0; color:var(--muted); font-size:13px; }
.tags { display:flex; flex-wrap:wrap; gap:4px; margin-top:auto; padding-top:4px; }
.tag { background:var(--chip); border-radius:10px; padding:1px 8px; font-size:12px; color:var(--muted); }
.tag.k-class { color:var(--fg); font-weight:600; }
.tag.k-language { display:none; }
.links { display:flex; gap:12px; font-size:13px; }
.links a { color:var(--muted); }
.links a:hover { color:var(--fg); }
footer { color:var(--muted); font-size:13px; }
</style>
</head>
<body>
<header>
  <h1>libKriging gallery</h1>
  <p>Notebooks showing <a href="https://github.com/libKriging/libKriging">libKriging</a> at work in Python, R, Octave and Julia,
  each compared with a reference (another package, an exact model, a known value). Click a card to view the executed
  notebook; Python notebooks can also be run in Colab or Binder, with the released pylibkriging.</p>
  <p>Executed in CI against libKriging __LIBKRIGING__, __DATE__ · <a href="https://github.com/__REPO__">sources</a> · __TOTALS__</p>
  <div id="filters"><div class="row"><input id="search" type="search" placeholder="search titles and ideas"><span id="count"></span></div>
  <details id="tagbox"><summary>filter by tags</summary>__FILTERS__</details></div>
</header>
<main class="grid">
__CARDS__
</main>
<footer>Built by <code>tools/site.py</code> from the CI artifacts of <a href="https://github.com/__REPO__">__REPO__</a>.</footer>
<script>
(function () {
  var on = new Set(), cards = Array.prototype.slice.call(document.querySelectorAll(".card"));
  var search = document.getElementById("search"), count = document.getElementById("count");
  function apply() {
    var q = search.value.trim().toLowerCase(), n = 0;
    cards.forEach(function (c) {
      var tags = JSON.parse(c.dataset.tags), ok = true;
      on.forEach(function (t) { if (tags.indexOf(t) < 0) ok = false; });
      if (q && c.dataset.text.indexOf(q) < 0) ok = false;
      c.classList.toggle("hidden", !ok); if (ok) n++;
    });
    count.textContent = n + " / " + cards.length + " notebooks";
  }
  document.querySelectorAll("#filters button").forEach(function (b) {
    b.addEventListener("click", function () {
      var t = b.dataset.tag;
      if (on.has(t)) on.delete(t); else on.add(t);
      b.classList.toggle("on"); apply();
    });
  });
  search.addEventListener("input", apply);
  if (window.innerWidth > 700) document.getElementById("tagbox").open = true;
  apply();
})();
</script>
</body>
</html>
"""


def filters_html(cards):
    rows = []
    for key in ("language",) + py2nb.TAG_KEYS:
        values = Counter(v for c in cards for v in ([c["language"]] if key == "language" else c["tags"].get(key, [])))
        if not values:
            continue
        buttons = "".join(f'<button type="button" data-tag="{html.escape(f'{key}: {v}')}">'
                          f'{html.escape(v)} <small>{n}</small></button>'
                          for v, n in sorted(values.items(), key=lambda kv: (-kv[1], kv[0].lower())))
        rows.append(f'<div class="row"><span class="key">{html.escape(key)}</span>{buttons}</div>')
    return "\n".join(rows)


def build(args):
    out = args.out
    if out.exists():
        shutil.rmtree(out)
    (out / "notebooks").mkdir(parents=True)
    (out / "thumbs").mkdir()
    cards = []
    for src in py2nb.sources():
        meta = py2nb.describe(src)
        meta["ext"] = src.suffix
        executed = args.notebooks / f"{src.stem}.ipynb" if args.notebooks else None
        if executed and executed.is_file():
            nb = nbformat.read(executed, as_version=4)
            ran = True
        else:
            nb, ran = py2nb.convert(src), False
        nbformat.write(nb, out / "notebooks" / f"{src.stem}.ipynb")
        for key in ("title", "idea", "reference"):  # plain text on the cards
            meta[key] = re.sub(r"`([^`]*)`", r"\1", meta[key] or "").replace("**", "")
        title = meta["title"].split(" · ", 1)[-1]
        img = first_image(nb)
        ext, data = img if img else ("svg", placeholder(meta["language"], title))
        (out / "thumbs" / f"{src.stem}.{ext}").write_bytes(data)
        counts = statuses(args.results, src.stem)
        cards.append({**meta, "title": title, "thumb": f"thumbs/{src.stem}.{ext}", "executed": ran,
                      "checks": counts, "status": verdict(counts, ran),
                      "links": links(meta, args.repo, args.branch)})
    order = {"Python": 0, "R": 1, "Octave": 2, "Julia": 3}
    cards.sort(key=lambda c: ("cross-binding" in c["tags"].get("topic", []), order.get(c["language"], 9), c["title"].lower()))
    total = Counter()
    for c in cards:
        total.update(c["checks"] or {})
    totals = ", ".join(f"{total[s]} {s}" for s in STATUSES if total[s]) or "no check results"
    page = (PAGE.replace("__CARDS__", "\n".join(map(card_html, cards)))
            .replace("__FILTERS__", filters_html(cards))
            .replace("__TOTALS__", html.escape(f"{len(cards)} notebooks, checks: {totals}"))
            .replace("__LIBKRIGING__", html.escape(args.libkriging))
            .replace("__DATE__", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
            .replace("__REPO__", html.escape(args.repo)))
    (out / "index.html").write_text(page)
    (out / "gallery.json").write_text(json.dumps(
        [{k: v for k, v in c.items() if k != "status"} | {"status": c["status"][0]} for c in cards], indent=1))
    shutil.copytree(ROOT / ".binder", out / ".binder")
    (out / ".nojekyll").write_text("")
    print(f"{out}: {len(cards)} cards ({sum(c['executed'] for c in cards)} executed), checks: {totals}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--notebooks", type=Path, help="directory of the executed notebooks")
    p.add_argument("--results", type=Path, help="directory of the *.checks.csv files")
    p.add_argument("--out", type=Path, default=Path("_site"))
    p.add_argument("--repo", default="libKriging/gallery", help="GitHub repository of the site")
    p.add_argument("--branch", default="gh-pages", help="branch the site is published on")
    p.add_argument("--libkriging", default="master", help="libKriging version, shown in the header")
    build(p.parse_args(argv))


if __name__ == "__main__":
    main()
