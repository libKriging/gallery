#!/usr/bin/env python3
"""Write the shared datasets of data/ (committed): the cross-binding notebooks of every
language read these exact files, so that all bindings fit the very same points.

Plain CSV, no header, full precision; X is n x d (rows = observations), y is a column.
Regenerating them is only needed when a dataset is added: the files are the reference,
not the script (a new scipy could draw other LHS points)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "python"))
from gallery import TESTS, design  # noqa: E402  (tools/python/gallery.py)

DATA = Path(__file__).resolve().parents[1] / "data"
DATASETS = [("branin", 25), ("hartmann3", 40), ("borehole", 80)]


def main():
    DATA.mkdir(exist_ok=True)
    for name, n in DATASETS:
        X, y, Xt, yt = design.dataset(TESTS[name], n, n_test=200, seed=2024)
        for stem, arr in [("X", X), ("y", y), ("Xt", Xt), ("yt", yt)]:
            np.savetxt(DATA / f"{name}_{stem}.csv", arr.reshape(len(arr), -1), delimiter=",", fmt="%.17g")
        print(f"data/{name}_*.csv: n={n}, d={X.shape[1]}, n_test={len(Xt)}")


if __name__ == "__main__":
    main()
