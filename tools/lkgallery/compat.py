"""What the installed pylibkriging offers, so that one notebook runs against both
the last release and the development head: a feature missing from the installed
version is recorded as SKIP (with the reason), never as a failure."""
import functools
import importlib

import numpy as np


def version():
    import pylibkriging as lk
    v = getattr(lk, "__version__", None)
    if v is None:
        try:
            from importlib.metadata import version as _v
            v = _v("pylibkriging")
        except Exception:  # noqa: BLE001
            v = "unknown"
    return v


@functools.lru_cache(None)
def has(feature):
    """Probe a feature on a tiny problem (cached)."""
    import pylibkriging as lk
    X = np.linspace(0, 1, 6).reshape(-1, 1)
    y = np.sin(3 * X[:, 0])
    probes = {
        "method": lambda: lk.Kriging(y, X, "matern5_2", method="vecchia(3)"),
        "noise": lambda: lk.Kriging(y, X, "matern5_2", noise="nugget"),
        "WarpKriging": lambda: lk.WarpKriging(y, X, ["none"], "matern5_2"),
        "MLPKriging": lambda: lk.MLPKriging(y, X, hidden_dims=[2], d_out=1, kernel="gauss"),
        "NestedKriging": lambda: lk.NestedKriging(y, X, "matern5_2", nb_groups=2),
        "subsetOfData": lambda: lk.Kriging.subsetOfData(X, n_max=3, method="kmeans", seed=1),
        "save": lambda: lk.Kriging(y, X, "matern5_2").save,
        "pickle": lambda: __import__("pickle").dumps(lk.Kriging(y, X, "matern5_2")),
        "NestedKriging.save": lambda: lk.NestedKriging(y, X, "matern5_2", nb_groups=2).save,
        "load(noise)": lambda: _roundtrip(lk, lk.Kriging(y, X, "matern5_2", noise="nugget")),
        "sklearn": lambda: importlib.import_module("pylibkriging.sklearn").KrigingRegressor,
    }
    try:
        probes[feature]()
        return True
    except Exception:  # noqa: BLE001
        return False


def _roundtrip(lk, k):
    import os
    import tempfile
    f = os.path.join(tempfile.mkdtemp(), "k.json")
    k.save(f)
    return lk.load(f)


def optional(module):
    """Import an optional comparison package, or return None."""
    try:
        return importlib.import_module(module)
    except Exception:  # noqa: BLE001 - broken installs count as absent
        return None
