"""Persistence checks shared by the notebooks of every model class: a model saved and
loaded back (class-specific and generic ``pylibkriging.load``), pickled, or deep-copied
must predict exactly as the original."""
import copy
import pickle
import tempfile
from pathlib import Path

import numpy as np

from . import compat


def check(suite, name, k, Xn, predict):
    """Record the persistence checks of model ``k`` (label ``name``) on the points ``Xn``.

    ``predict(model, Xn)`` returns (mean, stdev) as 1-D arrays; it carries the
    prediction method of an approximate fit (Vecchia / Nystrom)."""
    import pylibkriging as lk
    m0, s0 = predict(k, Xn)
    cls = type(k)
    path = Path(tempfile.mkdtemp()) / f"{name.replace(' ', '_')}.json"
    if not hasattr(k, "save"):
        suite.skip(f"{name}: save + load", "no save() in this pylibkriging")
    else:
        with suite.runs(f"{name}: save + load"):
            k.save(str(path))
            k1 = cls.load(str(path))
            m1, s1 = predict(k1, Xn)
            suite.close(f"{name}: load(save(k)) predicts the same mean", m1, m0, rtol=1e-12, atol=1e-12)
            suite.close(f"{name}: load(save(k)) predicts the same stdev", s1, s0, rtol=1e-10, atol=1e-12)
            if hasattr(k, "theta"):
                suite.close(f"{name}: same theta after load", np.ravel(k1.theta()), np.ravel(k.theta()),
                            rtol=1e-14, atol=0)
        if hasattr(lk, "load") and (compat.has("load(noise)") or "noise" not in name and "nugget" not in name):
            with suite.runs(f"{name}: generic pylibkriging.load"):
                k2 = lk.load(str(path))
                suite.true(f"{name}: generic load returns {cls.__name__}", type(k2).__name__ == cls.__name__,
                           detail=type(k2).__name__)
                suite.close(f"{name}: generic load predicts the same", predict(k2, Xn)[0], m0, rtol=1e-12,
                            atol=1e-12)
    if not compat.has("pickle"):
        suite.skip(f"{name}: pickle / deepcopy", "models not picklable in this pylibkriging")
        return
    with suite.runs(f"{name}: pickle round trip"):
        k3 = pickle.loads(pickle.dumps(k))
        suite.close(f"{name}: pickle predicts the same", predict(k3, Xn)[0], m0, rtol=1e-12, atol=1e-12)
    with suite.runs(f"{name}: deepcopy"):
        k4 = copy.deepcopy(k)
        suite.close(f"{name}: deepcopy predicts the same", predict(k4, Xn)[0], m0, rtol=1e-12, atol=1e-12)
