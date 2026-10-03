"""Test setup: stub `cdt.metrics` (tests/legacy_modules.py loads the modules in legacy/).

`cdt` needs R; the legacy files import it only for scoring helpers that the tests do not use.
"""
import sys
import types

import numpy as np


def _install_cdt_stub():
    try:
        import cdt.metrics  # noqa: F401
        return
    except Exception:
        pass
    cdt = types.ModuleType("cdt")
    metrics = types.ModuleType("cdt.metrics")

    def precision_recall(t, p):
        t = np.asarray(t) != 0
        p = np.asarray(p) != 0
        return (float((t & p).sum()) / max(int(p.sum()), 1), None)

    def SHD(t, p, double_for_anticausal=True):
        return int(((np.asarray(t) != 0) != (np.asarray(p) != 0)).sum())

    metrics.precision_recall = precision_recall
    metrics.SHD = SHD
    metrics.SID = lambda *a, **k: 0
    cdt.metrics = metrics
    sys.modules["cdt"] = cdt
    sys.modules["cdt.metrics"] = metrics


_install_cdt_stub()
