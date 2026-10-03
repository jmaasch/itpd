"""Order-based baseline: with the time order known, Z -> Y iff Z is dependent on Y given all other earlier nodes.

One test per candidate: candidates are all earlier nodes (within tau_max) except X = V^n_{t-1}; the edge X -> Y is
assumed as in ITPD. Conditioning set = (candidates minus Z) plus X, i.e. every other earlier node in the window.
This is the one-test-per-candidate scheme of Shiragur et al. (2024) and Franquesa Mones et al. (2026).
"""
from __future__ import annotations

import time

import numpy as np

from ..ci import InfeasibleTest
from ..itpd import Result


def run_s2(rec, N: int, T: int, alpha: float, *, tau_max: int | None = None, order: str = "variable",
           per_target: bool = False) -> Result:
    """per_target=True (rec with infeasible="raise"): a target with an infeasible test is reported infeasible, no parents
    are output for it and the test is never answered "independent"."""
    t0 = time.perf_counter()
    A = np.zeros((T * N, T * N), dtype=np.uint8)
    col = lambda n, t: t * N + n
    pa_pred, per_pair, bad = {}, [], []
    pairs = [(n, t) for n in range(N) for t in range(1, T)] if order == "variable" else \
        [(n, t) for t in range(1, T) for n in range(N)]
    for n, t in pairs:
        x, y = col(n, t - 1), col(n, t)
        lo = col(0, max(0, t - tau_max)) if tau_max is not None else 0
        window = list(range(lo, col(0, t)))
        cand = [c for c in window if c != x]
        rec.new_scope()
        raw0, uniq0 = rec.mark()
        try:
            parents = [z for z in cand if rec(y, z, [s for s in window if s != z], "order") <= alpha]
        except InfeasibleTest:
            if not per_target:
                raise
            raw1, uniq1 = rec.mark()
            bad.append(y)
            per_pair.append({"pair": (x, y), "n_cand": len(cand), "raw": raw1 - raw0, "new_unique": uniq1 - uniq0,
                             "max_size": rec.cur_max, "parents": [], "infeasible": True,
                             "first_infeasible_size": len(rec.last_infeasible[2]) if rec.last_infeasible else None})
            continue
        raw1, uniq1 = rec.mark()
        pa_pred[y] = parents
        A[x, y] = 1
        for p in parents:
            A[p, y] = 1
        per_pair.append({"pair": (x, y), "n_cand": len(cand), "raw": raw1 - raw0, "new_unique": uniq1 - uniq0,
                         "max_size": rec.cur_max, "parents": parents})
    return Result(A_hat=A, pa=pa_pred, per_pair=per_pair, seconds=time.perf_counter() - t0, summary=rec.summary(),
                  infeasible_targets=bad)


def run_s1(rec, N: int, tau: int, alpha: float) -> Result:
    t0 = time.perf_counter()
    W = (tau + 1) * N
    A = np.zeros((W, W), dtype=np.uint8)
    B = np.zeros((tau + 1, N, N), dtype=np.uint8)
    pa_pred, per_pair = {}, []
    window = list(range(tau * N))
    for n in range(N):
        x, y = (tau - 1) * N + n, tau * N + n
        cand = [c for c in window if c != x]
        rec.new_scope()
        raw0, uniq0 = rec.mark()
        parents = [z for z in cand if rec(y, z, [s for s in window if s != z], "order") <= alpha]
        raw1, uniq1 = rec.mark()
        pa_pred[y] = parents
        A[x, y] = 1
        B[1, n, n] = 1
        for p in parents:
            A[p, y] = 1
            B[tau - p // N, p % N, n] = 1
        per_pair.append({"pair": (x, y), "n_cand": len(cand), "raw": raw1 - raw0, "new_unique": uniq1 - uniq0,
                         "max_size": rec.cur_max, "parents": parents})
    return Result(A_hat=A, pa=pa_pred, per_pair=per_pair, seconds=time.perf_counter() - t0, summary=rec.summary(),
                  B_hat=B)
