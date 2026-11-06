"""ITPD_naive and ITPD on the unrolled time graph (S2) and as one run per series on a window slice (S1).

S2: one PaDL call per autocorrelation pair (V^n_{t-1}, V^n_t), candidates = all variables at times
[max(0, t - tau_max), t - 1] minus X (full history if tau_max is None). The edge X -> Y is assumed, never tested.
ITPD (reuse=True) additionally skips Z4/Z8 tests with labels from earlier pairs of the same variable:
  adjacency   parents of X are not Z4/Z8 for (X, Y)
  de_z1       adjacent Z1 candidates and their later copies, for later times
  de_z5       Z5,7 candidates and later copies: not Z4, not Z8
  de_z4       Z4 candidates and later copies: not Z8
These are the rules of the original code (legacy/itpd.py); `rules` switches them individually.
Optional rule (not in RULES, so not part of the ITPD of the paper):
  adjacency_self  X's own autocorrelation parent V^n_{t-2} is a parent of X, hence not Z4/Z8 for (X, Y). The adjacency
                  rule above only uses the predicted parents of X, which exclude that parent. Run it as
                  `rules=RULES + ("adjacency_self",)` (method_runner method name "itpd_adjacency_self").

Processing order is `order="variable"` (as in legacy/itpd.py) or `"time"` (as in legacy/itpd_naive.py). The rules only pass information along
one variable's chain, so the output and the test sets do not depend on the order.

S1: `run_s1` works on a window with slices 0..tau (column l * N + n) and runs PaDL once per series n, on
(X, Y) = (V^n_{tau-1}, V^n_tau) with candidates the other tau * N - 1 earlier window nodes. One pair per series means
no cross-pair deduction can fire, so ITPD and ITPD_naive coincide in S1.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .ci import InfeasibleTest
from .padl import padl

RULES = ("adjacency", "de_z1", "de_z5", "de_z4")
OPTIONAL_RULES = ("adjacency_self",)


@dataclass
class Result:
    A_hat: np.ndarray                       # S2: (T*N, T*N); S1: window (tau+1)*N square
    pa: dict                                # target column -> predicted parents (autocorrelation parent not included)
    per_pair: list = field(default_factory=list)
    skips: dict = field(default_factory=dict)
    seconds: float = 0.0
    summary: dict = field(default_factory=dict)
    B_hat: np.ndarray | None = None         # S1 only: (tau+1, N, N), B_hat[l, i, j] = V^i_{t-l} -> V^j_t
    irrelevant: dict | None = None          # final skip sets {col: {"Z4": {cand: reason}, "Z8": {...}}} (keep_labels only)
    infeasible_targets: list = field(default_factory=list)   # target columns whose PaDL call hit an infeasible test


def _new_irrelevant(N: int, T: int) -> dict:
    return {c: {"Z4": {}, "Z8": {}} for c in range(N * T)}


def run_s2(rec, N: int, T: int, alpha: float, *, reuse: bool, tau_max: int | None = None,
           variant: str = "paper", order: str = "variable", rules=RULES, keep_labels: bool = False,
           lazy: bool = False, per_target: bool = False, marginal_first: bool = False) -> Result:
    """`marginal_first` (itpd.padl): Z8 step never skipped, Y-marginal first; the Z4 rules stay in force."""
    if order not in ("variable", "time"):
        raise ValueError("order must be 'variable' or 'time'")
    unknown = set(rules) - set(RULES) - set(OPTIONAL_RULES)
    if unknown:
        raise ValueError(f"unknown rules {sorted(unknown)}")
    t0 = time.perf_counter()
    if order == "variable":
        pairs = [(n, t) for n in range(N) for t in range(1, T)]
    else:
        pairs = [(n, t) for t in range(1, T) for n in range(N)]
    irr = _new_irrelevant(N, T)
    A = np.zeros((T * N, T * N), dtype=np.uint8)
    pa_pred: dict = {}
    skips: dict = {}
    per_pair = []
    bad: list = []
    col = lambda n, t: t * N + n
    for n, t in pairs:
        x, y = col(n, t - 1), col(n, t)
        lo = col(0, max(0, t - tau_max) if tau_max is not None else 0)
        cand = [c for c in range(lo, col(0, t)) if c != x]
        if reuse and "adjacency" in rules and x in pa_pred:
            for p in pa_pred[x]:
                irr[y]["Z4"].setdefault(p, "adjacency")
                irr[y]["Z8"].setdefault(p, "adjacency")
        if reuse and "adjacency_self" in rules and t >= 2 and col(n, t - 2) >= lo:
            irr[y]["Z4"].setdefault(col(n, t - 2), "adjacency_self")
            irr[y]["Z8"].setdefault(col(n, t - 2), "adjacency_self")
        rec.new_scope()
        raw0, uniq0 = rec.mark()
        if per_target:
            # A target whose PaDL call needs an infeasible test (Fisher-z, n - |S| - 3 <= 0) is reported as infeasible:
            # no parents are output for it, its labels are not used to skip tests of later targets, and the test is
            # never answered "independent". `rec` must have infeasible="raise".
            try:
                res = padl(x, y, cand, rec, alpha, irrelevant=irr[y] if reuse else None, variant=variant, skips=skips,
                           lazy=lazy, marginal_first=marginal_first)
            except InfeasibleTest:
                raw1, uniq1 = rec.mark()
                bad.append(y)
                per_pair.append({"pair": (x, y), "n_cand": len(cand), "raw": raw1 - raw0, "new_unique": uniq1 - uniq0,
                                 "max_size": rec.cur_max, "parents": [], "infeasible": True,
                                 "first_infeasible_size": len(rec.last_infeasible[2]) if rec.last_infeasible else None})
                continue
        else:
            res = padl(x, y, cand, rec, alpha, irrelevant=irr[y] if reuse else None, variant=variant, skips=skips, lazy=lazy,
                       marginal_first=marginal_first)
        pa_pred[y] = res.parents
        A[x, y] = 1
        for p in res.parents:
            A[p, y] = 1
        if reuse:
            def later(z):  # z and its later copies, as columns
                zn, zt = z % N, z // N
                return [col(zn, s) for s in range(zt, T)]
            z1 = [c for z in res.z1_z3 for c in later(z)] if "de_z1" in rules else []
            z5 = [c for z in res.z57 for c in later(z)] if "de_z5" in rules else []
            z4 = [c for z in res.z4 for c in later(z)] if "de_z4" in rules else []
            for s in range(t + 1, T):
                yy = col(n, s)
                for c in z1:
                    irr[yy]["Z4"].setdefault(c, "de_z1")
                    irr[yy]["Z8"].setdefault(c, "de_z1")
                for c in z5:
                    irr[yy]["Z4"].setdefault(c, "de_z5")
                    irr[yy]["Z8"].setdefault(c, "de_z5")
                for c in z4:
                    irr[yy]["Z8"].setdefault(c, "de_z4")
        raw1, uniq1 = rec.mark()
        row = {"pair": (x, y), "n_cand": len(cand), "raw": raw1 - raw0, "new_unique": uniq1 - uniq0,
               "max_size": rec.cur_max, "parents": list(res.parents)}
        if keep_labels:
            row["labels"] = res.labels
        per_pair.append(row)
    return Result(A_hat=A, pa=pa_pred, per_pair=per_pair, skips={f"{k[0]}:{k[1]}": v for k, v in skips.items()},
                  seconds=time.perf_counter() - t0, summary=rec.summary(),
                  irrelevant=irr if keep_labels else None, infeasible_targets=bad)


def run_s1(rec, N: int, tau: int, alpha: float, *, variant: str = "paper", lazy: bool = False) -> Result:
    """One PaDL run per series on the last window slice. `rec` is defined on window columns l * N + n."""
    t0 = time.perf_counter()
    W = (tau + 1) * N
    A = np.zeros((W, W), dtype=np.uint8)
    B = np.zeros((tau + 1, N, N), dtype=np.uint8)
    pa_pred = {}
    per_pair = []
    for n in range(N):
        x, y = (tau - 1) * N + n, tau * N + n
        cand = [c for c in range(tau * N) if c != x]
        rec.new_scope()
        raw0, uniq0 = rec.mark()
        res = padl(x, y, cand, rec, alpha, variant=variant, lazy=lazy)
        raw1, uniq1 = rec.mark()
        pa_pred[y] = res.parents
        A[x, y] = 1
        B[1, n, n] = 1
        for p in res.parents:
            A[p, y] = 1
            l, i = p // N, p % N
            B[tau - l, i, n] = 1
        per_pair.append({"pair": (x, y), "n_cand": len(cand), "raw": raw1 - raw0, "new_unique": uniq1 - uniq0,
                         "max_size": rec.cur_max, "parents": list(res.parents)})
    return Result(A_hat=A, pa=pa_pred, per_pair=per_pair, seconds=time.perf_counter() - t0,
                  summary=rec.summary(), B_hat=B)
