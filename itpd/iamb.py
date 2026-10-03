"""Known-order IAMB per target on the unrolled time graph (S2), the "IAMB per target" baseline.

For each target Y = V^n_t (t >= 1), with X = V^n_{t-1} (self edge known, never tested, as in ITPD, blanket-screened shrink and the
full-conditioning baseline) and candidates C = every strictly earlier node except X (full history), IAMB (Tsamardinos, Aliferis, Statnikov 2003)
runs with the known self edge in every conditioning set (F = {X}):
  grow    CMB = [].  Repeat: test every Z in C - CMB given F | CMB (one CI test each); among the dependent ones (p <= alpha)
          add the one with the smallest p (ties, see below); stop when none is dependent.
  shrink  for each member Z of CMB in the order added: remove Z if Z _||_ Y | F | (CMB - Z) (p > alpha), testing against the
          current CMB (single pass; members already removed stay out).
The output parents are CMB. Under a CI oracle and faithfulness the strictly earlier nodes plus Y are ancestrally closed, so the
Markov blanket of the sink Y in them is pa(Y); the grow phase ends with CMB containing pa(Y) (a parent is adjacent to Y, hence
dependent given any set), and the shrink phase removes every non-parent (it is independent of Y given any superset of pa(Y) that
does not contain it). So the output is pa(Y) minus X for every grow order; the oracle check runs three tie rules.

Strength and ties. The recorder returns p-values only. The strongest candidate is the one with the smallest p. Candidates whose
p is equal (the oracle gives 0.0 for every dependent candidate; Fisher-z underflows to 0.0 at large M) are ranked by
-log p of the same Fisher-z statistic when the test object has `neglogp(x, y, S)` (an extra evaluation of an already counted test,
not a new CI test; `FisherZ.neglogp`), and then by `tie`: "last" (default; the candidate with the largest column index, i.e. the
latest time then the largest series index, first), "first", or "random:<seed>" (a fixed random order of the candidates). With finite data, ranking by -log p
agrees with ranking by |partial correlation| except for underflowed p-values.

Counting. Every call goes through the recorder, so the unique-test count is the same one the other methods report. Each grow
iteration re-tests every candidate outside CMB with a new conditioning set; the count is about |C| (|CMB| + 1) + |CMB| per target.
Largest conditioning set = rec.cur_max (includes X). A test that is infeasible (Fisher-z, n - |S| - 3 <= 0) makes the target
undecidable (no parents output, excluded from the score, never answered "independent"), as in the other methods; with sets of
size |CMB| + 1 this needs M <= |CMB| + 4.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .ci import InfeasibleTest
from .itpd import Result


@dataclass
class IAMBResult(Result):
    stats: dict = field(default_factory=dict)


def _rank_key(tie: str, n_cols: int):
    if tie == "last":
        return lambda z: -z
    if tie == "first":
        return lambda z: z
    if tie.startswith("random:"):
        perm = np.random.default_rng(int(tie.split(":")[1])).permutation(n_cols)
        return lambda z: int(perm[z])
    raise ValueError("tie must be last|first|random:<seed>")


def iamb_target(rec, y: int, x: int, cand, alpha: float, rank, stats: dict | None = None) -> list:
    """IAMB for one target. Raises InfeasibleTest from the recorder. Returns CMB after shrink, in the order added."""
    F = [x]
    cmb: list = []
    in_cmb: set = set()
    n_iter = 0
    while True:
        n_iter += 1
        dep = []
        for z in cand:
            if z in in_cmb:
                continue
            p = rec(z, y, F + cmb, "grow")
            if p <= alpha:
                dep.append((p, z))
        if not dep:
            break
        pmin = min(p for p, _ in dep)
        tied = [z for p, z in dep if p == pmin]
        if len(tied) > 1:
            nl = getattr(rec.ci, "neglogp", None)
            if nl is not None:
                S = sorted(F + cmb)
                best = max(nl(z, y, S) for z in tied)
                tied = [z for z in tied if nl(z, y, S) == best]
        z = min(tied, key=rank)
        cmb.append(z)
        in_cmb.add(z)
    n_grown = len(cmb)
    for z in list(cmb):
        rest = [c for c in cmb if c != z]
        if rec(z, y, F + rest, "shrink") > alpha:
            cmb.remove(z)
    if stats is not None:
        stats["grow_iters"] = n_iter
        stats["grown"] = n_grown
        stats["shrunk"] = n_grown - len(cmb)
    return cmb


def run_s2(rec, N: int, T: int, alpha: float, *, tau_max: int | None = None, order: str = "time",
           per_target: bool = True, tie: str = "last") -> Result:
    """Same interface as baselines.full_conditioning.run_s2 (`rec` with infeasible="raise" when per_target)."""
    t0 = time.perf_counter()
    rank = _rank_key(tie, T * N)
    A = np.zeros((T * N, T * N), dtype=np.uint8)
    col = lambda n, t: t * N + n
    pa_pred, per_pair, bad = {}, [], []
    pairs = [(n, t) for n in range(N) for t in range(1, T)] if order == "variable" else \
        [(n, t) for t in range(1, T) for n in range(N)]
    agg = {"grow_iters": 0, "grown": 0, "shrunk": 0}
    for n, t in pairs:
        x, y = col(n, t - 1), col(n, t)
        lo = col(0, max(0, t - tau_max)) if tau_max is not None else 0
        cand = [c for c in range(lo, col(0, t)) if c != x]
        rec.new_scope()
        raw0, uniq0 = rec.mark()
        st: dict = {}
        try:
            parents = iamb_target(rec, y, x, cand, alpha, rank, st)
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
        for k in agg:
            agg[k] += st[k]
        per_pair.append({"pair": (x, y), "n_cand": len(cand), "raw": raw1 - raw0, "new_unique": uniq1 - uniq0,
                         "max_size": rec.cur_max, "parents": [int(p) for p in parents], "grow_iters": st["grow_iters"],
                         "grown": st["grown"], "shrunk": st["shrunk"]})
    return IAMBResult(A_hat=A, pa=pa_pred, per_pair=per_pair, seconds=time.perf_counter() - t0, summary=rec.summary(),
                      infeasible_targets=bad, stats=agg)
