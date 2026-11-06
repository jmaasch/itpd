"""ITPD-S and ITPD-S+ (blanket-screened shrink) on the unrolled time graph (S2), time-major.

ITPD-S is the single pass (S = shrink); ITPD-S+ is ITPD-S with the re-check (`recheck=True`).

For each target Y = V^n_t (t >= 1), with X = V^n_{t-1} (self edge assumed, never tested, as in ITPD and the full-conditioning
baseline), candidates C = every earlier node except X (full history), forced set F = {X}:
  screening step  R = {Z in C : not (Z _||_ Y | S_A(Z))} at alpha_scr,  S_A(Z) = (B(Z) | {X}) - {Z};
  shrink step     O = {Z in R : not (Z _||_ Y | (R - {Z}) | {X})} at alpha_shr; O are the parents of Y besides X.
The shrink step is the shrink phase of Grow-Shrink and IAMB (Margaritis and Thrun 2000; Tsamardinos et al. 2003).
Re-check (`recheck=True`), the forward-backward second pass: A' = {Z in C - R : not (Z _||_ Y | (O | {X}) - {Z})} at alpha_shr, then the
output is shrink(R | A') at alpha_shr, issued even when A' is empty. With one dataset and a deterministic (memoised) test, shrink(R) for
A' = {} re-issues exactly the shrink-step tests and returns O, so the output equals the shortcut's ("return O if A' is empty");
only raw calls differ (by |R|). `stats["shortcut_diff"]` counts targets where they would differ (expected 0).

All targets at time t are decided before any target at t + 1; the blanket B(Z) for time t comes from G_hat, the graph learned
over the nodes at times < t (outputs of earlier slices plus the assumed self edges; an infeasible earlier target contributes
its self edge only). The argument `screening` chooses B(Z), the screening conditioning set besides X:
  "learned_blanket"  Markov blanket of Z in G_hat: parents, children, children's other parents (all at times < t);
  "shifted_parents"  the learned parents of X shifted one step forward ({V^j_{s+1} : V^j_s in pa_hat(X)}), the same for every Z;
  "blanket_shifted"  learned_blanket | shifted_parents;
  "true_blanket"     the Markov blanket of Z in the TRUE graph over the nodes at times < t (`A_true`): an oracle that
                     removes any error of G_hat from the screening sets;
  "own_lag"          the empty set: S_A = {X}, the previous value of the same series alone (an autoregressive, Granger-style test).
The two levels are `alpha_scr` (screening step) and `alpha_shr` (shrink step); screen-and-clean is a liberal alpha_scr with a strict alpha_shr.
Result files written before the renames carry the strings "mb", "shift", "oracle_mb", "none", "union", "oracle_blanket" and the
keys alpha_A, alpha_B (methods_registry.OLD_SCREENING_TO_NEW, OLD_OPTION_KEYS); `run_s2` accepts the earlier strings.
`A_true` is read by the algorithm only for screening="true_blanket"; otherwise only for diagnostics written to per_pair
(survivors that are not true parents, true parents lost in the screening step, screening tests of the true parents).

Cap (`cap`: None, an int, or "auto" = n - 4 for a test object with `n` samples, so every screening test is feasible for
Fisher-z, which needs n >= |S| + 4): when |S_A(Z)| > cap, S_A keeps X and the cap - 1 most recent members of B(Z)
(largest column index first: latest time, then largest series index); every truncation is counted (`n_capped`,
`capped_dropped`). The cap does not touch the shrink step, the re-check or the final shrink, so it cannot make a target
decidable that has a large survivor set.

Infeasible tests (the recorder raises `InfeasibleTest`; never answered "independent"):
  screening step                      the candidate is kept in R (cannot be pruned without a test) and counted in `a_inf`;
  shrink step, re-check, final shrink the target is infeasible: no parents output, excluded from its own score,
                                      counted (`infeasible_targets`), as in itpd.run_s2(per_target=True) and the
                                      full-conditioning baseline.
Recorder labels: "A" (screening step), "B" (shrink step), "C" (re-check), "V" (final shrink); unique tests are attributed to
the label that issued them first, so "V" tests that repeat shrink-step tests count as raw calls only.
ITPD-S and ITPD-S+ have no lazy / non-lazy distinction: every issued test can change the output.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .ci import InfeasibleTest
from .itpd import Result
from .methods_registry import OLD_SCREENING_TO_NEW

SCREENING_SETS = ("learned_blanket", "shifted_parents", "blanket_shifted", "true_blanket", "own_lag")


@dataclass
class ShrinkResult(Result):
    stats: dict = field(default_factory=dict)


def blankets(pa: list, ch: list, nodes) -> dict:
    """Markov blanket of every node in `nodes` within the graph given by parent sets `pa` and child sets `ch`
    (both restricted to `nodes` by the caller): pa(Z) | ch(Z) | pa(ch(Z)), minus Z."""
    out = {}
    for z in nodes:
        b = set(pa[z]) | ch[z]
        for c in ch[z]:
            b |= pa[c]
        b.discard(z)
        out[z] = b
    return out


def blanket_bound(pa: list, ch: list, z: int) -> int:
    """b(Z) = |pa(Z)| + sum over children c of |pa(c)| (an upper bound on |MB(Z)|)."""
    return len(pa[z]) + sum(len(pa[c]) for c in ch[z])


def _true_graph(A_true: np.ndarray, n_nodes: int):
    A = np.asarray(A_true) != 0
    pa = [set(np.nonzero(A[:, j])[0].tolist()) for j in range(n_nodes)]
    return pa


def run_s2(rec, N: int, T: int, alpha_scr: float, alpha_shr: float | None = None, *, screening: str = "learned_blanket",
           recheck: bool = False, cap=None, A_true: np.ndarray | None = None, per_target: bool = True,
           keep_parent_tests: bool = False) -> ShrinkResult:
    screening = OLD_SCREENING_TO_NEW.get(screening, screening)
    if screening not in SCREENING_SETS:
        raise ValueError(f"screening must be one of {SCREENING_SETS}")
    if screening == "true_blanket" and A_true is None:
        raise ValueError("screening='true_blanket' needs A_true")
    alpha_shr = alpha_scr if alpha_shr is None else alpha_shr
    if cap == "auto":
        cap = int(getattr(rec.ci, "n")) - 4
    t0 = time.perf_counter()
    D = T * N
    col = lambda n, t: t * N + n
    A = np.zeros((D, D), dtype=np.uint8)
    pa_hat = [set() for _ in range(D)]          # learned parents incl. the assumed self edge
    ch_hat = [set() for _ in range(D)]
    pa_true = _true_graph(A_true, D) if A_true is not None else None
    pa_pred, per_pair, bad = {}, [], []
    st = {"a_inf": 0, "n_capped": 0, "capped_dropped": 0, "sum_R": 0, "sum_cand": 0, "sum_Aprime": 0,
          "targets_Aprime": 0, "shortcut_diff": 0, "excess": 0, "lost": 0, "max_SA": 0, "max_SB": 0}

    for t in range(1, T):
        lo_t = col(0, t)                          # nodes at times < t are 0 .. lo_t - 1
        past = range(lo_t)
        mb = bnd = None
        if screening in ("learned_blanket", "blanket_shifted"):
            mb = blankets(pa_hat, ch_hat, past)
            bnd = {z: blanket_bound(pa_hat, ch_hat, z) for z in past}
        elif screening == "true_blanket":
            ch_t = [set() for _ in range(lo_t)]
            pa_t = [pa_true[z] for z in range(lo_t)]
            for c in range(lo_t):
                for p in pa_t[c]:
                    ch_t[p].add(c)
            mb = blankets(pa_t, ch_t, past)
            bnd = {z: blanket_bound(pa_t, ch_t, z) for z in past}
        new = {}
        for n in range(N):
            x, y = col(n, t - 1), col(n, t)
            cand = [c for c in past if c != x]
            shift = {p + N for p in pa_hat[x]} if screening in ("shifted_parents", "blanket_shifted") else set()
            rec.new_scope()
            raw0, uniq0 = rec.mark()
            row = {"pair": (x, y), "n_cand": len(cand), "a_inf": 0, "n_capped": 0}
            ptests = [] if (keep_parent_tests and pa_true is not None) else None
            tp = pa_true[y] - {x} if pa_true is not None else None
            try:
                # ---- screening step
                R, max_sa = [], 0
                for z in cand:
                    if screening == "own_lag":
                        H = set()
                    elif screening == "shifted_parents":
                        H = shift
                    elif screening == "blanket_shifted":
                        H = mb[z] | shift
                    else:
                        H = mb[z]
                    S = (H | {x}) - {z}
                    if cap is not None and len(S) > cap:
                        rest = sorted(S - {x}, reverse=True)
                        keep = rest[: max(cap - 1, 0)]
                        S2 = set(keep) | ({x} if cap >= 1 else set())
                        row["n_capped"] += 1
                        st["capped_dropped"] += len(S) - len(S2)
                        S = S2
                    max_sa = max(max_sa, len(S))
                    try:
                        p = rec(z, y, sorted(S), "A")
                    except InfeasibleTest:
                        row["a_inf"] += 1
                        R.append(z)
                        if ptests is not None and z in tp:
                            ptests.append([int(z), sorted(int(s) for s in S), None])
                        continue
                    if p <= alpha_scr:
                        R.append(z)
                    if ptests is not None and z in tp:
                        ptests.append([int(z), sorted(int(s) for s in S), float(p)])
                # ---- shrink step
                def verify(Rl, label):
                    out, mx = [], 0
                    for z in Rl:
                        S = [r for r in Rl if r != z] + [x]
                        mx = max(mx, len(S))
                        if rec(z, y, S, label) <= alpha_shr:
                            out.append(z)
                    return out, mx
                O, max_sb = verify(R, "B")
                parents, n_ap = O, 0
                if recheck:
                    Rs = set(R)
                    Os = set(O)
                    Ap = [z for z in cand if z not in Rs
                          and rec(z, y, sorted((Os | {x}) - {z}), "C") <= alpha_shr]
                    n_ap = len(Ap)
                    final, mx2 = verify(sorted(R + Ap), "V")
                    max_sb = max(max_sb, mx2)
                    if not Ap and final != O:          # cannot happen with a deterministic test (see docstring)
                        st["shortcut_diff"] += 1
                    parents = final
            except InfeasibleTest:
                if not per_target:
                    raise
                raw1, uniq1 = rec.mark()
                bad.append(y)
                row.update({"raw": raw1 - raw0, "new_unique": uniq1 - uniq0, "max_size": rec.cur_max, "parents": [],
                            "infeasible": True,
                            "first_infeasible_size": len(rec.last_infeasible[2]) if rec.last_infeasible else None})
                st["a_inf"] += row["a_inf"]
                st["n_capped"] += row["n_capped"]
                per_pair.append(row)
                new[y] = (x, None)
                continue
            raw1, uniq1 = rec.mark()
            row.update({"raw": raw1 - raw0, "new_unique": uniq1 - uniq0, "max_size": rec.cur_max,
                        "max_SA": max_sa, "max_SB": max_sb, "n_R": len(R), "n_Aprime": n_ap,
                        "parents": [int(p) for p in parents]})
            if bnd is not None:
                row["b_max"] = int(max((bnd[z] for z in cand), default=0))
            if tp is not None:
                Rs = set(R)
                row["excess"] = len(Rs - tp)
                row["lost"] = len(tp - Rs)
                st["excess"] += row["excess"]
                st["lost"] += row["lost"]
            if ptests is not None:
                row["ptests"] = ptests
            for k in ("a_inf", "n_capped"):
                st[k] += row[k]
            st["sum_R"] += len(R)
            st["sum_cand"] += len(cand)
            st["sum_Aprime"] += n_ap
            st["targets_Aprime"] += int(n_ap > 0)
            st["max_SA"] = max(st["max_SA"], max_sa)
            st["max_SB"] = max(st["max_SB"], max_sb)
            per_pair.append(row)
            new[y] = (x, parents)
        # commit the slice: G_hat now covers the nodes at times <= t
        for y, (x, parents) in new.items():
            pa_hat[y] = {x} | (set(parents) if parents is not None else set())
            for p in pa_hat[y]:
                ch_hat[p].add(y)
            if parents is not None:
                pa_pred[y] = list(parents)
                A[x, y] = 1
                for p in parents:
                    A[p, y] = 1
    st["n_infeasible_targets"] = len(bad)
    return ShrinkResult(A_hat=A, pa=pa_pred, per_pair=per_pair, seconds=time.perf_counter() - t0, summary=rec.summary(),
                     infeasible_targets=bad, stats=st)
