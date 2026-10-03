"""Hint-pruned verification (HPV) on the unrolled time graph (S2), time-major.

For each target Y = V^n_t (t >= 1), with X = V^n_{t-1} (self edge assumed, never tested, as in ITPD and the order-based
baseline), candidates C = every earlier node except X (full history), forced set F = {X}:
  phase A (prune)   R = {Z in C : not (Z _||_ Y | S_A(Z))} at alpha_A,  S_A(Z) = (H(Z) | {X}) - {Z};
  phase B (verify)  O = {Z in R : not (Z _||_ Y | (R - {Z}) | {X})} at alpha_B; O are the parents of Y besides X.
Re-check (`recheck=True`, the always-verify safe variant): A' = {Z in C - R : not (Z _||_ Y | (O | {X}) - {Z})}
at alpha_B, then the output is verify(R | A') at alpha_B, issued even when A' is empty. With one dataset and a deterministic
(memoised) test, verify(R) for A' = {} re-issues exactly the phase-B tests and returns O, so the always-verify output equals
the shortcut's ("return O if A' is empty"); only raw calls differ (by |R|). `stats["shortcut_diff"]` counts targets where
they would differ (expected 0).

All targets at time t are decided before any target at t + 1; the hints for time t come from G_hat, the graph learned over
the nodes at times < t (outputs of earlier slices plus the assumed self edges; an infeasible earlier target contributes its
self edge only). Hints H(Z):
  "learned_blanket"  Markov blanket of Z in G_hat: parents, children, children's other parents (all at times < t);
  "shifted_parents"  the learned parents of X shifted one step forward ({V^j_{s+1} : V^j_s in pa_hat(X)}), the same for every Z;
  "union"            learned_blanket | shifted_parents;
  "oracle_blanket"   the Markov blanket of Z in the TRUE graph over the nodes at times < t (`A_true`): an upper bound that
                     removes error propagation through G_hat;
  "none"             the empty hint: S_A = {X} (a screen given X only).
Result files written before the rename carry the hint strings "mb", "shift", "oracle_mb" (methods_registry.OLD_HINT_TO_NEW);
`run_s2` accepts both.
`A_true` is read by the algorithm only for hint="oracle_blanket"; otherwise only for diagnostics written to per_pair
(survivors that are not true parents, true parents lost in phase A, phase-A tests of the true parents).

Cap (`cap`: None, an int, or "auto" = n - 4 for a test object with `n` samples, so every phase-A test is feasible for
Fisher-z, which needs n >= |S| + 4): when |S_A(Z)| > cap, S_A keeps X and the cap - 1 most recent hint members (largest
column index first: latest time, then largest series index); every truncation is counted (`n_capped`, `capped_dropped`).
The cap does not touch phase B, the re-check or the final verification, so it cannot make a target decidable that has a
large survivor set.

Infeasible tests (the recorder raises `InfeasibleTest`; never answered "independent"):
  phase A    the candidate is kept in R (cannot be pruned without a test) and counted in `a_inf`;
  phase B, re-check, final verification   the target is infeasible: no parents output, excluded from its own score,
             counted (`infeasible_targets`), as in itpd.run_s2(per_target=True) and the order-based baseline.
Recorder labels: "A" (phase A), "B" (phase B), "C" (re-check), "V" (final verification); unique tests are attributed to
the label that issued them first, so "V" tests that repeat phase B count as raw calls only.
HPV has no lazy / non-lazy distinction: every issued test can change the output.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .ci import InfeasibleTest
from .itpd import Result
from .methods_registry import OLD_HINT_TO_NEW

HINTS = ("learned_blanket", "shifted_parents", "union", "oracle_blanket", "none")


@dataclass
class HPVResult(Result):
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


def run_s2(rec, N: int, T: int, alpha_A: float, alpha_B: float | None = None, *, hint: str = "learned_blanket",
           recheck: bool = False, cap=None, A_true: np.ndarray | None = None, per_target: bool = True,
           keep_parent_tests: bool = False) -> HPVResult:
    hint = OLD_HINT_TO_NEW.get(hint, hint)
    if hint not in HINTS:
        raise ValueError(f"hint must be one of {HINTS}")
    if hint == "oracle_blanket" and A_true is None:
        raise ValueError("hint='oracle_blanket' needs A_true")
    alpha_B = alpha_A if alpha_B is None else alpha_B
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
        if hint in ("learned_blanket", "union"):
            mb = blankets(pa_hat, ch_hat, past)
            bnd = {z: blanket_bound(pa_hat, ch_hat, z) for z in past}
        elif hint == "oracle_blanket":
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
            shift = {p + N for p in pa_hat[x]} if hint in ("shifted_parents", "union") else set()
            rec.new_scope()
            raw0, uniq0 = rec.mark()
            row = {"pair": (x, y), "n_cand": len(cand), "a_inf": 0, "n_capped": 0}
            ptests = [] if (keep_parent_tests and pa_true is not None) else None
            tp = pa_true[y] - {x} if pa_true is not None else None
            try:
                # ---- phase A
                R, max_sa = [], 0
                for z in cand:
                    if hint == "none":
                        H = set()
                    elif hint == "shifted_parents":
                        H = shift
                    elif hint == "union":
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
                    if p <= alpha_A:
                        R.append(z)
                    if ptests is not None and z in tp:
                        ptests.append([int(z), sorted(int(s) for s in S), float(p)])
                # ---- phase B
                def verify(Rl, label):
                    out, mx = [], 0
                    for z in Rl:
                        S = [r for r in Rl if r != z] + [x]
                        mx = max(mx, len(S))
                        if rec(z, y, S, label) <= alpha_B:
                            out.append(z)
                    return out, mx
                O, max_sb = verify(R, "B")
                parents, n_ap = O, 0
                if recheck:
                    Rs = set(R)
                    Os = set(O)
                    Ap = [z for z in cand if z not in Rs
                          and rec(z, y, sorted((Os | {x}) - {z}), "C") <= alpha_B]
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
    return HPVResult(A_hat=A, pa=pa_pred, per_pair=per_pair, seconds=time.perf_counter() - t0, summary=rec.summary(),
                     infeasible_targets=bad, stats=st)
