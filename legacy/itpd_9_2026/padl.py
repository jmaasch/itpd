"""PaDL (local causal partitioning), one call = parents of Y given the autocorrelation edge X -> Y.

Steps (Algorithm 1 of the paper, in the order of the original code):
 1. Z8:   Z indep X and Z indep Y (marginals).                          label z8x, z8y
 2. Z5,7: Z dep Y and Z indep Y given X.                                 label z57y (marginal if not memoized), z57
 3. Z4:   Z indep X and Z dep X given Y.                                 label z4x (marginal if not memoized), z4
 4. adjacency for the unlabelled set z': Z dep Y given X, Z4 and z' \\ Z.  label adj (+ adjx, adjy in the repo variant)
 5. parents inside Z4: Z dep Y given the other Z4, the adjacent set and X. label z4par

variant="paper": step 4 decides on the conditional test only (Algorithm 1).
variant="repo":  step 4 also requires X dep Z and Y dep Z marginally, as `test_adj_y` does in legacy/padl_itpd.py and
                 legacy/padl_naive.py; this loses a true Z4 parent whose Z4 test was skipped.

`irrelevant` (ITPD only) maps "Z4" / "Z8" to {candidate: reason}; the test of that partition is skipped for those
candidates. `skips` (optional dict) counts the skips used, keyed (partition, reason), and the marginal results reused
inside one call, keyed ("memo", "marginal").
Marginal results are memoized inside one call, exactly like the original code, so raw counts match it.
lazy=False evaluates both tests of a step even when the first already decides (the original code does); lazy=True
short-circuits (not issued: the conditional test of a step whose marginal already rules the label out).
marginal_first=True ("ITPD + marginal-first"): the Z8 step is never skipped (`irrelevant["Z8"]` is ignored) and
evaluates the Y-marginal first; the X-marginal is issued only if the Y-marginal does not reject (lazy) or always
(non-lazy). The labels are the same as without the flag; only the tests issued differ. Default False.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PaDLResult:
    parents: list
    z1_z3: list = field(default_factory=list)   # adjacent candidates among z'
    z4: list = field(default_factory=list)
    z4_parents: list = field(default_factory=list)
    z57: list = field(default_factory=list)     # named Z7 in legacy/padl_*.py; it is Z5 here (Z7 is empty)
    z8: list = field(default_factory=list)
    z_prime: list = field(default_factory=list)
    labels: dict = field(default_factory=dict)


def padl(x: int, y: int, cand, rec, alpha: float, *, irrelevant: dict | None = None,
         variant: str = "paper", skips: dict | None = None, lazy: bool = False,
         marginal_first: bool = False) -> PaDLResult:
    """Parents of the target y given the known edge x -> y among the candidates `cand`; `rec` is the Recorder."""
    if variant not in ("paper", "repo"):
        raise ValueError("variant must be 'paper' or 'repo'")
    irr4 = irrelevant["Z4"] if irrelevant else {}
    irr8 = irrelevant["Z8"] if irrelevant else {}
    mx: dict = {}
    my: dict = {}

    def memo_hit():
        if skips is not None:
            skips[("memo", "marginal")] = skips.get(("memo", "marginal"), 0) + 1

    def px(z, lab):
        if z not in mx:
            mx[z] = rec(x, z, [], lab)
        else:
            memo_hit()
        return mx[z]

    def py(z, lab):
        if z not in my:
            my[z] = rec(y, z, [], lab)
        else:
            memo_hit()
        return my[z]

    r = PaDLResult(parents=[])
    for z in cand:
        if z == x or z == y:
            continue
        if marginal_first:
            b8 = py(z, "z8y")
            a8 = px(z, "z8x") if not (lazy and b8 <= alpha) else 0.0
            if a8 > alpha and b8 > alpha:
                r.z8.append(z)
                r.labels[z] = "Z8"
                continue
        elif z not in irr8:
            a8 = px(z, "z8x")
            b8 = py(z, "z8y") if not (lazy and a8 <= alpha) else 0.0
            if a8 > alpha and b8 > alpha:
                r.z8.append(z)
                r.labels[z] = "Z8"
                continue
        elif skips is not None:
            k = ("Z8", irr8[z])
            skips[k] = skips.get(k, 0) + 1
        b57 = py(z, "z57y")
        c57 = rec(y, z, [x], "z57") if not (lazy and b57 > alpha) else 0.0
        if b57 <= alpha and c57 > alpha:
            r.z57.append(z)
            r.labels[z] = "Z5,7"
            continue
        if z not in irr4:
            a4 = px(z, "z4x")
            c4 = rec(x, z, [y], "z4") if not (lazy and a4 <= alpha) else 1.0
            if a4 > alpha and c4 <= alpha:
                r.z4.append(z)
                r.labels[z] = "Z4"
                continue
        elif skips is not None:
            k = ("Z4", irr4[z])
            skips[k] = skips.get(k, 0) + 1
        r.z_prime.append(z)

    base = r.z_prime + r.z4 + [x]
    for z in r.z_prime:
        if variant == "repo":
            a, b = px(z, "adjx"), py(z, "adjy")
        S = [s for s in base if s != z]
        pc = rec(y, z, S, "adj")
        adj = pc <= alpha and (variant == "paper" or (a <= alpha and b <= alpha))
        if adj:
            r.z1_z3.append(z)
            r.labels[z] = "Z1"
        else:
            r.labels[z] = "not identifiable"

    for z in r.z4:
        S = [s for s in r.z4 if s != z] + r.z1_z3 + [x]
        if rec(z, y, S, "z4par") <= alpha:
            r.z4_parents.append(z)
    r.parents = r.z1_z3 + r.z4_parents
    return r
