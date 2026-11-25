"""ITPD with single lines of Algorithm 1 changed, for the ablation tables.

Switches names the changes; its defaults are Algorithm 1. ITPDAblation is
ITPD with these changes. exhaustive_check runs it with a d-separation oracle
on every time graph of a small size.
"""
import math
from dataclasses import dataclass, fields

import numpy as np
from scipy.special import ndtr

from .. import ITPD, DSeparation, NotEvaluable
from ..graphs import self_edge_mask

# exhaustive_check: a config counts the enumerated graphs in blocks of this many masks.
BLOCK = 65536


@dataclass(frozen=True)
class Switches:
    """Changes to single lines of Algorithm 1. The defaults are Algorithm 1."""
    candidate_order: str = "descending"   # Line 6: "descending" | "ascending" time
    ancestor_gate: bool = True            # Line 7; False: Line 8 runs for every candidate
    marginal_test: bool = True            # Line 8; False: Lines 7 and 8 removed
    line9: str = "S"                      # "S" | "X" (given X only) | "off"
    dropped_lines: bool = False           # True: Lines 9' and 10' run before Line 9
    markov_test: bool = True              # Line 10; False: the candidate enters S
    line11: str = "X"                     # "X" | "same_step" (X and the members of S at the step of X)
    phase2: bool = True                   # False: S is returned
    phase2_order: str = "ascending"       # Line 13: "ascending" | "descending" time
    phase1_level: str = "alpha1"          # Lines 8, 9, 10: "alpha1" | "alpha"
    phase2_level: str = "alpha/m"         # Line 14: "alpha/m" | "alpha" | "extended_bic"

    VALUES = {"candidate_order": ("descending", "ascending"), "line9": ("S", "X", "off"), "line11": ("X", "same_step"),
              "phase2_order": ("ascending", "descending"), "phase1_level": ("alpha1", "alpha"),
              "phase2_level": ("alpha/m", "alpha", "extended_bic")}

    def __post_init__(self):
        for f in fields(self):
            value, allowed = getattr(self, f.name), self.VALUES.get(f.name, (True, False))
            if type(value) is not type(allowed[0]) or value not in allowed:
                raise ValueError(f"switch {f.name}: {value!r} is not one of {allowed}")


def extended_bic_level(M: int, m: int) -> float:
    """Return the extended-BIC level P(chi2_1 > log(M m)) for M observations and m candidates."""
    return float(2.0 * ndtr(-math.sqrt(math.log(M) + math.log(max(m, 1)))))


class ITPDAblation(ITPD):
    """ITPD with the changes of switches (a Switches, or its fields as a dict).

    screened_sizes_ maps a target to |S| before Phase II, X included. A
    target with a test that could not be evaluated in Phase I has no entry."""

    def __init__(self, *args, switches=Switches(), **kwargs):
        super().__init__(*args, **kwargs)
        self.switches = switches if isinstance(switches, Switches) else Switches(**switches)

    def fit(self, V=None, *, n_variables=None):
        self.screened_sizes_ = {}
        return super().fit(V, n_variables=n_variables)

    def _levels(self, t):
        alpha1, alpha2 = super()._levels(t)
        if self.switches.phase1_level == "alpha":
            alpha1 = self.alpha
        if self.switches.phase2_level == "alpha":
            alpha2 = self.alpha
        elif self.switches.phase2_level == "extended_bic":
            alpha2 = extended_bic_level(self.M_, (t + 1) * self.N_ - 1)
        return alpha1, alpha2

    def _discover(self, x, y, t):
        sw, N = self.switches, self.N_
        alpha1, alpha2 = self._levels(t)
        try:
            S = {x}
            ancestor_prefix = self._prefix[x].copy()
            # Line 6.
            for s in range(t, -1, -1) if sw.candidate_order == "descending" else range(t + 1):
                for n in range(N):
                    z = s * N + n
                    if z == x:
                        continue
                    # Lines 7-8.
                    if (sw.marginal_test and (s > ancestor_prefix[n] or not sw.ancestor_gate)
                            and self._independent(z, y, level=alpha1)):
                        continue
                    # Lines 9' and 10': Z and X are tested at the nominal level.
                    if (sw.dropped_lines and not self._independent(z, x, level=self.alpha)
                            and self._independent(z, y, {x}, level=alpha1)):
                        continue
                    # Line 9.
                    if sw.line9 != "off" and self._independent(z, y, S if sw.line9 == "S" else {x}, level=alpha1):
                        continue
                    # Line 10.
                    if sw.markov_test and self._independent(z, y, self.adjacency_[z], level=alpha1):
                        continue
                    S.add(z)
                    np.maximum(ancestor_prefix, self._prefix[z], out=ancestor_prefix)
            self.screened_sizes_[y] = len(S)
            if not sw.phase2:
                return set(S)
            # Line 11.
            parents = {x} if sw.line11 == "X" else {z for z in S if z // N == t}
            remaining = S.copy()
            # Lines 13-15.
            for z in sorted(S - parents, reverse=sw.phase2_order == "descending"):
                if self._independent(z, y, remaining - {z}, phase="II", level=alpha2):
                    remaining.remove(z)
                else:
                    parents.add(z)
        except NotEvaluable:
            self.untested_targets_.append(y)
            return {x}
        return parents


def exhaustive_check(switches, N: int, T: int, masks, *, latent_first_step: bool = False) -> dict:
    """Run ITPD with a d-separation oracle on enumerated time graphs and count the exact outputs.

    The graphs are the time graphs of N series and T steps with every lag-1
    self edge and no edge inside a time step. The other pairs (u, v) with u
    at an earlier step than v are the optional edges, in increasing order
    of (u, v); bit i of a mask says whether optional edge i is present.

    Parameters
    ----------
    switches : Switches, dict or None
        The changes to Algorithm 1; None runs the class ITPD.
    masks : iterable of int
        The graphs to check.
    latent_first_step : bool
        False: every step is observed, and an output is exact when it is
        the true graph. True: the method sees steps 2 to T and the oracle
        answers on the full graph; a target is judged when all its parents
        are observed, and an output is exact when every judged target has
        its true parents.

    Returns
    -------
    dict
        graphs, exact (number of graphs with an exact output),
        wrong_targets, judged_targets, requested_phase1, requested_phase2
        (CI tests), largest_set, and first_failure: None, or [number of
        optional edges, mask] of the failing graph with the fewest optional
        edges and then the smallest mask.
    """
    size, shift = N * T, N if latent_first_step else 0
    optional = [(u, v) for u in range(size) for v in range(size) if u // N < v // N and v != u + N]
    V = np.zeros((2, T - bool(shift), N))
    out = dict(graphs=0, exact=0, wrong_targets=0, judged_targets=0, requested_phase1=0, requested_phase2=0,
               largest_set=0, first_failure=None)
    for mask in masks:
        A = self_edge_mask(N, T)
        for i, edge in enumerate(optional):
            if mask >> i & 1:
                A[edge] = True
        oracle = DSeparation(A)

        def test(x, y, S):
            return float(oracle.separated(x + shift, y + shift, [s + shift for s in S]))

        model = ITPD(test) if switches is None else ITPDAblation(test, switches=switches)
        model.fit(V)
        # Judged targets: after the first observed step, and with no parent in the latent step.
        judged = (np.arange(size - shift) >= N) & ~A[:shift, shift:].any(0)
        wrong = int((model.get_adjacency_matrix().astype(bool) != A[shift:, shift:])[:, judged].any(0).sum())
        failure = [bin(mask).count("1"), mask]
        if wrong and (out["first_failure"] is None or failure < out["first_failure"]):
            out["first_failure"] = failure
        out["graphs"] += 1
        out["exact"] += not wrong
        out["wrong_targets"] += wrong
        out["judged_targets"] += int(judged.sum())
        out["requested_phase1"] += model.ci_queries_["I"]
        out["requested_phase2"] += model.ci_queries_["II"]
        out["largest_set"] = max(out["largest_set"], model.max_conditioning_size_)
    return out
