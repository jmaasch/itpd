"""The loop over targets that IAMB, IPaDL and FullCI share."""
from dataclasses import dataclass

import numpy as np

from ..ci_tests import NotEvaluable


@dataclass
class Result:
    """The estimate of a baseline.

    Attributes
    ----------
    adjacency : np.ndarray
        Boolean matrix (T * N, T * N) with [u, v] = True for u -> v.
    untested_targets : list of int
        Targets with a test that could not be evaluated. An untested target
        has no edge in adjacency, not even its lag-1 self edge.
    """
    adjacency: np.ndarray
    untested_targets: list


def run_per_target(find_parents, counter, N: int, T: int, alpha: float = 0.01,
                   corrected: bool = False) -> Result:
    """Find the parents of every target with one method.

    The targets are the nodes y = t * N + n for t = 1, ..., T - 1 and
    n = 0, ..., N - 1, in this order. The lag-1 self parent x = y - N is
    known. The candidates are all other nodes of the earlier steps, in
    increasing order.

    Parameters
    ----------
    find_parents : callable
        find_parents(pvalue, x, y, candidates, alpha) returns the parents of
        y among the candidates. A pair is dependent when pvalue(a, b, S) <=
        alpha.
    counter : CICounter
        The counted CI test.
    N, T : int
        Number of series and number of steps.
    alpha : float
        Significance level.
    corrected : bool
        If True, a target at step t is tested at the level alpha divided by
        its number of candidates, N * t - 1. The p-value is multiplied by
        this number and capped at 1, and the product is compared with alpha.
    """
    adjacency = np.zeros((T * N, T * N), dtype=bool)
    untested = []
    for t in range(1, T):
        scale = N * t - 1 if corrected else 1

        def pvalue(a, b, S):
            return min(1.0, counter(a, b, S) * scale)

        for y in range(t * N, (t + 1) * N):
            x = y - N
            candidates = [z for z in range(t * N) if z != x]
            try:
                parents = find_parents(pvalue, x, y, candidates, alpha)
            except NotEvaluable:
                untested.append(y)
                continue
            adjacency[[x, *parents], y] = True
    return Result(adjacency, untested)
