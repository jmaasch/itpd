"""IAMB (`iamb`): incremental association Markov blanket with known time order."""
from functools import partial

from ._targets import Result, run_per_target


def _parents(pvalue, x, y, candidates, alpha, neglogp=None):
    """`find_parents` function of IAMB for `run_per_target`.

    Return the blanket of y among `candidates` with a grow phase and one shrink
    pass, with x in every conditioning set. `neglogp(z, y, S)`, when given,
    ranks candidates with equal p-values.
    """
    blanket = []
    # Grow: test every candidate outside the blanket given the blanket, and
    # add the dependent candidate with the smallest p-value. Stop when no
    # candidate is dependent.
    while True:
        dependent = []
        for z in candidates:
            if z in blanket:
                continue
            p = pvalue(z, y, [x] + blanket)
            if p <= alpha:
                dependent.append((p, z))
        if not dependent:
            break
        smallest = min(p for p, _ in dependent)
        tied = [z for p, z in dependent if p == smallest]
        if len(tied) > 1 and neglogp is not None:
            # Equal p-values (for example p = 0): rank by -log p.
            S = sorted([x] + blanket)
            strength = [neglogp(z, y, S) for z in tied]
            tied = [z for z, s in zip(tied, strength) if s == max(strength)]
        blanket.append(max(tied))
    # Shrink: one pass in the order added. Remove a member that is
    # independent of the target given the other members.
    for z in list(blanket):
        others = [s for s in blanket if s != z]
        if pvalue(z, y, [x] + others) > alpha:
            blanket.remove(z)
    return blanket


def iamb(counter, N: int, T: int, alpha: float = 0.01, *,
         corrected: bool = False) -> Result:
    """Run IAMB on N series and T steps.

    The lag-1 self parent is in every conditioning set. Among the earlier
    nodes, the Markov blanket of a target is its parent set, so the blanket is
    the estimate. Candidates with equal p-values are ranked by
    `counter.test.neglogp` when the test has this method, and then the largest
    node wins. `counter` is a `CICounter`. See `run_per_target` for `alpha` and
    `corrected`.
    """
    neglogp = getattr(counter.test, "neglogp", None)
    return run_per_target(partial(_parents, neglogp=neglogp), counter, N, T,
                          alpha, corrected)
