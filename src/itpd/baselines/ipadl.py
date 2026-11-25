"""IPaDL: local causal partitioning (PaDL), run once for every target."""
from ._targets import Result, run_per_target


def _parents(pvalue, x, y, candidates, alpha):
    marginal_x, marginal_y = {}, {}

    # The marginal tests of one target are kept here, so a repeated use is
    # not a requested test.
    def p_x(z):
        if z not in marginal_x:
            marginal_x[z] = pvalue(x, z, [])
        return marginal_x[z]

    def p_y(z):
        if z not in marginal_y:
            marginal_y[z] = pvalue(y, z, [])
        return marginal_y[z]

    z4, unlabeled = [], []
    for z in candidates:
        # Z8: z is independent of x and of y.
        if p_x(z) > alpha and p_y(z) > alpha:
            continue
        # Z5 and Z7: z and y are dependent, and independent given x.
        if p_y(z) <= alpha and pvalue(y, z, [x]) > alpha:
            continue
        # Z4: z and x are independent, and dependent given y.
        if p_x(z) > alpha and pvalue(x, z, [y]) <= alpha:
            z4.append(z)
        else:
            unlabeled.append(z)
    # An unlabeled candidate is a parent when it depends on y given x, Z4
    # and the other unlabeled candidates.
    conditioning = unlabeled + z4 + [x]
    parents = [z for z in unlabeled
               if pvalue(y, z, [s for s in conditioning if s != z]) <= alpha]
    # A member of Z4 is a parent when it depends on y given the other
    # members, the parents found above and x.
    z4_parents = [z for z in z4
                  if pvalue(z, y, [s for s in z4 if s != z] + parents + [x]) <= alpha]
    return parents + z4_parents


def ipadl(counter, N: int, T: int, alpha: float = 0.01, *,
          corrected: bool = False) -> Result:
    """Run IPaDL on N series and T steps.

    PaDL partitions the candidates of a target y by their relation to the
    known edge x -> y and tests the remaining candidates for adjacency. No
    result of one target is used for another target. counter is a
    CICounter. See run_per_target for alpha and corrected.
    """
    return run_per_target(_parents, counter, N, T, alpha, corrected)
