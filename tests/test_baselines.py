"""Check the baselines."""
import numpy as np
import pytest

from itpd import CICounter

baselines = pytest.importorskip("itpd.baselines", reason="needs the baselines itpd.baselines")


class _TwoEqualCandidates:
    """N = 3, T = 2. Candidates 1 and 2 of target 3 both have p = 0.0 given the self parent 0.

    Each of them is independent of the target given the other one. Every other pair is independent."""

    def __call__(self, x, y, S):
        if y == 3 and x in (1, 2):
            return 1.0 if 3 - x in S else 0.0
        return 1.0


class _WithNeglogp(_TwoEqualCandidates):
    def neglogp(self, x, y, S):
        return {1: 2.0, 2: 1.0}.get(x, 0.0)


def _parents(test, target=3):
    result = baselines.iamb(CICounter(test), 3, 2, alpha=0.01)
    assert result.untested_targets == []
    return set(np.flatnonzero(result.adjacency[:, target]).tolist())


def test_iamb_breaks_a_tie_of_p_values_by_neglogp_and_then_by_the_largest_index():
    assert _parents(_WithNeglogp()) == {0, 1}            # Candidate 1 has the larger -log p.
    assert _parents(_TwoEqualCandidates()) == {0, 2}     # Without neglogp: the largest index.
    assert _parents(_WithNeglogp(), target=4) == {1}
