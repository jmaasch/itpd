"""Check the baselines in `baselines/`."""
import numpy as np
import pytest

from itpd import CICounter, DSeparation, FisherZ
from itpd.baselines import Result, full_conditioning, iamb, ipadl, lasso
from itpd.graphs import self_edge_mask, unroll

METHODS = [iamb, ipadl, full_conditioning]


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
    result = iamb(CICounter(test), 3, 2, alpha=0.01)
    assert result.untested_targets == []
    return set(np.flatnonzero(result.adjacency[:, target]).tolist())


def test_iamb_breaks_a_tie_of_p_values_by_neglogp_and_then_by_the_largest_index():
    assert _parents(_WithNeglogp()) == {0, 1}            # Candidate 1 has the larger -log p.
    assert _parents(_TwoEqualCandidates()) == {0, 2}     # Without `neglogp`: the largest index.
    assert _parents(_WithNeglogp(), target=4) == {1}


class _Recorded(DSeparation):
    """`DSeparation` that lists the tests it evaluates."""

    def __init__(self, A):
        super().__init__(A)
        self.evaluated = []

    def __call__(self, x, y, S=()):
        self.evaluated.append((x, y, tuple(S)))
        return super().__call__(x, y, S)


def test_iamb_adds_the_largest_index_first_under_the_oracle():
    # N = 3, T = 2: target 3 has the parents 0 (self), 1 and 2.
    A = np.zeros((6, 6), dtype=bool)
    A[[0, 1, 2, 1, 2], [3, 4, 5, 3, 3]] = True
    test = _Recorded(A)
    result = iamb(CICounter(test), 3, 2)
    assert np.array_equal(result.adjacency, A)
    # Both candidates have p = 0 given the self parent, and the oracle has no
    # `neglogp`. Candidate 2 enters the blanket first, so the third test of the
    # target is candidate 1 given {0, 2}.
    assert test.evaluated[:3] == [(1, 3, (0,)), (2, 3, (0,)), (1, 3, (0, 2))]


def _stationary_graph():
    """Build a graph with N = 3, T = 4, and lags 1 and 2.

    Includes every lag-1 self edge, 0 -> 1 at lag 1, and 2 -> 0 at lag 2."""
    B = np.zeros((3, 3, 3), dtype=bool)
    B[1] = np.eye(3, dtype=bool)
    B[1, 0, 1] = B[2, 2, 0] = True
    return unroll(B, 4)


@pytest.mark.parametrize("method", METHODS)
def test_exact_recovery_of_one_graph_under_the_oracle(method):
    A = _stationary_graph()
    for corrected in (False, True):
        result = method(CICounter(DSeparation(A)), 3, 4, corrected=corrected)
        assert isinstance(result, Result) and result.untested_targets == []
        assert result.adjacency.dtype == bool and np.array_equal(result.adjacency, A)


def test_ipadl_does_not_request_a_marginal_test_twice_for_one_target():
    # N = 2, T = 2: the self edges 0 -> 2 and 1 -> 3, and the edge 1 -> 2.
    A = np.zeros((4, 4), dtype=bool)
    A[[0, 1, 1], [2, 3, 2]] = True
    counter = CICounter(DSeparation(A))
    result = ipadl(counter, 2, 2)
    assert np.array_equal(result.adjacency, A)
    # Target 2 (self parent 0, candidate 1): the marginal tests (0, 1) and
    # (2, 1), then (2, 1 | 0), (0, 1 | 2) and again (1, 2 | 0). The second
    # and third use of a marginal test are not requested.
    # Target 3 (self parent 1, candidate 0): the marginal tests (1, 0) and (3, 0).
    assert counter.requested == 7
    assert counter.unique == 5


def test_full_conditioning_stops_a_target_at_its_first_test_that_cannot_be_evaluated():
    # N = 2, T = 3, M = 5. Fisher-z needs M - |S| - 3 > 0. A target of step 1
    # has |S| = 1, and a target of step 2 has |S| = 3.
    V = np.random.default_rng(0).normal(size=(5, 3, 2))
    counter = CICounter(FisherZ(V))
    result = full_conditioning(counter, 2, 3)
    assert result.untested_targets == [4, 5]
    assert not result.adjacency[:, [4, 5]].any()           # Not even the self edge.
    assert result.adjacency[0, 2] and result.adjacency[1, 3]
    # One test for each target of step 1, and one for each untested target
    # (it has three candidates).
    assert counter.requested == 4
    assert counter.not_evaluable == 2


@pytest.mark.parametrize("method", METHODS)
def test_an_untested_target_has_no_edge(method):
    counter = CICounter(lambda x, y, S: float("nan") if y == 5 else 1.0)
    result = method(counter, 2, 3)
    assert result.untested_targets == [5]
    assert np.array_equal(np.argwhere(result.adjacency), [[0, 2], [1, 3], [2, 4]])


@pytest.mark.parametrize("method", METHODS)
def test_corrected_level_compares_the_scaled_p_value_with_alpha(method):
    # N = 28, T = 2: a target has 27 candidates. With this p-value,
    # p * 27 <= alpha holds and p <= alpha / 27 does not.
    alpha, p = 0.01, 0.0003703703703703704
    assert p * 27 <= alpha and p > alpha / 27

    def edge(p_value, corrected):
        counter = CICounter(lambda x, y, S: p_value if (x, y) == (0, 29) else 1.0)
        result = method(counter, 28, 2, alpha, corrected=corrected)
        assert result.adjacency[1, 29] and result.adjacency.sum() == 28 + result.adjacency[0, 29]
        return bool(result.adjacency[0, 29])

    assert edge(p, corrected=True)
    assert edge(0.0005, corrected=False)
    assert not edge(0.0005, corrected=True)              # 0.0005 * 27 > alpha.


def test_lasso_finds_a_strong_parent_and_always_returns_the_self_edge():
    rng = np.random.default_rng(1)
    V = rng.normal(size=(200, 3, 2))
    V[:, 1, 1] += 2.0 * V[:, 0, 0]                       # 0 -> 3
    V[:, 2, 0] = 1.0                                     # Target 4 is constant.
    result = lasso(V)
    assert result.untested_targets == []
    assert result.adjacency[self_edge_mask(2, 3)].all()
    assert result.adjacency[0, 3]
    assert np.flatnonzero(result.adjacency[:, 4]).tolist() == [2]
    assert not np.tril(result.adjacency).any()
