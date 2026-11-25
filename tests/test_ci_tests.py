"""Check the CI tests in `ci_tests.py`: `FisherZ`, `GCM`, and `DSeparation`."""
import math

import networkx as nx
import numpy as np
import pytest
from scipy.stats import norm

from itpd import GCM, DSeparation, FisherZ, NotEvaluable


# Fisher-z.

def _three_columns():
    """Return a test with M = 103 and a correlation matrix set by hand."""
    test = FisherZ(np.random.default_rng(0).standard_normal((103, 3)))
    test.corr = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]])
    return test


def test_fisherz_closed_form():
    test = _three_columns()
    # Marginal: r = 0.5 and dof = 100.
    assert math.isclose(test(0, 1), 2 * norm.sf(10 * math.atanh(0.5)), rel_tol=1e-12)
    assert math.isclose(test(0, 1), 3.950252784999216e-08, rel_tol=1e-12)
    # Given column 2: r = (0.5 - 0.3 * 0.4) / sqrt((1 - 0.09) (1 - 0.16)) and dof = 99.
    r = 0.38 / math.sqrt(0.91 * 0.84)
    assert math.isclose(test(0, 1, [2]), 2 * norm.sf(math.sqrt(99) * math.atanh(r)), rel_tol=1e-12)
    assert math.isclose(test(0, 1, [2]), 3.610756564346025e-06, rel_tol=1e-12)
    assert math.isclose(test.neglogp(0, 1, [2]), -math.log(test(0, 1, [2])), rel_tol=1e-12)


def test_fisherz_matches_the_correlation_of_the_data():
    data = np.random.default_rng(1).standard_normal((60, 4))
    data[:, 1] += data[:, 0]
    r = np.corrcoef(data.T)[0, 1]
    assert math.isclose(FisherZ(data)(0, 1), 2 * norm.sf(math.sqrt(57) * abs(math.atanh(r))), rel_tol=1e-10)
    # Data (M, T, N) are read as (M, T * N).
    assert FisherZ(data.reshape(60, 2, 2))(0, 1, [3]) == FisherZ(data)(0, 1, [3])


def test_fisherz_ignores_the_pair_in_the_set():
    test = _three_columns()
    assert test(0, 1, [2, 0, 1]) == test(0, 1, [2])
    assert test(0, 1, [0]) == test(0, 1)


def test_fisherz_nan_when_dof_is_not_positive():
    test = FisherZ(np.random.default_rng(2).standard_normal((6, 5)))
    assert 0 <= test(0, 1, [2, 3]) <= 1                # dof = 6 - 2 - 3 = 1
    assert math.isnan(test(0, 1, [2, 3, 4]))           # dof = 0
    assert math.isnan(test.neglogp(0, 1, [2, 3, 4]))


def test_fisherz_edge_cases_of_the_partial_correlation():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(100)
    test = FisherZ(np.column_stack([x, 2 * x, rng.standard_normal(100)]))
    # |r| = 1 is clipped: the p-value is 0 and -log p stays finite.
    assert test(0, 1) == 0.0
    assert math.isfinite(test.neglogp(0, 1))
    # Column 0 has no variance left given column 1: p = 1.
    assert test(0, 2, [1]) == 1.0
    # A repeated member of S makes the system singular; the test still answers.
    assert 0 < test(0, 2, [1, 1]) <= 1


def test_fisherz_neglogp_ranks_pairs_with_p_equal_to_zero():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(2000)
    test = FisherZ(np.column_stack([x, x + 0.01 * rng.standard_normal(2000),
                                    x + 0.02 * rng.standard_normal(2000)]))
    assert test(0, 1) == 0.0 and test(0, 2) == 0.0
    assert math.isfinite(test.neglogp(0, 2)) and test.neglogp(0, 1) > test.neglogp(0, 2)


# GCM.

def _nonlinear_data(M=200):
    rng = np.random.default_rng(3)
    z = rng.standard_normal(M)
    x = np.tanh(z) + 0.3 * rng.standard_normal(M)
    y = np.sin(z) + 0.3 * rng.standard_normal(M)
    return np.column_stack([x, y, z])


def test_gcm_is_deterministic_and_finds_the_common_cause():
    data = _nonlinear_data()
    p_marginal, p_given = GCM(data)(0, 1), GCM(data)(0, 1, [2])
    assert GCM(data)(0, 1, [2]) == p_given
    assert p_marginal < 0.01 < p_given <= 1


def test_gcm_with_an_empty_set_is_a_correlation_test():
    data = _nonlinear_data()
    x, y = data[:, 0] - data[:, 0].mean(), data[:, 1] - data[:, 1].mean()
    R = x * y
    stat = math.sqrt(200) * R.mean() / math.sqrt((R ** 2).mean() - R.mean() ** 2)
    assert math.isclose(GCM(data)(0, 1), 2 * norm.sf(abs(stat)), rel_tol=1e-10)


def test_gcm_nan_with_fewer_than_eight_observations():
    data = _nonlinear_data()
    assert math.isnan(GCM(data[:7])(0, 1, [2]))
    assert not math.isnan(GCM(data[:8])(0, 1))


# d-separation oracle.

def test_dseparation_on_small_graphs():
    collider = np.zeros((3, 3), int)
    collider[0, 2] = collider[1, 2] = 1                     # 0 -> 2 <- 1
    oracle = DSeparation(collider)
    assert oracle(0, 1) == 1.0 and oracle(0, 1, [2]) == 0.0
    assert oracle.separated(0, 1) and not oracle.separated(0, 1, [2])
    chain = np.zeros((4, 4), int)
    chain[0, 1] = chain[1, 2] = chain[2, 3] = 1             # 0 -> 1 -> 2 -> 3
    oracle = DSeparation(chain)
    assert oracle(0, 3) == 0.0 and oracle(0, 3, [1]) == 1.0 and oracle(0, 2, [3]) == 0.0
    assert oracle(0, 3, [0, 3]) == oracle(0, 3)             # The pair is ignored in S.


def test_dseparation_agrees_with_networkx_on_random_dags():
    rng = np.random.default_rng(4)
    n_queries = 0
    for _ in range(20):
        n = int(rng.integers(5, 13))
        A = np.triu(rng.random((n, n)) < rng.uniform(0.1, 0.5), k=1)
        order = rng.permutation(n)
        A = A[np.ix_(order, order)]                         # A DAG whose order is not the index order.
        graph = nx.from_numpy_array(A, create_using=nx.DiGraph)
        oracle = DSeparation(A)
        for _ in range(200):
            x, y = (int(v) for v in rng.choice(n, 2, replace=False))
            others = [v for v in range(n) if v not in (x, y)]
            S = [int(s) for s in rng.choice(others, int(rng.integers(0, len(others) + 1)), replace=False)]
            assert oracle.separated(x, y, S) == nx.is_d_separator(graph, {x}, {y}, set(S))
            n_queries += 1
    assert n_queries == 4000


def test_dseparation_rejects_a_cycle():
    A = np.zeros((3, 3), int)
    A[0, 1] = A[1, 2] = A[2, 0] = 1
    with pytest.raises(ValueError):
        DSeparation(A)


def test_only_fisherz_has_neglogp():
    assert hasattr(FisherZ, "neglogp")
    assert not hasattr(GCM, "neglogp") and not hasattr(DSeparation, "neglogp")
    assert issubclass(NotEvaluable, RuntimeError)
