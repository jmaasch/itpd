"""Check Algorithm 1, its test levels, its CI tests, and finite-data results.

The finite-data tests at the end draw their data again and skip when the data
hash differs. Use -m "not slow" to omit them."""
import math
import warnings

import networkx as nx
import numpy as np
import pytest

from itpd import CICounter, DSeparation, FisherZ, ITPD, NotEvaluable, metrics


# Exact recovery with the d-separation oracle.

def _optional_edges(N, T):
    """Return forward edges other than the mandatory lag-1 self edges."""
    time = np.repeat(np.arange(T), N)
    P = N * T
    return [(u, v) for u in range(P) for v in range(P) if time[u] < time[v] and v != u + N]


def _graph(N, T, optional, mask):
    P = N * T
    A = np.zeros((P, P), np.uint8)
    A[np.arange(P - N), np.arange(N, P)] = 1
    for i, (u, v) in enumerate(optional):
        if mask >> i & 1:
            A[u, v] = 1
    return A


def _not_exact(N, T, masks):
    optional = _optional_edges(N, T)
    bad = []
    for mask in masks:
        A = _graph(N, T, optional, mask)
        model = ITPD(oracle_adjacency=A).fit(n_variables=N)
        assert model.untested_targets_ == []
        if not np.array_equal(model.get_adjacency_matrix(), A):
            bad.append(mask)
    return bad


def test_oracle_exact_on_all_graphs_n2_t3():
    assert len(_optional_edges(2, 3)) == 8
    assert _not_exact(2, 3, range(2 ** 8)) == []


@pytest.mark.parametrize("N, T, n_graphs", [(2, 4, 300), (3, 3, 300)])
def test_oracle_exact_on_random_graphs(N, T, n_graphs):
    K = len(_optional_edges(N, T))
    masks = np.random.default_rng(0).choice(2 ** K, n_graphs, replace=False).tolist()
    assert _not_exact(N, T, masks) == []


def _example_graph():
    """Three series, five steps; forward edges with probability 0.12."""
    N, T = 3, 5
    rng = np.random.default_rng(7)
    times = np.repeat(np.arange(T), N)
    A = ((rng.random((N * T, N * T)) < 0.12) & (times[:, None] < times[None, :])).astype(np.uint8)
    A[np.arange(N * (T - 1)), np.arange(N, N * T)] = 1
    return A, N, T


def test_oracle_counts_on_one_graph():
    A, N, T = _example_graph()
    counter = CICounter(DSeparation(A))
    model = ITPD(counter).fit(np.zeros((2, T, N)))
    assert np.array_equal(model.get_adjacency_matrix(), A) and A.sum() == 18
    assert model.ci_queries_ == {"I": 115, "II": 8} and model.max_conditioning_size_ == 5
    assert counter.summary(N, T) == {"requested": 123, "unique": 117, "largest_set": 5, "mean_set": 103 / 117,
                                     "unique_per_candidate": 1.5, "not_evaluable": 0}
    # The oracle given as a matrix issues the same tests.
    direct = ITPD(oracle_adjacency=A).fit(n_variables=N)
    assert direct.ci_queries_ == model.ci_queries_ and direct.adjacency_ == model.adjacency_
    truth = nx.from_numpy_array(A, create_using=nx.DiGraph)
    assert all(direct.ancestors(v) == nx.ancestors(truth, v) for v in range(N * T))


def test_oracle_matrix_is_checked():
    A, N, T = _example_graph()
    for u, v, value in [(0, 3, 0), (4, 1, 1), (3, 4, 1)]:    # A missing self edge, a backward edge, an edge inside a step.
        bad = A.copy()
        bad[u, v] = value
        with pytest.raises(ValueError):
            ITPD(oracle_adjacency=bad).fit(n_variables=N)
    with pytest.raises(ValueError):
        ITPD(oracle_adjacency=A).fit(n_variables=4)
    with pytest.raises(ValueError):
        ITPD().fit(n_variables=N)                            # No data and no oracle.


def test_plot_graph(tmp_path):
    pytest.importorskip("matplotlib")
    A, N, T = _example_graph()
    model = ITPD(oracle_adjacency=A).fit(n_variables=N)
    assert model.plot_graph(tmp_path / "graph").exists()
    assert (tmp_path / "graph.png").stat().st_size > 0


# Phase I and Phase II test levels.

def _fitted(N, T, **kwargs):
    return ITPD(lambda x, y, S: 1.0, **kwargs).fit(np.zeros((2, T, N)))


@pytest.mark.parametrize("N, t", [(2, 1), (2, 2), (3, 1), (20, 1), (20, 12), (20, 13), (20, 15), (300, 20)])
def test_levels_follow_the_formulas(N, t):
    """Check m = N * t - 1, with target time t + 1 in the paper's indexing."""
    model = _fitted(N, 2)
    m = N * t - 1
    alpha1, alpha2 = model._levels(t - 1)          # Convert the paper's index to zero-based time.
    assert alpha1 == max(0.01, min(0.2, 50 / m))
    assert alpha2 == 0.01 / m


def test_levels_known_values():
    model = _fitted(20, 2)
    assert model._levels(0) == (0.2, 0.01 / 19)                    # 50 / 19 > gamma
    assert model._levels(11)[0] == 0.2                             # 50 / 239 > gamma
    assert math.isclose(model._levels(12)[0], 50 / 259)            # gamma > 50 / 259 > alpha
    assert _fitted(300, 2)._levels(19) == (0.01, 0.01 / 5999)      # 50 / 5999 < alpha: the floor is alpha
    assert _fitted(1, 2)._levels(0) == (0.2, 0.01)                 # No candidates; use m = 1.


def test_levels_use_the_arguments():
    model = _fitted(20, 2, alpha=0.05, gamma=0.1, kappa=10)
    assert model._levels(0) == (0.1, 0.05 / 19)
    assert math.isclose(model._levels(9)[0], 10 / 199)
    assert math.isclose(model._levels(9)[1], 0.05 / 199)
    assert _fitted(300, 2, alpha=0.05)._levels(19) == (0.05, 0.05 / 5999)   # The floor is alpha.
    assert ITPD().alpha == 0.01 and ITPD().gamma == 0.2 and ITPD().kappa == 50


@pytest.mark.parametrize("kwargs", [{"alpha": 0.0}, {"alpha": 1.0}, {"gamma": 0.0}, {"gamma": 1.0}, {"kappa": 0}])
def test_invalid_levels_are_rejected(kwargs):
    with pytest.raises(ValueError):
        ITPD(**kwargs)


def _constant_p(p, **kwargs):
    """Fit N = 2, T = 2 with a constant p-value and collect target 2 calls.

    Each target has one candidate; target 2 has X = 0 and Z = 1.
    With default levels, alpha1 = gamma and alpha2 = alpha."""
    calls = []

    def ci(x, y, S):
        calls.append((x, y, tuple(S)))
        return p

    model = ITPD(ci, **kwargs).fit(np.zeros((2, 2, 2)))
    return model, [c for c in calls if c[1] == 2]


def test_levels_are_applied_on_the_lines():
    # alpha2 < p < alpha1: Lines 8-10 find dependence; Line 14 finds independence.
    model, calls = _constant_p(0.1)
    assert calls == [(1, 2, ()), (1, 2, (0,)), (1, 2, ()), (1, 2, (0,))]
    assert model.adjacency_[2] == {0}
    assert model.ci_queries_ == {"I": 6, "II": 2}
    # p > alpha1: Line 8 answers "independent".
    model, calls = _constant_p(0.3)
    assert calls == [(1, 2, ())] and model.adjacency_[2] == {0}
    # p < alpha2: Line 14 answers "dependent".
    model, calls = _constant_p(0.005)
    assert len(calls) == 4 and model.adjacency_[2] == {0, 1}
    # Changing gamma or alpha changes the test decisions.
    model, calls = _constant_p(0.1, gamma=0.05)
    assert calls == [(1, 2, ())]
    model, calls = _constant_p(0.1, alpha=0.15)
    assert len(calls) == 4 and model.adjacency_[2] == {0, 1}
    # p equal to the level counts as dependent.
    model, calls = _constant_p(0.2)
    assert len(calls) == 4 and model.adjacency_[2] == {0}


# NaN p-values and invalid p-values.

def test_nan_p_value_lists_the_target_and_warns():
    def ci(x, y, S):
        return float("nan") if y == 4 else 0.0

    with pytest.warns(RuntimeWarning, match="untested_targets_") as record:
        model = ITPD(ci).fit(np.zeros((2, 4, 2)))
    assert len(record) == 1
    assert model.untested_targets_ == [4]
    assert model.adjacency_[4] == {2}                    # Only the lag-1 self edge remains.
    assert model.adjacency_[5] == {0, 1, 2, 3}           # Discovery continues for other targets.
    assert model.adjacency_[6] == {0, 1, 2, 3, 4, 5}
    assert model.get_adjacency_matrix().shape == (8, 8)


def test_invalid_p_value_still_raises():
    with pytest.raises(ValueError):
        ITPD(lambda x, y, S: 1.5).fit(np.zeros((2, 2, 2)))


# The CI test of the fit.

def _linear_data(M, N=3, T=4, seed=0):
    """Each series depends on its own last value and on the last value of the series before it."""
    rng = np.random.default_rng(seed)
    V = rng.standard_normal((M, T, N))
    for t in range(1, T):
        V[:, t] += 0.6 * V[:, t - 1] + 0.5 * np.roll(V[:, t - 1], 1, axis=1)
    return V


@pytest.mark.parametrize("M", [200, 6])
def test_fisherz_by_name_equals_fisherz_through_the_counter(M):
    """M = 6: tests with |S| >= 3 cannot be evaluated; FisherZ returns NaN and the counter raises."""
    V = _linear_data(M)
    counter = CICounter(FisherZ(V))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        models = [ITPD("fisherz").fit(V), ITPD().fit(V), ITPD(FisherZ(V.reshape(M, -1))).fit(V), ITPD(counter).fit(V)]
    for model in models[1:]:
        assert model.adjacency_ == models[0].adjacency_
        assert model.ci_queries_ == models[0].ci_queries_
        assert model.max_conditioning_size_ == models[0].max_conditioning_size_
        assert model.untested_targets_ == models[0].untested_targets_
    assert counter.requested == sum(models[0].ci_queries_.values())
    assert counter.summary(3, 4)["largest_set"] == models[0].max_conditioning_size_
    assert bool(models[0].untested_targets_) == (M == 6) == bool(counter.not_evaluable)
    if M == 200:
        truth = {3 * t + n: {3 * (t - 1) + n, 3 * (t - 1) + (n - 1) % 3} for t in range(1, 4) for n in range(3)}
        assert all(models[0].adjacency_[y] == pa for y, pa in truth.items())


def test_gcm_by_name():
    V = _linear_data(200, N=2, T=2)
    model = ITPD("gcm").fit(V)
    assert model.adjacency_[2] == {0, 1} and model.adjacency_[3] == {0, 1}
    assert model.ci_queries_ == {"I": 6, "II": 2}


def test_invalid_ci_test_is_rejected():
    with pytest.raises(ValueError):
        ITPD("kci").fit(np.zeros((2, 2, 2)))


def test_a_test_that_raises_not_evaluable_gives_an_untested_target():
    def ci(x, y, S):
        if y == 4:
            raise NotEvaluable
        return 0.0

    with pytest.warns(RuntimeWarning, match="untested_targets_"):
        model = ITPD(ci).fit(np.zeros((2, 4, 2)))
    assert model.untested_targets_ == [4] and model.adjacency_[4] == {2}
    assert model.adjacency_[5] == {0, 1, 2, 3}


# Reproduce the paper's finite-data results.

# Linear Gaussian data: N = 20, T = 16, lag = 1, Fisher-z, alpha = 0.01.
# Each instance uses expected in-degree d, graph index g, and generator seed 0.
# Use the first M of 2000 observations. Scores exclude lag-1 self edges.
FIXTURES = [
    # Columns: d, g, M, SHA-1 of 2000 observations, tp, fp, fn, requested tests,
    # unique tests, and largest conditioning set.
    (2.0, 0, 200, "f7df382ae9f36ffc0f960f94ba2ed9e41afa36db", 554, 7, 69, 63470, 62495, 25),
    (2.0, 1, 500, "5e2ef56f5c751c391e42191ef9d5b5649476f6ca", 591, 3, 7, 64769, 63818, 28),
    (1.0, 0, 2000, "37c8fcbb6c0126ecd13ee7f9849a012930b09bb1", 312, 1, 0, 65468, 64501, 29),
]


@pytest.mark.slow
@pytest.mark.parametrize("d, g, M, sha1, tp, fp, fn, requested, unique, largest", FIXTURES,
                         ids=[f"d{f[0]:g}-g{f[1]}-M{f[2]}" for f in FIXTURES])
def test_stored_runs(d, g, M, sha1, tp, fp, fn, requested, unique, largest):
    linear = pytest.importorskip("itpd.simulation.linear", reason="needs the data generator itpd.simulation")
    N, T = 20, 16
    instance = linear.make_instance(N, T, 1, d, g, stationary=False)
    if linear.data_hash(instance.sample(2000)) != sha1:
        pytest.skip("the data differ on this platform")
    V = instance.sample(M)
    counter = CICounter(FisherZ(V))

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)   # Every target must be testable.
        model = ITPD(counter, alpha=0.01).fit(V)

    assert model.untested_targets_ == []
    counts = metrics.edge_counts(instance.A, model.get_adjacency_matrix(), N, T)
    assert (counts["tp"], counts["fp"], counts["fn"]) == (tp, fp, fn)
    summary = counter.summary(N, T)
    assert summary["requested"] == requested == sum(model.ci_queries_.values())
    assert summary["unique"] == unique
    assert summary["largest_set"] == largest == model.max_conditioning_size_
