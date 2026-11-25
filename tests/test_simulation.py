"""Tests of the simulated instances.

The frozen graph hashes depend on the random generator of NumPy only: these
graphs are drawn before any floating-point check, so the hashes are the same
on every platform. No test here compares a hash of data.
"""
import hashlib

import networkx as nx
import numpy as np
import pytest

from itpd.graphs import check_temporal, graph_hash, self_edge_mask, unroll
from itpd.simulation import Instance, linear, nonlinear, redrawn_edges, stepwise, violations
from itpd.simulation.graphs import edge_probability, longest_cross_lag, sample_nonstationary_graph

# (N, T, lag, in-degree, graph number, stationary, SHA-1 of the graph)
LINEAR_GRAPHS = [
    (20, 16, 1, 2, 0, False, "fe1e31e0af57ec124755a5c6eeaff435ce3135a9"),
    (20, 16, 1, 1, 0, False, "6fe773a4e16cfd7448b4cdec22038b159d4dc595"),
    (20, 16, 1, 1, 0, True, "541a6fdd353ee43da1c72a71068eb2ff88ffd3a7"),
    (10, 8, 1, 2, 0, True, "fc8e3813eb4c31b78e6fa4f36b4f0a47af45f6a0"),
    (10, 8, 1, 2, 0, False, "1763ce3bc49187c7699461f3496f4060473c7c2d"),
]


@pytest.mark.parametrize("N, T, lag, d, g, stationary, sha1", LINEAR_GRAPHS)
def test_graph_hashes(N, T, lag, d, g, stationary, sha1):
    graph_only = linear.make_instance(N, T, lag, d, g, stationary=stationary, weights=False)
    assert graph_only.graph_sha1 == sha1 == graph_hash(graph_only.A)
    assert graph_only.sample is None
    # The weights are drawn after the graph, so they do not change it.
    assert linear.make_instance(N, T, lag, d, g, stationary=stationary).graph_sha1 == sha1


def test_graph_hashes_of_the_other_generators():
    assert redrawn_edges.make_instance(20, 16, 2, 0.05, 0).graph_sha1 == "9fcb3d910e391addc1433bf77cac1a66d46876ea"
    assert redrawn_edges.make_instance(20, 16, 2, 1.0, 1).graph_sha1 == "e262e75021f2f069904c3d47ef990007e72c4d64"
    without_self = violations.make_instance(2, 0, no_self_edge_share=0.1)
    assert without_self.graph_sha1 == "865a3c1636ed41419c293871ff5743103afff6a7"
    assert without_self.info["series_without_self_edge"] == [10, 12]
    same_step = violations.make_instance(2, 0, same_step_share=0.1)
    assert same_step.graph_sha1 == "d2fd755be1d0183449a25586e5fdedbff6b5f90e"
    assert same_step.info["n_moved_edges"] == 62


@pytest.mark.parametrize("stationary", [True, False])
@pytest.mark.parametrize("lag", [1, 2, 4])
def test_linear_graphs_are_forward_in_time_with_all_self_edges(stationary, lag):
    N, T = 6, 7
    for g in range(3):
        instance = linear.make_instance(N, T, lag, 2, g, stationary=stationary)
        assert isinstance(instance, Instance) and instance.A.dtype == bool
        check_temporal(instance.A, N, T)
        sources, targets = np.nonzero(instance.A)
        assert (targets // N - sources // N).max() <= lag
        W = instance.info["weights"]
        assert W.shape == (T * N, T * N) and ((W != 0) == instance.A).all()
        if stationary:
            assert np.array_equal(unroll(instance.info["B"], T), instance.A)


def test_other_generators_give_valid_graphs():
    N, T = 6, 7
    for instance in (stepwise.make_instance(N, T, 2, 0), redrawn_edges.make_instance(N, T, 2, 0.3, 0),
                     nonlinear.make_instance(4, 4, 1, 1, 0, stationary=False)):
        check_temporal(instance.A, instance.N, instance.T)
        assert instance.lag == 1 and instance.graph_sha1 == graph_hash(instance.A)
    W3 = stepwise.make_instance(N, T, 2, 0).info["weights"]
    assert W3.shape == (T - 1, N, N)


def test_edge_probability():
    assert edge_probability(20, 1, 2) == 2 / 19
    assert edge_probability(10, 4, 2) == 2 / 39
    assert edge_probability(3, 1, 5) == 1.0
    assert edge_probability(1, 1, 2) == 0.0


def test_nonstationary_graph_draws_new_edges_at_every_step():
    N, T = 8, 6
    A = sample_nonstationary_graph(N, T, 2, 1, np.random.default_rng(0))
    blocks = {A[t * N:(t + 1) * N, (t + 1) * N:(t + 2) * N].tobytes() for t in range(T - 1)}
    assert len(blocks) == T - 1


def test_smaller_m_is_a_prefix():
    instances = [linear.make_instance(5, 6, 2, 2, 0, stationary=True),
                 linear.make_instance(5, 6, 1, 2, 0, stationary=False),
                 stepwise.make_instance(5, 6, 2, 0), redrawn_edges.make_instance(5, 6, 2, 0.5, 0),
                 violations.make_instance(2, 0, same_step_share=0.25, N=5, T=6),
                 nonlinear.make_instance(4, 4, 1, 1, 0, stationary=True)]
    for instance in instances:
        full = instance.sample(2000)
        assert full.shape == (2000, instance.T, instance.N) and full.dtype == np.float64
        assert np.array_equal(instance.sample(50), full[:50])
        assert np.array_equal(instance.sample(2000), full)   # the same draw on every call


def test_more_rows_than_m_max_are_another_draw():
    instance = linear.make_instance(5, 6, 1, 2, 0, stationary=False)
    larger = instance.sample(2500)
    assert larger.shape == (2500, 6, 5)
    assert not np.array_equal(larger[:2000], instance.sample(2000))


def test_data_hash_is_the_sha1_of_the_float64_bytes():
    X = np.arange(6.0).reshape(1, 2, 3)
    expected = hashlib.sha1(X.tobytes()).hexdigest()
    for same in (X, X.astype(np.float32), np.asfortranarray(X)):
        assert linear.data_hash(same) == expected


def test_population_covariance_matches_the_data():
    instance = linear.make_instance(4, 5, 1, 2, 0, stationary=False)
    A, W = instance.A, instance.info["weights"]
    _, Sigma = linear.propagate_covariance(A, W)
    X = instance.sample(2000).reshape(2000, -1)
    assert np.abs(np.cov(X, rowvar=False) - Sigma).max() < 0.35
    assert linear.partial_margin(A, Sigma) == pytest.approx(instance.info["min_partial_correlation"])
    assert linear.marginal_margin(A, Sigma) == pytest.approx(instance.info["min_marginal_correlation"])
    assert instance.info["guard_ok"] == (min(instance.info["min_partial_correlation"],
                                             instance.info["min_marginal_correlation"]) >= linear.MIN_STRENGTH)


@pytest.mark.parametrize("d, g", [(2, 0), (2, 1), (1, 0)])
def test_violations_with_shares_zero_equal_the_linear_instance(d, g):
    violated = violations.make_instance(d, g)
    standard = linear.make_instance(20, 16, 1, d, g, stationary=False)
    assert violated.graph_sha1 == standard.graph_sha1
    assert np.array_equal(violated.info["weights"], standard.info["weights"])
    assert np.array_equal(violated.sample(200), standard.sample(200))
    assert np.array_equal(violated.info["A_lag1"], standard.A)


def test_series_without_self_edge():
    N, T = 20, 16
    instance = violations.make_instance(2, 0, no_self_edge_share=0.25)
    series = instance.info["series_without_self_edge"]
    assert len(series) == 5
    assert set(series) > set(violations.make_instance(2, 0, no_self_edge_share=0.1).info["series_without_self_edge"])
    removed = instance.info["A_main"] & ~instance.A
    sources, targets = np.nonzero(removed)
    assert removed.sum() == 5 * (T - 1) and not (instance.A & ~instance.info["A_main"]).any()
    assert (removed <= self_edge_mask(N, T)).all() and set(sources % N) == set(series) == set(targets % N)


def test_edges_inside_a_time_step():
    N, T = 20, 16
    instance = violations.make_instance(2, 0, same_step_share=0.5)
    A, A_main, A_lag1 = instance.A, instance.info["A_main"], instance.info["A_lag1"]
    step = np.arange(N * T) // N
    same_step = A & (step[:, None] == step[None, :])
    assert same_step.sum() == instance.info["n_moved_edges"] > 0
    assert A.sum() == A_main.sum() and np.array_equal(A_lag1, A & ~same_step)
    assert (A_lag1 <= A_main).all() and A[self_edge_mask(N, T)].all()
    # Every edge inside a step replaces the lag-1 edge between the same two series.
    sources, targets = np.nonzero(same_step)
    assert A_main[sources - N, targets].all() and not A[sources - N, targets].any()
    assert nx.is_directed_acyclic_graph(nx.from_numpy_array(A, create_using=nx.DiGraph))


def test_both_violations_at_once_are_refused():
    with pytest.raises(ValueError):
        violations.make_instance(2, 0, no_self_edge_share=0.1, same_step_share=0.1)


def test_redrawn_edges_with_share_zero_is_stationary_and_shares_are_nested():
    N, T = 8, 7
    still = redrawn_edges.make_instance(N, T, 2, 0.0, 0)
    first = still.A[:N, N:2 * N]
    for t in range(1, T - 1):
        assert np.array_equal(still.A[t * N:(t + 1) * N, (t + 1) * N:(t + 2) * N], first)
    assert still.info["n_redrawn_slots"] == 0
    low, high = redrawn_edges.make_instance(N, T, 2, 0.2, 0), redrawn_edges.make_instance(N, T, 2, 0.6, 0)
    assert np.array_equal(low.A[:N, N:2 * N], first) and np.array_equal(high.A[:N, N:2 * N], first)
    assert 0 < low.info["n_redrawn_slots"] < high.info["n_redrawn_slots"]
    assert low.info["guard_tries"] == high.info["guard_tries"] == still.info["guard_tries"]


@pytest.mark.parametrize("stationary", [True, False])
def test_nonlinear_instance(stationary):
    N, T = 4, 4
    instance = nonlinear.make_instance(N, T, 1, 1, 0, stationary=stationary)
    assert instance.graph_sha1 == linear.make_instance(N, T, 1, 1, 0, stationary=stationary, weights=False).graph_sha1
    params = instance.info["parameters"]
    assert params.shape == (6, T * N, T * N)
    assert ((params[0] != 0) == instance.A).all()
    w = np.abs(params[0][instance.A])
    assert (w >= 0.8).all() and (w <= 1.6).all()
    if stationary:   # one function per slot of the window graph, used at every step
        assert np.array_equal(params[:, :N, N:2 * N], params[:, N:2 * N, 2 * N:3 * N])
    assert instance.info["guard_ok"] == (min(instance.info["min_partial_correlation"],
                                             instance.info["min_marginal_correlation"]) >= nonlinear.MIN_STRENGTH)
    assert np.abs(instance.sample(100)).max() < 50   # the functions are bounded


def test_longest_cross_lag_selects_the_graphs_of_the_lag_8_setting():
    # Stationary graphs with N 10, T 24, lag 8, in-degree 2: a graph number is
    # kept when its longest edge between two series has lag 8.
    N, T, lag = 10, 24, 8
    kept, g = [], 0
    while len(kept) < 20:
        instance = linear.make_instance(N, T, lag, 2, g, stationary=True, weights=False)
        if longest_cross_lag(instance.A, N) == lag:
            kept.append(g)
        g += 1
    assert kept == [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23]
    assert longest_cross_lag(self_edge_mask(N, T), N) == 0
