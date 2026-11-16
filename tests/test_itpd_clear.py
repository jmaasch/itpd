"""Tests of `itpd_clear/itpd.py`, the ITPD algorithm (Algorithm 1 of the paper).

The module is loaded by file path, so these tests do not need an installed package. The finite-data tests regenerate
their data with the repository's generator (`itpd_aistats`) and are skipped when it cannot be imported.

    python -m pytest tests/test_itpd_clear.py                 # all tests
    python -m pytest tests/test_itpd_clear.py -m "not slow"   # without the finite-data fixtures
"""
import importlib.util
import math
import sys
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("itpd_clear_itpd", ROOT / "itpd_clear" / "itpd.py")
itpd_clear = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(itpd_clear)
ITPD = itpd_clear.ITPD


# ---------------------------------------------------------------- exact recovery with the d-separation oracle

def _optional_edges(N, T):
    """Every forward edge except the lag-1 self edges, which are always present."""
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
        model = ITPD(backend="oracle", oracle_adjacency=A).fit(n_variables=N)
        assert model.untested_targets_ == []
        if not np.array_equal(model.get_adjacency_matrix(), A):   # equal matrices = every target exact
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


def test_run_example(tmp_path):
    model = itpd_clear.run_example(seed=7, output=str(tmp_path / "example.png"))
    assert (tmp_path / "example.png").exists()
    assert model.ci_queries_["I"] > 0 and model.ci_queries_["II"] > 0


# ---------------------------------------------------------------- the two levels

def _fitted(N, T, **kwargs):
    return ITPD(lambda x, y, S: 1.0, **kwargs).fit(np.zeros((2, T, N)))


@pytest.mark.parametrize("N, t", [(2, 1), (2, 2), (3, 1), (20, 1), (20, 12), (20, 13), (20, 15), (300, 20)])
def test_levels_follow_the_formulas(N, t):
    """`t` is the paper's index: the target is at time step t + 1 and has m = N * t - 1 candidates."""
    model = _fitted(N, 2)
    m = N * t - 1
    alpha1, alpha2 = model._levels(t - 1)          # the code counts time steps from 0
    assert alpha1 == max(0.01, min(0.2, 50 / m))
    assert alpha2 == 0.01 / m


def test_levels_known_values():
    model = _fitted(20, 2)
    assert model._levels(0) == (0.2, 0.01 / 19)                    # 50 / 19 > gamma
    assert model._levels(11)[0] == 0.2                             # 50 / 239 > gamma
    assert math.isclose(model._levels(12)[0], 50 / 259)            # gamma > 50 / 259 > alpha
    assert _fitted(300, 2)._levels(19) == (0.01, 0.01 / 5999)      # 50 / 5999 < alpha: the floor is alpha
    assert _fitted(1, 2)._levels(0) == (0.2, 0.01)                 # no candidate: m is set to 1


def test_levels_use_the_arguments():
    model = _fitted(20, 2, alpha=0.05, gamma=0.1, kappa=10)
    assert model._levels(0) == (0.1, 0.05 / 19)
    assert math.isclose(model._levels(9)[0], 10 / 199)
    assert math.isclose(model._levels(9)[1], 0.05 / 199)
    assert _fitted(300, 2, alpha=0.05)._levels(19) == (0.05, 0.05 / 5999)   # floor = the alpha argument
    assert ITPD().alpha == 0.01 and ITPD().gamma == 0.2 and ITPD().kappa == 50


@pytest.mark.parametrize("kwargs", [{"alpha": 0.0}, {"alpha": 1.0}, {"gamma": 0.0}, {"gamma": 1.0}, {"kappa": 0}])
def test_invalid_levels_are_rejected(kwargs):
    with pytest.raises(ValueError):
        ITPD(**kwargs)


def _constant_p(p, **kwargs):
    """N 2, T 2: every target has one candidate, so alpha1 = gamma and alpha2 = alpha. Target 2 has X = 0, Z = 1."""
    calls = []

    def ci(x, y, S):
        calls.append((x, y, tuple(S)))
        return p

    model = ITPD(ci, **kwargs).fit(np.zeros((2, 2, 2)))
    return model, [c for c in calls if c[1] == 2]


def test_levels_are_applied_on_the_lines():
    # alpha2 < p < alpha1: Lines 8, 9 and 10 answer "dependent", Line 14 answers "independent".
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
    # gamma and alpha move the two levels.
    model, calls = _constant_p(0.1, gamma=0.05)
    assert calls == [(1, 2, ())]
    model, calls = _constant_p(0.1, alpha=0.15)
    assert len(calls) == 4 and model.adjacency_[2] == {0, 1}
    # p equal to the level counts as dependent.
    model, calls = _constant_p(0.2)
    assert len(calls) == 4 and model.adjacency_[2] == {0}


# ---------------------------------------------------------------- a test that cannot be evaluated

def test_nan_p_value_lists_the_target_and_warns():
    def ci(x, y, S):
        return float("nan") if y == 4 else 0.0

    with pytest.warns(RuntimeWarning, match="untested_targets_") as record:
        model = ITPD(ci).fit(np.zeros((2, 4, 2)))
    assert len(record) == 1
    assert model.untested_targets_ == [4]
    assert model.adjacency_[4] == {2}                    # the lag-1 self edge only
    assert model.adjacency_[5] == {0, 1, 2, 3}           # the run continues
    assert model.adjacency_[6] == {0, 1, 2, 3, 4, 5}
    assert model.get_adjacency_matrix().shape == (8, 8)


def test_invalid_p_value_still_raises():
    with pytest.raises(ValueError):
        ITPD(lambda x, y, S: 1.5).fit(np.zeros((2, 2, 2)))


# ---------------------------------------------------------------- parity with runs behind the paper's results

# Linear Gaussian data, N 20, T 16, lag 1, Fisher-z, alpha 0.01. An instance is (expected in-degree d, graph index g)
# with generator seed 0; the data are the first M of 2000 rows. Scores exclude the lag-1 self edges.
FIXTURES = [
    # d, g, M, sha1 of the 2000 rows, tp, fp, fn, requested tests, unique tests, largest conditioning set
    (2.0, 0, 200, "f7df382ae9f36ffc0f960f94ba2ed9e41afa36db", 554, 7, 69, 63470, 62495, 25),
    (2.0, 1, 500, "5e2ef56f5c751c391e42191ef9d5b5649476f6ca", 591, 3, 7, 64769, 63818, 28),
    (1.0, 0, 2000, "37c8fcbb6c0126ecd13ee7f9849a012930b09bb1", 312, 1, 0, 65468, 64501, 29),
]


@pytest.mark.slow
@pytest.mark.parametrize("d, g, M, sha1, tp, fp, fn, requested, unique, largest", FIXTURES,
                         ids=[f"d{f[0]:g}-g{f[1]}-M{f[2]}" for f in FIXTURES])
def test_parity_with_stored_runs(d, g, M, sha1, tp, fp, fn, requested, unique, largest):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    observed_data = pytest.importorskip("itpd_aistats.observed_data")
    from itpd_aistats.ci import FisherZ, Recorder
    from itpd_aistats.metrics import edge_metrics

    N, T = 20, 16
    inst = observed_data.window_instance(N, T, 1, d, 0, g, "time")
    full = observed_data.observed_data(inst, 2000)
    if observed_data.data_hash(full) != sha1:
        pytest.skip("the random number streams of this platform give different data")
    X = np.ascontiguousarray(full[:M])
    recorder = Recorder(FisherZ(X.reshape(M, -1)), cache="run")

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)   # no target may be untested
        model = ITPD(lambda x, y, S: recorder(x, y, S), alpha=0.01).fit(X)

    assert model.untested_targets_ == []
    score = edge_metrics(inst["A"], model.get_adjacency_matrix(), N, T, include_self=False)
    assert (score["tp"], score["fp"], score["fn"]) == (tp, fp, fn)
    summary = recorder.summary()
    assert summary["raw_calls"] == requested == sum(model.ci_queries_.values())
    assert summary["unique_tests"] == unique
    assert max(size for size, count in summary["by_size_unique"].items() if count) == largest
    assert model.max_conditioning_size_ == largest
