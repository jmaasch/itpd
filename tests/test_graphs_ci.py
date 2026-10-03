"""d-separation oracle, the recorder (unique and raw counts), Fisher-z, GCM."""
import sys

import numpy as np
import networkx as nx
import pytest

from itpd import graphs, ci, sim


def _random_dag(n, p, rng):
    A = np.triu(rng.random((n, n)) < p, 1)
    return A


def test_dsep_matches_networkx():
    rng = np.random.default_rng(1)
    for _ in range(300):
        n = int(rng.integers(4, 11))
        A = _random_dag(n, rng.uniform(0.15, 0.5), rng)
        G = nx.from_numpy_array(A.astype(int), create_using=nx.DiGraph)
        d = graphs.DSep(A)
        for _ in range(8):
            x, y = rng.choice(n, 2, replace=False)
            rest = [v for v in range(n) if v not in (x, y)]
            S = [v for v in rest if rng.random() < 0.4]
            assert d.dsep(int(x), int(y), S) == nx.is_d_separator(G, {int(x)}, {int(y)}, set(S))


def test_dsep_does_not_mutate_and_ignores_x_y_in_S():
    A = np.zeros((3, 3), bool)
    A[0, 1] = A[1, 2] = True
    S = [0, 1, 2]
    d = graphs.DSep(A)
    assert d.dsep(0, 2, S) == d.dsep(0, 2, [1])
    assert S == [0, 1, 2]


def test_oracle_uses_full_graph_with_dropped_nodes():
    # The common cause must act as latent when it is not in S.
    # 0 -> 1, 0 -> 2: 1 and 2 are dependent given the empty set although 0 is not a candidate/conditioner.
    A = np.zeros((3, 3), bool)
    A[0, 1] = A[0, 2] = True
    o = ci.DSepCI(A)
    assert o(1, 2, []) == 0.0
    assert o(1, 2, [0]) == 1.0


def test_unroll_and_assumptions():
    rng = np.random.default_rng(0)
    B = sim.sample_window_graph(3, 2, 2, rng)
    A = graphs.unroll(B, 6)
    g = graphs.TimeGraph(A=A, N=3, T=6, tau=2, B=B)
    c = graphs.check_assumptions(g)
    assert c["autocorr_present"] and c["no_contemporaneous"] and c["forward_only"] and c["roots_at_t0"]
    assert c["max_lag"] <= 2


def test_recorder_counts():
    A = np.zeros((6, 6), bool)
    A[0, 3] = A[1, 3] = A[3, 5] = True
    rec = ci.Recorder(ci.DSepCI(A), cache="run")
    rec(0, 3, [], "a")
    rec(3, 0, [], "b")            # same unordered pair, repeat
    rec(0, 3, [1, 2], "a")
    rec(0, 3, [2, 1, 0], "c")      # x in S is stripped, same key as the previous one
    rec(1, 5, [0, 3], "c")
    s = rec.summary()
    assert s["raw_calls"] == 5 and s["unique_tests"] == 3
    assert sum(s["by_size_raw"].values()) == s["raw_calls"]
    assert sum(s["by_size_unique"].values()) == s["unique_tests"]
    assert s["by_size_raw"] == {0: 2, 2: 3}
    assert s["by_label_unique"] == {"a": 2, "c": 1}      # attributed to the first issuer
    assert s["cache_hits"] == 2 and s["evaluated"] == 3
    assert s["seconds_ci"] >= 0 and s["seconds_wall"] > 0


@pytest.mark.parametrize("scope,evaluated", [("none", 4), ("pair", 3), ("run", 2)])
def test_recorder_cache_scopes(scope, evaluated):
    rec = ci.Recorder(lambda x, y, S: 0.5, cache=scope)
    rec(0, 1, [])
    rec(0, 1, [])
    rec.new_scope()
    rec(0, 1, [])
    rec(2, 1, [])
    s = rec.summary()
    assert s["raw_calls"] == 4 and s["evaluated"] == evaluated
    assert s["unique_tests"] == 2


def test_fisherz_matches_causallearn():
    from causallearn.utils.cit import CIT
    rng = np.random.default_rng(3)
    X = rng.standard_normal((300, 6))
    X[:, 2] += 0.7 * X[:, 0] + 0.5 * X[:, 1]
    X[:, 3] += 0.6 * X[:, 2]
    fz = ci.FisherZ(X)
    ref = CIT(X, "fisherz")
    for x, y, S in [(0, 3, []), (0, 3, [2]), (1, 3, [0, 2]), (4, 5, [0, 1, 2]), (0, 1, [2])]:
        assert fz(x, y, S) == pytest.approx(ref(x, y, S), rel=1e-6, abs=1e-12)


def test_fisherz_infeasible_is_never_silent():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 14))
    fz = ci.FisherZ(X)
    S = list(range(2, 12))
    assert np.isnan(fz(0, 1, S))
    with pytest.raises(ci.InfeasibleTest):                 # default: raise
        ci.Recorder(fz)(0, 1, S)
    rec = ci.Recorder(fz, infeasible="flag")                # diagnostics: continue, count, flag
    assert rec(0, 1, S) == 1.0
    s = rec.summary()
    assert s["p_nan"] == 1 and s["infeasible"] and s["first_infeasible_size"] == 10


def test_infeasible_method_cell_is_reported_not_scored():
    from itpd import method_runner as runlib, sim
    rng = np.random.default_rng(0)
    r = sim.simulate_s2(4, 6, 12, 2, 2, rng)               # M = 12 < largest conditioning set + 3
    o = runlib.run_s2_method("order", r, ci_kind="fisherz")
    assert o["status"] == "infeasible" and o["metrics"] is None
    o = runlib.run_s2_method("order", r, ci_kind="fisherz", infeasible="flag")
    assert o["status"] == "infeasible" and o["metrics"] is None and o["tests"]["p_nan"] > 0


def test_gcm_detects_nonlinear_dependence():
    rng = np.random.default_rng(0)
    n = 600
    z = rng.standard_normal(n)
    x = np.sin(2 * z) + 0.3 * rng.standard_normal(n)
    y = np.cos(z) + 0.3 * rng.standard_normal(n)
    w = rng.standard_normal(n)
    D = np.column_stack([x, y, z, w])
    g = ci.GCM(D)
    assert g(0, 2, []) > -1          # runs
    assert g(0, 3, []) > 0.01        # independent
    assert g(0, 1, [2]) > 0.01       # independent given z
    y2 = np.tanh(x) + 0.3 * rng.standard_normal(n)
    D2 = np.column_stack([x, y2, w])
    assert ci.GCM(D2)(0, 1, [2]) < 1e-3     # dependent given an irrelevant variable


# ---- DSepCI answers the same under both oracles --------------------------------------------

def _oracle(monkeypatch, mode, A):
    monkeypatch.setenv("ITPD_ORACLE", mode)
    return ci.DSepCI(A)


def _edge():
    A = np.zeros((3, 3), bool)
    A[0, 1] = True
    return A


def _install_stub_fastdsep(monkeypatch):
    """A stand-in for the optional `fastdsep` package, so the selection logic is tested with and without the real one."""
    import types
    mod = types.ModuleType("fastdsep")

    class FastDSep:
        backend = "stub"

        def __init__(self, A):
            self.inner = graphs.DSep(A)

        def dsep(self, x, y, S):
            return self.inner.dsep(x, y, S)

    mod.FastDSep = FastDSep
    monkeypatch.setitem(sys.modules, "fastdsep", mod)


def test_oracle_switch_backend_names(monkeypatch):
    A = _edge()
    assert _oracle(monkeypatch, "old", A).backend == "python"
    assert _oracle(monkeypatch, "fast", A).backend in ("fastdsep-c", "fastdsep-python", "python")
    monkeypatch.delenv("ITPD_ORACLE")
    assert ci.DSepCI(A).backend == _oracle(monkeypatch, "fast", A).backend           # the default is "fast"
    monkeypatch.setenv("ITPD_ORACLE", "bogus")
    with pytest.raises(ValueError):
        ci.DSepCI(A)


def test_default_oracle_uses_fastdsep_when_importable_and_old_forces_python(monkeypatch):
    _install_stub_fastdsep(monkeypatch)
    A = _edge()
    monkeypatch.delenv("ITPD_ORACLE", raising=False)
    assert ci.DSepCI(A).backend == "fastdsep-stub"
    assert _oracle(monkeypatch, "fast", A).backend == "fastdsep-stub"
    old = _oracle(monkeypatch, "old", A)
    assert old.backend == "python" and isinstance(old.d, graphs.DSep)


def test_default_oracle_falls_back_to_python_without_fastdsep(monkeypatch):
    monkeypatch.setitem(sys.modules, "fastdsep", None)          # makes `import fastdsep` raise ImportError
    A = _edge()
    for mode in (None, "fast", "old"):
        if mode is None:
            monkeypatch.delenv("ITPD_ORACLE", raising=False)
        else:
            monkeypatch.setenv("ITPD_ORACLE", mode)
        o = ci.DSepCI(A)
        assert o.backend == "python" and isinstance(o.d, graphs.DSep)
        assert o(0, 1, []) == 0.0 and o(0, 2, []) == 1.0


def test_fast_oracle_parity_with_old_on_time_graphs(monkeypatch):
    """Same answers, DSepCI(old) vs DSepCI(fast), on random time graphs (both arms) and conditioning sets of every
    regime (empty, small, full history of the later node)."""
    rng = np.random.default_rng(7)
    checked = 0
    for kind in ("time", "window"):
        for (N, T, tau) in ((4, 5, 2), (6, 8, 1), (10, 6, 3)):
            r = sim.simulate_s2(N, T, 0, 2, tau, rng, graph=kind)
            A = r.graph.A
            old, fast = _oracle(monkeypatch, "old", A), _oracle(monkeypatch, "fast", A)
            n = A.shape[0]
            for _ in range(400):
                x, y = (int(v) for v in rng.choice(n, 2, replace=False))
                reg = int(rng.integers(0, 3))
                if reg == 0:
                    S = []
                elif reg == 1:
                    S = [int(v) for v in rng.choice(n, int(rng.integers(1, 5)), replace=False)]
                else:
                    S = list(range(0, max(x, y)))               # all earlier columns (time-major numbering)
                S = [s for s in S if s not in (x, y)]
                assert old(x, y, S) == fast(x, y, S)
                checked += 1
    assert checked == 2400


def test_method_outputs_and_counts_identical_under_both_oracles(monkeypatch):
    """Unique counts, raw counts, by-size histograms and graphs of ITPD_naive, ITPD and the order-based run do not
    depend on the oracle."""
    from itpd import method_runner as runlib
    rng = np.random.default_rng(3)
    g = sim.simulate_s2(5, 7, 0, 2, 2, rng, graph="time").graph
    for method in ("itpd_naive", "itpd", "order"):
        outs = []
        for mode in ("old", "fast"):
            monkeypatch.setenv("ITPD_ORACLE", mode)
            o = runlib.run_s2_method(method, None, graph=g, tau_max=None, variant="paper", order="time", lazy=True)
            t = o["tests"]
            outs.append((t["unique_tests"], t["raw_calls"], t["by_size_unique"], t["by_size_raw"],
                         o["metrics"]["exact"], o["metrics"]["fp"], o["metrics"]["fn"]))
        assert outs[0] == outs[1], method
