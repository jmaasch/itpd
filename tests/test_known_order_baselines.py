"""Known-order IAMB per target, known-order lasso, ITPD + marginal-first, and the finite-data harness."""

import numpy as np
import pytest

from itpd import ci, iamb, lasso, run_oracle_counts as oracle_counts, sim
from itpd import dataset_eval as finite, method_runner as runlib
from itpd.graphs import TimeGraph, unroll


def _graph(kind, N, T, tau, d, rng):
    if kind == "time":
        return sim.sample_time_graph(N, T, d, tau, rng)
    B = sim.sample_window_graph(N, tau, d, rng)
    return TimeGraph(A=unroll(B, T), N=N, T=T, tau=tau, B=B)


GRAPHS = [(kind, N, T, tau, d, s) for s, (kind, N, T, tau, d) in enumerate(
    [("time", 4, 5, 1, 2), ("window", 5, 6, 2, 2), ("time", 6, 6, 2, 3), ("window", 6, 5, 1, 3), ("window", 8, 6, 3, 1),
     ("time", 8, 5, 2, 2)])]


# ------------------------------------------------------------------------------------------------ marginal-first

@pytest.mark.parametrize("kind,N,T,tau,d,seed", GRAPHS)
@pytest.mark.parametrize("lazy", [True, False])
def test_marginal_first_oracle_exact(kind, N, T, tau, d, seed, lazy):
    g = _graph(kind, N, T, tau, d, np.random.default_rng(100 + seed))
    o = runlib.run_s2_method("itpd_marginal_first", None, graph=g, alpha=0.01, lazy=lazy, order="time", per_target=True)
    assert o["metrics"]["exact"] and o["n_infeasible_targets"] == 0
    naive = runlib.run_s2_method("itpd_naive", None, graph=g, alpha=0.01, lazy=lazy, order="time", per_target=True)
    assert o["tests"]["unique_tests"] <= naive["tests"]["unique_tests"]       # no extra test over the naive extension
    # decisions are those of the naive extension (the Z8 skip is never taken, only the order of the two marginals differs)


def test_marginal_first_default_off():
    """The flag is off for every existing method name: itpd and itpd_naive issue the same tests as before the flag."""
    g = _graph("window", 5, 6, 2, 2, np.random.default_rng(5))
    a = runlib.run_s2_method("itpd", None, graph=g, alpha=0.01, lazy=True, order="time", per_target=True)
    b = runlib.run_s2_method("itpd", None, graph=g, alpha=0.01, lazy=True, order="time", per_target=True)
    assert a["tests"]["unique_tests"] == b["tests"]["unique_tests"]
    assert sum(v for k, v in a["skips"].items() if k.startswith("Z8:")) > 0       # the paper's Z8 skip rules are still in force for "itpd"


def test_marginal_first_unique_test_counts():
    """Oracle unique tests on the window graphs 0-5 of the cell N = 10, T = 8, tau = 1, d = 2 (seed 0): regression counts."""
    tot = {"itpd_naive": 0, "itpd": 0, "itpd_marginal_first": 0}
    for gi in range(6):
        graph, _ = oracle_counts.make_instance(10, 8, 1, 2, 0, gi, "window")
        for m in tot:
            o = runlib.run_s2_method(m, None, graph=graph, alpha=0.01, lazy=True, order="time", per_target=True)
            assert o["metrics"]["exact"]
            tot[m] += o["tests"]["unique_tests"]
    assert tot == {"itpd_naive": 45865, "itpd": 45680, "itpd_marginal_first": 45674}


# ------------------------------------------------------------------------------------------------ IAMB

class CallLog:
    def __init__(self, A):
        self.o = ci.DSepCI(A)
        self.calls = []

    def __call__(self, x, y, S):
        self.calls.append((x, y, tuple(S)))
        return self.o(x, y, S)


@pytest.mark.parametrize("kind,N,T,tau,d,seed", GRAPHS)
@pytest.mark.parametrize("tie", ["last", "first", "random:3"])
def test_iamb_oracle_exact_counts_and_self_edge(kind, N, T, tau, d, seed, tie):
    g = _graph(kind, N, T, tau, d, np.random.default_rng(200 + seed))
    spy = CallLog(g.A)
    rec = ci.Recorder(spy, cache="run", infeasible="raise")
    res = iamb.run_s2(rec, N, T, 0.01, per_target=True, tie=tie)
    from itpd.metrics import edge_metrics
    m = edge_metrics(g.A, res.A_hat, N, T)
    assert m["exact"] and not res.infeasible_targets
    # the known self edge is never tested for its own target, and every conditioning set contains the target's self parent
    own = {tuple(sorted(p["pair"])) for p in res.per_pair}
    for x, y, S in spy.calls:
        assert (x, y) not in own
        assert (y - N) in S
    # counts: the grow phase issues (g + 1) passes over the candidates outside CMB (g = members grown, a shrinking list), all with new
    # conditioning sets, so its unique tests are sum (g + 1) nc - g (g + 1) / 2; the shrink phase issues at most g more
    grow = shrink = 0
    for p in res.per_pair:
        gn, nc = p["grown"], p["n_cand"]
        grow += (gn + 1) * nc - gn * (gn + 1) // 2
        shrink += gn
        assert p["max_size"] == (gn + 1 if gn < nc else gn) and len(p["parents"]) == gn - p["shrunk"]
        assert gn >= len(p["parents"])
    assert res.summary["by_label_unique"]["grow"] == grow
    assert res.summary["by_label_unique"].get("shrink", 0) <= shrink


def test_iamb_infeasible_targets_flagged():
    rng = np.random.default_rng(1)
    N, T, M = 3, 4, 5
    X = rng.standard_normal((M, T, N))
    rec = ci.Recorder(ci.FisherZ(X.reshape(M, T * N)), cache="run", infeasible="raise")
    res = iamb.run_s2(rec, N, T, 0.5, per_target=True)           # lenient alpha: grow adds members, sets of size 2 are infeasible (n = 5)
    assert len(res.infeasible_targets) > 0
    for y in res.infeasible_targets:
        assert y not in res.pa
    with pytest.raises(ci.InfeasibleTest):
        iamb.run_s2(ci.Recorder(ci.FisherZ(X.reshape(M, T * N)), cache="run", infeasible="raise"), N, T, 0.5, per_target=False)


def test_neglogp_matches_and_does_not_underflow():
    rng = np.random.default_rng(2)
    D = rng.standard_normal((300, 4))
    D[:, 1] += 0.4 * D[:, 0]
    D[:, 2] = D[:, 0] * 30 + 0.001 * rng.standard_normal(300)
    fz = ci.FisherZ(D)
    assert abs(fz.neglogp(0, 1, [3]) + np.log(fz(0, 1, [3]))) < 1e-8
    assert fz(0, 2, []) == 0.0 and fz.neglogp(0, 2, []) > 700                  # p underflows, -log p does not


def test_iamb_finite_data_sane_and_small_sets():
    rng = np.random.default_rng(3)
    s = sim.simulate_s2(6, 6, 2000, 2, 1, rng, graph="window")
    o = runlib.run_s2_method("iamb_known_order", s, ci_kind="fisherz", alpha=0.01, per_target=True, order="time")
    assert o["status"] == "ok" and o["metrics"]["f1"] > 0.8
    order = runlib.run_s2_method("order", s, ci_kind="fisherz", alpha=0.01, per_target=True, order="time")
    assert max(p["max_size"] for p in o["per_pair"]) < max(p["max_size"] for p in order["per_pair"])
    assert o["iamb_stats"]["grown"] >= o["iamb_stats"]["shrunk"]


# ------------------------------------------------------------------------------------------------ lasso

def test_lasso_known_order_recovers_parents_and_path_is_monotone():
    rng = np.random.default_rng(4)
    s = sim.simulate_s2(6, 6, 1500, 2, 1, rng, graph="window")
    N, T = 6, 6
    r = lasso.lasso_s2(s.X)
    from itpd.metrics import edge_metrics
    m = edge_metrics(s.graph.A, r.A_cv, N, T)
    assert m["recall"] > 0.8 and m["fp_nonforward"] == 0
    n_sel = [int((A != 0).sum()) for A in r.A_path]                  # lambdas decrease along the grid
    assert n_sel[0] < n_sel[-1] and n_sel[0] >= N * (T - 1)          # the self edges are always present
    assert m["recall"] <= edge_metrics(s.graph.A, r.A_path[-1], N, T)["recall"] + 0.1
    for t in range(1, T):
        for n in range(N):
            assert r.A_cv[(t - 1) * N + n, t * N + n] == 1             # self edge always present
    assert lasso.LAMBDAS[0] > lasso.LAMBDAS[-1]


# ------------------------------------------------------------------------------------------------ harness

def test_known_order_specs_through_run_dataset():
    rng = np.random.default_rng(5)
    s = sim.simulate_s2(5, 5, 300, 2, 1, rng, graph="window")
    specs = tuple(sp[:4] + ((0.01,),) for sp in finite.IAMB_MARGINAL_FIRST_SPECS)
    res = finite.run_dataset(s.graph.A, s.X, 1, specs, order="time")
    names = [r["name"] for r in res["runs"]]
    assert names == ["iamb_known_order", "itpd_marginal_first", "itpd_marginal_first_nonlazy"]
    r = res["runs"][0]
    assert r["status"] == "ok" and r["iamb_stats"]["grow_iters"] > 0 and len(r["target_max"]) == 5 * 4
    assert res["runs"][1]["unique"] <= res["runs"][2]["unique"]


def test_lasso_ebic_and_fixed_penalty_select_fewer_than_cv():
    rng = np.random.default_rng(6)
    s = sim.simulate_s2(8, 6, 400, 2, 1, rng, graph="window")
    N, T = 8, 6
    from itpd.metrics import edge_metrics
    r = lasso.lasso_s2(s.X, do_ebic=True, do_fixed=True, do_path=False)
    n = {k: edge_metrics(s.graph.A, getattr(r, a), N, T) for k, a in (("cv", "A_cv"), ("ebic", "A_ebic"), ("fixed", "A_fixed"))}
    assert n["ebic"]["fp"] <= n["cv"]["fp"] and n["fixed"]["fp"] <= n["cv"]["fp"]
    assert n["ebic"]["recall"] > 0.6 and n["fixed"]["recall"] > 0.6
    for A in (r.A_ebic, r.A_fixed):
        assert all(A[(t - 1) * N + n_, t * N + n_] == 1 for t in range(1, T) for n_ in range(N))
