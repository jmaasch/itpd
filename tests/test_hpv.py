"""HPV (itpd/hpv.py): hint construction, oracle exactness for every hint, call counts and set sizes, re-check,
cap and infeasible-test handling, finite-data harness."""
import numpy as np
import pytest

from itpd import ci, hpv, sim
from itpd import dataset_eval as finite, method_runner as runlib
from itpd.graphs import TimeGraph, unroll

HINTS = ("learned_blanket", "shifted_parents", "union", "oracle_blanket", "none")


def _graph(kind, N, T, tau, d, rng):
    if kind == "time":
        return sim.sample_time_graph(N, T, d, tau, rng)
    B = sim.sample_window_graph(N, tau, d, rng)
    return TimeGraph(A=unroll(B, T), N=N, T=T, tau=tau, B=B)


def _run(g, **kw):
    return runlib.run_s2_method("hpv", None, graph=g, alpha=0.01, per_target=True, hpv=kw)


class Spy:
    """Oracle CI that records the phase-A conditioning set of every candidate."""

    def __init__(self, A):
        self.o = ci.DSepCI(A)
        self.calls = []

    def __call__(self, x, y, S):
        self.calls.append((x, y, tuple(S)))
        return self.o(x, y, S)


def _true_mb(A, z, lo):
    """Markov blanket of z in the true graph restricted to the nodes < lo."""
    pa = set(np.nonzero(A[:lo, z])[0])
    ch = set(c for c in np.nonzero(A[z, :lo])[0])
    b = pa | ch
    for c in ch:
        b |= set(np.nonzero(A[:lo, c])[0])
    b.discard(z)
    return b


def test_blankets_from_learned_graph():
    # 0 -> 2, 1 -> 2, 2 -> 3, 4 -> 3, 5 isolated
    pa = [set(), set(), {0, 1}, {2, 4}, set(), set()]
    ch = [{2}, {2}, {3}, set(), {3}, set()]
    b = hpv.blankets(pa, ch, range(6))
    assert b[0] == {2, 1} and b[1] == {2, 0} and b[2] == {0, 1, 3, 4}
    assert b[3] == {2, 4} and b[4] == {3, 2} and b[5] == set()
    assert hpv.blanket_bound(pa, ch, 2) == 2 + 2 and hpv.blanket_bound(pa, ch, 5) == 0


@pytest.mark.parametrize("hint", HINTS)
def test_phase_A_sets_match_hint_definition(hint):
    # under the oracle G_hat = G, so the learned-graph hints must equal their true-graph definitions
    rng = np.random.default_rng(3)
    for kind in ("time", "window"):
        g = _graph(kind, 4, 6, 2, 2, rng)
        A, N = g.A, g.N
        spy = Spy(A)
        rec = ci.Recorder(spy, infeasible="raise")
        res = hpv.run_s2(rec, N, g.T, 0.01, hint=hint, A_true=A)
        assert res.infeasible_targets == []
        labels = rec.raw_label
        assert labels["A"] == sum(p["n_cand"] for p in res.per_pair)
        # recompute the expected phase-A key of every candidate
        expect = set()
        for t in range(1, g.T):
            lo = t * N
            for n in range(N):
                x, y = (t - 1) * N + n, t * N + n
                shift = {p + N for p in np.nonzero(A[:, x])[0]}
                for z in range(lo):
                    if z == x:
                        continue
                    H = {"none": set(), "shifted_parents": shift, "learned_blanket": _true_mb(A, z, lo), "oracle_blanket": _true_mb(A, z, lo),
                         "union": _true_mb(A, z, lo) | shift}[hint]
                    expect.add(ci.canonical_key(z, y, (H | {x}) - {z}))
        got = set(ci.canonical_key(*c) for c in spy.calls)
        assert expect <= got


def test_oracle_exact_every_hint_and_theorem4_on_500_graphs():
    rng = np.random.default_rng(2026)
    n_graphs, tpc = 0, []
    configs = [(N, T, tau, d, kind) for N in (2, 3, 4, 6) for T in (4, 6, 8) for tau in (1, 2, 3) for d in (1, 2, 3)
               for kind in ("time", "window")]
    for N, T, tau, d, kind in configs:
        reps = 3 if N <= 4 else 2
        for _ in range(reps):
            g = _graph(kind, N, T, tau, d, rng)
            A = g.A
            n_graphs += 1
            din, dout = int(A.sum(0).max()), int(A.sum(1).max())
            E_nonself = int(A.sum()) - N * (T - 1)
            runs = {}
            for hint in HINTS:
                o = _run(g, hint=hint)
                assert o["metrics"]["exact"], (hint, N, T, tau, d, kind)
                runs[hint] = o
            rc = _run(g, hint="learned_blanket", recheck=True)
            assert rc["metrics"]["exact"]
            sum_c = sum(p["n_cand"] for p in rc["per_pair"])
            # re-check under the oracle: A' empty; raw = 2|C| + |R| per target; always-verify == shortcut
            assert rc["hpv_stats"]["sum_Aprime"] == 0 and rc["hpv_stats"]["shortcut_diff"] == 0
            for p in rc["per_pair"]:
                assert p["raw"] == 2 * p["n_cand"] + p["n_R"]
            # calls identity for the blanket hints; perfect pruning
            for hint in ("learned_blanket", "oracle_blanket", "union"):
                o = runs[hint]
                assert o["tests"]["raw_calls"] == sum_c + E_nonself, hint
                assert o["hpv_stats"]["excess"] == 0 and o["hpv_stats"]["lost"] == 0
                for p in o["per_pair"]:
                    y = p["pair"][1]
                    npa = int(A[:, y].sum())
                    assert p["max_SB"] == (npa - 1 if npa > 1 else 0)
                    if hint != "union":
                        assert p["max_SA"] <= p["b_max"] + 1           # per target, b computed in G_hat = G
                        assert p["max_SA"] <= din * (1 + dout) + 1    # global degrees
            assert o["tests"]["unique_tests"] <= o["tests"]["raw_calls"]
            mb = runs["learned_blanket"]
            assert (mb["tests"]["raw_calls"] - sum_c) / sum_c <= 2 * (din - 1) / (N * T - 2) + 1e-12   # per-candidate overhead bound
            # oracle_blanket == learned_blanket under the oracle
            assert [(p["pair"], p["parents"], p["max_SA"], p["new_unique"]) for p in mb["per_pair"]] == \
                   [(p["pair"], p["parents"], p["max_SA"], p["new_unique"]) for p in runs["oracle_blanket"]["per_pair"]]
            tpc.append(mb["tests"]["unique_tests"] / sum_c)
            # shift on stationary window graphs: perfect pruning except at targets of time 2..tau, whose lag-s parents
            # sit at time 0 and have no shifted copy among X's parents
            if kind == "window":
                for p in runs["shifted_parents"]["per_pair"]:
                    ty = p["pair"][1] // N
                    if not 2 <= ty <= tau:
                        assert p["excess"] == 0, (tau, ty)
    assert n_graphs >= 500
    assert 1.0 <= min(tpc) and max(tpc) <= 2.0


def _dense_data(M, seed=5):
    rng = np.random.default_rng(seed)
    return sim.simulate_s2(6, 6, M, 3, 2, rng, graph="window")


def test_cap_never_issues_infeasible_phase_A_test():
    # the true-blanket hint is used so that the blankets are large (a learned graph at M = 14 is nearly empty)
    r = _dense_data(14)
    X = r.X.reshape(r.X.shape[0], -1)
    o_cap = runlib.run_s2_method("hpv", None, graph=r.graph, ci=ci.FisherZ(X), alpha=0.05, per_target=True,
                                 hpv={"hint": "oracle_blanket", "alpha_A": 0.2, "cap": "auto"})
    st = o_cap["hpv_stats"]
    assert st["n_capped"] > 0 and st["a_inf"] == 0
    for p in o_cap["per_pair"]:
        assert p.get("max_SA", 0) <= 14 - 4
    o_unc = runlib.run_s2_method("hpv", None, graph=r.graph, ci=ci.FisherZ(X), alpha=0.05, per_target=True,
                                 hpv={"hint": "oracle_blanket", "alpha_A": 0.2})
    assert o_unc["hpv_stats"]["a_inf"] > 0               # uncapped: infeasible phase-A tests happen, are counted
    for p in o_unc["per_pair"]:                          # ... and the candidate is kept, never pruned
        if not p.get("infeasible"):
            assert p["n_R"] >= p["a_inf"]


def test_infeasible_phase_B_marks_target_never_independent():
    class NanAbove:
        """Oracle answers, but NaN (infeasible) for |S| > k."""

        def __init__(self, A, k):
            self.o, self.k = ci.DSepCI(A), k

        def __call__(self, x, y, S):
            return float("nan") if len(S) > self.k else self.o(x, y, S)

    rng = np.random.default_rng(8)
    g = _graph("time", 5, 6, 2, 3, rng)
    rec = ci.Recorder(NanAbove(g.A, 2), infeasible="raise")
    res = hpv.run_s2(rec, g.N, g.T, 0.01, hint="none", A_true=g.A)
    assert res.infeasible_targets                      # some phase-B set has more than 2 members
    for p in res.per_pair:
        y = p["pair"][1]
        if p.get("infeasible"):
            assert p["parents"] == [] and y not in res.pa and res.A_hat[:, y].sum() == 0
        else:   # decided targets are still exact (phase-A NaN only keeps candidates)
            assert set(p["parents"]) == set(np.nonzero(g.A[:, y])[0]) - {p["pair"][0]}


def test_finite_harness_hpv_rows():
    r = _dense_data(300, seed=1)
    specs = (("hpv_mb_eq", "hpv", {"hint": "learned_blanket"}, True, (0.01, 0.05)),
             ("hpv_mb_len", "hpv", {"hint": "learned_blanket", "alpha_A": 0.1}, True, (0.01,)),
             ("hpv_omb", "hpv", {"hint": "oracle_blanket", "alpha_A": 0.1, "keep_parent_tests": True}, True, (0.01,)),
             ("hpv_rc", "hpv", {"hint": "learned_blanket", "alpha_A": 0.1, "recheck": True}, True, (0.01,)),
             ("order", "order", None, False, (0.01,)))
    from itpd.observed_data import true_edge_list
    out = finite.run_dataset(r.graph.A, r.X, 2, specs, order="time", edges=true_edge_list(r.graph.A, 6, 6))
    rows = {(x["name"], x["alpha"]): x for x in out["runs"]}
    a = rows[("hpv_mb_len", 0.01)]
    assert a["alpha_A"] == 0.1 and a["metrics_common"] is not None and "A" in a["by_label_unique"]
    assert len(a["target_n_R"]) == 30 and "fn_idx" in a
    assert rows[("hpv_mb_eq", 0.05)]["alpha_A"] == 0.05 and "target_n_R" not in rows[("hpv_mb_eq", 0.05)]
    assert rows[("hpv_omb", 0.01)]["hpv_ptests"]
    rc = rows[("hpv_rc", 0.01)]
    assert rc["hpv_stats"]["shortcut_diff"] == 0 and rc["unique"] >= a["unique"]
