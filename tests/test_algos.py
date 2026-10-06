"""PaDL / ITPD / ITPD_naive / full-conditioning: regression tests for earlier bugs of the original code, and exact recovery with an oracle."""
import numpy as np
import pytest

from itpd import ci, itpd as I, metrics, sim
from itpd import method_runner as runlib
from itpd.padl import padl

ALPHA = 0.01


def _run(g, method, tau_max=None, **kw):
    return runlib.run_s2_method(method, None, graph=g, tau_max=tau_max, **kw)


def test_paper_variant_keeps_skipped_z4_parent():
    # Z=0 -> Y=2 <- X=1, Z independent of X: a Z4 parent. If its Z4 test is skipped (unlicensed skip),
    # the variant of the original code ("repo") needs X dep Z marginally and drops it; the paper variant keeps it.
    A = np.zeros((3, 3), bool)
    A[0, 2] = A[1, 2] = True
    skip = {"Z4": {0: "manual"}, "Z8": {}}
    for variant, expect in (("paper", [0]), ("repo", [])):
        rec = ci.Recorder(ci.DSepCI(A))
        assert padl(1, 2, [0], rec, ALPHA, irrelevant=skip, variant=variant).parents == expect
    rec = ci.Recorder(ci.DSepCI(A))
    assert padl(1, 2, [0], rec, ALPHA, variant="repo").parents == [0]       # without the skip both are right


def test_oracle_full_graph_with_tau_max():
    # the induced-subgraph oracle of the original code made ITPD wrong on 60/60 such graphs (N=3, T=7, lags <= 2, tau_max=2)
    rng = np.random.default_rng(11)
    for _ in range(60):
        g = sim.sample_time_graph(3, 7, 2, 2, rng)
        for m in ("itpd_naive", "itpd", "full_conditioning"):
            r = runlib.run_s2_method(m, None, graph=g, tau_max=2)
            assert r["metrics"]["exact"], m


def test_counter_covers_all_sizes():
    rng = np.random.default_rng(0)
    g = sim.sample_time_graph(3, 6, 2, 2, rng)
    rec = ci.Recorder(ci.DSepCI(g.A), cache="none")
    res = I.run_s2(rec, 3, 6, ALPHA, reuse=False)
    s = res.summary
    assert sum(s["by_size_raw"].values()) == s["raw_calls"] == sum(p["raw"] for p in res.per_pair)
    assert 0 in s["by_size_raw"] and 1 in s["by_size_raw"]        # the sizes the counter of the original code never recorded
    # adjacency tests: the size recorded is exactly the size used (the original code logged it off by one).
    # a -> X, b -> X, a -> Y, b -> Y, X -> Y: both candidates are unlabelled, S = {other candidate, X}.
    A = np.zeros((4, 4), bool)
    A[0, 2] = A[1, 2] = A[0, 3] = A[1, 3] = A[2, 3] = True
    rec = ci.Recorder(ci.DSepCI(A), keep_calls=True)
    padl(2, 3, [0, 1], rec, ALPHA)
    adj = [(key, lab) for key, lab, _ in rec.calls if lab == "adj"]
    assert len(adj) == 2 and all(len(key[2]) == 2 for key, _ in adj)
    assert rec.summary()["by_label_size_unique"]["adj"] == {2: 2}


def test_inputs_not_mutated():
    A = np.zeros((4, 4), bool)
    A[0, 2] = A[1, 2] = A[2, 3] = True
    cand = [0, 3]
    rec = ci.Recorder(ci.DSepCI(A))
    padl(1, 2, cand, rec, ALPHA)
    assert cand == [0, 3]


def test_order_baseline_one_test_per_candidate():
    rng = np.random.default_rng(1)
    g = sim.sample_time_graph(4, 6, 2, 2, rng)
    r = _run(g, "full_conditioning")
    n_cand = sum(N * t - 1 for N in (4,) for t in range(1, 6)) * 4
    assert r["tests"]["unique_tests"] == r["tests"]["raw_calls"] == n_cand
    assert r["metrics"]["exact"]


def test_time_major_equals_variable_major():
    rng = np.random.default_rng(2)
    for _ in range(10):
        g = sim.sample_time_graph(4, 6, 2, 2, rng)
        for reuse in (False, True):
            a = I.run_s2(ci.Recorder(ci.DSepCI(g.A)), 4, 6, ALPHA, reuse=reuse, order="variable")
            b = I.run_s2(ci.Recorder(ci.DSepCI(g.A)), 4, 6, ALPHA, reuse=reuse, order="time")
            assert (a.A_hat == b.A_hat).all()
            assert a.summary["unique_tests"] == b.summary["unique_tests"]
            assert a.summary["by_size_unique"] == b.summary["by_size_unique"]


def test_itpd_skips_tests_and_reports_rules():
    rng = np.random.default_rng(3)
    g = sim.sample_time_graph(5, 8, 2, 2, rng)
    rn = I.run_s2(ci.Recorder(ci.DSepCI(g.A), cache="none"), 5, 8, ALPHA, reuse=False)
    ri = I.run_s2(ci.Recorder(ci.DSepCI(g.A), cache="none"), 5, 8, ALPHA, reuse=True)
    assert ri.summary["raw_calls"] < rn.summary["raw_calls"]
    assert sum(v for k, v in ri.skips.items() if k != "memo:marginal") > 0
    assert set(rn.skips) <= {"memo:marginal"}
    r_off = I.run_s2(ci.Recorder(ci.DSepCI(g.A), cache="none"), 5, 8, ALPHA, reuse=True, rules=())
    assert r_off.summary["raw_calls"] == rn.summary["raw_calls"]


def test_oracle_exact_recovery_500_graphs():
    """ITPD, ITPD_naive and the full-conditioning baseline recover the true graph; ITPD output == ITPD_naive output.
    Both full history and tau_max = the true max lag (tau_max below the true lag is not expected to be exact)."""
    rng = np.random.default_rng(2026)
    n = 0
    while n < 500:
        N = int(rng.integers(2, 7))
        T = int(rng.integers(3, 9))
        d = float(rng.choice([1.0, 2.0, 3.0]))
        tau = int(rng.integers(1, 4))
        g = sim.sample_time_graph(N, T, d, tau, rng) if rng.random() < 0.7 else \
            sim.sample_nonstationary_graph(N, T, d, tau, rng)
        tm = None if n % 2 == 0 else tau
        outs = {m: I_ for m, I_ in ((m, runlib.run_s2_method(m, None, graph=g, tau_max=tm,
                                                             order="time" if n % 3 == 0 else "variable"))
                                    for m in ("itpd_naive", "itpd", "full_conditioning"))}
        for m, o in outs.items():
            assert o["metrics"]["exact"], (m, N, T, d, tau, tm)
        a = I.run_s2(ci.Recorder(ci.DSepCI(g.A)), N, T, ALPHA, reuse=True, tau_max=tm)
        b = I.run_s2(ci.Recorder(ci.DSepCI(g.A)), N, T, ALPHA, reuse=False, tau_max=tm)
        assert (a.A_hat == b.A_hat).all()
        assert outs["itpd"]["tests"]["unique_tests"] <= outs["itpd_naive"]["tests"]["unique_tests"]
        n += 1


def test_s1_oracle_recovers_window_graph():
    rng = np.random.default_rng(7)
    for _ in range(60):
        N = int(rng.integers(2, 6))
        tau = int(rng.integers(1, 4))
        s = sim.simulate_s1(N, 25, float(rng.choice([1, 2, 3])), tau, rng)
        for m in ("itpd_naive", "itpd", "full_conditioning"):
            r = runlib.run_s1_method(m, s)
            assert r["metrics"]["exact"], (m, N, tau)
        a = runlib.run_s1_method("itpd", s)
        b = runlib.run_s1_method("itpd_naive", s)
        assert a["tests"]["unique_tests"] == b["tests"]["unique_tests"]       # no cross-pair rule fires in S1


def test_finite_data_pipeline_runs():
    rng = np.random.default_rng(0)
    r = sim.simulate_s2(3, 5, 400, 1, 2, rng)
    for m in ("itpd_naive", "itpd", "full_conditioning"):
        o = runlib.run_s2_method(m, r, ci_kind="fisherz", tau_max=2)
        assert o["tests"]["raw_calls"] > 0 and "f1" in o["metrics"]
    s1 = sim.simulate_s1(3, 400, 1, 2, rng)
    o = runlib.run_s1_method("itpd", s1, ci_kind="fisherz")
    assert o["tests"]["raw_calls"] > 0


def test_lazy_gives_same_graph_and_fewer_or_equal_tests():
    rng = np.random.default_rng(9)
    for _ in range(30):
        N, T = int(rng.integers(2, 6)), int(rng.integers(3, 8))
        g = sim.sample_time_graph(N, T, 2.0, 2, rng)
        for reuse in (False, True):
            for variant in ("paper", "repo"):
                a = I.run_s2(ci.Recorder(ci.DSepCI(g.A)), N, T, ALPHA, reuse=reuse, variant=variant, lazy=False)
                b = I.run_s2(ci.Recorder(ci.DSepCI(g.A)), N, T, ALPHA, reuse=reuse, variant=variant, lazy=True)
                assert (a.A_hat == b.A_hat).all() and (b.A_hat == g.A.astype(np.uint8)).all()
                assert b.summary["unique_tests"] <= a.summary["unique_tests"]


def test_instance_roundtrip_and_oracle_agreement(tmp_path):
    from itpd import instances
    rng = np.random.default_rng(0)
    r = sim.simulate_s2(3, 6, 0, 2, 2, rng, graph="window")
    h = instances.save_instance(str(tmp_path / "i.npz"), r.graph, W=r.W, d=2.0, seed=[0, 1], index=0, kind="window")
    inst = instances.load_instance(str(tmp_path / "i.npz"))
    assert inst["sha1"] == h and (inst["A"] == r.graph.A).all() and inst["W"].shape == r.W.shape
    o1, o2 = ci.DSepCI(r.graph.A), ci.DSepCI(inst["A"])
    q = np.random.default_rng(1)
    for _ in range(100):
        x, y = q.choice(18, 2, replace=False)
        S = [int(v) for v in q.choice(18, 4, replace=False) if v not in (x, y)]
        assert o1(int(x), int(y), S) == o2(int(x), int(y), S)
    X = sim.data_from_instance(inst, 50, np.random.default_rng(2))
    assert X.shape == (50, 6, 3)


def test_metrics_conventions():
    N, T = 2, 3
    A = np.zeros((6, 6), bool)
    A[0, 2] = A[1, 3] = A[2, 4] = A[3, 5] = True          # self edges
    A[0, 3] = True                                         # cross edge
    H = A.copy()
    H[1, 2] = True                                         # extra forward edge
    H[2, 3] = True                                         # same-time edge: FP, non-forward
    m = metrics.edge_metrics(A, H, N, T)
    assert (m["tp"], m["fp"], m["fn"], m["fp_nonforward"], m["shd"]) == (1, 2, 0, 1, 2)
    H2 = H.copy(); H2[0, 2] = False                        # missing self edge: invisible in the headline, visible with include_self
    assert metrics.edge_metrics(A, H2, N, T)["fn"] == 0
    assert metrics.edge_metrics(A, H2, N, T, include_self=True)["fn"] == 1


def test_instance_export_with_weights_keeps_the_graph():
    from itpd.experiments import oracle_counts
    for kind in ("time", "window"):
        g1, W1 = oracle_counts.make_instance(4, 6, 2, 2.0, 0, 3, kind, with_weights=False)
        g2, W2 = oracle_counts.make_instance(4, 6, 2, 2.0, 0, 3, kind, with_weights=True)
        assert W1 is None and W2 is not None
        from itpd.instances import graph_hash
        assert graph_hash(g1.A) == graph_hash(g2.A)


def test_oracle_counts_methods_subset():
    from itpd.experiments import oracle_counts
    res = oracle_counts.run_cell(3, 4, 1, 1.0, 2, 0, None, methods=oracle_counts.select_methods("order,itpd_naive"))
    assert {r["name"] for r in res["rows"]} == {"full_conditioning", "itpd_naive"}
    assert all(r["exact"] for r in res["rows"])


# ---------------------------------------------------------------- adjacency_self

def test_adjacency_self_not_in_default_rules_and_unknown_rule_rejected():
    rng = np.random.default_rng(31)
    g = sim.sample_time_graph(4, 7, 2.0, 2, rng)
    default = I.run_s2(ci.Recorder(ci.DSepCI(g.A)), 4, 7, ALPHA, reuse=True)
    explicit = I.run_s2(ci.Recorder(ci.DSepCI(g.A)), 4, 7, ALPHA, reuse=True, rules=I.RULES)
    assert "adjacency_self" not in I.RULES and I.OPTIONAL_RULES == ("adjacency_self",)
    assert default.summary["unique_tests"] == explicit.summary["unique_tests"]
    assert not any("adjacency_self" in k for k in default.skips)
    with pytest.raises(ValueError):
        I.run_s2(ci.Recorder(ci.DSepCI(g.A)), 4, 7, ALPHA, reuse=True, rules=("adjacency", "nope"))
    # the headline method names do not enable it
    for m in ("itpd", "itpd_naive"):
        o = runlib.run_s2_method(m, None, graph=g)
        assert not any("adjacency_self" in k for k in o["skips"])


def test_adjacency_self_oracle_exact_never_more_tests_and_skips_only_t_ge_2():
    rng = np.random.default_rng(77)
    saved, total = 0, 0
    for i in range(240):
        N, T = int(rng.integers(2, 7)), int(rng.integers(3, 9))
        tau = int(rng.integers(1, 4))
        g = sim.sample_time_graph(N, T, float(rng.choice([1.0, 2.0, 3.0])), tau, rng) if i % 3 else \
            sim.sample_nonstationary_graph(N, T, 2.0, tau, rng)
        tm = None if i % 2 == 0 else tau
        lazy = bool(i % 4 < 2)
        base = runlib.run_s2_method("itpd", None, graph=g, tau_max=tm, lazy=lazy, order="time" if i % 5 == 0 else "variable")
        adj = runlib.run_s2_method("itpd_adjself", None, graph=g, tau_max=tm, lazy=lazy,
                                   order="time" if i % 5 == 0 else "variable")
        assert adj["metrics"]["exact"], (N, T, tau, tm, lazy)
        ub, ua = base["tests"]["unique_tests"], adj["tests"]["unique_tests"]
        assert ua <= ub and adj["tests"]["raw_calls"] <= base["tests"]["raw_calls"]
        saved += ub - ua
        total += ub
        k4 = adj["skips"].get("Z4:adjacency_self", 0)
        k8 = adj["skips"].get("Z8:adjacency_self", 0)
        if T >= 3 and (tm is None or tm >= 2):
            assert k4 + k8 > 0
        if tm == 1:                                    # V_{t-2} is outside the candidate window: the rule never applies
            assert k4 + k8 == 0 and ua == ub
    assert saved > 0 and saved < total


def test_adjacency_self_saves_one_pair_of_tests_per_target_at_most():
    """In S2 with full history, the rule can skip at most the Z4 and Z8 issuance of V^n_{t-2} for pairs with t >= 2."""
    rng = np.random.default_rng(5)
    N, T = 4, 6
    g = sim.sample_time_graph(N, T, 2.0, 2, rng)
    o = runlib.run_s2_method("itpd_adjself", None, graph=g, lazy=False)
    assert o["skips"]["Z4:adjacency_self"] <= N * (T - 2) and o["skips"]["Z8:adjacency_self"] <= N * (T - 2)


def test_adjacency_self_finite_data_runs_and_oracle_counts_selects_it():
    from itpd.experiments import oracle_counts
    rng = np.random.default_rng(3)
    r = sim.simulate_s2(3, 6, 600, 1.5, 2, rng)
    o = runlib.run_s2_method("itpd_adjself", r, ci_kind="fisherz", tau_max=None)
    assert o["status"] == "ok" and "f1" in o["metrics"]
    assert [m[0] for m in oracle_counts.select_methods("itpd,itpd_adjself")] == ["itpd", "itpd_adjacency_self"]
