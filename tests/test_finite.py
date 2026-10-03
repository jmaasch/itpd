"""Shared p-value memo, per-target infeasibility, instances with data seeds, violations, S1 generator."""

import numpy as np
import pytest

from itpd import ci, instances, itpd as I, sim
from itpd import dataset_eval as finite, method_runner as runlib, observed_data as violations, run_single_series as s1x
from itpd.metrics import edge_metrics


def _data(N=4, T=6, M=300, seed=0, kind="window"):
    inst = violations.window_instance(N, T, 1, 2.0, seed, 0, kind)
    inst["data_seed"] = violations.data_seed_of(inst)
    return inst, violations.observed_data(inst, M)


def test_shared_memo_changes_nothing_but_the_work():
    inst, X = _data()
    g = inst["graph"]
    f = ci.FisherZ(X.reshape(X.shape[0], -1))
    shared = {}
    for m in ("itpd_naive", "itpd", "order"):
        a = runlib.run_s2_method(m, None, graph=g, ci=f, order="time", lazy=True, keep_graph=True)
        b = runlib.run_s2_method(m, None, graph=g, ci=f, order="time", lazy=True, keep_graph=True, shared=shared)
        assert (a["A_hat"] == b["A_hat"]).all() and a["tests"]["unique_tests"] == b["tests"]["unique_tests"]
    c = runlib.run_s2_method("itpd", None, graph=g, ci=f, order="time", lazy=True, shared=shared)
    assert c["tests"]["shared_hits"] > 0 and c["tests"]["evaluated"] == 0


def test_per_target_infeasibility_is_exact_for_order_and_never_independent():
    N, T, M = 4, 7, 20                                    # order needs M >= N t + 3: t <= 4 feasible
    inst, X = _data(N, T, M)
    g = inst["graph"]
    f = ci.FisherZ(X.reshape(M, -1))
    assert finite.common_tmax(N, T, M) == 4
    for m in ("order", "itpd", "itpd_naive"):
        o = runlib.run_s2_method(m, None, graph=g, ci=f, order="time", lazy=True, keep_graph=True, per_target=True,
                                 infeasible="raise")
        bad = set(o["infeasible_targets"])
        if m == "order":
            assert bad == {t * N + n for t in range(5, T) for n in range(N)}
            assert o["status"] == "partial" and o["n_infeasible_targets"] == len(bad)
        for y in bad:                                      # no parents are output for an infeasible target
            assert not o["A_hat"][:, y].any()
        assert not any(c < 5 * N for c in bad)             # common targets are feasible for every method
        assert o["metrics"] is not None
    # large M: per_target mode is identical to the plain run
    inst, X = _data(N, T, 2000)
    f = ci.FisherZ(X.reshape(2000, -1))
    a = runlib.run_s2_method("itpd", None, graph=g, ci=f, order="time", keep_graph=True)
    b = runlib.run_s2_method("itpd", None, graph=g, ci=f, order="time", keep_graph=True, per_target=True)
    assert (a["A_hat"] == b["A_hat"]).all() and b["status"] == "ok" and a["tests"]["unique_tests"] == b["tests"]["unique_tests"]


def test_infeasible_target_labels_are_not_propagated():
    N, T, M = 3, 6, 7
    inst, X = _data(N, T, M)
    rec = ci.Recorder(ci.FisherZ(X.reshape(M, -1)), infeasible="raise")
    res = I.run_s2(rec, N, T, 0.01, reuse=True, order="time", per_target=True)
    bad = set(res.infeasible_targets)
    assert bad and all(y not in res.pa for y in bad)
    assert all(p.get("infeasible") for p in res.per_pair if p["pair"][1] in bad)


def test_edge_metrics_targets_restriction():
    rng = np.random.default_rng(1)
    g = sim.sample_time_graph(3, 5, 2.0, 2, rng)
    A_hat = np.zeros_like(g.A, dtype=np.uint8)
    full = edge_metrics(g.A, A_hat, 3, 5)
    mask = np.zeros(15, dtype=bool)
    mask[:9] = True
    part = edge_metrics(g.A, A_hat, 3, 5, targets=mask)
    assert part["fn"] <= full["fn"] and part["fp"] == 0 and edge_metrics(g.A, g.A, 3, 5, targets=mask)["exact"]


def test_data_seed_prefix_and_instance_roundtrip(tmp_path):
    inst, X = _data(4, 6, 2000)
    X50 = violations.observed_data(inst, 50)
    assert (X50 == X[:50]).all()
    p = str(tmp_path / "i.npz")
    instances.save_instance(p, inst["graph"], W=inst["W"], d=2.0, index=0, kind="window", seed=[0, 4, 6, 1, 20, 0],
                            extra={"data_seed": inst["data_seed"], "M_max": 2000, "meta": inst["meta"],
                                   "data_sha1": violations.data_hash(X)})
    z = instances.load_instance(p)
    Xz = violations.observed_data(z, 2000)
    assert violations.data_hash(Xz) == z["data_sha1"] == violations.data_hash(X)
    assert (violations.observed_truth(z)["A_fwd"] == inst["A"]).all()


@pytest.mark.parametrize("spec", [{"kind": "hidden", "k": 2}, {"kind": "contemp", "p": 0.2},
                                  {"kind": "self_missing", "k": 2}, {"kind": "self_lag2", "k": 2},
                                  {"kind": "heavy", "noise": "laplace"}, {"kind": "measurement", "r": 0.5}])
def test_violation_instances(spec, tmp_path):
    N, T, tau = 6, 8, 2
    base = violations.build_instance(N, T, tau, 2.0, 0, 3)
    v = violations.build_instance(N, T, tau, 2.0, 0, 3, violation=spec, noise=spec.get("noise", "gauss"))
    v["data_seed"] = violations.data_seed_of(v)
    tr = violations.observed_truth(v)
    X = violations.observed_data(v, 400)
    k = spec["kind"]
    if k == "hidden":
        assert X.shape == (400, T, N - 2) and tr["N_obs"] == N - 2 and tr["A_fwd"].shape == (T * (N - 2),) * 2
        for n in v["meta"]["hidden"]:                      # a dropped series has outgoing edges to other series
            assert base["A"].reshape(T, N, T, N)[:, n, :, :].sum() > 0
    else:
        assert X.shape == (400, T, N)
    A, A0 = v["A"], base["A"]
    if k == "contemp":
        same = (np.arange(T * N)[:, None] // N) == (np.arange(T * N)[None, :] // N)
        L0 = A & same
        assert tr["n_lag0"] == int(L0.sum()) and not L0[:N].any()           # none at t = 0
        ii, jj = np.nonzero(L0)
        assert ((ii % N) < (jj % N)).all()
        assert (A & ~same == A0 & ~same).all()
        assert (tr["A_fwd"] == (A & ~same)).all()
    if k in ("self_missing", "self_lag2"):
        for n in v["meta"]["series"]:
            assert not any(A[t * N + n, (t + 1) * N + n] for t in range(T - 1))
            if k == "self_lag2":
                assert all(A[t * N + n, (t + 2) * N + n] for t in range(T - 2))
    if k not in ("contemp", "self_missing", "self_lag2"):
        assert (A == A0).all()
    assert v["meta"]["guard_ok"]
    if k == "measurement":   # observed variance = latent variance * (1 + r)
        Xl = sim._gen_linear(v["A"], v["W"], 2000, np.random.default_rng(np.random.SeedSequence(v["data_seed"])), "gauss")
        Xo = violations.observed_data(v, 2000).reshape(2000, -1)
        assert abs(Xo.var(0).mean() / Xl.var(0).mean() - 1.5) < 0.1


def test_edge_strengths_match_sample_partial_correlations():
    inst, X = _data(4, 5, 2000)
    Sig = violations.sigma_from_W(inst["W"])
    es = violations.edge_strengths(inst["A"], Sig, 4, 5)
    Xf = X.reshape(2000, -1)
    C = np.corrcoef(Xf.T)
    E = es["edges"]
    assert np.abs(es["marg"] - np.abs(C[E[:, 0], E[:, 1]])).max() < 0.08
    assert (es["cond_pa"] >= 0).all() and es["cond_pa"].min() >= min(sim.faithfulness_margin(inst["A"], Sig), 1) - 1e-9


def test_s1_generator_matches_unrolled_solution_and_window_cov():
    inst = s1x.s1_instance(3, 50, 2, 2.0, 0, 1)
    Wl = inst["Wl"]
    T = 12
    eps = np.random.default_rng(4).standard_normal((T, 3))
    X = s1x.gen_series(Wl, eps)
    Wu = sim._unroll_values(Wl, T)
    ref = np.linalg.solve(np.eye(T * 3) - Wu.T, eps.reshape(-1)).reshape(T, 3)
    assert np.allclose(X, ref)
    inst["T"] = 60000
    S = sim.windows(s1x.s1_data(inst), 2)
    assert np.abs(np.cov(S.T) - s1x.window_cov(Wl)).max() < 0.25
    a = dict(inst, T=200)
    b = dict(inst, T=300)
    assert (s1x.s1_data(a) == s1x.s1_data(b)[:200]).all()


def test_s1_every_slice_equals_last_slice_for_tau1():
    inst = s1x.s1_instance(5, 600, 1, 2.0, 0, 0)
    out = s1x.run_one(inst)
    r = {x["name"]: x for x in out["runs"]}
    assert r["itpd_last_slice"]["unique"] == r["itpd_naive_every_slice"]["unique"] == r["itpd_every_slice"]["unique"]
    assert r["itpd_last_slice"]["metrics"]["f1"] == r["itpd_naive_every_slice"]["metrics_union"]["f1"]


def test_s1_instance_file_regenerates_the_same_series(tmp_path):
    inst = s1x.s1_instance(5, 300, 2, 2.0, 0, 2)
    p = str(tmp_path / "s.npz")
    s1x.save_s1_instance(p, inst)
    z = s1x.load_s1_instance(p)
    assert (s1x.s1_data(z) == s1x.s1_data(inst)).all() and (z["B"] == inst["B"]).all()
