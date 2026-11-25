"""Simulators: assumptions, expected degree, data follow the graph, no duplicate or cancelling edges."""
import numpy as np
import pytest

from itpd import sim, graphs


def _ok(g):
    c = graphs.check_assumptions(g)
    assert c["autocorr_present"] and c["no_contemporaneous"] and c["forward_only"] and c["roots_at_t0"]
    return c


@pytest.mark.parametrize("kw", [dict(graph="time"), dict(graph="window"),
                                dict(nonstationary={"n_changes": 2, "frac": 0.4})])
def test_s2_assumptions_and_shapes(kw):
    rng = np.random.default_rng(0)
    r = sim.simulate_s2(4, 8, 30, 2, 3, rng, **kw)
    assert r.X.shape == (30, 8, 4)
    c = _ok(r.graph)
    assert c["max_lag"] <= 3
    assert r.W is not None and (r.W[r.graph.A] != 0).all() and (r.W[~r.graph.A] == 0).all()


def test_expected_out_degree():
    rng = np.random.default_rng(0)
    N, T, tau, d = 6, 12, 2, 2.0
    degs = []
    for _ in range(200):
        g = sim.sample_time_graph(N, T, d, tau, rng)
        A = g.A.copy()
        A &= ~graphs.autocorr_mask(N, T)
        degs.append(A[: (T - tau - 1) * N].sum(1).mean())      # nodes with the full forward horizon
    assert np.mean(degs) == pytest.approx(d, abs=0.1)


def test_data_follow_graph():
    # Earlier bug: with variable-major generation, a parent from a later-processed variable passed its raw noise only.
    # Here the empirical covariance must equal the exact covariance implied by (A, W).
    rng = np.random.default_rng(5)
    for graph in ("time", "window"):
        r = sim.simulate_s2(3, 5, 60000, 3, 2, rng, graph=graph)
        _, Sig = sim.propagate_cov(r.graph.A, r.W)
        X = r.X.reshape(r.X.shape[0], -1)
        emp = X.T @ X / X.shape[0]
        assert np.abs(emp - Sig).max() < 0.08 * max(1.0, Sig.max())


def test_no_cancelling_edges():
    rng = np.random.default_rng(2)
    for _ in range(20):
        r = sim.simulate_s2(3, 6, 10, 3, 2, rng)
        assert r.meta["guard_ok"] and r.meta["min_partial_corr"] >= 0.02
        assert (np.abs(r.W[r.graph.A]) > 0).all()


def test_variance_controlled():
    rng = np.random.default_rng(3)
    r = sim.simulate_s2(5, 20, 4000, 3, 3, rng)
    assert r.X.reshape(4000, -1).var(0).max() < 3.5       # signal variance capped at the noise variance


def test_window_stationary_spectral_radius():
    rng = np.random.default_rng(4)
    for _ in range(20):
        B = sim.sample_window_graph(4, 3, 3, rng)
        Wl = sim.stable_window_weights(B, rng)
        assert sim._spectral_radius(Wl) <= 0.9 + 1e-9


def _lag_block(A, N, t, tau):
    return np.stack([A[(t - l) * N:(t - l + 1) * N, t * N:(t + 1) * N] for l in range(1, tau + 1)])


def test_nonstationary_changes_shared_and_weights_fixed_by_default():
    N, T, tau = 4, 12, 2
    saw_change = False
    for seed in range(5):
        rng = np.random.default_rng(seed)
        r = sim.simulate_s2(N, T, 20, 2, tau, rng, nonstationary={"n_changes": 2, "frac": 0.5})
        ch = r.meta["change_times"]
        assert len(ch) == 2
        A = r.graph.A
        assert A[graphs.autocorr_mask(N, T)].all()                    # self edges never switch
        for t in range(tau + 1, T):
            if t not in ch:
                assert (_lag_block(A, N, t, tau) == _lag_block(A, N, t - 1, tau)).all()   # constant between changes
            else:
                saw_change |= bool((_lag_block(A, N, t, tau) != _lag_block(A, N, t - 1, tau)).any())
        # weights fixed in t: the weight of a slot is the same at every step where the edge exists
        for t in range(tau + 1, T):
            Wt = np.stack([r.W[(t - l) * N:(t - l + 1) * N, t * N:(t + 1) * N] for l in range(1, tau + 1)])
            Wt1 = np.stack([r.W[(t - 1 - l) * N:(t - l) * N, (t - 1) * N:t * N] for l in range(1, tau + 1)])
            both = (Wt != 0) & (Wt1 != 0)
            assert np.allclose(Wt[both], Wt1[both])
    assert saw_change


def test_nonstationary_per_step_weights_drift():
    rng = np.random.default_rng(1)
    r = sim.simulate_s2(4, 10, 20, 2, 2, rng, nonstationary={"n_changes": 2, "frac": 0.3, "weights": "per_step"})
    w = [r.W[(t - 1) * 4, t * 4] for t in range(1, 10)]
    assert len(set(np.round(w, 8))) > 1


def test_frac_zero_is_the_stationary_control():
    rng = np.random.default_rng(1)
    r = sim.simulate_s2(4, 10, 20, 2, 2, rng, nonstationary={"n_changes": 2, "frac": 0.0})
    N, T, tau = 4, 10, 2
    A = r.graph.A
    for t in range(tau + 1, T):
        assert (_lag_block(A, N, t, tau) == _lag_block(A, N, tau + 1, tau)).all()
    assert r.meta["weights"] == "stationary"


def test_guard_covers_window_graphs_and_logs_margins():
    rng = np.random.default_rng(0)
    for graph in ("window", "time"):
        r = sim.simulate_s2(4, 10, 5, 3, 2, rng, graph=graph)
        assert r.meta["min_partial_corr"] > 0 and r.meta["min_marginal_corr"] > 0
        assert r.meta["guard_ok"]
    r = sim.simulate_s1(4, 30, 3, 2, rng)
    assert "min_marginal_corr" in r.meta


def test_nonlinear_shapes_and_bounded():
    rng = np.random.default_rng(0)
    r = sim.simulate_s2(3, 10, 200, 2, 2, rng, nonlinear=True)
    assert r.X.shape == (200, 10, 3) and np.isfinite(r.X).all() and np.abs(r.X).max() < 40
    r1 = sim.simulate_s1(3, 50, 2, 2, rng, nonlinear=True)
    assert r1.X.shape == (50, 3)


def test_s1_shapes_and_windows():
    rng = np.random.default_rng(0)
    r = sim.simulate_s1(3, 40, 2, 2, rng)
    assert r.X.shape == (40, 3) and r.graph.B.shape == (3, 3, 3)
    Wd = sim.windows(r.X, 2)
    assert Wd.shape == (38, 9)
    assert np.allclose(Wd[5, 2 * 3 + 1], r.X[7, 1])           # column l*N+n = V^n_{s+l}
    assert (r.graph.A == graphs.unroll(r.graph.B, 40)).all()
