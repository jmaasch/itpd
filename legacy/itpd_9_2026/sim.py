"""Simulators satisfying the paper's assumptions: self edge V^n_t -> V^n_{t+1} for every n, t; no contemporaneous
edges; lags 1..tau; the process starts at t = 0 (those nodes are roots, no burn-in); all variables observed.

Expected out-degree d: every potential non-self forward edge (n, t) -> (m, t + l), l = 1..tau, (m, l) != (n, 1), is
present independently with probability p = d / (N * tau - 1). Nodes near the end of the horizon have fewer slots, so
their out-degree is below d (the unrolled graph has fewer slots there); the autocorrelation edge is not counted in d.

Linear-Gaussian data: X_v = sum_u W[u, v] X_u + eps_v, nodes generated in time order. Stability:
- shared weights (window graph, S1): the lag matrices are rescaled so the companion matrix has spectral radius <= rho;
- time-varying weights (S2 time graph, nonstationary variant): incoming weights of each node are rescaled in time order,
  using the exact covariance of the earlier nodes, so that the parent signal variance is at most max_snr * noise variance.
Nonlinear data: additive noise, X_v = sum_u w_uv g_uv(X_u) + eps_v with bounded g (tanh, sin, cos, hump).
Outputs: S2 data (M, T, N); S1 data (T, N). The time graph is returned as TimeGraph (and the window graph for S1).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .graphs import TimeGraph, autocorr_mask, unroll


@dataclass
class SimResult:
    X: np.ndarray | None
    graph: TimeGraph
    W: np.ndarray | None = None
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------- graphs

def _slot_prob(N: int, tau: int, d: float) -> float:
    slots = N * tau - 1
    return 0.0 if slots <= 0 else min(1.0, d / slots)


def sample_window_graph(N: int, tau: int, d: float, rng) -> np.ndarray:
    """Stationary lag graph B[l, i, j] (V^i_{t-l} -> V^j_t), l = 1..tau, with the self edge B[1, n, n] = 1."""
    B = np.zeros((tau + 1, N, N), dtype=bool)
    p = _slot_prob(N, tau, d)
    B[1:] = rng.random((tau, N, N)) < p
    B[1][np.arange(N), np.arange(N)] = True
    return B


def sample_time_graph(N: int, T: int, d: float, tau: int, rng) -> TimeGraph:
    """Graph without stationarity: every source node draws its own targets."""
    A = np.zeros((T * N, T * N), dtype=bool)
    p = _slot_prob(N, tau, d)
    for t in range(T - 1):
        for l in range(1, min(tau, T - 1 - t) + 1):
            blk = rng.random((N, N)) < p
            A[t * N:(t + 1) * N, (t + l) * N:(t + l + 1) * N] = blk
    A |= autocorr_mask(N, T)
    return TimeGraph(A=A, N=N, T=T, tau=tau, meta={"graph": "time"})


def sample_nonstationary_graph(N: int, T: int, d: float, tau: int, rng, n_changes: int = 2, frac: float = 0.3) -> TimeGraph:
    """Base window graph; at `n_changes` change times (shared by all trajectories) a fraction `frac` of the
    non-self edges is switched off and as many absent slots are switched on. Self edges never change."""
    B = sample_window_graph(N, tau, d, rng)
    cross = np.ones((tau + 1, N, N), dtype=bool)
    cross[0] = False
    cross[1][np.arange(N), np.arange(N)] = False
    n_changes = min(n_changes, max(T - 2, 0))
    changes = sorted(rng.choice(np.arange(1, T), size=n_changes, replace=False).tolist()) if n_changes else []
    segs = [B.copy()]
    for _ in changes:
        cur = segs[-1].copy()
        on = np.argwhere(cur & cross)
        off = np.argwhere(~cur & cross)
        m = max(1, int(round(frac * len(on)))) if (len(on) and frac > 0) else 0
        m_off = min(m, len(on))
        if m_off:
            for l, i, j in on[rng.choice(len(on), size=m_off, replace=False)]:
                cur[l, i, j] = False
        m_on = min(m_off, len(off))
        if m_on:
            for l, i, j in off[rng.choice(len(off), size=m_on, replace=False)]:
                cur[l, i, j] = True
        segs.append(cur)
    A = np.zeros((T * N, T * N), dtype=bool)
    for t in range(1, T):
        k = sum(c <= t for c in changes)
        Bt = segs[k]
        for l in range(1, min(tau, t) + 1):
            A[(t - l) * N:(t - l + 1) * N, t * N:(t + 1) * N] = Bt[l]
    B_union = np.logical_or.reduce(segs)
    return TimeGraph(A=A, N=N, T=T, tau=tau, B=None,
                     meta={"graph": "nonstationary", "change_times": changes, "n_segments": len(segs),
                           "B_union": B_union, "frac": frac})


# ---------------------------------------------------------------- weights and variance control

def _draw_weights(shape, rng, wmin, wmax):
    return rng.uniform(wmin, wmax, size=shape) * rng.choice([-1.0, 1.0], size=shape)


def _spectral_radius(Wl: np.ndarray) -> float:
    tau, N = Wl.shape[0] - 1, Wl.shape[1]
    C = np.zeros((N * tau, N * tau))
    for l in range(1, tau + 1):
        C[:N, (l - 1) * N:l * N] = Wl[l].T
    if tau > 1:
        C[N:, :-N] = np.eye(N * (tau - 1))
    return float(np.max(np.abs(np.linalg.eigvals(C))))


def stable_window_weights(B: np.ndarray, rng, wmin=0.3, wmax=0.8, rho=0.9) -> np.ndarray:
    Wl = B * _draw_weights(B.shape, rng, wmin, wmax)
    Wl[0] = 0
    r = _spectral_radius(Wl) if B.shape[0] > 1 else 0.0
    if r > rho:
        c = rho / r
        for l in range(1, Wl.shape[0]):
            Wl[l] *= c ** l
    return Wl


def _unroll_values(Wl, T):
    tau, N = Wl.shape[0] - 1, Wl.shape[1]
    W = np.zeros((T * N, T * N))
    for t in range(1, T):
        for l in range(1, min(tau, t) + 1):
            W[(t - l) * N:(t - l + 1) * N, t * N:(t + 1) * N] = Wl[l]
    return W


def propagate_cov(A: np.ndarray, W: np.ndarray, sigma2: float = 1.0, max_snr: float | None = None):
    """Exact covariance of the linear SCM, nodes in index (= time) order. If max_snr is given, the incoming weights of
    each node are scaled down so that the parent signal variance is at most max_snr * sigma2. Returns (W, Sigma)."""
    n = A.shape[0]
    W = W.copy()
    Sig = np.zeros((n, n))
    for v in range(n):
        pa = np.nonzero(A[:, v])[0]
        if len(pa) == 0:
            Sig[v, v] = sigma2
            continue
        w = W[pa, v]
        sig = float(w @ Sig[np.ix_(pa, pa)] @ w)
        if max_snr is not None and sig > max_snr * sigma2:
            w = w * np.sqrt(max_snr * sigma2 / sig)
            W[pa, v] = w
            sig = max_snr * sigma2
        row = w @ Sig[pa, :v]
        Sig[v, :v] = row
        Sig[:v, v] = row
        Sig[v, v] = sig + sigma2
    return W, Sig


def faithfulness_margin(A: np.ndarray, Sig: np.ndarray) -> float:
    """Smallest |partial correlation| of a true edge u -> v given the other parents of v (exact, from Sigma)."""
    best = np.inf
    for v in range(A.shape[0]):
        pa = np.nonzero(A[:, v])[0]
        if len(pa) == 0:
            continue
        idx = list(pa) + [v]
        P = np.linalg.inv(Sig[np.ix_(idx, idx)])
        k = len(pa)
        pc = np.abs(-P[:k, k] / np.sqrt(np.abs(P[:k, :k].diagonal() * P[k, k])))
        best = min(best, float(pc.min()))
    return best


def marginal_margin(A: np.ndarray, Sig: np.ndarray) -> float:
    """Smallest marginal |correlation| of a true edge (what the Z8 / Z5,7 steps of PaDL look at)."""
    ii, jj = np.nonzero(A)
    if len(ii) == 0:
        return np.inf
    sd = np.sqrt(np.diag(Sig))
    return float(np.min(np.abs(Sig[ii, jj]) / (sd[ii] * sd[jj])))


# ---------------------------------------------------------------- data

def _noise(rng, shape, kind: str):
    if kind == "gauss":
        return rng.standard_normal(shape)
    if kind == "laplace":
        return rng.laplace(size=shape) / np.sqrt(2.0)
    if kind == "student3":
        return rng.standard_t(3, size=shape) / np.sqrt(3.0)
    raise ValueError(kind)


def _gen_linear(A, W, M, rng, noise):
    n = A.shape[0]
    X = np.empty((M, n))
    for v in range(n):
        x = _noise(rng, M, noise)
        pa = np.nonzero(A[:, v])[0]
        if len(pa):
            x = x + X[:, pa] @ W[pa, v]
        X[:, v] = x
    return X


_FNS = (np.tanh, np.sin, np.cos, lambda x: x * np.exp(0.5 - 0.5 * x * x))


def _gen_nonlinear(A, W, M, rng, noise, noise_sd, fn_ids):
    n = A.shape[0]
    X = np.empty((M, n))
    for v in range(n):
        x = noise_sd * _noise(rng, M, noise)
        for u in np.nonzero(A[:, v])[0]:
            x = x + W[u, v] * _FNS[fn_ids[u, v]](X[:, u])
        X[:, v] = x
    return X


def _simulate(graph: TimeGraph, M: int, rng, *, nonlinear=False, noise="gauss", max_snr=1.0, weights="stationary",
              wmin=0.3, wmax=0.8, min_strength=0.02, max_tries=25, noise_sd=1.0) -> SimResult:
    """Weights, then data (M = 0: weights and meta only, X = None, for instance export).

    weights="stationary": one lag-weight array shared by all time steps (spectral radius <= 0.9), used when the
    graph has a window graph (`graph.B`) or a union of window graphs (nonstationary structure); weights="per_step":
    every edge gets its own weight (drift), variance capped node by node (max_snr). A faithfulness guard resamples
    the weights (same graph) until both the smallest partial correlation of a true edge given the other parents and
    its smallest marginal correlation are >= min_strength (window graphs included); `meta["guard_ok"]` says if it held."""
    A, N, T = graph.A, graph.N, graph.T
    meta = dict(graph.meta)
    Bw = graph.B if graph.B is not None else (meta.get("B_union") if weights == "stationary" else None)
    if nonlinear:
        W = _draw_weights(A.shape, rng, 0.6, 1.6) * A
        fn_ids = rng.integers(0, len(_FNS), size=A.shape)
        meta.update(kind="nonlinear", noise=noise, weights="per_edge")
        X = None if M == 0 else _gen_nonlinear(A, W, M, rng, noise, noise_sd, fn_ids).reshape(M, T, N)
        return SimResult(X=X, graph=graph, W=W, meta=meta)
    for tries in range(1, max_tries + 1):
        if Bw is not None:
            Wl = stable_window_weights(Bw, rng, wmin, wmax)
            W, Sig = propagate_cov(A, _unroll_values(Wl, T) * A)
        else:
            W, Sig = propagate_cov(A, _draw_weights(A.shape, rng, wmin, wmax) * A, max_snr=max_snr)
        pm, mm = faithfulness_margin(A, Sig), marginal_margin(A, Sig)
        if min(pm, mm) >= min_strength:
            break
    meta.update(kind="linear", noise=noise, weights="stationary" if Bw is not None else "per_step",
                min_partial_corr=pm, min_marginal_corr=mm, guard_tries=tries,
                guard_ok=bool(min(pm, mm) >= min_strength))
    X = None if M == 0 else _gen_linear(A, W, M, rng, noise).reshape(M, T, N)
    return SimResult(X=X, graph=graph, W=W, meta=meta)


def simulate_s2(N: int, T: int, M: int, d: float, tau: int, rng, *, graph: str = "time", nonlinear: bool = False,
                noise: str = "gauss", nonstationary: dict | None = None, **kw) -> SimResult:
    """S2: M independent trajectories of one system, data (M, T, N), full time graph as truth (M = 0: no data).

    graph="time": edges drawn per source node (no stationarity, per-edge weights); "window": one stationary lag graph
    with weights fixed in t. nonstationary={"n_changes": k, "frac": f, "weights": "stationary"|"per_step"}: edges
    switch on/off at k change times shared by all trajectories; weights="stationary" (default) keeps the weights
    fixed in t, so frac = 0 (or k = 0) is the stationary control; "per_step" redraws every edge weight per step.
    """
    if nonstationary is not None:
        ns = dict(nonstationary)
        w = ns.pop("weights", "stationary")
        g = sample_nonstationary_graph(N, T, d, tau, rng, **ns)
        return _simulate(g, M, rng, nonlinear=nonlinear, noise=noise, weights=w, **kw)
    if graph == "time":
        return _simulate(sample_time_graph(N, T, d, tau, rng), M, rng, nonlinear=nonlinear, noise=noise,
                         weights="per_step", **kw)
    if graph == "window":
        B = sample_window_graph(N, tau, d, rng)
        g = TimeGraph(A=unroll(B, T), N=N, T=T, tau=tau, B=B, meta={"graph": "window"})
        return _simulate(g, M, rng, nonlinear=nonlinear, noise=noise, weights="stationary", **kw)
    raise ValueError(graph)


def simulate_s1(N: int, T: int, d: float, tau: int, rng, *, nonlinear: bool = False, noise: str = "gauss", **kw) -> SimResult:
    """S1: one trajectory of a stationary system, data (T, N); truth: window graph B and its unrolled time graph.
    The process starts at t = 0 with roots (no burn-in); the S1 run scores the last window slice only (the targets
    at t >= tau, whose parents are all observed)."""
    B = sample_window_graph(N, tau, d, rng)
    g = TimeGraph(A=unroll(B, T), N=N, T=T, tau=tau, B=B, meta={"graph": "window"})
    r = _simulate(g, 1, rng, nonlinear=nonlinear, noise=noise, weights="stationary", **kw)
    r.X = r.X[0]
    return r


def windows(X: np.ndarray, tau: int) -> np.ndarray:
    """S1 data (T, N) -> overlapping windows (T - tau, (tau + 1) * N); column l * N + n = V^n_{s + l}."""
    T, N = X.shape
    return np.stack([X[s:s + tau + 1].reshape(-1) for s in range(T - tau)])


def data_from_instance(inst: dict, M: int, rng, noise: str = "gauss") -> np.ndarray:
    """Linear-Gaussian data (M, T, N) from a saved instance (`itpd.instances`): uses the stored weights W."""
    A, W = inst["A"], inst["W"].astype(float)
    return _gen_linear(A, W, M, rng, noise).reshape(M, inst["T"], inst["N"])
