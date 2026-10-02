"""Nonlinear data: additive-noise S2 instances on window (stationary lag) graphs, per-edge random smooth functions.

SCM, nodes in column order (time-major, edges only forward in time):
    X_v = sum_{u in pa(v)} g_uv(X_u) + sigma * eps_v,   eps_v ~ N(0, 1) i.i.d.,   sigma = NOISE_SD
    g_uv(x) = w * ( a * tanh(s * x + c) + (1 - a) * sin(k * x + phi) )
with per-lag-slot random parameters, shared by every time step (the lag graph and its functions are stationary, as the
linear window arm): w = +-U(0.8, 1.6), a ~ U(0.4, 1.0) (tanh part dominant, so the edge has a monotone component and a
nonzero correlation in general), s ~ U(0.7, 2.0), c ~ U(-0.5, 0.5), k ~ U(0.7, 2.0), phi ~ U(0, 2 pi). Every g is bounded
(|g| <= 1.6), so the process cannot blow up; the self edge V^n_{t-1} -> V^n_t is a nonlinear edge of the same family.
The graph and its sha1 are the ones of the window arm of the oracle and finite-data runs of the same cell and index
(`run_oracle_counts.make_instance`), so every nonlinear graph pairs with the linear-Gaussian finite-data run of the same graph.

Faithfulness guard (a proxy, stated as such): on a reference sample of GUARD_N = 20,000 rows, every true edge u -> v must have
|marginal Pearson correlation(X_u, X_v)| >= MIN_STRENGTH and |partial Pearson correlation(X_u, X_v | the other parents of v)|
>= MIN_STRENGTH (0.05; the linear-DGP guard used 0.02). The function parameters are resampled (same graph) up to MAX_TRIES
times; `meta["guard_ok"]` says whether it held. This is a linear-correlation screen: it excludes edges invisible to a
covariance-based test, it does not certify nonlinear faithfulness.

Data convention as `observed_data.py`: the data of an instance are the first M rows of one matrix of M_MAX = 2,000 rows drawn
with default_rng(SeedSequence(data_seed)); smaller M are prefixes (paired across M and across methods).
"""
from __future__ import annotations

import numpy as np

from . import sim
from .instances import load_instance, save_instance
from .observed_data import data_hash

M_MAX = 2000
NOISE_SD = 1.0
GUARD_N = 20000
MIN_STRENGTH = 0.05
MAX_TRIES = 50


def draw_lag_params(B: np.ndarray, rng) -> np.ndarray:
    """(6, tau + 1, N, N) per-lag-slot function parameters (rows: w, a, s, c, k, phi); zero weight where B = 0."""
    sh = B.shape
    P = np.empty((6,) + sh)
    P[0] = rng.uniform(0.8, 1.6, sh) * rng.choice([-1.0, 1.0], sh) * (B != 0)
    P[1] = rng.uniform(0.4, 1.0, sh)
    P[2] = rng.uniform(0.7, 2.0, sh)
    P[3] = rng.uniform(-0.5, 0.5, sh)
    P[4] = rng.uniform(0.7, 2.0, sh)
    P[5] = rng.uniform(0.0, 2 * np.pi, sh)
    P[0][0] = 0.0                                   # no same-time edges
    return P


def unroll_params(P: np.ndarray, T: int) -> np.ndarray:
    """(6, T*N, T*N) parameters of the unrolled time graph (same lag slot -> same function at every t)."""
    return np.stack([sim._unroll_values(P[i], T) for i in range(P.shape[0])])


def generate(A: np.ndarray, PU: np.ndarray, M: int, rng, noise_sd: float = NOISE_SD) -> np.ndarray:
    """(M, T*N) data from the unrolled parameters `PU` (6, n, n); nodes in column (= topological) order."""
    n = A.shape[0]
    w, a, s, c, k, phi = PU
    X = np.empty((M, n))
    for v in range(n):
        x = noise_sd * rng.standard_normal(M)
        for u in np.nonzero(A[:, v])[0]:
            xu = X[:, u]
            x += w[u, v] * (a[u, v] * np.tanh(s[u, v] * xu + c[u, v]) + (1.0 - a[u, v]) * np.sin(k[u, v] * xu + phi[u, v]))
        X[:, v] = x
    return X


def _partial_corr(X: np.ndarray, u: int, v: int, pa: list) -> float:
    others = [p for p in pa if p != u]
    if not others:
        return float(np.corrcoef(X[:, u], X[:, v])[0, 1])
    Z = np.column_stack([np.ones(len(X)), X[:, others]])
    ru = X[:, u] - Z @ np.linalg.lstsq(Z, X[:, u], rcond=None)[0]
    rv = X[:, v] - Z @ np.linalg.lstsq(Z, X[:, v], rcond=None)[0]
    return float(np.corrcoef(ru, rv)[0, 1])


def guard_margins(A: np.ndarray, X: np.ndarray, N: int, T: int) -> tuple[float, float]:
    """(smallest |partial corr| given the other parents, smallest |marginal corr|) over all true edges, self edges included
    (the methods never test those, but their strength shapes the series)."""
    E = np.argwhere(A != 0)
    pm, mm = 1.0, 1.0
    pars = {}
    for u, v in E:
        pa = pars.setdefault(int(v), [int(p) for p in np.nonzero(A[:, v])[0]])
        mm = min(mm, abs(float(np.corrcoef(X[:, u], X[:, v])[0, 1])))
        pm = min(pm, abs(_partial_corr(X, int(u), int(v), pa)))
    return pm, mm


def build_instance(N: int, T: int, tau: int, d: float, seed: int, g: int) -> dict:
    """Window graph of the oracle and finite-data cell and index (same sha1), nonlinear functions, guard, data seed."""
    from .run_oracle_counts import make_instance
    gr, _ = make_instance(N, T, tau, d, seed, g, "window", with_weights=False)
    A = gr.A
    rng = np.random.default_rng(np.random.SeedSequence([seed, N, T, tau, int(round(d * 10)), g, 11]))
    ds = [seed, N, T, tau, int(round(d * 10)), g, 11011]
    for tries in range(1, MAX_TRIES + 1):
        P = draw_lag_params(gr.B, rng)
        PU = unroll_params(P, T)
        Xr = generate(A, PU, GUARD_N, np.random.default_rng(np.random.SeedSequence(ds + [tries])))
        pm, mm = guard_margins(A, Xr, N, T)
        if min(pm, mm) >= MIN_STRENGTH:
            break
    meta = {"kind": "nonlinear", "noise": "gauss", "noise_sd": NOISE_SD, "min_partial_corr": pm, "min_marginal_corr": mm,
            "guard_tries": tries, "guard_ok": bool(min(pm, mm) >= MIN_STRENGTH), "guard_n": GUARD_N,
            "min_strength": MIN_STRENGTH, "violation": None, "nonstationary": None, "hidden": []}
    return {"A": A, "graph": gr, "P": P, "N": N, "T": T, "tau": tau, "d": d, "seed": seed, "g": g, "kind": "window",
            "data_seed": ds, "M_max": M_MAX, "meta": meta}


def data(inst: dict, M: int) -> np.ndarray:
    """(M, T, N) data of a nonlinear instance (built here or loaded with `load_nonlinear_instance`): the first M of M_MAX rows."""
    PU = unroll_params(inst["P"], inst["T"])
    X = generate(inst["A"], PU, int(inst.get("M_max", M_MAX)), np.random.default_rng(np.random.SeedSequence(inst["data_seed"])),
                 inst.get("meta", {}).get("noise_sd", NOISE_SD))
    return np.ascontiguousarray(X[:M].reshape(M, inst["T"], inst["N"]))


def save(path: str, inst: dict) -> str:
    X = data(inst, inst["M_max"])
    return save_instance(path, inst["graph"], W=None, d=inst["d"], index=inst["g"], kind="window",
                         seed=[inst["seed"], inst["N"], inst["T"], inst["tau"], int(round(inst["d"] * 10)), inst["g"]],
                         extra={"data_seed": inst["data_seed"], "M_max": inst["M_max"], "meta": inst["meta"],
                                "data_sha1": data_hash(X), "fn_lag": inst["P"]})


def load_nonlinear_instance(path: str) -> dict:
    """Load a saved nonlinear instance; `data(inst, M)` regenerates the data (check `data_sha1`)."""
    inst = load_instance(path)
    inst["P"] = inst["fn_lag"]
    inst["g"] = inst["index"]
    return inst
