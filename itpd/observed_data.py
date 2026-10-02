"""The observed data and truth of a finite-data instance: data seed, optional assumption violation and nonstationary replicates.
(instances.py = the stored graph instances: structure, weights, seeds; this module builds the data and the truth from them.)

Everything a runner needs to rebuild the observed data of one instance is in the instance file (format: instances.py):
`observed_data(inst, M)` returns the (M, T, N_obs) array and `observed_truth(inst)` the matching truth, so every runner
that reads the same instance gets bit-identical data (check `data_sha1`).

Data convention: the data of an instance are the first M rows of one matrix of `M_max` rows drawn with
`default_rng(SeedSequence(data_seed))`; smaller M are prefixes (paired across M).

How each violation enters the simulator (base = stationary window graph, lag-weights fixed in t, spectral radius <= 0.9,
the faithfulness guard of sim.py on every true edge; the baseline setting "none" is the same call without a violation;
the base graph and base weights of a graph index are identical across violation settings):
  hidden k        k series are removed from the observed data (all T copies). The k series are drawn at random among the
                  series with at least one outgoing non-self edge (a series without effects on others is neither a
                  confounder nor a mediator). Truth = the induced subgraph of the true time graph on the observed
                  nodes: edges that run through a hidden node are NOT in the truth, so methods that report them score
                  false positives.
  contemp p       every same-time pair (series n < m, so the process stays acyclic in column order) gets the edge
                  V^n_t -> V^m_t with probability p for t >= 1 (V_0 stay independent roots), with its own weight.
                  Truth for the headline score = lagged edges only (no tested method can report a same-time edge);
                  the number of true same-time edges is recorded (`n_lag0_true`) as unrecoverable false negatives.
                  The incoming weights are rescaled (propagate_cov max_snr = 4) to keep the variance bounded.
  self_missing k  k random series lose their lag-1 self edge for every t (V^n_{t-1} -> V^n_t absent). ITPD and all
                  tested methods still assume it (they output it and never test it); scored with include_self = True
                  beside the headline (the false self edge is a false positive there).
  self_lag2 k     as self_missing, and the series gets the lag-2 self edge V^n_{t-2} -> V^n_t instead (needs tau >= 2).
  heavy           the same SCM with Laplace noise (`laplace`, unit variance) or Student-t with 3 dof (`student3`,
                  unit variance, infinite fourth moment); Fisher-z is applied unchanged.
  measurement r   the observed data are X + e with e independent Gaussian, variance r * Var(X_v) per node (Var from the
                  exact covariance), truth unchanged (X = latent, Y = observed).
Nonstationary replicates: `sim.sample_nonstationary_graph` with n_changes change times shared by all replicates and
a fraction `frac` of the non-self edges switched off (and as many absent slots switched on) at each change; the lag
weights are fixed in t (stationary option), so frac = 0 is the stationary control; the base graph, the change times and
the base weights of a graph index are shared across frac values (the later random draws differ).
Frozen strings: the violation kinds above ("hidden", "contemp", "self_missing", "self_lag2", "heavy", "measurement", and "none")
enter the data seed through `_tag` (crc32); renaming one would change the data of every stored instance, so none is renamed.
"""
from __future__ import annotations

import hashlib
import json
import zlib

import numpy as np

from . import sim
from .graphs import autocorr_mask, unroll

M_MAX = 2000


# ---------------------------------------------------------------- population quantities

def sigma_from_W(W: np.ndarray) -> np.ndarray:
    """Exact covariance of X = W^T X + eps with unit-variance noise (nodes in topological = column order)."""
    n = W.shape[0]
    Mi = np.linalg.inv(np.eye(n) - np.asarray(W, float).T)
    return Mi @ Mi.T


def true_edge_list(A: np.ndarray, N: int, T: int) -> np.ndarray:
    """(E, 2) array of the non-self true edges (u, v) in a fixed order (row-major), self edges excluded."""
    return np.argwhere((np.asarray(A) != 0) & ~autocorr_mask(N, T))


def edge_strengths(A: np.ndarray, Sig: np.ndarray, N: int, T: int) -> dict:
    """Population strength of every non-self true edge u -> v, in `true_edge_list` order:
    marg = |corr(u, v)|; cond_x = |partial corr(u, v | autocorrelation parent of v)|; cond_pa = |partial corr(u, v |
    all other parents of v)| (the quantity the faithfulness guard bounds). Used to stratify false negatives."""
    A = np.asarray(A) != 0
    E = true_edge_list(A, N, T)
    sd = np.sqrt(np.diag(Sig))
    marg = np.abs(Sig[E[:, 0], E[:, 1]]) / (sd[E[:, 0]] * sd[E[:, 1]])
    cond_x = np.empty(len(E))
    cond_pa = np.empty(len(E))
    pars = {}
    for k, (u, v) in enumerate(E):
        if v not in pars:
            pa = np.nonzero(A[:, v])[0]
            idx = list(pa) + [v]
            P = np.linalg.inv(Sig[np.ix_(idx, idx)])
            pars[v] = (list(pa), P)
        pa, P = pars[v]
        i = pa.index(u)
        cond_pa[k] = abs(-P[i, -1] / np.sqrt(abs(P[i, i] * P[-1, -1])))
        x = v - N
        if x >= 0 and A[x, v] and x != u:
            idx = [u, v, x]
            Pm = np.linalg.inv(Sig[np.ix_(idx, idx)])
            cond_x[k] = abs(-Pm[0, 1] / np.sqrt(abs(Pm[0, 0] * Pm[1, 1])))
        else:
            cond_x[k] = marg[k]
    return {"edges": E, "marg": marg, "cond_x": cond_x, "cond_pa": cond_pa}


def _linear_weights(A, Bw, T, rng, *, lag0=None, max_snr=None, min_strength=0.02, tries=25):
    """Stationary lag weights on the lag graph Bw (+ optional same-time weights), covariance, faithfulness guard on every
    true edge of A (partial correlation given the other parents and marginal correlation >= min_strength)."""
    for k in range(1, tries + 1):
        Wl = sim.stable_window_weights(Bw, rng)
        Wu = sim._unroll_values(Wl, T)
        if lag0 is not None:
            Wu = Wu + sim._draw_weights(A.shape, rng, 0.3, 0.8) * lag0
        W, Sig = sim.propagate_cov(A, Wu * A, max_snr=max_snr)
        pm, mm = sim.faithfulness_margin(A, Sig), sim.marginal_margin(A, Sig)
        if min(pm, mm) >= min_strength:
            break
    return W, Sig, {"min_partial_corr": float(pm), "min_marginal_corr": float(mm), "guard_tries": k,
                    "guard_ok": bool(min(pm, mm) >= min_strength)}


def _sub(*parts) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(p) for p in parts]))


def _tag(s: str) -> int:
    return zlib.crc32(s.encode()) & 0x7FFFFFFF


# ---------------------------------------------------------------- builders

def build_instance(N: int, T: int, tau: int, d: float, seed: int, g: int, *, violation: dict | None = None,
                   nonstationary: dict | None = None, noise: str = "gauss") -> dict:
    """Window-graph instance (base of the violation settings) or nonstationary instance. Returns the arrays and meta of the instance.

    violation: None | {"kind": "hidden", "k": k} | {"kind": "contemp", "p": p} | {"kind": "self_missing", "k": k} |
               {"kind": "self_lag2", "k": k} | {"kind": "heavy", "noise": "laplace"|"student3"} |
               {"kind": "measurement", "r": r}
    nonstationary: {"n_changes": 2, "frac": f}
    """
    base = [seed, N, T, tau, int(round(d * 10)), g]
    rng_g = _sub(*base)
    meta = {"violation": violation, "nonstationary": nonstationary, "noise": noise}
    vk = violation["kind"] if violation else "none"
    rng_w = _sub(*base, 17)
    lag0 = None
    max_snr = None
    hidden: list = []
    if nonstationary is not None:
        gr = sim.sample_nonstationary_graph(N, T, d, tau, rng_g, **nonstationary)
        A, Bw = gr.A.copy(), gr.meta["B_union"]
        meta.update(change_times=gr.meta["change_times"], frac=nonstationary.get("frac"))
        kind = "nonstationary"
    else:
        B = sim.sample_window_graph(N, tau, d, rng_g)
        rng_v = _sub(*base, 31, _tag(vk), int(round(1000 * (violation.get("p", 0) + violation.get("r", 0)))) if violation else 0,
                     violation.get("k", 0) if violation else 0)
        if vk in ("self_missing", "self_lag2"):
            k = min(int(violation["k"]), N)
            ser = sorted(rng_v.choice(N, size=k, replace=False).tolist())
            for n in ser:
                B[1, n, n] = False
                if vk == "self_lag2":
                    if tau < 2:
                        raise ValueError("self_lag2 needs tau >= 2")
                    B[2, n, n] = True
            meta["series"] = ser
        A = unroll(B, T)
        if vk == "contemp":
            p = float(violation["p"])
            lag0 = np.zeros_like(A)
            for t in range(1, T):
                blk = np.triu(rng_v.random((N, N)) < p, k=1)
                lag0[t * N:(t + 1) * N, t * N:(t + 1) * N] = blk
            A = A | lag0
            max_snr = 4.0
        if vk == "hidden":
            cross = B[1:].copy()
            cross[:, np.arange(N), np.arange(N)] = False
            has_out = np.nonzero(cross.any(axis=(0, 2)))[0]
            k = min(int(violation["k"]), len(has_out))
            hidden = sorted(rng_v.choice(has_out, size=k, replace=False).tolist()) if k else []
        Bw = B
        kind = "window"
    W, Sig, guard = _linear_weights(A, Bw, T, rng_w, lag0=lag0, max_snr=max_snr)
    meta.update(guard, hidden=hidden)
    if vk == "heavy":
        meta["noise"] = violation["noise"]
    if vk == "measurement":
        meta["meas_r"] = float(violation["r"])
    return {"A": A, "W": W, "N": N, "T": T, "tau": tau, "d": d, "seed": seed, "g": g, "kind": kind, "meta": meta,
            "Sig": Sig}


def window_instance(N, T, tau, d, seed, g, arm: str) -> dict:
    """Finite-data instance: the same graph and weights as `run_oracle_counts --with-weights` (so counts pair with the oracle runs)."""
    from .run_oracle_counts import make_instance
    gr, W = make_instance(N, T, tau, d, seed, g, arm, with_weights=True)
    return {"A": gr.A, "W": W, "N": N, "T": T, "tau": tau, "d": d, "seed": seed, "g": g, "kind": arm,
            "meta": {"violation": None, "nonstationary": None, "noise": "gauss", "hidden": []}, "graph": gr}


# ---------------------------------------------------------------- data and truth from an instance

def data_seed_of(inst: dict) -> list:
    seed = int(np.asarray(inst["seed"]).ravel()[0])
    g = int(inst["g"]) if "g" in inst else int(inst["index"])
    return [seed, inst["N"], inst["T"], inst["tau"], int(round(inst["d"] * 10)), g, 90210]


def observed_data(inst: dict, M: int) -> np.ndarray:
    """(M, T, N_obs) observed data of an instance dict (loaded with instances.load_instance or built here)."""
    A, W, N, T = inst["A"], inst["W"].astype(float), inst["N"], inst["T"]
    meta = inst["meta"] if isinstance(inst["meta"], dict) else json.loads(inst["meta"])
    ds = list(inst["data_seed"]) if "data_seed" in inst else data_seed_of(inst)
    Mmax = int(inst.get("M_max", M_MAX))
    X = sim._gen_linear(A, W, Mmax, np.random.default_rng(np.random.SeedSequence(ds)), meta.get("noise", "gauss"))
    if "meas_r" in meta:
        sd = np.sqrt(np.diag(sigma_from_W(W))) * np.sqrt(meta["meas_r"])
        X = X + sd * np.random.default_rng(np.random.SeedSequence(ds + [1])).standard_normal(X.shape)
    X = X[:M].reshape(M, T, N)
    hid = meta.get("hidden") or []
    if hid:
        X = X[:, :, [n for n in range(N) if n not in hid]]
    return np.ascontiguousarray(X)


def data_hash(X: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(X, dtype=np.float64).tobytes()).hexdigest()


def observed_truth(inst: dict) -> dict:
    """Truth for scoring in the observed-node indexing (column t * N_obs + n'):
    A_obs = induced true time graph on the observed nodes (all edges incl. same-time ones),
    A_fwd = A_obs without same-time edges (headline truth), n_lag0 = number of true same-time edges among observed."""
    A, N, T = np.asarray(inst["A"]) != 0, inst["N"], inst["T"]
    meta = inst["meta"] if isinstance(inst["meta"], dict) else json.loads(inst["meta"])
    hid = meta.get("hidden") or []
    obs = [n for n in range(N) if n not in hid]
    cols = [t * N + n for t in range(T) for n in obs]
    Ao = A[np.ix_(cols, cols)]
    No = len(obs)
    ti = np.arange(T * No) // No
    same = ti[:, None] == ti[None, :]
    return {"A_obs": Ao, "A_fwd": Ao & ~same, "n_lag0": int((Ao & same).sum()), "N_obs": No, "obs": obs}
