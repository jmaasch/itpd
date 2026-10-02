"""Instance files shared by every runner (ITPD and the baselines): one .npz per random graph.

Keys of the npz (all numpy arrays / scalars):
  A         (T*N, T*N) uint8   unrolled time graph, A[i, j] = 1 iff i -> j, column = t * N + n (time-major), t from 0
  W         (T*N, T*N) float32 (float64 when data_seed is present) linear-Gaussian weights of the SCM (same layout, zero where A = 0); absent for oracle-only
  N, T, tau, d                 sizes and the generator parameters (tau = max lag, d = expected non-self out-degree)
  seed, index                  numpy SeedSequence entropy list and graph index that generated it
  kind                         "time" (per-source edges) | "window" (stationary lag graph) | "nonstationary"
  B         (tau+1, N, N)      window graph, only for kind = "window"
  sha1                         hash of (shape, A bytes); every result row of every method carries it
Finite-data extension (keys absent in the oracle-only instances; `itpd.observed_data`):
  data_seed  int64 list          SeedSequence entropy of the data; the data are the FIRST M rows of one matrix of M_max rows
  M_max      int                 number of rows generated (2000); smaller M are prefixes, so data are paired across M
  meta_json  str (JSON)          {"noise": "gauss"|"laplace"|"student3", "violation": {...}|null, "nonstationary":
                                 {"n_changes", "frac"}|null, "hidden": [series dropped from the observed data],
                                 "meas_r": r (observed = X + e, Var(e) = r Var(X_v)), "series": [affected series],
                                 "change_times": [...], guard values (min_partial_corr, min_marginal_corr, guard_ok)}
  data_sha1  str                 sha1 of the float64 observed data at M = M_max (check after regenerating)
  kind                           also "nonstationary" and "violation" (robustness runs; the instance holds the TRUE graph A with the
                                 modified self edges / same-time edges, W its weights; hidden series are dropped only from the data)
Data and truth of a finite-data instance, bit-identical for every runner:
  inst = instances.load_instance(path); X = observed_data.observed_data(inst, M)  # (M, T, N_obs) observed data
  tr = observed_data.observed_truth(inst)   # tr["A_fwd"] = headline truth (observed nodes, no same-time edges, column
                                         # t * N_obs + n'), tr["A_obs"] incl. same-time edges, tr["n_lag0"], tr["obs"]
S1 instances (`itpd.run_single_series`, kind = "s1"): no unrolled graph (T up to 10,000); keys B (tau+1, N, N), Wl (tau+1, N, N) lag weights,
T, N, tau, d, data_seed, burn = 0, meta_json; data from `run_single_series.s1_data(inst)`; truth = B.
The process starts at t = 0 (those nodes are roots); the self edge V^n_t -> V^n_{t+1} is in A for all n, t.
Oracle: `itpd.ci.DSepCI(A)` (d-separation on the full graph); every method uses it for the oracle tests.
Data for a finite-data run: `itpd.sim.data_from_instance(inst, M, rng)`.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np

from .graphs import TimeGraph


def graph_hash(A: np.ndarray) -> str:
    A = np.ascontiguousarray(np.asarray(A) != 0, dtype=np.uint8)
    return hashlib.sha1(str(A.shape).encode() + A.tobytes()).hexdigest()


def save_instance(path: str, graph: TimeGraph, *, W=None, d: float, seed, index: int, kind: str, extra=None) -> str:
    """`extra`: optional dict of finite-data keys (data_seed, M_max, meta (dict, stored as meta_json), data_sha1)."""
    h = graph_hash(graph.A)
    arrays = dict(A=(graph.A != 0).astype(np.uint8), N=graph.N, T=graph.T, tau=graph.tau, d=d,
                  seed=np.asarray(seed, dtype=np.int64), index=index, kind=kind, sha1=h)
    if W is not None:     # finite-data instances keep float64 weights: the data are regenerated from W (bit-identical)
        arrays["W"] = np.asarray(W, dtype=np.float64 if extra and "data_seed" in extra else np.float32)
    if graph.B is not None:
        arrays["B"] = (graph.B != 0).astype(np.uint8)
    for k, v in (extra or {}).items():
        if k == "meta":
            arrays["meta_json"] = json.dumps(v, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        elif k == "data_seed":
            arrays[k] = np.asarray(v, dtype=np.int64)
        else:
            arrays[k] = v
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + f".tmp{os.getpid()}.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)               # atomic: concurrent writers of the same (deterministic) instance are safe
    return h


def load_instance(path: str) -> dict:
    z = np.load(path, allow_pickle=False)
    out = {k: z[k] for k in z.files}
    for k in ("N", "T", "tau", "index"):
        out[k] = int(out[k])
    out["d"] = float(out["d"])
    out["kind"] = str(out["kind"])
    out["sha1"] = str(out["sha1"])
    out["A"] = out["A"].astype(bool)
    if "meta_json" in out:
        out["meta"] = json.loads(str(out["meta_json"]))
    if "data_seed" in out:
        out["data_seed"] = [int(v) for v in out["data_seed"]]
    if "M_max" in out:
        out["M_max"] = int(out["M_max"])
    if "data_sha1" in out:
        out["data_sha1"] = str(out["data_sha1"])
    out["graph"] = TimeGraph(A=out["A"], N=out["N"], T=out["T"], tau=out["tau"],
                             B=out["B"].astype(bool) if "B" in out else None)
    assert graph_hash(out["A"]) == out["sha1"]
    return out
