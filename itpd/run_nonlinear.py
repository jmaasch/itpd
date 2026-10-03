"""Nonlinear driver: nonlinear additive-noise S2 data, gradient-boosting GCM test, one JSON per (cell, graph, M), resumable.

    python -m itpd.run_nonlinear --out-dir DIR --N 10 --T 8 --M 500,2000 --graphs 20 --workers 16 [--alphas 0.01,...]
        [--tau 1 --d 2 --seed 0 --offset 0 --budget-sec 780 --small]

Cell = (N, T, tau, d); the graph of index g is the stationary window graph of the window arm of the oracle and finite-data runs of the same cell (same
sha1), the data are nonlinear (`itpd.nonlinear`: per-edge tanh / sine mixtures, noise sd 1, first M of 2,000 rows).
Test: `itpd.ci.GCM` (generalized covariance measure, HistGradientBoostingRegressor, 2-fold cross-fitting); one GCM object and one
p-value memo per dataset, shared by the methods and the alpha levels (a distinct test is computed once). Methods:
itpd_naive, itpd (paper variant), full_conditioning, lazy headline (`specs_for`); full history, process order = time. The GCM has no
hard feasibility rule (only n < 8), so every target is a common target.
Resume: a task file is never recomputed; the p-value memo of an unfinished task is checkpointed after every (method, alpha) run
(DIR/memo/<task>.pkl, written every 90 s and after every run), so a job killed at the wall-clock limit resumes with the tests it already computed.
Files: DIR/<cell>/g<idx>_M<M>.json; instances: DIR/instances/<cell>/g<idx>.npz (nonlinear.save / load_nonlinear_instance).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import pickle
import time


from . import dataset_eval, nonlinear, observed_data
from .ci import GCM
from .instances import graph_hash
from .methods_registry import spec

NONLINEAR_ALPHAS = (0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 1e-4, 1e-6)
PRIMARY = dataset_eval.PRIMARY


def specs_for(alphas):
    return (spec("itpd_naive", alphas), spec("itpd", alphas), spec("full_conditioning", alphas))


def cell_name(N, T, tau, d):
    return f"nonlinear_N{N}_T{T}_tau{tau}_d{d:g}"


def task_path(out, cell, g, M):
    return os.path.join(out, cell, f"g{g:02d}_M{M}.json")


def memo_path(out, cell, g, M):
    return os.path.join(out, "memo", f"{cell}_g{g:02d}_M{M}.pkl")


class CkptDict(dict):
    """p-value memo that writes itself to disk every `every` seconds (a task killed at the wall-clock limit loses at most that)."""

    def __init__(self, path, every=90.0, init=None):
        super().__init__(init or {})
        self.path, self.every, self._t = path, every, time.time()

    def __setitem__(self, k, v):
        super().__setitem__(k, v)
        if time.time() - self._t > self.every:
            self.save()

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + f".tmp{os.getpid()}"
        with open(tmp, "wb") as f:
            pickle.dump(dict(self), f)
        os.replace(tmp, self.path)
        self._t = time.time()


def run_task(a):
    out, N, T, tau, d, seed, g, M, alphas = a
    cell = cell_name(N, T, tau, d)
    path = task_path(out, cell, g, M)
    if os.path.exists(path):
        return path, "exists", 0.0
    t0 = time.time()
    inst = nonlinear.build_instance(N, T, tau, d, seed, g)
    ip = os.path.join(out, "instances", cell, f"g{g:02d}.npz")
    if not os.path.exists(ip):
        nonlinear.save(ip, inst)
    X = nonlinear.data(inst, M)
    edges = observed_data.true_edge_list(inst["A"], N, T)
    mp = memo_path(out, cell, g, M)
    init = {}
    if os.path.exists(mp):
        with open(mp, "rb") as f:
            init = pickle.load(f)
    shared = CkptDict(mp, init=init)
    resumed = len(shared)
    gcm = GCM(X.reshape(M, T * N))

    def ckpt(memo):
        memo.save()

    res = dataset_eval.run_dataset(inst["A"], X, tau, specs_for(alphas), order="time", edges=edges, ci=gcm, shared=shared, on_run=ckpt)
    res.update({"cell": cell, "arm": "window", "dgp": "nonlinear", "ci": "gcm_hgb", "N": N, "T": T, "tau": tau, "d": d, "seed": seed,
                "graph": g, "sha1": graph_hash(inst["A"]), "data_seed": inst["data_seed"], "M_max": nonlinear.M_MAX,
                "n_true_edges_nonself": int(len(edges)), "guard": {k: inst["meta"][k] for k in
                                                                   ("min_partial_corr", "min_marginal_corr", "guard_tries", "guard_ok")},
                "memo_resumed": resumed, "memo_size": len(shared), "gcm_fits": gcm.fits,
                "gcm_calls_by_size": {int(k): v for k, v in sorted(gcm.calls_by_size.items())},
                "gcm_sec_by_size": {int(k): v for k, v in sorted(gcm.sec_by_size.items())},
                "seconds": time.time() - t0})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(res, f, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(tmp, path)
    if os.path.exists(mp):
        os.remove(mp)
    return path, "done", time.time() - t0


def _worker(args):
    a, deadline = args
    if time.time() > deadline:
        return a, "skipped", 0.0
    return run_task(a)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--N", default="10")
    ap.add_argument("--T", default="8")
    ap.add_argument("--M", default="500,2000")
    ap.add_argument("--tau", type=int, default=1)
    ap.add_argument("--d", type=float, default=2.0)
    ap.add_argument("--graphs", type=int, default=20)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alphas", default=",".join(f"{x:g}" for x in NONLINEAR_ALPHAS))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--task", default=None, help="run only the k-th task of the full sorted task list (array jobs)")
    ap.add_argument("--budget-sec", type=float, default=1e9, help="start no new task after this many seconds")
    a = ap.parse_args(argv)
    ints = lambda s: [int(x) for x in s.split(",")]
    alphas = tuple(float(x) for x in a.alphas.split(","))
    tasks = []
    for N, T, M in itertools.product(ints(a.N), ints(a.T), ints(a.M)):
        for g in range(a.offset, a.offset + a.graphs):
            tasks.append((a.out_dir, N, T, a.tau, a.d, a.seed, g, M, alphas))
    done = lambda t: os.path.exists(task_path(t[0], cell_name(*t[1:5]), t[6], t[7]))
    tasks.sort(key=lambda t: (-t[7] * t[1] * t[2], t[6], t[7]))     # big first; fixed order, so --task k is stable
    if a.task is not None:                                           # array job: the k-th task of the full list
        tasks = tasks[int(a.task):int(a.task) + 1]
    todo = [t for t in tasks if not done(t)]
    print(f"tasks {len(tasks)} todo {len(todo)}", flush=True)
    deadline = time.time() + a.budget_sec
    t0 = time.time()
    if a.workers <= 1:
        for t in todo:
            r = _worker((t, deadline))
            print(r[1], os.path.basename(r[0]) if isinstance(r[0], str) else "", round(r[2], 1), flush=True)
    else:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(a.workers) as pool:
            for r in pool.imap_unordered(_worker, [(t, deadline) for t in todo], chunksize=1):
                print(r[1], os.path.basename(r[0]) if isinstance(r[0], str) else "", round(r[2], 1), flush=True)
    left = len([t for t in tasks if not done(t)])
    print(f"done in {time.time() - t0:.0f}s; remaining {left}", flush=True)


if __name__ == "__main__":
    main()
