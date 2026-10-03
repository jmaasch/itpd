"""Finite-data driver: Fisher-z runs of the ITPD family and the full-conditioning baseline on S2 instances, one JSON per (cell, graph, M), resumable.

    python -m itpd.run_finite_data --out-dir DIR --arm window --N 10,20 --T 8,16 --M 50,100,200,500,2000 --graphs 20 --workers 16
        [--tau 1 --d 2 --seed 0 --budget-sec 780 --small]

Cell = (arm, N, T, tau, d); arm "window" (stationary lag graph, lag weights fixed in t) or "time"
(per-source edges, per-step weights); instances and weights are the ones of the oracle runs (`run_oracle_counts.make_instance`,
same seed convention), so a finite-data graph has the same sha1 as the oracle-run graph of the same cell and index. Data: the first
M rows of one M_max = 2000 matrix per graph (paired across M and across methods). Linear-Gaussian, Fisher-z, alpha grid
`dataset_eval.ALPHAS` (the headline alpha is 0.01), full history, process order = time. Methods: `dataset_eval.ITPD_AND_ORDER_SPECS`
(lazy headline, non-lazy beside it for alpha = 0.01, ITPD + adjacency_self as an extra); `--small` runs
`dataset_eval.PRIMARY_ALPHA_SPECS`. One memoised Fisher-z per dataset.
Files: DIR/<cell>/g<idx>_M<M>.json; instances: DIR/instances/<cell>/g<idx>.npz (format: itpd/instances.py, with data seed).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time


from . import dataset_eval, observed_data
from .instances import graph_hash, save_instance

M_MAX = observed_data.M_MAX


def cell_name(arm, N, T, tau, d):
    return f"{arm}_N{N}_T{T}_tau{tau}_d{d:g}"


def task_path(out, cell, g, M):
    return os.path.join(out, cell, f"g{g:02d}_M{M}.json")


def instance_path(out, cell, g):
    return os.path.join(out, "instances", cell, f"g{g:02d}.npz")


def make_inst(arm, N, T, tau, d, seed, g):
    inst = observed_data.window_instance(N, T, tau, d, seed, g, arm)
    inst["data_seed"] = observed_data.data_seed_of(inst)
    inst["M_max"] = M_MAX
    return inst


def write_instance(out, cell, inst, overwrite=False):
    path = instance_path(out, cell, g := inst["g"])
    if os.path.exists(path) and not overwrite:
        return path
    X = observed_data.observed_data(inst, M_MAX)
    save_instance(path, inst["graph"], W=inst["W"], d=inst["d"], index=g, kind=inst["kind"],
                  seed=[inst["seed"], inst["N"], inst["T"], inst["tau"], int(inst["d"] * 10), g],
                  extra={"data_seed": inst["data_seed"], "M_max": M_MAX, "meta": inst["meta"],
                         "data_sha1": observed_data.data_hash(X)})
    return path


def run_task(a):
    out, arm, N, T, tau, d, seed, g, M, specs_name = a
    cell = cell_name(arm, N, T, tau, d)
    path = task_path(out, cell, g, M)
    if os.path.exists(path):
        return path, "exists", 0.0
    t0 = time.time()
    inst = make_inst(arm, N, T, tau, d, seed, g)
    write_instance(out, cell, inst)
    X = observed_data.observed_data(inst, M)
    edges = observed_data.true_edge_list(inst["A"], N, T)
    specs = dataset_eval.PRIMARY_ALPHA_SPECS if specs_name == "small" else dataset_eval.ITPD_AND_ORDER_SPECS
    res = dataset_eval.run_dataset(inst["A"], X, tau, specs, order="time", edges=edges)
    res.update({"cell": cell, "arm": arm, "N": N, "T": T, "tau": tau, "d": d, "seed": seed, "graph": g,
                "sha1": graph_hash(inst["A"]), "data_seed": inst["data_seed"], "M_max": M_MAX,
                "n_true_edges_nonself": int(len(edges)), "seconds": time.time() - t0})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(res, f, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(tmp, path)
    return path, "done", time.time() - t0


def est_cost(N, T, M):
    """Rough seconds per task, for ordering only (feasible large cells first)."""
    n = N * T
    feas = min(1.0, max(0.05, (M - N) / max(n - N, 1)))
    return (n / 320.0) ** 2.2 * 400.0 * feas


def _guarded(a, deadline):
    if time.time() > deadline:
        return a, "skipped", 0.0
    return run_task(a)


def _worker(args):
    a, deadline = args
    return _guarded(a, deadline)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--arm", default="window", choices=["window", "time"])
    ap.add_argument("--N", default="10,20")
    ap.add_argument("--T", default="8,16")
    ap.add_argument("--M", default="50,100,200,500,2000")
    ap.add_argument("--tau", type=int, default=1)
    ap.add_argument("--d", type=float, default=2.0)
    ap.add_argument("--graphs", type=int, default=20)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--instances-only", action="store_true", help="(re)write the instance files only (overwrites)")
    ap.add_argument("--small", action="store_true", help="alpha = 0.01 only, methods itpd_naive, itpd, order (+ nl)")
    ap.add_argument("--budget-sec", type=float, default=1e9, help="start no new task after this many seconds")
    a = ap.parse_args(argv)
    ints = lambda s: [int(x) for x in s.split(",")]
    tasks = []
    for N, T, M in itertools.product(ints(a.N), ints(a.T), ints(a.M)):
        for g in range(a.offset, a.offset + a.graphs):
            tasks.append((a.out_dir, a.arm, N, T, a.tau, a.d, a.seed, g, M, "small" if a.small else "full"))
    if a.instances_only:
        for N, T in itertools.product(ints(a.N), ints(a.T)):
            for g in range(a.offset, a.offset + a.graphs):
                write_instance(a.out_dir, cell_name(a.arm, N, T, a.tau, a.d), make_inst(a.arm, N, T, a.tau, a.d, a.seed, g), True)
        return
    todo = [t for t in tasks if not os.path.exists(task_path(t[0], cell_name(*t[1:6]), t[7], t[8]))]
    todo.sort(key=lambda t: -est_cost(t[2], t[3], t[8]))
    print(f"tasks {len(tasks)} todo {len(todo)}", flush=True)
    deadline = time.time() + a.budget_sec
    t0 = time.time()
    if a.workers <= 1:
        for t in todo:
            r = _guarded(t, deadline)
            print(r[1], os.path.basename(r[0]) if isinstance(r[0], str) else "", round(r[2], 1), flush=True)
    else:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(a.workers) as pool:
            for r in pool.imap_unordered(_worker, [(t, deadline) for t in todo], chunksize=1):
                pass
    left = len([t for t in tasks if not os.path.exists(task_path(t[0], cell_name(*t[1:6]), t[7], t[8]))])
    print(f"done in {time.time() - t0:.0f}s; remaining {left}", flush=True)


if __name__ == "__main__":
    main()
