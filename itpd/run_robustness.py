"""Robustness drivers: nonstationary replicates with shared change times (`--exp nonstationary`) and assumption violations one at a
time (`--exp violations`).

    python -m itpd.run_robustness --out-dir DIR --exp nonstationary|violations [--N 10 --T 12 --tau 2 --d 2 --M 500,2000 --graphs 20 --workers 16]

Nonstationary settings: frac in 0, 0.25, 0.5 (n_changes = 2 shared change times, lag weights fixed in t; frac = 0 is the stationary
control). Violation settings: none, hidden k=1,2,3, contemp p=0.05,0.1,0.2, self_missing k=1,3, self_lag2 k=1,3, heavy laplace /
student3, measurement r=0.1,0.5 (how each enters the simulator: itpd/observed_data.py docstring). Methods: itpd_naive, itpd (paper
variant), order_based, plus the non-lazy rows of the first two (`ROBUSTNESS_SPECS`), all at alpha = 0.01, Fisher-z, full history, linear data, one
shared p-value memo per dataset; scored with `dataset_eval.run_dataset` (own-feasible targets, common targets, self edges
included in `metrics_self`). One file per (setting, graph, M): DIR/<exp>/<setting>/g<idx>_M<M>.json, instances in
DIR/instances/<setting>/g<idx>.npz (format: itpd/instances.py; read them with
observed_data.observed_data / observed_truth). Resumable.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time


from . import dataset_eval, observed_data
from .graphs import TimeGraph
from .instances import graph_hash, save_instance

NONSTATIONARY_SETTINGS = {f"frac{f:g}": {"nonstationary": {"n_changes": 2, "frac": f}} for f in (0.0, 0.25, 0.5)}
VIOLATION_SETTINGS = {
    "none": {"violation": None},
    **{f"hidden_k{k}": {"violation": {"kind": "hidden", "k": k}} for k in (1, 2, 3)},
    **{f"contemp_p{p:g}": {"violation": {"kind": "contemp", "p": p}} for p in (0.05, 0.1, 0.2)},
    **{f"selfmissing_k{k}": {"violation": {"kind": "self_missing", "k": k}} for k in (1, 3)},
    **{f"selflag2_k{k}": {"violation": {"kind": "self_lag2", "k": k}} for k in (1, 3)},
    "heavy_laplace": {"violation": {"kind": "heavy", "noise": "laplace"}, "noise": "laplace"},
    "heavy_student3": {"violation": {"kind": "heavy", "noise": "student3"}, "noise": "student3"},
    **{f"meas_r{r:g}": {"violation": {"kind": "measurement", "r": r}} for r in (0.1, 0.5)},
}
SETTINGS = {"nonstationary": NONSTATIONARY_SETTINGS, "violations": VIOLATION_SETTINGS}
ROBUSTNESS_SPECS = dataset_eval.PRIMARY_ALPHA_SPECS


def build(exp, setting, N, T, tau, d, seed, g):
    spec = SETTINGS[exp][setting]
    inst = observed_data.build_instance(N, T, tau, d, seed, g, violation=spec.get("violation"),
                                     nonstationary=spec.get("nonstationary"), noise=spec.get("noise", "gauss"))
    inst["data_seed"] = observed_data.data_seed_of(inst)
    inst["M_max"] = observed_data.M_MAX
    inst["meta"]["setting"] = setting
    return inst


def write_instance(out, exp, setting, inst, overwrite=False):
    N, T, tau, d, g = inst["N"], inst["T"], inst["tau"], inst["d"], inst["g"]
    ipath = os.path.join(out, "instances", exp, setting, f"g{g:02d}.npz")
    if os.path.exists(ipath) and not overwrite:
        return ipath
    X2 = observed_data.observed_data(inst, observed_data.M_MAX)
    save_instance(ipath, TimeGraph(A=inst["A"], N=N, T=T, tau=tau), W=inst["W"], d=d, index=g, kind=inst["kind"],
                  seed=[inst["seed"], N, T, tau, int(round(d * 10)), g],
                  extra={"data_seed": inst["data_seed"], "M_max": inst["M_max"], "meta": inst["meta"],
                         "data_sha1": observed_data.data_hash(X2)})
    return ipath


def run_task(a):
    out, exp, setting, N, T, tau, d, seed, g, M = a
    path = os.path.join(out, exp, setting, f"g{g:02d}_M{M}.json")
    if os.path.exists(path):
        return path, "exists", 0.0
    t0 = time.time()
    inst = build(exp, setting, N, T, tau, d, seed, g)
    write_instance(out, exp, setting, inst)
    X = observed_data.observed_data(inst, M)
    tr = observed_data.observed_truth(inst)
    No = tr["N_obs"]
    edges = observed_data.true_edge_list(tr["A_fwd"], No, T)
    res = dataset_eval.run_dataset(tr["A_fwd"], X, tau, ROBUSTNESS_SPECS, order="time", edges=edges, score_self=True, n_lag0=tr["n_lag0"])
    meta = inst["meta"]
    res.update({"exp": exp, "setting": setting, "N": N, "N_obs": No, "T": T, "tau": tau, "d": d, "seed": seed, "graph": g,
                "sha1": graph_hash(inst["A"]), "data_seed": inst["data_seed"], "n_lag0_true": tr["n_lag0"],
                "n_true_edges_nonself": int(len(edges)), "hidden": meta.get("hidden"), "series": meta.get("series"),
                "change_times": meta.get("change_times"), "guard_ok": meta.get("guard_ok"),
                "min_partial_corr": meta.get("min_partial_corr"), "seconds": time.time() - t0})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(res, f, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(tmp, path)
    return path, "done", time.time() - t0


def _worker(args):
    a, deadline = args
    return (a, "skipped", 0.0) if time.time() > deadline else run_task(a)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--exp", required=True, choices=["nonstationary", "violations"])
    ap.add_argument("--N", type=int, default=10)
    ap.add_argument("--T", type=int, default=None)
    ap.add_argument("--tau", type=int, default=2)
    ap.add_argument("--d", type=float, default=2.0)
    ap.add_argument("--M", default="500,2000")
    ap.add_argument("--graphs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--budget-sec", type=float, default=1e9)
    ap.add_argument("--instances-only", action="store_true", help="(re)write the instance files only (overwrites)")
    a = ap.parse_args(argv)
    T = a.T or (12 if a.exp == "nonstationary" else 10)
    if a.instances_only:
        for s_, g in itertools.product(SETTINGS[a.exp], range(a.graphs)):
            write_instance(a.out_dir, a.exp, s_, build(a.exp, s_, a.N, T, a.tau, a.d, a.seed, g), True)
        return
    tasks = [(a.out_dir, a.exp, s, a.N, T, a.tau, a.d, a.seed, g, int(M))
             for s, g, M in itertools.product(SETTINGS[a.exp], range(a.graphs), a.M.split(","))]
    exists = lambda t: os.path.exists(os.path.join(t[0], t[1], t[2], f"g{t[8]:02d}_M{t[9]}.json"))
    todo = [t for t in tasks if not exists(t)]
    print(f"tasks {len(tasks)} todo {len(todo)}", flush=True)
    deadline, t0 = time.time() + a.budget_sec, time.time()
    if a.workers <= 1:
        for t in todo:
            _worker((t, deadline))
    else:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(a.workers) as pool:
            for _ in pool.imap_unordered(_worker, [(t, deadline) for t in todo], chunksize=1):
                pass
    print(f"done in {time.time() - t0:.0f}s; remaining {len([t for t in tasks if not exists(t)])}", flush=True)


if __name__ == "__main__":
    main()
