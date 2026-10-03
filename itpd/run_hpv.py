"""HPV drivers: oracle runs on the instances of earlier oracle runs (`oracle`) and finite-data runs on the window instances of earlier
finite-data runs (`finite`). Nothing is regenerated except the data of a finite-data instance, which is checked against its stored
data_sha1. One JSON per task, resumable (an existing file is skipped). The earlier runs are read from the results directory R
(`--results`, default $ITPD_RESULTS or ./results): `python -m itpd.run_oracle_counts ... --out R/oracle/<arm>/N<N>_T<T>_tau<tau>_d2.json
--instances-dir R/instances/<arm> --with-weights` and `python -m itpd.run_finite_data --arm window --out-dir R/finite/window ...`.

    python -m itpd.run_hpv oracle --out-dir OUT/oracle --workers 12 [--arm time,window --N 10,20 --T 8,16 --tau 1,2 --graphs 20]
    python -m itpd.run_hpv finite --out-dir OUT/finite --workers 12 [--N 10,20 --T 8,16 --M 50,100,200,500,2000 --graphs 20]
        [--budget-sec S]   (start no new task after S seconds)
        [--specs hpv_safe,hpv_safe_lenient]   (run only these entries of HPV_FINITE_SPECS; default all; use a new --out-dir)
        [--shard K/NSH]    (this process runs tasks K, K + NSH, ... of the cost-sorted list; one shard per batch job)
        [--summary FILE]   (one-line summary written at the end)

oracle: instances R/instances/<arm>/N<N>_T<T>_tau<tau>_d2_<arm>_g<g>.npz, sha1 checked against the stored oracle-run row of the same
graph (R/oracle/<arm>/N<N>_T<T>_tau<tau>_d2.json, graph_stats.sha1) and Sum|C| against its n_cand. HPV variants (`HPV_ORACLE_VARIANTS`):
hints learned_blanket, oracle_blanket, none, union, learned_blanket + re-check (hpv_safe), and shifted_parents (window arm only); exact
d-separation oracle on the full graph.
finite: instances R/finite/window/instances/window_N<N>_T<T>_tau1_d2/g<g>.npz, sha1 and data_seed checked against the stored finite-data task
JSON, data regenerated and checked against data_sha1 (M = 2000), the first M rows used (paired across M).
Methods `HPV_FINITE_SPECS` through dataset_eval.run_dataset (Fisher-z, one shared p-value memo per dataset, per-target infeasibility,
common targets t <= (M - 3) / N). For the true parents, the population partial correlation given the phase-A set actually
used is computed from the exact covariance (rows "hpv_rhoA": [edge index, |S_A|, |rho|, p-value]).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from functools import lru_cache

import numpy as np

from . import dataset_eval, method_runner, observed_data
from .instances import load_instance
from .methods_registry import OLD_TO_NEW, hpv_options, spec

R_DEFAULT = os.environ.get("ITPD_RESULTS", "results")

ALPHAS = dataset_eval.ALPHAS
ALPHAS_LEN = tuple(a for a in ALPHAS if a <= 0.1 + 1e-12)
HPV_FINITE_SPECS = (                      # built from methods_registry.spec; set by main(--specs); forked workers inherit it
    spec("hpv_single_pass", ALPHAS, keep_parent_tests=True),
    spec("hpv_single_pass_lenient", ALPHAS_LEN, keep_parent_tests=True),
    spec("hpv_single_pass_oracle_blanket", (dataset_eval.PRIMARY,), keep_parent_tests=True),
    spec("hpv_single_pass_oracle_blanket_lenient", (dataset_eval.PRIMARY,), keep_parent_tests=True),
    # the re-check (always-verify, HPV-safe) at equal alpha with the full 13-point sweep (matched-false-positive curves); the lenient
    # row sweeps alpha_B with alpha_A = 0.1
    spec("hpv_safe", ALPHAS),
    spec("hpv_safe_lenient", ALPHAS_LEN),
)
ACTIVE_SPECS = HPV_FINITE_SPECS
HPV_ORACLE_VARIANTS = tuple((n, hpv_options(n)) for n in (
    "hpv_single_pass", "hpv_single_pass_oracle_blanket", "hpv_single_pass_no_hint", "hpv_single_pass_union", "hpv_safe",
    "hpv_single_pass_shifted_parents"))


def _dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, default=lambda o: o.item() if hasattr(o, "item") else (o.tolist() if hasattr(o, "tolist") else str(o)))
    os.replace(tmp, path)


# ------------------------------------------------------------------------------------------------ oracle

@lru_cache(maxsize=None)
def _oracle_rows(oracle_dir, arm, N, T, tau):
    d = json.load(open(os.path.join(oracle_dir, arm, f"N{N}_T{T}_tau{tau}_d2.json")))
    out = {}
    for r in d["rows"]:
        if OLD_TO_NEW.get(r["name"], r["name"]) == "order_based":
            out[r["graph"]] = (r["graph_stats"]["sha1"], r["n_cand"])
    return out


def oracle_task(a):
    out_dir, R, arm, N, T, tau, g = a
    path = os.path.join(out_dir, arm, f"N{N}_T{T}_tau{tau}_d2", f"g{g:02d}.json")
    if os.path.exists(path):
        return path, "exists", 0.0
    t0 = time.time()
    inst = load_instance(os.path.join(R, "instances", arm, f"N{N}_T{T}_tau{tau}_d2_{arm}_g{g}.npz"))
    sha_stored, ncand_stored = _oracle_rows(os.path.join(R, "oracle"), arm, N, T, tau)[g]
    assert inst["sha1"] == sha_stored, (arm, N, T, tau, g)
    gr = inst["graph"]
    A = gr.A
    din, dout = int(A.sum(0).max()), int(A.sum(1).max())
    e_nonself = int(A.sum()) - N * (T - 1)
    rows = []
    for name, kw in HPV_ORACLE_VARIANTS:
        if name == "hpv_single_pass_shifted_parents" and arm != "window":
            continue
        o = method_runner.run_s2_method("hpv", None, graph=gr, ci_kind="oracle", alpha=0.01, per_target=True, hpv=kw)
        pp, t = o["per_pair"], o["tests"]
        sum_c = sum(p["n_cand"] for p in pp)
        assert sum_c == ncand_stored
        by_t = {}
        for p in pp:
            ty = p["pair"][1] // N
            b = by_t.setdefault(ty, [0, 0])
            b[0] += p["excess"]
            b[1] += p["n_R"]
        rows.append({
            "name": name, "exact": o["metrics"]["exact"], "fp": o["metrics"]["fp"], "fn": o["metrics"]["fn"],
            "unique": t["unique_tests"], "raw": t["raw_calls"], "n_cand": sum_c,
            "by_label_unique": t["by_label_unique"], "by_label_raw": t["by_label_raw"], "by_size_unique": t["by_size_unique"],
            "max_size": max(p["max_size"] for p in pp),
            "mean_target_max_frac": float(np.mean([p["max_size"] / p["n_cand"] for p in pp if p["n_cand"]])),
            "mean_target_tpc": float(np.mean([p["new_unique"] / p["n_cand"] for p in pp if p["n_cand"]])),
            "max_SA": max(p["max_SA"] for p in pp), "max_SB": max(p["max_SB"] for p in pp),
            "calls_identity": t["raw_calls"] == sum_c + e_nonself,
            "overhead_bound": 1 + 2 * (din - 1) / (N * T - 2),
            # learned_blanket / oracle_blanket: |S_A| <= b(Z) + |F|; union adds |pa_hat(X)| (= true in-degree of X here)
            "n_bound_b_viol": int(sum(1 for p in pp if "b_max" in p and p["max_SA"] > p["b_max"] + 1
                                      + (int(A[:, p["pair"][0]].sum()) if name == "hpv_single_pass_union" else 0))),
            "n_bound_glob_viol": int(sum(1 for p in pp if p["max_SA"] > din * (1 + dout) + 1)),
            "glob_bound": din * (1 + dout) + 1,
            "excess_by_t": {str(k): v[0] for k, v in sorted(by_t.items())},
            "hpv_stats": o["hpv_stats"], "seconds": o["seconds"]})
    _dump(path, {"arm": arm, "N": N, "T": T, "tau": tau, "d": 2, "graph": g, "sha1": inst["sha1"], "d_in": din,
                 "d_out": dout, "edges_nonself": e_nonself, "rows": rows, "seconds": time.time() - t0})
    return path, "done", time.time() - t0


# ------------------------------------------------------------------------------------------------ finite data

def _pcorr(Sig, a, b, S):
    idx = [a, b] + [int(s) for s in S]
    P = np.linalg.inv(Sig[np.ix_(idx, idx)])
    return abs(-P[0, 1] / np.sqrt(abs(P[0, 0] * P[1, 1])))


def finite_task(a):
    out_dir, R, N, T, g, M = a
    cell = f"window_N{N}_T{T}_tau1_d2"
    path = os.path.join(out_dir, cell, f"g{g:02d}_M{M}.json")
    if os.path.exists(path):
        return path, "exists", 0.0
    t0 = time.time()
    fin = os.path.join(R, "finite", "window")
    inst = load_instance(os.path.join(fin, "instances", cell, f"g{g:02d}.npz"))
    ref = json.load(open(os.path.join(fin, cell, f"g{g:02d}_M{M}.json")))
    assert inst["sha1"] == ref["sha1"] and list(inst["data_seed"]) == list(ref["data_seed"]), (cell, g, M)
    Xf = observed_data.observed_data(inst, inst["M_max"])
    assert observed_data.data_hash(Xf) == inst["data_sha1"], (cell, g)
    X = np.ascontiguousarray(Xf[:M])
    A = inst["A"]
    edges = observed_data.true_edge_list(A, N, T)
    eidx = {(int(u), int(v)): k for k, (u, v) in enumerate(edges)}
    res = dataset_eval.run_dataset(A, X, inst["tau"], ACTIVE_SPECS, order="time", edges=edges)
    Sig = observed_data.sigma_from_W(inst["W"])
    for r in res["runs"]:
        pt = r.pop("hpv_ptests", None)
        if pt:
            r["hpv_rhoA"] = [[eidx[(z, y)], len(S), float(_pcorr(Sig, z, y, S)), p]
                             for y, lst in pt for z, S, p in lst]
    res.update({"cell": cell, "arm": "window", "N": N, "T": T, "tau": inst["tau"], "d": inst["d"], "graph": g,
                "sha1": inst["sha1"], "data_sha1_ok": True, "n_true_edges_nonself": int(len(edges)),
                "seconds": time.time() - t0})
    _dump(path, res)
    return path, "done", time.time() - t0


# ------------------------------------------------------------------------------------------------ driver

def _guard(args):
    fn, a, deadline = args
    if time.time() > deadline:
        return a, "skipped", 0.0
    try:
        return fn(a)
    except Exception as e:                       # report and continue; the task file is not written
        return a, f"error {type(e).__name__}: {e}", 0.0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["oracle", "finite"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--results", default=R_DEFAULT, help="results directory of the earlier runs (read only)")
    ap.add_argument("--arm", default="time,window")
    ap.add_argument("--N", default="10,20")
    ap.add_argument("--T", default="8,16")
    ap.add_argument("--tau", default="1,2")
    ap.add_argument("--M", default="50,100,200,500,2000")
    ap.add_argument("--graphs", type=int, default=20)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--budget-sec", type=float, default=1e9)
    ap.add_argument("--specs", default=None, help="comma list of HPV_FINITE_SPECS names (finite mode); default all")
    ap.add_argument("--shard", default=None, help="K/NSH: run only tasks K, K+NSH, ... of the cost-sorted list")
    ap.add_argument("--summary", default=None, help="file for a one-line summary at the end")
    a = ap.parse_args(argv)
    global ACTIVE_SPECS
    if a.specs:
        want = [OLD_TO_NEW.get(n, n) for n in a.specs.split(",")]
        ACTIVE_SPECS = tuple(sp for sp in HPV_FINITE_SPECS if sp[0] in want)
        assert len(ACTIVE_SPECS) == len(want), (want, [sp[0] for sp in HPV_FINITE_SPECS])
    ints = lambda s: [int(x) for x in s.split(",")]
    if a.mode == "oracle":
        fn = oracle_task
        tasks = [(a.out_dir, a.results, arm, N, T, tau, g) for arm in a.arm.split(",") for N in ints(a.N)
                 for T in ints(a.T) for tau in ints(a.tau) for g in range(a.graphs)]
        cost = lambda t: (t[3] * t[4]) ** 2
    else:
        fn = finite_task
        tasks = [(a.out_dir, a.results, N, T, g, M) for N, T, M in itertools.product(ints(a.N), ints(a.T), ints(a.M))
                 for g in range(a.graphs)]
        cost = lambda t: (t[2] * t[3]) ** 2.2
    tasks.sort(key=lambda t: -cost(t))
    if a.shard:
        k, nsh = (int(x) for x in a.shard.split("/"))
        tasks = tasks[k::nsh]
    deadline = time.time() + a.budget_sec
    t0 = time.time()
    print(f"tasks {len(tasks)}", flush=True)
    n_err = 0
    if a.workers <= 1:
        it = map(_guard, [(fn, t, deadline) for t in tasks])
        pool = None
    else:
        import multiprocessing as mp
        pool = mp.get_context("fork").Pool(a.workers)
        it = pool.imap_unordered(_guard, [(fn, t, deadline) for t in tasks], chunksize=1)
    for r in it:
        if str(r[1]).startswith("error"):
            n_err += 1
            print(r[1], r[0], flush=True)
    if pool is not None:
        pool.close()
        pool.join()
    left = sum(1 for t in tasks if not os.path.exists(_task_path(a.mode, t)))
    msg = f"done in {time.time() - t0:.0f}s; errors {n_err}; remaining {left}; tasks {len(tasks)}" + (
        f"; shard {a.shard}" if a.shard else "")
    print(msg, flush=True)
    if a.summary:
        with open(a.summary, "w") as f:
            f.write(msg + "\n")


def _task_path(mode, t):
    if mode == "oracle":
        out_dir, R, arm, N, T, tau, g = t
        return os.path.join(out_dir, arm, f"N{N}_T{T}_tau{tau}_d2", f"g{g:02d}.json")
    out_dir, R, N, T, g, M = t
    return os.path.join(out_dir, f"window_N{N}_T{T}_tau1_d2", f"g{g:02d}_M{M}.json")


if __name__ == "__main__":
    main()
