"""Known-order baselines: IAMB per target and lasso per target (both take the self edge as known), plus ITPD + marginal-first, on the
instances of earlier runs. ITPD + marginal-first (itpd_marginal_first, padl.py `marginal_first`) is an ITPD variant, not a baseline; it is run
here because it is compared with the same baselines on the same instances.

    python -m itpd.run_known_order_baselines oracle --out-dir OUT/oracle --workers 16 [--arm time,window --N 5,10,20 --T 4,8,16 --tau 1,2,3 --d 1,2,3]
    python -m itpd.run_known_order_baselines finite --out-dir OUT/finite --workers 16 [--cells 10x8:0-49,20x8:0-49,10x16:0-19,20x16:0-19
        --M 50,100,200,500,2000] [--no-lasso] [--budget-sec S] [--shard K/NSH] [--summary FILE]
    python -m itpd.run_known_order_baselines lasso_ebic_fixed --out-dir OUT/lasso --workers 16 [--cells ... --M ...]
        (extended-BIC and fixed-penalty lasso, see itpd/lasso.py)
    python -m itpd.run_known_order_baselines timing --out-dir OUT/timing --workers 4 [--cells 10x8:0-4,...] [--M 200,2000]

oracle: one task = one oracle-run cell (arm, N, T, tau, d) = 20 graphs of R/instances/<arm> (R = `--results`, default $ITPD_RESULTS or ./results); the sha1 of every graph is
checked against the row of the earlier oracle run; per graph: IAMB (three tie rules), ITPD + marginal-first (lazy and non-lazy), exactness and
unique tests, next to the unique counts of itpd_naive, itpd and full_conditioning of the same graph (paired by graph).
finite: instances R/finite/window/instances/window_N<N>_T<T>_tau1_d2/g<g>.npz (the finite-data instances of the earlier run, with sha1, data_seed and
data_sha1 checked against its task JSON); one JSON per (cell, graph, M) with `runs` (dataset_eval.IAMB_MARGINAL_FIRST_SPECS
through dataset_eval.run_dataset: Fisher-z, one shared p-value memo per dataset, per-target infeasibility, common targets) and `lasso`
(itpd.lasso rows, scored by the same scorer on all targets and on the common targets). Resumable (an existing file is skipped).
timing: wall-clock per method at alpha 0.01 with a fresh test object and no shared memo for each method (the finite runs share
one p-value memo among the methods of a dataset, so their `seconds` are not comparable); per task one process, methods one
after another, perf_counter and process CPU time.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from functools import lru_cache

import numpy as np

from . import dataset_eval, lasso, method_runner, observed_data
from .ci import DSepCI, FisherZ
from .instances import load_instance
from .methods_registry import BY_NAME, OLD_TO_NEW, shrink_options
from .metrics import edge_metrics

R_DEFAULT = os.environ.get("ITPD_RESULTS", "results")
CELLS_DEFAULT = "10x8:0-49,20x8:0-49,10x16:0-19,20x16:0-19"
ACTIVE_SPECS = dataset_eval.IAMB_MARGINAL_FIRST_SPECS


def _dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, default=lambda o: o.item() if hasattr(o, "item") else (o.tolist() if hasattr(o, "tolist") else str(o)))
    os.replace(tmp, path)


# ------------------------------------------------------------------------------------------------ oracle

@lru_cache(maxsize=None)
def _stored_oracle_cell(oracle_dir, arm, N, T, tau, d):
    """Rows of a stored oracle-run cell by graph and (current) row name; results written before the rename carry old names."""
    j = json.load(open(os.path.join(oracle_dir, arm, f"N{N}_T{T}_tau{tau}_d{d}.json")))
    out = {}
    for r in j["rows"]:
        out.setdefault(r["graph"], {})[OLD_TO_NEW.get(r["name"], r["name"])] = r
    return out


# (row name, runner method, extra kwargs), from methods_registry: IAMB with three tie rules and ITPD + marginal-first, lazy and non-lazy
ORACLE_SPECS = tuple((n, BY_NAME[n].function, {k: v for k, v in BY_NAME[n].options.items() if k in ("tie", "lazy")}) for n in (
    "iamb_known_order", "iamb_known_order_tie_first", "iamb_known_order_tie_random", "itpd_marginal_first",
    "itpd_marginal_first_nonlazy"))


def oracle_task(a):
    out_dir, R, arm, N, T, tau, d = a
    path = os.path.join(out_dir, arm, f"N{N}_T{T}_tau{tau}_d{d}.json")
    if os.path.exists(path):
        return path, "exists", 0.0
    t0 = time.time()
    stored = _stored_oracle_cell(os.path.join(R, "oracle"), arm, N, T, tau, d)
    graphs = []
    for g in sorted(stored):
        inst = load_instance(os.path.join(R, "instances", arm, f"N{N}_T{T}_tau{tau}_d{d}_{arm}_g{g}.npz"))
        ref = stored[g]
        assert inst["sha1"] == ref["full_conditioning"]["graph_stats"]["sha1"], (arm, N, T, tau, d, g)
        gr = inst["graph"]
        ci = DSepCI(gr.A)
        n_cand_ref = ref["full_conditioning"]["n_cand"]
        row = {"graph": g, "sha1": inst["sha1"], "n_cand": n_cand_ref,
               "stored_oracle_run": {k: {"unique": ref[k]["unique"], "raw": ref[k]["raw"], "max_size": ref[k]["max_size"],
                                   "exact": ref[k]["exact"]} for k in ("itpd_naive", "itpd", "full_conditioning")}}
        for name, method, kw in ORACLE_SPECS:
            kw = dict(kw)
            tie = kw.pop("tie", None)
            if tie is not None:                                # IAMB tie rule: call the module through method_runner's recorder
                from . import iamb as _iamb
                from .ci import Recorder
                rec = Recorder(ci, cache="run", infeasible="raise")
                res = _iamb.run_s2(rec, N, T, 0.01, per_target=True, tie=tie)
                m = edge_metrics(gr.A, res.A_hat, N, T)
                t = res.summary
                pp = res.per_pair
                bad = len(res.infeasible_targets)
                sec = res.seconds
            else:
                o = method_runner.run_s2_method(method, None, graph=gr, ci=ci, alpha=0.01, per_target=True, order="time", **kw)
                m, t, pp, bad, sec = o["metrics"], o["tests"], o["per_pair"], o["n_infeasible_targets"], o["seconds"]
            assert sum(p["n_cand"] for p in pp) == n_cand_ref
            row[name] = {"exact": m["exact"], "fp": m["fp"], "fn": m["fn"], "unique": t["unique_tests"], "raw": t["raw_calls"],
                         "max_size": max(p["max_size"] for p in pp), "by_size_unique": t["by_size_unique"],
                         "mean_target_tpc": float(np.mean([p["new_unique"] / p["n_cand"] for p in pp if p["n_cand"]])),
                         "n_infeasible": bad, "seconds": sec}
        graphs.append(row)
    _dump(path, {"arm": arm, "N": N, "T": T, "tau": tau, "d": d, "graphs": graphs, "seconds": time.time() - t0,
                 "backend": ci.backend})
    return path, "done", time.time() - t0


# ------------------------------------------------------------------------------------------------ finite

def _load_finite(R, N, T, g, M):
    cell = f"window_N{N}_T{T}_tau1_d2"
    fin = os.path.join(R, "finite", "window")
    inst = load_instance(os.path.join(fin, "instances", cell, f"g{g:02d}.npz"))
    ref = json.load(open(os.path.join(fin, cell, f"g{g:02d}_M{M}.json")))
    assert inst["sha1"] == ref["sha1"] and list(inst["data_seed"]) == list(ref["data_seed"]), (cell, g, M)
    Xf = observed_data.observed_data(inst, inst["M_max"])
    assert observed_data.data_hash(Xf) == inst["data_sha1"], (cell, g)
    return cell, inst, ref, np.ascontiguousarray(Xf[:M])


def lasso_rows(A_truth, X, N, T, M):
    ct = dataset_eval.common_tmax(N, T, M)
    cmask = dataset_eval._cols_mask(N, T, ct)
    r = lasso.lasso_s2(X)
    rows = []
    for name, A, lam, sec, cpu in [("lasso_cv", r.A_cv, None, r.seconds_cv, r.cpu_cv)] + [
            ("lasso_path", Ak, l, None, None) for Ak, l in zip(r.A_path, r.lambdas)]:
        row = {"name": name, "lambda": lam, "metrics": edge_metrics(A_truth, A, N, T), "seconds": sec, "cpu_seconds": cpu}
        row["metrics_common"] = edge_metrics(A_truth, A, N, T, targets=cmask) if ct >= 1 else None
        rows.append(row)
    return {"rows": rows, "seconds_cv": r.seconds_cv, "seconds_path": r.seconds_path, "cpu_cv": r.cpu_cv,
            "cpu_path": r.cpu_path, "common_tmax": ct, "median_cv_lambda": float(np.median(r.n_cv_alpha))}


def finite_task(a):
    out_dir, R, N, T, g, M, do_lasso = a
    cell = f"window_N{N}_T{T}_tau1_d2"
    path = os.path.join(out_dir, cell, f"g{g:02d}_M{M}.json")
    if os.path.exists(path):
        return path, "exists", 0.0
    t0 = time.time()
    cell, inst, ref, X = _load_finite(R, N, T, g, M)
    A = inst["A"]
    edges = observed_data.true_edge_list(A, N, T)
    res = dataset_eval.run_dataset(A, X, inst["tau"], ACTIVE_SPECS, order="time", edges=edges)
    assert res["common_tmax"] == ref["common_tmax"] and len(edges) == ref["n_true_edges_nonself"]
    if do_lasso:
        res["lasso"] = lasso_rows(A, X, N, T, M)
    res.update({"cell": cell, "arm": "window", "N": N, "T": T, "tau": inst["tau"], "d": inst["d"], "graph": g,
                "sha1": inst["sha1"], "data_sha1_ok": True, "n_true_edges_nonself": int(len(edges)),
                "seconds": time.time() - t0})
    _dump(path, res)
    return path, "done", time.time() - t0


def lasso_ebic_fixed_task(a):
    """The two lasso rules added after the cross-validated lasso over-selected: lasso_ebic (headline lasso) and lasso_fixed, own file per task."""
    out_dir, R, N, T, g, M = a
    cell = f"window_N{N}_T{T}_tau1_d2"
    path = os.path.join(out_dir, cell, f"g{g:02d}_M{M}.json")
    if os.path.exists(path):
        return path, "exists", 0.0
    t0 = time.time()
    cell, inst, ref, X = _load_finite(R, N, T, g, M)
    A = inst["A"]
    ct = dataset_eval.common_tmax(N, T, M)
    cmask = dataset_eval._cols_mask(N, T, ct)
    r = lasso.lasso_s2(X, do_cv=False, do_path=False, do_ebic=True, do_fixed=True)
    rows = []
    for name, Ah, sec, cpu in (("lasso_ebic", r.A_ebic, r.seconds_ebic, r.cpu_ebic), ("lasso_fixed", r.A_fixed, r.seconds_fixed, r.cpu_fixed)):
        rows.append({"name": name, "metrics": edge_metrics(A, Ah, N, T), "metrics_common": edge_metrics(A, Ah, N, T, targets=cmask),
                     "seconds": sec, "cpu_seconds": cpu})
    _dump(path, {"cell": cell, "N": N, "T": T, "graph": g, "M": M, "sha1": inst["sha1"], "common_tmax": ct, "rows": rows,
                 "seconds": time.time() - t0})
    return path, "done", time.time() - t0


# ------------------------------------------------------------------------------------------------ timing

TIMING_SPECS = (("itpd_naive", "itpd_naive", {"lazy": True}), ("itpd", "itpd", {"lazy": True}),
                ("itpd_marginal_first", "itpd_marginal_first", {"lazy": True}), ("full_conditioning", "full_conditioning", {}),
                ("iamb_known_order", "iamb_known_order", {}),
                ("blanket_screened_shrink_recheck", "blanket_screened_shrink", {"shrink": shrink_options("blanket_screened_shrink_recheck")}))


def timing_task(a):
    out_dir, R, N, T, g, M = a
    cell = f"window_N{N}_T{T}_tau1_d2"
    path = os.path.join(out_dir, cell, f"g{g:02d}_M{M}.json")
    if os.path.exists(path):
        return path, "exists", 0.0
    t0 = time.time()
    cell, inst, ref, X = _load_finite(R, N, T, g, M)
    tg_A = inst["A"]
    rows = []
    for name, method, kw in TIMING_SPECS:
        ci = FisherZ(X.reshape(M, T * N))
        c0, w0 = time.process_time(), time.perf_counter()
        o = method_runner.run_s2_method(method, None, graph=inst["graph"], ci=ci, alpha=0.01, order="time", per_target=True,
                                 infeasible="raise", **kw)
        rows.append({"name": name, "unique": o["tests"]["unique_tests"], "raw": o["tests"]["raw_calls"],
                     "seconds_wall": time.perf_counter() - w0, "seconds_cpu": time.process_time() - c0,
                     "seconds_ci": o["tests"]["seconds_ci"], "n_inf_targets": o["n_infeasible_targets"],
                     "n_targets": o["n_targets"], "metrics": o["metrics"]})
    lr = lasso_rows(tg_A, X, N, T, M)
    rows.append({"name": "lasso_cv", "seconds_wall": lr["seconds_cv"], "seconds_cpu": lr["cpu_cv"]})
    rows.append({"name": "lasso_path_all_lambdas", "seconds_wall": lr["seconds_path"], "seconds_cpu": lr["cpu_path"]})
    _dump(path, {"cell": cell, "N": N, "T": T, "graph": g, "M": M, "rows": rows, "host": os.uname().nodename,
                 "seconds": time.time() - t0})
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


def parse_cells(s):
    out = []
    for part in s.split(","):
        nt, gr = part.split(":")
        N, T = (int(v) for v in nt.split("x"))
        lo, hi = (int(v) for v in gr.split("-"))
        out.append((N, T, range(lo, hi + 1)))
    return out


def _task_path(mode, t):
    if mode == "oracle":
        out_dir, R, arm, N, T, tau, d = t
        return os.path.join(out_dir, arm, f"N{N}_T{T}_tau{tau}_d{d}.json")
    out_dir, R, N, T, g, M = t[:6]
    return os.path.join(out_dir, f"window_N{N}_T{T}_tau1_d2", f"g{g:02d}_M{M}.json")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["oracle", "finite", "timing", "lasso_ebic_fixed"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--results", default=R_DEFAULT, help="results directory of the earlier runs (read only)")
    ap.add_argument("--arm", default="time,window")
    ap.add_argument("--N", default="5,10,20")
    ap.add_argument("--T", default="4,8,16")
    ap.add_argument("--tau", default="1,2,3")
    ap.add_argument("--d", default="1,2,3")
    ap.add_argument("--cells", default=CELLS_DEFAULT)
    ap.add_argument("--M", default="50,100,200,500,2000")
    ap.add_argument("--no-lasso", action="store_true")
    ap.add_argument("--specs", default=None, help="comma list of IAMB_MARGINAL_FIRST_SPECS names (finite mode); default all (use a new --out-dir)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--budget-sec", type=float, default=1e9)
    ap.add_argument("--shard", default=None)
    ap.add_argument("--summary", default=None)
    a = ap.parse_args(argv)
    global ACTIVE_SPECS
    if a.specs:
        want = [OLD_TO_NEW.get(n, n) for n in a.specs.split(",")]
        ACTIVE_SPECS = tuple(sp for sp in dataset_eval.IAMB_MARGINAL_FIRST_SPECS if sp[0] in want)
        assert len(ACTIVE_SPECS) == len(want), want
    ints = lambda s: [int(x) for x in s.split(",")]
    if a.mode == "oracle":
        fn = oracle_task
        tasks = [(a.out_dir, a.results, arm, N, T, tau, d) for arm in a.arm.split(",") for N in ints(a.N) for T in ints(a.T)
                 for tau in ints(a.tau) for d in ints(a.d)]
        cost = lambda t: (t[3] * t[4]) ** 2
    else:
        fn = {"finite": finite_task, "timing": timing_task, "lasso_ebic_fixed": lasso_ebic_fixed_task}[a.mode]
        tasks = []
        for N, T, gs in parse_cells(a.cells):
            for M in ints(a.M):
                for g in gs:
                    tasks.append((a.out_dir, a.results, N, T, g, M) + ((not a.no_lasso,) if a.mode == "finite" else ()))
        cost = lambda t: (t[2] * t[3]) ** 2.2
    tasks = [t for t in tasks if not os.path.exists(_task_path(a.mode, t))]
    tasks.sort(key=lambda t: -cost(t))
    if a.shard:
        k, nsh = (int(x) for x in a.shard.split("/"))
        tasks = tasks[k::nsh]
    deadline = time.time() + a.budget_sec
    t0 = time.time()
    print(f"tasks todo {len(tasks)}", flush=True)
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


if __name__ == "__main__":
    main()
