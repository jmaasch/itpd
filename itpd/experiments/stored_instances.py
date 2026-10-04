"""Runs on the stored instances of earlier runs: the known-order baselines (`known_order`) and ITPD-S and ITPD-S+ (`itpd_s`).
Nothing is regenerated except the data of a finite-data instance, which is checked against its stored data_sha1. One JSON per
task, resumable (an existing file is skipped). The earlier runs are read from the results directory R (`--results`, default
$ITPD_RESULTS or ./results): `python -m itpd.experiments oracle_counts cell ... --out R/oracle/<arm>/N<N>_T<T>_tau<tau>_d2.json
--instances-dir R/instances/<arm> --with-weights` and `python -m itpd.experiments finite_data --arm window --out-dir R/finite/window ...`.

    python -m itpd.experiments stored_instances known_order oracle --out-dir OUT/oracle --workers 16 [--arm time,window --N 5,10,20 --T 4,8,16 --tau 1,2,3 --d 1,2,3]
    python -m itpd.experiments stored_instances known_order finite --out-dir OUT/finite --workers 16 [--cells 10x8:0-49,20x8:0-49,10x16:0-19,20x16:0-19
        --M 50,100,200,500,2000] [--no-lasso] [--budget-sec S] [--shard K/NSH] [--summary FILE]
    python -m itpd.experiments stored_instances known_order lasso_ebic_fixed --out-dir OUT/lasso --workers 16 [--cells ... --M ...]
        (extended-BIC and fixed-penalty lasso, see itpd/lasso.py)
    python -m itpd.experiments stored_instances known_order timing --out-dir OUT/timing --workers 4 [--cells 10x8:0-4,...] [--M 200,2000]
    python -m itpd.experiments stored_instances itpd_s oracle --out-dir OUT/oracle --workers 12 [--arm time,window --N 10,20 --T 8,16 --tau 1,2 --graphs 20]
    python -m itpd.experiments stored_instances itpd_s finite --out-dir OUT/finite --workers 12 [--N 10,20 --T 8,16 --M 50,100,200,500,2000 --graphs 20]
        [--budget-sec S]   (start no new task after S seconds)
        [--specs itpd_s_plus,itpd_s_plus_screen_clean]   (run only these entries of SHRINK_FINITE_SPECS; default all; use a new --out-dir)
        [--shard K/NSH]    (this process runs tasks K, K + NSH, ... of the cost-sorted list; one shard per batch job)
        [--summary FILE]   (one-line summary written at the end)

known_order: IAMB per target and lasso per target (both take the self edge as known), plus ITPD + marginal-first, on the
instances of earlier runs. ITPD + marginal-first (itpd_marginal_first, padl.py `marginal_first`) is an ITPD variant, not a baseline; it is run
here because it is compared with the same baselines on the same instances.
oracle: one task = one oracle-run cell (arm, N, T, tau, d) = 20 graphs of R/instances/<arm>; the sha1 of every graph is
checked against the row of the earlier oracle run; per graph: IAMB (three tie rules), ITPD + marginal-first (lazy and non-lazy), exactness and
unique tests, next to the unique counts of itpd_naive, itpd and full_conditioning of the same graph (paired by graph).
finite: instances R/finite/window/instances/window_N<N>_T<T>_tau1_d2/g<g>.npz (the finite-data instances of the earlier run, with sha1, data_seed and
data_sha1 checked against its task JSON); one JSON per (cell, graph, M) with `runs` (dataset_eval.IAMB_MARGINAL_FIRST_SPECS
through dataset_eval.run_dataset: Fisher-z, one shared p-value memo per dataset, per-target infeasibility, common targets) and `lasso`
(itpd.lasso rows, scored by the same scorer on all targets and on the common targets). Resumable (an existing file is skipped).
timing: wall-clock per method at alpha 0.01 with a fresh test object and no shared memo for each method (the finite runs share
one p-value memo among the methods of a dataset, so their `seconds` are not comparable); per task one process, methods one
after another, perf_counter and process CPU time.

itpd_s oracle: instances R/instances/<arm>/N<N>_T<T>_tau<tau>_d2_<arm>_g<g>.npz, sha1 checked against the stored oracle-run row of the same
graph (R/oracle/<arm>/N<N>_T<T>_tau<tau>_d2.json, graph_stats.sha1) and Sum|C| against its n_cand. ITPD-S variants (`SHRINK_ORACLE_VARIANTS`):
screening conditioning sets learned_blanket, true_blanket, own_lag, blanket_shifted, learned_blanket + second pass (itpd_s_plus), and shifted_parents (window arm only); exact
d-separation oracle on the full graph.
itpd_s finite: instances R/finite/window/instances/window_N<N>_T<T>_tau1_d2/g<g>.npz, sha1 and data_seed checked against the stored finite-data task
JSON, data regenerated and checked against data_sha1 (M = 2000), the first M rows used (paired across M).
Methods `SHRINK_FINITE_SPECS` through dataset_eval.run_dataset (Fisher-z, one shared p-value memo per dataset, per-target infeasibility,
common targets t <= (M - 3) / N). For the true parents, the population partial correlation given the screening set actually
used is computed from the exact covariance (rows "screening_rho": [edge index, |S_A|, |rho|, p-value]).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from functools import lru_cache

import numpy as np

from .. import dataset_eval, iamb, lasso, method_runner, observed_data
from ..ci import DSepCI, FisherZ, Recorder
from ..instances import load_instance
from ..methods_registry import BY_NAME, OLD_TO_NEW, shrink_options, spec
from ..metrics import edge_metrics
from . import common

R_DEFAULT = os.environ.get("ITPD_RESULTS", "results")
CELLS_DEFAULT = "10x8:0-49,20x8:0-49,10x16:0-19,20x16:0-19"
ACTIVE_KNOWN_ORDER_SPECS = dataset_eval.IAMB_MARGINAL_FIRST_SPECS     # set by main(--specs); forked workers inherit it


# ------------------------------------------------------------------------------------------------ stored finite-data instances

def _finite_cell(N, T):
    return f"window_N{N}_T{T}_tau1_d2"


def _load_finite(R, N, T, g, M):
    cell = _finite_cell(N, T)
    fin = os.path.join(R, "finite", "window")
    inst = load_instance(os.path.join(fin, "instances", cell, f"g{g:02d}.npz"))
    ref = json.load(open(os.path.join(fin, cell, f"g{g:02d}_M{M}.json")))
    assert inst["sha1"] == ref["sha1"] and list(inst["data_seed"]) == list(ref["data_seed"]), (cell, g, M)
    Xf = observed_data.observed_data(inst, inst["M_max"])
    assert observed_data.data_hash(Xf) == inst["data_sha1"], (cell, g)
    return cell, inst, ref, np.ascontiguousarray(Xf[:M])


# ------------------------------------------------------------------------------------------------ known-order baselines: oracle

@lru_cache(maxsize=None)
def _stored_oracle_cell(oracle_dir, arm, N, T, tau, d):
    """Rows of a stored oracle-run cell by graph and (current) row name; results written before the rename carry old names."""
    j = json.load(open(os.path.join(oracle_dir, arm, f"N{N}_T{T}_tau{tau}_d{d}.json")))
    out = {}
    for r in j["rows"]:
        out.setdefault(r["graph"], {})[OLD_TO_NEW.get(r["name"], r["name"])] = r
    return out


# (row name, runner method, extra kwargs), from methods_registry: IAMB with three tie rules and ITPD + marginal-first, lazy and non-lazy
KNOWN_ORDER_ORACLE_SPECS = tuple((n, BY_NAME[n].function, {k: v for k, v in BY_NAME[n].options.items() if k in ("tie", "lazy")}) for n in (
    "iamb_known_order", "iamb_known_order_tie_first", "iamb_known_order_tie_random", "itpd_marginal_first",
    "itpd_marginal_first_nonlazy"))


def known_order_oracle_task(a):
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
        for name, method, kw in KNOWN_ORDER_ORACLE_SPECS:
            kw = dict(kw)
            tie = kw.pop("tie", None)
            if tie is not None:                                # IAMB tie rule: call the module through method_runner's recorder
                rec = Recorder(ci, cache="run", infeasible="raise")
                res = iamb.run_s2(rec, N, T, 0.01, per_target=True, tie=tie)
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
    common.dump_json(path, {"arm": arm, "N": N, "T": T, "tau": tau, "d": d, "graphs": graphs, "seconds": time.time() - t0,
                            "backend": ci.backend})
    return path, "done", time.time() - t0


# ------------------------------------------------------------------------------------------------ known-order baselines: finite data

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


def known_order_finite_task(a):
    out_dir, R, N, T, g, M, do_lasso = a
    cell = _finite_cell(N, T)
    path = os.path.join(out_dir, cell, f"g{g:02d}_M{M}.json")
    if os.path.exists(path):
        return path, "exists", 0.0
    t0 = time.time()
    cell, inst, ref, X = _load_finite(R, N, T, g, M)
    A = inst["A"]
    edges = observed_data.true_edge_list(A, N, T)
    res = dataset_eval.run_dataset(A, X, inst["tau"], ACTIVE_KNOWN_ORDER_SPECS, order="time", edges=edges)
    assert res["common_tmax"] == ref["common_tmax"] and len(edges) == ref["n_true_edges_nonself"]
    if do_lasso:
        res["lasso"] = lasso_rows(A, X, N, T, M)
    res.update({"cell": cell, "arm": "window", "N": N, "T": T, "tau": inst["tau"], "d": inst["d"], "graph": g,
                "sha1": inst["sha1"], "data_sha1_ok": True, "n_true_edges_nonself": int(len(edges)),
                "seconds": time.time() - t0})
    common.dump_json(path, res)
    return path, "done", time.time() - t0


def lasso_ebic_fixed_task(a):
    """The two lasso rules added after the cross-validated lasso over-selected: lasso_ebic (headline lasso) and lasso_fixed, own file per task."""
    out_dir, R, N, T, g, M = a
    cell = _finite_cell(N, T)
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
    common.dump_json(path, {"cell": cell, "N": N, "T": T, "graph": g, "M": M, "sha1": inst["sha1"], "common_tmax": ct, "rows": rows,
                            "seconds": time.time() - t0})
    return path, "done", time.time() - t0


# ------------------------------------------------------------------------------------------------ known-order baselines: timing

TIMING_SPECS = (("itpd_naive", "itpd_naive", {"lazy": True}), ("itpd", "itpd", {"lazy": True}),
                ("itpd_marginal_first", "itpd_marginal_first", {"lazy": True}), ("full_conditioning", "full_conditioning", {}),
                ("iamb_known_order", "iamb_known_order", {}),
                ("itpd_s_plus", "itpd_s", {"shrink": shrink_options("itpd_s_plus")}))


def timing_task(a):
    out_dir, R, N, T, g, M = a
    cell = _finite_cell(N, T)
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
    common.dump_json(path, {"cell": cell, "N": N, "T": T, "graph": g, "M": M, "rows": rows, "host": os.uname().nodename,
                            "seconds": time.time() - t0})
    return path, "done", time.time() - t0


def parse_cells(s):
    out = []
    for part in s.split(","):
        nt, gr = part.split(":")
        N, T = (int(v) for v in nt.split("x"))
        lo, hi = (int(v) for v in gr.split("-"))
        out.append((N, T, range(lo, hi + 1)))
    return out


# ------------------------------------------------------------------------------------------------ ITPD-S and ITPD-S+: variants and specs

ALPHAS = dataset_eval.ALPHAS
ALPHAS_LEN = tuple(a for a in ALPHAS if a <= 0.1 + 1e-12)
SHRINK_FINITE_SPECS = (                      # built from methods_registry.spec; set by main(--specs); forked workers inherit it
    spec("itpd_s", ALPHAS, keep_parent_tests=True),
    spec("itpd_s_screen_clean", ALPHAS_LEN, keep_parent_tests=True),
    spec("itpd_s_true_blanket", (dataset_eval.PRIMARY,), keep_parent_tests=True),
    spec("itpd_s_true_blanket_screen_clean", (dataset_eval.PRIMARY,), keep_parent_tests=True),
    # the re-check (always-verify) at equal alpha with the full 13-point sweep (matched-false-positive curves); the screen-and-clean
    # row sweeps alpha_shr with alpha_scr = 0.1
    spec("itpd_s_plus", ALPHAS),
    spec("itpd_s_plus_screen_clean", ALPHAS_LEN),
)
ACTIVE_SHRINK_SPECS = SHRINK_FINITE_SPECS
SHRINK_ORACLE_VARIANTS = tuple((n, shrink_options(n)) for n in (
    "itpd_s", "itpd_s_true_blanket", "itpd_s_own_lag", "itpd_s_blanket_shifted", "itpd_s_plus",
    "itpd_s_shifted_parents"))


# ------------------------------------------------------------------------------------------------ ITPD-S and ITPD-S+: oracle

@lru_cache(maxsize=None)
def _oracle_rows(oracle_dir, arm, N, T, tau):
    d = json.load(open(os.path.join(oracle_dir, arm, f"N{N}_T{T}_tau{tau}_d2.json")))
    out = {}
    for r in d["rows"]:
        if OLD_TO_NEW.get(r["name"], r["name"]) == "full_conditioning":
            out[r["graph"]] = (r["graph_stats"]["sha1"], r["n_cand"])
    return out


def shrink_oracle_task(a):
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
    for name, kw in SHRINK_ORACLE_VARIANTS:
        if name == "itpd_s_shifted_parents" and arm != "window":
            continue
        o = method_runner.run_s2_method("itpd_s", None, graph=gr, ci_kind="oracle", alpha=0.01, per_target=True, shrink=kw)
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
            # learned_blanket / true_blanket: |S_A| <= b(Z) + |F|; blanket_shifted adds |pa_hat(X)| (= true in-degree of X here)
            "n_bound_b_viol": int(sum(1 for p in pp if "b_max" in p and p["max_SA"] > p["b_max"] + 1
                                      + (int(A[:, p["pair"][0]].sum()) if name == "itpd_s_blanket_shifted" else 0))),
            "n_bound_glob_viol": int(sum(1 for p in pp if p["max_SA"] > din * (1 + dout) + 1)),
            "glob_bound": din * (1 + dout) + 1,
            "excess_by_t": {str(k): v[0] for k, v in sorted(by_t.items())},
            "shrink_stats": o["shrink_stats"], "seconds": o["seconds"]})
    common.dump_json(path, {"arm": arm, "N": N, "T": T, "tau": tau, "d": 2, "graph": g, "sha1": inst["sha1"], "d_in": din,
                            "d_out": dout, "edges_nonself": e_nonself, "rows": rows, "seconds": time.time() - t0})
    return path, "done", time.time() - t0


# ------------------------------------------------------------------------------------------------ ITPD-S and ITPD-S+: finite data

def _pcorr(Sig, a, b, S):
    idx = [a, b] + [int(s) for s in S]
    P = np.linalg.inv(Sig[np.ix_(idx, idx)])
    return abs(-P[0, 1] / np.sqrt(abs(P[0, 0] * P[1, 1])))


def shrink_finite_task(a):
    out_dir, R, N, T, g, M = a
    cell = _finite_cell(N, T)
    path = os.path.join(out_dir, cell, f"g{g:02d}_M{M}.json")
    if os.path.exists(path):
        return path, "exists", 0.0
    t0 = time.time()
    _, inst, ref, X = _load_finite(R, N, T, g, M)
    A = inst["A"]
    edges = observed_data.true_edge_list(A, N, T)
    eidx = {(int(u), int(v)): k for k, (u, v) in enumerate(edges)}
    res = dataset_eval.run_dataset(A, X, inst["tau"], ACTIVE_SHRINK_SPECS, order="time", edges=edges)
    Sig = observed_data.sigma_from_W(inst["W"])
    for r in res["runs"]:
        pt = r.pop("screening_parent_tests", None)
        if pt:
            r["screening_rho"] = [[eidx[(z, y)], len(S), float(_pcorr(Sig, z, y, S)), p]
                             for y, lst in pt for z, S, p in lst]
    res.update({"cell": cell, "arm": "window", "N": N, "T": T, "tau": inst["tau"], "d": inst["d"], "graph": g,
                "sha1": inst["sha1"], "data_sha1_ok": True, "n_true_edges_nonself": int(len(edges)),
                "seconds": time.time() - t0})
    common.dump_json(path, res)
    return path, "done", time.time() - t0


# ------------------------------------------------------------------------------------------------ driver

def _known_order_path(mode, t):
    if mode == "oracle":
        out_dir, R, arm, N, T, tau, d = t
        return os.path.join(out_dir, arm, f"N{N}_T{T}_tau{tau}_d{d}.json")
    out_dir, R, N, T, g, M = t[:6]
    return os.path.join(out_dir, _finite_cell(N, T), f"g{g:02d}_M{M}.json")


def _shrink_path(mode, t):
    if mode == "oracle":
        out_dir, R, arm, N, T, tau, g = t
        return os.path.join(out_dir, arm, f"N{N}_T{T}_tau{tau}_d2", f"g{g:02d}.json")
    out_dir, R, N, T, g, M = t
    return os.path.join(out_dir, _finite_cell(N, T), f"g{g:02d}_M{M}.json")


def _run(a, fn, tasks, cost, path_of, drop_done):
    """Biggest tasks first, optionally one shard of the list, stop starting tasks after the budget; the closing line reports errors,
    remaining and tasks. known_order lists only the tasks still to do (so shards are cut from that list), itpd_s shards the full list."""
    if drop_done:
        tasks = [t for t in tasks if not os.path.exists(path_of(t))]
    tasks.sort(key=lambda t: -cost(t))
    tasks = common.take_shard(tasks, a.shard)
    t0 = time.time()
    print(f"tasks todo {len(tasks)}" if drop_done else f"tasks {len(tasks)}", flush=True)
    n_err = common.run_tasks(fn, tasks, a.workers, t0 + a.budget_sec)
    left = sum(1 for t in tasks if not os.path.exists(path_of(t)))
    common.finish(t0, left, n_err, len(tasks), a.shard, a.summary)


def _known_order(a):
    global ACTIVE_KNOWN_ORDER_SPECS
    if a.specs:
        want = [OLD_TO_NEW.get(n, n) for n in a.specs.split(",")]
        ACTIVE_KNOWN_ORDER_SPECS = tuple(sp for sp in dataset_eval.IAMB_MARGINAL_FIRST_SPECS if sp[0] in want)
        assert len(ACTIVE_KNOWN_ORDER_SPECS) == len(want), want
    if a.mode == "oracle":
        fn = known_order_oracle_task
        tasks = [(a.out_dir, a.results, arm, N, T, tau, d) for arm in a.arm.split(",") for N in common.ints(a.N) for T in common.ints(a.T)
                 for tau in common.ints(a.tau) for d in common.ints(a.d)]
        cost = lambda t: (t[3] * t[4]) ** 2
    else:
        fn = {"finite": known_order_finite_task, "timing": timing_task, "lasso_ebic_fixed": lasso_ebic_fixed_task}[a.mode]
        tasks = []
        for N, T, gs in parse_cells(a.cells):
            for M in common.ints(a.M):
                for g in gs:
                    tasks.append((a.out_dir, a.results, N, T, g, M) + ((not a.no_lasso,) if a.mode == "finite" else ()))
        cost = lambda t: (t[2] * t[3]) ** 2.2
    _run(a, fn, tasks, cost, lambda t: _known_order_path(a.mode, t), drop_done=True)


def _shrink(a):
    global ACTIVE_SHRINK_SPECS
    if a.specs:
        want = [OLD_TO_NEW.get(n, n) for n in a.specs.split(",")]
        ACTIVE_SHRINK_SPECS = tuple(sp for sp in SHRINK_FINITE_SPECS if sp[0] in want)
        assert len(ACTIVE_SHRINK_SPECS) == len(want), (want, [sp[0] for sp in SHRINK_FINITE_SPECS])
    if a.mode == "oracle":
        fn = shrink_oracle_task
        tasks = [(a.out_dir, a.results, arm, N, T, tau, g) for arm in a.arm.split(",") for N in common.ints(a.N)
                 for T in common.ints(a.T) for tau in common.ints(a.tau) for g in range(a.graphs)]
        cost = lambda t: (t[3] * t[4]) ** 2
    else:
        fn = shrink_finite_task
        tasks = [(a.out_dir, a.results, N, T, g, M) for N, T, M in itertools.product(common.ints(a.N), common.ints(a.T), common.ints(a.M))
                 for g in range(a.graphs)]
        cost = lambda t: (t[2] * t[3]) ** 2.2
    _run(a, fn, tasks, cost, lambda t: _shrink_path(a.mode, t), drop_done=False)


def _family_args(sp, mode_choices, N, T, tau, specs_help):
    sp.add_argument("mode", choices=mode_choices)
    sp.add_argument("--out-dir", required=True)
    sp.add_argument("--results", default=R_DEFAULT, help="results directory of the earlier runs (read only)")
    sp.add_argument("--arm", default="time,window")
    sp.add_argument("--N", default=N)
    sp.add_argument("--T", default=T)
    sp.add_argument("--tau", default=tau)
    sp.add_argument("--M", default="50,100,200,500,2000")
    sp.add_argument("--specs", default=None, help=specs_help)
    common.add_pool_args(sp)
    common.add_shard_args(sp)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m itpd.experiments stored_instances", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    fam = ap.add_subparsers(dest="family", required=True, metavar="{known_order,itpd_s}")
    ko = fam.add_parser("known_order", help="IAMB, lasso and ITPD + marginal-first on stored instances")
    _family_args(ko, ["oracle", "finite", "timing", "lasso_ebic_fixed"], "5,10,20", "4,8,16", "1,2,3",
                 "comma list of IAMB_MARGINAL_FIRST_SPECS names (finite mode); default all (use a new --out-dir)")
    ko.add_argument("--d", default="1,2,3")
    ko.add_argument("--cells", default=CELLS_DEFAULT)
    ko.add_argument("--no-lasso", action="store_true")
    ko.set_defaults(run=_known_order)
    sh = fam.add_parser("itpd_s", help="ITPD-S and ITPD-S+ on stored instances")
    _family_args(sh, ["oracle", "finite"], "10,20", "8,16", "1,2",
                 "comma list of SHRINK_FINITE_SPECS names (finite mode); default all")
    sh.add_argument("--graphs", type=int, default=20)
    sh.set_defaults(run=_shrink)
    a = ap.parse_args(argv)
    a.run(a)


if __name__ == "__main__":
    main()
