"""Oracle runs: exactness and test counts of the ITPD family and the full-conditioning baseline on random S2 graphs, one cell (N, T, tau, d)
per job, one summary JSON per job.

    python -m itpd.run_oracle_counts --N 20 --T 16 --tau 2 --d 2 --graphs 20 --seed 0 --out results/oracle/N20_T16_tau2_d2.json
        [--graph time|window] [--tau-max none|tau|INT] [--order time|variable] [--instances-dir DIR [--with-weights]]

Per graph: a random graph (process starts at t = 0 with roots, self edge V^n_t -> V^n_{t+1}, no contemporaneous edges),
then the methods below, each with the exact d-separation oracle on the FULL graph and one run-wide cache:
  itpd_naive, itpd (paper variant), itpd_repo_variant (step 4 of the original code: extra marginal conjunct)   headline: LAZY evaluation
  itpd_naive_nonlazy, itpd_nonlazy, itpd_repo_variant_nonlazy                                  non-lazy (as in the original code) beside it
  full_conditioning                                                                                  one test per candidate
Lazy = a test is issued only when its result can change the label (a step's second test is skipped when the first
already decides it); non-lazy = both tests of every step, as in the original code. Both give the same graphs.
Candidates: full history (--tau-max none), the DGP lag (--tau-max tau) or an integer.
JSON: {"config", "rows"}; one row per graph and method with exact/fp/fn, raw and unique counts, by-size and by-rule
histograms, largest conditioning set per target, rule skip counters, per_step (new unique tests, raw calls and
candidates when time step t is appended, with --order time), seconds, graph_stats (incl. sha1 of the graph).
Instances (--instances-dir): one npz per graph, format documented in itpd/instances.py, shared with the baselines.
An existing output file is not recomputed (resume = rerun the same command). Result files written before the rename carry the
pre-rename row names (methods_registry.OLD_TO_NEW); `--methods` accepts both.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from . import method_runner, sim
from .graphs import TimeGraph, check_assumptions, unroll
from .instances import graph_hash, save_instance
from .methods_registry import OLD_TO_NEW, spec

# (row name, method, PaDL variant, lazy), built from methods_registry.spec. Headline = lazy; the non-lazy count is always beside it.
ORACLE_SPECS = tuple(spec(n, ())[:4] for n in ("itpd_naive", "itpd", "itpd_repo_variant", "itpd_naive_nonlazy", "itpd_nonlazy",
                                                 "itpd_repo_variant_nonlazy", "full_conditioning"))
# selectable with --methods, not part of the default set
EXTRA_ORACLE_SPECS = tuple(spec(n, ())[:4] for n in ("itpd_adjacency_self", "itpd_adjacency_self_nonlazy"))


def make_instance(N, T, tau, d, seed, g, kind, with_weights=False):
    rng = np.random.default_rng(np.random.SeedSequence([seed, N, T, tau, int(d * 10), g]))
    if with_weights:
        r = sim.simulate_s2(N, T, 0, d, tau, rng, graph=kind)
        return r.graph, r.W
    if kind == "time":
        return sim.sample_time_graph(N, T, d, tau, rng), None
    B = sim.sample_window_graph(N, tau, d, rng)
    return TimeGraph(A=unroll(B, T), N=N, T=T, tau=tau, B=B, meta={"graph": "window"}), None


def select_methods(names: str | None):
    if not names:
        return ORACLE_SPECS
    want = [OLD_TO_NEW.get(n.strip(), n.strip()) for n in names.split(",")]
    allm = ORACLE_SPECS + EXTRA_ORACLE_SPECS
    sel = tuple(m for m in allm if m[0] in want)
    if len(sel) != len(want):
        raise SystemExit(f"unknown method in {want}; choose from {[m[0] for m in allm]}")
    return sel


def run_cell(N: int, T: int, tau: int, d: float, graphs: int, seed: int, tau_max, methods=ORACLE_SPECS,
             order: str = "time", kind: str = "time", instances_dir: str | None = None,
             with_weights: bool = False, offset: int = 0) -> dict:
    rows = []
    t0 = time.time()
    for g in range(offset, offset + graphs):
        graph, W = make_instance(N, T, tau, d, seed, g, kind, with_weights)
        chk = check_assumptions(graph)
        assert chk["autocorr_present"] and chk["no_contemporaneous"] and chk["roots_at_t0"]
        h = graph_hash(graph.A)
        if instances_dir:
            save_instance(os.path.join(instances_dir, f"N{N}_T{T}_tau{tau}_d{d:g}_{kind}_g{g}.npz"), graph, W=W, d=d,
                          seed=[seed, N, T, tau, int(d * 10), g], index=g, kind=kind)
        tm = tau if tau_max == "tau" else tau_max
        gstat = {"edges": int(graph.A.sum()), "max_in_degree": int(graph.A.sum(0).max()),
                 "max_lag": chk["max_lag"], "sha1": h,
                 "mean_out_degree_nonself": float((graph.A.sum(1)[: (T - 1) * N]).mean() - 1.0)}
        for name, method, variant, lazy in methods:
            o = method_runner.run_s2_method(method, None, graph=graph, tau_max=tm, variant=variant or "paper",
                                     order=order, lazy=lazy)
            res_pairs = o.pop("per_pair")
            t = o["tests"]
            ms = [p["max_size"] for p in res_pairs]
            nc = [p["n_cand"] for p in res_pairs]
            step = {}
            for pr in res_pairs:                       # tests when step t is appended (order = time)
                tt = pr["pair"][1] // N
                a_ = step.setdefault(tt, [0, 0, 0])
                a_[0] += pr["new_unique"]; a_[1] += pr["raw"]; a_[2] += pr["n_cand"]
            rows.append({"graph": g, "name": name, "lazy": lazy, "exact": o["metrics"]["exact"],
                         "fp": o["metrics"]["fp"], "fn": o["metrics"]["fn"], "raw": t["raw_calls"],
                         "unique": t["unique_tests"], "n_cand": sum(nc), "by_size_unique": t["by_size_unique"],
                         "by_size_raw": t["by_size_raw"], "by_label_unique": t["by_label_unique"],
                         "by_label_raw": t["by_label_raw"], "max_size": max(ms),
                         "mean_target_max_size": float(np.mean(ms)),
                         "mean_target_max_frac": float(np.mean([m / c for m, c in zip(ms, nc) if c])),
                         "per_step": {str(k): v for k, v in sorted(step.items())},
                         "skips": o["skips"], "seconds": o["seconds"], "graph_stats": gstat})
    return {"config": {"N": N, "T": T, "tau": tau, "d": d, "graphs": graphs, "seed": seed, "tau_max": tau_max,
                       "order": order, "graph_kind": kind, "offset": offset, "ci": "oracle", "cache": "run",
                       "headline_lazy": True, "methods": [m[0] for m in methods]},
            "rows": rows, "seconds_total": time.time() - t0}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--T", type=int, required=True)
    ap.add_argument("--tau", type=int, required=True)
    ap.add_argument("--d", type=float, required=True)
    ap.add_argument("--graphs", type=int, default=20)
    ap.add_argument("--methods", default=None, help="comma list of row names (default all): " + ",".join(m[0] for m in ORACLE_SPECS))
    ap.add_argument("--offset", type=int, default=0, help="first graph index (split a big cell over several jobs; file names must differ)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tau-max", default="none", help="none | tau | integer")
    ap.add_argument("--order", default="time", choices=["variable", "time"],
                    help="processing order; time = targets appended step by step (needed for the per-step counts)")
    ap.add_argument("--graph", default="time", choices=["time", "window"],
                    help="time: per-source edges (nonstationary structure); window: stationary lag graph")
    ap.add_argument("--instances-dir", default=None, help="write each instance as .npz (format: itpd/instances.py)")
    ap.add_argument("--with-weights", action="store_true", help="also store SCM weights W in the instances (slower)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        print("exists, skipping", a.out)
        return
    tm = None if a.tau_max == "none" else ("tau" if a.tau_max == "tau" else int(a.tau_max))
    res = run_cell(a.N, a.T, a.tau, a.d, a.graphs, a.seed, tm, order=a.order, kind=a.graph,
                   instances_dir=a.instances_dir, with_weights=a.with_weights, offset=a.offset,
                   methods=select_methods(a.methods))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f)
    os.replace(tmp, a.out)
    print("wrote", a.out, "seconds", round(res["seconds_total"], 1))


if __name__ == "__main__":
    main()
