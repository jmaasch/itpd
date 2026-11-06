"""Oracle runs: exactness and test counts of the ITPD family and the full-conditioning baseline on random S2 graphs, one cell (N, T, tau, d)
per job, one summary JSON per job.

    python -m itpd.experiments oracle_counts cell --N 20 --T 16 --tau 2 --d 2 --graphs 20 --seed 0 --out results/oracle/N20_T16_tau2_d2.json
        [--graph time|window] [--tau-max none|tau|INT] [--order time|variable] [--instances-dir DIR [--with-weights]]
    python -m itpd.experiments oracle_counts grid --out-dir results/oracle/window --graph window --workers 16 --budget-sec 780
        [--N 5,10,20 --T 4,8,16 --tau 1,2,3 --d 1,2,3 --graphs 20 --seed 0 --tau-max none --instances-dir DIR --methods ...]
    python -m itpd.experiments oracle_counts large OUT_DIR --cells "80x25,40x50" --graphs 0-4 --arms window
        [--methods itpd_naive,itpd,full_conditioning --workers 16 --start-by-sec 480 --task-timeout 780 --oracle fast]

cell: per graph, a random graph (process starts at t = 0 with roots, self edge V^n_t -> V^n_{t+1}, no contemporaneous edges),
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

grid: the cells of the product N x T x tau x d in parallel inside one allocation, one summary JSON per cell
OUT_DIR/N<N>_T<T>_tau<tau>_d<d>.json (the files of `cell`; resumable). Defaults: N 5,10,20; T 4,8,16; tau 1,2,3; d 1,2,3; 20 graphs;
tau-max none. Cells whose JSON exists are skipped; no new cell is started when it would not finish inside --budget-sec (so a short
job can be repeated until `remaining 0`). For the full-scale run pass e.g. --N 40,80 --T 32,50 and a longer budget.

large: tasks of large cells, one process per (arm, N, T, graph, method), --workers at a time (resumable, time-boxed). A cell is NxT
(tau 1, d 2, full history, lazy headline). Task output: OUT_DIR/<arm>/N<N>_T<T>_g<g>_<method>.json (one graph per file via `cell
--graphs 1 --offset g`; instance: OUT_DIR/instances/<arm>/...). Existing files are skipped; no task is started after --start-by-sec
(so a short job can be repeated until "remaining 0"). Biggest cells first. Prints one summary line at the end.
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .. import method_runner, sim
from ..graphs import TimeGraph, check_assumptions, unroll
from ..instances import graph_hash, save_instance
from ..methods_registry import OLD_TO_NEW, spec
from . import common

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


def parse_tau_max(s: str):
    return None if s == "none" else ("tau" if s == "tau" else int(s))


def run_cell_file(out: str, methods: str | None = None, **kw) -> tuple:
    """Compute one cell (keywords of `run_cell`; `methods` is a comma list of row names) and write it to `out`, unless `out` exists."""
    if os.path.exists(out):
        print("exists, skipping", out)
        return out, "exists", 0.0
    res = run_cell(methods=select_methods(methods), **kw)
    common.dump_json(out, res)
    print("wrote", out, "seconds", round(res["seconds_total"], 1))
    return out, "done", res["seconds_total"]


# ------------------------------------------------------------------------------------------------ cell

def _cell(a):
    run_cell_file(a.out, methods=a.methods, N=a.N, T=a.T, tau=a.tau, d=a.d, graphs=a.graphs, seed=a.seed,
                  tau_max=parse_tau_max(a.tau_max), order=a.order, kind=a.graph, instances_dir=a.instances_dir,
                  with_weights=a.with_weights, offset=a.offset)


def _cell_args(ap):
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
    ap.set_defaults(run=_cell)


# ------------------------------------------------------------------------------------------------ grid

def cell_path(out_dir, N, T, tau, d):
    return os.path.join(out_dir, f"N{N}_T{T}_tau{tau}_d{d:g}.json")


def est_seconds(N, T, graphs):
    return graphs * 12.0 * (N * T / 320.0) ** 2.2 * 1.3 + 2


def _grid_task(job):
    out, kw = job
    return run_cell_file(out, **kw)


def _grid(a):
    cells = [(N, T, tau, float(d)) for N in common.ints(a.N) for T in common.ints(a.T) for tau in common.ints(a.tau)
             for d in common.ints(a.d)]
    todo = [c for c in cells if not os.path.exists(cell_path(a.out_dir, *c))]
    todo.sort(key=lambda c: -est_seconds(c[0], c[1], a.graphs))
    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()
    # greedy schedule on `workers` virtual machines using the cost estimate
    load = [0.0] * a.workers
    chosen = []
    for c in todo:
        e = est_seconds(c[0], c[1], a.graphs)
        i = min(range(a.workers), key=lambda k: load[k])
        if load[i] + e <= a.budget_sec:
            load[i] += e
            chosen.append(c)
    print(f"cells total {len(cells)} todo {len(todo)} chosen {len(chosen)}", flush=True)
    jobs = [(cell_path(a.out_dir, *c), dict(N=c[0], T=c[1], tau=c[2], d=c[3], graphs=a.graphs, seed=a.seed,
                                            tau_max=parse_tau_max(a.tau_max), kind=a.graph, instances_dir=a.instances_dir,
                                            methods=a.methods)) for c in chosen]
    common.run_tasks(_grid_task, jobs, a.workers, math.inf)
    left = len([c for c in cells if not os.path.exists(cell_path(a.out_dir, *c))])
    common.finish(t0, left)


def _grid_args(ap):
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--N", default="5,10,20")
    ap.add_argument("--T", default="4,8,16")
    ap.add_argument("--tau", default="1,2,3")
    ap.add_argument("--d", default="1,2,3")
    ap.add_argument("--graphs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tau-max", default="none")
    ap.add_argument("--graph", default="time", choices=["time", "window"])
    ap.add_argument("--instances-dir", default=None)
    ap.add_argument("--methods", default=None, help="comma list of row names of ORACLE_SPECS (default all; old names accepted)")
    common.add_pool_args(ap, workers=16, budget_sec=780)
    ap.set_defaults(run=_grid)


# ------------------------------------------------------------------------------------------------ large

def _graph_indices(s):
    if "-" in s:
        lo, hi = s.split("-")
        return list(range(int(lo), int(hi) + 1))
    return common.ints(s)


def _large(a):
    t0 = time.time()
    tasks = []
    for arm in a.arms.split(","):
        for c in a.cells.split(","):
            N, T = (int(x) for x in c.split("x"))
            for g in _graph_indices(a.graphs):
                for m in a.methods.split(","):
                    m = OLD_TO_NEW.get(m, m)
                    path = os.path.join(a.out, arm, f"N{N}_T{T}_g{g}_{m}.json")
                    tasks.append((N * T, arm, N, T, g, m, path))

    def done(t):
        # a task is done when its file exists under the current or an earlier name of the method (files of runs started before the rename)
        return any(os.path.exists(t[-1].replace(f"_{t[5]}.json", f"_{old}.json")) for old in [t[5]] + [k for k, v in OLD_TO_NEW.items() if v == t[5]])

    tasks = [t for t in tasks if not done(t)]
    tasks.sort(key=lambda t: (-t[0], t[4]))
    print(f"tasks todo {len(tasks)}", flush=True)
    env = dict(os.environ, ITPD_ORACLE=a.oracle, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")

    def run(t):
        _, arm, N, T, g, m, path = t
        if time.time() - t0 > a.start_by_sec:
            return "skipped"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cmd = [sys.executable, "-m", "itpd.experiments", "oracle_counts", "cell", "--N", str(N), "--T", str(T), "--tau", "1", "--d", "2",
               "--graphs", "1", "--offset", str(g), "--graph", arm, "--methods", m,
               "--instances-dir", os.path.join(a.out, "instances", arm), "--out", path]
        try:
            r = subprocess.run(cmd, env=env, timeout=a.task_timeout, capture_output=True, text=True)
            if r.returncode:
                print("FAIL", path, r.stderr[-300:], flush=True)
                return "fail"
        except subprocess.TimeoutExpired:
            print("TIMEOUT", path, flush=True)
            return "timeout"
        return "done"

    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(run, tasks))
    left = sum(1 for t in tasks if not done(t))
    print(f"done {res.count('done')} skipped {res.count('skipped')} fail {res.count('fail')} timeout {res.count('timeout')} in {time.time() - t0:.0f}s; remaining {left}", flush=True)


def _large_args(ap):
    ap.add_argument("out")
    ap.add_argument("--cells", required=True)
    ap.add_argument("--graphs", default="0-4")
    ap.add_argument("--arms", default="window")
    ap.add_argument("--methods", default="itpd_naive,itpd,full_conditioning")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--start-by-sec", type=float, default=480)
    ap.add_argument("--task-timeout", type=float, default=780)
    ap.add_argument("--oracle", default="fast")
    ap.set_defaults(run=_large)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m itpd.experiments oracle_counts", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = ap.add_subparsers(dest="mode", required=True, metavar="{cell,grid,large}")
    for name, add_args, text in (("cell", _cell_args, "one cell (N, T, tau, d): one summary JSON"),
                                 ("grid", _grid_args, "all cells of a grid in parallel: one summary JSON per cell"),
                                 ("large", _large_args, "tasks of large cells: one process per (arm, N, T, graph, method)")):
        add_args(modes.add_parser(name, help=text, description=text))
    a = ap.parse_args(argv)
    a.run(a)


if __name__ == "__main__":
    main()
