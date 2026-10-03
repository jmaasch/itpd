"""Run the oracle-run grid cells in parallel inside one allocation, one summary JSON per cell (resumable).

    python scripts/oracle_grid.py --out-dir results/oracle/window --graph window --workers 16 --budget-sec 780
Defaults: N 5,10,20; T 4,8,16; tau 1,2,3; d 1,2,3; 20 graphs; tau-max none. Cells whose JSON exists are skipped;
no new cell is started when it would not finish inside --budget-sec (so a short job can be repeated until
`remaining 0`). For the full-scale run pass e.g. --N 40,80 --T 32,50 and a longer budget.
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from itpd import run_oracle_counts  # noqa: E402


def cell_path(out_dir, N, T, tau, d):
    return os.path.join(out_dir, f"N{N}_T{T}_tau{tau}_d{d:g}.json")


def est_seconds(N, T, graphs):
    return graphs * 12.0 * (N * T / 320.0) ** 2.2 * 1.3 + 2


def work(args):
    N, T, tau, d, graphs, seed, tm, out, kind, inst, methods = args
    run_oracle_counts.main((["--instances-dir", inst] if inst else []) + (["--methods", methods] if methods else []) + ["--N", str(N), "--T", str(T), "--tau", str(tau), "--d", str(d), "--graphs", str(graphs),
             "--seed", str(seed), "--tau-max", tm, "--graph", kind, "--out", out])
    return out


def main():
    ap = argparse.ArgumentParser()
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
    ap.add_argument("--methods", default=None, help="comma list of row names of itpd.run_oracle_counts.ORACLE_SPECS (default all; old names accepted)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--budget-sec", type=float, default=780)
    a = ap.parse_args()
    ints = lambda s: [int(x) for x in s.split(",")]
    cells = [(N, T, tau, float(d)) for N in ints(a.N) for T in ints(a.T) for tau in ints(a.tau) for d in ints(a.d)]
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
    jobs = [(N, T, tau, d, a.graphs, a.seed, a.tau_max, cell_path(a.out_dir, N, T, tau, d), a.graph, a.instances_dir, a.methods)
            for N, T, tau, d in chosen]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for f in as_completed([ex.submit(work, j) for j in jobs]):
            f.result()
    left = len([c for c in cells if not os.path.exists(cell_path(a.out_dir, *c))])
    print(f"done in {time.time() - t0:.0f}s; remaining {left}", flush=True)


if __name__ == "__main__":
    main()
