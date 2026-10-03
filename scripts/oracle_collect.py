"""Collect the oracle-run cell JSONs (itpd.run_oracle_counts) into markdown tables and a JSON of aggregates.

    python scripts/oracle_collect.py results/oracle [--out results/oracle/TABLES.md]
Numbers only. unique = distinct (unordered pair, sorted conditioning set); raw = every call. Headline rows are the
lazy evaluation; rows with suffix _nonlazy are the non-lazy (original code) evaluation. Skip counters are skipped Z4/Z8 step
issuances, not saved unique tests (saved unique = naive unique - ITPD unique). All ratios are paired per graph;
the table shows median [Q1, Q3] over graphs.
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from itpd.methods_registry import OLD_TO_NEW  # noqa: E402

H = ["itpd_naive", "itpd", "itpd_repo_variant", "order_based"]           # headline (lazy)
NL = ["itpd_naive_nonlazy", "itpd_nonlazy", "itpd_repo_variant_nonlazy"]
ALL = H + NL


def q(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return "-"
    a, b, c = np.percentile(x, [50, 25, 75])
    f = lambda v: f"{v:.0f}" if abs(v) >= 100 else (f"{v:.2f}" if abs(v) < 10 else f"{v:.1f}")
    return f"{f(a)} [{f(b)}, {f(c)}]"


def load(d):
    merged = {}
    for p in sorted(glob.glob(os.path.join(d, "N*.json"))):       # files of one cell (graph offsets) are merged
        c = json.load(open(p))
        cfg = c["config"]
        key = (cfg["N"], cfg["T"], cfg["tau"], cfg["d"])
        if key not in merged:
            merged[key] = (cfg, defaultdict(dict))
        for r in c["rows"]:
            merged[key][1][r["graph"]][OLD_TO_NEW.get(r["name"], r["name"])] = r
        merged[key][0]["graphs"] = len(merged[key][1])
    return [(cfg, [by[g] for g in sorted(by)]) for cfg, by in merged.values()]


def gather(cells, f):
    return [f(g) for _, graphs in cells for g in graphs]


def md(head, rows):
    return "\n".join(["| " + " | ".join(map(str, head)) + " |", "|" + "---|" * len(head)] +
                     ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def slope(xs, ys):
    return float(np.polyfit(np.log(np.asarray(xs, float)), np.log(np.asarray(ys, float)), 1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--out")
    a = ap.parse_args()
    cells = load(a.dir)
    if not cells:
        raise SystemExit("no cells")
    c0 = cells[0][0]
    ng = sum(len(g) for _, g in cells)
    L = [f"Conditions: oracle (d-separation on the full graph), S2, graph kind = {c0['graph_kind']}, process starts at "
         f"t=0 (the paper's t=1), candidates = full history (tau_max {c0['tau_max']}), cache = {c0['cache']}, "
         f"processing order = {c0['order']}, seed {c0['seed']}, {len(cells)} cells, {ng} graphs "
         f"({c0['graphs']} per cell). Headline rows = lazy evaluation, rows _nonlazy = non-lazy (as in the original code). "
         f"Median [Q1, Q3] over graphs."]
    agg = {}
    uq = lambda m: (lambda g: g[m]["unique"])
    ratio = lambda m, b: (lambda g: g[m]["unique"] / g[b]["unique"])

    # exact recovery
    rows = [[m, f"{sum(gather(cells, lambda g: g[m]['exact']))}/{ng}"] for m in ALL]
    L += ["\n### Exact recovery (graphs recovered exactly / graphs)\n", md(["method", "exact"], rows)]
    bad = [(c["N"], c["T"], c["tau"], c["d"], i, m) for c, gs in cells for i, g in enumerate(gs) for m in ALL
           if not g[m]["exact"]]
    L.append(f"\nInexact (N, T, tau, d, graph, method): {bad[:10] if bad else 'none'}. "
             f"With 0 misses in {ng} graphs the 95% upper bound on the miss rate is {3 / ng:.4f}.")

    # ITPD vs ITPD_naive
    L.append("\n### Unique tests, ITPD vs ITPD_naive (paired per graph)\n")
    rows = [["lazy (headline)", q(gather(cells, ratio("itpd", "itpd_naive"))), q(gather(cells, ratio("itpd_repo_variant", "itpd_naive"))),
             q(gather(cells, ratio("order_based", "itpd_naive")))],
            ["non-lazy", q(gather(cells, ratio("itpd_nonlazy", "itpd_naive_nonlazy"))), q(gather(cells, ratio("itpd_repo_variant_nonlazy", "itpd_naive_nonlazy"))),
             q(gather(cells, ratio("order_based", "itpd_naive_nonlazy")))]]
    L.append(md(["evaluation", "itpd / itpd_naive", "itpd_repo_variant / itpd_naive", "order / itpd_naive"], rows))
    rows = []
    for key in ("N", "T", "tau", "d"):
        for v in sorted({c[key] for c, _ in cells}):
            sub = [(c, gs) for c, gs in cells if c[key] == v]
            rows.append([f"{key}={v:g}", q(gather(sub, ratio("itpd", "itpd_naive"))),
                         q(gather(sub, ratio("itpd_nonlazy", "itpd_naive_nonlazy"))),
                         q(gather(sub, lambda g: g["itpd_naive"]["unique"] - g["itpd"]["unique"])),
                         q(gather(sub, lambda g: g["itpd_naive_nonlazy"]["unique"] - g["itpd_nonlazy"]["unique"]))])
    L += ["\nBy factor (other factors pooled):\n",
          md(["", "itpd/naive lazy", "itpd/naive non-lazy", "saved unique lazy", "saved unique non-lazy"], rows)]

    def size_sum(m, lo, hi):
        return sum(v for _, gs in cells for g in gs for k, v in g[m]["by_size_unique"].items() if lo <= int(k) <= hi)
    for tag, mm in (("lazy", H), ("non-lazy", ["itpd_naive_nonlazy", "itpd_nonlazy", "itpd_repo_variant_nonlazy", "order_based"])):
        rows = []
        for name, lo, hi in (("0", 0, 0), ("1", 1, 1), (">=2", 2, 10 ** 9)):
            v = [size_sum(m, lo, hi) for m in mm]
            rows.append([name] + v + [v[0] - v[1], v[0] - v[2]])
        L += [f"\nUnique tests by conditioning-set size, summed over all graphs ({tag}):\n",
              md(["|S|"] + mm + ["naive - itpd", "naive - itpd_repo_variant"], rows)]
    d2 = sum(1 for _, gs in cells for g in gs for a_, b_ in (("itpd", "itpd_naive"), ("itpd_nonlazy", "itpd_naive_nonlazy"))
             if {k: v for k, v in g[a_]["by_size_unique"].items() if int(k) >= 2} !=
             {k: v for k, v in g[b_]["by_size_unique"].items() if int(k) >= 2})
    L.append(f"\nGraph-mode pairs where ITPD and ITPD_naive differ in the unique count at some size >= 2: {d2} of {2 * ng}.")
    for tag, m in (("lazy", "itpd"), ("non-lazy", "itpd_nonlazy")):
        keys = sorted({k for _, gs in cells for g in gs for k in g[m]["skips"]})
        rows = [[k, q(gather(cells, lambda g, k=k: g[m]["skips"].get(k, 0)))] for k in keys]
        L += [f"\nRule counters per graph, ITPD paper variant, {tag} (skipped Z4/Z8 step issuances; memo = marginal results "
              f"reused inside one PaDL call):\n", md(["counter (partition:rule)", "per graph"], rows)]
    rows = [[m, q(gather(cells, lambda g, m=m: g[m]["raw"])), q(gather(cells, uq(m))),
             q(gather(cells, lambda g, m=m: g[m]["raw"] - g[m]["unique"]))] for m in ALL]
    L += ["\nRaw calls beside unique tests (repeats = raw - unique, all marginal repeats across pairs):\n",
          md(["method", "raw", "unique", "repeats"], rows)]

    # tests per candidate
    L.append("\n### Unique tests per candidate (candidates = sum over targets; order-based is 1.00 by construction)\n")
    per = lambda m: (lambda g: g[m]["unique"] / g[m]["n_cand"])
    rows = [[m, q(gather(cells, per(m))), q(gather(cells, lambda g, m=m: g[m]["mean_target_max_frac"]))] for m in ALL]
    L.append(md(["method", "unique per candidate", "largest conditioning set / candidates (mean over targets)"], rows))
    rows = []
    for key in ("T", "N"):
        for v in sorted({c[key] for c, _ in cells}):
            sub = [(c, gs) for c, gs in cells if c[key] == v]
            rows.append([f"{key}={v}"] + [q(gather(sub, per(m))) for m in H + ["itpd_naive_nonlazy", "itpd_nonlazy"]])
    L += ["\n", md(["unique per candidate"] + H + ["itpd_naive_nonlazy", "itpd_nonlazy"], rows)]

    # scaling
    L.append("\n### Scaling\n")
    cellmed = {}
    rows = []
    for N in sorted({c["N"] for c, _ in cells}):
        for T in sorted({c["T"] for c, _ in cells}):
            sub = [(c, gs) for c, gs in cells if c["N"] == N and c["T"] == T]
            if not sub:
                continue
            cand = np.median(gather(sub, lambda g: g["order_based"]["n_cand"]))
            meds = {m: float(np.median(gather(sub, uq(m)))) for m in ALL}
            frac = {m: float(np.median(gather(sub, lambda g, m=m: g[m]["mean_target_max_frac"]))) for m in ("itpd_naive", "itpd")}
            cellmed[(N, T)] = (cand, meds)
            rows.append([N, T, f"{cand:.0f}"] + [f"{meds[m]:.0f}" for m in ALL] +
                        [f"{meds['itpd'] / meds['itpd_naive']:.3f}", f"{meds['itpd_nonlazy'] / meds['itpd_naive_nonlazy']:.3f}",
                         f"{frac['itpd']:.2f}"])
    L += ["Median unique tests per graph by (N, T), tau and d pooled:\n",
          md(["N", "T", "candidates"] + ALL + ["itpd/naive lazy", "itpd/naive non-lazy", "largest set / candidates (itpd)"], rows)]
    Ns, Ts = sorted({k[0] for k in cellmed}), sorted({k[1] for k in cellmed})
    rows, agg["slopes"] = [], {}
    for m in ALL:
        sN = [slope(Ns, [cellmed[(N, T)][1][m] for N in Ns]) for T in Ts if all((N, T) in cellmed for N in Ns)]
        sT = [slope(Ts, [cellmed[(N, T)][1][m] for T in Ts]) for N in Ns if all((N, T) in cellmed for T in Ts)]
        sC = slope([v[0] for v in cellmed.values()], [v[1][m] for v in cellmed.values()])
        rows.append([m, ", ".join(f"{s:.2f}" for s in sN), ", ".join(f"{s:.2f}" for s in sT), f"{sC:.2f}"])
        agg["slopes"][m] = {"vs_N_by_T": sN, "vs_T_by_N": sT, "vs_candidates": sC}
    L += [f"\nLog-log slopes of the per-(N, T) median of unique tests: vs N (one per T in {Ts}), vs T (one per N in {Ns}), "
          f"vs candidates (all cells; the candidate count is exact):\n", md(["method", "vs N", "vs T", "vs candidates"], rows)]
    rows = []
    for key in ("tau", "d"):
        for v in sorted({c[key] for c, _ in cells}):
            sub = [(c, gs) for c, gs in cells if c[key] == v]
            rows.append([f"{key}={v:g}"] + [q(gather(sub, uq(m))) for m in H])
    L += ["\nUnique tests by tau and d (other factors pooled):\n", md([""] + H, rows)]
    rows = [[f"T={T}"] + [q(gather([(c, gs) for c, gs in cells if c["T"] == T], lambda g, m=m: g[m]["mean_target_max_frac"]))
                          for m in ("itpd_naive", "itpd", "order_based")] for T in Ts]
    L += ["\nLargest conditioning set per target as a share of its candidates, by T:\n",
          md(["", "itpd_naive", "itpd", "order_based"], rows)]
    big = [(c, gs) for c, gs in cells if c["N"] == Ns[-1] and c["T"] == Ts[-1]]
    rows = [[m, q(gather(big, lambda g, m=m: g[m]["seconds"])),
             q(gather(big, lambda g, m=m: 1e6 * g[m]["seconds"] / max(g[m]["raw"], 1)))] for m in ALL]
    L += [f"\nWall-clock (oracle; not comparable across methods), cells with N={Ns[-1]}, T={Ts[-1]}:\n",
          md(["method", "seconds per graph", "us per raw call"], rows)]
    # realised degree
    L.append("\nRealised mean non-self out-degree (average over all nodes with t <= T-2): " +
             q(gather(cells, lambda g: g["order_based"]["graph_stats"]["mean_out_degree_nonself"])))

    # tests when a time step is appended
    L.append("\n### Tests when a time step is appended (order = time; new unique tests summed over the N targets of step t)\n")
    Tmax = Ts[-1]
    for N in Ns:
        sub = [(c, gs) for c, gs in cells if c["N"] == N and c["T"] == Tmax]
        if not sub:
            continue
        rows = []
        for t in range(1, Tmax):
            f = lambda m, i: gather(sub, lambda g: g[m]["per_step"][str(t)][i])
            rows.append([t, q(f("order_based", 2))] + [q(f(m, 0)) for m in ("itpd_naive", "itpd", "order_based")] +
                        [f"{np.median(f('itpd_naive', 0)) / N:.1f}", f"{np.median(f('itpd', 0)) / N:.1f}"])
        L += [f"\nN={N}, T={Tmax} (tau, d pooled), t = index of the appended step:\n",
              md(["t", "candidates (all N targets)", "itpd_naive", "itpd", "order_based", "itpd_naive per target", "itpd per target"], rows)]
    txt = "\n".join(L) + "\n"
    out = a.out or os.path.join(a.dir, "TABLES.md")
    open(out, "w").write(txt)
    json.dump(agg, open(os.path.join(a.dir, "aggregates.json"), "w"))
    print(txt)


if __name__ == "__main__":
    main()
