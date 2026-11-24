"""Tables of the oracle runs (the cell JSONs of `itpd.experiments oracle_counts`): the grid cells, the large cells, the scaling slopes.

    python -m itpd.tables oracle_counts cells DIR [--out DIR/TABLES.md]
    python -m itpd.tables oracle_counts large BIG_DIR --oracle ORACLE_DIR [--arm window] [--out BIG_DIR/TABLES-<arm>.md]
    python -m itpd.tables oracle_counts slopes DIR [--out DIR/SLOPES.md] [--boot 2000] [--seed 0]

cells: numbers only. unique = distinct (unordered pair, sorted conditioning set); raw = every call. Headline rows are the
lazy evaluation; rows with suffix _nonlazy are the non-lazy (original code) evaluation. Skip counters are skipped Z4/Z8 step
issuances, not saved unique tests (saved unique = naive unique - ITPD unique). All ratios are paired per graph;
the table shows median [Q1, Q3] over graphs. Writes TABLES.md and aggregates.json in DIR.

large: per-(cell, graph, method) JSONs of `oracle_counts large` (3 methods) plus the small oracle-run cells at tau 1, d 2.
BIG_DIR/<arm>/N*_T*_g*_<method>.json (the glob takes files of both spellings of a method, `_order.json` and `_order_based.json`);
ORACLE_DIR = the results of the small cells of the same arm (cells N*_T*_tau1_d2.json, graphs 0..19, all methods, same instances:
graph index g and the seed convention are shared, so graph g of a big cell is a different graph from every small cell, the sha1
says which). Numbers only. Headline = lazy unique tests; ratios paired per graph (graphs where all three methods finished).

slopes: log-log slopes with bootstrap confidence intervals over graphs, from the saved oracle-run cells.
Each oracle-run cell (N, T, tau, d) holds 20 independent random graphs. Graph g of one cell is not the same graph as graph g
of another cell, so there is no per-graph slope across N or T. What is estimated instead:
 (a) cell-median slopes: per (N, T) the median over all graphs of the (tau, d) cells, then the
     OLS slope of log(median) on log(N) at fixed T (3 levels), on log(T) at fixed N (3 levels), and on log(candidates)
     over the 9 (N, T) groups; the CI is a percentile bootstrap that resamples graphs WITHIN each (N, T, tau, d) cell
     (same resample for all methods, so the CIs are paired) and refits;
 (b) graph-level slope: OLS of log(unique) on log(candidates) over all graphs (1,620 per arm), same bootstrap;
 (c) tests per candidate (unique / candidates) against N, T and candidates: the slope of log(unique / candidates)
     (0 = tests proportional to candidates);
 (d) the paired ratio itpd / naive: median over graphs per (N, T), slope of log(ratio) on log(N), log(T), log(candidates)
     (0 = the ratio does not drift), same bootstrap.
Only 3 levels of N and of T exist in the default grid: slope CIs are wide for that reason. Numbers only.
Writes SLOPES.md and SLOPES.json.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

from .common import CORE, LAZY, md, new_name, q, slope

NL = ["itpd_naive_nonlazy", "itpd_nonlazy", "itpd_repo_variant_nonlazy"]
ALL = LAZY + NL
SLOPE_METHODS = ["itpd_naive", "itpd", "itpd_repo_variant", "full_conditioning", "itpd_naive_nonlazy", "itpd_nonlazy"]
LARGE_BINS = [(0, 0, "0"), (1, 1, "1"), (2, 4, "2-4"), (5, 19, "5-19"), (20, 99, "20-99"), (100, 499, "100-499"), (500, 10 ** 9, ">=500")]


# ------------------------------------------------------------------------------------------------ cells

def load_cells(d):
    merged = {}
    for p in sorted(glob.glob(os.path.join(d, "N*.json"))):       # files of one cell (graph offsets) are merged
        c = json.load(open(p))
        cfg = c["config"]
        key = (cfg["N"], cfg["T"], cfg["tau"], cfg["d"])
        if key not in merged:
            merged[key] = (cfg, defaultdict(dict))
        for r in c["rows"]:
            merged[key][1][r["graph"]][new_name(r["name"])] = r
        merged[key][0]["graphs"] = len(merged[key][1])
    return [(cfg, [by[g] for g in sorted(by)]) for cfg, by in merged.values()]


def gather(cells, f):
    return [f(g) for _, graphs in cells for g in graphs]


def collect_cells(a):
    cells = load_cells(a.dir)
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
             q(gather(cells, ratio("full_conditioning", "itpd_naive")))],
            ["non-lazy", q(gather(cells, ratio("itpd_nonlazy", "itpd_naive_nonlazy"))), q(gather(cells, ratio("itpd_repo_variant_nonlazy", "itpd_naive_nonlazy"))),
             q(gather(cells, ratio("full_conditioning", "itpd_naive_nonlazy")))]]
    L.append(md(["evaluation", "itpd / itpd_naive", "itpd_repo_variant / itpd_naive", "full conditioning / itpd_naive"], rows))
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
    for tag, mm in (("lazy", LAZY), ("non-lazy", ["itpd_naive_nonlazy", "itpd_nonlazy", "itpd_repo_variant_nonlazy", "full_conditioning"])):
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
    L.append("\n### Unique tests per candidate (candidates = sum over targets; full conditioning is 1.00 by construction)\n")
    per = lambda m: (lambda g: g[m]["unique"] / g[m]["n_cand"])
    rows = [[m, q(gather(cells, per(m))), q(gather(cells, lambda g, m=m: g[m]["mean_target_max_frac"]))] for m in ALL]
    L.append(md(["method", "unique per candidate", "largest conditioning set / candidates (mean over targets)"], rows))
    rows = []
    for key in ("T", "N"):
        for v in sorted({c[key] for c, _ in cells}):
            sub = [(c, gs) for c, gs in cells if c[key] == v]
            rows.append([f"{key}={v}"] + [q(gather(sub, per(m))) for m in LAZY + ["itpd_naive_nonlazy", "itpd_nonlazy"]])
    L += ["\n", md(["unique per candidate"] + LAZY + ["itpd_naive_nonlazy", "itpd_nonlazy"], rows)]

    # scaling
    L.append("\n### Scaling\n")
    cellmed = {}
    rows = []
    for N in sorted({c["N"] for c, _ in cells}):
        for T in sorted({c["T"] for c, _ in cells}):
            sub = [(c, gs) for c, gs in cells if c["N"] == N and c["T"] == T]
            if not sub:
                continue
            cand = np.median(gather(sub, lambda g: g["full_conditioning"]["n_cand"]))
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
            rows.append([f"{key}={v:g}"] + [q(gather(sub, uq(m))) for m in LAZY])
    L += ["\nUnique tests by tau and d (other factors pooled):\n", md([""] + LAZY, rows)]
    rows = [[f"T={T}"] + [q(gather([(c, gs) for c, gs in cells if c["T"] == T], lambda g, m=m: g[m]["mean_target_max_frac"]))
                          for m in ("itpd_naive", "itpd", "full_conditioning")] for T in Ts]
    L += ["\nLargest conditioning set per target as a share of its candidates, by T:\n",
          md(["", "itpd_naive", "itpd", "full_conditioning"], rows)]
    big = [(c, gs) for c, gs in cells if c["N"] == Ns[-1] and c["T"] == Ts[-1]]
    rows = [[m, q(gather(big, lambda g, m=m: g[m]["seconds"])),
             q(gather(big, lambda g, m=m: 1e6 * g[m]["seconds"] / max(g[m]["raw"], 1)))] for m in ALL]
    L += [f"\nWall-clock (oracle; not comparable across methods), cells with N={Ns[-1]}, T={Ts[-1]}:\n",
          md(["method", "seconds per graph", "us per raw call"], rows)]
    # realised degree
    L.append("\nRealised mean non-self out-degree (average over all nodes with t <= T-2): " +
             q(gather(cells, lambda g: g["full_conditioning"]["graph_stats"]["mean_out_degree_nonself"])))

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
            rows.append([t, q(f("full_conditioning", 2))] + [q(f(m, 0)) for m in ("itpd_naive", "itpd", "full_conditioning")] +
                        [f"{np.median(f('itpd_naive', 0)) / N:.1f}", f"{np.median(f('itpd', 0)) / N:.1f}"])
        L += [f"\nN={N}, T={Tmax} (tau, d pooled), t = index of the appended step:\n",
              md(["t", "candidates (all N targets)", "itpd_naive", "itpd", "full_conditioning", "itpd_naive per target", "itpd per target"], rows)]
    txt = "\n".join(L) + "\n"
    out = a.out or os.path.join(a.dir, "TABLES.md")
    open(out, "w").write(txt)
    json.dump(agg, open(os.path.join(a.dir, "aggregates.json"), "w"))
    print(txt)



# ------------------------------------------------------------------------------------------------ large cells

def load_rows(paths, src):
    out = defaultdict(dict)       # (N, T) -> graph -> method -> row (+ src)
    for p in paths:
        try:
            c = json.load(open(p))
        except Exception:
            continue
        cfg = c["config"]
        if cfg["tau"] != 1 or abs(cfg["d"] - 2) > 1e-9 or cfg.get("tau_max") not in (None, "none"):
            continue
        for r in c["rows"]:
            r = dict(r, name=new_name(r["name"]))      # results written before the rename carry old names
            if r["name"] in CORE:
                r = dict(r, src=src)
                out[(cfg["N"], cfg["T"])].setdefault((src, r["graph"]), {})[r["name"]] = r
    return out


def collect_large(a):
    cells = defaultdict(dict)
    for src, paths in (("oracle", glob.glob(os.path.join(a.oracle, "N*_tau1_d2.json"))),
                       ("big", glob.glob(os.path.join(a.big, a.arm, "N*_g*_*.json")))):
        for k, v in load_rows(sorted(paths), src).items():
            cells[k].update(v)
    keys = sorted(cells)
    L = [f"Conditions: exact d-separation oracle on the full graph (fastdsep), S2, arm = {a.arm}, tau = 1, d = 2, process starts at t = 0, "
         f"candidates = full history, one run-wide cache, processing order = time, lazy headline; methods itpd_naive, itpd (paper variant), full conditioning. "
         f"Cells N x T with N T <= 2,000. Median [Q1, Q3] over graphs of the cell; ratios paired per graph (graphs with all three methods)."]
    rows, per_cell = [], {}
    for (N, T) in keys:
        gs = {g: d for g, d in cells[(N, T)].items() if all(m in d for m in CORE)}
        if not gs:
            continue
        G = list(gs.values())
        cand = np.median([g["full_conditioning"]["n_cand"] for g in G])
        u = {m: [g[m]["unique"] for g in G] for m in CORE}
        per_cell[(N, T)] = {"cand": cand, "u": {m: float(np.median(u[m])) for m in CORE}, "n": len(G)}
        rows.append([N, T, N * T, len(G), f"{cand:.0f}"] + [q(u[m]) for m in CORE] +
                    [q([g[m]["unique"] / g[m]["n_cand"] for g in G]) for m in CORE] +
                    [q([g["itpd"]["unique"] / g["itpd_naive"]["unique"] for g in G]), q([g["full_conditioning"]["unique"] / g["itpd_naive"]["unique"] for g in G])] +
                    [f"{sum(g[m]['exact'] for g in G)}/{len(G)}" for m in CORE])
    L += ["\n### Unique tests and tests per candidate by (N, T)\n",
          md(["N", "T", "nodes", "graphs", "candidates"] + [f"unique {m}" for m in CORE] + [f"per candidate {m}" for m in CORE] +
             ["itpd / naive", "full conditioning / naive"] + [f"exact {m}" for m in CORE], rows)]
    rows = []
    for (N, T) in keys:
        gs = [d for d in cells[(N, T)].values() if all(m in d for m in CORE)]
        if not gs:
            continue
        rows.append([N, T, len(gs)] + [q([g[m]["mean_target_max_frac"] for g in gs]) for m in ("itpd_naive", "itpd", "full_conditioning")] +
                    [q([g[m]["max_size"] for g in gs]) for m in ("itpd_naive", "itpd", "full_conditioning")] + [q([g["full_conditioning"]["n_cand"] / (N * (T - 1)) for g in gs])])
    L += ["\n### Largest conditioning set per target as a share of its candidates (mean over the targets of a graph), and the largest set of the graph\n",
          md(["N", "T", "graphs", "share naive", "share itpd", "share full conditioning", "largest naive", "largest itpd", "largest full conditioning", "mean candidates per target"], rows)]
    rows = []
    for (N, T) in keys:
        gs = [d for d in cells[(N, T)].values() if all(m in d for m in CORE)]
        if not gs or N * T < 600:
            continue
        for m in CORE:
            b = []
            for g in gs:
                bs = g[m]["by_size_unique"]
                b.append([sum(v for k, v in bs.items() if lo <= int(k) <= hi) for lo, hi, _ in LARGE_BINS])
            b = np.array(b)
            rows.append([N, T, m] + [f"{int(np.median(b[:, i]))}" for i in range(len(LARGE_BINS))])
    L += ["\n### Unique tests by conditioning-set size (median over graphs of the count in the bin), cells with at least 600 nodes\n",
          md(["N", "T", "method"] + [b[2] for b in LARGE_BINS], rows)]
    # slopes over all cells with data
    if len(per_cell) >= 3:
        xs = [v["cand"] for v in per_cell.values()]
        rows = [[m, f"{slope(xs, [v['u'][m] for v in per_cell.values()]):.3f}"] for m in CORE]
        L += [f"\n### Log-log slope of the cell-median unique tests vs the exact candidate count, {len(per_cell)} cells (N, T) = {sorted(per_cell)}\n",
              md(["method", "slope"], rows)]
    # scaling at fixed N / T
    for axis, fixed_vals in (("T", sorted({k[0] for k in per_cell})), ("N", sorted({k[1] for k in per_cell}))):
        rows = []
        for fv in fixed_vals:
            sub = sorted(k for k in per_cell if (k[0] == fv if axis == "T" else k[1] == fv))
            if len(sub) < 2:
                continue
            for m in CORE:
                s = slope([sub_k[1] if axis == "T" else sub_k[0] for sub_k in sub], [per_cell[k]["u"][m] for k in sub])
                rows.append([("N" if axis == "T" else "T") + f"={fv}", m, ", ".join(str(k[1] if axis == "T" else k[0]) for k in sub), f"{s:.2f}"])
        if rows:
            L += [f"\n### Log-log slope of the cell-median unique tests vs {axis} (other factor fixed)\n", md(["fixed", "method", f"{axis} values", "slope"], rows)]
    rows = []
    for (N, T) in keys:
        gs = [d for d in cells[(N, T)].values() if all(m in d for m in CORE) and all(d[m]["src"] == "big" for m in CORE)]
        if not gs:
            continue
        rows.append([N, T, len(gs)] + [q([g[m]["seconds"] for g in gs]) for m in CORE] +
                    [q([1e6 * g[m]["seconds"] / max(g[m]["raw"], 1) for g in gs]) for m in CORE])
    L += ["\n### Wall-clock per graph (seconds, one core; the recorder, hashing and the list building dominate the oracle) and microseconds per raw call, big cells only\n",
          md(["N", "T", "graphs"] + [f"s {m}" for m in CORE] + [f"us/call {m}" for m in CORE], rows)]
    txt = "\n".join(L) + "\n"
    out = a.out or os.path.join(a.big, f"TABLES-{a.arm}.md")
    open(out, "w").write(txt)
    print("wrote", out, len(txt))



# ------------------------------------------------------------------------------------------------ slopes

def load_arrays(d):
    cells = defaultdict(lambda: defaultdict(dict))
    for p in sorted(glob.glob(os.path.join(d, "N*.json"))):
        c = json.load(open(p))
        cfg = c["config"]
        key = (cfg["N"], cfg["T"], cfg["tau"], cfg["d"])
        for r in c["rows"]:
            cells[key][r["graph"]][new_name(r["name"])] = (r["unique"], r["n_cand"], r.get("mean_target_max_frac"), r.get("max_size"))
    out = {}
    for key, by in cells.items():
        gs = sorted(by)
        out[key] = {"uniq": {m: np.array([by[g][m][0] for g in gs], float) for m in SLOPE_METHODS},
                    "cand": np.array([by[g]["full_conditioning"][1] for g in gs], float),
                    "share": {m: np.array([by[g][m][2] for g in gs], float) for m in SLOPE_METHODS}}
    return out


def stats(cells, idx):
    """All slope statistics for one (re)sample. idx[key] = graph indices of that cell."""
    NT = defaultdict(lambda: {"cand": [], **{m: [] for m in SLOPE_METHODS}})
    for key, c in cells.items():
        N, T = key[0], key[1]
        i = idx[key]
        NT[(N, T)]["cand"].append(c["cand"][i])
        for m in SLOPE_METHODS:
            NT[(N, T)][m].append(c["uniq"][m][i])
    Ns = sorted({k[0] for k in NT})
    Ts = sorted({k[1] for k in NT})
    med = {}
    for k, v in NT.items():
        med[k] = {"cand": float(np.median(np.concatenate(v["cand"])))}
        cand_all = np.concatenate(v["cand"])
        for m in SLOPE_METHODS:
            u = np.concatenate(v[m])
            med[k][m] = float(np.median(u))
            med[k][m + "/cand"] = float(np.median(u / cand_all))
        med[k]["ratio_itpd"] = float(np.median(np.concatenate(v["itpd"]) / np.concatenate(v["itpd_naive"])))
        med[k]["ratio_itpd_nonlazy"] = float(np.median(np.concatenate(v["itpd_nonlazy"]) / np.concatenate(v["itpd_naive_nonlazy"])))
    out = {}
    # (a) cell-median slopes, (c) tests per candidate, (d) paired ratio
    for m in SLOPE_METHODS:
        for T in Ts:
            out[f"a|{m}|vs N|T={T}"] = slope(Ns, [med[(N, T)][m] for N in Ns])
            out[f"c|{m}|per cand vs N|T={T}"] = slope(Ns, [med[(N, T)][m + "/cand"] for N in Ns])
        for N in Ns:
            out[f"a|{m}|vs T|N={N}"] = slope(Ts, [med[(N, T)][m] for T in Ts])
            out[f"c|{m}|per cand vs T|N={N}"] = slope(Ts, [med[(N, T)][m + "/cand"] for T in Ts])
        keys = sorted(med)
        out[f"a|{m}|vs candidates|all"] = slope([med[k]["cand"] for k in keys], [med[k][m] for k in keys])
        out[f"c|{m}|per cand vs candidates|all"] = slope([med[k]["cand"] for k in keys], [med[k][m + "/cand"] for k in keys])
    for r in ("ratio_itpd", "ratio_itpd_nonlazy"):
        for T in Ts:
            out[f"d|{r}|vs N|T={T}"] = slope(Ns, [med[(N, T)][r] for N in Ns])
        for N in Ns:
            out[f"d|{r}|vs T|N={N}"] = slope(Ts, [med[(N, T)][r] for T in Ts])
        keys = sorted(med)
        out[f"d|{r}|vs candidates|all"] = slope([med[k]["cand"] for k in keys], [med[k][r] for k in keys])
    # (b) graph-level OLS
    cand = np.concatenate([c["cand"][idx[k]] for k, c in cells.items()])
    for m in SLOPE_METHODS:
        u = np.concatenate([c["uniq"][m][idx[k]] for k, c in cells.items()])
        out[f"b|{m}|graph-level vs candidates|all"] = slope(cand, u)
    return out



def collect_slopes(a):
    cells = load_arrays(a.dir)
    keys = sorted(cells)
    n = {k: len(cells[k]["cand"]) for k in keys}
    point = stats(cells, {k: np.arange(n[k]) for k in keys})
    rng = np.random.default_rng(a.seed)
    boots = {k: [] for k in point}
    for _ in range(a.boot):
        s = stats(cells, {k: rng.integers(0, n[k], n[k]) for k in keys})
        for kk, v in s.items():
            boots[kk].append(v)
    res = {kk: (point[kk], *np.percentile(boots[kk], [2.5, 97.5])) for kk in point}
    fmt = lambda t: f"{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]"
    c0 = json.load(open(sorted(glob.glob(os.path.join(a.dir, "N*.json")))[0]))["config"]
    L = [f"Conditions: oracle, S2, graph kind = {c0['graph_kind']}, full history, order = {c0['order']}, "
         f"{len(keys)} cells x {n[keys[0]]} graphs = {sum(n.values())} graphs, bootstrap over graphs within cells "
         f"({a.boot} resamples, percentile 95% interval, seed {a.seed}); lazy rows are the headline, _nonlazy non-lazy. "
         f"Cell = (N, T, tau, d); levels N in 5,10,20, T in 4,8,16 (3 levels each). Point estimate [2.5%, 97.5%]."]
    Ns = sorted({k[0] for k in keys}); Ts = sorted({k[1] for k in keys})

    def tab(title, kind, cols, fn):
        L.append(f"\n### {title}\n")
        L.append("| method | " + " | ".join(cols) + " |")
        L.append("|---|" + "---|" * len(cols))
        for m in SLOPE_METHODS:
            L.append(f"| {m} | " + " | ".join(fmt(res[fn(m, c)]) for c in cols) + " |")

    tab("(a) slope of the cell-median unique tests vs N (columns: T)", "a", [f"T={T}" for T in Ts],
        lambda m, c: f"a|{m}|vs N|{c}")
    tab("(a) slope of the cell-median unique tests vs T (columns: N)", "a", [f"N={N}" for N in Ns],
        lambda m, c: f"a|{m}|vs T|{c}")
    L.append("\n### (a) vs candidates over the 9 (N, T) groups, and (b) graph-level OLS vs candidates\n")
    L.append("| method | (a) cell medians | (b) all graphs |\n|---|---|---|")
    for m in SLOPE_METHODS:
        L.append(f"| {m} | {fmt(res[f'a|{m}|vs candidates|all'])} | {fmt(res[f'b|{m}|graph-level vs candidates|all'])} |")
    tab("(c) slope of tests per candidate vs N (columns: T)", "c", [f"T={T}" for T in Ts],
        lambda m, c: f"c|{m}|per cand vs N|{c}")
    tab("(c) slope of tests per candidate vs T (columns: N)", "c", [f"N={N}" for N in Ns],
        lambda m, c: f"c|{m}|per cand vs T|{c}")
    L.append("\n### (c) tests per candidate vs candidates (9 (N, T) groups)\n")
    L.append("| method | slope |\n|---|---|")
    for m in SLOPE_METHODS:
        L.append(f"| {m} | {fmt(res[f'c|{m}|per cand vs candidates|all'])} |")
    L.append("\n### (d) paired ratio itpd / itpd_naive: slope of log(median ratio); 0 = no drift\n")
    cols = [f"vs N, T={T}" for T in Ts] + [f"vs T, N={N}" for N in Ns] + ["vs candidates"]
    L.append("| ratio | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols))
    for r, nm in (("ratio_itpd", "itpd / naive (lazy)"), ("ratio_itpd_nonlazy", "itpd_nonlazy / itpd_naive_nonlazy")):
        cells_ = [f"d|{r}|vs N|T={T}" for T in Ts] + [f"d|{r}|vs T|N={N}" for N in Ns] + [f"d|{r}|vs candidates|all"]
        L.append(f"| {nm} | " + " | ".join(fmt(res[c]) for c in cells_) + " |")
    # largest conditioning set as a share of the target's candidates, by (N, T), tau and d pooled (mean over targets of a graph)
    L.append("\n### largest conditioning set per target as a share of its candidates, by (N, T) "
             "(per graph: mean over its targets; median [Q1, Q3] over the graphs of the 9 (tau, d) cells; bootstrap 95% CI of the median)\n")
    cols = [f"N={N}, T={T}" for N in Ns for T in Ts]
    L.append("| method | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols))
    rngb = np.random.default_rng(1)
    for m in ("itpd_naive", "itpd", "itpd_repo_variant", "full_conditioning"):
        row = []
        for N in Ns:
            for T in Ts:
                v = np.concatenate([cells[k]["share"][m] for k in keys if k[0] == N and k[1] == T])
                b = [np.median(v[rngb.integers(0, len(v), len(v))]) for _ in range(500)]
                q1, q2, q3 = np.percentile(v, [50, 25, 75])
                row.append(f"{q1:.2f} [{q2:.2f}, {q3:.2f}] CI [{np.percentile(b, 2.5):.2f}, {np.percentile(b, 97.5):.2f}]")
        L.append(f"| {m} | " + " | ".join(row) + " |")
    txt = "\n".join(L) + "\n"
    out = a.out or os.path.join(a.dir, "SLOPES.md")
    open(out, "w").write(txt)
    json.dump({k: list(v) for k, v in res.items()}, open(os.path.splitext(out)[0] + ".json", "w"))
    print("wrote", out)



def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m itpd.tables oracle_counts", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = ap.add_subparsers(dest="mode", required=True, metavar="{cells,large,slopes}")
    p = modes.add_parser("cells", help="tables of the grid cells", description="Tables of the cell JSONs of one oracle grid.")
    p.add_argument("dir", help="directory of the cell JSONs N<N>_T<T>_tau<tau>_d<d>.json")
    p.add_argument("--out", help="markdown file (default DIR/TABLES.md)")
    p = modes.add_parser("large", help="tables of the large cells", description="Tables of the large cells next to the small oracle-run cells.")
    p.add_argument("big", help="directory BIG_DIR of the large-cell JSONs")
    p.add_argument("--oracle", required=True, help="directory of the small cells N*_tau1_d2.json of the same arm")
    p.add_argument("--arm", default="window")
    p.add_argument("--out", help="markdown file (default BIG_DIR/TABLES-<arm>.md)")
    p = modes.add_parser("slopes", help="log-log scaling slopes with bootstrap intervals",
                         description="Log-log slopes with bootstrap confidence intervals over graphs.")
    p.add_argument("dir", help="directory of the cell JSONs")
    p.add_argument("--out", help="markdown file (default DIR/SLOPES.md; the numbers go next to it as SLOPES.json)")
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    {"cells": collect_cells, "large": collect_large, "slopes": collect_slopes}[a.mode](a)


if __name__ == "__main__":
    main()
