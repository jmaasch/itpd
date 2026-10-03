"""Log-log slopes with bootstrap confidence intervals over graphs, from the saved oracle-run cells.

    python scripts/oracle_scaling_slopes.py results/oracle/time [--out results/oracle/time/SLOPES.md] [--boot 2000]

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
Only 3 levels of N and of T exist in the grid in the default grid: slope CIs are wide for that reason.
Numbers only.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from itpd.methods_registry import OLD_TO_NEW  # noqa: E402

METHODS = ["itpd_naive", "itpd", "itpd_repo_variant", "order_based", "itpd_naive_nonlazy", "itpd_nonlazy"]


def load(d):
    cells = defaultdict(lambda: defaultdict(dict))
    for p in sorted(glob.glob(os.path.join(d, "N*.json"))):
        c = json.load(open(p))
        cfg = c["config"]
        key = (cfg["N"], cfg["T"], cfg["tau"], cfg["d"])
        for r in c["rows"]:
            cells[key][r["graph"]][OLD_TO_NEW.get(r["name"], r["name"])] = (r["unique"], r["n_cand"], r.get("mean_target_max_frac"), r.get("max_size"))
    out = {}
    for key, by in cells.items():
        gs = sorted(by)
        out[key] = {"uniq": {m: np.array([by[g][m][0] for g in gs], float) for m in METHODS},
                    "cand": np.array([by[g]["order_based"][1] for g in gs], float),
                    "share": {m: np.array([by[g][m][2] for g in gs], float) for m in METHODS}}
    return out


def fit(x, y):
    return float(np.polyfit(np.log(np.asarray(x, float)), np.log(np.asarray(y, float)), 1)[0])


def stats(cells, idx):
    """All slope statistics for one (re)sample. idx[key] = graph indices of that cell."""
    NT = defaultdict(lambda: {"cand": [], **{m: [] for m in METHODS}})
    for key, c in cells.items():
        N, T = key[0], key[1]
        i = idx[key]
        NT[(N, T)]["cand"].append(c["cand"][i])
        for m in METHODS:
            NT[(N, T)][m].append(c["uniq"][m][i])
    Ns = sorted({k[0] for k in NT})
    Ts = sorted({k[1] for k in NT})
    med = {}
    for k, v in NT.items():
        med[k] = {"cand": float(np.median(np.concatenate(v["cand"])))}
        cand_all = np.concatenate(v["cand"])
        for m in METHODS:
            u = np.concatenate(v[m])
            med[k][m] = float(np.median(u))
            med[k][m + "/cand"] = float(np.median(u / cand_all))
        med[k]["ratio_itpd"] = float(np.median(np.concatenate(v["itpd"]) / np.concatenate(v["itpd_naive"])))
        med[k]["ratio_itpd_nonlazy"] = float(np.median(np.concatenate(v["itpd_nonlazy"]) / np.concatenate(v["itpd_naive_nonlazy"])))
    out = {}
    # (a) cell-median slopes, (c) tests per candidate, (d) paired ratio
    for m in METHODS:
        for T in Ts:
            out[f"a|{m}|vs N|T={T}"] = fit(Ns, [med[(N, T)][m] for N in Ns])
            out[f"c|{m}|per cand vs N|T={T}"] = fit(Ns, [med[(N, T)][m + "/cand"] for N in Ns])
        for N in Ns:
            out[f"a|{m}|vs T|N={N}"] = fit(Ts, [med[(N, T)][m] for T in Ts])
            out[f"c|{m}|per cand vs T|N={N}"] = fit(Ts, [med[(N, T)][m + "/cand"] for T in Ts])
        keys = sorted(med)
        out[f"a|{m}|vs candidates|all"] = fit([med[k]["cand"] for k in keys], [med[k][m] for k in keys])
        out[f"c|{m}|per cand vs candidates|all"] = fit([med[k]["cand"] for k in keys], [med[k][m + "/cand"] for k in keys])
    for r in ("ratio_itpd", "ratio_itpd_nonlazy"):
        for T in Ts:
            out[f"d|{r}|vs N|T={T}"] = fit(Ns, [med[(N, T)][r] for N in Ns])
        for N in Ns:
            out[f"d|{r}|vs T|N={N}"] = fit(Ts, [med[(N, T)][r] for T in Ts])
        keys = sorted(med)
        out[f"d|{r}|vs candidates|all"] = fit([med[k]["cand"] for k in keys], [med[k][r] for k in keys])
    # (b) graph-level OLS
    cand = np.concatenate([c["cand"][idx[k]] for k, c in cells.items()])
    for m in METHODS:
        u = np.concatenate([c["uniq"][m][idx[k]] for k, c in cells.items()])
        out[f"b|{m}|graph-level vs candidates|all"] = fit(cand, u)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--out")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cells = load(a.dir)
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
        for m in METHODS:
            L.append(f"| {m} | " + " | ".join(fmt(res[fn(m, c)]) for c in cols) + " |")

    tab("(a) slope of the cell-median unique tests vs N (columns: T)", "a", [f"T={T}" for T in Ts],
        lambda m, c: f"a|{m}|vs N|{c}")
    tab("(a) slope of the cell-median unique tests vs T (columns: N)", "a", [f"N={N}" for N in Ns],
        lambda m, c: f"a|{m}|vs T|{c}")
    L.append("\n### (a) vs candidates over the 9 (N, T) groups, and (b) graph-level OLS vs candidates\n")
    L.append("| method | (a) cell medians | (b) all graphs |\n|---|---|---|")
    for m in METHODS:
        L.append(f"| {m} | {fmt(res[f'a|{m}|vs candidates|all'])} | {fmt(res[f'b|{m}|graph-level vs candidates|all'])} |")
    tab("(c) slope of tests per candidate vs N (columns: T)", "c", [f"T={T}" for T in Ts],
        lambda m, c: f"c|{m}|per cand vs N|{c}")
    tab("(c) slope of tests per candidate vs T (columns: N)", "c", [f"N={N}" for N in Ns],
        lambda m, c: f"c|{m}|per cand vs T|{c}")
    L.append("\n### (c) tests per candidate vs candidates (9 (N, T) groups)\n")
    L.append("| method | slope |\n|---|---|")
    for m in METHODS:
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
    for m in ("itpd_naive", "itpd", "itpd_repo_variant", "order_based"):
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


if __name__ == "__main__":
    main()
