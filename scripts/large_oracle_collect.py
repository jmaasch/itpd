"""Large-cell tables: per-(cell, graph, method) JSONs of itpd.run_oracle_counts (3 methods) plus the small oracle-run cells at tau 1, d 2.

    python scripts/large_oracle_collect.py BIG_DIR --oracle ORACLE_DIR [--arm window] [--out BIG_DIR/TABLES-<arm>.md]
BIG_DIR/<arm>/N*_T*_g*_<method>.json (the glob takes files of both spellings of a method, `_order.json` and `_order_based.json`; files written by `python -m itpd.run_oracle_counts --graphs 1 --offset g --methods m`); ORACLE_DIR = the
results of the small cells of the same arm (cells N*_T*_tau1_d2.json, graphs 0..19, all methods, same instances: graph index g and the
seed convention are shared, so graph g of a big cell is a different graph from every small cell, the sha1 says which).
Numbers only. Headline = lazy unique tests; ratios paired per graph (graphs where all three methods finished).
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

M3 = ["itpd_naive", "itpd", "order_based"]
BINS = [(0, 0, "0"), (1, 1, "1"), (2, 4, "2-4"), (5, 19, "5-19"), (20, 99, "20-99"), (100, 499, "100-499"), (500, 10 ** 9, ">=500")]


def f(v):
    if v is None or (isinstance(v, float) and v != v):
        return "-"
    return f"{v:.0f}" if abs(v) >= 100 else (f"{v:.2f}" if abs(v) < 10 else f"{v:.1f}")


def q(x):
    x = [v for v in x if v is not None]
    if not x:
        return "-"
    a, b, c = np.percentile(np.asarray(x, float), [50, 25, 75])
    return f"{f(a)} [{f(b)}, {f(c)}]"


def md(head, rows):
    return "\n".join(["| " + " | ".join(map(str, head)) + " |", "|" + "---|" * len(head)] +
                     ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


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
            r = dict(r, name=OLD_TO_NEW.get(r["name"], r["name"]))      # results written before the rename carry old names
            if r["name"] in M3:
                r = dict(r, src=src)
                out[(cfg["N"], cfg["T"])].setdefault((src, r["graph"]), {})[r["name"]] = r
    return out


def slope(xs, ys):
    return float(np.polyfit(np.log(np.asarray(xs, float)), np.log(np.asarray(ys, float)), 1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("big")
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--arm", default="window")
    ap.add_argument("--out")
    a = ap.parse_args()
    cells = defaultdict(dict)
    for src, paths in (("oracle", glob.glob(os.path.join(a.oracle, "N*_tau1_d2.json"))),
                       ("big", glob.glob(os.path.join(a.big, a.arm, "N*_g*_*.json")))):
        for k, v in load_rows(sorted(paths), src).items():
            cells[k].update(v)
    keys = sorted(cells)
    L = [f"Conditions: exact d-separation oracle on the full graph (fastdsep), S2, arm = {a.arm}, tau = 1, d = 2, process starts at t = 0, "
         f"candidates = full history, one run-wide cache, processing order = time, lazy headline; methods itpd_naive, itpd (paper variant), order. "
         f"Cells N x T with N T <= 2,000. Median [Q1, Q3] over graphs of the cell; ratios paired per graph (graphs with all three methods)."]
    rows, per_cell = [], {}
    for (N, T) in keys:
        gs = {g: d for g, d in cells[(N, T)].items() if all(m in d for m in M3)}
        if not gs:
            continue
        G = list(gs.values())
        cand = np.median([g["order_based"]["n_cand"] for g in G])
        u = {m: [g[m]["unique"] for g in G] for m in M3}
        per_cell[(N, T)] = {"cand": cand, "u": {m: float(np.median(u[m])) for m in M3}, "n": len(G)}
        rows.append([N, T, N * T, len(G), f"{cand:.0f}"] + [q(u[m]) for m in M3] +
                    [q([g[m]["unique"] / g[m]["n_cand"] for g in G]) for m in M3] +
                    [q([g["itpd"]["unique"] / g["itpd_naive"]["unique"] for g in G]), q([g["order_based"]["unique"] / g["itpd_naive"]["unique"] for g in G])] +
                    [f"{sum(g[m]['exact'] for g in G)}/{len(G)}" for m in M3])
    L += ["\n### Unique tests and tests per candidate by (N, T)\n",
          md(["N", "T", "nodes", "graphs", "candidates"] + [f"unique {m}" for m in M3] + [f"per candidate {m}" for m in M3] +
             ["itpd / naive", "order / naive"] + [f"exact {m}" for m in M3], rows)]
    rows = []
    for (N, T) in keys:
        gs = [d for d in cells[(N, T)].values() if all(m in d for m in M3)]
        if not gs:
            continue
        rows.append([N, T, len(gs)] + [q([g[m]["mean_target_max_frac"] for g in gs]) for m in ("itpd_naive", "itpd", "order_based")] +
                    [q([g[m]["max_size"] for g in gs]) for m in ("itpd_naive", "itpd", "order_based")] + [q([g["order_based"]["n_cand"] / (N * (T - 1)) for g in gs])])
    L += ["\n### Largest conditioning set per target as a share of its candidates (mean over the targets of a graph), and the largest set of the graph\n",
          md(["N", "T", "graphs", "share naive", "share itpd", "share order", "largest naive", "largest itpd", "largest order", "mean candidates per target"], rows)]
    rows = []
    for (N, T) in keys:
        gs = [d for d in cells[(N, T)].values() if all(m in d for m in M3)]
        if not gs or N * T < 600:
            continue
        for m in M3:
            b = []
            for g in gs:
                bs = g[m]["by_size_unique"]
                b.append([sum(v for k, v in bs.items() if lo <= int(k) <= hi) for lo, hi, _ in BINS])
            b = np.array(b)
            rows.append([N, T, m] + [f"{int(np.median(b[:, i]))}" for i in range(len(BINS))])
    L += ["\n### Unique tests by conditioning-set size (median over graphs of the count in the bin), cells with at least 600 nodes\n",
          md(["N", "T", "method"] + [b[2] for b in BINS], rows)]
    # slopes over all cells with data
    if len(per_cell) >= 3:
        xs = [v["cand"] for v in per_cell.values()]
        rows = [[m, f"{slope(xs, [v['u'][m] for v in per_cell.values()]):.3f}"] for m in M3]
        L += [f"\n### Log-log slope of the cell-median unique tests vs the exact candidate count, {len(per_cell)} cells (N, T) = {sorted(per_cell)}\n",
              md(["method", "slope"], rows)]
    # scaling at fixed N / T
    for axis, fixed_vals in (("T", sorted({k[0] for k in per_cell})), ("N", sorted({k[1] for k in per_cell}))):
        rows = []
        for fv in fixed_vals:
            sub = sorted(k for k in per_cell if (k[0] == fv if axis == "T" else k[1] == fv))
            if len(sub) < 2:
                continue
            for m in M3:
                s = slope([sub_k[1] if axis == "T" else sub_k[0] for sub_k in sub], [per_cell[k]["u"][m] for k in sub])
                rows.append([("N" if axis == "T" else "T") + f"={fv}", m, ", ".join(str(k[1] if axis == "T" else k[0]) for k in sub), f"{s:.2f}"])
        if rows:
            L += [f"\n### Log-log slope of the cell-median unique tests vs {axis} (other factor fixed)\n", md(["fixed", "method", f"{axis} values", "slope"], rows)]
    rows = []
    for (N, T) in keys:
        gs = [d for d in cells[(N, T)].values() if all(m in d for m in M3) and all(d[m]["src"] == "big" for m in M3)]
        if not gs:
            continue
        rows.append([N, T, len(gs)] + [q([g[m]["seconds"] for g in gs]) for m in M3] +
                    [q([1e6 * g[m]["seconds"] / max(g[m]["raw"], 1) for g in gs]) for m in M3])
    L += ["\n### Wall-clock per graph (seconds, one core; the recorder, hashing and the list building dominate the oracle) and microseconds per raw call, big cells only\n",
          md(["N", "T", "graphs"] + [f"s {m}" for m in M3] + [f"us/call {m}" for m in M3], rows)]
    txt = "\n".join(L) + "\n"
    out = a.out or os.path.join(a.big, f"TABLES-{a.arm}.md")
    open(out, "w").write(txt)
    print("wrote", out, len(txt))


if __name__ == "__main__":
    main()
