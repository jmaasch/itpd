"""Tables from the per-task JSONs of itpd.run_single_series.   python scripts/single_series_collect.py DIR [--out FILE]   Numbers only."""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from finite_common import load_tasks, md, q  # noqa: E402


def get(t, name):
    for r in t["runs"]:
        if r["name"] == name:
            return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--out")
    a = ap.parse_args()
    tasks = load_tasks(os.path.join(a.dir, "N*", "g*.json"))
    by = defaultdict(list)
    for t in tasks:
        by[(t["N"], t["tau"], t["T"])].append(t)
    keys = sorted(by)
    L = [f"Conditions: S1 (one trajectory per graph), stationary VAR(tau), lag weights fixed in t, spectral radius <= 0.9, 100 burn-in steps "
         f"discarded, Gaussian noise, d = {tasks[0]['d']:g}, Fisher-z with n = T - tau overlapping window rows (dependent samples), alpha 0.01, "
         f"lazy, {len({t['graph'] for t in tasks})} graphs per cell, {len(tasks)} runs. Truth = lag graph B (lags 1..tau, self and lag 0 excluded); "
         f"ITPD = ITPD_naive on the last slice by construction (one pair per series). Median [Q1, Q3] over graphs."]
    rows = []
    for k in keys:
        ts = by[k]
        g = lambda n, f_: [get(t, n)[f_] for t in ts if get(t, n)["status"] == "ok"]
        gm = lambda n, key: [get(t, n)["metrics"][key] for t in ts if get(t, n)["status"] == "ok"]
        inf = sum(1 for t in ts for r in t["runs"] if r["status"] != "ok")
        rows.append([k[0], k[1], k[2], len(ts), q(gm("itpd_last_slice", "f1")), q(gm("itpd_last_slice", "recall")), q(gm("itpd_last_slice", "precision")),
                     q(g("itpd_last_slice", "unique")), q(gm("order_based_last_slice", "f1")), q(g("order_based_last_slice", "unique")),
                     q([get(t, "itpd_naive_every_slice")["metrics_union"]["f1"] for t in ts if get(t, "itpd_naive_every_slice")["status"] == "ok"]) if k[1] > 1 else "= last slice",
                     q(g("itpd_naive_every_slice", "unique")) if k[1] > 1 else "= last slice", inf])
    L += ["\n### Last slice vs every slice (F1 on the lag graph; unique tests per graph)\n",
          md(["N", "tau", "T", "graphs", "itpd last F1", "recall", "precision", "itpd last unique", "order last F1", "order unique",
              "every-slice F1 (union of slices)", "every-slice unique (naive)", "infeasible runs"], rows)]
    rows = []
    for k in keys:
        if k[1] == 1:
            continue
        ts = by[k]
        for nm in ("itpd_naive_every_slice", "itpd_every_slice"):
            ok = [get(t, nm) for t in ts if get(t, nm)["status"] == "ok"]
            tot = {s: [sum(r["per_slice"][str(s)]["wrong_targets"] for r in [x]) for x in ok] for s in range(1, k[1] + 1)}
            f1s = {s: [2 * r["per_slice"][str(s)]["tp"] / max(2 * r["per_slice"][str(s)]["tp"] + r["per_slice"][str(s)]["fp"] + r["per_slice"][str(s)]["fn"], 1) for r in ok]
                   for s in range(1, k[1] + 1)}
            rows.append([k[0], k[1], k[2], nm] + [f"{np.sum(tot[s])}/{k[0] * len(ok)} (F1 {np.median(f1s[s]):.2f})" for s in range(1, k[1] + 1)] + ["-"] * (3 - k[1]) +
                        [q([get(t, 'itpd_last_slice')['unique'] for t in ts]), q([r["unique"] for r in ok])])
    L += ["\n### Every-slice run: wrong targets / targets per window slice (median F1 of that slice against the in-window true parents); slice tau = the last slice\n",
          md(["N", "tau", "T", "method", "slice 1", "slice 2", "slice 3", "last-slice unique", "every-slice unique"], rows)]
    rows = []
    for k in keys:
        ts = by[k]
        rows.append([k[0], k[1], k[2], q([get(t, "itpd_last_slice")["max_size"] for t in ts if get(t, "itpd_last_slice")["status"] == "ok"]),
                     k[0] * k[1] - 1, q([t["n_rows"] for t in ts])])
    L += ["\n### Largest conditioning set (last slice) vs candidates per target (N tau - 1) and rows\n",
          md(["N", "tau", "T", "largest set, itpd last", "candidates per target", "rows n"], rows)]
    out = a.out or os.path.join(a.dir, "TABLES.md")
    open(out, "w").write("\n".join(L) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
