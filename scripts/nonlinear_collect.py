"""Tables from the per-task JSONs of itpd.run_nonlinear (nonlinear data, GCM test).

    python scripts/nonlinear_collect.py RUNS_DIR [--lin FINITE_WINDOW_DIR] [--out FILE.md]
Numbers only. Median [Q1, Q3] over graphs. `--lin` = results/finite/window (linear-Gaussian Fisher-z, same graphs by sha1) for the
paired comparison.
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from finite_common import q, md, run_of, boot_ci, fmt_ci, size_bins, BIN_LABELS, f, rename_rows  # noqa

METHODS = ["itpd_naive", "itpd", "full_conditioning"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--lin", default=None)
    ap.add_argument("--out")
    a = ap.parse_args()
    tasks = []
    for p in sorted(glob.glob(os.path.join(a.dir, "nonlinear_*", "g*_M*.json"))):
        if ".tmp" not in p:
            tasks.append(rename_rows(json.load(open(p))))
    L = []
    if tasks:
        by = defaultdict(list)
        for t in tasks:
            by[(t["N"], t["T"], t["M"])].append(t)
        keys = sorted(by)
        t0 = tasks[0]
        L.append(f"Conditions: S2, stationary window graph (the graph of the same index in the oracle and finite-data runs, same sha1), tau = {t0['tau']}, d = {t0['d']:g}, nonlinear additive noise "
                 f"(`itpd.nonlinear`: tanh / sine mixtures per edge, N(0, 1) noise), test = GCM with HistGradientBoostingRegressor (60 trees, 2-fold cross-fit), "
                 f"alpha 0.01 unless stated, full history, process order = time, lazy headline, {len(tasks)} (graph, M) tasks. Every target is feasible (GCM has no "
                 f"sample-size rule beyond n >= 8). Faithfulness proxy guard held on {sum(1 for t in tasks if t['guard']['guard_ok'])} of {len(tasks)} tasks.")
        rows = []
        for k in keys:
            ts = by[k]
            for m in METHODS:
                rs = [run_of(t, m) for t in ts]
                x = [r["metrics_common"] for r in rs]
                rows.append([k[0], k[1], k[2], m, len(ts), q([r["unique"] for r in rs]), q([r["raw"] for r in rs]),
                             q([r["unique"] / run_of(t, "full_conditioning")["unique"] for r, t in zip(rs, ts)]),
                             q([y["recall"] for y in x]), q([y["precision"] for y in x]), q([y["f1"] for y in x]), q([y["fp"] for y in x]), q([y["fn"] for y in x]),
                             q([max(r["target_max"]) for r in rs]), q([np.mean(r["target_max"]) for r in rs])])
        L += ["\n### Unique tests, accuracy and conditioning sizes (alpha 0.01; `unique / full conditioning` = unique tests relative to the full-conditioning run of the same graph)\n",
              md(["N", "T", "M", "method", "graphs", "unique tests", "raw calls", "unique / full conditioning", "recall", "precision", "F1", "FP", "FN",
                  "largest set", "mean target-max set"], rows)]
        rows = []
        for k in keys:
            for m in METHODS:
                b = np.array([size_bins(run_of(t, m)["by_size_unique"]) for t in by[k]])
                rows.append([k[0], k[1], k[2], m] + [f"{int(np.median(b[:, i]))}" for i in range(b.shape[1])])
        L += ["\n### Unique tests by conditioning-set size (median over graphs of the count in the bin; alpha 0.01)\n", md(["N", "T", "M", "method"] + BIN_LABELS, rows)]
        rows = []
        for k in keys:
            ts = by[k]
            r_ = lambda m1, m2: q([run_of(t, m1)["unique"] / run_of(t, m2)["unique"] for t in ts])
            rows.append([k[0], k[1], k[2], r_("itpd", "itpd_naive"), r_("full_conditioning", "itpd_naive")])
        L += ["\n### Unique-test ratios, paired per graph\n", md(["N", "T", "M", "itpd / naive", "full conditioning / naive"], rows)]
        # matched FP
        alphas = sorted({r["alpha"] for t in tasks for r in t["runs"]})
        rows = []
        for k in keys:
            ts = by[k]
            target = sum(run_of(t, "full_conditioning")["metrics_common"]["fp"] for t in ts)
            o = [run_of(t, "full_conditioning")["metrics_common"] for t in ts]
            rows.append([k[0], k[1], k[2], "full conditioning (alpha 0.01)", "0.01", target, q([x["recall"] for x in o]), q([x["precision"] for x in o]), q([x["f1"] for x in o])])
            for m in ("itpd_naive", "itpd"):
                best = None
                for al in alphas:
                    rs = [run_of(t, m, al) for t in ts]
                    fp = sum(r["metrics_common"]["fp"] for r in rs)
                    key = (abs(fp - target), al)
                    if best is None or key < best[0]:
                        best = (key, al, fp, rs)
                _, al, fp, rs = best
                x = [r["metrics_common"] for r in rs]
                rows.append([k[0], k[1], k[2], m, f"{al:g}", fp, q([y["recall"] for y in x]), q([y["precision"] for y in x]), q([y["f1"] for y in x])])
        L += ["\n### Matched false positives: each ITPD variant at the alpha whose total FP over the cell's graphs is closest to the full-conditioning total at alpha 0.01\n",
              f"alpha grid: {', '.join(f'{x:g}' for x in alphas)}\n", md(["N", "T", "M", "method", "alpha", "FP total", "recall", "precision", "F1"], rows)]
        rows = []
        for k in keys:
            ts = by[k]
            for m in ("itpd", "itpd_naive"):
                rows.append([k[0], k[1], k[2], f"{m} - full conditioning"] + [fmt_ci(boot_ci([run_of(t, m)["metrics_common"][key] - run_of(t, "full_conditioning")["metrics_common"][key] for t in ts]))
                                                                  for key in ("f1", "recall", "precision")])
        L += ["\n### Paired differences per graph (alpha 0.01): mean [95% bootstrap CI over graphs]\n", md(["N", "T", "M", "pair", "F1", "recall", "precision"], rows)]
        # cost per test
        rows = []
        for k in keys:
            ts = by[k]
            calls, secs = defaultdict(int), defaultdict(float)
            for t in ts:
                for s, v in t["gcm_calls_by_size"].items():
                    calls[int(s)] += v
                for s, v in t["gcm_sec_by_size"].items():
                    secs[int(s)] += v
            tot_c, tot_s = sum(calls.values()), sum(secs.values())
            bins = [(0, 0), (1, 1), (2, 4), (5, 19), (20, 99), (100, 10 ** 9)]
            cells = []
            for lo, hi in bins:
                c = sum(v for s, v in calls.items() if lo <= s <= hi)
                s_ = sum(v for s, v in secs.items() if lo <= s <= hi)
                cells.append(f"{s_ / c:.3f}" if c else "-")
            rows.append([k[0], k[1], k[2], len(ts), q([t["memo_size"] for t in ts]), f"{sum(1 for t in ts if t['memo_resumed'])}", tot_c // len(ts), f"{tot_s / tot_c:.3f}", f"{tot_s / len(ts):.0f}"] + cells + [f"{q([t['seconds'] for t in ts])}"])
        L += ["\n### Cost: GCM evaluations (distinct tests actually computed per dataset, shared by the three methods and all alpha), seconds per evaluation by conditioning size, "
              "CPU seconds in the test per dataset, wall seconds per task (single core); a task resumed from a checkpoint reports evaluations, test seconds and wall seconds of its last process only, so the totals per dataset are the memo size times the mean seconds per evaluation\n",
              md(["N", "T", "M", "datasets", "distinct tests per dataset (memo size, all alpha and methods)", "tasks resumed from a checkpoint", "evaluations timed in the last process (per dataset)", "s per evaluation (mean)", "test seconds per dataset (last process)"] + [f"s/eval |S| {l}" for l in BIN_LABELS] + ["task wall seconds"], rows)]
        # linear comparison
        if a.lin:
            lin = {}
            for p in glob.glob(os.path.join(a.lin, "*", "g*_M*.json")):
                if ".tmp" in p:
                    continue
                d = rename_rows(json.load(open(p)))
                lin[(d["N"], d["T"], d["M"], d["graph"])] = d
            rows = []
            for k in keys:
                pairs = [(t, lin[(k[0], k[1], k[2], t["graph"])]) for t in by[k] if (k[0], k[1], k[2], t["graph"]) in lin]
                if not pairs:
                    continue
                assert all(t["sha1"] == l["sha1"] for t, l in pairs)
                for m in METHODS:
                    rows.append([k[0], k[1], k[2], m, len(pairs),
                                 q([run_of(l, m)["unique"] for _, l in pairs]), q([run_of(t, m)["unique"] for t, _ in pairs]),
                                 q([run_of(t, m)["unique"] / run_of(l, m)["unique"] for t, l in pairs if run_of(l, m)["n_inf_targets"] == 0]),
                                 q([run_of(l, m)["metrics_common"]["f1"] if run_of(l, m)["metrics_common"] else None for _, l in pairs]),
                                 q([run_of(t, m)["metrics_common"]["f1"] for t, _ in pairs]),
                                 q([max(run_of(l, m)["target_max"]) for _, l in pairs]), q([max(run_of(t, m)["target_max"]) for t, _ in pairs])])
            L += ["\n### Nonlinear + GCM vs linear-Gaussian + Fisher-z on the same graphs (alpha 0.01; linear F1 on the common targets of its M)\n",
                  md(["N", "T", "M", "method", "graphs", "unique linear", "unique nonlinear", "ratio nl / lin", "F1 linear", "F1 nonlinear", "largest set linear", "largest set nonlinear"], rows)]
    txt = "\n".join(L) + "\n"
    out = a.out or os.path.join(a.dir, "TABLES.md")
    open(out, "w").write(txt)
    print("wrote", out, len(txt))


if __name__ == "__main__":
    main()
