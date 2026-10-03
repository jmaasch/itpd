"""Tables from the per-task JSONs of itpd.run_finite_data.   python scripts/finite_collect.py DIR [--out DIR/TABLES.md]
Numbers only. Median [Q1, Q3] over graphs unless stated; all graphs of every cell are in every table (an infeasible
target is excluded from its method's score, the share is shown; a graph is never dropped)."""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from finite_common import BIN_LABELS, OLD_TO_NEW, boot_ci, fmt_ci, load_tasks, md, q, run_of, size_bins  # noqa: E402

LAZY = ["itpd_naive", "itpd", "itpd_repo_variant", "full_conditioning"]
EXTRA = ["itpd_adjacency_self"]
NL = {"itpd_naive": "itpd_naive_nonlazy", "itpd": "itpd_nonlazy", "itpd_repo_variant": "itpd_repo_variant_nonlazy", "itpd_adjacency_self": "itpd_adjacency_self_nonlazy"}
ALPHAS = None
BINS = [(0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 1.01)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--out")
    ap.add_argument("--oracle", default=None, help="results directory of the oracle runs of the same arm (pairing / finite-vs-oracle counts)")
    a = ap.parse_args()
    tasks = load_tasks(os.path.join(a.dir, "*", "g*_M*.json"))
    if not tasks:
        raise SystemExit("no tasks")
    by = defaultdict(list)
    for t in tasks:
        by[(t["N"], t["T"], t["M"])].append(t)
    keys = sorted(by)
    arm = tasks[0]["arm"]
    L = [f"Conditions: S2, arm = {arm} (same graph and weight draws as the oracle runs, same seeds), linear-Gaussian, Fisher-z, full history, DGP lag tau = "
         f"{tasks[0]['tau']}, expected out-degree d = {tasks[0]['d']:g}, process starts at t = 0, process order = time, data = first M rows of "
         f"one 2,000-row matrix per graph (paired across methods and M), alpha 0.01 unless stated, {len(tasks)} (graph, M) tasks, "
         f"{len(set((t['N'], t['T'], t['graph']) for t in tasks))} graphs. Headline = lazy evaluation, `_nonlazy` = non-lazy. Targets = the "
         f"N (T - 1) nodes with t >= 1. An infeasible test (M < |S| + 4) is never answered `independent`: the target is infeasible, "
         f"excluded from the score, and counted. `common targets` = t <= floor((M - 3) / N), feasible for every method."]

    # feasibility (analytic)
    rows = []
    for (N, T) in sorted({(k[0], k[1]) for k in keys}):
        Ms = sorted({k[2] for k in keys if k[:2] == (N, T)})
        rows.append([N, T, N * (T - 1) + 3] + [f"{finite_ct(N, T, M)}/{T - 1}" for M in Ms])
    L += ["\n### Feasibility: full conditioning needs M >= N (T - 1) + 3 for the last target; common targets (t <= (M - 3) / N) by M\n",
          md(["N", "T", "M for all targets"] + [f"common t, M={M}" for M in Ms], rows)]

    def cellrows(metric_key, uniq_key, methods, title, note):
        rows = []
        for k in keys:
            ts = by[k]
            for m in methods:
                rs = [run_of(t, m) for t in ts]
                if any(r is None for r in rs):
                    continue
                mt = [r[metric_key] for r in rs]
                ok = [x for x in mt if x is not None]
                if metric_key == "metrics_common" and not ok:
                    continue
                inf = [r["n_inf_targets"] / r["n_targets"] for r in rs]
                nl = [run_of(t, NL[m]) for t in ts] if m in NL else []
                rows.append([k[0], k[1], k[2], m, f"{np.mean(inf):.2f} ({sum(1 for r in rs if r['n_inf_targets'] > 0)}/{len(rs)})",
                             q([x["recall"] for x in ok]), q([x["precision"] for x in ok]), q([x["f1"] for x in ok]),
                             q([x["shd"] for x in ok]), q([r.get(uniq_key) for r in rs]),
                             q([r.get(uniq_key) for r in nl]) if nl and all(r is not None for r in nl) else "-"])
        L.extend(["\n### " + title + "\n", note, md(["N", "T", "M", "method", "infeasible targets: mean share (graphs with any)", "recall",
                                                      "precision", "F1", "SHD", "unique tests, lazy", "unique tests, non-lazy"], rows)])

    cellrows("metrics", "unique", LAZY + EXTRA, "Own-feasible targets (each method scored on the targets it could decide; unique counts include the tests "
             "issued before an infeasible target was abandoned)", "alpha 0.01; directed time-indexed edges, self edges excluded.")
    cellrows("metrics_common", "unique_common", LAZY + EXTRA, "Common targets (feasible for every method): comparable across methods",
             "alpha 0.01; edges into targets t <= floor((M - 3) / N) only; unique tests summed over those targets.")

    # matched FP
    alphas = sorted({r["alpha"] for t in tasks for r in t["runs"] if r["name"] == "full_conditioning"})
    rows = []
    for k in keys:
        ts = [t for t in by[k] if t["common_tmax"] >= 1]
        if not ts:
            continue
        target = sum(run_of(t, "full_conditioning")["metrics_common"]["fp"] for t in ts)
        o = [run_of(t, "full_conditioning")["metrics_common"] for t in ts]
        rows.append([k[0], k[1], k[2], "full conditioning (alpha 0.01)", "0.01", target, q([x["recall"] for x in o]), q([x["precision"] for x in o]),
                     q([x["f1"] for x in o]), q([x["shd"] for x in o])])
        for m in ("itpd_naive", "itpd", "itpd_repo_variant"):
            best = None
            for al in alphas:
                rs = [run_of(t, m, al) for t in ts]
                if any(r is None or r["metrics_common"] is None for r in rs):
                    continue
                fp = sum(r["metrics_common"]["fp"] for r in rs)
                key = (abs(fp - target), al)
                if best is None or key < best[0]:
                    best = (key, al, fp, rs)
            if best is None:
                continue
            _, al, fp, rs = best
            x = [r["metrics_common"] for r in rs]
            rows.append([k[0], k[1], k[2], m, f"{al:g}", fp, q([y["recall"] for y in x]), q([y["precision"] for y in x]),
                         q([y["f1"] for y in x]), q([y["shd"] for y in x])])
    L += ["\n### Matched false positives (common targets): each ITPD variant at the alpha whose total FP over the cell's graphs is closest to the "
          "full-conditioning total at alpha 0.01 (ties: smaller alpha)\n",
          f"alpha grid: {', '.join(f'{x:g}' for x in alphas)}; the `FP total` column is the sum over the cell's graphs (target = the full-conditioning row).\n",
          md(["N", "T", "M", "method", "alpha", "FP total", "recall", "precision", "F1", "SHD"], rows)]

    # paired differences
    rows = []
    for k in keys:
        ts = [t for t in by[k] if t["common_tmax"] >= 1]
        if len(ts) < 3:
            continue
        def d(m1, m2, key="f1"):
            return boot_ci([run_of(t, m1)["metrics_common"][key] - run_of(t, m2)["metrics_common"][key] for t in ts])
        rows.append([k[0], k[1], k[2], fmt_ci(d("itpd", "full_conditioning")), fmt_ci(d("itpd_naive", "full_conditioning")), fmt_ci(d("itpd", "itpd_naive")),
                     fmt_ci(d("itpd", "itpd_repo_variant")), fmt_ci(d("itpd", "full_conditioning", "recall")), fmt_ci(d("itpd", "full_conditioning", "precision"))])
    L += ["\n### Paired differences per graph (common targets, alpha 0.01): mean [95% bootstrap CI over graphs]\n",
          md(["N", "T", "M", "F1 itpd - full conditioning", "F1 naive - full conditioning", "F1 itpd - naive", "F1 itpd - itpd_repo_variant", "recall itpd - full conditioning",
              "precision itpd - full conditioning"], rows)]

    # tests by size and ratios
    rows = []
    for k in keys:
        for m in LAZY + EXTRA:
            rs = [run_of(t, m) for t in by[k]]
            b = np.array([size_bins(r["by_size_unique"]) for r in rs])
            rows.append([k[0], k[1], k[2], m] + [f"{int(np.median(b[:, i]))}" for i in range(b.shape[1])])
    L += ["\n### Unique tests by conditioning size (alpha 0.01, lazy; median over graphs of the count in the bin; all targets, incl. "
          "tests issued before an infeasible target was abandoned)\n", md(["N", "T", "M", "method"] + BIN_LABELS, rows)]
    rows = []
    for k in keys:
        ts = by[k]
        def r_(m1, m2, mode="lazy"):
            return q([run_of(t, m1)["unique"] / run_of(t, m2)["unique"] for t in ts if run_of(t, m2)["unique"] and
                      run_of(t, m1)["n_inf_targets"] == 0 and run_of(t, m2)["n_inf_targets"] == 0])
        rows.append([k[0], k[1], k[2], r_("itpd", "itpd_naive"), r_("itpd_nonlazy", "itpd_naive_nonlazy"), r_("itpd_repo_variant", "itpd_naive"),
                     r_("itpd_adjacency_self", "itpd"), r_("full_conditioning", "itpd_naive")])
    L += ["\n### Unique-test ratios, paired per graph (graphs where both methods have no infeasible target)\n",
          md(["N", "T", "M", "itpd / naive (lazy)", "itpd / naive (non-lazy)", "itpd_repo_variant / naive", "itpd_adjacency_self / itpd", "full conditioning / naive"], rows)]

    # largest conditioning set vs M
    rows = []
    for (N, T) in sorted({(k[0], k[1]) for k in keys}):
        for m in LAZY:
            big = [run_of(t, m) for t in by.get((N, T, 2000), [])]
            if not big:
                continue
            tm = [max(r["target_max"]) for r in big]
            rows.append([N, T, m, q(tm), f"{int(np.median(tm)) + 4}", q([np.median(r["target_max"]) for r in big]),
                         " ".join(f"{k[2]}:{np.mean([run_of(t, m)['n_inf_targets'] / run_of(t, m)['n_targets'] for t in by[k]]):.2f}"
                                  for k in keys if k[:2] == (N, T))])
    L += ["\n### Largest conditioning set per graph (max over targets, from the M = 2,000 runs, all targets feasible) and the M it needs "
          "(= largest + 4); median per target; measured infeasible-target share by M (mean over graphs)\n",
          md(["N", "T", "method", "largest set per graph", "M needed (median)", "median target-max", "infeasible share by M (M:share)"], rows)]

    # FN by population strength
    from itpd import observed_data
    import itpd.instances as inst_mod
    rows = []
    for k in keys:
        acc = {m: np.zeros((len(BINS), 2)) for m in LAZY}
        for t in by[k]:
            if t["common_tmax"] < 1:
                continue
            ip = os.path.join(a.dir, "instances", t["cell"], f"g{t['graph']:02d}.npz")
            if not os.path.exists(ip):
                continue
            inst = inst_mod.load_instance(ip)
            Sig = observed_data.sigma_from_W(inst["W"])
            es = observed_data.edge_strengths(inst["A"], Sig, k[0], k[1])
            inc = (es["edges"][:, 1] // k[0]) <= t["common_tmax"]
            for m in LAZY:
                r = run_of(t, m)
                fn = set(r.get("fn_idx", []))
                for bi, (lo, hi) in enumerate(BINS):
                    sel = np.nonzero(inc & (es["cond_pa"] >= lo) & (es["cond_pa"] < hi))[0]
                    acc[m][bi, 0] += len(sel)
                    acc[m][bi, 1] += sum(1 for i in sel if i in fn)
        for m in LAZY:
            if acc[m][:, 0].sum():
                rows.append([k[0], k[1], k[2], m] + [f"{int(acc[m][i, 1])}/{int(acc[m][i, 0])}" for i in range(len(BINS))])
    L += ["\n### Missed true edges (FN / true edges) by population |partial correlation given the other parents|, common targets, alpha 0.01, "
          "summed over graphs\n", md(["N", "T", "M", "method"] + [f"[{lo:g}, {min(hi, 1):g})" for lo, hi in BINS], rows)]

    # alpha sensitivity
    rows = []
    for k in [kk for kk in keys if kk in ((10, 8, 500), (20, 16, 2000), (20, 8, 500))]:
        ts = [t for t in by[k] if t["common_tmax"] >= 1]
        for m in LAZY:
            for al in (0.001, 0.01, 0.05):
                rs = [run_of(t, m, al) for t in ts]
                if any(r is None for r in rs):
                    continue
                x = [r["metrics_common"] for r in rs]
                rows.append([k[0], k[1], k[2], m, f"{al:g}", q([r["unique"] for r in rs]), q([y["recall"] for y in x]), q([y["precision"] for y in x]),
                             q([y["f1"] for y in x]), q([y["fp"] for y in x])])
    L += ["\n### Alpha sensitivity (common targets)\n", md(["N", "T", "M", "method", "alpha", "unique tests", "recall", "precision", "F1", "FP"], rows)]

    # pairing with the oracle runs and finite vs oracle counts
    if a.oracle:
        import json as _j
        rows = []
        for (N, T) in sorted({(k[0], k[1]) for k in keys}):
            pf = os.path.join(a.oracle, f"N{N}_T{T}_tau{tasks[0]['tau']}_d{tasks[0]['d']:g}.json")
            if not os.path.exists(pf) or (N, T, 2000) not in by:
                continue
            orc = _j.load(open(pf))
            o = {(r["graph"], OLD_TO_NEW.get(r["name"], r["name"])): r for r in orc["rows"]}
            tk = [t for t in by[(N, T, 2000)] if (t["graph"], "full_conditioning") in o]      # graphs beyond the oracle grid (index >= 20) have no oracle pair
            same = all(o[(t["graph"], "full_conditioning")]["graph_stats"]["sha1"] == t["sha1"] for t in tk)
            for m in ("itpd_naive", "itpd", "full_conditioning"):
                rat = [run_of(t, m)["unique"] / o[(t["graph"], m)]["unique"] for t in tk if run_of(t, m)["n_inf_targets"] == 0]
                rows.append([N, T, m, "yes" if same else "NO", q(rat)])
        L += ["\n### Finite-data unique tests (M = 2,000, alpha 0.01) / oracle unique tests, same graph (sha1 equal), per graph\n",
              md(["N", "T", "method", "same sha1 as the oracle run", "ratio"], rows)]
    txt = "\n".join(L) + "\n"
    out = a.out or os.path.join(a.dir, "TABLES.md")
    open(out, "w").write(txt)
    print("wrote", out, len(txt))


def finite_ct(N, T, M):
    return int(max(0, min(T - 1, (M - 3) // N)))


if __name__ == "__main__":
    main()
