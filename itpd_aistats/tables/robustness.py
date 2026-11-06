"""Tables from the per-task JSONs of `itpd.experiments robustness`.

    python -m itpd.tables robustness DIR --exp nonstationary|violations [--out FILE]

Numbers only. Median [Q1, Q3] over graphs; deltas are paired per graph (same base graph and data seed) with a 95% bootstrap CI of the mean.
"""
import argparse
import os
from collections import defaultdict

from .common import CORE, boot_ci, fmt_ci, load_tasks, md, q, run_of


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m itpd.tables robustness", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--exp", required=True)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    tasks = load_tasks(os.path.join(a.dir, a.exp, "*", "g*_M*.json"))
    by = defaultdict(dict)
    for t in tasks:
        by[(t["setting"], t["M"])][t["graph"]] = t
    base = "none" if a.exp == "violations" else "frac0"
    settings = sorted({k[0] for k in by}, key=lambda s: (s != base, s))
    Ms = sorted({k[1] for k in by})
    t0 = tasks[0]
    L = [f"Conditions: S2, N = {t0['N']}, T = {t0['T']}, DGP lag tau = {t0['tau']}, d = {t0['d']:g}, linear (Gaussian noise unless the setting "
         f"says otherwise), stationary lag weights (spectral radius <= 0.9), Fisher-z, alpha 0.01, full history, process starts at t = 0, "
         f"{len({t['graph'] for t in tasks})} graphs per setting, data = first M rows of one 2,000-row matrix per graph (same data seed across "
         f"settings), methods lazy headline (itpd = paper variant). Headline score = directed time-indexed edges, self edges excluded; "
         f"`F1 self` includes the self edges (the assumed self edge is output for every series). Infeasible targets: "
         f"{sum(r['n_inf_targets'] for t in tasks for r in t['runs'])} in total."]
    rows = []
    for s in settings:
        for M in Ms:
            ts = [by[(s, M)][g] for g in sorted(by[(s, M)])] if (s, M) in by else []
            if not ts:
                continue
            for m in CORE:
                rs = [run_of(t, m) for t in ts]
                x = [r["metrics"] for r in rs]
                nl = [run_of(t, m + "_nonlazy") for t in ts] if m != "full_conditioning" else []
                dl = ""
                if (base, M) in by and s != base:
                    pairs = [(run_of(by[(s, M)][g], m)["metrics"]["f1"], run_of(by[(base, M)][g], m)["metrics"]["f1"]) for g in sorted(by[(s, M)]) if g in by[(base, M)]]
                    dl = fmt_ci(boot_ci([p - b for p, b in pairs]))
                row = [s, M, m, q([y["recall"] for y in x]), q([y["precision"] for y in x]), q([y["f1"] for y in x]), q([y["shd"] for y in x]),
                       q([r["metrics_self"]["f1"] for r in rs]), dl, q([r["unique"] for r in rs]),
                       q([r["unique"] for r in nl]) if nl else "-"]
                if a.exp == "violations":
                    row += [q([t["n_lag0_true"] for t in ts]) if s.startswith("contemp") else "-", q([t["N_obs"] for t in ts]) if s.startswith("hidden") else "-"]
                rows.append(row)
    head = ["setting", "M", "method", "recall", "precision", "F1", "SHD", "F1 self", f"delta F1 vs {base}", "unique (lazy)", "unique (non-lazy)"]
    if a.exp == "violations":
        head += ["true same-time edges", "observed series"]
    L += [f"\n### Results: {a.exp}\n", md(head, rows)]
    rows = []
    for s in settings:
        for M in Ms:
            if (s, M) not in by:
                continue
            ts = [by[(s, M)][g] for g in sorted(by[(s, M)])]
            d = lambda m1, m2: fmt_ci(boot_ci([run_of(t, m1)["metrics"]["f1"] - run_of(t, m2)["metrics"]["f1"] for t in ts]))
            ratio = q([run_of(t, "itpd")["unique"] / run_of(t, "itpd_naive")["unique"] for t in ts])
            rows.append([s, M, d("itpd", "full_conditioning"), d("itpd_naive", "full_conditioning"), d("itpd", "itpd_naive"), ratio,
                         q([t["n_true_edges_nonself"] for t in ts]), f"{sum(1 for t in ts if t.get('guard_ok'))}/{len(ts)}"])
    L += ["\n### paired F1 differences and test ratios (per graph)\n",
          md(["setting", "M", "F1 itpd - full conditioning", "F1 naive - full conditioning", "F1 itpd - naive", "unique itpd / naive", "true non-self edges", "guard ok"], rows)]
    if a.exp == "nonstationary":
        rows = [[s, M, q([len(t["change_times"] or []) for t in by[(s, M)].values()]), str(sorted({tuple(t["change_times"] or []) for t in by[(s, M)].values()})[:3]),
                 q([t["n_true_edges_nonself"] for t in by[(s, M)].values()])] for s in settings for M in Ms if (s, M) in by]
        L += ["\n### Change times (shared by all replicates of a graph; first three distinct sets) and true non-self edge counts\n",
              md(["setting", "M", "changes", "change times", "true non-self edges"], rows)]
    out = a.out or os.path.join(a.dir, a.exp, "TABLES.md")
    open(out, "w").write("\n".join(L) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
