"""Tables of the blanket-screened shrink runs: the oracle runs against the stored oracle-run rows of ITPD, ITPD_naive and the full-conditioning baseline, and
the finite-data runs against the stored finite-data rows. Reads, under the results directory R (`--results`, default $ITPD_RESULTS
or ./results): R/blanket_screened_shrink/oracle (itpd.run_blanket_screened_shrink oracle), R/blanket_screened_shrink/single_pass (itpd.run_blanket_screened_shrink finite), R/oracle and R/finite/window (the
earlier runs). Writes tables_oracle.md, tables_finite.md and aggregates.json in `--out` (default R/blanket_screened_shrink).

    python scripts/blanket_screened_shrink_collect.py [--results DIR] [--out DIR] [--only oracle,finite]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

from itpd import observed_data
from itpd.methods_registry import OLD_TO_NEW
from itpd.dataset_eval import common_tmax
from itpd.instances import load_instance

RES = os.environ.get("ITPD_RESULTS", "results")
ORACLE_STORED = ("itpd_naive", "itpd", "full_conditioning")
FINITE_STORED = ("itpd_naive", "itpd", "full_conditioning")
FINITE_SHRINK = ("blanket_screened_shrink", "blanket_screened_shrink_lenient", "blanket_screened_shrink_oracle_blanket", "blanket_screened_shrink_oracle_blanket_lenient",
         "blanket_screened_shrink_recheck_lenient")
LABEL = {"blanket_screened_shrink": "blanket-screened shrink", "blanket_screened_shrink_lenient": "blanket-screened shrink lenient",
         "blanket_screened_shrink_oracle_blanket": "blanket-screened shrink oracle blanket",
         "blanket_screened_shrink_oracle_blanket_lenient": "blanket-screened shrink oracle blanket lenient",
         "blanket_screened_shrink_recheck_lenient": "blanket-screened shrink re-check lenient", "itpd": "ITPD",
         "itpd_naive": "ITPD_naive", "full_conditioning": "full conditioning"}
new_name = lambda n: OLD_TO_NEW.get(n, n)          # result files written before the method-name rename carry old names
PRIMARY = 0.01
BINS = (0.0, 0.05, 0.1, 0.2, 0.3, np.inf)


def mq(x, fmt="{:.2f}"):
    x = np.asarray([v for v in x if v is not None], float)
    if len(x) == 0:
        return "-"
    q1, m, q3 = np.percentile(x, [25, 50, 75])
    return f"{fmt.format(m)} [{fmt.format(q1)}, {fmt.format(q3)}]"


def mr(x, fmt="{:.3f}"):
    x = np.asarray(x, float)
    return f"{fmt.format(np.median(x))} [{fmt.format(x.min())}, {fmt.format(x.max())}]"


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# ============================================================================================ oracle runs

def collect_oracle(res, out):
    lines = ["# Blanket-screened shrink oracle runs: tables", "",
             "Conditions: S2, instances of the oracle runs (sha1 checked per graph), exact d-separation oracle on the full graph, full "
             "history, F = {X}, d = 2, 20 graphs per cell. tpc = unique tests / Sum|C| (whole graph); per-target tpc = mean "
             "over targets of new unique tests / |C|; share = mean over targets of (largest set / |C|). Stored methods: "
             "lazy headline rows of the oracle runs (full conditioning is one test per candidate). Medians [min, max] over graphs.", ""]
    agg = {}
    rows = []
    tflat = defaultdict(dict)
    for arm in ("time", "window"):
        for N in (10, 20):
            for T in (8, 16):
                for tau in (1, 2):
                    cell = f"N{N}_T{T}_tau{tau}_d2"
                    files = sorted(glob.glob(os.path.join(res, "blanket_screened_shrink", "oracle", arm, cell, "g*.json")))
                    if not files:
                        continue
                    D = [json.load(open(f)) for f in files]
                    orc = json.load(open(os.path.join(res, "oracle", arm, cell + ".json")))
                    by = defaultdict(list)
                    for d in D:
                        for r in d["rows"]:
                            by[new_name(r["name"])].append(r)
                    for nm in ORACLE_STORED:
                        by[nm] = [r for r in orc["rows"] if new_name(r["name"]) == nm and r["graph"] < len(D)]
                    for nm, rr in by.items():
                        tpc = [r["unique"] / r["n_cand"] for r in rr]
                        row = [arm, N, T, tau, nm, f"{sum(r['exact'] for r in rr)}/{len(rr)}", mr(tpc),
                               mr([r["raw"] / r["n_cand"] for r in rr]),
                               mr([r["mean_target_tpc"] for r in rr]) if "mean_target_tpc" in rr[0] else "-",
                               mr([r["max_size"] for r in rr], "{:.0f}"), mr([r["mean_target_max_frac"] for r in rr], "{:.2f}")]
                        if nm in ORACLE_STORED:
                            row += ["-"] * 5
                        else:
                            row += [f"{sum(r['calls_identity'] for r in rr)}/{len(rr)}",
                                    f"{sum(r['n_bound_b_viol'] for r in rr)} / {sum(r['n_bound_glob_viol'] for r in rr)}",
                                    sum(r["shrink_stats"]["excess"] for r in rr), sum(r["shrink_stats"]["sum_Aprime"] for r in rr),
                                    " ".join(f"{k}:{np.median([r['by_label_unique'].get(k, 0) / r['n_cand'] for r in rr]):.3f}"
                                             for k in ("A", "B", "C") if any(k in r["by_label_unique"] for r in rr))]
                        rows.append(row)
                        agg[f"{arm}|{cell}|{nm}"] = {"tpc_med": float(np.median(tpc)), "tpc_min": float(min(tpc)),
                                                     "tpc_max": float(max(tpc)),
                                                     "max_med": float(np.median([r["max_size"] for r in rr])),
                                                     "share_med": float(np.median([r["mean_target_max_frac"] for r in rr])),
                                                     "exact": int(sum(r["exact"] for r in rr)), "n": len(rr)}
                        tflat[(arm, N, tau, nm)][T] = float(np.median([r["max_size"] for r in rr]))
                    # shift excess by target time (window)
                    if arm == "window" and "blanket_screened_shrink_shifted_parents" in by:
                        ex = defaultdict(int)
                        for r in by["blanket_screened_shrink_shifted_parents"]:
                            for k, v in r["excess_by_t"].items():
                                ex[int(k)] += v
                        agg[f"{arm}|{cell}|shift_excess_by_t"] = {str(k): v for k, v in sorted(ex.items())}
                    # per-candidate overhead bound check for the single-pass variant
                    if "blanket_screened_shrink" in by:
                        agg[f"{arm}|{cell}|overhead_bound_ok"] = int(sum((r["raw"] - r["n_cand"]) / r["n_cand"] <= r["overhead_bound"] - 1 + 1e-12
                                                                for r in by["blanket_screened_shrink"]))
    lines.append(table(["arm", "N", "T", "tau", "method", "exact", "tpc", "raw/cand", "per-target tpc", "largest set",
                        "share", "calls = Sum|C| + |E| - |E_F|", "bound viol. (b, global)", "excess survivors",
                        "A' (re-check adds)", "unique tpc by step (A, B, C)"], rows))
    lines += ["", "#### Shifted-parents screening set (window arm): excess survivors (non-parents surviving the screening step) summed over graphs, by target time", ""]
    srows = [[k.split("|")[1], json.dumps(v)] for k, v in agg.items() if k.endswith("shift_excess_by_t")]
    lines.append(table(["cell", "excess by target time"], srows))
    lines += ["", "#### Largest set per graph: median at T = 16 / median at T = 8 (same arm, N, tau)", ""]
    trows = []
    for (arm, N, tau, nm), v in sorted(tflat.items()):
        if 8 in v and 16 in v:
            trows.append([arm, N, tau, nm, f"{v[16]:.0f} / {v[8]:.0f} = {v[16] / v[8]:.2f}"])
    lines.append(table(["arm", "N", "tau", "method", "ratio"], trows))
    lines += ["", "#### Per-candidate overhead bound 1 + 2(d_in - 1)/(NT - 2), single-pass blanket-screened shrink: graphs within bound", ""]
    lines.append(table(["cell", "graphs within"], [[k.rsplit("|", 1)[0], v] for k, v in agg.items() if k.endswith("overhead_bound_ok")]))
    open(os.path.join(out, "tables_oracle.md"), "w").write("\n".join(lines) + "\n")
    return agg


# ============================================================================================ F

def _ncand(N, T):
    return N * sum(N * t - 1 for t in range(1, T))


def _share(target_max, N, T, target_inf=None):
    v = []
    for i, m in enumerate(target_max):
        if target_inf is not None and target_inf[i]:
            continue
        t = i // N + 1
        v.append(m / (N * t - 1))
    return float(np.mean(v)) if v else None


def _interp(points, target):
    """points: (fp, recall) along the alpha order; recall at fp = target by linear interpolation in log(fp) between the
    two neighbouring points (sorted by fp); None if the target is outside [min fp, max fp]."""
    pts = sorted((max(f, 0.5), r) for f, r in points)
    lt = np.log(max(target, 0.5))
    xs = [np.log(f) for f, _ in pts]
    if lt < xs[0] - 1e-12 or lt > xs[-1] + 1e-12:
        return None
    for (x0, (_, r0)), (x1, (_, r1)) in zip(zip(xs, pts), zip(xs[1:], pts[1:])):
        if x0 - 1e-12 <= lt <= x1 + 1e-12:
            return r0 if x1 == x0 else r0 + (r1 - r0) * (lt - x0) / (x1 - x0)
    return pts[-1][1]


def collect_finite(res, out):
    fin = os.path.join(res, "finite", "window")
    lines = ["# Blanket-screened shrink finite-data runs: tables", "",
             "Conditions: S2 window arm (stationary lag weights, spectral radius <= 0.9), tau = 1, d = 2, linear-Gaussian, "
             "Fisher-z, full history, the instances g00-g19 per (N, T) of the finite-data runs (sha1, data_seed and data_sha1 checked), data = "
             "first M rows (paired across M and methods). Blanket-screened shrink: equal = alpha_A = alpha_B; lenient = alpha_A 0.10; alpha_B = 0.01 "
             "unless stated. Stored rows (ITPD paper variant, ITPD_naive, full conditioning) at alpha 0.01, lazy. Common targets: "
             "t <= (M - 3)/N. Medians [Q1, Q3] over 20 graphs; totals are sums over the 20 graphs.", ""]
    edge_cache = {}
    A_rows, B_rows, C_rows, D_rows, E_rows, H_rows = [], [], [], [], [], []
    fnbins = defaultdict(lambda: defaultdict(lambda: np.zeros((2, len(BINS) - 1), int)))   # (M, method) -> measure -> [fn, all]
    agg = {}
    for N in (10, 20):
        for T in (8, 16):
            cell = f"window_N{N}_T{T}_tau1_d2"
            nc = _ncand(N, T)
            for g in range(20):
                if (cell, g) not in edge_cache:
                    inst = load_instance(os.path.join(fin, "instances", cell, f"g{g:02d}.npz"))
                    st = observed_data.edge_strengths(inst["A"], observed_data.sigma_from_W(inst["W"]), N, T)
                    edge_cache[(cell, g)] = st
            for M in (50, 100, 200, 500, 2000):
                ct = common_tmax(N, T, M)
                rows_by = defaultdict(list)      # method -> list of rows at alpha_B = 0.01 (one per graph)
                sweep = defaultdict(lambda: defaultdict(lambda: np.zeros(3)))   # method -> alpha -> [tp, fp, fn] pooled
                for g in range(20):
                    fh = os.path.join(res, "blanket_screened_shrink", "single_pass", cell, f"g{g:02d}_M{M}.json")
                    fs = os.path.join(fin, cell, f"g{g:02d}_M{M}.json")
                    if not os.path.exists(fh):
                        continue
                    dh, ds = json.load(open(fh)), json.load(open(fs))
                    assert dh["sha1"] == ds["sha1"]
                    for r in dh["runs"] + [r for r in ds["runs"] if new_name(r["name"]) in FINITE_STORED]:
                        nm = new_name(r["name"])
                        mc = r.get("metrics_common")
                        if mc is not None:
                            sweep[nm][r["alpha"]] += [mc["tp"], mc["fp"], mc["fn"]]
                        if abs(r["alpha"] - PRIMARY) < 1e-12:
                            r = dict(r)
                            r["_g"] = g
                            rows_by[nm].append(r)
                if not rows_by:
                    continue
                full_cond_fp = sweep["full_conditioning"][PRIMARY][1]
                full_cond_rec = sweep["full_conditioning"][PRIMARY][0] / max(1, sweep["full_conditioning"][PRIMARY][0] + sweep["full_conditioning"][PRIMARY][2])
                for nm in FINITE_SHRINK + FINITE_STORED:
                    rr = rows_by.get(nm, [])
                    if not rr:
                        continue
                    key = f"{cell}|M{M}|{nm}"
                    tpc = [r["unique"] / nc for r in rr]
                    lab = rr[0].get("by_label_unique")
                    by_step = " ".join(f"{k}:{np.median([r['by_label_unique'].get(k, 0) / nc for r in rr]):.3f}"
                                     for k in ("A", "B", "C") if lab and k in lab) if lab else "-"
                    tmaxs = [max(r["target_max"]) for r in rr]
                    shares = [_share(r["target_max"], N, T, r["target_inf"]) for r in rr]
                    inf = [r["n_inf_targets"] / r["n_targets"] for r in rr]
                    A_rows.append([N, T, M, LABEL[nm], mq(tpc, "{:.3f}"), mq([r["raw"] / nc for r in rr], "{:.3f}"), by_step,
                                   mq(tmaxs, "{:.0f}"), mq(shares), f"{np.mean(inf):.2f} ({sum(1 for x in inf if x > 0)}/20)"])
                    mcs = [r["metrics_common"] for r in rr if r.get("metrics_common")]
                    tot = np.sum([[m["tp"], m["fp"], m["fn"]] for m in mcs], axis=0) if mcs else np.zeros(3)
                    B_rows.append([N, T, M, f"{ct}/{T - 1}", LABEL[nm], mq([m["recall"] for m in mcs]),
                                   mq([m["precision"] for m in mcs]), mq([m["f1"] for m in mcs]), int(tot[1]), int(tot[2])])
                    own = [r["metrics"] for r in rr if r.get("metrics")]
                    C_rows.append([N, T, M, LABEL[nm], mq([m["recall"] for m in own]), mq([m["precision"] for m in own]),
                                   mq([m["f1"] for m in own]), f"{np.mean(inf):.2f}"])
                    # matched FP (oracle-tuned diagnostic): methods with a sweep
                    if len(sweep[nm]) > 1:
                        pts = [(v[1], v[0] / max(1, v[0] + v[2])) for a, v in sorted(sweep[nm].items())]
                        rec_at = _interp(pts, full_cond_fp)
                        fps = [p[0] for p in pts]
                        D_rows.append([N, T, M, LABEL[nm], int(full_cond_fp), f"{full_cond_rec:.3f}",
                                       "not reached" if rec_at is None else f"{rec_at:.3f}",
                                       f"{int(min(fps))}-{int(max(fps))}",
                                       " ".join(f"{a:g}:{int(v[1])}/{v[0] / max(1, v[0] + v[2]):.2f}"
                                                for a, v in sorted(sweep[nm].items(), reverse=True))])
                        agg[key + "|matched_recall"] = rec_at
                    # FN by population |rho| (common targets)
                    for r in rr:
                        st = edge_cache[(cell, r["_g"])]
                        E = st["edges"]
                        inc = (E[:, 1] // N) <= ct
                        fn = np.zeros(len(E), bool)
                        fn[r.get("fn_idx", [])] = True
                        meas = {"marg": st["marg"], "cond_x": st["cond_x"], "cond_pa": st["cond_pa"]}
                        if r.get("screening_rho"):
                            ra = np.full(len(E), np.nan)
                            for k, _, rho, _ in r["screening_rho"]:
                                ra[k] = rho
                            meas["rho_A"] = ra
                        for mname, vals in meas.items():
                            ok = inc & ~np.isnan(vals)
                            b = np.digitize(vals[ok], BINS[1:-1])
                            fnbins[(M, nm)][mname][0] += np.bincount(b[fn[ok]], minlength=len(BINS) - 1)
                            fnbins[(M, nm)][mname][1] += np.bincount(b, minlength=len(BINS) - 1)
                    agg[key] = {"tpc_med": float(np.median(tpc)), "max_med": float(np.median(tmaxs)),
                                "share_med": float(np.median([s for s in shares if s is not None])) if any(s is not None for s in shares) else None,
                                "inf_share": float(np.mean(inf)),
                                "rec_med": float(np.median([m["recall"] for m in mcs])) if mcs else None,
                                "prec_med": float(np.median([m["precision"] for m in mcs])) if mcs else None,
                                "f1_med": float(np.median([m["f1"] for m in mcs])) if mcs else None,
                                "fp_tot": int(tot[1]), "fn_tot": int(tot[2]), "tp_tot": int(tot[0]),
                                "f1_by_g": {int(r["_g"]): r["metrics_common"]["f1"] for r in rr if r.get("metrics_common")},
                                "tpc_by_g": {int(r["_g"]): r["unique"] / nc for r in rr}}
                    if nm.startswith("blanket_screened_shrink"):
                        hs = [r["shrink_stats"] for r in rr]
                        agg[key].update({k: int(sum(h[k] for h in hs)) for k in
                                         ("excess", "lost", "sum_R", "sum_Aprime", "targets_Aprime", "shortcut_diff", "a_inf", "n_capped")})
                        H_rows.append([N, T, M, LABEL[nm], agg[key]["excess"], agg[key]["lost"], agg[key]["sum_Aprime"],
                                       agg[key]["targets_Aprime"], agg[key]["shortcut_diff"], agg[key]["a_inf"]])
                # error propagation: learned blanket vs true blanket
                for a, b in (("blanket_screened_shrink", "blanket_screened_shrink_oracle_blanket"), ("blanket_screened_shrink_lenient", "blanket_screened_shrink_oracle_blanket_lenient")):
                    ka, kb = f"{cell}|M{M}|{a}", f"{cell}|M{M}|{b}"
                    if ka in agg and kb in agg:
                        gs = sorted(set(agg[ka]["f1_by_g"]) & set(agg[kb]["f1_by_g"]))
                        df = [agg[kb]["f1_by_g"][g] - agg[ka]["f1_by_g"][g] for g in gs]
                        dt = [agg[ka]["tpc_by_g"][g] - agg[kb]["tpc_by_g"][g] for g in gs]
                        E_rows.append([N, T, M, "lenient" if a.endswith("lenient") else "equal",
                                       f"{np.mean(df):+.3f} [{np.min(df):+.3f}, {np.max(df):+.3f}]" if df else "-",
                                       f"{np.mean(dt):+.3f}", agg[ka]["excess"], agg[kb]["excess"], agg[ka]["lost"], agg[kb]["lost"],
                                       agg[ka]["fn_tot"], agg[kb]["fn_tot"], agg[ka]["fp_tot"], agg[kb]["fp_tot"]])
    lines += ["#### Tests and conditioning sets (all targets; a method with infeasible targets includes the tests issued before "
              "it abandoned them). tpc = unique / Sum|C|; share = mean over feasible targets of largest set / |C|", ""]
    lines.append(table(["N", "T", "M", "method", "unique tpc", "raw tpc", "unique tpc by step (A, B, C)", "largest set per graph",
                        "largest-set share", "infeasible-target share (graphs with any)"], A_rows))
    lines += ["", "#### Common targets (t <= (M-3)/N; comparable across methods), alpha (alpha_B) 0.01", ""]
    lines.append(table(["N", "T", "M", "common t", "method", "recall", "precision", "F1", "FP total", "FN total"], B_rows))
    lines += ["", "#### Own-feasible targets (each method on the targets it could decide; NOT comparable across methods)", ""]
    lines.append(table(["N", "T", "M", "method", "recall", "precision", "F1", "infeasible share"], C_rows))
    lines += ["", "#### Matched FP, oracle-tuned diagnostic (needs the truth; not a usable procedure). Common targets, pooled over "
              "the 20 graphs. Recall of each swept method interpolated linearly in log(FP) at full conditioning's pooled FP at alpha 0.01; "
              "'not reached' if that FP lies outside the method's swept FP range (no extrapolation). Sweeps: blanket-screened shrink "
              "alpha in 13 values 0.2-1e-6; blanket-screened shrink lenient alpha_B in 12 values 0.1-1e-6 (alpha_A 0.1); ITPD, ITPD_naive 13 values. "
              "Last column: alpha:FP/recall.", ""]
    lines.append(table(["N", "T", "M", "method", "full conditioning FP", "full conditioning recall", "recall at full conditioning FP", "swept FP range", "curve"], D_rows))
    lines += ["", "#### Error propagation: learned-graph screening set (learned_blanket) vs true-blanket screening set (oracle_blanket), same alphas. "
              "dF1 = F1(oracle_blanket) - F1(learned_blanket), mean [min, max] over graphs (common targets); dtpc = tpc(learned_blanket) - tpc(oracle_blanket); "
              "excess = non-parents surviving the screening step; lost = true parents removed in the screening step (all targets)", ""]
    lines.append(table(["N", "T", "M", "alphas", "dF1", "dtpc", "excess learned", "excess oracle", "lost learned", "lost oracle",
                        "FN learned", "FN oracle", "FP learned", "FP oracle"], E_rows))
    lines += ["", "#### Blanket-screened shrink internals (totals over 20 graphs, all targets): excess survivors, lost parents, re-check adds "
              "(A' members, targets with A' non-empty), always-verify vs shortcut output differences, screening-step tests that were "
              "infeasible (candidate kept)", ""]
    lines.append(table(["N", "T", "M", "method", "excess", "lost", "A' total", "targets with A'", "shortcut diff", "screening-step infeasible"], H_rows))
    lines += ["", "#### False negatives by population |rho| (common targets, alpha 0.01, pooled over the 4 (N, T) cells): FN / true "
              "edges in the bin. marg = |corr(Z, Y)|, cond_x = |pcorr(Z, Y | X)|, cond_pa = |pcorr(Z, Y | other parents)|, "
              "rho_A = |pcorr(Z, Y | screening set actually used)| (blanket-screened shrink only)", ""]
    bl = [f"[{BINS[i]:g}, {BINS[i + 1]:g})" for i in range(len(BINS) - 1)]
    G_rows = []
    for (M, nm), d in sorted(fnbins.items(), key=lambda x: (x[0][0], x[0][1])):
        for mname, arr in d.items():
            G_rows.append([M, LABEL[nm], mname] + [f"{a}/{b}" for a, b in zip(arr[0], arr[1])] + [f"{arr[0].sum()}"])
            agg[f"fnbins|M{M}|{nm}|{mname}"] = arr.tolist()
    lines.append(table(["M", "method", "measure"] + bl + ["FN total"], G_rows))
    open(os.path.join(out, "tables_finite.md"), "w").write("\n".join(lines) + "\n")
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=RES)
    ap.add_argument("--out", default=None, help="output directory (default: <results>/blanket_screened_shrink)")
    ap.add_argument("--only", default="oracle,finite")
    a = ap.parse_args()
    out = a.out or os.path.join(a.results, "blanket_screened_shrink")
    os.makedirs(out, exist_ok=True)
    agg = {}
    if "oracle" in a.only:
        agg["oracle"] = collect_oracle(a.results, out)
    if "finite" in a.only:
        agg["finite"] = collect_finite(a.results, out)
    json.dump(agg, open(os.path.join(out, "aggregates.json"), "w"), default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("wrote", out)


if __name__ == "__main__":
    main()
