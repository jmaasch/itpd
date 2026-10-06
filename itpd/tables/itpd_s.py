"""Tables of the ITPD-S and ITPD-S+ runs: the oracle runs against the stored oracle-run rows of ITPD, ITPD_naive and the full-conditioning baseline, the
finite-data runs against the stored finite-data rows, and the pooled recall-false-positive curves with the recall at matched false positives.
Reads, under the results directory R (`--results`, default $ITPD_RESULTS or ./results): R/itpd_s/oracle
(`itpd.experiments stored_instances itpd_s oracle`), R/itpd_s/single_pass and R/itpd_s/recheck
(`stored_instances itpd_s finite`, the latter with `--specs itpd_s_plus,itpd_s_plus_screen_clean`),
R/oracle and R/finite/window (the earlier runs, with their instances). Writes in `--out` (default R/itpd_s):

  --only oracle    tables_oracle.md
  --only finite    tables_finite.md
  (both)           aggregates.json
  --only curves    curves.csv (pooled recall-FP curves, one row per cell, M, method and alpha), matched_fp.csv (recall and F1
                   interpolated in log FP at the declared FP levels, an oracle-tuned diagnostic), tables.md, aggregates_curves.json

    python -m itpd.tables stored_instances itpd_s [--results DIR] [--out DIR] [--only oracle,finite,curves]
        [--arm time,window] [--N 10,20] [--T 8,16] [--tau 1,2] [--M 50,100,200,500,2000] [--graphs 20]

The default of `--only` is `oracle,finite`. The grid is the one of the runs: `oracle` uses --arm, --N, --T, --tau and --graphs (the files
that do not exist are skipped), `finite` and `curves` use --N, --T, --M and --graphs (tau 1, d 2, window arm; `curves` needs every file of the grid).

Pooled = sums of tp, fp, fn over the graphs of a cell, common targets (t <= (M - 3)/N). Interpolation: recall linear in log FP
(FP clamped at 0.5) between the two neighbouring alphas of the grid, sorted by FP; no extrapolation (a level outside the swept FP
range is reported as below / above the range).
"""
from __future__ import annotations

import csv
import glob
import json
import os
from collections import defaultdict
from typing import NamedTuple

import numpy as np

from itpd import observed_data
from itpd.dataset_eval import common_tmax
from itpd.experiments.common import ints
from itpd.instances import load_instance

from .common import ALPHA, CORE, RESULTS, interp, md, ncand, new_name

FINITE_SHRINK = ("itpd_s", "itpd_s_screen_clean", "itpd_s_true_blanket", "itpd_s_true_blanket_screen_clean",
         "itpd_s_plus_screen_clean")
LABEL = {"itpd_s": "ITPD-S", "itpd_s_screen_clean": "ITPD-S screen-and-clean",
         "itpd_s_true_blanket": "ITPD-S true-blanket screen",
         "itpd_s_true_blanket_screen_clean": "ITPD-S true-blanket screen-and-clean",
         "itpd_s_plus_screen_clean": "ITPD-S+ screen-and-clean", "itpd": "ITPD",
         "itpd_naive": "ITPD_naive", "full_conditioning": "full conditioning"}
BINS = (0.0, 0.05, 0.1, 0.2, 0.3, np.inf)
# key -> (source, name in the JSON, display label)
METH = {
    "eq": ("single_pass", "itpd_s", "ITPD-S"),
    "rc_eq": ("recheck", "itpd_s_plus", "ITPD-S+"),
    "len": ("single_pass", "itpd_s_screen_clean", "ITPD-S screen-and-clean"),
    "rc_len": ("recheck", "itpd_s_plus_screen_clean", "ITPD-S+ screen-and-clean"),
    "itpd": ("finite", "itpd", "ITPD"),
    "naive": ("finite", "itpd_naive", "ITPD_naive"),
    "full_conditioning": ("finite", "full_conditioning", "full conditioning"),
}
KEYS = list(METH)
LEVELS = {"a": ("full_conditioning", "full conditioning's FP at alpha 0.01"), "b": ("eq", "ITPD-S's own FP (alpha 0.01)"),
          "c": ("itpd", "ITPD's own FP (alpha 0.01)"), "d": ("rc_eq", "ITPD-S+'s own FP (alpha 0.01)")}
UNDECIDABLE_MS = (50, 100)                 # the M of the undecidable-target tables
NEAR_CANCELLED_MS = (200, 500, 2000)       # the M of the near-cancelled-parents table


class Grid(NamedTuple):
    arms: list
    Ns: list
    Ts: list
    taus: list
    Ms: list
    graphs: int

    @property
    def cells(self):
        return [(N, T) for N in self.Ns for T in self.Ts]


def _cells_word(n):
    """'four cells' (a small count as a word) for the table titles."""
    word = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight")[n] if n < 9 else str(n)
    return f"{word} cell" + ("" if n == 1 else "s")


def mq(x, fmt="{:.2f}"):
    x = np.asarray([v for v in x if v is not None], float)
    if len(x) == 0:
        return "-"
    q1, m, q3 = np.percentile(x, [25, 50, 75])
    return f"{fmt.format(m)} [{fmt.format(q1)}, {fmt.format(q3)}]"


def mr(x, fmt="{:.3f}"):
    x = np.asarray(x, float)
    return f"{fmt.format(np.median(x))} [{fmt.format(x.min())}, {fmt.format(x.max())}]"


# ============================================================================================ oracle runs

def collect_oracle(res, out, grid):
    lines = ["# ITPD-S and ITPD-S+ oracle runs: tables", "",
             "Conditions: S2, instances of the oracle runs (sha1 checked per graph), exact d-separation oracle on the full graph, full "
             f"history, F = {{X}}, d = 2, {grid.graphs} graphs per cell. tpc = unique tests / Sum|C| (whole graph); per-target tpc = mean "
             "over targets of new unique tests / |C|; share = mean over targets of (largest set / |C|). Stored methods: "
             "lazy headline rows of the oracle runs (full conditioning is one test per candidate). Medians [min, max] over graphs.", ""]
    agg = {}
    rows = []
    tflat = defaultdict(dict)
    for arm in grid.arms:
        for N in grid.Ns:
            for T in grid.Ts:
                for tau in grid.taus:
                    cell = f"N{N}_T{T}_tau{tau}_d2"
                    files = sorted(glob.glob(os.path.join(res, "itpd_s", "oracle", arm, cell, "g*.json")))
                    if not files:
                        continue
                    D = [json.load(open(f)) for f in files]
                    orc = json.load(open(os.path.join(res, "oracle", arm, cell + ".json")))
                    by = defaultdict(list)
                    for d in D:
                        for r in d["rows"]:
                            by[new_name(r["name"])].append(r)
                    for nm in CORE:
                        by[nm] = [r for r in orc["rows"] if new_name(r["name"]) == nm and r["graph"] < len(D)]
                    for nm, rr in by.items():
                        tpc = [r["unique"] / r["n_cand"] for r in rr]
                        row = [arm, N, T, tau, nm, f"{sum(r['exact'] for r in rr)}/{len(rr)}", mr(tpc),
                               mr([r["raw"] / r["n_cand"] for r in rr]),
                               mr([r["mean_target_tpc"] for r in rr]) if "mean_target_tpc" in rr[0] else "-",
                               mr([r["max_size"] for r in rr], "{:.0f}"), mr([r["mean_target_max_frac"] for r in rr], "{:.2f}")]
                        if nm in CORE:
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
                    if arm == "window" and "itpd_s_shifted_parents" in by:
                        ex = defaultdict(int)
                        for r in by["itpd_s_shifted_parents"]:
                            for k, v in r["excess_by_t"].items():
                                ex[int(k)] += v
                        agg[f"{arm}|{cell}|shift_excess_by_t"] = {str(k): v for k, v in sorted(ex.items())}
                    # per-candidate overhead bound check for the single-pass variant
                    if "itpd_s" in by:
                        agg[f"{arm}|{cell}|overhead_bound_ok"] = int(sum((r["raw"] - r["n_cand"]) / r["n_cand"] <= r["overhead_bound"] - 1 + 1e-12
                                                                for r in by["itpd_s"]))
    lines.append(md(["arm", "N", "T", "tau", "method", "exact", "tpc", "raw/cand", "per-target tpc", "largest set",
                     "share", "calls = Sum|C| + |E| - |E_F|", "bound viol. (b, global)", "excess survivors",
                     "A' (re-check adds)", "unique tpc by step (A, B, C)"], rows))
    lines += ["", "#### Shifted-parents screening set (window arm): excess survivors (non-parents surviving the screening step) summed over graphs, by target time", ""]
    srows = [[k.split("|")[1], json.dumps(v)] for k, v in agg.items() if k.endswith("shift_excess_by_t")]
    lines.append(md(["cell", "excess by target time"], srows))
    lines += ["", "#### Largest set per graph: median at T = 16 / median at T = 8 (same arm, N, tau)", ""]
    trows = []
    for (arm, N, tau, nm), v in sorted(tflat.items()):
        if 8 in v and 16 in v:
            trows.append([arm, N, tau, nm, f"{v[16]:.0f} / {v[8]:.0f} = {v[16] / v[8]:.2f}"])
    lines.append(md(["arm", "N", "tau", "method", "ratio"], trows))
    lines += ["", "#### Per-candidate overhead bound 1 + 2(d_in - 1)/(NT - 2), ITPD-S: graphs within bound", ""]
    lines.append(md(["cell", "graphs within"], [[k.rsplit("|", 1)[0], v] for k, v in agg.items() if k.endswith("overhead_bound_ok")]))
    open(os.path.join(out, "tables_oracle.md"), "w").write("\n".join(lines) + "\n")
    return agg


# ============================================================================================ F

def _share(target_max, N, T, target_inf=None):
    v = []
    for i, m in enumerate(target_max):
        if target_inf is not None and target_inf[i]:
            continue
        t = i // N + 1
        v.append(m / (N * t - 1))
    return float(np.mean(v)) if v else None


def _recall_at_fp(points, target):
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


def collect_finite(res, out, grid):
    fin = os.path.join(res, "finite", "window")
    lines = ["# ITPD-S and ITPD-S+ finite-data runs: tables", "",
             "Conditions: S2 window arm (stationary lag weights, spectral radius <= 0.9), tau = 1, d = 2, linear-Gaussian, "
             "Fisher-z, full history, the instances g00-g19 per (N, T) of the finite-data runs (sha1, data_seed and data_sha1 checked), data = "
             "first M rows (paired across M and methods). ITPD-S and ITPD-S+: equal = alpha_scr = alpha_shr; screen-and-clean = alpha_scr 0.10; alpha_shr = 0.01 "
             "unless stated. Stored rows (ITPD paper variant, ITPD_naive, full conditioning) at alpha 0.01, lazy. Common targets: "
             f"t <= (M - 3)/N. Medians [Q1, Q3] over {grid.graphs} graphs; totals are sums over the {grid.graphs} graphs.", ""]
    edge_cache = {}
    A_rows, B_rows, C_rows, D_rows, E_rows, H_rows = [], [], [], [], [], []
    fnbins = defaultdict(lambda: defaultdict(lambda: np.zeros((2, len(BINS) - 1), int)))   # (M, method) -> measure -> [fn, all]
    agg = {}
    for N in grid.Ns:
        for T in grid.Ts:
            cell = f"window_N{N}_T{T}_tau1_d2"
            nc = ncand(N, T)
            for g in range(grid.graphs):
                if (cell, g) not in edge_cache:
                    inst = load_instance(os.path.join(fin, "instances", cell, f"g{g:02d}.npz"))
                    st = observed_data.edge_strengths(inst["A"], observed_data.sigma_from_W(inst["W"]), N, T)
                    edge_cache[(cell, g)] = st
            for M in grid.Ms:
                ct = common_tmax(N, T, M)
                rows_by = defaultdict(list)      # method -> list of rows at alpha_shr = 0.01 (one per graph)
                sweep = defaultdict(lambda: defaultdict(lambda: np.zeros(3)))   # method -> alpha -> [tp, fp, fn] pooled
                for g in range(grid.graphs):
                    fh = os.path.join(res, "itpd_s", "single_pass", cell, f"g{g:02d}_M{M}.json")
                    fs = os.path.join(fin, cell, f"g{g:02d}_M{M}.json")
                    if not os.path.exists(fh):
                        continue
                    dh, ds = json.load(open(fh)), json.load(open(fs))
                    assert dh["sha1"] == ds["sha1"]
                    for r in dh["runs"] + [r for r in ds["runs"] if new_name(r["name"]) in CORE]:
                        nm = new_name(r["name"])
                        mc = r.get("metrics_common")
                        if mc is not None:
                            sweep[nm][r["alpha"]] += [mc["tp"], mc["fp"], mc["fn"]]
                        if abs(r["alpha"] - ALPHA) < 1e-12:
                            r = dict(r)
                            r["_g"] = g
                            rows_by[nm].append(r)
                if not rows_by:
                    continue
                full_cond_fp = sweep["full_conditioning"][ALPHA][1]
                full_cond_rec = sweep["full_conditioning"][ALPHA][0] / max(1, sweep["full_conditioning"][ALPHA][0] + sweep["full_conditioning"][ALPHA][2])
                for nm in FINITE_SHRINK + tuple(CORE):
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
                                   mq(tmaxs, "{:.0f}"), mq(shares), f"{np.mean(inf):.2f} ({sum(1 for x in inf if x > 0)}/{grid.graphs})"])
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
                        rec_at = _recall_at_fp(pts, full_cond_fp)
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
                    if nm.startswith("itpd_s"):
                        hs = [r["shrink_stats"] for r in rr]
                        agg[key].update({k: int(sum(h[k] for h in hs)) for k in
                                         ("excess", "lost", "sum_R", "sum_Aprime", "targets_Aprime", "shortcut_diff", "a_inf", "n_capped")})
                        H_rows.append([N, T, M, LABEL[nm], agg[key]["excess"], agg[key]["lost"], agg[key]["sum_Aprime"],
                                       agg[key]["targets_Aprime"], agg[key]["shortcut_diff"], agg[key]["a_inf"]])
                # error propagation: learned blanket vs true blanket
                for a, b in (("itpd_s", "itpd_s_true_blanket"), ("itpd_s_screen_clean", "itpd_s_true_blanket_screen_clean")):
                    ka, kb = f"{cell}|M{M}|{a}", f"{cell}|M{M}|{b}"
                    if ka in agg and kb in agg:
                        gs = sorted(set(agg[ka]["f1_by_g"]) & set(agg[kb]["f1_by_g"]))
                        df = [agg[kb]["f1_by_g"][g] - agg[ka]["f1_by_g"][g] for g in gs]
                        dt = [agg[ka]["tpc_by_g"][g] - agg[kb]["tpc_by_g"][g] for g in gs]
                        E_rows.append([N, T, M, "screen-and-clean" if a.endswith("screen_clean") else "equal",
                                       f"{np.mean(df):+.3f} [{np.min(df):+.3f}, {np.max(df):+.3f}]" if df else "-",
                                       f"{np.mean(dt):+.3f}", agg[ka]["excess"], agg[kb]["excess"], agg[ka]["lost"], agg[kb]["lost"],
                                       agg[ka]["fn_tot"], agg[kb]["fn_tot"], agg[ka]["fp_tot"], agg[kb]["fp_tot"]])
    lines += ["#### Tests and conditioning sets (all targets; a method with infeasible targets includes the tests issued before "
              "it abandoned them). tpc = unique / Sum|C|; share = mean over feasible targets of largest set / |C|", ""]
    lines.append(md(["N", "T", "M", "method", "unique tpc", "raw tpc", "unique tpc by step (A, B, C)", "largest set per graph",
                     "largest-set share", "infeasible-target share (graphs with any)"], A_rows))
    lines += ["", "#### Common targets (t <= (M-3)/N; comparable across methods), alpha (alpha_shr) 0.01", ""]
    lines.append(md(["N", "T", "M", "common t", "method", "recall", "precision", "F1", "FP total", "FN total"], B_rows))
    lines += ["", "#### Own-feasible targets (each method on the targets it could decide; NOT comparable across methods)", ""]
    lines.append(md(["N", "T", "M", "method", "recall", "precision", "F1", "infeasible share"], C_rows))
    lines += ["", "#### Matched FP, oracle-tuned diagnostic (needs the truth; not a usable procedure). Common targets, pooled over "
              f"the {grid.graphs} graphs. Recall of each swept method interpolated linearly in log(FP) at full conditioning's pooled FP at alpha 0.01; "
              "'not reached' if that FP lies outside the method's swept FP range (no extrapolation). Sweeps: ITPD-S "
              "alpha in 13 values 0.2-1e-6; ITPD-S screen-and-clean alpha_shr in 12 values 0.1-1e-6 (alpha_scr 0.1); ITPD, ITPD_naive 13 values. "
              "Last column: alpha:FP/recall.", ""]
    lines.append(md(["N", "T", "M", "method", "full conditioning FP", "full conditioning recall", "recall at full conditioning FP", "swept FP range", "curve"], D_rows))
    lines += ["", "#### Error propagation: learned-graph screening set (learned_blanket) vs true-blanket screening set (true_blanket), same alphas. "
              "dF1 = F1(true_blanket) - F1(learned_blanket), mean [min, max] over graphs (common targets); dtpc = tpc(learned_blanket) - tpc(true_blanket); "
              "excess = non-parents surviving the screening step; lost = true parents removed in the screening step (all targets)", ""]
    lines.append(md(["N", "T", "M", "alphas", "dF1", "dtpc", "excess learned", "excess oracle", "lost learned", "lost oracle",
                     "FN learned", "FN oracle", "FP learned", "FP oracle"], E_rows))
    lines += ["", f"#### ITPD-S and ITPD-S+ internals (totals over {grid.graphs} graphs, all targets): excess survivors, lost parents, re-check adds "
              "(A' members, targets with A' non-empty), always-verify vs shortcut output differences, screening-step tests that were "
              "infeasible (candidate kept)", ""]
    lines.append(md(["N", "T", "M", "method", "excess", "lost", "A' total", "targets with A'", "shortcut diff", "screening-step infeasible"], H_rows))
    lines += ["", f"#### False negatives by population |rho| (common targets, alpha 0.01, pooled over the {len(grid.cells)} (N, T) cell{'' if len(grid.cells) == 1 else 's'}): FN / true "
              "edges in the bin. marg = |corr(Z, Y)|, cond_x = |pcorr(Z, Y | X)|, cond_pa = |pcorr(Z, Y | other parents)|, "
              "rho_A = |pcorr(Z, Y | screening set actually used)| (ITPD-S and ITPD-S+ only)", ""]
    bl = [f"[{BINS[i]:g}, {BINS[i + 1]:g})" for i in range(len(BINS) - 1)]
    G_rows = []
    for (M, nm), d in sorted(fnbins.items(), key=lambda x: (x[0][0], x[0][1])):
        for mname, arr in d.items():
            G_rows.append([M, LABEL[nm], mname] + [f"{a}/{b}" for a, b in zip(arr[0], arr[1])] + [f"{arr[0].sum()}"])
            agg[f"fnbins|M{M}|{nm}|{mname}"] = arr.tolist()
    lines.append(md(["M", "method", "measure"] + bl + ["FN total"], G_rows))
    open(os.path.join(out, "tables_finite.md"), "w").write("\n".join(lines) + "\n")
    return agg


# ============================================================================================ curves and matched false positives

def load_all(res, grid):
    """data[(N, T, M)][g] = {"single_pass": json, "recheck": json, "finite": json}"""
    fin = os.path.join(res, "finite", "window")
    data = {}
    for N, T in grid.cells:
        cell = f"window_N{N}_T{T}_tau1_d2"
        for M in grid.Ms:
            per = {}
            for g in range(grid.graphs):
                d = {"finite": json.load(open(f"{fin}/{cell}/g{g:02d}_M{M}.json")),
                     "single_pass": json.load(open(f"{res}/itpd_s/single_pass/{cell}/g{g:02d}_M{M}.json")),
                     "recheck": json.load(open(f"{res}/itpd_s/recheck/{cell}/g{g:02d}_M{M}.json"))}
                for k in ("single_pass", "recheck"):
                    assert d[k]["sha1"] == d["finite"]["sha1"] and d[k]["common_tmax"] == d["finite"]["common_tmax"], (cell, g, M, k)
                    assert d[k]["n_true_edges_nonself"] == d["finite"]["n_true_edges_nonself"]
                per[g] = d
            data[(N, T, M)] = per
    return data


def rows_of(d, key, alpha=None):
    src, name, _ = METH[key]
    return [r for r in d[src]["runs"] if new_name(r["name"]) == name and (alpha is None or abs(r["alpha"] - alpha) < 1e-12)]


def pooled_curves(per, key):
    """{alpha: [fp, tp, fn, n_inf, n_targets]} pooled over graphs; asserts every graph has the same alpha set."""
    out = {}
    alph = None
    for g, d in per.items():
        rr = rows_of(d, key)
        a = sorted(r["alpha"] for r in rr)
        assert alph is None or a == alph, (key, g)
        alph = a
        for r in rr:
            m = r["metrics_common"]
            v = out.setdefault(r["alpha"], [0, 0, 0, 0, 0])
            v[0] += m["fp"]; v[1] += m["tp"]; v[2] += m["fn"]; v[3] += r["n_inf_targets"]; v[4] += r["n_targets"]
            assert r["common_bad"] == 0
    return out


def f1_at(recall, E, fp):
    tp = recall * E
    return 2 * tp / (2 * tp + fp + (E - tp))


def build_curves(res, out, grid):
    data = load_all(res, grid)
    agg = {"curves": {}, "matched": {}, "primary": {}}
    curves_rows, matched_rows = [], []
    for (N, T, M), per in data.items():
        cell = f"N{N}T{T}"
        C = {k: pooled_curves(per, k) for k in KEYS}
        E = C["itpd"][ALPHA][1] + C["itpd"][ALPHA][2]
        for k in KEYS:
            assert all(v[1] + v[2] == E for v in C[k].values()), (cell, M, k)
            for a, (fp, tp, fn, ni, nt) in sorted(C[k].items(), reverse=True):
                curves_rows.append([f"window_N{N}_T{T}_tau1_d2", N, T, M, METH[k][2], a, 0.1 if k in ("len", "rc_len") else a,
                                    fp, tp, fn, f"{tp / E:.6f}", f"{tp / (tp + fp) if tp + fp else 1.0:.6f}",
                                    f"{2 * tp / (2 * tp + fp + fn):.6f}", ni, nt])
        agg["curves"][f"{cell}|{M}"] = {k: {str(a): v for a, v in C[k].items()} for k in KEYS}
        for lv, (ref, _) in LEVELS.items():
            fp0 = C[ref][ALPHA][0]
            for k in KEYS:
                if k == ref:
                    v = C[k][ALPHA]
                    rec, st, br = v[1] / E, "own", (ALPHA, ALPHA)
                else:
                    rec, st, br = interp(C[k], fp0)
                f1 = None if rec is None else f1_at(rec, E, fp0)
                matched_rows.append([lv, f"window_N{N}_T{T}_tau1_d2", N, T, M, METH[k][2], fp0, st,
                                     "" if rec is None else f"{rec:.6f}", "" if f1 is None else f"{f1:.6f}", br[0], br[1]])
                agg["matched"][f"{lv}|{cell}|{M}|{k}"] = {"fp0": fp0, "status": st, "recall": rec, "f1": f1, "bracket": br}
    with open(os.path.join(out, "curves.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell", "N", "T", "M", "method", "alpha_or_alpha_shr", "alpha_scr", "fp", "tp", "fn", "recall", "precision", "f1",
                    "undecidable_targets", "targets"])
        w.writerows(curves_rows)
    with open(os.path.join(out, "matched_fp.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["level", "cell", "N", "T", "M", "method", "fp_level", "status", "recall", "f1", "alpha_lo", "alpha_hi"])
        w.writerows(matched_rows)
    return data, agg


# ---------------------------------------------------------------------------------------------------------- tables

def fmt_rec(m):
    if m["status"] == "ok" or m["status"] == "own":
        return f"{m['recall']:.3f}"
    return {"below": "<range", "above": ">range"}[m["status"]]


def matched_tables(agg, grid):
    lines = []
    for lv, (ref, desc) in LEVELS.items():
        rows = []
        for N, T in grid.cells:
            for M in grid.Ms:
                row = [N, T, M, agg["matched"][f"{lv}|N{N}T{T}|{M}|{ref}"]["fp0"]]
                for k in KEYS:
                    row.append(fmt_rec(agg["matched"][f"{lv}|N{N}T{T}|{M}|{k}"]))
                rows.append(row)
        lines += [f"##### Level ({lv}): FP matched at {desc}", "",
                  md(["N", "T", "M", "FP level (pooled)"] + [METH[k][2] for k in KEYS], rows), ""]
    # differences to ITPD at levels b and c, per M: one entry per cell
    cell_names = " / ".join(f"N{N}T{T}" for N, T in grid.cells)
    for lv in ("b", "c"):
        ref = LEVELS[lv][0]
        rows = []
        for M in grid.Ms:
            row = [M]
            for k in ("eq", "rc_eq", "len", "rc_len", "naive"):
                ds = []
                for N, T in grid.cells:
                    a = agg["matched"][f"{lv}|N{N}T{T}|{M}|{k}"]
                    b = agg["matched"][f"{lv}|N{N}T{T}|{M}|itpd"]
                    ds.append("n/r" if a["recall"] is None or b["recall"] is None else f"{a['recall'] - b['recall']:+.3f}")
                row.append(" / ".join(ds))
            rows.append(row)
        lines += [f"##### Recall minus ITPD's recall, both at level ({lv}) ({LEVELS[lv][1]}); cells {cell_names}", "",
                  md(["M"] + [METH[k][2] for k in ("eq", "rc_eq", "len", "rc_len", "naive")], rows), ""]
    return lines


def pm(vals):
    v = np.asarray([x for x in vals if x is not None], float)
    return "-" if len(v) == 0 else f"{np.median(v):.3f}"


def summary_tables(data, agg, grid):
    """Medians over the graphs at alpha (alpha_shr) = 0.01 on common targets: tpc, largest set, undecidable share, recall, precision, F1."""
    lines = []
    keys = KEYS
    T = {q: [] for q in ("tpc", "max", "inf", "rec", "prec", "f1", "fp", "fn")}
    for N, Tt in grid.cells:
        nc = ncand(N, Tt)
        for M in grid.Ms:
            per = data[(N, Tt, M)]
            ro = {q: [N, Tt, M] for q in T}
            for k in keys:
                rr = [rows_of(per[g], k, ALPHA)[0] for g in range(grid.graphs)]
                mc = [r["metrics_common"] for r in rr]
                tpc = [r["unique"] / nc for r in rr]
                mx = [max(r["target_max"]) for r in rr]
                inf = [r["n_inf_targets"] / r["n_targets"] for r in rr]
                ro["tpc"].append(pm(tpc))
                ro["max"].append(f"{np.median(mx):.1f}")
                ro["inf"].append(f"{np.mean(inf):.3f}")
                ro["rec"].append(pm([m["recall"] for m in mc]))
                ro["prec"].append(pm([m["precision"] for m in mc]))
                ro["f1"].append(pm([m["f1"] for m in mc]))
                ro["fp"].append(sum(m["fp"] for m in mc))
                ro["fn"].append(sum(m["fn"] for m in mc))
                agg["primary"][f"N{N}T{Tt}|{M}|{k}"] = {"tpc": float(np.median(tpc)), "max": float(np.median(mx)),
                                                         "inf": float(np.mean(inf)), "rec": float(np.median([m["recall"] for m in mc])),
                                                         "prec": float(np.median([m["precision"] for m in mc])),
                                                         "f1": float(np.median([m["f1"] for m in mc])),
                                                         "fp": int(sum(m["fp"] for m in mc)), "fn": int(sum(m["fn"] for m in mc)),
                                                         "tp": int(sum(m["tp"] for m in mc))}
            for q in T:
                T[q].append(ro[q])
    head = ["N", "T", "M"] + [METH[k][2] for k in keys]
    G = grid.graphs
    names = {"tpc": f"unique tests per candidate (median over {G} graphs; all targets; ITPD, ITPD_naive lazy)",
             "max": f"largest conditioning set issued per graph (median over {G} graphs)",
             "inf": f"undecidable-target share (mean over {G} graphs; all targets)",
             "rec": f"recall, common targets (median over {G} graphs)", "prec": "precision, common targets (median)",
             "f1": "F1, common targets (median)", "fp": f"FP total over {G} graphs, common targets",
             "fn": f"FN total over {G} graphs, common targets"}
    for q in ("tpc", "max", "inf", "rec", "prec", "f1", "fp", "fn"):
        lines += [f"##### {names[q]}", "", md(head, T[q]), ""]
    return lines


def undecidable_tables(data, agg, grid):
    """Undecidable targets / all targets, pooled over the graphs, across alpha (alpha_shr), at M 50, 100."""
    lines = []
    for N, Tt in grid.cells:
        for M in [M for M in UNDECIDABLE_MS if M in grid.Ms]:
            per = data[(N, Tt, M)]
            rows = []
            alphas = (0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001)
            for k in KEYS:
                C = pooled_curves(per, k)
                rows.append([METH[k][2]] + [("-" if a not in C else f"{C[a][3]}/{C[a][4]}") for a in alphas])
            lines += [f"##### Undecidable targets / targets, N{N} T{Tt} M{M} (pooled over {grid.graphs} graphs; alpha, or alpha_shr for the screen-and-clean variants)", "",
                      md(["method"] + [f"{a:g}" for a in alphas], rows), ""]
    return lines


def recheck_table(data, grid):
    rows = []
    for N, Tt in grid.cells:
        for M in grid.Ms:
            per = data[(N, Tt, M)]
            r = []
            for k in ("eq", "rc_eq", "len", "rc_len"):
                rr = [rows_of(per[g], k, ALPHA)[0] for g in range(grid.graphs)]
                fn = sum(x["metrics_common"]["fn"] for x in rr)
                fp = sum(x["metrics_common"]["fp"] for x in rr)
                r.append((fn, fp, rr))
            hs = [x["shrink_stats"] for x in r[1][2]]
            hl = [x["shrink_stats"] for x in r[3][2]]
            rows.append([N, Tt, M, f"{r[0][0]} -> {r[1][0]}", f"{r[0][1]} -> {r[1][1]}",
                         f"{sum(h['sum_Aprime'] for h in hs)} ({sum(h['targets_Aprime'] for h in hs)})", sum(h["shortcut_diff"] for h in hs),
                         f"{r[2][0]} -> {r[3][0]}", f"{r[2][1]} -> {r[3][1]}",
                         f"{sum(h['sum_Aprime'] for h in hl)} ({sum(h['targets_Aprime'] for h in hl)})", sum(h["shortcut_diff"] for h in hl)])
    return ["##### Re-check (always-verify) against ITPD-S, same alpha: FN and FP totals (common targets), A' members (targets with A' non-empty), "
            f"always-verify vs shortcut output differences (all targets), pooled over {grid.graphs} graphs", "",
            md(["N", "T", "M", "FN equal -> re-check", "FP equal -> re-check", "A' (targets) re-check equal", "shortcut diff equal", "FN screen-and-clean -> re-check",
                "FP screen-and-clean -> re-check", "A' (targets) re-check screen-and-clean", "shortcut diff screen-and-clean"], rows), ""]


def near_cancelled(res, data, grid):
    fin = os.path.join(res, "finite", "window")
    out = {}
    lines = []
    keys = KEYS
    for M in [M for M in NEAR_CANCELLED_MS if M in grid.Ms]:
        tot = 0
        miss = {k: set() for k in keys}
        for N, T in grid.cells:
            c = f"window_N{N}_T{T}_tau1_d2"
            for g in range(grid.graphs):
                inst = load_instance(f"{fin}/instances/{c}/g{g:02d}.npz")
                st = observed_data.edge_strengths(inst["A"], observed_data.sigma_from_W(inst["W"]), N, T)
                E = st["edges"]
                d = data[(N, T, M)][g]
                ct = d["finite"]["common_tmax"]
                near = {i for i in range(len(E)) if st["cond_x"][i] < 0.05 and E[i][1] // N <= ct}
                tot += len(near)
                for k in keys:
                    r = rows_of(d, k, ALPHA)[0]
                    miss[k] |= {(c, g, i) for i in r["fn_idx"] if i in near}
        out[M] = {"near": tot, **{k: len(v) for k, v in miss.items()},
                  "eq_also_itpd": len(miss["eq"] & miss["itpd"]), "rc_eq_also_itpd": len(miss["rc_eq"] & miss["itpd"])}
        lines.append([M, tot] + [len(miss[k]) for k in keys])
    return [f"##### Near-cancelled parents (|pcorr(Z, Y | X)| < 0.05; population values; alpha 0.01; common targets; {_cells_word(len(grid.cells))} pooled): "
            "missed / all", "", md(["M", "near-cancelled edges"] + [METH[k][2] for k in keys], lines), ""], out


def add_args(ap):
    ap.add_argument("--results", default=RESULTS)
    ap.add_argument("--out", default=None, help="output directory (default: <results>/itpd_s)")
    ap.add_argument("--only", default="oracle,finite", help="comma list of the parts to write: oracle, finite, curves")
    ap.add_argument("--arm", default="time,window", help="oracle part only")
    ap.add_argument("--N", default="10,20")
    ap.add_argument("--T", default="8,16")
    ap.add_argument("--tau", default="1,2", help="oracle part only (the finite-data runs have tau 1)")
    ap.add_argument("--M", default="50,100,200,500,2000", help="finite and curves parts")
    ap.add_argument("--graphs", type=int, default=20)


def run(a):
    grid = Grid(a.arm.split(","), ints(a.N), ints(a.T), ints(a.tau), ints(a.M), a.graphs)
    out = a.out or os.path.join(a.results, "itpd_s")
    os.makedirs(out, exist_ok=True)
    parts = a.only.split(",")
    agg = {}
    if "oracle" in parts:
        agg["oracle"] = collect_oracle(a.results, out, grid)
    if "finite" in parts:
        agg["finite"] = collect_finite(a.results, out, grid)
    if agg:
        json.dump(agg, open(os.path.join(out, "aggregates.json"), "w"), default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    if "curves" in parts:
        data, agg = build_curves(a.results, out, grid)
        L = ["# ITPD-S and ITPD-S+ tables", "",
             f"Conditions: S2 window arm, tau = 1, d = 2, linear-Gaussian, Fisher-z, full history, instances g00-g{grid.graphs - 1:02d} per (N, T) of the finite-data runs, data = first M rows "
             f"(sha1, data_seed, data_sha1 checked; the ITPD-S and ITPD-S+ runs join the same tasks; common_tmax and true-edge counts equal). Pooled = sums over the {grid.graphs} "
             "graphs, common targets t <= (M - 3)/N. ITPD-S = single pass, learned-blanket screening set, alpha_scr = alpha_shr; ITPD-S+ = the same plus the "
             "always-verify re-check (alpha 0.01 in both steps unless a sweep is stated); ITPD-S screen-and-clean = alpha_scr 0.1 with alpha_shr swept; ITPD-S+ screen-and-clean = the same "
             "with the re-check. Sweeps: 13 alphas (0.2 to 1e-6) for the equal-alpha variants, ITPD, ITPD_naive and full conditioning; 12 alpha_shr (0.1 to 1e-6) for the screen-and-clean variants. "
             "Matched-FP tables are an oracle-tuned diagnostic (the level is chosen with the truth).", ""]
        L += ["#### Matched FP (oracle-tuned diagnostic): recall interpolated in log FP", ""] + matched_tables(agg, grid)
        L += ["#### Alpha = 0.01 operating point", ""] + summary_tables(data, agg, grid)
        L += ["#### Undecidable targets across alpha", ""] + undecidable_tables(data, agg, grid)
        L += ["#### Re-check cost and effect", ""] + recheck_table(data, grid)
        nl, nc = near_cancelled(a.results, data, grid)
        agg["near_cancelled"] = nc
        L += ["#### Near-cancelled parents", ""] + nl
        open(os.path.join(out, "tables.md"), "w").write("\n".join(L) + "\n")
        json.dump(agg, open(os.path.join(out, "aggregates_curves.json"), "w"),
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("wrote", out)
