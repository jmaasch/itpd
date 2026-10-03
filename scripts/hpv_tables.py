"""Tables for the HPV runs: pooled recall-false-positive curves and recall at matched false positives, next to ITPD, ITPD_naive and
the order-based baseline. Reads, under the results directory R (`--results`, default $ITPD_RESULTS or ./results):
  R/finite/window/<cell>/g<g>_M<M>.json (+ instances)   itpd.run_finite_data rows of ITPD, ITPD_naive and order_based
  R/hpv/single_pass/<cell>/g<g>_M<M>.json               itpd.run_hpv finite: hpv_single_pass, hpv_single_pass_lenient
  R/hpv/safe/<cell>/g<g>_M<M>.json                      itpd.run_hpv finite --specs hpv_safe,hpv_safe_lenient
Writes in `--out` (default R/hpv):
  curves.csv        pooled recall-FP curves, one row per (cell, M, method, alpha)
  matched_fp.csv    recall and F1 interpolated in log FP at the declared FP levels (oracle-tuned diagnostic)
  tables.md         the tables
  aggregates_curves.json   numbers behind the tables

    python scripts/hpv_tables.py [--results DIR] [--out DIR]

Pooled = sums of tp, fp, fn over the 20 graphs of a cell, common targets (t <= (M - 3)/N). Interpolation: recall linear in log FP
(FP clamped at 0.5) between the two neighbouring alphas of the grid, sorted by FP; no extrapolation (a level outside the swept FP
range is reported as below / above the range).
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

from itpd import observed_data
from itpd.instances import load_instance
from itpd.methods_registry import OLD_TO_NEW

RES = os.environ.get("ITPD_RESULTS", "results")
CELLS = [(10, 8), (10, 16), (20, 8), (20, 16)]
MS = [50, 100, 200, 500, 2000]
NG = 20
PRIMARY = 0.01
# key -> (source, name in the JSON, display label)
METH = {
    "eq": ("single_pass", "hpv_single_pass", "HPV-eq"),
    "rc_eq": ("safe", "hpv_safe", "HPV-safe eq"),
    "len": ("single_pass", "hpv_single_pass_lenient", "HPV-len"),
    "rc_len": ("safe", "hpv_safe_lenient", "HPV-safe len"),
    "itpd": ("finite", "itpd", "ITPD"),
    "naive": ("finite", "itpd_naive", "ITPD_naive"),
    "order": ("finite", "order_based", "order"),
}
KEYS = list(METH)
LEVELS = {"a": ("order", "order's FP at alpha 0.01"), "b": ("eq", "HPV-eq's own FP (alpha 0.01)"),
          "c": ("itpd", "ITPD's own FP (alpha 0.01)"), "d": ("rc_eq", "HPV-safe eq's own FP (alpha 0.01)")}


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def ncand(N, T):
    return N * sum(N * t - 1 for t in range(1, T))


def load_all(res):
    """data[(N, T, M)][g] = {"single_pass": json, "safe": json, "finite": json}"""
    fin = os.path.join(res, "finite", "window")
    data = {}
    for N, T in CELLS:
        cell = f"window_N{N}_T{T}_tau1_d2"
        for M in MS:
            per = {}
            for g in range(NG):
                d = {"finite": json.load(open(f"{fin}/{cell}/g{g:02d}_M{M}.json")),
                     "single_pass": json.load(open(f"{res}/hpv/single_pass/{cell}/g{g:02d}_M{M}.json")),
                     "safe": json.load(open(f"{res}/hpv/safe/{cell}/g{g:02d}_M{M}.json"))}
                for k in ("single_pass", "safe"):
                    assert d[k]["sha1"] == d["finite"]["sha1"] and d[k]["common_tmax"] == d["finite"]["common_tmax"], (cell, g, M, k)
                    assert d[k]["n_true_edges_nonself"] == d["finite"]["n_true_edges_nonself"]
                per[g] = d
            data[(N, T, M)] = per
    return data


def rows_of(d, key, alpha=None):
    src, name, _ = METH[key]
    return [r for r in d[src]["runs"] if OLD_TO_NEW.get(r["name"], r["name"]) == name and (alpha is None or abs(r["alpha"] - alpha) < 1e-12)]


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


def interp(curve, fp0):
    """(recall, tp-free status, bracket alphas). status 'ok' | 'below' | 'above' (level outside the swept FP range)."""
    pts = sorted((v[0], v[1] / (v[1] + v[2]), a) for a, v in curve.items())
    fps = [p[0] for p in pts]
    if fp0 < fps[0]:
        return None, "below", (pts[0][2], pts[0][2])
    if fp0 > fps[-1]:
        return None, "above", (pts[-1][2], pts[-1][2])
    for (f1, r1, a1), (f2, r2, a2) in zip(pts, pts[1:]):
        if f1 <= fp0 <= f2:
            if f1 == f2:
                return max(r1, r2), "ok", (a1, a2)
            lf1, lf2 = np.log(max(f1, 0.5)), np.log(max(f2, 0.5))
            w = (np.log(max(fp0, 0.5)) - lf1) / (lf2 - lf1) if lf2 != lf1 else 0.0
            return r1 + w * (r2 - r1), "ok", (a1, a2)
    return pts[-1][1], "ok", (pts[-1][2], pts[-1][2])


def f1_at(recall, E, fp):
    tp = recall * E
    return 2 * tp / (2 * tp + fp + (E - tp))


def build(res, out):
    data = load_all(res)
    agg = {"curves": {}, "matched": {}, "primary": {}}
    curves_rows, matched_rows = [], []
    for (N, T, M), per in data.items():
        cell = f"N{N}T{T}"
        C = {k: pooled_curves(per, k) for k in KEYS}
        E = C["itpd"][PRIMARY][1] + C["itpd"][PRIMARY][2]
        for k in KEYS:
            assert all(v[1] + v[2] == E for v in C[k].values()), (cell, M, k)
            for a, (fp, tp, fn, ni, nt) in sorted(C[k].items(), reverse=True):
                curves_rows.append([f"window_N{N}_T{T}_tau1_d2", N, T, M, METH[k][2], a, 0.1 if k in ("len", "rc_len") else a,
                                    fp, tp, fn, f"{tp / E:.6f}", f"{tp / (tp + fp) if tp + fp else 1.0:.6f}",
                                    f"{2 * tp / (2 * tp + fp + fn):.6f}", ni, nt])
        agg["curves"][f"{cell}|{M}"] = {k: {str(a): v for a, v in C[k].items()} for k in KEYS}
        for lv, (ref, _) in LEVELS.items():
            fp0 = C[ref][PRIMARY][0]
            for k in KEYS:
                if k == ref:
                    v = C[k][PRIMARY]
                    rec, st, br = v[1] / E, "own", (PRIMARY, PRIMARY)
                else:
                    rec, st, br = interp(C[k], fp0)
                f1 = None if rec is None else f1_at(rec, E, fp0)
                matched_rows.append([lv, f"window_N{N}_T{T}_tau1_d2", N, T, M, METH[k][2], fp0, st,
                                     "" if rec is None else f"{rec:.6f}", "" if f1 is None else f"{f1:.6f}", br[0], br[1]])
                agg["matched"][f"{lv}|{cell}|{M}|{k}"] = {"fp0": fp0, "status": st, "recall": rec, "f1": f1, "bracket": br}
    with open(os.path.join(out, "curves.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell", "N", "T", "M", "method", "alpha_or_alpha_B", "alpha_A", "fp", "tp", "fn", "recall", "precision", "f1",
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


def matched_tables(agg):
    lines = []
    for lv, (ref, desc) in LEVELS.items():
        rows = []
        for N, T in CELLS:
            for M in MS:
                row = [N, T, M, agg["matched"][f"{lv}|N{N}T{T}|{M}|{ref}"]["fp0"]]
                for k in KEYS:
                    row.append(fmt_rec(agg["matched"][f"{lv}|N{N}T{T}|{M}|{k}"]))
                rows.append(row)
        lines += [f"##### Level ({lv}): FP matched at {desc}", "",
                  table(["N", "T", "M", "FP level (pooled)"] + [METH[k][2] for k in KEYS], rows), ""]
    # differences to ITPD at levels b and c, per M: four cells
    for lv in ("b", "c"):
        ref = LEVELS[lv][0]
        rows = []
        for M in MS:
            row = [M]
            for k in ("eq", "rc_eq", "len", "rc_len", "naive"):
                ds = []
                for N, T in CELLS:
                    a = agg["matched"][f"{lv}|N{N}T{T}|{M}|{k}"]
                    b = agg["matched"][f"{lv}|N{N}T{T}|{M}|itpd"]
                    ds.append("n/r" if a["recall"] is None or b["recall"] is None else f"{a['recall'] - b['recall']:+.3f}")
                row.append(" / ".join(ds))
            rows.append(row)
        lines += [f"##### Recall minus ITPD's recall, both at level ({lv}) ({LEVELS[lv][1]}); cells N10T8 / N10T16 / N20T8 / N20T16", "",
                  table(["M", "HPV-eq", "HPV-safe eq", "HPV-len", "HPV-safe len", "ITPD_naive"], rows), ""]
    return lines


def pm(vals):
    v = np.asarray([x for x in vals if x is not None], float)
    return "-" if len(v) == 0 else f"{np.median(v):.3f}"


def summary_tables(data, agg):
    """Medians over 20 graphs at alpha (alpha_B) = 0.01 on common targets: tpc, largest set, undecidable share, recall, precision, F1."""
    lines = []
    keys = KEYS
    T = {q: [] for q in ("tpc", "max", "inf", "rec", "prec", "f1", "fp", "fn")}
    for N, Tt in CELLS:
        nc = ncand(N, Tt)
        for M in MS:
            per = data[(N, Tt, M)]
            ro = {q: [N, Tt, M] for q in T}
            for k in keys:
                rr = [rows_of(per[g], k, PRIMARY)[0] for g in range(NG)]
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
    names = {"tpc": "unique tests per candidate (median over 20 graphs; all targets; ITPD, ITPD_naive lazy)",
             "max": "largest conditioning set issued per graph (median over 20 graphs)",
             "inf": "undecidable-target share (mean over 20 graphs; all targets)",
             "rec": "recall, common targets (median over 20 graphs)", "prec": "precision, common targets (median)",
             "f1": "F1, common targets (median)", "fp": "FP total over 20 graphs, common targets",
             "fn": "FN total over 20 graphs, common targets"}
    for q in ("tpc", "max", "inf", "rec", "prec", "f1", "fp", "fn"):
        lines += [f"##### {names[q]}", "", table(head, T[q]), ""]
    return lines


def undecidable_tables(data, agg):
    """Undecidable targets / all targets, pooled over 20 graphs, across alpha (alpha_B), for N20 T16 and the other cells at M 50, 100."""
    lines = []
    for N, Tt in CELLS:
        for M in (50, 100):
            per = data[(N, Tt, M)]
            rows = []
            alphas = (0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001)
            for k in KEYS:
                C = pooled_curves(per, k)
                rows.append([METH[k][2]] + [("-" if a not in C else f"{C[a][3]}/{C[a][4]}") for a in alphas])
            lines += [f"##### Undecidable targets / targets, N{N} T{Tt} M{M} (pooled over 20 graphs; alpha, or alpha_B for the len family)", "",
                      table(["method"] + [f"{a:g}" for a in alphas], rows), ""]
    return lines


def recheck_table(data):
    rows = []
    for N, Tt in CELLS:
        for M in MS:
            per = data[(N, Tt, M)]
            r = []
            for k in ("eq", "rc_eq", "len", "rc_len"):
                rr = [rows_of(per[g], k, PRIMARY)[0] for g in range(NG)]
                fn = sum(x["metrics_common"]["fn"] for x in rr)
                fp = sum(x["metrics_common"]["fp"] for x in rr)
                r.append((fn, fp, rr))
            hs = [x["hpv_stats"] for x in r[1][2]]
            hl = [x["hpv_stats"] for x in r[3][2]]
            rows.append([N, Tt, M, f"{r[0][0]} -> {r[1][0]}", f"{r[0][1]} -> {r[1][1]}",
                         f"{sum(h['sum_Aprime'] for h in hs)} ({sum(h['targets_Aprime'] for h in hs)})", sum(h["shortcut_diff"] for h in hs),
                         f"{r[2][0]} -> {r[3][0]}", f"{r[2][1]} -> {r[3][1]}",
                         f"{sum(h['sum_Aprime'] for h in hl)} ({sum(h['targets_Aprime'] for h in hl)})", sum(h["shortcut_diff"] for h in hl)])
    return ["##### Re-check (always-verify) against single-pass HPV, same alpha: FN and FP totals (common targets), A' members (targets with A' non-empty), "
            "always-verify vs shortcut output differences (all targets), pooled over 20 graphs", "",
            table(["N", "T", "M", "FN eq -> safe eq", "FP eq -> safe eq", "A' (targets) safe eq", "shortcut diff", "FN len -> safe len",
                   "FP len -> safe len", "A' (targets) safe len", "shortcut diff len"], rows), ""]


def near_cancelled(res, data):
    fin = os.path.join(res, "finite", "window")
    out = {}
    lines = []
    keys = KEYS
    for M in (200, 500, 2000):
        tot = 0
        miss = {k: set() for k in keys}
        for N, T in CELLS:
            c = f"window_N{N}_T{T}_tau1_d2"
            for g in range(NG):
                inst = load_instance(f"{fin}/instances/{c}/g{g:02d}.npz")
                st = observed_data.edge_strengths(inst["A"], observed_data.sigma_from_W(inst["W"]), N, T)
                E = st["edges"]
                d = data[(N, T, M)][g]
                ct = d["finite"]["common_tmax"]
                near = {i for i in range(len(E)) if st["cond_x"][i] < 0.05 and E[i][1] // N <= ct}
                tot += len(near)
                for k in keys:
                    r = rows_of(d, k, PRIMARY)[0]
                    miss[k] |= {(c, g, i) for i in r["fn_idx"] if i in near}
        out[M] = {"near": tot, **{k: len(v) for k, v in miss.items()},
                  "eq_also_itpd": len(miss["eq"] & miss["itpd"]), "rc_eq_also_itpd": len(miss["rc_eq"] & miss["itpd"])}
        lines.append([M, tot] + [len(miss[k]) for k in keys])
    return ["##### Near-cancelled parents (|pcorr(Z, Y | X)| < 0.05; population values; alpha 0.01; common targets; four cells pooled): "
            "missed / all", "", table(["M", "near-cancelled edges"] + [METH[k][2] for k in keys], lines), ""], out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=RES)
    ap.add_argument("--out", default=None, help="output directory (default: <results>/hpv)")
    a = ap.parse_args()
    out = a.out or os.path.join(a.results, "hpv")
    os.makedirs(out, exist_ok=True)
    data, agg = build(a.results, out)
    L = ["# HPV tables", "",
         "Conditions: S2 window arm, tau = 1, d = 2, linear-Gaussian, Fisher-z, full history, instances g00-g19 per (N, T) of the finite-data runs, data = first M rows "
         "(sha1, data_seed, data_sha1 checked; the HPV runs join the same tasks; common_tmax and true-edge counts equal). Pooled = sums over the 20 "
         "graphs, common targets t <= (M - 3)/N. HPV-eq = single pass, learned-blanket hint, alpha_A = alpha_B; HPV-safe eq = the same plus the "
         "always-verify re-check (alpha 0.01 in both phases unless a sweep is stated); HPV-len = alpha_A 0.1 with alpha_B swept; HPV-safe len = the same "
         "with the re-check. Sweeps: 13 alphas (0.2 to 1e-6) for eq, safe eq, ITPD, ITPD_naive, order; 12 alpha_B (0.1 to 1e-6) for the len family. "
         "Matched-FP tables are an oracle-tuned diagnostic (the level is chosen with the truth).", ""]
    L += ["#### Matched FP (oracle-tuned diagnostic): recall interpolated in log FP", ""] + matched_tables(agg)
    L += ["#### Alpha = 0.01 operating point", ""] + summary_tables(data, agg)
    L += ["#### Undecidable targets across alpha", ""] + undecidable_tables(data, agg)
    L += ["#### Re-check cost and effect", ""] + recheck_table(data)
    nl, nc = near_cancelled(a.results, data)
    agg["near_cancelled"] = nc
    L += ["#### Near-cancelled parents", ""] + nl
    open(os.path.join(out, "tables.md"), "w").write("\n".join(L) + "\n")
    json.dump(agg, open(os.path.join(out, "aggregates_curves.json"), "w"),
              default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("wrote", out)


if __name__ == "__main__":
    main()
