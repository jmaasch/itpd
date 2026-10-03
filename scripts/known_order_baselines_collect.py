"""Tables of the known-order baselines: oracle check (IAMB, ITPD + marginal-first), finite-data comparison on the window instances
of the finite-data runs (IAMB, ITPD + marginal-first, lasso next to the ITPD, ITPD_naive and full-conditioning rows and the blanket-screened shrink rows),
paired per-graph differences, matched-FP, wall-clock. Reads, under the results directory R ($ITPD_RESULTS or ./results):
R/baselines/{oracle,finite,lasso_ebic_fixed,timing} (itpd.run_known_order_baselines), R/finite/window and R/oracle (the earlier
runs) and R/blanket_screened_shrink/{single_pass,recheck,oracle} (itpd.run_blanket_screened_shrink, optional). Writes R/baselines/TABLES.md and aggregates.json.

    python scripts/known_order_baselines_collect.py [--out DIR]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from blanket_screened_shrink_tables import interp, ncand, table   # noqa: E402
from itpd.methods_registry import OLD_TO_NEW   # noqa: E402

RES = os.environ.get("ITPD_RESULTS", "results")
CELLS = [(10, 8, 50), (20, 8, 50), (10, 16, 20), (20, 16, 20)]
MS = [50, 100, 200, 500, 2000]
A0 = 0.01
SRC = {"iamb": ("baselines", "iamb_known_order"), "itpd_mf": ("baselines", "itpd_marginal_first"), "itpd": ("finite", "itpd"), "naive": ("finite", "itpd_naive"),
       "full_conditioning": ("finite", "full_conditioning"), "blanket_screened_shrink_recheck": ("recheck", "blanket_screened_shrink_recheck"),
       "blanket_screened_shrink": ("single_pass", "blanket_screened_shrink")}
LASSO = {"lasso_cv": ("baselines", "lasso_cv"), "lasso_ebic": ("lasso", "lasso_ebic"), "lasso_fixed": ("lasso", "lasso_fixed")}
LABEL = {"lasso_ebic": "lasso EBIC (headline)", "lasso_fixed": "lasso fixed penalty", "iamb": "IAMB", "itpd_mf": "ITPD+mf", "itpd": "ITPD", "naive": "ITPD_naive", "full_conditioning": "full conditioning",
         "blanket_screened_shrink_recheck": "blanket-screened shrink re-check", "blanket_screened_shrink": "blanket-screened shrink", "lasso_cv": "lasso CV", "lasso_path": "lasso path"}
rng = np.random.default_rng(0)


def rename_names(d):
    """Result files written before the method-name rename carry old names (row `name`, keys of the oracle rows): translate."""
    new = lambda n: OLD_TO_NEW.get(n, n)
    for lst in (d.get("runs"), d.get("rows"), (d.get("lasso") or {}).get("rows")):
        for r in lst or []:
            if isinstance(r, dict) and "name" in r:
                r["name"] = new(r["name"])
    for i, r in enumerate(d.get("graphs") or []):           # oracle files: rows keyed by method name
        r = {new(k): v for k, v in r.items()}
        if "stored_oracle_run" in r:
            r["stored_oracle_run"] = {new(k): v for k, v in r["stored_oracle_run"].items()}
        d["graphs"][i] = r
    return d


def jl(p):
    with open(p) as f:
        return rename_names(json.load(f))


def run_of(d, key, alpha=A0):
    src, name = SRC[key]
    r = [q for q in d[src]["runs"] if q["name"] == name and abs(q["alpha"] - alpha) < 1e-12]
    return r[0] if r else None


def load(out):
    """data[(N, T, M)][g] = {"baselines": json, "finite": json, "lasso": json?, "single_pass": json?, "recheck": json?}"""
    data = {}
    for N, T, ng in CELLS:
        cell = f"window_N{N}_T{T}_tau1_d2"
        for M in MS:
            per = {}
            for g in range(ng):
                bp = f"{out}/finite/{cell}/g{g:02d}_M{M}.json"
                if not os.path.exists(bp):
                    continue
                d = {"baselines": jl(bp), "finite": jl(f"{RES}/finite/window/{cell}/g{g:02d}_M{M}.json")}
                pl = f"{out}/lasso_ebic_fixed/{cell}/g{g:02d}_M{M}.json"
                if os.path.exists(pl):
                    d["lasso"] = jl(pl)
                for k in ("single_pass", "recheck"):
                    pk = f"{RES}/blanket_screened_shrink/{k}/{cell}/g{g:02d}_M{M}.json"
                    if g < 20 and os.path.exists(pk):
                        d[k] = jl(pk)
                for k in ("finite", "single_pass", "recheck"):
                    if k in d:
                        assert d[k]["sha1"] == d["baselines"]["sha1"] and d[k]["common_tmax"] == d["baselines"]["common_tmax"], (cell, g, M, k)
                per[g] = d
            data[(N, T, M)] = per
    return data


def lrow(d, key):
    src, name = LASSO[key]
    if src == "baselines":
        L = d["baselines"].get("lasso")
        rows = [r for r in (L["rows"] if L else []) if r["name"] == name]
    else:
        rows = [r for r in d.get("lasso", {}).get("rows", []) if r["name"] == name]
    return rows[0] if rows else None


def have(d, key):
    return (lrow(d, key) is not None) if key in LASSO else (SRC[key][0] in d)


def med(v):
    v = list(v)
    return float(np.median(v)) if len(v) else float("nan")


def boot_ci(diff, n=2000):
    diff = np.asarray(diff, dtype=float)
    if len(diff) < 2:
        return float("nan"), float("nan")
    idx = rng.integers(0, len(diff), (n, len(diff)))
    m = diff[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def f3(x):
    return "-" if x is None or x != x else f"{x:.3f}"


# --------------------------------------------------------------------------------------------- finite tables

def per_graph(per, key, field):
    out = {}
    for g, d in per.items():
        if not have(d, key):
            continue
        r = lrow(d, key) if key in LASSO else run_of(d, key)
        if r is None:
            continue
        if field in ("recall", "precision", "f1", "fp"):
            out[g] = r["metrics_common"][field]
        elif field == "n_sel":
            out[g] = r["metrics_common"]["tp"] + r["metrics_common"]["fp"]
        elif key in LASSO:
            continue
        else:
            out[g] = r[field]
    return out


def summary_tables(data, subset):
    """subset 'all' (every graph of the cell, no blanket-screened shrink) or 'shrink' (graphs 0-19 incl. blanket-screened shrink rows)."""
    L = []
    keys = ["iamb", "itpd_mf", "itpd", "naive", "full_conditioning"] + (["blanket_screened_shrink_recheck", "blanket_screened_shrink"] if subset == "shrink" else [])
    for N, T, ng in CELLS:
        rows = []
        for M in MS:
            per = {g: d for g, d in data[(N, T, M)].items() if subset == "all" or g < 20}
            if not per:
                continue
            nc = ncand(N, T)
            for k in keys + list(LASSO):
                if k in LASSO:
                    rec = per_graph(per, k, "recall"); pre = per_graph(per, k, "precision")
                    f1 = per_graph(per, k, "f1"); fp = per_graph(per, k, "fp"); ns = per_graph(per, k, "n_sel")
                    if not rec:
                        continue
                    rows.append([M, LABEL[k], len(rec), "-", "-", "0 / " + str(len(rec) * (T - 1) * N), f3(med(rec.values())),
                                 f3(med(pre.values())), f3(med(f1.values())), f"{med(fp.values()):.0f}", f"{med(ns.values()):.0f}", "-"])
                    continue
                u = per_graph(per, k, "unique")
                if not u:
                    continue
                tm = {g: max(run_of(d, k)["target_max"]) for g, d in per.items() if have(d, k) and run_of(d, k)}
                ni = per_graph(per, k, "n_inf_targets"); nt = per_graph(per, k, "n_targets")
                rec = per_graph(per, k, "recall"); pre = per_graph(per, k, "precision")
                f1 = per_graph(per, k, "f1"); fp = per_graph(per, k, "fp"); ns = per_graph(per, k, "n_sel")
                rows.append([M, LABEL[k], len(u), f"{sum(u.values()) / (len(u) * nc):.3f}", f"{med(tm.values()):.0f} / {max(tm.values())}",
                             f"{sum(ni.values())} / {sum(nt.values())}", f3(med(rec.values())), f3(med(pre.values())),
                             f3(med(f1.values())), f"{med(fp.values()):.0f}", f"{med(ns.values()):.0f}", f"{med(list(u.values())):.0f}"])
        L += [f"\n#### N {N}, T {T}, " + ("all graphs of the cell (" + str(ng) + "), without blanket-screened shrink" if subset == "all" else "graphs 0-19, with the blanket-screened shrink rows") + "\n",
              table(["M", "method", "graphs", "unique tests per candidate", "largest set (median / max over graphs of the per-graph max)",
                     "undecidable targets / targets (all graphs)", "median recall", "median precision", "median F1", "median FP",
                     "median selected edges (common targets)", "median unique tests"], rows)]
    return L


def paired_tables(data):
    L = []
    pairs = [("iamb", "itpd", "all"), ("itpd_mf", "itpd", "all"), ("iamb", "full_conditioning", "all"), ("iamb", "blanket_screened_shrink_recheck", "shrink"),
             ("iamb", "blanket_screened_shrink", "shrink"), ("itpd", "blanket_screened_shrink_recheck", "shrink"), ("lasso_ebic", "itpd", "all"),
             ("lasso_ebic", "iamb", "all"), ("lasso_ebic", "blanket_screened_shrink_recheck", "shrink"), ("lasso_cv", "lasso_ebic", "all")]
    for a, b, sub in pairs:
        rows = []
        for N, T, ng in CELLS:
            for M in MS:
                per = {g: d for g, d in data[(N, T, M)].items() if sub == "all" or g < 20}
                ga, gb = (per_graph(per, a, "f1"), per_graph(per, b, "f1"))
                gs = sorted(set(ga) & set(gb))
                if not gs:
                    continue
                cells = []
                for fld in ("recall", "precision", "f1"):
                    xa, xb = per_graph(per, a, fld), per_graph(per, b, fld)
                    diff = np.array([xa[g] - xb[g] for g in gs])
                    lo, hi = boot_ci(diff)
                    cells.append(f"{diff.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]; {int((diff > 1e-12).sum())}/{int((diff < -1e-12).sum())}")
                if a in LASSO or b in LASSO:
                    rows.append([f"N{N} T{T}", M, len(gs)] + cells + ["-"])
                    continue
                ua, ub = per_graph(per, a, "unique"), per_graph(per, b, "unique")
                ratio = np.array([ua[g] / ub[g] for g in gs])
                lo, hi = boot_ci(ratio)
                rows.append([f"N{N} T{T}", M, len(gs)] + cells + [f"{ratio.mean():.3f} [{lo:.3f}, {hi:.3f}]"])
        L += [f"\n#### {LABEL[a]} minus {LABEL[b]} per graph " + ("(graphs 0-19)" if sub == "shrink" else "(all graphs of the cell)") + "\n",
              table(["cell", "M", "graphs", "recall: mean diff [95% bootstrap CI over graphs]; graphs higher/lower",
                     "precision: same", "F1: same", "unique tests ratio, mean [CI]"], rows)]
    return L


def pooled(per, key, lam=False):
    out = {}
    for g, d in per.items():
        if key in ("lasso_path",):
            L = d["baselines"].get("lasso")
            if not L:
                continue
            for r in [r for r in L["rows"] if r["name"] == "lasso_path"]:
                m = r["metrics_common"]
                v = out.setdefault(r["lambda"], [0, 0, 0, 0, 0]); v[0] += m["fp"]; v[1] += m["tp"]; v[2] += m["fn"]
            continue
        if not have(d, key):
            continue
        for r in [q for q in d[SRC[key][0]]["runs"] if q["name"] == SRC[key][1]]:
            m = r["metrics_common"]
            v = out.setdefault(r["alpha"], [0, 0, 0, 0, 0]); v[0] += m["fp"]; v[1] += m["tp"]; v[2] += m["fn"]
    return out


def matched_tables(data):
    L = []
    levels = {"a": ("full_conditioning", "full conditioning's FP at alpha 0.01"), "c": ("itpd", "ITPD's FP at alpha 0.01"), "e": ("iamb", "IAMB's FP at alpha 0.01"),
              "d": ("blanket_screened_shrink_recheck", "blanket-screened shrink re-check's FP at alpha 0.01"), "f": ("lasso_ebic", "EBIC lasso's own FP"),
              "g": ("lasso_fixed", "fixed-penalty lasso's own FP"), "h": ("lasso_cv", "CV lasso's own FP")}
    for sub in ("all", "shrink"):
        rows = []
        mk = ["iamb", "itpd", "itpd_mf", "full_conditioning", "naive", "lasso_path"] + (["blanket_screened_shrink_recheck"] if sub == "shrink" else [])
        for N, T, ng in CELLS:
            for M in MS:
                per = {g: d for g, d in data[(N, T, M)].items() if sub == "all" or g < 20}
                if not per:
                    continue
                C = {k: pooled(per, k) for k in mk}
                if not all(C[k] for k in mk):
                    continue
                for lv, (ref, desc) in levels.items():
                    if ref in LASSO:
                        lf = per_graph(per, ref, "fp")
                        if not lf:
                            continue
                        fp0 = sum(lf.values())
                    else:
                        if ref not in C and ref != "iamb":
                            continue
                        if lv == "d" and sub != "shrink":
                            continue
                        fp0 = C[ref][A0][0]
                    cells = []
                    for k in mk:
                        r, st, _ = interp(C[k], fp0)
                        cells.append(f3(r) if st == "ok" else ("below range" if st == "below" else "above range"))
                    rows.append([f"N{N} T{T}", M, lv, f"{fp0}"] + cells)
        L += [f"\n#### Matched false positives, " + ("all graphs of the cell" if sub == "all" else "graphs 0-19 incl. blanket-screened shrink re-check") +
              ": recall (pooled over graphs, common targets) interpolated in log FP on each method's sweep (alpha for the CI methods, 13 fixed lambdas for the lasso); levels: a = full conditioning's FP at 0.01, c = ITPD's, e = IAMB's"
              + (", d = blanket-screened shrink re-check's" if sub == "shrink" else "") + "; oracle-tuned comparison\n",
              table(["cell", "M", "level", "FP pooled"] + [LABEL[k] for k in mk], rows)]
    return L


# --------------------------------------------------------------------------------------------- oracle tables

_ORACLE_ROWS = {}


def oracle_nonlazy_unique(arm, N, T, tau, d, g):
    """Stored non-lazy oracle-run unique counts (itpd_naive_nonlazy, itpd_nonlazy) of one graph."""
    k = (arm, N, T, tau, d)
    if k not in _ORACLE_ROWS:
        j = jl(f"{RES}/oracle/{arm}/N{N}_T{T}_tau{tau}_d{d}.json")
        _ORACLE_ROWS[k] = {}
        for r in j["rows"]:
            _ORACLE_ROWS[k].setdefault(r["graph"], {})[r["name"]] = r["unique"]
    return _ORACLE_ROWS[k][g]["itpd_naive_nonlazy"], _ORACLE_ROWS[k][g]["itpd_nonlazy"]


def oracle_tables(out):
    L = []
    files = sorted(glob.glob(f"{out}/oracle/*/*.json"))
    if not files:
        return ["\n(no oracle files)\n"], {}
    agg = {}
    n_graphs = n_bad = 0
    bad = []
    for f in files:
        d = jl(f)
        for r in d["graphs"]:
            n_graphs += 1
            key = (d["arm"], d["N"], d["T"])
            a = agg.setdefault(key, {"g": 0, "ex": {m: 0 for m in ("iamb_known_order", "iamb_known_order_tie_first", "iamb_known_order_tie_random", "itpd_marginal_first", "itpd_marginal_first_nonlazy")},
                                     "ncand": 0, "u": {}, "ms": {}})
            a["g"] += 1
            a["ncand"] += r["n_cand"]
            for m in a["ex"]:
                a["ex"][m] += int(r[m]["exact"])
                if not r[m]["exact"]:
                    n_bad += 1
                    bad.append((f, r["graph"], m))
            for m in ("iamb_known_order", "itpd_marginal_first", "itpd_marginal_first_nonlazy"):
                a["u"][m] = a["u"].get(m, 0) + r[m]["unique"]
                a["ms"][m] = max(a["ms"].get(m, 0), r[m]["max_size"])
            nn, ni = oracle_nonlazy_unique(d["arm"], d["N"], d["T"], d["tau"], d["d"], r["graph"])
            a["u"]["itpd_naive_nonlazy"] = a["u"].get("itpd_naive_nonlazy", 0) + nn
            a["u"]["itpd_nonlazy"] = a["u"].get("itpd_nonlazy", 0) + ni
            for m in ("itpd_naive", "itpd", "full_conditioning"):
                a["u"][m] = a["u"].get(m, 0) + r["stored_oracle_run"][m]["unique"]
                a["ms"][m] = max(a["ms"].get(m, 0), r["stored_oracle_run"][m]["max_size"])
    rows1, rows2 = [], []
    for (arm, N, T), a in sorted(agg.items()):
        ex = " / ".join(f"{a['ex'][m]}" for m in ("iamb_known_order", "iamb_known_order_tie_first", "iamb_known_order_tie_random", "itpd_marginal_first", "itpd_marginal_first_nonlazy"))
        rows1.append([arm, N, T, a["g"], ex] + [f"{a['u'][m] / a['ncand']:.3f}" for m in ("iamb_known_order", "itpd_naive", "itpd", "full_conditioning")] +
                     [f"{a['ms'][m]}" for m in ("iamb_known_order", "itpd_naive", "itpd", "full_conditioning")])
        rows2.append([arm, N, T, a["g"], f"{a['u']['itpd_marginal_first'] / a['u']['itpd_naive']:.4f}", f"{a['u']['itpd_marginal_first'] / a['u']['itpd']:.4f}",
                      f"{a['u']['itpd'] / a['u']['itpd_naive']:.4f}", f"{a['u']['itpd_marginal_first_nonlazy'] / a['u']['itpd_naive_nonlazy']:.4f}",
                      f"{a['u']['itpd_nonlazy'] / a['u']['itpd_naive_nonlazy']:.4f}"])
    L += [f"\nOracle check: {n_graphs} graphs, {n_bad} not exact" + ("" if not bad else f" (BUG: {bad[:5]})") + ".\n",
          "\n#### IAMB per target under the oracle on the oracle-run instances (all tau 1-3, d 1-3 pooled per arm, N, T): graphs exact out of graphs for IAMB with tie rules last / first / random, ITPD+mf lazy / non-lazy; unique tests per candidate (paired by graph with the stored oracle-run rows) and largest set (max over graphs)\n",
          table(["arm", "N", "T", "graphs", "exact: IAMB / IAMB first / IAMB rand / mf / mf non-lazy", "IAMB tests per cand.", "ITPD_naive", "ITPD", "full conditioning",
                 "IAMB largest set", "naive", "ITPD", "full conditioning"], rows1),
          "\n#### ITPD + marginal-first under the oracle: unique tests relative to ITPD_naive and ITPD (pooled over the graphs of the group), lazy, and non-lazy (the evaluation of the original code) in the last two columns\n",
          table(["arm", "N", "T", "graphs", "mf / naive (lazy)", "mf / ITPD (lazy)", "ITPD / naive (lazy, stored)", "mf / naive (both non-lazy)",
                 "ITPD / naive (both non-lazy, stored)"], rows2)]
    # blanket-screened shrink oracle rows (d = 2, N 10/20, T 8/16, tau 1/2) on the same graphs
    rows3 = []
    for f in sorted(glob.glob(f"{out}/oracle/*/*_d2.json")):
        d = jl(f)
        pdir = f"{RES}/blanket_screened_shrink/oracle/{d['arm']}/N{d['N']}_T{d['T']}_tau{d['tau']}_d2"
        if not os.path.isdir(pdir):
            continue
        ui = u_sh = u_rc = nc = 0
        mi = m_sh = m_rc = 0
        for r in d["graphs"]:
            h = jl(f"{pdir}/g{r['graph']:02d}.json")
            assert h["sha1"] == r["sha1"]
            rr = {x["name"]: x for x in h["rows"]}
            ui += r["iamb_known_order"]["unique"]; u_sh += rr["blanket_screened_shrink"]["unique"]; u_rc += rr["blanket_screened_shrink_recheck"]["unique"]; nc += r["n_cand"]
            mi = max(mi, r["iamb_known_order"]["max_size"]); m_sh = max(m_sh, rr["blanket_screened_shrink"]["max_size"]); m_rc = max(m_rc, rr["blanket_screened_shrink_recheck"]["max_size"])
        rows3.append([d["arm"], d["N"], d["T"], d["tau"], len(d["graphs"]), f"{ui / nc:.3f}", f"{u_sh / nc:.3f}", f"{u_rc / nc:.3f}", mi, m_sh, m_rc])
    L += ["\n#### O3. IAMB, blanket-screened shrink and blanket-screened shrink re-check under the oracle on the same 20 graphs per cell (d = 2): unique tests per candidate and largest set (max over graphs)\n",
          table(["arm", "N", "T", "tau", "graphs", "IAMB tests/cand", "blanket-screened shrink", "blanket-screened shrink re-check", "IAMB largest set",
                 "blanket-screened shrink", "blanket-screened shrink re-check"], rows3)]
    return L, {"graphs": n_graphs, "not_exact": n_bad}


def timing_tables(out):
    files = sorted(glob.glob(f"{out}/timing/*/*.json"))
    if not files:
        return ["\n(no timing files)\n"]
    by = {}
    for f in files:
        d = jl(f)
        for r in d["rows"]:
            by.setdefault((d["N"], d["T"], d["M"]), {}).setdefault(r["name"], []).append((r["seconds_wall"], r["seconds_cpu"], r.get("unique")))
        pl = f"{out}/lasso_ebic_fixed/{d['cell']}/g{d['graph']:02d}_M{d['M']}.json"
        if os.path.exists(pl):
            for r in jl(pl)["rows"]:
                by[(d["N"], d["T"], d["M"])].setdefault(r["name"], []).append((r["seconds"], r["cpu_seconds"], None))
    names = ["itpd_naive", "itpd", "itpd_marginal_first", "full_conditioning", "iamb_known_order", "blanket_screened_shrink_recheck", "lasso_cv", "lasso_path_all_lambdas", "lasso_ebic",
             "lasso_fixed"]
    rows = []
    for k in sorted(by):
        rows.append([f"N{k[0]} T{k[1]}", k[2], len(by[k]["iamb_known_order"])] + [(f"{med([x[0] for x in by[k][n]]):.2f} / {med([x[1] for x in by[k][n]]):.2f}" if n in by[k] else "-") for n in names])
    return ["\n#### Wall-clock (median over the timed graphs; wall / CPU seconds per dataset), alpha 0.01, fresh Fisher-z object and no shared memo per method, one process, 4 timed tasks in parallel on one node\n",
            table(["cell", "M", "graphs"] + names, rows)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{RES}/baselines")
    a = ap.parse_args()
    L = ["# Known-order baselines: tables (generated by scripts/known_order_baselines_collect.py)\n"]
    o, agg = oracle_tables(a.out)
    L += ["\n## Oracle\n"] + o
    data = load(a.out)
    n = {k: len(v) for k, v in data.items()}
    L += [f"\n## Finite data (window instances of the finite-data runs, tau 1, d 2, linear-Gaussian, Fisher-z, alpha 0.01 unless stated); tasks present per (N, T, M): {n}\n",
          "\n### Operating point alpha 0.01, common targets (t <= (M - 3) / N)\n"] + summary_tables(data, "all") + summary_tables(data, "shrink")
    L += ["\n### Paired per-graph differences (common targets)\n"] + paired_tables(data)
    L += ["\n### Matched false positives\n"] + matched_tables(data)
    L += ["\n### Wall-clock\n"] + timing_tables(a.out)
    with open(f"{a.out}/TABLES.md", "w") as f:
        f.write("\n".join(L) + "\n")
    json.dump({"oracle": agg, "tasks": {f"{k[0]}x{k[1]}xM{k[2]}": v for k, v in n.items()}}, open(f"{a.out}/aggregates.json", "w"))
    print("wrote", f"{a.out}/TABLES.md")


if __name__ == "__main__":
    main()
