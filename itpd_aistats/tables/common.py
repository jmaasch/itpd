"""Helpers shared by the table collectors: per-task JSON loading, number formats, markdown tables, bootstrap intervals, log-log slopes
and the recall-at-matched-false-positives interpolation. Results written before the method-name rename carry old row names;
`new_name`, `rename_rows` and `load_tasks` translate them with methods_registry.OLD_TO_NEW, so every collector reads old and new
files with the new names; the same functions translate the `shrink` options and the `alpha_A` key of a row (OLD_OPTION_KEYS)."""
import glob
import json
import os

import numpy as np

from itpd.methods_registry import OLD_OPTION_KEYS, OLD_TO_NEW, current_shrink_options

RESULTS = os.environ.get("ITPD_RESULTS", "results")        # results directory of the earlier runs
ALPHA = 0.01                                               # the operating point of every table
CORE = ["itpd_naive", "itpd", "full_conditioning"]
LAZY = ["itpd_naive", "itpd", "itpd_repo_variant", "full_conditioning"]
SIZE_BINS = ((0, 0), (1, 1), (2, 4), (5, 19), (20, 99), (100, 10 ** 9))
BIN_LABELS = ["0", "1", "2-4", "5-19", "20-99", ">=100"]


def new_name(name):
    return OLD_TO_NEW.get(name, name)


def rename_rows(task):
    """Old row names, `shrink` options and level keys -> current ones in the lists of rows a task JSON can hold (runs, rows, lasso rows)."""
    lists = [task.get("runs"), task.get("rows"), (task.get("lasso") or {}).get("rows")]
    for lst in lists:
        for r in lst or []:
            if not isinstance(r, dict):
                continue
            if r.get("name") in OLD_TO_NEW:
                r["name"] = OLD_TO_NEW[r["name"]]
            if isinstance(r.get("shrink"), dict):
                r["shrink"] = current_shrink_options(r["shrink"])
            for old, new_ in OLD_OPTION_KEYS.items():
                if old in r:
                    r[new_] = r.pop(old)
    return task


def load_tasks(pattern):
    out = []
    for p in sorted(glob.glob(pattern)):
        if ".tmp" in p:
            continue
        out.append(rename_rows(json.load(open(p))))
    return out


def run_of(task, name, alpha=ALPHA):
    for r in task["runs"]:
        if r["name"] == name and abs(r["alpha"] - alpha) < 1e-12:
            return r
    return None


def _num(v):
    if v is None or (isinstance(v, float) and v != v):
        return "-"
    return f"{v:.0f}" if abs(v) >= 100 else (f"{v:.2f}" if abs(v) < 10 else f"{v:.1f}")


def q(x):
    """Median [Q1, Q3] of the non-missing values."""
    x = [v for v in x if v is not None]
    if not x:
        return "-"
    a, b, c = np.percentile(np.asarray(x, float), [50, 25, 75])
    return f"{_num(a)} [{_num(b)}, {_num(c)}]"


def md(head, rows):
    return "\n".join(["| " + " | ".join(map(str, head)) + " |", "|" + "---|" * len(head)] +
                     ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def boot_ci(x, B=2000, seed=0, stat=np.mean):
    """(statistic, 2.5%, 97.5%) of a percentile bootstrap over x; NaNs for fewer than two values."""
    x = np.asarray(x, float)
    if len(x) < 2:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    bs = [stat(x[rng.integers(0, len(x), len(x))]) for _ in range(B)]
    return float(stat(x)), *[float(v) for v in np.percentile(bs, [2.5, 97.5])]


def fmt_ci(t):
    return "-" if t[0] != t[0] else f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"


def size_bins(by_size, bins=SIZE_BINS):
    """Counts of a {conditioning-set size: count} dict summed over the bins."""
    out = []
    for lo, hi in bins:
        out.append(sum(int(v) for k, v in by_size.items() if lo <= int(k) <= hi))
    return out


def slope(xs, ys):
    """Least-squares slope of log(ys) on log(xs)."""
    return float(np.polyfit(np.log(np.asarray(xs, float)), np.log(np.asarray(ys, float)), 1)[0])


def ncand(N, T):
    """Sum |C| over the N (T - 1) targets with t >= 1: a target at time t has N t - 1 candidates (full history)."""
    return N * sum(N * t - 1 for t in range(1, T))


def interp(curve, fp0):
    """(recall, status, bracket alphas) at the false-positive level fp0 on a pooled curve {alpha: [fp, tp, fn, ...]}: recall linear in
    log FP (FP clamped at 0.5) between the two neighbouring alphas, sorted by FP; status 'ok' | 'below' | 'above' (level outside the swept range)."""
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
