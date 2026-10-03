"""Shared helpers of the finite-data collectors (finite-data, robustness, single-series, nonlinear): load per-task
JSONs, median [Q1, Q3], markdown tables. Results written before the method-name rename carry old row names; `load_tasks`
translates them with methods_registry.OLD_TO_NEW, so every collector reads old and new files with the new names."""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from itpd.methods_registry import OLD_TO_NEW  # noqa: E402


def rename_rows(task):
    """Old row names -> current names in the lists of rows a task JSON can hold (runs, rows, lasso rows)."""
    lists = [task.get("runs"), task.get("rows"), (task.get("lasso") or {}).get("rows")]
    for lst in lists:
        for r in lst or []:
            if isinstance(r, dict) and r.get("name") in OLD_TO_NEW:
                r["name"] = OLD_TO_NEW[r["name"]]
    return task


def load_tasks(pattern):
    out = []
    for p in sorted(glob.glob(pattern)):
        if ".tmp" in p:
            continue
        out.append(rename_rows(json.load(open(p))))
    return out


def run_of(task, name, alpha=0.01):
    for r in task["runs"]:
        if r["name"] == name and abs(r["alpha"] - alpha) < 1e-12:
            return r
    return None


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


def boot_ci(x, B=2000, seed=0, stat=np.mean):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    bs = [stat(x[rng.integers(0, len(x), len(x))]) for _ in range(B)]
    return float(stat(x)), *[float(v) for v in np.percentile(bs, [2.5, 97.5])]


def fmt_ci(t):
    return "-" if t[0] != t[0] else f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"


def size_bins(by_size, bins=((0, 0), (1, 1), (2, 4), (5, 19), (20, 99), (100, 10 ** 9))):
    out = []
    for lo, hi in bins:
        out.append(sum(int(v) for k, v in by_size.items() if lo <= int(k) <= hi))
    return out


BIN_LABELS = ["0", "1", "2-4", "5-19", "20-99", ">=100"]
