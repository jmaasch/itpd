"""Run large oracle-cell tasks, one process per (arm, N, T, graph, method), 16 at a time (resumable, time-boxed).

    python scripts/large_oracle_run.py OUT_DIR --cells "80x25,40x50" --graphs 0-4 --arms window [--methods itpd_naive,itpd,full_conditioning]
        [--workers 16] [--start-by-sec 480] [--task-timeout 780] [--oracle fast]
A cell is NxT (tau 1, d 2, full history, lazy headline). Task output: OUT_DIR/<arm>/N<N>_T<T>_g<g>_<method>.json (one graph per file
via `itpd.run_oracle_counts --graphs 1 --offset g`; instance: OUT_DIR/instances/<arm>/...). Existing files are skipped; no task is started after
--start-by-sec (so a short job can be repeated until "remaining 0"). Biggest cells first. Prints one summary line at the end.
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from itpd.methods_registry import OLD_TO_NEW  # noqa: E402


def rng(s):
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--cells", required=True)
    ap.add_argument("--graphs", default="0-4")
    ap.add_argument("--arms", default="window")
    ap.add_argument("--methods", default="itpd_naive,itpd,full_conditioning")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--start-by-sec", type=float, default=480)
    ap.add_argument("--task-timeout", type=float, default=780)
    ap.add_argument("--oracle", default="fast")
    a = ap.parse_args()
    t0 = time.time()
    tasks = []
    for arm in a.arms.split(","):
        for c in a.cells.split(","):
            N, T = (int(x) for x in c.split("x"))
            for g in rng(a.graphs):
                for m in a.methods.split(","):
                    m = OLD_TO_NEW.get(m, m)
                    path = os.path.join(a.out, arm, f"N{N}_T{T}_g{g}_{m}.json")
                    tasks.append((N * T, arm, N, T, g, m, path))
    # a task is done when its file exists under the current or an earlier name of the method (files of runs started before the rename)
    done = lambda t: any(os.path.exists(t[-1].replace(f"_{t[5]}.json", f"_{old}.json")) for old in [t[5]] + [k for k, v in OLD_TO_NEW.items() if v == t[5]])
    tasks = [t for t in tasks if not done(t)]
    tasks.sort(key=lambda t: (-t[0], t[4]))
    print(f"tasks todo {len(tasks)}", flush=True)
    env = dict(os.environ, ITPD_ORACLE=a.oracle, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONPATH=ROOT)

    def run(t):
        _, arm, N, T, g, m, path = t
        if time.time() - t0 > a.start_by_sec:
            return "skipped"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cmd = [sys.executable, "-m", "itpd.run_oracle_counts", "--N", str(N), "--T", str(T), "--tau", "1", "--d", "2", "--graphs", "1", "--offset", str(g),
               "--graph", arm, "--methods", m, "--instances-dir", os.path.join(a.out, "instances", arm), "--out", path]
        try:
            r = subprocess.run(cmd, env=env, cwd=ROOT, timeout=a.task_timeout, capture_output=True, text=True)
            if r.returncode:
                print("FAIL", path, r.stderr[-300:], flush=True)
                return "fail"
        except subprocess.TimeoutExpired:
            print("TIMEOUT", path, flush=True)
            return "timeout"
        return "done"

    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(run, tasks))
    left = sum(1 for t in tasks if not done(t))
    print(f"done {res.count('done')} skipped {res.count('skipped')} fail {res.count('fail')} timeout {res.count('timeout')} in {time.time() - t0:.0f}s; remaining {left}", flush=True)


if __name__ == "__main__":
    main()
