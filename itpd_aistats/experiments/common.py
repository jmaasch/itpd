"""Plumbing shared by the experiment drivers: integer lists, atomic JSON files, the time-boxed worker pool, the closing line.

Every driver builds a list of tasks, runs `fn(task) -> (path, status, seconds)` over it with `run_tasks`, and ends with `finish`.
A task is skipped when its file exists (the task function checks), so repeating the same command resumes a run; `--budget-sec`
stops new tasks from starting, so a time-limited job is repeated until the closing line says `remaining 0`.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import time


def ints(s: str) -> list[int]:
    """'5,10,20' -> [5, 10, 20]."""
    return [int(x) for x in s.split(",")]


def dump_json(path: str, obj) -> None:
    """Write obj as JSON through a temporary file and a rename, so that a killed run never leaves a partial result."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, default=lambda o: o.item() if hasattr(o, "item") else (o.tolist() if hasattr(o, "tolist") else str(o)))
    os.replace(tmp, path)


def add_pool_args(ap, workers: int = 1, budget_sec: float = 1e9) -> None:
    ap.add_argument("--workers", type=int, default=workers, help="worker processes (1 = run in this process)")
    ap.add_argument("--budget-sec", type=float, default=budget_sec, help="start no new task after this many seconds")


def add_shard_args(ap) -> None:
    ap.add_argument("--shard", default=None, help="K/NSH: run only tasks K, K+NSH, ... of the cost-sorted list (one shard per batch job)")
    ap.add_argument("--summary", default=None, help="file for the closing line")


def take_shard(tasks: list, shard: str | None) -> list:
    if not shard:
        return tasks
    k, nsh = (int(x) for x in shard.split("/"))
    return tasks[k::nsh]


def guard(job):
    fn, task, deadline = job
    if time.time() > deadline:
        return task, "skipped", 0.0
    try:
        return fn(task)
    except Exception as e:                       # report and continue; the task file is not written
        return task, f"error {type(e).__name__}: {e}", 0.0


def print_progress(r) -> None:
    """One line per finished task: status, file name, seconds."""
    print(r[1], os.path.basename(r[0]) if isinstance(r[0], str) else "", round(r[2], 1), flush=True)


def run_tasks(fn, tasks: list, workers: int, deadline: float, on_result=None) -> int:
    """Apply fn to every task, in this process (workers <= 1) or on a pool of forked processes.

    A task that would start after `deadline` is skipped; a task that raises is printed and does not stop the others.
    on_result(r) sees every result that is not an error. Returns the number of tasks that raised."""
    jobs = [(fn, t, deadline) for t in tasks]
    pool = None
    if workers <= 1:
        it = map(guard, jobs)
    else:
        pool = mp.get_context("fork").Pool(workers)
        it = pool.imap_unordered(guard, jobs, chunksize=1)
    n_err = 0
    for r in it:
        if str(r[1]).startswith("error"):
            n_err += 1
            print(r[1], r[0], flush=True)
        elif on_result is not None:
            on_result(r)
    if pool is not None:
        pool.close()
        pool.join()
    return n_err


def finish(t0: float, left: int, n_err: int | None = None, n_tasks: int | None = None, shard: str | None = None,
           summary: str | None = None) -> None:
    """Print `done in Ns; [errors E; ]remaining L[; tasks T][; shard K/NSH]` (also into the file `summary`)."""
    parts = [f"done in {time.time() - t0:.0f}s"]
    if n_err is not None:
        parts.append(f"errors {n_err}")
    parts.append(f"remaining {left}")
    if n_tasks is not None:
        parts.append(f"tasks {n_tasks}")
    if shard:
        parts.append(f"shard {shard}")
    msg = "; ".join(parts)
    print(msg, flush=True)
    if summary:
        with open(summary, "w") as f:
            f.write(msg + "\n")
