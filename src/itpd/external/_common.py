"""Shared code of the external-baseline scripts.

The scripts run in the interpreter of their tool, so this module needs only
the standard library and numpy. Data have shape (M, T, N). The node of
series n at step t is t * N + n, and A[u, v] means u -> v.
"""
import argparse
import hashlib
import json
import sys
import time
from importlib import metadata

import numpy as np


def parser(description):
    """Return the argument parser with the two options of every script."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--input", required=True, help=".npz file with the data X of shape (M, T, N)")
    p.add_argument("--output", required=True, help=".npz file for the result")
    return p


def read_input(path):
    """Return the data X (M, T, N) of an input file."""
    with np.load(path, allow_pickle=False) as z:
        return np.ascontiguousarray(z["X"], dtype=np.float64)


def write_output(path, A_hat, seconds, packages, status="ok", counter=None):
    """Write the result file.

    A_hat is the estimated time graph (T * N, T * N), status is "ok" or
    "timeout", and seconds is the time of the method call. versions is a
    JSON text with the version of Python and of each named package. A
    counter adds requested, unique, largest_set and mean_set."""
    result = {"A_hat": np.asarray(A_hat) != 0, "status": status, "seconds": seconds}
    versions = {"python": sys.version.split()[0]}
    versions.update({name: metadata.version(name) for name in packages})
    result["versions"] = json.dumps(versions)
    if counter is not None:
        result.update(counter.summary())
    np.savez(path, **result)


def window_to_time_graph(win, T):
    """Return the time graph that repeats a window graph at every step.

    win has shape (L + 1, N, N). win[l, i, j] means that series i at step
    t - l is a parent of series j at step t, for l = 1, ..., L."""
    N = win.shape[1]
    A = np.zeros((T * N, T * N), dtype=bool)
    for t in range(1, T):
        for l in range(1, min(win.shape[0] - 1, t) + 1):
            A[(t - l) * N:(t - l + 1) * N, t * N:(t + 1) * N] |= win[l] != 0
    return A


class TestCounter:
    """Count the CI tests that a method requests.

    A unique test is an unordered pair of variables with a conditioning
    set. record raises TimeoutError when time_limit seconds have passed
    since the counter was made."""

    def __init__(self, time_limit=None):
        self.requested = 0
        self.seen = set()
        self.unique_by_size = {}
        self.start = time.perf_counter()
        self.time_limit = time_limit

    def record(self, x, y, S):
        """Count one test of x and y given S."""
        if self.time_limit is not None and time.perf_counter() - self.start > self.time_limit:
            raise TimeoutError
        x, y = int(x), int(y)
        S = tuple(sorted({int(s) for s in S} - {x, y}))
        k = len(S)
        self.requested += 1
        # The set holds hashes, not tuples, so that 1e8 tests fit in memory.
        if k <= 8:
            key = hash((min(x, y), max(x, y), S))
        else:
            digest = hashlib.blake2b(np.asarray(S, dtype=np.int32).tobytes(), digest_size=12).digest()
            key = hash((min(x, y), max(x, y), k, digest))
        if key not in self.seen:
            self.seen.add(key)
            self.unique_by_size[k] = self.unique_by_size.get(k, 0) + 1

    def summary(self):
        """Return requested, unique, largest_set and mean_set (over the unique tests)."""
        unique = len(self.seen)
        total = sum(k * count for k, count in self.unique_by_size.items())
        return {"requested": self.requested, "unique": unique,
                "largest_set": max(self.unique_by_size, default=0),
                "mean_set": total / unique if unique else float("nan")}
