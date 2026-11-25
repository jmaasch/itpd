"""Random time graphs.

In-degree d is the expected number of parents of a node from other slots: every
edge slot (a source series at one of the lags 1, ..., `lag`, except the lag-1
self slot) holds an edge with probability d / (N * lag - 1). The lag-1 self
edge is always present.
"""
import numpy as np

from ..graphs import self_edge_mask


def edge_probability(N: int, lag: int, d: float) -> float:
    """Return the probability of an edge in one slot."""
    slots = N * lag - 1
    return 0.0 if slots <= 0 else min(1.0, d / slots)


def sample_stationary_graph(N: int, lag: int, d: float, rng) -> np.ndarray:
    """Return a window graph B (lag + 1, N, N), the same at every step.

    B[l, i, j] means that series i at step t - l is a parent of series j at
    step t. B[0] is empty. `itpd.graphs.unroll` gives the time graph."""
    B = np.zeros((lag + 1, N, N), dtype=bool)
    p = edge_probability(N, lag, d)
    B[1:] = rng.random((lag, N, N)) < p
    B[1][np.arange(N), np.arange(N)] = True
    return B


def sample_nonstationary_graph(N: int, T: int, d: float, lag: int, rng) -> np.ndarray:
    """Return a time graph (T * N, T * N) with edges drawn at each step."""
    A = np.zeros((T * N, T * N), dtype=bool)
    p = edge_probability(N, lag, d)
    for t in range(T - 1):
        for l in range(1, min(lag, T - 1 - t) + 1):
            # The self slot takes a random number too; its edge is set below.
            A[t * N:(t + 1) * N, (t + l) * N:(t + l + 1) * N] = rng.random((N, N)) < p
    A |= self_edge_mask(N, T)
    return A


def longest_cross_lag(A, N: int) -> int:
    """Return the largest lag of an edge between two different series (0 if none)."""
    sources, targets = np.nonzero(np.asarray(A) != 0)
    cross = (sources % N) != (targets % N)
    lags = targets // N - sources // N
    return int(lags[cross].max()) if cross.any() else 0
