"""Time graphs as adjacency matrices.

A time graph of N series and T steps has T * N nodes. The node of series n
at step t is t * N + n (both zero-based). A[u, v] != 0 means u -> v.
"""
import hashlib

import numpy as np


def node(t: int, n: int, N: int) -> int:
    """Return the node of series n at step t."""
    return t * N + n


def unroll(B: np.ndarray, T: int) -> np.ndarray:
    """Return the time graph (T * N, T * N) of a stationary window graph.

    B has shape (lag + 1, N, N). B[l, i, j] != 0 means that series i at step
    t - l is a parent of series j at step t, for l = 1, ..., lag. B[0] is
    not used."""
    lag, N = B.shape[0] - 1, B.shape[1]
    A = np.zeros((T * N, T * N), dtype=bool)
    for t in range(1, T):
        for l in range(1, min(lag, t) + 1):
            A[(t - l) * N:(t - l + 1) * N, t * N:(t + 1) * N] = B[l] != 0
    return A


def self_edge_mask(N: int, T: int) -> np.ndarray:
    """Return the Boolean matrix of the lag-1 self edges."""
    mask = np.zeros((T * N, T * N), dtype=bool)
    sources = np.arange((T - 1) * N)
    mask[sources, sources + N] = True
    return mask


def check_temporal(A, N: int, T: int) -> None:
    """Raise ValueError unless A is a time graph that ITPD assumes.

    Every edge points to a later step, and every lag-1 self edge is present."""
    A = np.asarray(A) != 0
    if A.shape != (T * N, T * N):
        raise ValueError(f"The matrix must have shape ({T * N}, {T * N})")
    sources, targets = np.nonzero(A)
    if (sources // N >= targets // N).any():
        raise ValueError("Edges must point strictly forward in time")
    if not A[self_edge_mask(N, T)].all():
        raise ValueError("The graph must contain every lag-1 self edge")


def graph_hash(A) -> str:
    """Return the SHA-1 of the shape and of the Boolean entries of A."""
    A = np.ascontiguousarray(np.asarray(A) != 0, dtype=np.uint8)
    return hashlib.sha1(str(A.shape).encode() + A.tobytes()).hexdigest()
