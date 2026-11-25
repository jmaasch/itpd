"""Nonstationary linear-Gaussian instances that break one assumption of ITPD.

The instance starts from linear.make_instance(N, T, 1, d, g,
stationary=False) and repeats its random draws in the same order. With both
shares at 0 it is that instance, unless that instance fails its weight
check: this generator draws the weights up to MAX_TRIES times.

No lag-1 self edge: round(share * N) series, the first ones of one random
order of the series, lose their self edge at every step. The other edges and
the noise stay.

Edges inside a time step: at every step t >= 1, round(share * n) of the n
edges between two series into step t are moved from lag 1 to lag 0, so that
i at step t - 1 -> j at step t becomes i at step t -> j at step t. For
every step one order of the series is fixed, and only an edge that points
forward in this order can be moved, so the graph stays acyclic. A moved edge
has the weight of its own slot.
"""
import zlib

import numpy as np

from ..graphs import graph_hash
from . import Instance
from .graphs import sample_nonstationary_graph
from .linear import (MIN_STRENGTH, check_record, draw_weights, first_rows, marginal_margin, partial_margin,
                     propagate_covariance)

MAX_TRIES = 500   # draws of the weights; linear.make_instance stops after 25


def series_order(edges: np.ndarray, key: np.ndarray) -> list:
    """Return an order of the nodes of a directed graph with few edges that point backward.

    Heuristic of Eades, Lin and Smyth: sinks go last, sources go first;
    when there is neither, the node with the largest out-degree minus
    in-degree goes first. `key` decides between equal nodes."""
    alive = np.ones(edges.shape[0], bool)
    left, right = [], []
    while alive.any():
        changed = True
        while changed:
            changed = False
            for v in sorted(np.nonzero(alive)[0], key=lambda a: key[a]):
                if not alive[v]:
                    continue
                if int(edges[v][alive].sum()) == 0:
                    right.append(int(v))
                    alive[v] = False
                    changed = True
                elif int(edges[:, v][alive].sum()) == 0:
                    left.append(int(v))
                    alive[v] = False
                    changed = True
        if alive.any():
            best, chosen = None, None
            for v in sorted(np.nonzero(alive)[0], key=lambda a: key[a]):
                delta = int(edges[v][alive].sum()) - int(edges[:, v][alive].sum())
                if best is None or delta > best:
                    best, chosen = delta, int(v)
            left.append(chosen)
            alive[chosen] = False
    return left + right[::-1]


def _propagate(A, W, order):
    """linear.propagate_covariance with the nodes taken in `order` (None: index order)."""
    if order is None:
        return propagate_covariance(A, W, max_signal=1.0)
    ix = np.ix_(order, order)
    Wp, Sp = propagate_covariance(A[ix], W[ix], max_signal=1.0)
    W, Sig = np.zeros_like(Wp), np.zeros_like(Sp)
    W[ix], Sig[ix] = Wp, Sp
    return W, Sig


def make_instance(d: float, g: int, *, no_self_edge_share: float = 0.0, same_step_share: float = 0.0, seed: int = 0,
                  N: int = 20, T: int = 16) -> Instance:
    """Return graph number g with one violation (at most one share may be above 0).

    info["A_main"] is the graph without the violation and info["A_lag1"]
    the edges of A between two different steps. Instance.A holds every
    edge, also those inside a step."""
    if no_self_edge_share > 0 and same_step_share > 0:
        raise ValueError("Only one of the two shares may be above 0")
    stream = [seed, N, T, 1, int(d * 10), g]
    rng = np.random.default_rng(np.random.SeedSequence(stream))
    A_main = sample_nonstationary_graph(N, T, d, 1, rng)
    A = A_main.copy()
    info = {"A_main": A_main}
    order = None   # order in which the nodes are generated (None: index order)
    if no_self_edge_share > 0:
        choice = np.random.default_rng(np.random.SeedSequence(stream + [31, zlib.crc32(b"self_missing") & 0x7FFFFFFF]))
        k = int(round(no_self_edge_share * N))
        series = sorted(int(s) for s in choice.permutation(N)[:k])
        for n in series:
            for t in range(T - 1):
                A[t * N + n, (t + 1) * N + n] = False
        info["series_without_self_edge"] = series
    if same_step_share > 0:
        choice = np.random.default_rng(np.random.SeedSequence(stream + [31, zlib.crc32(b"contemp") & 0x7FFFFFFF]))
        order, n_moved = list(range(N)), 0
        for t in range(1, T):
            cross = A_main[(t - 1) * N:t * N, t * N:(t + 1) * N].copy()
            np.fill_diagonal(cross, False)
            step_order = series_order(cross, choice.random(N))
            position = np.empty(N, int)
            position[step_order] = np.arange(N)
            ii, jj = np.nonzero(cross)
            k = int(round(same_step_share * len(ii)))
            forward = np.nonzero(position[ii] < position[jj])[0]
            moved = choice.permutation(forward)[:k] if len(forward) else np.array([], int)
            for e in moved:
                i, j = int(ii[e]), int(jj[e])
                A[(t - 1) * N + i, t * N + j] = False
                A[t * N + i, t * N + j] = True
            order += [t * N + n for n in step_order]
            n_moved += len(moved)
        order = np.array(order, dtype=int)
        info["n_moved_edges"] = n_moved
    for tries in range(1, MAX_TRIES + 1):
        raw = draw_weights(A.shape, rng)
        W, Sig = _propagate(A, raw * A, order)
        partial, marginal = partial_margin(A, Sig), marginal_margin(A, Sig)
        if min(partial, marginal) >= MIN_STRENGTH:
            break
    step = np.arange(N * T) // N
    info.update(weights=W, A_lag1=A & (step[:, None] < step[None, :]), **check_record(tries, partial, marginal))
    data_seed = [seed, N, T, 1, int(round(d * 10)), g, 90210]

    def draw(rows):
        noise = np.random.default_rng(np.random.SeedSequence(data_seed))
        X = np.empty((rows, N * T))
        for v in range(N * T):   # noise in index order, as linear.sample_linear
            X[:, v] = noise.standard_normal(rows)
        for v in (range(N * T) if order is None else order):
            pa = np.nonzero(A[:, v])[0]
            if len(pa):
                X[:, v] = X[:, v] + X[:, pa] @ W[pa, v]
        return X
    return Instance(A, N, T, 1, first_rows(draw, N, T), graph_hash(A), info)
