"""Linear-Gaussian data on a lag-1 graph in which a share of the edge slots is redrawn at every step.

The first transition (step 0 to step 1) has one lag-1 graph and one weight
matrix, drawn as for a stationary graph. At every later transition each of
the N x N slots is redrawn with probability `share`: a redrawn slot holds an
edge with the edge probability (the self edge is always present) and gets
another weight, and any other slot keeps the edge and the weight of the
transition before it. share = 0 is a stationary graph; with share = 1 every
transition is drawn on its own. The weight matrix of every transition is
scaled to a spectral radius of at most 0.9.

The random streams do not depend on the share or on d, so the sets of
redrawn slots are nested in the share, and the noise is shared.
"""
import numpy as np

from ..graphs import graph_hash
from . import Instance
from .graphs import edge_probability, sample_stationary_graph
from .linear import (MAX_TRIES, MIN_STRENGTH, check_record, draw_weights, first_rows, marginal_margin, partial_margin,
                     propagate_covariance, sample_linear, spectral_radius)


def _rng(*parts):
    return np.random.default_rng(np.random.SeedSequence([int(p) for p in parts]))


def make_instance(N: int, T: int, d: float, share: float, g: int, seed: int = 0, M_max: int = 2000) -> Instance:
    """Return graph number g with weights and data.

    The weights are drawn again (up to MAX_TRIES times) until the graph of
    share 0 passes the check of linear.py; that draw is used for every
    share, and info["guard_ok"] says whether the check holds at this share."""
    p = edge_probability(N, 1, d)
    first = sample_stationary_graph(N, 1, d, _rng(seed, g, 2, 101))[1]
    coins = _rng(seed, g, 3, 201).random((T - 1, N, N))
    slot_edges = _rng(seed, g, 3, 202).random((T - 1, N, N)) < p
    diagonal = np.eye(N, dtype=bool)

    def transitions(share):
        """Return the edges and the redrawn slots of every transition."""
        edges, redrawn = [first.copy()], [np.zeros((N, N), dtype=bool)]
        for u in range(1, T - 1):
            r_u = coins[u] < share
            e_u = np.where(r_u, slot_edges[u], edges[-1])
            e_u[diagonal] = True
            edges.append(e_u)
            redrawn.append(r_u)
        A = np.zeros((T * N, T * N), dtype=bool)
        for u in range(T - 1):
            A[u * N:(u + 1) * N, (u + 1) * N:(u + 2) * N] = edges[u]
        return A, edges, redrawn

    def weights(A, edges, redrawn, tries):
        """Return (W, Sigma) for draw number `tries` of the weights."""
        w_u = draw_weights((2, N, N), _rng(seed, g, 2, 103, tries))[1]
        slot_weights = draw_weights((T - 1, N, N), _rng(seed, g, 3, 203, tries))
        W = np.zeros((T * N, T * N))
        for u in range(T - 1):
            if u > 0:
                w_u = np.where(redrawn[u], slot_weights[u], w_u)
            Wl = np.zeros((2, N, N))
            Wl[1] = edges[u] * w_u
            r = spectral_radius(Wl)
            if r > 0.9:
                Wl[1] *= 0.9 / r
            W[u * N:(u + 1) * N, (u + 1) * N:(u + 2) * N] = Wl[1]
        return propagate_covariance(A, W * A)

    still = transitions(0.0)
    for tries in range(1, MAX_TRIES + 1):
        W, Sig = weights(*still, tries)
        if min(partial_margin(still[0], Sig), marginal_margin(still[0], Sig)) >= MIN_STRENGTH:
            break
    A, edges, redrawn = transitions(share)
    if share > 0:
        W, Sig = weights(A, edges, redrawn, tries)
    info = {"weights": W, "n_redrawn_slots": int(sum(r.sum() for r in redrawn)),
            **check_record(tries, partial_margin(A, Sig), marginal_margin(A, Sig))}

    def draw(rows):
        return sample_linear(A, W, rows, _rng(seed, g, 2, 105, 0))
    return Instance(A, N, T, 1, first_rows(draw, M_max, N, T), graph_hash(A), info)
