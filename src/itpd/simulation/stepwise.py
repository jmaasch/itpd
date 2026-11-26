"""The nonstationary linear-Gaussian model of lag 1, step by step.

The model is that of `linear.make_instance(..., lag=1, stationary=False)`. All
parents of a node are in the step before it, so only the N x N covariance of
one step is kept, and settings with many series stay cheap. The weights of each
node are checked on their own (up to `MAX_TRIES` draws per node); the random
numbers differ from those of `linear.make_instance`. Look at `make_instance`.
"""
import numpy as np

from ..graphs import graph_hash
from . import Instance
from .graphs import edge_probability
from .linear import MAX_TRIES, MIN_STRENGTH, first_rows


def make_instance(N: int, T: int, d: float, g: int, seed: int = 0) -> Instance:
    """Return graph number g with weights and data.

    `info["weights"]` has shape (T - 1, N, N): entry [t - 1, i, j] is the
    weight of the edge from series i at step t - 1 to series j at step t.
    `info["guard_failures"]` counts the nodes that kept their last draw
    although the check failed."""
    stream = [seed, N, T, 1, int(round(10 * d)), g]
    rng = np.random.default_rng(np.random.SeedSequence(stream))
    p = edge_probability(N, 1, d)
    W3 = np.zeros((T - 1, N, N))
    Sig = np.eye(N)   # covariance of step 0
    failures = 0
    min_partial = min_marginal = np.inf
    for t in range(1, T):
        edges = rng.random((N, N)) < p
        edges[np.arange(N), np.arange(N)] = True
        Wt = np.zeros((N, N))
        for j in range(N):
            pa = np.nonzero(edges[:, j])[0]
            S = Sig[np.ix_(pa, pa)]
            for _ in range(MAX_TRIES):
                w = rng.uniform(0.3, 0.8, size=len(pa)) * rng.choice([-1.0, 1.0], size=len(pa))
                sig = float(w @ S @ w)
                if sig > 1.0:   # parent signal variance at most 1
                    w = w * np.sqrt(1.0 / sig)
                    sig = 1.0
                cov = S @ w   # covariance of each parent with the child
                var = sig + 1.0
                marginal = np.abs(cov) / (np.sqrt(np.diag(S)) * np.sqrt(var))
                J = np.empty((len(pa) + 1, len(pa) + 1))   # covariance of the parents and the child
                J[:-1, :-1] = S
                J[:-1, -1] = cov
                J[-1, :-1] = cov
                J[-1, -1] = var
                P = np.linalg.inv(J)
                partial = np.abs(-P[:-1, -1] / np.sqrt(np.abs(np.diag(P)[:-1] * P[-1, -1])))
                if min(marginal.min(), partial.min()) >= MIN_STRENGTH:
                    break
            else:
                failures += 1
            min_partial = min(min_partial, float(partial.min()))
            min_marginal = min(min_marginal, float(marginal.min()))
            Wt[pa, j] = w
        W3[t - 1] = Wt
        Sig = Wt.T @ Sig @ Wt + np.eye(N)
    A = np.zeros((T * N, T * N), dtype=bool)
    for t in range(1, T):
        A[(t - 1) * N:t * N, t * N:(t + 1) * N] = W3[t - 1] != 0

    def draw(rows):
        noise = np.random.default_rng(np.random.SeedSequence(stream + [90210]))
        X = np.empty((rows, T, N))
        X[:, 0] = noise.standard_normal((rows, N))
        for t in range(1, T):
            X[:, t] = X[:, t - 1] @ W3[t - 1] + noise.standard_normal((rows, N))
        return X
    info = {"weights": W3, "guard_failures": failures, "guard_ok": failures == 0,
            "min_partial_correlation": float(min_partial), "min_marginal_correlation": float(min_marginal)}
    return Instance(A, N, T, 1, first_rows(draw, N, T), graph_hash(A), info)
