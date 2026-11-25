"""Linear-Gaussian data on a random time graph.

Model: X_v = sum over the parents u of W[u, v] * X_u + noise_v, with standard
normal noise; the nodes of step 0 have no parents. Weights have a size in [0.3,
0.8] and a random sign. A draw of the weights is accepted when every edge has a
partial correlation (given the other parents of its child) and a correlation of
at least `MIN_STRENGTH` in absolute value. Look at `make_instance`.
"""
import hashlib

import numpy as np

from ..graphs import graph_hash, unroll
from . import Instance
from .graphs import sample_nonstationary_graph, sample_stationary_graph

MIN_STRENGTH = 0.02
MAX_TRIES = 25
ROWS = 2000   # rows of one draw of the data


def data_hash(X) -> str:
    """Return the SHA-1 of the float64 bytes of X."""
    return hashlib.sha1(np.ascontiguousarray(X, dtype=np.float64).tobytes()).hexdigest()


def draw_weights(shape, rng, low: float = 0.3, high: float = 0.8) -> np.ndarray:
    """Return weights with a size in [`low`, `high`] and a random sign."""
    return rng.uniform(low, high, size=shape) * rng.choice([-1.0, 1.0], size=shape)


def spectral_radius(Wl: np.ndarray) -> float:
    """Return the spectral radius of the process with lag weights `Wl`.

    `Wl` has shape (lag + 1, N, N)."""
    lag, N = Wl.shape[0] - 1, Wl.shape[1]
    C = np.zeros((N * lag, N * lag))
    for l in range(1, lag + 1):
        C[:N, (l - 1) * N:l * N] = Wl[l].T
    if lag > 1:
        C[N:, :-N] = np.eye(N * (lag - 1))
    return float(np.max(np.abs(np.linalg.eigvals(C))))


def stationary_weights(B: np.ndarray, rng, radius: float = 0.9) -> np.ndarray:
    """Return lag weights (lag + 1, N, N) of the window graph B.

    If the spectral radius r is above `radius`, the weights of lag l are
    multiplied by (radius / r) ** l."""
    Wl = B * draw_weights(B.shape, rng)   # The lag-0 slice is drawn and not used.
    Wl[0] = 0
    r = spectral_radius(Wl) if B.shape[0] > 1 else 0.0
    if r > radius:
        c = radius / r
        for l in range(1, Wl.shape[0]):
            Wl[l] *= c ** l
    return Wl


def unroll_weights(Wl: np.ndarray, T: int) -> np.ndarray:
    """Return the (T * N, T * N) matrix repeating `Wl` at every step."""
    lag, N = Wl.shape[0] - 1, Wl.shape[1]
    W = np.zeros((T * N, T * N))
    for t in range(1, T):
        for l in range(1, min(lag, t) + 1):
            W[(t - l) * N:(t - l + 1) * N, t * N:(t + 1) * N] = Wl[l]
    return W


def propagate_covariance(A: np.ndarray, W: np.ndarray, max_signal=None):
    """Return (W, Sigma): the weights and the exact covariance.

    Nodes are taken in index order, so every parent must have a smaller index.
    If `max_signal` is given, the incoming weights of a node are scaled down so
    that the variance of its parent signal is at most `max_signal` (the noise
    variance is 1)."""
    n = A.shape[0]
    W = W.copy()
    Sig = np.zeros((n, n))
    for v in range(n):
        pa = np.nonzero(A[:, v])[0]
        if len(pa) == 0:
            Sig[v, v] = 1.0
            continue
        w = W[pa, v]
        sig = float(w @ Sig[np.ix_(pa, pa)] @ w)
        if max_signal is not None and sig > max_signal:
            w = w * np.sqrt(max_signal / sig)
            W[pa, v] = w
            sig = max_signal
        row = w @ Sig[pa, :v]
        Sig[v, :v] = row
        Sig[:v, v] = row
        Sig[v, v] = sig + 1.0
    return W, Sig


def partial_margin(A: np.ndarray, Sig: np.ndarray) -> float:
    """Return the smallest |partial correlation| of an edge.

    Each edge is conditioned on the other parents of its child."""
    best = np.inf
    for v in range(A.shape[0]):
        pa = np.nonzero(A[:, v])[0]
        if len(pa) == 0:
            continue
        idx = list(pa) + [v]
        P = np.linalg.inv(Sig[np.ix_(idx, idx)])
        k = len(pa)
        pc = np.abs(-P[:k, k] / np.sqrt(np.abs(P[:k, :k].diagonal() * P[k, k])))
        best = min(best, float(pc.min()))
    return best


def marginal_margin(A: np.ndarray, Sig: np.ndarray) -> float:
    """Return the smallest |correlation| of an edge."""
    ii, jj = np.nonzero(A)
    if len(ii) == 0:
        return np.inf
    sd = np.sqrt(np.diag(Sig))
    return float(np.min(np.abs(Sig[ii, jj]) / (sd[ii] * sd[jj])))


def check_record(tries: int, partial: float, marginal: float) -> dict:
    """Return the numbers of the weight check for `Instance.info`."""
    return {"guard_tries": tries, "guard_ok": bool(min(partial, marginal) >= MIN_STRENGTH),
            "min_partial_correlation": float(partial), "min_marginal_correlation": float(marginal)}


def sample_linear(A: np.ndarray, W: np.ndarray, M: int, rng) -> np.ndarray:
    """Return M observations (M, T * N) drawn in node order.

    Draws one noise vector per node."""
    n = A.shape[0]
    X = np.empty((M, n))
    for v in range(n):
        x = rng.standard_normal(M)
        pa = np.nonzero(A[:, v])[0]
        if len(pa):
            x = x + X[:, pa] @ W[pa, v]
        X[:, v] = x
    return X


def first_rows(draw, N: int, T: int):
    """Return `sample(M)`: the first M rows of `draw(max(ROWS, M))`.

    The returned array has shape (M, T, N). The noise is drawn per node at full
    length, so the draw for another number of rows is another data set, not an
    extension."""
    def sample(M: int) -> np.ndarray:
        return np.ascontiguousarray(draw(max(ROWS, M))[:M].reshape(M, T, N))
    return sample


def make_instance(N: int, T: int, lag: int, d: float, g: int, *, stationary: bool, seed: int = 0,
                  weights: bool = True) -> Instance:
    """Return graph number g of a setting, with weights and data.

    - `stationary=True`: one window graph and one set of lag weights for all
      steps, scaled to a spectral radius of at most 0.9.
    - `stationary=False`: edges and weights are drawn at every step, and the
      parent signal of a node has variance at most 1.

    The graph and the weights share one random generator (graph first);
    `weights=False` stops after the graph. After `MAX_TRIES` draws that fail
    the check, the last draw is kept and `info["guard_ok"]` is `False`."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, N, T, lag, int(d * 10), g]))
    info = {}
    if stationary:
        B = info["B"] = sample_stationary_graph(N, lag, d, rng)
        A = unroll(B, T)
    else:
        A = sample_nonstationary_graph(N, T, d, lag, rng)
    if not weights:
        return Instance(A, N, T, lag, None, graph_hash(A), info)
    for tries in range(1, MAX_TRIES + 1):
        if stationary:
            W, Sig = propagate_covariance(A, unroll_weights(stationary_weights(B, rng), T) * A)
        else:
            W, Sig = propagate_covariance(A, draw_weights(A.shape, rng) * A, max_signal=1.0)
        partial, marginal = partial_margin(A, Sig), marginal_margin(A, Sig)
        if min(partial, marginal) >= MIN_STRENGTH:
            break
    info.update(weights=W, **check_record(tries, partial, marginal))
    data_seed = [seed, N, T, lag, int(round(d * 10)), g, 90210]   # round(d * 10) here; the seed of the graph and the weights uses int(d * 10)

    def draw(rows):
        return sample_linear(A, W, rows, np.random.default_rng(np.random.SeedSequence(data_seed)))
    return Instance(A, N, T, lag, first_rows(draw, N, T), graph_hash(A), info)
