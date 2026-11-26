"""Nonlinear data with additive normal noise on a random time graph.

Model: X_v = sum over the parents u of f_uv(X_u) + noise_v, with standard
normal noise and f(x) = w * (a * tanh(s * x + c) + (1 - a) * sin(k * x + phi)).
Ranges: |w| in [0.8, 1.6] with a random sign, a in [0.4, 1], s and k in [0.7,
2], c in [-0.5, 0.5], phi in [0, 2 pi]. Every f is bounded, and the lag-1 self
edge is a function of the same family.

A draw of the parameters is checked on `REFERENCE_ROWS` simulated rows: every
edge needs a correlation and a partial correlation (given the other parents of
its child) of at least `MIN_STRENGTH` in absolute value. The check uses linear
correlations only. Look at `make_instance`.
"""
import numpy as np

from . import Instance
from .linear import first_rows, make_instance as _graph_instance, unroll_weights

MIN_STRENGTH = 0.05
REFERENCE_ROWS = 20000
MAX_TRIES = 50     # stationary graph: draws of all parameters
MAX_ROUNDS = 200   # nonstationary graph: rounds in which the failing edges are drawn again


def draw_parameters(edges: np.ndarray, rng) -> np.ndarray:
    """Return parameters (6, `*edges.shape`) for every slot.

    Parameters are w, a, s, c, k, `phi`; w is 0 where `edges` is 0."""
    shape = edges.shape
    P = np.empty((6,) + shape)
    P[0] = rng.uniform(0.8, 1.6, shape) * rng.choice([-1.0, 1.0], shape) * (np.asarray(edges) != 0)
    P[1] = rng.uniform(0.4, 1.0, shape)
    P[2] = rng.uniform(0.7, 2.0, shape)
    P[3] = rng.uniform(-0.5, 0.5, shape)
    P[4] = rng.uniform(0.7, 2.0, shape)
    P[5] = rng.uniform(0.0, 2 * np.pi, shape)
    return P


def _redraw_edge(P, u, v, rng) -> None:
    """Draw the six parameters of the edge u -> v again."""
    P[0, u, v] = rng.uniform(0.8, 1.6) * rng.choice([-1.0, 1.0])
    P[1, u, v] = rng.uniform(0.4, 1.0)
    P[2, u, v] = rng.uniform(0.7, 2.0)
    P[3, u, v] = rng.uniform(-0.5, 0.5)
    P[4, u, v] = rng.uniform(0.7, 2.0)
    P[5, u, v] = rng.uniform(0.0, 2 * np.pi)


def sample_nonlinear(A: np.ndarray, params: np.ndarray, M: int, rng) -> np.ndarray:
    """Return M observations (M, T * N) for `params` (6, T * N, T * N).

    Nodes are generated in index order."""
    n = A.shape[0]
    w, a, s, c, k, phi = params
    X = np.empty((M, n))
    for v in range(n):
        x = rng.standard_normal(M)
        for u in np.nonzero(A[:, v])[0]:
            xu = X[:, u]
            x += w[u, v] * (a[u, v] * np.tanh(s[u, v] * xu + c[u, v])
                            + (1.0 - a[u, v]) * np.sin(k[u, v] * xu + phi[u, v]))
        X[:, v] = x
    return X


def _partial_correlation(X, u, v, parents) -> float:
    """Return sample partial correlation of columns u and v of X.

    Conditions on the other members of `parents` (the parents of v without u).
    Without other parents, return the correlation."""
    others = [p for p in parents if p != u]
    if not others:
        return float(np.corrcoef(X[:, u], X[:, v])[0, 1])
    Z = np.column_stack([np.ones(len(X)), X[:, others]])
    ru = X[:, u] - Z @ np.linalg.lstsq(Z, X[:, u], rcond=None)[0]
    rv = X[:, v] - Z @ np.linalg.lstsq(Z, X[:, v], rcond=None)[0]
    return float(np.corrcoef(ru, rv)[0, 1])


def edge_margins(A: np.ndarray, X: np.ndarray) -> dict:
    """Return |partial correlation| and |correlation| for every edge.

    Returns `{(u, v): (partial, marginal)}` where `partial` conditions on the
    other parents of v."""
    out, parents = {}, {}
    for u, v in np.argwhere(A):
        u, v = int(u), int(v)
        pa = parents.setdefault(v, [int(p) for p in np.nonzero(A[:, v])[0]])
        marginal = abs(float(np.corrcoef(X[:, u], X[:, v])[0, 1]))
        out[(u, v)] = (abs(_partial_correlation(X, u, v, pa)), marginal)
    return out


def make_instance(N: int, T: int, lag: int, d: float, g: int, *, stationary: bool, seed: int = 0) -> Instance:
    """Return graph number g of a setting, with parameters and data.

    The graph is that of `linear.make_instance` with the same arguments.
    - `stationary=True`: one function per slot of the window graph, used at
      every step; all parameters are drawn again until the check passes (up to
      `MAX_TRIES` draws).
    - `stationary=False`: one function per edge of the time graph; in every
      round only the edges that fail the check are drawn again (up to
      `MAX_ROUNDS` rounds).

    When the check still fails at the limit, `make_instance` keeps the
    parameters drawn last and sets `info["guard_ok"]` to `False`.
    `info["parameters"]` has shape (6, T * N, T * N)."""
    base = _graph_instance(N, T, lag, d, g, stationary=stationary, seed=seed, weights=False)
    A = base.A
    stream = [seed, N, T, lag, int(round(d * 10)), g]
    rng = np.random.default_rng(np.random.SeedSequence(stream + [11 if stationary else 12]))
    data_seed = stream + [11011 if stationary else 12012]

    def reference(params, number):
        reference_rng = np.random.default_rng(np.random.SeedSequence(data_seed + [number]))
        return edge_margins(A, sample_nonlinear(A, params, REFERENCE_ROWS, reference_rng))

    info = dict(base.info)
    if stationary:
        for tries in range(1, MAX_TRIES + 1):
            P = draw_parameters(base.info["B"], rng)
            P[0][0] = 0.0   # no edge inside a time step
            params = np.stack([unroll_weights(P[i], T) for i in range(6)])
            partial = marginal = 1.0
            for pm, mm in reference(params, tries).values():
                partial, marginal = min(partial, pm), min(marginal, mm)
            if min(partial, marginal) >= MIN_STRENGTH:
                break
        ok = bool(min(partial, marginal) >= MIN_STRENGTH)
    else:
        params = draw_parameters(A, rng)
        info["n_redrawn_edges"] = 0
        for tries in range(1, MAX_ROUNDS + 1):
            margins = {e: tuple(m if np.isfinite(m) else 0.0 for m in pair)
                       for e, pair in reference(params, tries).items()}
            failing = [e for e, pair in margins.items() if min(pair) < MIN_STRENGTH]
            if not failing:
                break
            for u, v in failing:
                _redraw_edge(params, u, v, rng)
            info["n_redrawn_edges"] += len(failing)
        partial, marginal = min(m[0] for m in margins.values()), min(m[1] for m in margins.values())
        ok = not failing
    info.update(parameters=params, guard_tries=tries, guard_ok=ok,
                min_partial_correlation=partial, min_marginal_correlation=marginal)

    def draw(rows):
        return sample_nonlinear(A, params, rows, np.random.default_rng(np.random.SeedSequence(data_seed)))
    return Instance(A, N, T, lag, first_rows(draw, N, T), base.graph_sha1, info)
