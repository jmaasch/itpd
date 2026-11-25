"""Lasso: one L1-penalized regression per target with a fixed penalty."""
import warnings

import numpy as np

from ._targets import Result


def lasso(V) -> Result:
    """Run the Lasso baseline on data V of shape (M, T, N).

    For a target y, the lag-1 self parent x is projected out of y and of
    every candidate. The residuals are scaled to unit variance, and y is
    regressed on the candidates with the penalty sqrt(2 log(p) / M) for p
    candidates. The parents are the candidates with a nonzero coefficient.
    The self edge is always in the estimate, and no target is untested.
    """
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import lasso_path

    V = np.asarray(V, dtype=float)
    M, T, N = V.shape
    D = V.reshape(M, T * N)
    D = D - D.mean(0)
    adjacency = np.zeros((T * N, T * N), dtype=bool)
    for y in range(N, T * N):
        x = y - N
        candidates = [z for z in range(y // N * N) if z != x]
        adjacency[x, y] = True
        xv = D[:, x]
        ry, Z = D[:, y], D[:, candidates]
        xx = float(xv @ xv)
        if xx > 0:
            ry = ry - xv * (float(xv @ ry) / xx)
            Z = Z - np.outer(xv, (xv @ Z) / xx)
        sd = Z.std(0)
        keep = sd > 1e-12                  # Drop constant columns.
        sy = ry.std()
        if sy <= 1e-12 or not keep.any():
            continue
        Zs = Z[:, keep] / sd[keep]
        ys = ry / sy
        penalty = float(np.sqrt(2.0 * np.log(max(Zs.shape[1], 2)) / M))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            _, coefficients, _ = lasso_path(Zs, ys, alphas=np.array([penalty]),
                                            max_iter=5000)
        kept = [z for z, k in zip(candidates, keep) if k]
        for j in np.nonzero(coefficients[:, 0])[0]:
            adjacency[kept[j], y] = True
    return Result(adjacency, [])
