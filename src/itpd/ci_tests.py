"""Conditional independence (CI) tests.

A CI test is a callable (x, y, S) -> p-value. x and y are column indices of
the data, and S is a collection of column indices. The pair is dependent at a
level when p <= level. A test returns NaN when it cannot be evaluated.
Members of S equal to x or y are ignored.

FisherZ is the test for linear Gaussian data, GCM is a test for nonlinear
data, and DSeparation answers from a known graph.
"""
import numpy as np
from scipy.special import erfc, log_ndtr


class NotEvaluable(RuntimeError):
    """A CI test could not be evaluated, for example with too few observations."""


def _as_columns(data) -> np.ndarray:
    """Return data as a float array (M, D). Data (M, T, N) become (M, T * N)."""
    data = np.asarray(data, dtype=float)
    return data.reshape(data.shape[0], -1)


class FisherZ:
    """Fisher-z test of the partial correlation.

    data has shape (M, D): M independent observations of D columns. The
    correlation matrix is computed once. With dof = M - |S| - 3, the statistic
    is sqrt(dof) * |atanh(r)| for the partial correlation r of x and y given
    S, and the p-value is two-sided normal. The test returns NaN when
    dof <= 0.

    The last bits of a p-value depend on the order of x, y and S.
    """

    def __init__(self, data):
        data = _as_columns(data)
        self.n = data.shape[0]
        self.corr = np.corrcoef(data.T)

    def _statistic(self, x: int, y: int, S):
        """Return sqrt(dof) * |z|, or None when dof <= 0."""
        S = [int(s) for s in S if s != x and s != y]
        dof = self.n - len(S) - 3
        if dof <= 0:
            return None
        C = self.corr
        xy = [x, y]
        # Partial covariance of (x, y) given S: a Schur complement of the
        # correlation matrix.
        P = C[np.ix_(xy, xy)]
        if S:
            Css = C[np.ix_(S, S)]
            Csx = C[np.ix_(S, xy)]
            try:
                K = np.linalg.solve(Css, Csx)
            except np.linalg.LinAlgError:
                K = np.linalg.lstsq(Css, Csx, rcond=None)[0]
            P = P - Csx.T @ K
        d = abs(P[0, 0] * P[1, 1])
        if d <= 0:
            return 0.0
        r = P[0, 1] / np.sqrt(d)
        if abs(r) >= 1.0:
            r = (1.0 - np.finfo(float).eps) * np.sign(r)
        z = 0.5 * np.log((1.0 + r) / (1.0 - r))
        return np.sqrt(dof) * abs(z)

    def __call__(self, x: int, y: int, S=()) -> float:
        stat = self._statistic(x, y, S)
        if stat is None:
            return float("nan")
        return float(erfc(stat / np.sqrt(2.0)))

    def neglogp(self, x: int, y: int, S=()) -> float:
        """Return -log p of the same test, finite when p underflows to 0.

        It ranks candidates with equal p-values. NaN when dof <= 0."""
        stat = self._statistic(x, y, S)
        if stat is None:
            return float("nan")
        return float(-(np.log(2.0) + log_ndtr(-stat)))


class GCM:
    """Generalized covariance measure with gradient-boosted regressions.

    data has shape (M, D). The observations are split once into two folds.
    The residual of x given S on one fold comes from a
    HistGradientBoostingRegressor fitted on the other fold, and likewise for
    y. With R = residual of x times residual of y, the statistic is
    sqrt(M) * mean(R) / sd(R), and the p-value is two-sided normal. With an
    empty S the residuals are the centered columns. The test returns NaN only
    when M < 8.

    The p-values depend on the scikit-learn version.
    """

    def __init__(self, data, max_iter: int = 60, seed: int = 0):
        self.data = _as_columns(data)
        self.n = self.data.shape[0]
        self.max_iter = max_iter
        self.seed = seed
        self._fold = np.random.default_rng(seed).permutation(self.n) % 2

    def _residual(self, target: np.ndarray, Z) -> np.ndarray:
        if Z is None:
            return target - target.mean()
        from sklearn.ensemble import HistGradientBoostingRegressor
        residual = np.empty(len(target))
        for fold in (0, 1):
            train, test = self._fold != fold, self._fold == fold
            model = HistGradientBoostingRegressor(max_iter=self.max_iter,
                                                  random_state=self.seed)
            model.fit(Z[train], target[train])
            residual[test] = target[test] - model.predict(Z[test])
        return residual

    def __call__(self, x: int, y: int, S=()) -> float:
        S = [int(s) for s in S if s != x and s != y]
        if self.n < 8:
            return float("nan")
        Z = self.data[:, S] if S else None
        R = self._residual(self.data[:, x], Z) * self._residual(self.data[:, y], Z)
        m, v = R.mean(), (R ** 2).mean() - R.mean() ** 2
        if v <= 1e-300:
            return 1.0
        stat = np.sqrt(self.n) * m / np.sqrt(v)
        return float(erfc(abs(stat) / np.sqrt(2.0)))


class DSeparation:
    """d-separation oracle of a known DAG.

    A is a square matrix with A[u, v] != 0 for an edge u -> v. A call returns
    1.0 when x and y are d-separated given S and 0.0 otherwise. These values
    are decisions, not p-values. A graph with a directed cycle raises
    ValueError.

    Sets of nodes are bit sets (Python integers). x and y are d-separated
    given S when S cuts every path between them in the moral graph of the
    ancestors of x, y and S.
    """

    def __init__(self, A):
        A = np.asarray(A) != 0
        n = A.shape[0]
        self._children = [np.nonzero(A[i, :])[0].tolist() for i in range(n)]
        parents = [np.nonzero(A[:, j])[0].tolist() for j in range(n)]
        self._parent_bits = [sum(1 << p for p in ps) for ps in parents]
        self._child_bits = [sum(1 << c for c in cs) for cs in self._children]
        # Topological order, then ancestors of every node (the node included).
        indegree = [len(ps) for ps in parents]
        order = [i for i in range(n) if indegree[i] == 0]
        k = 0
        while k < len(order):
            u = order[k]
            k += 1
            for c in self._children[u]:
                indegree[c] -= 1
                if indegree[c] == 0:
                    order.append(c)
        if len(order) != n:
            raise ValueError("The graph has a directed cycle")
        self._ancestors = [0] * n
        for u in order:
            bits = 1 << u
            for p in parents[u]:
                bits |= self._ancestors[p]
            self._ancestors[u] = bits

    def separated(self, x: int, y: int, S=()) -> bool:
        """Return whether x and y are d-separated given S."""
        given = 0
        for s in S:
            if s != x and s != y:
                given |= 1 << int(s)
        ancestors = self._ancestors
        if given == 0:
            return not (ancestors[x] & ancestors[y])
        ancestral = ancestors[x] | ancestors[y]
        rest = given
        while rest:
            low = rest & -rest
            ancestral |= ancestors[low.bit_length() - 1]
            rest ^= low
        allowed = ancestral & ~given
        parent_bits, child_bits = self._parent_bits, self._child_bits
        # Search the moral ancestral graph from x without entering S.
        visited = 1 << x
        stack = [x]
        y_bit = 1 << y
        while stack:
            u = stack.pop()
            neighbors = (parent_bits[u] | child_bits[u]) & allowed
            for c in self._children[u]:
                if (ancestral >> c) & 1:
                    neighbors |= parent_bits[c] & allowed
            neighbors &= ~visited
            if neighbors & y_bit:
                return False
            visited |= neighbors
            while neighbors:
                low = neighbors & -neighbors
                stack.append(low.bit_length() - 1)
                neighbors ^= low
        return True

    def __call__(self, x: int, y: int, S=()) -> float:
        return 1.0 if self.separated(x, y, S) else 0.0
