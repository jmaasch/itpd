"""Iterative Temporal Parent Discovery (ITPD), Algorithm 1.

Where to look:
- `ITPD.fit` runs Lines 1 to 4 and Line 16 of Algorithm 1 (the roots, the loop
  over time steps and series, and the update of the adjacency list).
- `ITPD._discover` runs Lines 5 to 15 for one target; its comments give the
  line numbers.
- `ITPD._levels` returns `alpha1` and `alpha2`.

Data have shape (M, T, N): M independent observations of N series at T time
steps. The node of series n at step t is t * N + n, both zero-based. The paper
counts time from 1 and a target is at time t + 1; then m = N * t - 1 counts its
candidates without the lag-1 self parent. The code counts steps from 0.

Lines 8, 9 and 10 use alpha1 = max(alpha, min(gamma, kappa / m)) (`alpha1` is
the paper's alpha_1); Line 14 uses alpha2 = alpha / m (`alpha2` is alpha_2). A
test answers independent when p > level (dependent when p <= level). Line 6
visits candidates in descending time; Line 13 uses ascending time. Both visit
series in increasing index within a time step.

A CI test that returns a NaN p-value, or raises `NotEvaluable`, cannot be
evaluated. For example, `FisherZ` returns NaN when M - |S| - 3 <= 0. The target
of such a test keeps only its lag-1 self edge and is listed in
`untested_targets_`. The run continues, and `fit` emits one warning for all
such targets. Other exceptions of a CI test propagate.

The algorithm assumes a causally sufficient, faithful temporal DAG with no
edges inside a time step, no parents in the first time step, and every lag-1
self edge. Exact recovery with a d-separation oracle does not imply exact
recovery from finite data.

Lag-1 self edges make the ancestors of a node a prefix of each series: for a
node and a series, the ancestors in that series are the steps 0, ..., k for
some k, and the fitted object stores k. Do not mutate `graph_` after `fit`;
`get_graph()` returns an independent copy."""

import warnings
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path

import networkx as nx
import numpy as np

from .ci_tests import GCM, DSeparation, FisherZ, NotEvaluable
from .graphs import check_temporal


class ITPD:
    """Algorithm 1: discover parent sets in chronological order.

    Parameters
    ----------
    ci_test : {'fisherz', 'gcm'} or callable
        'fisherz' (default) is the Fisher-z test `FisherZ` and 'gcm' is the
        test `GCM`, each on the data given to `fit`. A callable (x, y,
        S) -> p-value takes node IDs and a sorted tuple of node IDs. Use a test
        appropriate to the data.
    alpha : float
        Nominal significance level (default 0.01). Independence is rejected
        when p <= the test level.
    gamma, kappa : float
        Phase I constants (defaults 0.2 and 50). For m candidates of a target,
        alpha1 = max(alpha, min(gamma, kappa / m)); Phase II uses
        alpha2 = alpha / m.
    oracle_adjacency : array-like or None
        True graph as a square matrix with nonzero [u, v] for u -> v. When
        given, the CI test is the d-separation oracle of this graph
        (`DSeparation`) and `ci_test` is not used. The graph must meet the
        temporal assumptions above.

    Attributes
    ----------
    adjacency_ : dict
        Maps node IDs to frozen parent sets (`frozenset`).
    graph_ : networkx.DiGraph
        The learned directed graph.
    ci_queries_ : Counter
        Counts the requested tests by phase (keys `'I'` and `'II'`).
    max_conditioning_size_ : int
        Largest conditioning set requested.
    untested_targets_ : list of int
        Targets with a test that could not be evaluated; each keeps only its
        lag-1 self edge, and `fit` emits one warning if the list is nonempty.
    M_, T_, N_ : int
        Number of observations, time steps, and series."""

    def __init__(self, ci_test="fisherz", *, alpha=0.01, gamma=0.2, kappa=50,
                 oracle_adjacency=None):
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie strictly between 0 and 1")
        if not 0 < gamma < 1:
            raise ValueError("gamma must lie strictly between 0 and 1")
        if not kappa > 0:
            raise ValueError("kappa must be positive")
        self.ci_test = ci_test
        self.alpha = alpha
        self.gamma = gamma
        self.kappa = kappa
        self.oracle_adjacency = oracle_adjacency

    def fit(self, V: np.ndarray | None = None, *, n_variables: int | None = None):
        """Learn parent sets and return this estimator.

        Parameters
        ----------
        V : ndarray or None
            Finite real data of shape (M, T, N), with M >= 2 independent
            observations, T >= 1 time steps, and N >= 1 series. With
            `oracle_adjacency`, only the shape is used, and V may be omitted.
        n_variables : int or None
            Number of series N. Required when V is omitted; T is then the
            dimension of `oracle_adjacency` divided by `n_variables`. If V is
            supplied, this value must agree with `V.shape[2]`.

        Returns
        -------
        self : ITPD
            This fitted estimator. With `oracle_adjacency` and no V, `M_` is 0."""
        self._fitted = False
        if V is None:
            matrix = np.asarray(self.oracle_adjacency)
            if (self.oracle_adjacency is None or matrix.ndim != 2
                    or matrix.shape[0] != matrix.shape[1]
                    or not isinstance(n_variables, (int, np.integer))
                    or isinstance(n_variables, (bool, np.bool_))
                    or n_variables < 1 or matrix.shape[0] == 0
                    or matrix.shape[0] % n_variables):
                raise ValueError("Without V, supply a square oracle matrix and a "
                                 "positive n_variables dividing its dimension")
            self.M_, self.T_, self.N_ = 0, matrix.shape[0] // n_variables, int(n_variables)
            self.data_ = None
        else:
            V = np.asarray(V)
            if V.ndim != 3 or min(V.shape) < 1 or V.shape[0] < 2:
                raise ValueError("V must have shape (M >= 2, T >= 1, N >= 1)")
            if V.dtype.kind not in "biuf" or not np.isfinite(V).all():
                raise ValueError("V must contain finite real numeric values")
            self.M_, self.T_, self.N_ = V.shape
            if n_variables is not None and n_variables != self.N_:
                raise ValueError("n_variables disagrees with V.shape[2]")
            self.data_ = V.reshape(self.M_, -1)
        p = self.T_ * self.N_
        self.graph_ = nx.DiGraph()
        self.adjacency_ = {}
        self._prefix = np.full((p, self.N_), -1, dtype=np.intp)
        self.ci_queries_ = Counter()
        self.max_conditioning_size_ = 0
        self.untested_targets_ = []
        self._pvalue = self._make_test()
        for n in range(self.N_):
            self._record(n, set())
        for t in range(self.T_ - 1):
            for n in range(self.N_):
                x, y = t * self.N_ + n, (t + 1) * self.N_ + n
                self._record(y, self._discover(x, y, t))
        if self.untested_targets_:
            warnings.warn(
                f"{len(self.untested_targets_)} target(s) had a CI test that "
                "could not be evaluated (NaN p-value); each keeps only its "
                "lag-1 self edge. See untested_targets_.",
                RuntimeWarning, stacklevel=2)
        self._fitted = True
        return self

    def _require_fitted(self):
        if not getattr(self, "_fitted", False):
            raise RuntimeError("Call fit successfully before accessing the final graph")

    def get_graph(self) -> nx.DiGraph:
        """Return an independent copy of the fitted graph."""
        self._require_fitted()
        return self.graph_.copy()

    def get_adjacency_matrix(self) -> np.ndarray:
        """Return a binary matrix in node-ID order, with [u, v] = 1 for u -> v."""
        self._require_fitted()
        return nx.to_numpy_array(self.graph_, nodelist=range(self.T_ * self.N_),
                                 dtype=np.uint8, weight=None)

    def plot_graph(self, path="itpd.png", *, figsize=None, dpi=200) -> Path:
        """Save all nodes and edges as PNG, with time on x and variable on y.

        Labels are zero-based (time, variable) pairs. Return the output path,
        create parent directories, and replace any existing output file.
        Use a headless canvas without changing Matplotlib's global backend."""
        self._require_fitted()
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        path = Path(path).expanduser().with_suffix(".png")
        path.parent.mkdir(parents=True, exist_ok=True)
        if figsize is None:
            figsize = (max(6, min(30, 1.8 * self.T_)), max(3, min(24, 1.5 * self.N_)))
        fig = Figure(figsize=figsize, layout="constrained")
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        positions = {v: (v // self.N_, -(v % self.N_)) for v in self.graph_}
        labels = {v: str(self.node_label(v)) for v in self.graph_}
        nx.draw_networkx(self.graph_, pos=positions, labels=labels, ax=ax,
                         node_color=[v % self.N_ for v in self.graph_],
                         cmap="tab20", node_size=650, font_size=8,
                         arrows=True, arrowsize=16, width=1,
                         connectionstyle="arc3,rad=0.15")
        ax.set_title("ITPD: time increases left to right; labels are (time, variable)")
        ax.margins(0.15)
        fig.savefig(path, format="png", dpi=dpi)
        return path

    def node_label(self, node: int) -> tuple[int, int]:
        """Return the zero-based (time, variable) pair for a node ID."""
        return divmod(node, self.N_)

    def ancestors(self, node: int) -> set[int]:
        """Return strict ancestors from the learned graph's cached prefixes."""
        if node not in self.adjacency_:
            raise KeyError(node)
        return {t * self.N_ + n
                for n, end in enumerate(self._prefix[node])
                for t in range(int(end) + 1)}

    def _record(self, node: int, parents: set[int]):
        """Store the parents of a node and its ancestor prefixes."""
        self.adjacency_[node] = frozenset(parents)
        self.graph_.add_node(node)
        self.graph_.add_edges_from((parent, node) for parent in sorted(parents))
        t, n = self.node_label(node)
        if t == 0:
            return
        previous = node - self.N_
        # Inherit ancestors through the mandatory lag-1 self edge.
        self._prefix[node] = self._prefix[previous]
        self._prefix[node, n] = t - 1
        for parent in self.graph_.predecessors(node):
            if parent == previous:
                continue
            pt, pn = self.node_label(parent)
            np.maximum(self._prefix[node], self._prefix[parent],
                       out=self._prefix[node])
            self._prefix[node, pn] = max(self._prefix[node, pn], pt)

    def _levels(self, t: int) -> tuple[float, float]:
        """Return (alpha1, alpha2) for a target at zero-based time t + 1.

        The candidate count is m = N * (t + 1) - 1. When N = 1 and t = 0,
        no CI test runs; use m = 1 to avoid division by zero."""
        m = max((t + 1) * self.N_ - 1, 1)
        return max(self.alpha, min(self.gamma, self.kappa / m)), self.alpha / m

    def _discover(self, x: int, y: int, t: int) -> set[int]:
        alpha1, alpha2 = self._levels(t)
        try:
            # Line 5: S starts with the lag-1 self parent X.
            S = {x}
            ancestor_prefix = self._prefix[x].copy()
            # Line 6: descending time, then increasing variable index.
            for s in range(t, -1, -1):
                for n in range(self.N_):
                    z = s * self.N_ + n
                    if z == x:
                        continue
                    # Lines 7-8: test marginal independence only if Z is not
                    # a learned ancestor of any member of S.
                    if (s > ancestor_prefix[n]
                            and self._independent(z, y, level=alpha1)):
                        continue
                    # Line 9: condition on the current S.
                    if self._independent(z, y, S, level=alpha1):
                        continue
                    # Line 10: dependence given the learned parents of Z adds Z to S.
                    if not self._independent(z, y, self.adjacency_[z],
                                             level=alpha1):
                        S.add(z)
                        np.maximum(ancestor_prefix, self._prefix[z],
                                   out=ancestor_prefix)

            # Line 11: pa(Y) starts with X, which is not tested.
            parents = {x}
            # Line 12: S' starts as a copy of S.
            remaining = S.copy()
            # Line 13: ascending time, then increasing variable index.
            for z in sorted(S - parents):
                # Line 14: dependence given S' minus Z makes Z a parent.
                if self._independent(z, y, remaining - {z}, phase="II",
                                     level=alpha2):
                    # Line 15: remove Z from S' on independence.
                    remaining.remove(z)
                else:
                    parents.add(z)
        except NotEvaluable:
            self.untested_targets_.append(y)
            return {x}
        # Line 16: fit stores the parent set.
        return parents

    def _independent(self, x: int, y: int, conditions: Iterable[int] = (),
                     *, phase="I", level: float | None = None) -> bool:
        conditions = tuple(sorted(conditions))
        self.ci_queries_[phase] += 1
        self.max_conditioning_size_ = max(self.max_conditioning_size_,
                                          len(conditions))
        p = float(self._pvalue(x, y, conditions))
        if np.isnan(p):
            raise NotEvaluable
        if not 0 <= p <= 1:
            raise ValueError(f"CI test returned invalid p-value {p} for "
                             f"({x}, {y} | {conditions})")
        return p > (self.alpha if level is None else level)

    def _make_test(self) -> Callable:
        """Return the CI test callable `(x, y, S) -> p-value` for this fit.

        Use the d-separation oracle if `oracle_adjacency` is given, else
        `FisherZ` or `GCM` on the data, else the given callable. Raise
        `ValueError` otherwise."""
        if self.oracle_adjacency is not None:
            return self._make_oracle()
        if self.ci_test == "fisherz":
            return FisherZ(self.data_)
        if self.ci_test == "gcm":
            return GCM(self.data_)
        if not callable(self.ci_test):
            raise ValueError("ci_test must be 'fisherz', 'gcm', or a callable "
                             "(x, y, S) -> p-value")
        return self.ci_test

    def _make_oracle(self) -> Callable:
        """Validate `oracle_adjacency` and return its `DSeparation` oracle.

        Require a finite numeric matrix of shape (T * N, T * N) that passes
        `graphs.check_temporal`, else raise `ValueError`."""
        matrix = np.asarray(self.oracle_adjacency)
        p = self.T_ * self.N_
        if (matrix.shape != (p, p) or matrix.dtype.kind not in "biuf"
                or not np.isfinite(matrix).all()):
            raise ValueError(f"oracle_adjacency must be a finite numeric ({p}, {p}) matrix")
        check_temporal(matrix, self.N_, self.T_)
        return DSeparation(matrix != 0)
