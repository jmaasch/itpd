"""Iterative Temporal Parent Discovery (ITPD), Algorithm 1.

Data have shape (replicates, times, variables). Each CI observation is an
independent replicate trajectory. Node IDs are t * N + n, with zero-based
time t and variable n.

For a target at the paper's time t + 1, m = N * t - 1 excludes its lag-1
self parent. Lines 8, 9, and 10 use alpha1 = max(alpha, min(gamma, kappa / m));
Line 14 uses alpha2 = alpha / m. A test returns independent when p > level.
Line 6 visits candidates in descending time; Line 13 uses ascending time.
Both visit variables in increasing index within a time step.

A CI test that returns a NaN p-value cannot be evaluated. For example,
itpd_aistats.ci.FisherZ returns NaN when M - |S| - 3 <= 0; causal-learn's
fisherz returns p = 1 at 0 and raises an error below 0. A target with a NaN
p-value keeps only its lag-1 self edge and is listed in untested_targets_.
The run continues, and fit emits one warning for all such targets.
Backend exceptions propagate.

The algorithm assumes a causally sufficient, faithful temporal DAG with no
contemporaneous edges, empty first-slice parent sets, and every lag-1 self
edge. Oracle correctness does not imply finite-sample correctness.

Lag-1 self edges make each series' ancestors a prefix. Cached prefix
endpoints and initial-root bitsets describe the learned graph. Do not mutate
graph_ after fit; get_graph() returns an independent copy.

Run the oracle example with python itpd_clear/itpd.py. It requires NumPy,
NetworkX, and Matplotlib; --seed and --output set the seed and PNG path."""

import warnings
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path

import networkx as nx
import numpy as np


class _NotEvaluable(Exception):
    """A CI test returned a NaN p-value."""


class ITPD:
    """Discover parent sets in chronological order.

    Parameters
    ----------
    ci_test : str, callable, or Tigramite CondIndTest
        A causal-learn CIT method name (default 'fisherz'), a bound callable
        (x_id, y_id, conditioning_ids) -> p-value, or a configured Tigramite
        test such as ParCorr() or CMIknn(). Use a test appropriate to the data.
    backend : {'causal-learn', 'tigramite', 'oracle'}
        CI backend. Defaults to 'causal-learn'.
    alpha : float
        Base significance level (default 0.01). Independence is rejected
        when p <= the test level.
    gamma, kappa : float
        Phase I constants (defaults 0.2 and 50). For m target candidates,
        alpha1 = max(alpha, min(gamma, kappa / m)); Phase II uses alpha / m.
    ci_kwargs : dict or None
        Options for causal-learn's CIT constructor. Configure Tigramite
        options on the supplied test object.
    oracle_adjacency : array-like or None
        Square ground-truth matrix with nonzero [u, v] indicating u -> v.
        Requires backend='oracle' and the temporal assumptions above.
        Oracle d-separation returns 1 for independence and 0 for dependence.
        These values encode decisions, not statistical p-values.

    After fit, adjacency_ maps node IDs to frozen parent sets, and graph_
    contains the learned directed graph. ci_queries_ counts requested calls
    by phase, including calls a backend serves from its cache.
    max_conditioning_size_ records the largest conditioning set requested.
    untested_targets_ lists targets with a NaN p-value; each keeps only its
    lag-1 self edge, and fit emits one warning if the list is nonempty.

    Graph caches use O(N^2 T) prefix entries and O(N^2 T) root bits.
    CI backends may allocate additional caches or dense matrices."""

    def __init__(self, ci_test="fisherz", *, backend="causal-learn",
                 alpha=0.01, gamma=0.2, kappa=50, ci_kwargs=None,
                 oracle_adjacency=None):
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie strictly between 0 and 1")
        if not 0 < gamma < 1:
            raise ValueError("gamma must lie strictly between 0 and 1")
        if not kappa > 0:
            raise ValueError("kappa must be positive")
        if backend not in {"causal-learn", "tigramite", "oracle"}:
            raise ValueError("backend must be 'causal-learn', 'tigramite', or 'oracle'")
        if backend != "oracle" and oracle_adjacency is not None:
            raise ValueError("oracle_adjacency requires backend='oracle'")
        self.ci_test = ci_test
        self.backend = backend
        self.alpha = alpha
        self.gamma = gamma
        self.kappa = kappa
        self.ci_kwargs = dict(ci_kwargs or {})
        self.oracle_adjacency = oracle_adjacency

    def fit(self, V: np.ndarray | None = None, *, n_variables: int | None = None):
        """Learn parent sets and return this estimator.

        Parameters
        ----------
        V : ndarray or None
            Finite real data of shape (M, T, N), with M >= 2 independent
            replicate trajectories, T >= 1 time steps, and N >= 1 variables.
            Oracle mode uses the shape and ignores the values after validation.
        n_variables : int or None
            Number of variables. Required when V is omitted in oracle mode;
            T is then the oracle matrix dimension divided by n_variables.
            If V is supplied, this value must agree with V.shape[2]."""
        self._fitted = False
        if V is None:
            matrix = np.asarray(self.oracle_adjacency)
            if (self.backend != "oracle" or matrix.ndim != 2
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
        self._roots = [0] * p
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

    def marginally_connected(self, a: int, b: int) -> bool:
        """Return whether two nodes share an initial root in the learned DAG.

        This is equivalent to marginal d-connection. Each root counts as its
        own root for this query."""
        if a not in self.adjacency_ or b not in self.adjacency_:
            raise KeyError("Both nodes must have learned parent sets")
        return bool(self._roots[a] & self._roots[b])

    def _record(self, node: int, parents: set[int]):
        """Store parents and compute ancestor prefixes and root bitsets."""
        self.adjacency_[node] = frozenset(parents)
        self.graph_.add_node(node)
        self.graph_.add_edges_from((parent, node) for parent in sorted(parents))
        t, n = self.node_label(node)
        if t == 0:
            self._roots[node] = 1 << n
            return
        previous = node - self.N_
        # Inherit ancestors through the mandatory lag-1 self edge.
        self._prefix[node] = self._prefix[previous]
        self._prefix[node, n] = t - 1
        self._roots[node] = self._roots[previous]
        for parent in self.graph_.predecessors(node):
            if parent == previous:
                continue
            pt, pn = self.node_label(parent)
            np.maximum(self._prefix[node], self._prefix[parent],
                       out=self._prefix[node])
            self._prefix[node, pn] = max(self._prefix[node, pn], pt)
            self._roots[node] |= self._roots[parent]

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
        except _NotEvaluable:
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
            raise _NotEvaluable
        if not 0 <= p <= 1:
            raise ValueError(f"CI test returned invalid p-value {p} for "
                             f"({x}, {y} | {conditions})")
        return p > (self.alpha if level is None else level)

    def _make_test(self) -> Callable:
        if self.backend == "oracle":
            return self._make_oracle()
        if self.backend == "causal-learn":
            if isinstance(self.ci_test, str):
                from causallearn.utils.cit import CIT
                return CIT(self.data_, method=self.ci_test, **self.ci_kwargs)
            if not callable(self.ci_test) or self.ci_kwargs:
                raise ValueError("Supply a CIT method name or a bound p-value callable")
            return self.ci_test
        if self.ci_kwargs or not hasattr(self.ci_test, "run_test_raw"):
            raise ValueError("Supply a configured Tigramite test object")
        if getattr(self.ci_test, "significance", None) == "fixed_thres":
            raise ValueError("Use a Tigramite p-value test, not fixed_thres")

        def pvalue(x, y, conditions):
            result = self.ci_test.run_test_raw(
                x=self.data_[:, [x]], y=self.data_[:, [y]],
                z=self.data_[:, list(conditions)] if conditions else None,
            )
            return result[1]

        return pvalue

    def _make_oracle(self) -> Callable:
        matrix = np.asarray(self.oracle_adjacency)
        p = self.T_ * self.N_
        if (matrix.shape != (p, p) or matrix.dtype.kind not in "biuf"
                or not np.isfinite(matrix).all()):
            raise ValueError(f"oracle_adjacency must be a finite numeric ({p}, {p}) matrix")
        if self.ci_kwargs:
            raise ValueError("ci_kwargs are not used by the oracle")
        graph = nx.from_numpy_array(matrix != 0, create_using=nx.DiGraph)
        if any(u // self.N_ >= v // self.N_ for u, v in graph.edges):
            raise ValueError("Oracle edges must point strictly forward in time")
        if any(not graph.has_edge(v - self.N_, v) for v in range(self.N_, p)):
            raise ValueError("Oracle graph must contain every lag-1 autocorrelation edge")

        def pvalue(x, y, conditions):
            return float(nx.is_d_separator(graph, {x}, {y}, set(conditions)))

        return pvalue


def run_example(*, seed=7, output="itpd_example.png") -> ITPD:
    """Generate a temporal DAG, check oracle recovery, and save its graph.

    Use three series and five time points. Sample optional forward edges
    independently with probability 0.12 and include every lag-1 self edge.
    The oracle uses the DAG's d-separation relations without observations."""
    N, T = 3, 5
    rng = np.random.default_rng(seed)
    times = np.repeat(np.arange(T), N)
    adjacency = ((rng.random((N * T, N * T)) < 0.12)
                 & (times[:, None] < times[None, :])).astype(np.uint8)
    sources = np.arange(N * (T - 1))
    adjacency[sources, sources + N] = 1
    truth = nx.from_numpy_array(adjacency, create_using=nx.DiGraph)

    model = ITPD(backend="oracle", oracle_adjacency=adjacency).fit(n_variables=N)
    if not np.array_equal(model.get_adjacency_matrix(), adjacency):
        learned = set(model.get_graph().edges)
        expected = set(truth.edges)
        raise AssertionError(f"Oracle recovery failed: missing={expected - learned}, "
                             f"extra={learned - expected}")
    path = model.plot_graph(output)
    print(f"Exact recovery verified: {N * T} nodes, {truth.number_of_edges()} edges "
          f"(seed={seed}).")
    print(f"CI queries: Phase I={model.ci_queries_['I']}, "
          f"Phase II={model.ci_queries_['II']}; "
          f"maximum conditioning size={model.max_conditioning_size_}.")
    print(f"Learned graph saved to {path.resolve()}")
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ITPD's random-DAG oracle example.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed (default: 7)")
    parser.add_argument("--output", default="itpd_example.png", help="Output PNG path")
    args = parser.parse_args()
    run_example(seed=args.seed, output=args.output)
