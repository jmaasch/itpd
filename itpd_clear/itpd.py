"""Iterative Temporal Parent Discovery (ITPD).

Install: pip install numpy networkx matplotlib causal-learn tigramite

Run the reproducible oracle example: python itpd.py
Optional: python itpd.py --seed 42 --output learned_graph.png
The example requires only numpy, networkx, and matplotlib.

    model = ITPD(ci_test="fisherz", alpha=0.05).fit(V)
    parents = model.adjacency_                 # node -> frozenset of parents
    graph = model.graph_                      # networkx.DiGraph

    from tigramite.independence_tests.parcorr import ParCorr
    model = ITPD(ParCorr(), backend="tigramite").fit(V)

    model = ITPD(backend="oracle", oracle_adjacency=A).fit(n_variables=N)
    graph = model.get_graph()                  # independent copy
    matrix = model.get_adjacency_matrix()      # matrix[u, v] = 1 iff u -> v
    model.plot_graph("itpd.png")

V has shape (replicates, times, variables). CI observations are independent
replicate trajectories, NOT pooled time points. Nodes use zero-based integer
IDs t * N + n; node_label(node) returns (t, n).

Assumes a causally sufficient, faithful temporal DAG, no contemporaneous edges,
empty first-slice parent sets, and every lag-1 self edge. Oracle correctness does
not imply finite-sample correctness. Choose a CI test appropriate to the data.

Ancestry is stored as N prefix endpoints per node: lag-1 self edges make each
series' ancestors a prefix. Endpoints are computed once when parents are fixed,
first from autocorrelation, then from the remaining parent adjacency entries.
Marginal d-connection is equivalent to sharing an initial root, so cached root
bitsets replace repeated d-separation traversals. These caches describe the
learned graph, not necessarily the true graph. Do not mutate graph_ after fit.
"""

from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path

import networkx as nx
import numpy as np


class ITPD:
    """Discover parents chronologically, screening candidates in reverse time.

    Parameters
    ----------
    ci_test : str, callable, or Tigramite CondIndTest
        For causal-learn, a CIT method name (default 'fisherz'), or a bound
        callable (x_id, y_id, conditioning_ids) -> p-value. For Tigramite,
        supply an instantiated test, e.g. ParCorr() or CMIknn().
    backend : {'causal-learn', 'tigramite', 'oracle'}
    alpha : float
        Reject independence when p <= alpha.
    ci_kwargs : dict or None
        Options passed to causal-learn's CIT constructor. Configure Tigramite
        options on the supplied test object instead.
    oracle_adjacency : array-like or None
        Square ground-truth matrix with nonzero [u, v] indicating u -> v.
        Used only by backend='oracle'; must satisfy the temporal assumptions.
        Oracle CI uses NetworkX d-separation, returning 1 for independence and
        0 for dependence. These are decision encodings, not statistical p-values.

    Attributes after fit
    --------------------
    adjacency_ : dict[int, frozenset[int]]
    graph_ : networkx.DiGraph
    ci_queries_ : Counter
        Calls requested by phase. A backend may cache repeated calls internally.
    max_conditioning_size_ : int

    Graph-cache storage is O(N^2 T) prefix entries plus O(N^2 T) root bits.
    CI backends may allocate additional caches or dense matrices.
    """

    def __init__(self, ci_test="fisherz", *, backend="causal-learn",
                 alpha=0.05, ci_kwargs=None, oracle_adjacency=None):
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie strictly between 0 and 1")
        if backend not in {"causal-learn", "tigramite", "oracle"}:
            raise ValueError("backend must be 'causal-learn', 'tigramite', or 'oracle'")
        if backend != "oracle" and oracle_adjacency is not None:
            raise ValueError("oracle_adjacency requires backend='oracle'")
        self.ci_test = ci_test
        self.backend = backend
        self.alpha = alpha
        self.ci_kwargs = dict(ci_kwargs or {})
        self.oracle_adjacency = oracle_adjacency

    def fit(self, V: np.ndarray | None = None, *, n_variables: int | None = None):
        """Fit (M, T, N) data, or omit V for oracle mode and specify N.

        With no V, T is inferred from the oracle matrix dimension divided by N.
        Oracle mode also accepts V, using its shape rather than its values.
        """
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
        self._pvalue = self._make_test()
        for n in range(self.N_):
            self._record(n, set())
        for t in range(self.T_ - 1):
            for n in range(self.N_):
                x, y = t * self.N_ + n, (t + 1) * self.N_ + n
                self._record(y, self._discover(x, y, t))
        self._fitted = True
        return self

    def _require_fitted(self):
        if not getattr(self, "_fitted", False):
            raise RuntimeError("Call fit successfully before accessing the final graph")

    def get_graph(self) -> nx.DiGraph:
        """Return a copy of the final graph, protecting internal graph caches."""
        self._require_fitted()
        return self.graph_.copy()

    def get_adjacency_matrix(self) -> np.ndarray:
        """Return a binary matrix in node-ID order; entry [u, v] means u -> v."""
        self._require_fitted()
        return nx.to_numpy_array(self.graph_, nodelist=range(self.T_ * self.N_),
                                 dtype=np.uint8, weight=None)

    def plot_graph(self, path="itpd.png", *, figsize=None, dpi=200) -> Path:
        """Export all nodes and edges as PNG, with time on x and series on y.

        Node labels are (time, variable), both zero-based. Returns the output
        path; creates parent directories and replaces any existing output file.
        Uses a headless canvas without changing Matplotlib's global backend.
        """
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
        """Return the zero-based (time, variable) for an integer node ID."""
        return divmod(node, self.N_)

    def ancestors(self, node: int) -> set[int]:
        """Expand cached prefixes into the strict ancestors of a learned node."""
        if node not in self.adjacency_:
            raise KeyError(node)
        return {t * self.N_ + n
                for n, end in enumerate(self._prefix[node])
                for t in range(int(end) + 1)}

    def marginally_connected(self, a: int, b: int) -> bool:
        """Exact marginal d-connection in the learned DAG, using common roots.

        A collider-free path exists iff the nodes share an ancestor, equivalently
        an initial root. A root counts as its own root for this query.
        """
        if a not in self.adjacency_ or b not in self.adjacency_:
            raise KeyError("Both nodes must have learned parent sets")
        return bool(self._roots[a] & self._roots[b])

    def _record(self, node: int, parents: set[int]):
        """Commit parents and compute caches once, autocorrelation first."""
        self.adjacency_[node] = frozenset(parents)
        self.graph_.add_node(node)
        self.graph_.add_edges_from((parent, node) for parent in sorted(parents))
        t, n = self.node_label(node)
        if t == 0:
            self._roots[node] = 1 << n
            return
        previous = node - self.N_
        # Mandatory self-chain supplies the own-series prefix and inherited
        # cross-series ancestors without traversing that chain again.
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

    def _discover(self, x: int, y: int, t: int) -> set[int]:
        S = {x}
        ancestor_prefix = self._prefix[x].copy()
        # Phase I: descending time, increasing variable ID within each slice.
        for s in range(t, -1, -1):
            for n in range(self.N_):
                z = s * self.N_ + n
                if z in S:
                    continue
                if s > ancestor_prefix[n] and self._independent(z, y):
                    continue
                if (self.marginally_connected(z, x)
                        and self._independent(z, y, (x,))):
                    continue
                if not self._independent(z, y, self.adjacency_[z]):
                    S.update(self.ancestors(z))
                    S.add(z)
                    np.maximum(ancestor_prefix, self._prefix[z],
                               out=ancestor_prefix)

        # Phase II: fixed iteration set, mutable conditioning set including
        # all known parents.
        parents = {z for z in S if z // self.N_ == t}
        remaining = S.copy()
        for z in S - parents:
            if self._independent(z, y, remaining - {z}, phase="II"):
                remaining.remove(z)
            else:
                parents.add(z)
        return parents

    def _independent(self, x: int, y: int, conditions: Iterable[int] = (),
                     *, phase="I") -> bool:
        conditions = tuple(sorted(conditions))
        self.ci_queries_[phase] += 1
        self.max_conditioning_size_ = max(self.max_conditioning_size_,
                                          len(conditions))
        p = float(self._pvalue(x, y, conditions))
        if not np.isfinite(p) or not 0 <= p <= 1:
            raise ValueError(f"CI test returned invalid p-value {p} for "
                             f"({x}, {y} | {conditions})")
        return p > self.alpha

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
        # Keep ground truth separate from the incrementally recovered graph.
        nx.freeze(graph)

        def pvalue(x, y, conditions):
            return float(nx.is_d_separator(graph, {x}, {y}, set(conditions)))

        return pvalue


def run_example(*, seed=7, output="itpd_example.png") -> ITPD:
    """Generate a temporal DAG, verify oracle recovery, and export its graph.

    Uses three series and five time points. Optional forward edges are sampled
    independently with probability 0.12; every lag-1 self edge is then forced.
    The oracle supplies the faithful DAG independence model directly, so no
    structural equations, observations, or statistical CI tests are needed.
    """
    N, T = 3, 5
    rng = np.random.default_rng(seed)
    times = np.repeat(np.arange(T), N)
    adjacency = ((rng.random((N * T, N * T)) < 0.12)
                 & (times[:, None] < times[None, :])).astype(np.uint8)
    sources = np.arange(N * (T - 1))
    adjacency[sources, sources + N] = 1
    truth = nx.from_numpy_array(adjacency, create_using=nx.DiGraph)
    assert nx.is_directed_acyclic_graph(truth)
    assert all(times[u] < times[v] for u, v in truth.edges)
    assert all(truth.has_edge(u, u + N) for u in sources)
    assert all(truth.in_degree(n) == 0 for n in range(N))

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
