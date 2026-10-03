"""Unrolled time graphs and the d-separation oracle (always on the FULL graph).

The oracle never sees a sub-graph: tests issued with a restricted candidate set (tau_max, S1 windows)
are answered on the full unrolled graph, the nodes that are not in the conditioning set act as latent.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TimeGraph:
    """Unrolled time graph on T*N nodes; node (variable n, time t) is column t * N + n. `B` is the window graph
    (tau+1, N, N) if the graph is stationary."""

    A: np.ndarray
    N: int
    T: int
    tau: int
    B: np.ndarray | None = None
    meta: dict = field(default_factory=dict)


def unroll(B: np.ndarray, T: int) -> np.ndarray:
    """Window graph B[l, i, j] (V^i_{t-l} -> V^j_t, l = 1..tau; B[0] unused) to the unrolled (T*N, T*N) matrix."""
    tau, N = B.shape[0] - 1, B.shape[1]
    A = np.zeros((T * N, T * N), dtype=bool)
    for t in range(1, T):
        for l in range(1, min(tau, t) + 1):
            A[(t - l) * N:(t - l + 1) * N, t * N:(t + 1) * N] = B[l] != 0
    return A


def autocorr_mask(N: int, T: int) -> np.ndarray:
    """Boolean (T*N, T*N) mask of the autocorrelation edges V^n_t -> V^n_{t+1}."""
    m = np.zeros((T * N, T * N), dtype=bool)
    for t in range(T - 1):
        for n in range(N):
            m[t * N + n, (t + 1) * N + n] = True
    return m


def check_assumptions(g: TimeGraph) -> dict:
    """Paper assumptions: self edge for every node and step, no contemporaneous edge, edges point forward in time."""
    N, T, A = g.N, g.T, g.A
    ac = autocorr_mask(N, T)
    ii, jj = np.nonzero(A)
    ti, tj = ii // N, jj // N
    return {
        "autocorr_present": bool(A[ac].all()),
        "no_contemporaneous": bool((ti != tj).all()),
        "forward_only": bool((tj > ti).all()),
        "max_lag": int((tj - ti).max()) if len(ii) else 0,
        "roots_at_t0": bool(not A[:, :N].any()),
    }


class DSep:
    """d-separation by the ancestral moral graph, with bitsets (python ints).

    `A[i, j]` = i -> j. Query `dsep(x, y, S)`; x, y are removed from S if present.
    Marginal queries use: d-connected given the empty set iff an'(x) and an'(y) intersect.
    """

    def __init__(self, A: np.ndarray):
        A = np.asarray(A) != 0
        n = A.shape[0]
        self.n = n
        self.par = [np.nonzero(A[:, j])[0].tolist() for j in range(n)]
        self.ch = [np.nonzero(A[i, :])[0].tolist() for i in range(n)]
        self.parmask = [sum(1 << p for p in ps) for ps in self.par]
        self.chmask = [sum(1 << c for c in cs) for cs in self.ch]
        # ancestors including self, by a topological order (Kahn)
        indeg = [len(p) for p in self.par]
        order = [i for i in range(n) if indeg[i] == 0]
        k = 0
        while k < len(order):
            u = order[k]
            k += 1
            for c in self.ch[u]:
                indeg[c] -= 1
                if indeg[c] == 0:
                    order.append(c)
        if len(order) != n:
            raise ValueError("graph has a directed cycle")
        self.anc = [0] * n
        for u in order:
            m = 1 << u
            for p in self.par[u]:
                m |= self.anc[p]
            self.anc[u] = m

    def dsep(self, x: int, y: int, S=()) -> bool:
        Sm = 0
        for s in S:
            if s != x and s != y:
                Sm |= 1 << s
        anc = self.anc
        if Sm == 0:
            return not (anc[x] & anc[y])
        Aset = anc[x] | anc[y]
        m = Sm
        while m:
            low = m & -m
            Aset |= anc[low.bit_length() - 1]
            m ^= low
        allowed = Aset & ~Sm
        parmask, chmask, ch = self.parmask, self.chmask, self.ch
        visited = 1 << x
        stack = [x]
        ybit = 1 << y
        while stack:
            u = stack.pop()
            nb = (parmask[u] | chmask[u]) & allowed
            for c in ch[u]:
                if (Aset >> c) & 1:
                    nb |= parmask[c] & allowed
            nb &= ~visited
            if nb & ybit:
                return False
            visited |= nb
            while nb:
                low = nb & -nb
                stack.append(low.bit_length() - 1)
                nb ^= low
        return True
