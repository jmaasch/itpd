"""DF-PC on data of shape (M, T, N), with the known time order.

    python dfpc.py --input in.npz --output out.npz --alpha 0.01 --source DIR [--time-limit 600]

`DIR` is a clone of the DF-PC repository (`requirements/pcmci-dfpc.txt` names
the address and the commit). See `_common.py` for I/O and test counting.
- Uses the DF-PC deduction engine (`cddd.inference.Deductor`: pure deduction,
  dependence first) and the interface of its CI tester.
- The variables are the T * N nodes of the time graph; there is no maximum lag.
- The CI test is the Fisher-z test. The counted tests are the tests that DF-PC
  computes, after its cache and its deductions.
- When the time limit passes, the status is "timeout" and the graph is empty.

The function `search` is adapted from `PCStable.run` of the DF-PC repository
(`cddd/algorithms.py`), which has the license below. The adaptation adds the
known edges and the time order.

MIT License

Copyright (c) 2026 DF-PC contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import sys
import time
from itertools import combinations

import numpy as np
from scipy.stats import norm

import _common


class FisherZ:
    """Fisher-z test of partial correlation on the columns of data (M, D).

    The partial correlation comes from the inverse of a submatrix of the
    correlation matrix. The test has M - |S| - 3 degrees of freedom."""

    def __init__(self, data):
        self.rows = data.shape[0]
        self.corr = np.corrcoef(data, rowvar=False)

    def __call__(self, x, y, S):
        """Return the two-sided p-value of x and y given S.

        Raise `ValueError` when M - |S| - 3 <= 0 or when the partial
        correlation is not finite."""
        S = [int(s) for s in S if s != x and s != y]
        df = self.rows - len(S) - 3
        if df <= 0:
            raise ValueError("The Fisher-z test needs more rows than the conditioning set has members plus 3")
        idx = [x, y] + S
        sub = self.corr[np.ix_(idx, idx)]
        try:
            P = np.linalg.inv(sub)
        except np.linalg.LinAlgError:
            P = np.linalg.pinv(sub)
        d = P[0, 0] * P[1, 1]
        if not np.isfinite(d) or d <= 0:
            raise ValueError("The partial correlation is not finite")
        r = -P[0, 1] / np.sqrt(d)
        r = min(max(r, -0.9999999), 0.9999999)
        z = np.sqrt(df) * 0.5 * np.log((1 + r) / (1 - r))
        return float(2 * (1 - norm.cdf(abs(z))))


def search(D, step, known, independent):
    """Return the adjacency matrix (D, D) of the skeleton search.

    step[v] is the time step of node v. Only pairs of nodes at different steps
    can be adjacent. A pair is tested with the later node as the target, given
    sets of nodes that are adjacent to the target and at steps before its step.
    The pairs in `known` are adjacent and are never tested. independent(target,
    candidate, set) gives the decision for one set."""
    adj = np.ones((D, D), dtype=np.int8)
    np.fill_diagonal(adj, 0)
    adj[step[:, None] == step[None, :]] = 0
    never_tested = set()
    for a, b in known:
        never_tested.update([(a, b), (b, a)])
        adj[a, b] = adj[b, a] = 1
    for k in range(D - 1):
        removed = []
        any_pool = False
        for target in range(D):
            adj_t = [i for i in range(D) if adj[i, target] == 1]
            for cand in adj_t:
                if (cand, target) in never_tested or step[cand] > step[target]:
                    continue
                pool = list(set(adj_t) - {cand})        # the element order of PCStable.run
                pool = [z for z in pool if step[z] < step[target]]
                if len(pool) < k:
                    continue
                any_pool = True
                for cs in combinations(pool, k):
                    if independent(target, cand, cs):
                        removed.append((target, cand))
                        break
        for a, b in removed:
            adj[a, b] = adj[b, a] = 0
        if not any_pool:
            break
    return adj


def main():
    """Read input, run DF-PC with counted Fisher-z tests, and write the graph."""
    p = _common.parser(__doc__.split("\n")[0])
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--time-limit", type=float, default=None, help="seconds")
    p.add_argument("--source", required=True, help="folder of the DF-PC clone")
    args = p.parse_args()
    sys.path.insert(0, args.source)
    from cddd.independence import CITester
    from cddd.inference import Deductor

    X = _common.read_input(args.input)
    M, T, N = X.shape
    D = T * N
    step = np.repeat(np.arange(T), N)
    test = FisherZ(X.reshape(M, D))
    counter = _common.TestCounter(args.time_limit)

    class Tester(CITester):
        """CI tester of DF-PC that computes the counted Fisher-z test."""

        def _compute_p_value(self, data, X, Y, cond_set):
            counter.record(X, Y, cond_set)
            p = test(X, Y, cond_set)
            return p, -p

    deductor = Deductor(Tester(), args.alpha, priority="dep", pure=True)
    known = {(t * N + n, (t + 1) * N + n) for t in range(T - 1) for n in range(N)}
    start = time.perf_counter()
    try:
        adj = search(D, step, known, lambda a, b, cs: deductor.deduce(None, a, b, set(cs)) == "ind")
        status = "ok"
    except TimeoutError:
        adj, status = np.zeros((D, D), dtype=np.int8), "timeout"
    seconds = time.perf_counter() - start
    A_hat = (adj != 0) & (step[:, None] < step[None, :])        # every edge points to the later node
    _common.write_output(args.output, A_hat, seconds, ["causal-learn", "numpy", "scipy"], status, counter)


if __name__ == "__main__":
    main()
