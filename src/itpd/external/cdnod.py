"""CD-NOD (`andrey-core`, release 0.17.1) on data of shape (M, T, N).

    python cdnod.py --input in.npz --output out.npz --alpha 0.01 --max-lag 1

See `_common.py` for I/O and time graphs.
- The data are pooled into one window: one row for each observation and each
  time step t >= max-lag, with the N series at steps t - max-lag, ..., t - 1
  and at step t.
- The time step t of the row is the context of CD-NOD.
- The CI test is the Fisher-z test of the library.
- Background knowledge: the lagged columns come before the columns of step t;
  no edge between two columns of step t; no edge between two lagged columns;
  the lag-1 self edge of every series is required.
- A lagged series is a parent of a series at step t when the two columns are
  adjacent in the returned graph (the orientation is not used). This window
  graph is repeated at every time step.
"""
import itertools
import time
import warnings

import andrey
import numpy as np

import _common

RELEASE = "0.17.1"


def pooled(X, L):
    """Return window data (M (T - L), (L + 1) N) and the step of each row.

    Column blocks of N series: lag L, ..., lag 1, step t."""
    M, T, N = X.shape
    blocks = [X[:, L - l:T - l, :] for l in range(L, -1, -1)]
    data = np.concatenate(blocks, axis=2).reshape(M * (T - L), (L + 1) * N)
    steps = np.broadcast_to(np.arange(L, T, dtype=np.float64)[None, :], (M, T - L)).reshape(-1)
    return np.ascontiguousarray(data), np.ascontiguousarray(steps)


def knowledge(L, N):
    """Return the background knowledge for L lags of N series."""
    lagged = list(range(L * N))
    current = list(range(L * N, (L + 1) * N))
    forbidden = list(itertools.permutations(current, 2)) + list(itertools.permutations(lagged, 2))
    required = [((L - 1) * N + j, L * N + j) for j in range(N)]        # the lag-1 self edges
    return andrey.Knowledge(tiers=[lagged, current], forbidden=forbidden, required=required)


def main():
    """Read input, run CD-NOD on pooled window data, and write the time graph.

    Raise `RuntimeError` when the installed `andrey-core` is not release
    `RELEASE`."""
    p = _common.parser(__doc__.split("\n")[0])
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--max-lag", type=int, default=1)
    args = p.parse_args()
    if andrey.__version__ != RELEASE:
        raise RuntimeError(f"andrey-core {andrey.__version__} is installed; this script needs {RELEASE}")
    X = _common.read_input(args.input)
    M, T, N = X.shape
    L = args.max_lag
    start = time.perf_counter()
    data, steps = pooled(X, L)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = andrey.cdnod(data, steps, alpha=args.alpha, indep_test="fisherz", knowledge=knowledge(L, N))
    seconds = time.perf_counter() - start
    marks = np.asarray(out.structure.adjacency)
    adjacent = (marks != 0) | (marks != 0).T
    win = np.zeros((L + 1, N, N), dtype=bool)
    for l in range(1, L + 1):
        win[l] = adjacent[(L - l) * N:(L - l + 1) * N, L * N:]        # [series at lag l, series at step t]
    _common.write_output(args.output, _common.window_to_time_graph(win, T), seconds,
                         ["andrey-core", "numpy", "scipy"])


if __name__ == "__main__":
    main()
