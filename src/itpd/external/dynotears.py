"""DYNOTEARS (causalnex) on data of shape (M, T, N).

    python dynotears.py --input in.npz --output out.npz --max-lag 1 [--unit-variance]

The M observations are given as M data frames. The penalties are
lambda_w = lambda_a = 0.1 and the weight threshold is 0.05; nothing is
tuned. The fitted model is one window graph with lags 0, ..., max-lag. The
edges with lag 1 or more are repeated at every time step; the edges inside
a time step are dropped. --unit-variance divides every series by the root of
its mean square over all observations and time steps.
"""
import time

import numpy as np
import pandas as pd
from causalnex.structure.dynotears import from_pandas_dynamic

import _common


def main():
    p = _common.parser(__doc__.split("\n")[0])
    p.add_argument("--max-lag", type=int, default=1)
    p.add_argument("--unit-variance", action="store_true")
    args = p.parse_args()
    X = _common.read_input(args.input)
    if args.unit_variance:
        X = np.ascontiguousarray(X / np.sqrt((X ** 2).mean(axis=(0, 1))))
    M, T, N = X.shape
    K = args.max_lag
    start = time.perf_counter()
    columns = [f"v{i}" for i in range(N)]
    frames = [pd.DataFrame(X[m], columns=columns) for m in range(M)]
    model = from_pandas_dynamic(frames, p=K, lambda_w=0.1, lambda_a=0.1, w_threshold=0.05)
    seconds = time.perf_counter() - start
    win = np.zeros((K + 1, N, N), dtype=bool)
    for source, target in model.edges:        # node names: "v3_lag1" -> "v5_lag0"
        (i, lag), (j, target_lag) = source[1:].split("_lag"), target[1:].split("_lag")
        if int(target_lag) == 0 and int(lag) >= 1:
            win[int(lag), int(i), int(j)] = True
    _common.write_output(args.output, _common.window_to_time_graph(win, T), seconds,
                         ["causalnex", "numpy", "pandas", "scipy"])


if __name__ == "__main__":
    main()
