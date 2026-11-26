"""Recover a random temporal DAG exactly with the d-separation oracle.

The graph has three series and five time steps. Every lag-1 self edge is
present, and every other forward edge is present with probability 0.12. ITPD
gets the d-separation relations of the graph and no data.

Run:

    python examples/oracle_exact_recovery.py [--seed 7] [--output graph.png]

The PNG needs Matplotlib; without `--output` no figure is written.
"""
import argparse

import numpy as np

from itpd import ITPD


def run_example(*, seed=7, output=None) -> ITPD:
    """Draw the graph, fit ITPD with the oracle, and check exact recovery."""
    N, T = 3, 5
    rng = np.random.default_rng(seed)
    times = np.repeat(np.arange(T), N)
    adjacency = ((rng.random((N * T, N * T)) < 0.12)
                 & (times[:, None] < times[None, :])).astype(np.uint8)
    sources = np.arange(N * (T - 1))
    adjacency[sources, sources + N] = 1

    model = ITPD(oracle_adjacency=adjacency).fit(n_variables=N)
    learned = model.get_adjacency_matrix()
    if not np.array_equal(learned, adjacency):
        missing = np.argwhere((adjacency == 1) & (learned == 0)).tolist()
        extra = np.argwhere((adjacency == 0) & (learned == 1)).tolist()
        raise AssertionError(f"Oracle recovery failed: missing={missing}, extra={extra}")
    print(f"Exact recovery: {N * T} nodes, {int(adjacency.sum())} edges (seed={seed}).")
    print(f"CI tests: Phase I={model.ci_queries_['I']}, "
          f"Phase II={model.ci_queries_['II']}; "
          f"largest conditioning set={model.max_conditioning_size_}.")
    if output is not None:
        print(f"Learned graph saved to {model.plot_graph(output)}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=7, help="random seed (default: 7)")
    parser.add_argument("--output", default=None, help="path of a PNG of the learned graph")
    args = parser.parse_args()
    run_example(seed=args.seed, output=args.output)
