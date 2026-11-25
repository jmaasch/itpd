"""Fit ITPD on simulated data and score the estimate.

The data are 500 observations of 5 series over 8 time steps from a
nonstationary linear-Gaussian time graph: the edges and their weights are drawn
at every step, with 2 parents per node on average besides the lag-1 self
parent. ITPD runs with the Fisher-z test at `alpha` = 0.01.

Run:

    python examples/fit_simulated_data.py [--graph 0] [--M 500]
"""
import argparse

from itpd import ITPD, CICounter, FisherZ
from itpd.metrics import score
from itpd.simulation import linear


def run_example(*, graph=0, M=500) -> ITPD:
    """Draw the graph and data, fit ITPD, and print scores and CI test counts."""
    N, T = 5, 8
    instance = linear.make_instance(N, T, 1, 2, graph, stationary=False)
    V = instance.sample(M)                  # shape (M, T, N)
    counter = CICounter(FisherZ(V))         # counts the CI tests that ITPD requests
    model = ITPD(counter, alpha=0.01).fit(V)

    result = score(instance.A, model.get_adjacency_matrix(), N, T, M, model.untested_targets_)["all"]
    counts = counter.summary(N, T)
    print(f"Data: {M} observations of {N} series over {T} time steps (graph {graph}).")
    print(f"Edges (lag-1 self edges not scored): {result['tp']} true, {result['fp']} false, {result['fn']} missed; "
          f"F1 = {result['f1']:.3f}.")
    print(f"CI tests: {counts['requested']} requested, {counts['unique']} unique "
          f"({counts['unique_per_candidate']:.2f} per candidate); largest conditioning set = {counts['largest_set']}.")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--graph", type=int, default=0, help="index of the random graph (default: 0)")
    parser.add_argument("--M", type=int, default=500, help="number of observations (default: 500)")
    args = parser.parse_args()
    run_example(graph=args.graph, M=args.M)
