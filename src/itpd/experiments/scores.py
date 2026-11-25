"""Scores for time graphs that break an assumption of ITPD.

The instances of itpd.simulation.violations can have edges inside a time
step. Every method here returns edges from earlier steps only, so two
truths are scored:

lag1_edges        the true edges between two different steps;
markov_boundary   for every target, the earlier variables that stay
                  dependent on it given all other earlier variables: the
                  earlier-step parents of the target and of each same-step
                  ancestor of the target.

Both are views of the table of means (itpd.experiments.aggregate). They
return the dict of itpd.metrics.score, so the lag-1 self edges are not
scored and an untested target has no parents.
"""
import numpy as np

from .. import metrics


def _steps(A, N):
    """Return A as a Boolean matrix, and the Boolean matrices "u is at an earlier step than v" and "at the same step"."""
    A = np.asarray(A) != 0
    step = np.arange(len(A)) // N
    return A, step[:, None] < step[None, :], step[:, None] == step[None, :]


def lag1_edges(truth, A_hat, M=None, untested=()) -> dict:
    """Score the estimate A_hat against the edges of truth.A between two different steps.

    truth is an itpd.simulation.Instance. M sets the common targets (None:
    every target). untested lists the untested targets."""
    A, earlier, _ = _steps(truth.A, truth.N)
    return metrics.score(A & earlier, (np.asarray(A_hat) != 0) & earlier, truth.N, truth.T, M, untested)


def markov_boundary_among_earlier(A, N: int) -> np.ndarray:
    """Return the Boolean matrix P with P[u, v] iff u is an earlier-step parent of v or of a same-step ancestor of v."""
    A, earlier, same = _steps(A, N)
    reach = (A & same) | np.eye(len(A), dtype=bool)      # reach[w, v]: w is v or a same-step ancestor of v
    while True:
        longer = (reach.astype(np.int32) @ reach.astype(np.int32)) > 0
        if (longer == reach).all():
            break
        reach = longer
    return ((A & earlier).astype(np.int32) @ reach.astype(np.int32)) > 0


def markov_boundary(truth, A_hat, M=None, untested=()) -> dict:
    """Score the estimate A_hat against the Markov boundary of every target among the earlier variables.

    The arguments are those of lag1_edges."""
    _, earlier, _ = _steps(truth.A, truth.N)
    P = markov_boundary_among_earlier(truth.A, truth.N)
    return metrics.score(P, (np.asarray(A_hat) != 0) & earlier, truth.N, truth.T, M, untested)
