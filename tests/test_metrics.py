"""Check the scores in `metrics.py`."""
import math

import numpy as np

from itpd import metrics


def _two_targets():
    """Return a graph with N = 2, T = 2.

    The true parents of node 2 are {0, 1}, and those of node 3 are {1, 0}."""
    A = np.zeros((4, 4), int)
    A[[0, 1, 0, 1], [2, 2, 3, 3]] = 1
    return A


def test_self_edges_are_not_scored():
    A = _two_targets()
    no_self = A.copy()
    no_self[0, 2] = no_self[1, 3] = 0
    assert metrics.edge_counts(A, A, 2, 2) == {"tp": 2, "fp": 0, "fn": 0}
    assert metrics.edge_counts(A, no_self, 2, 2) == {"tp": 2, "fp": 0, "fn": 0}
    assert metrics.edge_counts(no_self, A, 2, 2) == {"tp": 2, "fp": 0, "fn": 0}


def test_an_untested_target_has_no_parents():
    A = _two_targets()
    assert metrics.edge_counts(A, A, 2, 2, untested=[3]) == {"tp": 1, "fp": 0, "fn": 1}
    result = metrics.score(A, A, 2, 2, untested=[3])
    assert result["all"] == {"tp": 1, "fp": 0, "fn": 1, "precision": 1.0, "recall": 0.5, "f1": 2 / 3}
    assert result["n_untested"] == 1
    # A node of the first step is not a target.
    assert metrics.score(A, A, 2, 2, untested=[0, 3])["n_untested"] == 1


def test_false_edges():
    A = _two_targets()
    A_hat = np.zeros((4, 4), int)
    A_hat[0, 2] = A_hat[1, 3] = 1          # Self edges only: both other edges are missed.
    A_hat[3, 2] = 1                        # An edge inside a time step is a false positive.
    assert metrics.edge_counts(A, A_hat, 2, 2) == {"tp": 0, "fp": 1, "fn": 2}


def test_last_common_step():
    assert metrics.last_common_step(20, 16, 200) == 9
    assert metrics.last_common_step(20, 16, 500) == 15
    assert metrics.last_common_step(20, 16, 22) == 0
    assert metrics.last_common_step(20, 16, 23) == 1
    assert metrics.last_common_step(20, 16) == 15


def test_score_on_common_targets():
    N, T = 2, 4
    A = np.zeros((8, 8), int)
    A[np.arange(6), np.arange(2, 8)] = 1
    A[0, 3] = A[2, 5] = A[1, 6] = 1        # One edge into each of the steps 1, 2, 3.
    A_hat = A.copy()
    A_hat[1, 6] = 0
    A_hat[0, 7] = 1
    result = metrics.score(A, A_hat, N, T, M=7)             # (7 - 3) // 2 = 2
    assert result["last_common_step"] == 2
    assert result["common"] == {"tp": 2, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert result["all"] == {"tp": 2, "fp": 1, "fn": 1, "precision": 2 / 3, "recall": 2 / 3, "f1": 2 / 3}
    assert metrics.score(A, A_hat, N, T)["common"] == result["all"]     # Without M every target is common.
    few = metrics.score(A, A_hat, N, T, M=4)
    assert few["last_common_step"] == 0 and few["common"] is None and few["all"] == result["all"]


def test_f1_precision_recall():
    assert metrics.f1(554, 7, 69) == 1108 / 1184 == 0.9358108108108109
    assert metrics.f1(0, 3, 0) == 0.0
    assert math.isnan(metrics.f1(0, 0, 0))
    assert math.isnan(metrics.precision(0, 0)) and math.isnan(metrics.recall(0, 0))
    assert metrics.precision(3, 1) == 0.75 and metrics.recall(3, 9) == 0.25
