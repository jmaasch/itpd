"""Check the helpers for time graphs."""
import numpy as np
import pytest

from itpd.graphs import check_temporal, graph_hash, node, self_edge_mask, unroll


def _window():
    """Two series, lag 2: 0 -> 0, 0 -> 1, 1 -> 1 at lag 1 and 1 -> 0 at lag 2."""
    B = np.zeros((3, 2, 2), int)
    B[1] = [[1, 1], [0, 1]]
    B[2] = [[0, 0], [1, 0]]
    return B


def test_node():
    assert node(0, 0, 20) == 0 and node(3, 4, 20) == 64


def test_unroll():
    A = unroll(_window(), 4)
    edges = {(int(u), int(v)) for u, v in np.argwhere(A)}
    lag1 = {(t * 2 + i, (t + 1) * 2 + j) for t in range(3) for i, j in [(0, 0), (0, 1), (1, 1)]}
    lag2 = {(t * 2 + 1, (t + 2) * 2) for t in range(2)}
    assert A.shape == (8, 8) and A.dtype == bool and edges == lag1 | lag2
    check_temporal(A, 2, 4)


def test_self_edge_mask():
    mask = self_edge_mask(3, 4)
    assert mask.sum() == 9 and all(mask[u, u + 3] for u in range(9))


def test_graph_hash():
    A = unroll(_window(), 4)
    assert graph_hash(A) == "c5f53652201ae1063d809f3d15fb3f15186b4a4a"
    assert graph_hash(A.astype(float) * 0.5) == graph_hash(A)          # Only the nonzero pattern counts.
    assert graph_hash(np.zeros((2, 8))) != graph_hash(np.zeros((4, 4)))


def test_check_temporal_rejects_other_graphs():
    A = unroll(_window(), 4)
    for u, v, value in [(2, 2, 1), (3, 2, 1), (5, 0, 1), (0, 2, 0)]:   # Two edges inside a step, a backward edge, a missing self edge.
        bad = A.copy()
        bad[u, v] = value
        with pytest.raises(ValueError):
            check_temporal(bad, 2, 4)
    with pytest.raises(ValueError):
        check_temporal(A, 2, 3)
