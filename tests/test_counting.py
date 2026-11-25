"""Check the counter of CI tests."""
import math

import pytest

from itpd import CICounter, NotEvaluable
from itpd.counting import n_candidates


class _Test:
    """A CI test that keeps its calls. It cannot be evaluated when |S| >= limit."""

    def __init__(self, limit=math.inf):
        self.calls, self.limit = [], limit

    def __call__(self, x, y, S):
        self.calls.append((x, y, S))
        return float("nan") if len(S) >= self.limit else 1 / (1 + x + y + len(S))

    def neglogp(self, x, y, S):
        return 0.0


def test_requested_and_unique_tests():
    test = _Test()
    counter = CICounter(test)
    p = counter(3, 1, [5, 4])
    # The key ignores the order of the pair and of S, repeats in S, and the pair in S.
    assert counter(1, 3, [4, 5]) == counter(3, 1, (5, 4, 4)) == counter(1, 3, [1, 3, 4, 5]) == p
    assert counter(1, 3) == 0.2 and counter(1, 3, [4]) != p
    assert (counter.requested, counter.unique) == (6, 3)
    assert counter.by_size == {2: 1, 0: 1, 1: 1}
    # The test is called once per unique test: pair in increasing order, S a sorted tuple.
    assert test.calls == [(1, 3, (4, 5)), (1, 3, ()), (1, 3, (4,))]
    assert counter.test is test and counter.test.neglogp(1, 3, ()) == 0.0


def test_large_sets_are_told_apart():
    counter = CICounter(_Test())
    counter(0, 1, range(2, 12))
    counter(1, 0, reversed(range(2, 12)))
    counter(0, 1, range(3, 13))
    assert (counter.requested, counter.unique, counter.by_size) == (3, 2, {10: 2})


def test_a_test_that_cannot_be_evaluated():
    test = _Test(limit=3)
    counter = CICounter(test)
    counter(0, 1, [2])
    for _ in range(2):
        with pytest.raises(NotEvaluable):
            counter(0, 1, [2, 3, 4])
    # It counts as requested and as unique, sets the largest set, and is not kept.
    assert (counter.requested, counter.unique, counter.not_evaluable) == (3, 2, 1)
    assert len(test.calls) == 3
    summary = counter.summary(2, 3)
    assert summary == {"requested": 3, "unique": 2, "largest_set": 3, "mean_set": 2.0,
                       "unique_per_candidate": 2 / 8, "not_evaluable": 1}


def test_summary_without_tests():
    summary = CICounter(_Test()).summary(20, 16)
    assert summary["requested"] == summary["unique"] == summary["not_evaluable"] == 0
    assert summary["largest_set"] is None and summary["mean_set"] is None


def test_n_candidates():
    assert n_candidates(20, 16) == 47700
    assert n_candidates(2, 3) == 2 * 1 + 2 * 3
    assert n_candidates(1, 2) == 0
