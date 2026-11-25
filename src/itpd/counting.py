"""Count the CI tests that a method requests.

`CICounter` wraps a CI test. Every method calls the test through it, so all
methods are counted in the same way:
- a requested test is one call of the counter;
- a unique test is a distinct (unordered pair, conditioning set).
"""
import hashlib

import numpy as np

from .ci_tests import NotEvaluable


def n_candidates(N: int, T: int) -> int:
    """Return the number of (candidate, target) pairs of N series and T steps.

    A target at step t = 1, ..., T - 1 has t * N - 1 candidates: the earlier
    nodes without its lag-1 self parent."""
    return sum(N * (t * N - 1) for t in range(1, T))


class CICounter:
    """Call a CI test, count the calls, and keep the p-values.

    The test is called as `test(a, b, S)` with a < b and S a sorted tuple
    without a and b. The p-value of a unique test is kept, and a repeated
    request returns the kept value. A conditioning set with more than 8 members
    is told apart by its size and a 12-byte BLAKE2b digest of its members. A
    test that cannot be evaluated (NaN) counts as requested and as unique, is
    not kept, and raises `NotEvaluable`.

    Attributes
    ----------
    test : callable
        The wrapped CI test.
    requested : int
        Number of calls.
    unique : int
        Number of distinct tests.
    by_size : dict
        Conditioning-set size -> number of unique tests.
    not_evaluable : int
        Number of unique tests that could not be evaluated.
    """

    def __init__(self, test):
        self.test = test
        self.requested = 0
        self.by_size = {}
        self._seen = set()
        self._pvalues = {}
        self._not_evaluable = set()

    @property
    def unique(self) -> int:
        return len(self._seen)

    @property
    def not_evaluable(self) -> int:
        return len(self._not_evaluable)

    def __call__(self, x: int, y: int, S=()) -> float:
        """Count one request and return the p-value of x and y given S.

        Return the kept p-value on a repeated request without calling `test`.
        Raise `NotEvaluable` when `test` returns NaN."""
        a, b = (x, y) if x < y else (y, x)
        S = tuple(sorted({int(s) for s in S} - {a, b}))
        size = len(S)
        key = (a, b, S)
        if size > 8:
            # Save memory: keep a 12-byte digest of a large set, not the set.
            digest = hashlib.blake2b(np.asarray(S, dtype=np.int32).tobytes(),
                                     digest_size=12).digest()
            key = (a, b, size, digest)
        self.requested += 1
        if key not in self._seen:
            self._seen.add(key)
            self.by_size[size] = self.by_size.get(size, 0) + 1
        if key in self._pvalues:
            return self._pvalues[key]
        p = self.test(a, b, S)
        if p != p:
            self._not_evaluable.add(key)
            raise NotEvaluable(f"The test of ({a}, {b}) given {size} variables "
                               "cannot be evaluated")
        self._pvalues[key] = p
        return p

    def summary(self, N: int, T: int) -> dict:
        """Return the counts of a run on N series and T steps.

        `largest_set` and `mean_set` are over the unique tests. Both are `None`
        when there is no test."""
        unique = self.unique
        sizes = [size for size, count in self.by_size.items() if count]
        return {
            "requested": self.requested,
            "unique": unique,
            "largest_set": max(sizes) if sizes else None,
            "mean_set": (sum(size * count for size, count in self.by_size.items()) / unique
                         if unique else None),
            "unique_per_candidate": unique / n_candidates(N, T),
            "not_evaluable": self.not_evaluable,
        }
