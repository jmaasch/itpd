"""CI tests (oracle, Fisher-z, GCM) and the instrumented recorder every method calls through.

A CI test is a callable `ci(x, y, S) -> p-value` on global column indices (oracle: 1.0 if d-separated, else 0.0).
`Recorder` wraps one of them and is the only place where tests are counted:

- raw_calls: every call that reaches the recorder (the algorithms memoize marginals inside one PaDL call themselves).
- unique_tests: distinct (unordered pair, sorted conditioning set); the headline count.
- evaluated: calls that actually ran the underlying test, given the cache scope
  ("none": every call, "pair": cache cleared by `new_scope()` per target, "run": one cache for the whole run).
- by size and by rule label, both raw and unique (a unique test is attributed to the label that first issued it).
- cache hits, seconds spent inside the test, wall-clock since creation, number of infeasible tests (`p_nan`).
- `shared` (optional dict): p-values (NaN for infeasible) shared between recorders that run on the SAME data, so that
  many methods and many alpha levels on one dataset compute each distinct test once. The p-value of a test does not
  depend on alpha or on the method, so counts and decisions are unchanged; `evaluated` and `seconds_ci` then count
  only the tests this recorder really computed (the others are `shared_hits`).
Infeasible tests (the test returns NaN) never silently count as "independent": by default (`infeasible="raise"`)
the recorder raises `InfeasibleTest`; with `infeasible="flag"` it records the test, counts it in `p_nan`, keeps the
first offending key and returns p = 1 so the run can finish for diagnostics; the run summary then has
`infeasible = True` and `method_runner` reports the method-cell as infeasible instead of scoring it.
"""
from __future__ import annotations

import hashlib
import os
import time
from collections import defaultdict

import numpy as np
from scipy.special import erfc, log_ndtr


# ---------------------------------------------------------------------------------------------- CI tests
# Section 1: InfeasibleTest, the d-separation oracle `DSepCI`, `FisherZ`, `GCM` and the `MappedCI` wrapper.
# Section 2 (end of file): `Recorder`, the counting wrapper every method calls through.


class InfeasibleTest(RuntimeError):
    """A CI test that cannot be evaluated (Fisher-z with n - |S| - 3 <= 0, too few samples for the regression)."""


def canonical_key(x: int, y: int, S) -> tuple:
    a, b = (x, y) if x < y else (y, x)
    return (a, b, tuple(sorted({int(s) for s in S} - {a, b})))


def compact_key(key: tuple) -> tuple:
    """Memory-light stand-in for a canonical key: (x, y, |S|, 12-byte digest of S). Used for the cache and the unique
    set, so a run with millions of tests at |S| ~ 1000 does not store the sets (collision probability ~ 2^-96)."""
    S = key[2]
    return (key[0], key[1], len(S), hashlib.blake2b(np.asarray(S, dtype=np.int32).tobytes(), digest_size=12).digest())


def oracle_mode() -> str:
    """Which d-separation oracle `DSepCI` uses, set by the environment variable ITPD_ORACLE.

    "fast" (the default) uses the optional `fastdsep` C extension when it is installed and the pure-Python
    `itpd.graphs.DSep` otherwise. "old" always uses `itpd.graphs.DSep`. Both give the same answers."""
    mode = os.environ.get("ITPD_ORACLE", "fast").strip().lower()
    if mode not in ("fast", "old"):
        raise ValueError("ITPD_ORACLE must be fast|old")
    return mode


class DSepCI:
    """Oracle CI test on the full graph: p = 1.0 if d-separated else 0.0.

    With ITPD_ORACLE unset or "fast" the oracle is `fastdsep.FastDSep` when the optional `fastdsep` package can be
    imported; it is 30 to 170 times faster than `graphs.DSep` for a non-empty S. Otherwise, and with ITPD_ORACLE=old,
    the oracle is the pure-Python `graphs.DSep`. `self.backend` names the one in use: "fastdsep-c" or "fastdsep-python"
    (the backend that fastdsep reports) or "python". The answers are identical (tests/test_graphs_ci.py)."""

    def __init__(self, A: np.ndarray):
        FastDSep = None
        if oracle_mode() != "old":
            try:
                from fastdsep import FastDSep
            except ImportError:
                pass
        if FastDSep is not None:
            self.d, self.backend = FastDSep(A), "fastdsep-" + FastDSep.backend
        else:
            from .graphs import DSep
            self.d, self.backend = DSep(A), "python"

    def __call__(self, x: int, y: int, S) -> float:
        return 1.0 if self.d.dsep(x, y, S) else 0.0


class FisherZ:
    """Fisher-z partial-correlation test from a precomputed correlation matrix of `data` (n, D).

    Same statistic as causal-learn's `fisherz` (|r| clipped below 1, dof = n - |S| - 3).
    Returns NaN when dof <= 0 (infeasible). The recorder then raises `InfeasibleTest` by default; only with
    `infeasible="flag"` does it count the test in `p_nan` and return p = 1 for diagnostics (see the module docstring).
    """

    def __init__(self, data: np.ndarray):
        data = np.asarray(data, dtype=float)
        self.n = data.shape[0]
        self.corr = np.corrcoef(data.T)

    def _stat(self, x: int, y: int, S) -> float | None:
        """|z| * sqrt(dof) of the partial correlation of (x, y | S), or None when the test is infeasible (dof <= 0)."""
        S = [int(s) for s in S if s != x and s != y]
        dof = self.n - len(S) - 3
        if dof <= 0:
            return None
        C = self.corr
        xy = [x, y]
        P = C[np.ix_(xy, xy)]
        if S:
            Css = C[np.ix_(S, S)]
            Csx = C[np.ix_(S, xy)]
            try:
                K = np.linalg.solve(Css, Csx)
            except np.linalg.LinAlgError:
                K = np.linalg.lstsq(Css, Csx, rcond=None)[0]
            P = P - Csx.T @ K
        d = abs(P[0, 0] * P[1, 1])
        if d <= 0:
            return 0.0
        r = P[0, 1] / np.sqrt(d)
        if abs(r) >= 1.0:
            r = (1.0 - np.finfo(float).eps) * np.sign(r)
        z = 0.5 * np.log((1.0 + r) / (1.0 - r))
        return np.sqrt(dof) * abs(z)

    def __call__(self, x: int, y: int, S) -> float:
        stat = self._stat(x, y, S)
        if stat is None:
            return float("nan")
        return float(erfc(stat / np.sqrt(2.0)))

    def neglogp(self, x: int, y: int, S) -> float:
        """-log p of the same test, without underflow at large n. Not a new CI test: it ranks candidates whose
        p-values are equal (ties in IAMB). NaN when the test is infeasible."""
        stat = self._stat(x, y, S)
        if stat is None:
            return float("nan")
        return float(-(np.log(2.0) + log_ndtr(-stat)))


class GCM:
    """Generalized covariance measure (Shah and Peters 2020) with a gradient-boosting regressor.

    Test of x _||_ y | S from n i.i.d. rows. Statistic:
        r^x_i = x_i - f_hat(z_i),  r^y_i = y_i - g_hat(z_i),  R_i = r^x_i * r^y_i,
        T_n = sqrt(n) * mean(R) / sqrt(mean(R^2) - mean(R)^2),   p = 2 (1 - Phi(|T_n|)) = erfc(|T_n| / sqrt 2),
    asymptotically N(0, 1) under x _||_ y | S when both regressions converge fast enough (product of the two rates
    o(n^-1/2)). f_hat and g_hat: sklearn HistGradientBoostingRegressor (`max_iter` = 60 trees of default depth
    (31 leaves), learning rate 0.1, min_samples_leaf 20, squared loss, fixed random_state), 2-fold cross-fitting with
    one fixed random split of the rows (each residual is predicted by a model fit on the other half; no row is
    predicted by a model that saw it). S empty: no regression, the residuals are the centered variables, so the test
    is the Pearson correlation test (it has no power against a dependence that is uncorrelated, for example an even
    function of a symmetric variable). The statistic is compared with the standard normal (no t correction).
    Feasibility: NaN (infeasible) only when n < 8; there is no hard condition like Fisher-z's n > |S| + 3.
    Bookkeeping for the cost per test: `calls_by_size[|S|]`, `sec_by_size[|S|]` count the evaluations that ran the
    test (a memo hit upstream is not an evaluation), `fits` the number of boosted-tree fits.
    """

    def __init__(self, data: np.ndarray, max_iter: int = 60, seed: int = 0):
        self.data = np.asarray(data, dtype=float)
        self.n = self.data.shape[0]
        self.max_iter = max_iter
        self.seed = seed
        self.fits = 0
        self.calls_by_size: dict = {}
        self.sec_by_size: dict = {}
        self._fold = np.random.default_rng(seed).permutation(self.n) % 2

    def _resid(self, target: np.ndarray, Z: np.ndarray | None) -> np.ndarray:
        if Z is None or Z.shape[1] == 0:
            return target - target.mean()
        from sklearn.ensemble import HistGradientBoostingRegressor
        res = np.empty(len(target))
        for f in (0, 1):
            tr, te = self._fold != f, self._fold == f
            m = HistGradientBoostingRegressor(max_iter=self.max_iter, random_state=self.seed)
            m.fit(Z[tr], target[tr])
            self.fits += 1
            res[te] = target[te] - m.predict(Z[te])
        return res

    def __call__(self, x: int, y: int, S) -> float:
        S = [int(s) for s in S if s != x and s != y]
        if self.n < 8:
            return float("nan")
        t0 = time.perf_counter()
        Z = self.data[:, S] if S else None
        R = self._resid(self.data[:, x], Z) * self._resid(self.data[:, y], Z)
        m, v = R.mean(), (R ** 2).mean() - R.mean() ** 2
        k = len(S)
        self.calls_by_size[k] = self.calls_by_size.get(k, 0) + 1
        self.sec_by_size[k] = self.sec_by_size.get(k, 0.0) + time.perf_counter() - t0
        if v <= 1e-300:
            return 1.0
        stat = np.sqrt(self.n) * m / np.sqrt(v)
        return float(erfc(abs(stat) / np.sqrt(2.0)))


class MappedCI:
    """Wrap a CI test defined on other column indices (S1 windows: local column -> global column)."""

    def __init__(self, ci, cols):
        self.ci, self.cols = ci, list(cols)

    def __call__(self, x, y, S):
        return self.ci(self.cols[x], self.cols[y], [self.cols[s] for s in S])


# ---------------------------------------------------------------------------------------------- counting


class Recorder:
    def __init__(self, ci, cache: str = "run", keep_calls: bool = False, infeasible: str = "raise",
                 shared: dict | None = None):
        if cache not in ("none", "pair", "run"):
            raise ValueError("cache must be none|pair|run")
        if infeasible not in ("raise", "flag"):
            raise ValueError("infeasible must be raise|flag")
        self.ci, self.cache_scope, self.keep_calls, self.infeasible = ci, cache, keep_calls, infeasible
        self.shared = shared
        self.shared_hits = 0
        self.first_infeasible = None
        self.last_infeasible = None
        self._nan_seen: set = set()
        self._cache: dict = {}
        self._seen: dict = {}          # key -> label of first issuer (global uniqueness)
        self.raw = 0
        self.evaluated = 0
        self.hits = 0
        self.p_nan = 0
        self.sec_ci = 0.0
        self.t0 = time.perf_counter()
        self.raw_size = defaultdict(int)
        self.uniq_size = defaultdict(int)
        self.raw_label = defaultdict(int)
        self.uniq_label = defaultdict(int)
        self.uniq_label_size = defaultdict(lambda: defaultdict(int))
        self.calls: list = []
        self.cur_max = 0               # largest conditioning set issued since the last new_scope()

    def new_scope(self) -> None:
        self.cur_max = 0
        if self.cache_scope == "pair":
            self._cache.clear()

    def mark(self) -> tuple[int, int]:
        return self.raw, len(self._seen)

    def __call__(self, x: int, y: int, S, label: str = "") -> float:
        key = canonical_key(x, y, S)
        size = len(key[2])
        ck = compact_key(key) if size > 8 else key
        self.raw += 1
        if size > self.cur_max:
            self.cur_max = size
        self.raw_size[size] += 1
        self.raw_label[label] += 1
        if ck not in self._seen:
            self._seen[ck] = label
            self.uniq_size[size] += 1
            self.uniq_label[label] += 1
            self.uniq_label_size[label][size] += 1
        if self.cache_scope != "none" and ck in self._cache:
            self.hits += 1
            p = self._cache[ck]
        else:
            p = self.shared.get(ck) if self.shared is not None else None
            if p is not None:
                self.shared_hits += 1
            else:
                t = time.perf_counter()
                p = self.ci(key[0], key[1], key[2])
                self.sec_ci += time.perf_counter() - t
                self.evaluated += 1
                if self.shared is not None:
                    self.shared[ck] = p
            if p != p:
                self.p_nan += 1
                self._nan_seen.add(ck)
                self.last_infeasible = key
                if self.first_infeasible is None:
                    self.first_infeasible = key
                if self.infeasible == "raise":
                    raise InfeasibleTest(f"test {key[:2]} given {len(key[2])} variables is infeasible")
                p = 1.0
            if self.cache_scope != "none":
                self._cache[ck] = p
        if self.keep_calls:
            self.calls.append((key, label, p))
        return p

    def summary(self) -> dict:
        def srt(d):
            return {int(k): int(v) for k, v in sorted(d.items())}
        return {
            "raw_calls": self.raw,
            "unique_tests": len(self._seen),
            "evaluated": self.evaluated,
            "shared_hits": self.shared_hits,
            "cache_scope": self.cache_scope,
            "cache_hits": self.hits,
            "p_nan": self.p_nan,
            "p_nan_unique": len(self._nan_seen),
            "infeasible": self.p_nan > 0,
            "first_infeasible_size": None if self.first_infeasible is None else len(self.first_infeasible[2]),
            "by_size_raw": srt(self.raw_size),
            "by_size_unique": srt(self.uniq_size),
            "by_label_raw": {k: int(v) for k, v in sorted(self.raw_label.items())},
            "by_label_unique": {k: int(v) for k, v in sorted(self.uniq_label.items())},
            "by_label_size_unique": {k: srt(v) for k, v in sorted(self.uniq_label_size.items())},
            "seconds_ci": self.sec_ci,
            "seconds_wall": time.perf_counter() - self.t0,
        }
