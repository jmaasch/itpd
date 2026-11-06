"""Known-order lasso per target on the unrolled time graph (S2), finite data only (no CI tests, no oracle).

For each target Y = V^n_t (t >= 1), X = V^n_{t-1} (self edge known, as in every other method) and candidates C = every strictly
earlier node except X:
  1. centre the columns of the (M, T N) data; project X out of Y and of every candidate (X enters the model unpenalised, the
     lasso analogue of conditioning on the known self edge);
  2. standardise the projected candidates and the projected Y to unit variance (the penalty is then on a correlation scale);
  3. L1 regression of Y on the candidates (sklearn, objective (1 / 2M) ||y - Z w||^2 + lambda ||w||_1), parents = nonzero w.
Two selection rules, both reported:
  `lasso_cv`    sklearn LassoCV (5 folds, no shuffle, 100 values of lambda on sklearn's automatic grid per target, the lambda
                with the smallest mean squared error); the cross-validated lasso is a prediction-optimal choice and selects
                more variables than a support-consistent one, so its precision is the lasso's weak spot;
  `lasso_path`  the support at each of 13 fixed absolute lambdas LAMBDAS (log-spaced from 0.6 to 0.003 on the standardised
                scale; lambda >= 1 selects nothing), one point of a recall-false-positive curve each, for the oracle-tuned
                matched-FP comparison with the CI methods' alpha sweep.
Two further rules (the cross-validated lasso over-selects; the headline lasso row is `lasso_ebic`):
  `lasso_ebic`  extended BIC (Chen and Chen 2008, gamma = 1) along sklearn's automatic 100-value lambda path of each target:
                EBIC = M log(RSS / M) + df log M + 2 log C(p, df), RSS of the lasso fit, df = number of nonzero coefficients
                (Zou, Hastie, Tibshirani 2007), p = number of candidates; lambdas with df > M / 2 are not considered. The plain
                profile BIC (no binomial term) saturates when M <= p (RSS near zero), and sklearn's LassoLarsIC needs a noise-variance
                estimate that does not exist for M <= p, so the extended form is used;
  `lasso_fixed` one fixed penalty without tuning, lambda = sqrt(2 log p / M) on the standardised scale (noise standard deviation
                taken as 1, which is conservative because the standardised, X-projected response has residual variance below 1).
No test is ever infeasible, so no target is undecidable. The numbers of tests are zero by construction and are not reported;
wall-clock (perf_counter and process CPU time) is.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np

LAMBDAS = tuple(float(v) for v in np.logspace(np.log10(0.6), np.log10(0.003), 13))


@dataclass
class LassoResult:
    A_cv: np.ndarray                                   # (T N, T N) selected edges incl. the assumed self edges
    A_path: list = field(default_factory=list)         # one (T N, T N) per lambda in `lambdas`
    lambdas: tuple = LAMBDAS
    seconds_cv: float = 0.0
    seconds_path: float = 0.0
    cpu_cv: float = 0.0
    cpu_path: float = 0.0
    n_cv_alpha: list = field(default_factory=list)     # chosen lambda per target (LassoCV)
    A_ebic: np.ndarray | None = None
    A_fixed: np.ndarray | None = None
    seconds_ebic: float = 0.0
    seconds_fixed: float = 0.0
    cpu_ebic: float = 0.0
    cpu_fixed: float = 0.0


def lasso_s2(X: np.ndarray, *, lambdas=LAMBDAS, do_cv: bool = True, do_path: bool = True, do_ebic: bool = False,
             do_fixed: bool = False, max_iter: int = 5000) -> LassoResult:
    """X: (M, T, N) observed data. Targets t = 1 .. T - 1, column of V^n_t = t N + n (time-major, as everywhere)."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LassoCV, lasso_path
    M, T, N = X.shape
    D = np.asarray(X, dtype=float).reshape(M, T * N)
    D = D - D.mean(0)
    A_cv = np.zeros((T * N, T * N), dtype=np.uint8)
    A_path = [np.zeros((T * N, T * N), dtype=np.uint8) for _ in lambdas]
    out = LassoResult(A_cv=A_cv, A_path=A_path, lambdas=tuple(lambdas))
    if do_ebic:
        out.A_ebic = np.zeros((T * N, T * N), dtype=np.uint8)
    if do_fixed:
        out.A_fixed = np.zeros((T * N, T * N), dtype=np.uint8)
    grid = np.array(sorted(lambdas, reverse=True))
    order_back = [int(np.where(grid == l)[0][0]) for l in lambdas]
    for t in range(1, T):
        for n in range(N):
            x, y = (t - 1) * N + n, t * N + n
            cols = [c for c in range(t * N) if c != x]
            xv = D[:, x]
            xx = float(xv @ xv)
            ry = D[:, y] - xv * (float(xv @ D[:, y]) / xx) if xx > 0 else D[:, y]
            Z = D[:, cols] - np.outer(xv, (xv @ D[:, cols]) / xx) if xx > 0 else D[:, cols]
            sd = Z.std(0)
            keep = sd > 1e-12
            sy = ry.std()
            for A in [A_cv] + A_path + [out.A_ebic, out.A_fixed]:
                if A is not None:
                    A[x, y] = 1
            if sy <= 1e-12 or not keep.any():
                continue
            Zs = Z[:, keep] / sd[keep]
            ys = ry / sy
            idx = [c for c, k in zip(cols, keep) if k]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                if do_cv:
                    t0, c0 = time.perf_counter(), time.process_time()
                    m = LassoCV(cv=5, alphas=100, fit_intercept=False, max_iter=max_iter, selection="cyclic").fit(Zs, ys)
                    out.seconds_cv += time.perf_counter() - t0
                    out.cpu_cv += time.process_time() - c0
                    out.n_cv_alpha.append(float(m.alpha_))
                    for j in np.nonzero(m.coef_)[0]:
                        A_cv[idx[j], y] = 1
                if do_ebic:
                    from scipy.special import gammaln
                    t0, c0 = time.perf_counter(), time.process_time()
                    _, coefs, _ = lasso_path(Zs, ys, alphas=100, max_iter=max_iter)
                    pp, nn = Zs.shape[1], Zs.shape[0]
                    rss = ((ys[:, None] - Zs @ coefs) ** 2).sum(0)
                    df = (coefs != 0).sum(0)
                    lc = gammaln(pp + 1) - gammaln(df + 1) - gammaln(pp - df + 1)
                    ebic = nn * np.log(np.maximum(rss, 1e-12 * nn) / nn) + df * np.log(nn) + 2.0 * lc
                    ebic = np.where(df <= nn // 2, ebic, np.inf)
                    k = int(np.argmin(ebic))
                    out.seconds_ebic += time.perf_counter() - t0
                    out.cpu_ebic += time.process_time() - c0
                    for j in np.nonzero(coefs[:, k])[0]:
                        out.A_ebic[idx[j], y] = 1
                if do_fixed:
                    t0, c0 = time.perf_counter(), time.process_time()
                    lam = float(np.sqrt(2.0 * np.log(max(Zs.shape[1], 2)) / Zs.shape[0]))
                    _, c1, _ = lasso_path(Zs, ys, alphas=np.array([lam]), max_iter=max_iter)
                    out.seconds_fixed += time.perf_counter() - t0
                    out.cpu_fixed += time.process_time() - c0
                    for j in np.nonzero(c1[:, 0])[0]:
                        out.A_fixed[idx[j], y] = 1
                if do_path:
                    t0, c0 = time.perf_counter(), time.process_time()
                    _, coefs, _ = lasso_path(Zs, ys, alphas=grid, max_iter=max_iter)   # coefs (p, len(grid)), decreasing lambda
                    out.seconds_path += time.perf_counter() - t0
                    out.cpu_path += time.process_time() - c0
                    for k, ib in enumerate(order_back):
                        for j in np.nonzero(coefs[:, ib])[0]:
                            A_path[k][idx[j], y] = 1
    return out
