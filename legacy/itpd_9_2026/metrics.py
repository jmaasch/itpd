"""The one scorer for every method: precision, recall, F1 and SHD on directed time-indexed edges.

Conventions (identical for every method):
- Headline: the autocorrelation (self) edges V^n_t -> V^n_{t+1} are excluded from truth and estimate (they are built in);
  `include_self=True` scores them too (robustness arm with missing or false self edges).
- All edges point forward in time, so SHD = FP + FN. Estimated edges that are not forward in time (same-time or
  backward entries of A_hat) count as FP and are also reported in `fp_nonforward`.
- F1 is computed per graph; aggregate by median over graphs. 0/0 precision or recall is 1.
- S1 window graphs B[l, i, j] (V^i_{t-l} -> V^j_t, l = 0..tau): lag 0 and the self edge are excluded by default;
  `include_lag0=True` scores same-time edges (robustness arm).
"""
from __future__ import annotations

import numpy as np

from .graphs import autocorr_mask


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def edge_metrics(A_true: np.ndarray, A_hat: np.ndarray, N: int, T: int, include_self: bool = False,
                 targets=None) -> dict:
    """Full time graph (T*N, T*N). `targets` (bool array over columns, or None): score only edges whose child is a
    target in the set (truth and estimate both restricted); used for infeasible-target handling and for the targets
    feasible for every method."""
    t, h = np.asarray(A_true) != 0, np.asarray(A_hat) != 0
    if not include_self:
        keep = ~autocorr_mask(N, T)
        t, h = t & keep, h & keep
    if targets is not None:
        keep_c = np.asarray(targets, dtype=bool)[None, :]
        t, h = t & keep_c, h & keep_c
    tp, fp, fn = int((t & h).sum()), int((~t & h).sum()), int((t & ~h).sum())
    ii, jj = np.nonzero(h)
    nonfwd = int(((jj // N) <= (ii // N)).sum())
    p, r, f = _prf(tp, fp, fn)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f, "shd": fp + fn,
            "fp_nonforward": nonfwd, "exact": fp + fn == 0}


def lag_metrics(B_true: np.ndarray, B_hat: np.ndarray, include_self: bool = False, include_lag0: bool = False) -> dict:
    t, h = (np.asarray(B_true) != 0).copy(), (np.asarray(B_hat) != 0).copy()
    if not include_lag0:
        t[0] = h[0] = False
    if not include_self:
        i = np.arange(t.shape[1])
        t[1][i, i] = h[1][i, i] = False
    tp, fp, fn = int((t & h).sum()), int((~t & h).sum()), int((t & ~h).sum())
    p, r, f = _prf(tp, fp, fn)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f, "shd": fp + fn, "exact": fp + fn == 0}
