"""Score an estimated time graph against the true one.

Start at `score`: it returns precision, recall, and F1 on all targets and on
the common targets.

Edges are directed and time-indexed. The lag-1 self edges are not scored. A
target is a node at step 1 or later; its edges are the entries of its column.
An untested target has no parents in the estimate, so its true edges count as
missed.

Common targets are the targets that every method can test with M observations:
the Fisher-z test of a target at step t given all earlier nodes needs
M >= N * t + 3.
"""
import numpy as np

from .graphs import self_edge_mask


def last_common_step(N: int, T: int, M: int | None = None) -> int:
    """Return the last step whose targets are common targets.

    M is `None` for a CI test that needs no minimum number of observations
    (`DSeparation`, `GCM`): every target is then a common target."""
    if M is None:
        return T - 1
    return int(max(0, min(T - 1, (M - 3) // N)))


def edge_counts(A_true, A_hat, N: int, T: int, *, last_step: int | None = None,
                untested=()) -> dict:
    """Return `tp`, `fp` and `fn` over the targets at steps 1, ..., `last_step`.

    `last_step` defaults to T - 1. `untested` lists the untested targets."""
    keep = ~self_edge_mask(N, T)
    truth = (np.asarray(A_true) != 0) & keep
    estimate = (np.asarray(A_hat) != 0) & keep
    estimate[:, [int(c) for c in untested]] = False
    step = np.arange(T * N) // N
    scored = (step >= 1) & (step <= (T - 1 if last_step is None else last_step))
    return {"tp": int((estimate & truth)[:, scored].sum()),
            "fp": int((estimate & ~truth)[:, scored].sum()),
            "fn": int((truth & ~estimate)[:, scored].sum())}


def precision(tp: int, fp: int) -> float:
    """Return tp / (tp + fp), or NaN when there is no estimated edge."""
    return tp / (tp + fp) if tp + fp else float("nan")


def recall(tp: int, fn: int) -> float:
    """Return tp / (tp + fn), or NaN when there is no true edge."""
    return tp / (tp + fn) if tp + fn else float("nan")


def f1(tp: int, fp: int, fn: int) -> float:
    """Return 2 tp / (2 tp + fp + fn), or NaN when the denominator is 0."""
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float("nan")


def score(A_true, A_hat, N: int, T: int, M: int | None = None, untested=()) -> dict:
    """Return the scores on all targets and on the common targets.

    The keys `'all'` and `'common'` each hold `tp`, `fp`, `fn`, `precision`,
    `recall` and `f1`. `'common'` is `None` when no target is common
    (`last_common_step` < 1). `n_untested` counts the untested targets."""
    last = last_common_step(N, T, M)
    untested = sorted({int(c) for c in untested if c >= N})

    def scores(last_step):
        counts = edge_counts(A_true, A_hat, N, T, last_step=last_step, untested=untested)
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        return {**counts, "precision": precision(tp, fp), "recall": recall(tp, fn),
                "f1": f1(tp, fp, fn)}

    return {"all": scores(T - 1), "common": scores(last) if last >= 1 else None,
            "last_common_step": last, "n_untested": len(untested)}
