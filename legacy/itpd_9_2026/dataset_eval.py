"""Finite-data harness shared by the finite-data, robustness, nonlinear, baseline and ITPD-S drivers: run the methods of a spec list on
one dataset (one test object, one shared p-value memo), per target feasibility, scoring on own-feasible and on common-feasible targets.
Spec lists (name, method, variant, lazy, alphas) are built from methods_registry.py.

Infeasible tests (Fisher-z with n - |S| - 3 <= 0, i.e. M < |S| + 4) are never answered "independent": the PaDL call (one
target) that needs one is reported infeasible, no parents are output for it, it is excluded from the score, and its labels
are not used for the skip rules of later targets (itpd.run_s2(per_target=True)). Per run we record:
 - own-feasible targets: precision / recall / F1 / SHD over the targets the method could decide, infeasible-target count;
 - common targets: targets t <= common_tmax = floor((M - 3) / N), the targets whose full-conditioning set (N t - 1 variables)
   is feasible, hence feasible for every method (ITPD and ITPD_naive sets are subsets of the same candidate window);
   metrics on these are comparable across methods;
 - unique / raw tests (all targets, and summed over the common targets), by conditioning size, per-target largest
   conditioning set and the size of the first infeasible set (detail alpha only).
"""
from __future__ import annotations

import numpy as np

from . import method_runner
from .ci import FisherZ
from .graphs import TimeGraph
from .methods_registry import current_shrink_options, spec
from .metrics import edge_metrics

ALPHAS = (0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 5e-4, 2e-4, 1e-4, 1e-5, 1e-6)
PRIMARY = 0.01

# Spec = (row name, method, PaDL variant or shrink options, lazy, alphas); built from methods_registry.spec (names in methods_registry.py).
# Lazy headline for the whole alpha grid, non-lazy counting beside it at PRIMARY only.
ITPD_AND_ORDER_SPECS = (
    spec("itpd_naive", ALPHAS),
    spec("itpd", ALPHAS),
    spec("itpd_repo_variant", ALPHAS),
    spec("full_conditioning", ALPHAS),
    spec("itpd_naive_nonlazy", (PRIMARY,)),
    spec("itpd_nonlazy", (PRIMARY,)),
    spec("itpd_repo_variant_nonlazy", (PRIMARY,)),
    spec("itpd_adjacency_self", (PRIMARY,)),
    spec("itpd_adjacency_self_nonlazy", (PRIMARY,)),
)
# Known-order IAMB per target and ITPD + marginal-first (lazy headline + non-lazy beside it at PRIMARY); run by experiments/stored_instances.py (known_order).
IAMB_MARGINAL_FIRST_SPECS = (
    spec("iamb_known_order", ALPHAS),
    spec("itpd_marginal_first", ALPHAS),
    spec("itpd_marginal_first_nonlazy", (PRIMARY,)),
)
# The five rows of ITPD_AND_ORDER_SPECS that the robustness and quick runs repeat, at PRIMARY only.
PRIMARY_ALPHA_SPECS = tuple(s[:4] + ((PRIMARY,),) for s in ITPD_AND_ORDER_SPECS
                            if s[0] in ("itpd_naive", "itpd", "full_conditioning", "itpd_naive_nonlazy", "itpd_nonlazy"))


def common_tmax(N: int, T: int, M: int) -> int:
    """Largest time index whose full-conditioning set (N t - 1 variables) is feasible: M >= N t + 3."""
    return int(max(0, min(T - 1, (M - 3) // N)))


def _cols_mask(N: int, T: int, tmax: int) -> np.ndarray:
    m = np.zeros(T * N, dtype=bool)
    m[: (tmax + 1) * N] = True
    return m


def run_dataset(A_truth: np.ndarray, X: np.ndarray, tau: int, specs=ITPD_AND_ORDER_SPECS, *, order: str = "time",
                edges: np.ndarray | None = None, score_self: bool = False, n_lag0: int = 0, ci=None,
                shared: dict | None = None, on_run=None) -> dict:
    """`A_truth` (T*N, T*N): headline truth on the observed nodes (self edges present as they are in the DGP);
    X (M, T, N). Returns {"M", "common_tmax", "runs": [...]}.
    `ci` (default: Fisher-z on X): any CI test object on the columns of X.reshape(M, T * N), e.g. `itpd.ci.GCM`; a test
    without a sample-size feasibility rule (GCM) makes every target common (common_tmax = T - 1).
    `shared`: p-value memo to reuse and extend (checkpoint / resume); `on_run(shared)` is called after every (method, alpha) run."""
    M, T, N = X.shape
    tg = TimeGraph(A=np.asarray(A_truth) != 0, N=N, T=T, tau=tau)
    ct = common_tmax(N, T, M) if ci is None else T - 1
    if ci is None:
        ci = FisherZ(X.reshape(M, T * N))
    shared = {} if shared is None else shared
    cmask = _cols_mask(N, T, ct)
    runs = []
    for name, method, variant, lazy, alphas in specs:
        for alpha in alphas:
            detail = abs(alpha - PRIMARY) < 1e-12
            hk = current_shrink_options(variant) if method == "itpd_s" else None   # `variant` is the dict of shrink options, alpha = alpha_shr
            o = method_runner.run_s2_method(method, None, graph=tg, ci=ci, alpha=alpha,
                                     variant="paper" if hk is not None else (variant or "paper"), order=order,
                                     lazy=lazy, infeasible="raise", shared=shared, keep_graph=True, per_target=True, shrink=hk)
            pp = o["per_pair"]
            A_hat = o["A_hat"]
            t = o["tests"]
            row = {"name": name, "alpha": alpha, "lazy": lazy, "status": o["status"], "n_targets": o["n_targets"],
                   "n_inf_targets": o["n_infeasible_targets"], "unique": t["unique_tests"], "raw": t["raw_calls"],
                   "p_nan_unique": t["p_nan_unique"], "seconds": o["seconds"], "metrics": o["metrics"]}
            if hk is not None:
                row["alpha_scr"] = alpha if hk.get("alpha_scr") is None else hk["alpha_scr"]
                row["shrink"] = {k: v for k, v in hk.items() if k != "keep_parent_tests"}
                row["shrink_stats"] = o["shrink_stats"]
                row["by_label_unique"] = t["by_label_unique"]
                row["by_label_raw"] = t["by_label_raw"]
            if method == "iamb_known_order":
                row["iamb_stats"] = o["iamb_stats"]
                row["by_label_unique"] = t["by_label_unique"]
            # common targets (feasible for every method): metrics, tests, infeasible targets inside the set (must be 0)
            bad = set(o["infeasible_targets"])
            row["common_bad"] = int(sum(1 for c in bad if c < (ct + 1) * N))
            if ct >= 1:
                row["metrics_common"] = edge_metrics(tg.A, A_hat, N, T, targets=cmask)
                cp = [p for p in pp if p["pair"][1] < (ct + 1) * N]
                row["unique_common"] = int(sum(p["new_unique"] for p in cp))
                row["raw_common"] = int(sum(p["raw"] for p in cp))
            else:
                row["metrics_common"] = None
            if score_self:
                tm = np.ones(T * N, dtype=bool)
                tm[list(bad)] = False
                row["metrics_self"] = edge_metrics(tg.A, A_hat, N, T, include_self=True, targets=tm)
                row["n_lag0_true"] = int(n_lag0)
            if detail:
                row["by_size_unique"] = t["by_size_unique"]
                row["by_size_raw"] = t["by_size_raw"]
                row["target_max"] = [int(p["max_size"]) for p in pp]
                row["target_inf"] = [int(bool(p.get("infeasible"))) for p in pp]
                row["first_inf_size"] = [p.get("first_infeasible_size") for p in pp if p.get("infeasible")]
                row["skips"] = o["skips"]
                if method == "iamb_known_order":
                    for k in ("grown", "shrunk", "grow_iters"):
                        row["target_" + k] = [p.get(k) for p in pp]
                if hk is not None:
                    for k in ("n_R", "max_SA", "max_SB", "excess", "lost", "n_Aprime", "a_inf"):
                        row["target_" + k] = [p.get(k) for p in pp]
                    pt = [[int(p["pair"][1]), p["ptests"]] for p in pp if p.get("ptests")]
                    if pt:
                        row["screening_parent_tests"] = pt
                if edges is not None and ct >= 1:
                    u, v = edges[:, 0], edges[:, 1]
                    inc = (v // N) <= ct
                    row["fn_idx"] = [int(i) for i in np.nonzero(inc & (A_hat[u, v] == 0))[0]]
            runs.append(row)
            if on_run is not None:
                on_run(shared)
    return {"M": M, "common_tmax": ct, "runs": runs}
