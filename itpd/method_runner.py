"""Run one method on one instance and return a JSON-able summary (config, metrics, recorder summary, rules fired).

`run_s2_method(method, ...)`: `method` is one of "itpd_naive", "itpd", "itpd_adjacency_self", "itpd_marginal_first", "full_conditioning",
"iamb_known_order", "blanket_screened_shrink" (the `function` column of methods_registry.py); the strings of earlier releases
("order", "order_based", "hpv", "itpd_adjself", ...; `OLD_TO_NEW`) are still accepted and written back under the new name.
`run_s1_method` is the single-series setting.
"blanket_screened_shrink" (itpd/blanket_screened_shrink.py) takes its options as `shrink=dict(screening=..., alpha_A=...,
recheck=..., cap=..., keep_parent_tests=...)`; `alpha` is alpha_B (alpha_A defaults to alpha). The truth graph is passed to
it for screening="oracle_blanket" and for diagnostics only. "iamb_known_order" (itpd/iamb.py) is IAMB per target with the self
edge known and has no options. "itpd_marginal_first" is ITPD with the Z8 step never skipped and the Y-marginal
evaluated first (`marginal_first` in itpd/padl.py); it uses the Z4 rules of "itpd".
CI kinds: "oracle" (d-separation on the full graph), "fisherz", "gcm". The shared cache policy is `cache`
("run" by default for every method, so unique tests are what is counted and what is evaluated).
"""
from __future__ import annotations

import time

import numpy as np

from . import blanket_screened_shrink as _shrink
from . import iamb as _iamb
from . import itpd as _itpd
from .baselines import full_conditioning as _full_conditioning
from .ci import DSepCI, FisherZ, GCM, InfeasibleTest, MappedCI, Recorder
from .metrics import edge_metrics, lag_metrics
from .methods_registry import OLD_SCREENING_TO_NEW, OLD_TO_NEW
from .sim import SimResult, windows

ADJACENCY_SELF_RULES = _itpd.RULES + ("adjacency_self",)


def make_ci(kind: str, graph=None, data: np.ndarray | None = None):
    """`data` is (n, D) with D matching the graph columns (S2: data.reshape(M, T*N))."""
    if kind == "oracle":
        return DSepCI(graph.A)
    if kind == "fisherz":
        return FisherZ(data)
    if kind == "gcm":
        return GCM(data)
    raise ValueError(kind)


def run_s2_method(method: str, sim: SimResult | None, *, graph=None, ci_kind: str = "oracle", alpha: float = 0.01,
                  tau_max: int | None = None, variant: str = "paper", order: str = "variable", cache: str = "run",
                  ci=None, lazy: bool = False, infeasible: str = "raise", rules=None, shared: dict | None = None,
                  keep_graph: bool = False, per_target: bool = False, shrink: dict | None = None) -> dict:
    """`itpd_adjacency_self` = ITPD plus the optional adjacency_self rule; `rules` overrides the rule set of "itpd"."""
    method = OLD_TO_NEW.get(method, method)
    if shrink is not None and shrink.get("screening") in OLD_SCREENING_TO_NEW:
        shrink = {**shrink, "screening": OLD_SCREENING_TO_NEW[shrink["screening"]]}
    g = graph if graph is not None else sim.graph
    N, T = g.N, g.T
    if ci is None:
        data = None if sim is None or sim.X is None else sim.X.reshape(sim.X.shape[0], T * N)
        ci = make_ci(ci_kind, g, data)
    rec = Recorder(ci, cache=cache, infeasible=infeasible, shared=shared)
    t_start = time.perf_counter()
    try:
        if method == "itpd_naive":
            res = _itpd.run_s2(rec, N, T, alpha, reuse=False, tau_max=tau_max, variant=variant, order=order, lazy=lazy,
                               per_target=per_target)
        elif method in ("itpd", "itpd_adjacency_self"):
            rl = rules if rules is not None else (ADJACENCY_SELF_RULES if method == "itpd_adjacency_self" else _itpd.RULES)
            res = _itpd.run_s2(rec, N, T, alpha, reuse=True, tau_max=tau_max, variant=variant, order=order, lazy=lazy,
                               rules=rl, per_target=per_target)
        elif method == "full_conditioning":
            res = _full_conditioning.run_s2(rec, N, T, alpha, tau_max=tau_max, order=order, per_target=per_target)
        elif method == "itpd_marginal_first":
            res = _itpd.run_s2(rec, N, T, alpha, reuse=True, tau_max=tau_max, variant=variant, order=order, lazy=lazy,
                               rules=_itpd.RULES, per_target=per_target, marginal_first=True)
        elif method == "iamb_known_order":
            res = _iamb.run_s2(rec, N, T, alpha, tau_max=tau_max, order=order, per_target=per_target)
        elif method == "blanket_screened_shrink":
            kw = dict(shrink or {})
            aA = kw.pop("alpha_A", None)
            res = _shrink.run_s2(rec, N, T, alpha if aA is None else aA, alpha, A_true=g.A, per_target=per_target, **kw)
        else:
            raise ValueError(method)
    except InfeasibleTest:
        return {"method": method, "setting": "S2", "N": N, "T": T, "tau_max": tau_max, "alpha": alpha, "ci": ci_kind,
                "status": "infeasible", "metrics": None, "tests": rec.summary(), "seconds": time.perf_counter() - t_start}
    if per_target:                     # infeasible targets are excluded from the score, never answered "independent"
        bad = list(res.infeasible_targets)
        tmask = np.ones(T * N, dtype=bool)
        tmask[bad] = False
        n_scored_targets = (T - 1) * N
        n_bad = len(bad)
        out = {"status": "ok" if n_bad == 0 else ("infeasible" if n_bad == n_scored_targets else "partial"),
               "method": method, "setting": "S2", "N": N, "T": T, "tau_max": tau_max, "alpha": alpha, "ci": ci_kind,
               "variant": variant if method not in ("full_conditioning", "iamb_known_order") else None, "order": order, "lazy": lazy,
               "metrics": None if n_bad == n_scored_targets else edge_metrics(g.A, res.A_hat, N, T, targets=tmask),
               "n_targets": n_scored_targets, "n_infeasible_targets": n_bad, "infeasible_targets": bad,
               "tests": res.summary, "skips": res.skips, "seconds": res.seconds, "per_pair": res.per_pair}
        if method == "blanket_screened_shrink":
            out["variant"], out["shrink"], out["shrink_stats"] = None, dict(shrink or {}), res.stats
        if method == "iamb_known_order":
            out["iamb_stats"] = res.stats
        if keep_graph:
            out["A_hat"] = res.A_hat
        return out
    if res.summary["infeasible"]:      # infeasible="flag": finished, but not scored (`metrics` stays None)
        out = {"method": method, "setting": "S2", "N": N, "T": T, "tau_max": tau_max, "alpha": alpha, "ci": ci_kind,
               "status": "infeasible", "metrics": None, "tests": res.summary, "seconds": res.seconds,
               "per_pair": res.per_pair,
               # diagnostic only: the infeasible tests were answered "independent" (flagged and counted in tests.p_nan)
               "metrics_flagged": edge_metrics(g.A, res.A_hat, N, T), "skips": res.skips}
        if keep_graph:
            out["A_hat"] = res.A_hat
        return out
    out = {"status": "ok", "method": method, "setting": "S2", "N": N, "T": T, "tau_max": tau_max, "alpha": alpha, "ci": ci_kind,
           "variant": variant if method not in ("full_conditioning", "iamb_known_order") else None, "order": order, "lazy": lazy,
           "metrics": edge_metrics(g.A, res.A_hat, N, T), "tests": res.summary, "skips": res.skips,
           "seconds": res.seconds, "per_pair": res.per_pair}
    if method == "blanket_screened_shrink":
        out["variant"], out["shrink"], out["shrink_stats"] = None, dict(shrink or {}), res.stats
    if method == "iamb_known_order":
        out["iamb_stats"] = res.stats
    if keep_graph:
        out["A_hat"] = res.A_hat
    return out


def run_s1_method(method: str, sim: SimResult, *, ci_kind: str = "oracle", alpha: float = 0.01, variant: str = "paper",
                  cache: str = "run", infeasible: str = "raise") -> dict:
    """S1: one run per series on the last window slice; truth = window graph sim.graph.B."""
    method = OLD_TO_NEW.get(method, method)
    g = sim.graph
    N, T, tau = g.N, g.T, g.tau
    if ci_kind == "oracle":
        cols = [(T - tau - 1 + l) * N + n for l in range(tau + 1) for n in range(N)]
        ci = MappedCI(DSepCI(g.A), cols)
    else:
        ci = make_ci(ci_kind, None, windows(sim.X, tau))
    rec = Recorder(ci, cache=cache, infeasible=infeasible)
    try:
        if method in ("itpd", "itpd_naive"):
            res = _itpd.run_s1(rec, N, tau, alpha, variant=variant)
        elif method == "full_conditioning":
            res = _full_conditioning.run_s1(rec, N, tau, alpha)
        else:
            raise ValueError(method)
    except InfeasibleTest:
        return {"method": method, "setting": "S1", "N": N, "T": T, "tau": tau, "status": "infeasible",
                "metrics": None, "tests": rec.summary()}
    bad = res.summary["infeasible"]
    return {"status": "infeasible" if bad else "ok", "method": method, "setting": "S1", "N": N, "T": T, "tau": tau,
            "alpha": alpha, "ci": ci_kind, "variant": variant if method != "full_conditioning" else None,
            "metrics": None if bad else lag_metrics(g.B, res.B_hat),
            "tests": res.summary, "seconds": res.seconds}

