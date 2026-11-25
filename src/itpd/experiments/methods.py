"""Run one method on one instance.

The methods are ITPD, the baselines IAMB, IPaDL, FullCI and Lasso of
itpd.baselines, and the external baselines PCMCI, DF-PC, DYNOTEARS and CD-NOD
of itpd.external. Every method with CI tests of this package calls the test
through its own CICounter.

IAMB, IPaDL, FullCI, PCMCI, DF-PC and CD-NOD run at one of two levels: "alpha"
is the nominal level, and "corrected" is alpha divided by the number of
candidate parents of a target.
"""
import re
import time
import warnings

import numpy as np

from .. import GCM, ITPD, CICounter, DSeparation, FisherZ
from ..baselines import full_conditioning, iamb, ipadl, lasso
from ..external import SCRIPTS, corrected_level, run_external

CI_TESTS = {"fisherz": FisherZ, "gcm": GCM}
BASELINES = {"IAMB": iamb, "IPaDL": ipadl, "FullCI": full_conditioning}
LEVELS = ("alpha", "corrected")
# Method -> the keys of its entry in a config, besides name.
OPTIONS = {"ITPD": ("label", "switches"), "IAMB": ("levels",), "IPaDL": ("levels",), "FullCI": ("levels",), "Lasso": (),
           "PCMCI": ("levels", "max_lag", "time_limit"), "DF-PC": ("levels", "time_limit"),
           "DYNOTEARS": ("max_lag", "unit_variance", "time_limit"), "CD-NOD": ("levels", "max_lag", "time_limit")}


def slug(label: str) -> str:
    """Return a label as a part of a file name."""
    return re.sub(r"[^A-Za-z0-9]+", "-", label).strip("-")


def expand(entry, lag: int, ci_test: str) -> list:
    """Return the runs of one method entry of a config: one dict per level, with the name and the options.

    lag is the lag of the graphs: PCMCI, DYNOTEARS and CD-NOD search up to
    max_lag, which defaults to it. A method with levels runs at "alpha"
    unless the entry lists its levels. Raise ValueError for an entry that
    names no method, holds an unknown key, or cannot run with the CI test
    of the setting."""
    name = entry.get("name") if isinstance(entry, dict) else None
    if name not in OPTIONS:
        raise ValueError(f"unknown method: {entry!r}")
    unknown = sorted(set(entry) - set(OPTIONS[name]) - {"name"})
    if unknown:
        raise ValueError(f"method {name}: unknown key {', '.join(map(str, unknown))}")
    if ci_test == "oracle" and (name in SCRIPTS or name == "Lasso"):
        raise ValueError(f"{name} needs data; the oracle has none")
    if ci_test != "fisherz" and name in SCRIPTS and "levels" in OPTIONS[name]:
        raise ValueError(f"{name} runs its own Fisher-z test; it cannot run with the CI test {ci_test}")
    run = {"name": name}
    if "label" in entry or "switches" in entry:
        # A run with switches has its own file name, so it never takes the file of Algorithm 1.
        label = entry.get("label")
        if not isinstance(entry.get("switches"), dict) or not isinstance(label, str) or not slug(label) or slug(label) in OPTIONS:
            raise ValueError("an ITPD entry with switches needs a label that is not a method name, and a label needs switches")
        run.update(label=label, switches=entry["switches"])
    if "max_lag" in OPTIONS[name]:
        run["max_lag"] = entry.get("max_lag", lag)
    if "time_limit" in entry:
        run["time_limit"] = entry["time_limit"]
    if entry.get("unit_variance"):
        run["unit_variance"] = True
    if "levels" not in OPTIONS[name]:
        return [run]
    levels = entry.get("levels", ["alpha"])
    if not isinstance(levels, list) or not levels or set(levels) - set(LEVELS):
        raise ValueError(f"{name}: levels must be a list from {', '.join(LEVELS)}")
    return [dict(run, level=level) for level in levels]


def _itpd(method, counter, instance, V, alpha):
    if "switches" in method:
        from .ablations import ITPDAblation
        model = ITPDAblation(counter, alpha=alpha, switches=method["switches"])
    else:
        model = ITPD(counter, alpha=alpha)
    with warnings.catch_warnings():
        # The result lists the untested targets.
        warnings.filterwarnings("ignore", message=".*could not be evaluated", category=RuntimeWarning)
        # The oracle needs no data: fit reads only the shape.
        model.fit(np.zeros((2, instance.T, instance.N)) if V is None else V)
    if sum(model.ci_queries_.values()) != counter.requested:
        raise RuntimeError("ITPD and the counter disagree on the number of requested tests")
    return model.get_adjacency_matrix(), model.untested_targets_


def _external(method, instance, V, alpha) -> dict:
    name = method["name"]
    options = {key: method[key] for key in ("max_lag", "unit_variance") if key in method}
    if "level" in method:
        options["alpha"] = alpha if method["level"] == "alpha" else corrected_level(
            name, alpha, instance.N, instance.T, method.get("max_lag", 1))
    out = run_external(name, V, time_limit=method.get("time_limit"), **options)
    kept = ("status", "message", "requested", "unique", "largest_set", "mean_set", "seconds", "versions")
    result = {key: out[key] for key in kept if key in out}
    # Only a run that ended is an estimate.
    result["edges"] = np.flatnonzero(out["A_hat"]).tolist() if out["status"] == "ok" else None
    result["untested"] = []
    return result


def run_method(method: dict, instance, V, ci_test: str, alpha: float) -> dict:
    """Run a method and return what it found.

    Parameters
    ----------
    method : dict
        name, and where they apply level ("alpha" or "corrected"), max_lag,
        time_limit (seconds), unit_variance, switches.
    instance : itpd.simulation.Instance
        Gives N, T and, for the oracle, the true graph.
    V : ndarray (M, T, N) or None
        The data; None with the oracle.
    ci_test : {"fisherz", "gcm", "oracle"}
        The CI test of ITPD, IAMB, IPaDL and FullCI.
    alpha : float
        Nominal significance level.

    Returns
    -------
    dict
        status: "ok"; "partial" when a target is untested; for an external
        baseline also "timeout", "skipped" or "error" (then with a message).
        edges: the estimate as a list of u * T N + v for u -> v, or None when
        there is no estimate. untested: the untested targets. requested,
        unique, largest_set, mean_set: the counts of CI tests, where the
        method has them. seconds: time of the method call.
    """
    name, N, T = method["name"], instance.N, instance.T
    if name in SCRIPTS:
        return _external(method, instance, V, alpha)
    start = time.perf_counter()
    if name == "Lasso":
        result, counts = lasso(V), {}
        A_hat, untested = result.adjacency, result.untested_targets
    else:
        counter = CICounter(DSeparation(instance.A) if ci_test == "oracle" else CI_TESTS[ci_test](V))
        if name == "ITPD":
            A_hat, untested = _itpd(method, counter, instance, V, alpha)
        else:
            result = BASELINES[name](counter, N, T, alpha, corrected=method["level"] == "corrected")
            A_hat, untested = result.adjacency, result.untested_targets
        counts = counter.summary(N, T)
    seconds = time.perf_counter() - start
    return {"status": "partial" if untested else "ok", "edges": np.flatnonzero(A_hat).tolist(),
            "untested": [int(y) for y in untested], **counts, "seconds": seconds}
