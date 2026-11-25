"""Check runs of the paper's experiments against stored runs.

A run is one method on one instance at the significance level 0.01, made as the experiment runner makes it:
itpd.experiments.instances.make_instance builds the instance, and itpd.experiments.methods.run_method runs the method.

Simulated data pass through BLAS, so another CPU type or BLAS build can change their last bits. A run on data
first compares the SHA-1 of its data with the stored one. When they differ, the run is skipped with the reason
"the data differ on this platform"; with ITPD_REQUIRE_REFERENCE_DATA=1 in the environment it fails instead.
The run with the oracle has no data and is never skipped.
"""
import functools
import os

import numpy as np
import pytest

from itpd import metrics
from itpd.experiments import instances
from itpd.experiments.methods import run_method
from itpd.simulation import linear

# Cell -> graph block of a config, data mechanism, CI test, graph index, M, SHA-1, and the number of first rows of
# the data that the SHA-1 is of (the oracle has no data: the SHA-1 is that of the graph).
CELLS = {
    "main-d2-g0-M200": ({"generator": "standard", "N": 20, "T": 16, "lag": 1, "in_degree": 2, "stationary": False},
                        "linear_gaussian", "fisherz", 0, 200, "f7df382ae9f36ffc0f960f94ba2ed9e41afa36db", 2000),
    "main-d2-g1-M500": ({"generator": "standard", "N": 20, "T": 16, "lag": 1, "in_degree": 2, "stationary": False},
                        "linear_gaussian", "fisherz", 1, 500, "5e2ef56f5c751c391e42191ef9d5b5649476f6ca", 2000),
    "main-d1-g0-M2000": ({"generator": "standard", "N": 20, "T": 16, "lag": 1, "in_degree": 1, "stationary": False},
                         "linear_gaussian", "fisherz", 0, 2000, "37c8fcbb6c0126ecd13ee7f9849a012930b09bb1", 2000),
    "lag1-stationary-g0-M500": ({"generator": "standard", "N": 10, "T": 16, "lag": 1, "in_degree": 2, "stationary": True},
                                "linear_gaussian", "fisherz", 0, 500, "349f0227b95160b11ca8cd9dfabc80b463c8d19b", 2000),
    "lag2-nonstationary-g0-M500": ({"generator": "standard", "N": 10, "T": 16, "lag": 2, "in_degree": 2, "stationary": False},
                                   "linear_gaussian", "fisherz", 0, 500, "d2074b2468713565401963889f556631c8437476", 2000),
    "oracle-d1-g0": ({"generator": "standard", "N": 20, "T": 16, "lag": 1, "in_degree": 1, "stationary": True},
                     "none", "oracle", 0, None, "541a6fdd353ee43da1c72a71068eb2ff88ffd3a7", None),
    "redrawn-edges-share0.05-g0-M500": ({"generator": "redrawn_edges", "N": 20, "T": 16, "in_degree": 2, "share": 0.05},
                                        "linear_gaussian", "fisherz", 0, 500, "e4ff4a39447ad9038baabad4eac136d54ba844b2", 500),
    "nonlinear-stationary-g0-M500": ({"generator": "standard", "N": 10, "T": 8, "lag": 1, "in_degree": 2, "stationary": True},
                                     "nonlinear", "gcm", 0, 500, "05d17335019d7d0f3d03edc060b293e64059c524", 2000),
}

# Counts of stored runs of the paper's experiments: tp, fp and fn over all targets, requested and unique CI tests, largest conditioning set, untested targets.
# Columns: cell, method, (tp, fp, fn), (requested, unique, largest set) or None for a method without CI tests, untested targets.
RUNS = [
    ("main-d2-g0-M200", {"name": "ITPD"}, (554, 7, 69), (63470, 62495, 25), 0),
    ("main-d2-g1-M500", {"name": "ITPD"}, (591, 3, 7), (64769, 63818, 28), 0),
    ("main-d1-g0-M2000", {"name": "ITPD"}, (312, 1, 0), (65468, 64501, 29), 0),
    ("lag1-stationary-g0-M500", {"name": "ITPD"}, (269, 4, 1), (17878, 17378, 21), 0),
    ("lag1-stationary-g0-M500", {"name": "IAMB", "level": "alpha"}, (270, 89, 0), (43847, 43707, 8), 0),
    ("lag1-stationary-g0-M500", {"name": "IAMB", "level": "corrected"}, (270, 2, 0), (34141, 34001, 5), 0),
    ("lag1-stationary-g0-M500", {"name": "IPaDL", "level": "alpha"}, (258, 93, 12), (36026, 24985, 84), 0),
    ("lag1-stationary-g0-M500", {"name": "IPaDL", "level": "corrected"}, (259, 10, 11), (32125, 21078, 54), 0),
    ("lag1-stationary-g0-M500", {"name": "FullCI", "level": "alpha"}, (270, 109, 0), (11850, 11850, 149), 0),
    ("lag1-stationary-g0-M500", {"name": "FullCI", "level": "corrected"}, (270, 0, 0), (11850, 11850, 149), 0),
    ("lag1-stationary-g0-M500", {"name": "Lasso"}, (269, 71, 1), None, 0),
    ("lag2-nonstationary-g0-M500", {"name": "ITPD"}, (275, 9, 6), (16963, 16502, 25), 0),
    ("oracle-d1-g0", {"name": "ITPD"}, (240, 0, 0), (52820, 52669, 4), 0),
    ("redrawn-edges-share0.05-g0-M500", {"name": "ITPD"}, (604, 3, 6), (65893, 64905, 29), 0),
    ("nonlinear-stationary-g0-M500", {"name": "Lasso"}, (97, 14, 1), None, 0),
    ("main-d2-g0-M200", {"name": "ITPD", "switches": {"ancestor_gate": False}}, (554, 6, 69), (69247, 67898, 21), 0),
]

# More stored runs, marked slow: the baselines on the first cell of the main setting take 1 to 9 seconds each.
SLOW_RUNS = [
    ("main-d2-g0-M200", {"name": "IAMB", "level": "alpha"}, (603, 416, 20), (225572, 225291, 11), 0),
    ("main-d2-g0-M200", {"name": "IAMB", "level": "corrected"}, (553, 8, 70), (133415, 133157, 7), 0),
    ("main-d2-g0-M200", {"name": "IPaDL", "level": "alpha"}, (557, 98, 66), (104508, 59915, 41), 0),
    ("main-d2-g0-M200", {"name": "IPaDL", "level": "corrected"}, (427, 2, 196), (98500, 53896, 14), 0),
    ("main-d2-g0-M200", {"name": "FullCI", "level": "alpha"}, (254, 184, 369), (17940, 17940, 299), 120),
    ("main-d2-g0-M200", {"name": "FullCI", "level": "corrected"}, (154, 2, 469), (17940, 17940, 299), 120),
    ("main-d2-g0-M200", {"name": "Lasso"}, (547, 59, 76), None, 0),
]


def _name(run) -> str:
    cell, method = run[:2]
    return "-".join([cell, method["name"], *([method["level"]] if "level" in method else []), *method.get("switches", {})])


@functools.cache
def _instance(cell):
    """Return the instance of a cell, its draw of data (None for the oracle), and whether the SHA-1 is the stored one."""
    graph, mechanism, _, g, _, sha1, hashed = CELLS[cell]
    instance = instances.make_instance(graph, mechanism, 0, g)
    if mechanism == "none":
        return instance, None, instance.graph_sha1 == sha1
    X = instance.sample(linear.ROWS)
    return instance, X, linear.data_hash(X[:hashed]) == sha1


@pytest.mark.parametrize("run", [*RUNS, *(pytest.param(run, marks=pytest.mark.slow) for run in SLOW_RUNS)], ids=_name)
def test_stored_run(run):
    cell, method, counts, tests, untested = run
    _, mechanism, ci_test, _, M, _, _ = CELLS[cell]
    instance, X, same = _instance(cell)
    if mechanism == "none":
        assert same
    elif not same:
        if os.environ.get("ITPD_REQUIRE_REFERENCE_DATA") == "1":
            pytest.fail("the data differ on this platform")
        pytest.skip("the data differ on this platform")

    result = run_method(method, instance, None if X is None else X[:M], ci_test, 0.01)

    A_hat = np.zeros(instance.A.size, dtype=bool)
    A_hat[result["edges"]] = True
    score = metrics.score(instance.A, A_hat.reshape(instance.A.shape), instance.N, instance.T, untested=result["untested"])["all"]
    assert (score["tp"], score["fp"], score["fn"]) == counts
    assert len(result["untested"]) == untested and result["status"] == ("partial" if untested else "ok")
    if tests is None:
        assert "requested" not in result
    else:
        assert (result["requested"], result["unique"], result["largest_set"]) == tests
