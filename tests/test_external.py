"""Tests of the external-baseline launcher and shared script code."""
import sys

import numpy as np
import pytest

from itpd import CICounter
from itpd.experiments.methods import run_method
from itpd.external import _common, corrected_level, run_external
from itpd.graphs import unroll
from itpd.simulation import linear


def test_missing_interpreter_gives_skipped(no_interpreters, monkeypatch):
    X = np.zeros((5, 4, 2))
    result = run_external("PCMCI", X, alpha=0.01)
    assert result == {"status": "skipped", "message": "ITPD_PYTHON_PCMCI is not set"}
    monkeypatch.setenv("ITPD_PYTHON_PCMCI", "no-such-interpreter")
    assert run_external("PCMCI", X, alpha=0.01)["message"] == "ITPD_PYTHON_PCMCI names no file"
    # DF-PC also needs the folder of its clone.
    monkeypatch.setenv("ITPD_PYTHON_DFPC", sys.executable)
    assert run_external("DF-PC", X, alpha=0.01) == {"status": "skipped", "message": "ITPD_DFPC_SOURCE does not name a DF-PC clone"}


def test_a_skipped_method_has_no_estimate(no_interpreters):
    instance = linear.make_instance(3, 4, 1, 1, 0, stationary=False)
    for method in ({"name": "PCMCI", "level": "alpha", "max_lag": 1}, {"name": "DF-PC", "level": "corrected", "time_limit": 5},
                   {"name": "DYNOTEARS", "max_lag": 1, "unit_variance": True}, {"name": "CD-NOD", "level": "corrected", "max_lag": 1}):
        result = run_method(method, instance, instance.sample(50), "fisherz", 0.01)
        assert result["status"] == "skipped" and result["edges"] is None and result["untested"] == []


def test_corrected_level():
    # PCMCI and DF-PC: the candidates of a target at the last step. CD-NOD: the lagged columns of its window.
    assert corrected_level("PCMCI", 0.01, 20, 16) == corrected_level("DF-PC", 0.01, 20, 16) == 0.01 / 299
    assert corrected_level("CD-NOD", 0.01, 20, 16) == 0.01 / 20
    assert corrected_level("CD-NOD", 0.01, 10, 16, max_lag=4) == 0.01 / 40


def test_window_graph_of_a_script_is_repeated_at_every_step():
    window = np.random.default_rng(0).random((3, 4, 4)) < 0.3
    A = _common.window_to_time_graph(window, 6)
    assert A.dtype == bool and np.array_equal(A, unroll(window, 6))


def test_counter_of_the_scripts_counts_as_cicounter():
    rng = np.random.default_rng(1)
    script_counter, counter = _common.TestCounter(), CICounter(lambda x, y, S: 0.5)
    for _ in range(300):
        x, y = (int(v) for v in rng.choice(6, 2, replace=False))
        S = [int(s) for s in rng.choice(30, int(rng.integers(0, 13)))]      # with repeats, and with x or y
        for _ in range(int(rng.integers(1, 3))):
            script_counter.record(x, y, S)
            counter(y, x, S)
    summary = counter.summary(2, 2)
    assert script_counter.summary() == {key: summary[key] for key in ("requested", "unique", "largest_set", "mean_set")}
    assert summary["requested"] > summary["unique"] > 0 and summary["largest_set"] > 8


def test_counter_of_the_scripts_stops_at_the_time_limit():
    counter = _common.TestCounter(time_limit=-1.0)
    with pytest.raises(TimeoutError):
        counter.record(0, 1, [])
    assert counter.requested == 0
