"""Nonlinear instances (itpd.nonlinear), GCM bookkeeping, the nonlinear driver on a tiny cell."""
import json
import os

import numpy as np

from itpd import ci, nonlinear, instances
from itpd.experiments import nonlinear_data as driver, oracle_counts


def test_nonlinear_instance_graph_pairs_with_oracle_window_and_data_reproducible(tmp_path):
    inst = nonlinear.build_instance(4, 5, 1, 2.0, 0, 3)
    gr, _ = oracle_counts.make_instance(4, 5, 1, 2.0, 0, 3, "window")
    assert instances.graph_hash(inst["A"]) == instances.graph_hash(gr.A)
    X = nonlinear.data(inst, 200)
    assert X.shape == (200, 5, 4) and np.isfinite(X).all()
    assert np.array_equal(X, nonlinear.data(inst, 400)[:200])           # prefixes: paired across M
    path = str(tmp_path / "i.npz")
    nonlinear.save(path, inst)
    inst2 = nonlinear.load_nonlinear_instance(path)
    assert nonlinear.data_hash(nonlinear.data(inst2, nonlinear.M_MAX)) == inst2["data_sha1"]
    assert np.array_equal(nonlinear.data(inst2, 200), X)
    assert inst["meta"]["guard_ok"]


def test_nonlinear_data_is_nonlinear():
    # a true edge with a sine-dominated function has near-zero linear correlation possible; here just check bounded edge effect
    inst = nonlinear.build_instance(4, 4, 1, 2.0, 1, 0)
    X = nonlinear.data(inst, 2000).reshape(2000, -1)
    assert np.abs(X).max() < 60


def test_gcm_counts_by_size():
    rng = np.random.default_rng(0)
    D = rng.standard_normal((300, 4))
    g = ci.GCM(D)
    g(0, 1, []); g(0, 1, [2]); g(0, 1, [2, 3])
    assert g.calls_by_size == {0: 1, 1: 1, 2: 1} and g.fits == 8       # 2 targets x 2 folds x 2 non-empty tests


def test_nonlinear_driver_tiny_cell(tmp_path):
    out = str(tmp_path)
    driver.main(["--out-dir", out, "--N", "3", "--T", "4", "--M", "150", "--graphs", "2", "--alphas", "0.01,0.05"])
    fs = sorted(os.listdir(os.path.join(out, "nonlinear_N3_T4_tau1_d2")))
    assert len(fs) == 2
    r = json.load(open(os.path.join(out, "nonlinear_N3_T4_tau1_d2", fs[0])))
    names = {(x["name"], x["alpha"]) for x in r["runs"]}
    assert names == {(m, a) for m in ("itpd_naive", "itpd", "full_conditioning") for a in (0.01, 0.05)}
    assert r["common_tmax"] == 3 and all(x["status"] == "ok" for x in r["runs"])
    assert sum(r["gcm_calls_by_size"].values()) > 0
    driver.main(["--out-dir", out, "--N", "3", "--T", "4", "--M", "150", "--graphs", "2", "--alphas", "0.01,0.05"])
