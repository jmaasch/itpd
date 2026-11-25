"""Tests of `itpd.experiments`.

Covers the config reader, the runner, the table of means, the scores, the
ablations, and the command line."""
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from itpd import ITPD, CICounter, FisherZ, metrics
from itpd.experiments import gene_networks, run, scores, table
from itpd.experiments.ablations import ITPDAblation, Switches, exhaustive_check, extended_bic_level
from itpd.experiments.aggregate import cells, columns, rows
from itpd.experiments.__main__ import main
from itpd.experiments.config import ConfigError, indices, load
from itpd.experiments.runner import run_path, specifications
from itpd.graphs import graph_hash
from itpd.simulation import Instance, linear, nonlinear, redrawn_edges, stepwise, violations

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
GENE_DATA = CONFIGS.parent / "data" / "gene_networks"
SETTING = "standard_N3_T4_lag1_in_degree1_nonstationary"

# Three series, four steps. With 9 observations the last step cannot be tested given all earlier nodes.
SMALL = {
    "alpha": 0.01,
    "graph": {"generator": "standard", "N": 3, "T": 4, "lag": 1, "in_degree": 1, "stationary": False},
    "data": {"mechanism": "linear_gaussian", "M": [9, 100]},
    "graphs": "0-1",
    "methods": [{"name": "ITPD"}, {"name": "IAMB", "levels": ["alpha", "corrected"]}, {"name": "IPaDL"},
                {"name": "FullCI", "levels": ["corrected"]}, {"name": "Lasso"}, {"name": "PCMCI", "levels": ["corrected"]}],
}


def write(folder, name="small.yaml", **changes) -> str:
    folder.mkdir(exist_ok=True)
    path = folder / name
    path.write_text(yaml.safe_dump({**SMALL, **changes}))
    return str(path)


def read(out, name, g=0, M=100, ci_test="fisherz", setting=SETTING) -> dict:
    folder = Path(out) / setting / f"g{g:02d}" / (f"M{M}" if M else "")
    return json.loads((folder / ci_test / f"{name}.json").read_text())


# The config reader.

def test_every_config_loads():
    paths = sorted(CONFIGS.glob("*.yaml"))
    assert paths
    for path in paths:
        config = load(path)
        assert config["settings"] and config["paper_item"] and config["folder"] == str(CONFIGS)


def test_main_setting_config():
    config = load(CONFIGS / "main_setting_f1.yaml")
    assert [setting["graph"]["in_degree"] for setting in config["settings"]] == [1, 2, 3, 5]
    setting = config["settings"][1]
    assert setting["graph"] == {"generator": "standard", "N": 20, "T": 16, "lag": 1, "in_degree": 2, "stationary": False}
    assert setting["data"] == {"mechanism": "linear_gaussian", "M": [200, 500, 2000]}
    assert setting["graphs"] == list(range(20)) and (setting["alpha"], setting["ci_test"], setting["seed"]) == (0.01, "fisherz", 0)
    files = [Path(run_path(spec)).name for spec in specifications(config, graphs=[0], M=[200]) if spec["graph"]["in_degree"] == 2]
    assert files == ["ITPD.json", "IAMB__alpha.json", "IAMB__corrected.json", "IPaDL__alpha.json", "IPaDL__corrected.json",
                     "FullCI__alpha.json", "FullCI__corrected.json", "Lasso.json", "PCMCI__alpha__lag1.json",
                     "PCMCI__corrected__lag1.json", "DF-PC__alpha.json", "DF-PC__corrected.json", "DYNOTEARS__lag1.json",
                     "CD-NOD__alpha__lag1.json", "CD-NOD__corrected__lag1.json"]
    assert len(specifications(config)) == 4 * 20 * 3 * 15


@pytest.mark.parametrize("changes, message", [
    ({"colour": 1}, "unknown key colour"),
    ({"graph": {**SMALL["graph"], "degree": 2}}, "unknown key degree"),
    ({"graph": {"generator": "standard", "N": 3}}, "graph needs the keys"),
    ({"graph": {**SMALL["graph"], "generator": "ring"}}, "graph.generator"),
    ({"data": {"mechanism": "uniform", "M": 50}}, "data.mechanism"),
    ({"methods": [{"name": "ITPD", "levels": ["alpha"]}]}, "unknown key levels"),
    ({"methods": [{"name": "PC"}]}, "unknown method"),
    ({"methods": [{"name": "IAMB", "levels": ["strict"]}]}, "levels"),
    ({"report": {"plot": True}}, "unknown key plot"),
    ({"scoring": {"rule": 1}}, "unknown key rule"),
    ({"settings": [{"paper_item": "x"}]}, "unknown key paper_item"),
    ({"ci_test": "oracle"}, "go together"),
    ({"ci_test": "oracle", "data": {"mechanism": "none"}, "methods": [{"name": "Lasso"}]}, "needs data"),
    # PCMCI, DF-PC and CD-NOD have their own Fisher-z test.
    ({"ci_test": "gcm", "methods": [{"name": "PCMCI"}]}, "its own Fisher-z test"),
    ({"data": {"mechanism": "linear_gaussian"}}, "data.M"),
    ({"graphs": "a-b"}, "indices"),
    # An entry with `switches` has its own file name.
    ({"methods": [{"name": "ITPD", "switches": {}}]}, "label"),
    ({"methods": [{"name": "ITPD", "switches": {}, "label": "ITPD"}]}, "label"),
    ({"methods": [{"name": "ITPD", "label": "plain"}]}, "label"),
    ({"methods": [{"name": "ITPD", "gamma": 0.1}]}, "label"),
    ({"methods": [{"name": "ITPD", "label": "level", "gamma": 2}]}, "gamma"),
    ({"methods": [{"name": "ITPD", "label": "changed", "switches": {"line12": "X"}}]}, "line12"),
    # YAML reads an unquoted `off` as `False`.
    ({"methods": [{"name": "ITPD", "label": "changed", "switches": {"line9": False}}]}, "line9"),
    ({"ci_test": "oracle", "data": {"mechanism": "none"},
      "methods": [{"name": "ITPD", "label": "level", "switches": {"phase2_level": "extended_bic"}}]}, "number of observations"),
    ({"ci_test": "oracle", "data": {"mechanism": "none"}, "methods": [{"name": "IAMB"}],
      "graph": {"generator": "enumerate", "N": 2, "T": 3, "latent_first_step": False}}, "ITPD only"),
])
def test_invalid_config_is_an_error(tmp_path, changes, message):
    with pytest.raises(ConfigError, match=message):
        load(write(tmp_path, **changes))


def test_settings_merge_over_the_top_level(tmp_path):
    entries = [{"graph": {"in_degree": [1, 2], "lag": 2}, "graphs": [3, "5-6"]},
               {"ci_test": "oracle", "data": {"mechanism": "none"}, "methods": [{"name": "ITPD"}],
                "graph": {"generator": "standard", "N": 2, "T": 3, "lag": 1, "in_degree": 0.5, "stationary": True}}]
    first, second, oracle = load(write(tmp_path, settings=entries))["settings"]
    assert first["graph"] == {**SMALL["graph"], "lag": 2} and second["graph"] == {**SMALL["graph"], "lag": 2, "in_degree": 2}
    assert first["graphs"] == [3, 5, 6] and first["data"]["M"] == [9, 100]
    # PCMCI searches up to the lag of the graph unless the entry gives
    # `max_lag`.
    assert first["methods"][-1] == {"name": "PCMCI", "max_lag": 2, "level": "corrected"}
    assert oracle["graph"]["N"] == 2 and oracle["data"] == {"mechanism": "none", "M": [None]} and oracle["graphs"] == [0, 1]
    spec = specifications({"settings": [oracle]})[0]
    assert run_path(spec) == str(Path("standard_N2_T3_lag1_in_degree0.5_stationary", "g00", "oracle", "ITPD.json"))
    assert indices("0-2,7") == [0, 1, 2, 7] and indices(4) == [4]


def test_gene_network_data(tmp_path, monkeypatch, no_interpreters):
    """Test stored gene-network data.

    The path is relative to the folder of the config, and every M has its own
    centered array."""
    folder = tmp_path / "somewhere"
    folder.mkdir()
    (folder / "data").symlink_to(GENE_DATA, target_is_directory=True)
    graph = {"generator": "gene_networks", "network": 3, "path": "data"}
    config = write(folder, graph=graph, data={"mechanism": "stored", "M": [30, 40]}, graphs=[0], methods=[{"name": "Lasso"}],
                   scoring={"view": "regulatory"})
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out"
    assert run(config, out) == {"written": 2, "skipped": 0}

    instance = gene_networks.load(3, 0, GENE_DATA)
    X = instance.sample(200)
    assert X.shape == (200, 40, 20) and float(np.abs(X.mean(axis=0)).max()) < 1e-12
    assert linear.data_hash(X) == "869a7968ba9795e125170e91687f990a004f2ddf"
    for M in (30, 40):
        stored = read(out, "Lasso", M=M, setting="gene_networks_network3")
        assert stored["spec"]["graph"] == graph and stored["graph_sha1"] == instance.graph_sha1 and stored["status"] == "ok"
        assert stored["data_sha1"] == linear.data_hash(instance.sample(M))
    # The table needs the truth tables only.
    scored = rows(load(config), out)
    assert [row["scores"]["regulatory"]["common"]["tp"] > 0 for row in scored] == [True, True]
    # The repository holds 200 observations of data set 0.
    with pytest.raises(FileNotFoundError, match="net3_g0.npy"):
        instance.sample(201)
    more = write(folder, "more.yaml", graph=graph, data={"mechanism": "stored", "M": 30}, graphs=[1], methods=[{"name": "Lasso"}])
    with pytest.raises(ConfigError, match="net3_g1.npy.*README.md"):
        run(more, out)


# The runner.

def test_run_writes_one_file_per_run(tmp_path, no_interpreters):
    config, out = write(tmp_path), tmp_path / "out"
    assert run(config, out) == {"written": 28, "skipped": 0}
    files = sorted(path.relative_to(out) for path in out.rglob("*.json"))
    assert len(files) == 28 and not list(out.rglob("*.tmp"))
    assert [path.name for path in files[:7]] == ["FullCI__corrected.json", "IAMB__alpha.json", "IAMB__corrected.json",
                                                 "IPaDL__alpha.json", "ITPD.json", "Lasso.json", "PCMCI__corrected__lag1.json"]

    instance = linear.make_instance(3, 4, 1, 1, 0, stationary=False)
    X = instance.sample(2000)
    itpd = read(out, "ITPD")
    assert itpd["spec"] == {"graph": SMALL["graph"], "graph_index": 0, "seed": 0, "data": {"mechanism": "linear_gaussian", "M": 100},
                            "ci_test": "fisherz", "alpha": 0.01, "method": {"name": "ITPD"}, "repeat": 1}
    assert itpd["graph_sha1"] == instance.graph_sha1 and itpd["data_sha1"] == linear.data_hash(X)
    # The file holds the estimate and the counts of a direct call on the first 100 observations.
    counter = CICounter(FisherZ(X[:100]))
    model = ITPD(counter, alpha=0.01).fit(X[:100])
    assert itpd["edges"] == np.flatnonzero(model.get_adjacency_matrix()).tolist() and itpd["status"] == "ok"
    assert {key: itpd[key] for key in counter.summary(3, 4)} == counter.summary(3, 4)
    assert len(itpd["seconds"]) == 1 and itpd["versions"]["numpy"] == np.__version__
    assert read(out, "IAMB__corrected")["spec"]["method"] == {"name": "IAMB", "level": "corrected"}
    assert read(out, "Lasso")["edges"] and "requested" not in read(out, "Lasso")
    # FullCI cannot test the targets of the last step with 9 observations; they are untested and have no edge.
    full = read(out, "FullCI__corrected", M=9)
    assert full["status"] == "partial" and full["untested"] == [9, 10, 11]
    assert all(edge % 12 < 9 for edge in full["edges"])
    # No interpreter is given for PCMCI.
    pcmci = read(out, "PCMCI__corrected__lag1")
    assert pcmci["status"] == "skipped" and pcmci["edges"] is None and "ITPD_PYTHON_PCMCI" in pcmci["message"]


def test_second_run_skips_finished_runs(tmp_path, no_interpreters):
    config, out = write(tmp_path), tmp_path / "out"
    run(config, out)
    before = {path: path.read_bytes() for path in out.rglob("*.json")}
    # Only the 4 runs of PCMCI, which have no interpreter, are tried again. A finished run is not made again:
    # its file, which holds the seconds of the run, stays as it is.
    assert run(config, out) == {"written": 4, "skipped": 24}
    assert {path: path.read_bytes() for path in out.rglob("*.json")} == before
    # A further config can share the folder: equal runs share their file.
    more = write(tmp_path, "more.yaml", graphs="0-2", methods=[{"name": "ITPD"}])
    assert run(more, out) == {"written": 2, "skipped": 4}


@pytest.mark.parametrize("changes", [{"alpha": 0.05}, {"seed": 1}, {"repeat": 2}])
def test_changed_specification_stops_the_runner(tmp_path, changes):
    out = tmp_path / "out"
    methods = [{"name": "ITPD"}, {"name": "Lasso"}]
    run(write(tmp_path, methods=methods), out)
    before = {path: path.read_bytes() for path in out.rglob("*.json")}
    # The folder has the 8 runs; graph 2 is not in it. The runner stops before it runs anything.
    changed = write(tmp_path, "changed.yaml", methods=methods, graphs="0-2", **changes)
    with pytest.raises(ConfigError, match=f"another specification .*{next(iter(changes))}"):
        run(changed, out)
    assert {path: path.read_bytes() for path in out.rglob("*.json")} == before


def test_repetitions_and_selection(tmp_path, capsys):
    config, out = write(tmp_path, repeat=2), tmp_path / "out"
    assert run(config, out, graphs=[1], methods=["ITPD", "IPaDL"], M=[100]) == {"written": 2, "skipped": 0}
    assert sorted(path.name for path in out.rglob("*.json")) == ["IPaDL__alpha.json", "ITPD.json"]
    itpd = read(out, "ITPD", g=1)
    assert itpd["spec"]["repeat"] == 2 and len(itpd["seconds"]) == 2 and all(s > 0 for s in itpd["seconds"])
    with pytest.raises(ConfigError, match="no method PC"):
        run(config, out, methods=["PC"])
    with pytest.raises(ConfigError, match="no run"):
        run(config, out, graphs=[7])
    assert "2 runs written, 0 finished runs skipped" in capsys.readouterr().out


def test_oracle_runs_recover_the_graph(tmp_path):
    graph = {"generator": "standard", "N": 3, "T": 5, "lag": 2, "in_degree": 1, "stationary": True}
    config = write(tmp_path, ci_test="oracle", graph=graph, data={"mechanism": "none"}, graphs=[0],
                   methods=[{"name": "ITPD"}, {"name": "IAMB"}, {"name": "IPaDL"}, {"name": "FullCI"}])
    out = tmp_path / "out"
    assert run(config, out) == {"written": 4, "skipped": 0}
    instance = linear.make_instance(3, 5, 2, 1, 0, stationary=True, weights=False)
    folder = out / "standard_N3_T5_lag2_in_degree1_stationary" / "g00" / "oracle"
    for name in ("ITPD", "IAMB__alpha", "IPaDL__alpha", "FullCI__alpha"):
        result = json.loads((folder / f"{name}.json").read_text())
        assert result["edges"] == np.flatnonzero(instance.A).tolist() and result["status"] == "ok"
        assert result["graph_sha1"] == instance.graph_sha1 and result["data_sha1"] is None and result["requested"] > 0


@pytest.mark.parametrize("graph, mechanism, ci_test, make", [
    ({"generator": "stepwise", "N": 3, "T": 4, "in_degree": 1}, "linear_gaussian", "fisherz",
     lambda g: stepwise.make_instance(3, 4, 1, g)),
    ({"generator": "redrawn_edges", "N": 3, "T": 4, "in_degree": 1, "share": 0.5}, "linear_gaussian", "fisherz",
     lambda g: redrawn_edges.make_instance(3, 4, 1, 0.5, g)),
    ({"generator": "standard", "N": 2, "T": 3, "lag": 1, "in_degree": 1, "stationary": False}, "nonlinear", "gcm",
     lambda g: nonlinear.make_instance(2, 3, 1, 1, g, stationary=False)),
    ({"generator": "violations", "N": 3, "T": 4, "in_degree": 1, "no_self_edge_share": 0, "same_step_share": 0.5}, "linear_gaussian",
     "fisherz", lambda g: violations.make_instance(1, g, same_step_share=0.5, N=3, T=4)),
])
def test_other_generators_and_the_gcm_test(tmp_path, no_interpreters, graph, mechanism, ci_test, make):
    config = write(tmp_path, graph=graph, data={"mechanism": mechanism, "M": 60}, ci_test=ci_test, graphs=[1],
                   methods=[{"name": "ITPD"}, {"name": "Lasso"}, {"name": "DYNOTEARS"}])
    out = tmp_path / "out"
    assert run(config, out) == {"written": 3, "skipped": 0}
    instance = make(1)
    scored = {row["method"]: row for row in rows(load(config), out)}
    for name in ("ITPD", "Lasso"):
        stored = scored[name]["run"]
        assert stored["graph_sha1"] == instance.graph_sha1 and stored["data_sha1"] == linear.data_hash(instance.sample(2000))
        assert stored["spec"]["graph"] == graph and stored["spec"]["ci_test"] == ci_test and stored["status"] == "ok"
    assert scored["ITPD"]["run"]["requested"] > 0 and scored["DYNOTEARS, max lag 1"]["status"] == "skipped"
    # The folder of a setting is the generator and its parameters.
    words = [graph["generator"]] + [("" if v else "non") + k if isinstance(v, bool) else f"{k}{v}" for k, v in graph.items() if k != "generator"]
    assert (out / "_".join(words) / "g01" / "M60" / ci_test / "ITPD.json").exists()


# The table of means.

def test_configs_share_runs():
    """Configs for the same run share its file and specification."""
    def runs(name, keep=lambda spec: True):
        return {run_path(spec): spec for spec in specifications(load(CONFIGS / f"{name}.yaml")) if keep(spec)}
    f1 = runs("main_setting_f1")
    for name in ("main_setting_tests", "frontier_tests", "frontier_seconds"):
        assert runs(name) and all(f1.get(path) == spec for path, spec in runs(name).items())
    # `sample_size`: with 20 series and 200, 500, or 2000 observations, the
    # runs are runs of the main setting.
    shared = runs("sample_size", lambda spec: spec["graph"]["N"] == 20 and spec["data"]["M"] >= 200)
    assert len(shared) == 10 * 3 * 13 and all(f1.get(path) == spec for path, spec in shared.items())
    # Scaling of seconds: the methods that are not timed there are runs of the main setting and of the scalability table.
    others = runs("scaling_time", lambda spec: spec["method"]["name"] not in ("ITPD", "IAMB", "Lasso"))
    known = {**f1, **runs("scalability")}
    assert len(others) == 20 * 4 + 2 * 5 * 2 and all(known.get(path) == spec for path, spec in others.items())
    # The runs of ITPD as Algorithm 1 in the tables on levels are runs of the main setting.
    for name in ("extended_bic", "gamma_kappa"):
        shared = runs(name, lambda spec: spec["method"] == {"name": "ITPD"} and spec["graph"] in (f1[path]["graph"] for path in f1))
        assert len(shared) >= 2 * 3 * 5 and all(f1.get(path) == spec for path, spec in shared.items())
    # Gene networks: the two tables score the same runs, and the table on levels has runs of ITPD among them.
    genes = runs("gene_networks")
    assert runs("gene_networks_indirect") == genes
    shared = runs("extended_bic", lambda spec: spec["method"] == {"name": "ITPD"} and spec["graph"]["generator"] == "gene_networks")
    assert len(shared) == 2 * 3 * 3 and all(genes.get(path) == spec for path, spec in shared.items())
    # Dropped steps: Algorithm 1 and the changed Line 11 are runs of the ablation table.
    ablation = runs("ablation")
    shared = runs("dropped_steps", lambda spec: "9'" not in spec["method"]["label"])
    assert len(shared) == 12 * 20 * 2 and all(ablation.get(path) == spec for path, spec in shared.items())


def test_a_changed_run_never_has_the_file_of_plain_itpd():
    plain, changed = set(), set()
    for path in sorted(CONFIGS.glob("*.yaml")):
        for spec in specifications(load(path)):
            if spec["method"]["name"] == "ITPD":
                (changed if set(spec["method"]) - {"name"} else plain).add(run_path(spec))
    assert plain and changed and not plain & changed and all(Path(path).name != "ITPD.json" for path in changed)


def test_every_config_has_a_table_and_expected_means(tmp_path):
    for path in sorted(CONFIGS.glob("*.yaml")):
        config = load(path)
        planned = rows(config, tmp_path)
        text = table(path, tmp_path, compare=True).splitlines()
        assert text[0] == config["paper_item"] and text[1] == f"{len(planned)} runs: {len(planned)} missing"
        assert text[-1].startswith("0 means compared")
        # Every expected mean belongs to one cell, method, level and metric of the config.
        known = {(row["setting"], str(row["M"] or ""), row["method"], row["level"] or "") for row in planned}
        with open(CONFIGS / "expected" / f"{path.stem}.csv", newline="") as f:
            expected = list(csv.DictReader(line for line in f if line[0] != "#"))
        assert expected and all((r["setting"], r["M"], r["method"], r["level"]) in known for r in expected)
        assert {r["metric"] for r in expected} <= set(columns(config))
        assert all(float(r["mean"]) >= 0 and r["better_level"] in ("True", "False") and int(r["graphs"]) for r in expected)


ONE_METRIC = {"settings": [{"scoring": {}}], "report": {"metrics": ["f1"]}}


def cell_rows(f1, method="IAMB", level=None) -> list:
    """Build the rows of one cell.

    Returns one row per graph with the F1 values `f1`; `None` is a run without
    a result."""
    return [{"setting": "s", "graph": {}, "M": 100, "graph_index": g, "method": method, "level": level, "status": "ok",
             **({} if value is None else {"values": {"f1": value, "level_by": value}})} for g, value in enumerate(f1)]


@pytest.mark.parametrize("alpha, corrected, better", [
    ([0.5, 0.7], [0.6, 0.6], ["corrected"]),              # A tie goes to the corrected level.
    ([0.5, 0.7 + 1e-13], [0.6, 0.6], ["corrected"]),      # The means are rounded to 12 digits.
    ([0.5, 0.7 + 1e-9], [0.6, 0.6], ["alpha"]),
    ([0.6, None], [0.1, 0.9], ["alpha"]),                 # Each level has the mean of its own graphs: 0.6 and 0.5.
    ([0.1], [None], ["alpha"]),                           # A level without a result is not a candidate.
    ([None], [None], []),
])
def test_better_level(alpha, corrected, better):
    groups = cells(ONE_METRIC, cell_rows(alpha, level="alpha") + cell_rows(corrected, level="corrected"))
    assert [group["level"] for group in groups if group["better_level"]] == better
    assert [(group["mean"]["f1"] or {}).get("graphs", 0) for group in groups] == [sum(v is not None for v in alpha), sum(v is not None for v in corrected)]


def test_bootstrap_interval_and_paired_difference():
    rng = np.random.default_rng(1)
    a = rng.random(20)
    b = a + 0.1 + 0.05 * rng.random(20)
    report = {"metrics": ["f1"], "bootstrap": {"resamples": 1000, "seed": 7}, "relative_to": "ITPD"}
    config = {"settings": [{"scoring": {}}], "report": report}
    values = [*a[:19].tolist(), None]                     # ITPD has no result on the last graph.
    reference, other = cells(config, cell_rows(values, "ITPD") + cell_rows(b.tolist(), "IAMB", "alpha"))
    index = {n: np.random.default_rng(7).integers(0, n, (1000, n)) for n in (19, 20)}
    lo, hi = (np.percentile(b[index[20]].mean(1), q) for q in (2.5, 97.5))
    assert other["mean"]["f1"] == {"mean": b.mean(), "lo": lo, "hi": hi, "graphs": 20}
    d = (b - a)[:19]                                      # The difference is on the graphs that both have.
    lo, hi = (np.percentile(d[index[19]].mean(1), q) for q in (2.5, 97.5))
    assert other["difference"]["f1"] == {"mean": d.mean(), "lo": lo, "hi": hi, "graphs": 19, "verdict": "higher"}
    assert reference["difference"] == {} and reference["mean"]["f1"]["graphs"] == 19
    del report["bootstrap"]
    assert cells(config, cell_rows([0.5, 0.7], "ITPD") + cell_rows([0.5, 0.9]))[1]["difference"]["f1"]["verdict"] is None


def test_table_of_a_small_config(tmp_path, no_interpreters, capsys):
    report = {"metrics": ["f1", "unique_per_candidate"], "relative_to": "ITPD"}
    config, out = write(tmp_path, report=report), tmp_path / "out"
    run(config, out)
    scored = rows(load(config), out)
    assert len(scored) == 28 and [row["status"] for row in scored[:7]] == ["ok"] * 4 + ["partial", "ok", "skipped"]
    # The score of a run comes from the edges of its file and the graph of its specification.
    instance = linear.make_instance(3, 4, 1, 1, 0, stationary=False, weights=False)
    for row, name, M, last in ((scored[7], "ITPD", 100, 3), (scored[4], "FullCI__corrected", 9, 2)):
        stored = read(out, name, M=M)
        A_hat = np.zeros(144, dtype=bool)
        A_hat[stored["edges"]] = True
        score = metrics.score(instance.A, A_hat.reshape(12, 12), 3, 4, M, stored["untested"])
        for targets in ("common", "all"):
            assert all(row["scores"]["edges"][targets][key] == score[targets][key] for key in ("tp", "fp", "fn", "f1"))
        assert row["scores"]["edges"]["last_common_step"] == last and row["M"] == M
        assert row["values"]["f1"] == score["common"]["f1"] == row["values"]["level_by"]
        assert row["values"]["exact"] == float(score["all"]["fp"] == 0 and score["all"]["fn"] == 0)
        assert row["values"]["unique_per_candidate"] == stored["unique"] / 45 and row["method"] == name.split("__")[0]
    assert scored[4]["scores"]["edges"]["all"]["fn"] > scored[4]["scores"]["edges"]["common"]["fn"] and scored[4]["status"] == "partial"
    assert scored[6]["method"] == "PCMCI, max lag 1" and "values" not in scored[6]

    text = table(config, out).splitlines()
    assert text[1] == "28 runs: 24 with a result, 4 without a result" and text[3] == "f1"
    assert text[4].split() == ["M", "method", "level", "graphs", "mean", "minus", "ITPD"]     # No bootstrap: no `lo`, `hi`, `verdict`.
    assert [line.split()[1] for line in text[5:11]] == ["ITPD", "IAMB", "IPaDL", "FullCI", "Lasso", "PCMCI,"]
    assert text[10].split()[4:8] == ["1", "corrected", "0", "-"] and "unique_per_candidate" in text
    main(["table", config, "--runs", str(out)])
    assert capsys.readouterr().out.splitlines()[-len(text):] == text

    # Expected means: `configs/expected/<config>.csv` beside the config.
    with pytest.raises(ConfigError, match="no expected means"):
        table(config, out, compare=True)
    mean = next(group for group in cells(load(config), scored) if group["M"] == 100)["mean"]["f1"]
    (tmp_path / "expected").mkdir()
    (tmp_path / "expected" / "small.csv").write_text(
        f"# source\nsetting,M,method,level,better_level,metric,mean,graphs\n{SETTING},100,ITPD,,True,f1,{mean['mean'] - 0.25!r},2\n")
    text = table(config, out, compare=True).splitlines()
    assert text[-1] == ("1 means compared with the expected means: largest absolute difference 2.5e-01; "
                        "0 with another number of graphs; 0 with another better level")

    # A missing file is a missing run; a file of another specification or of another graph is an error.
    path = out / SETTING / "g00" / "M100" / "fisherz" / "Lasso.json"
    stored = json.loads(path.read_text())
    path.write_text(json.dumps({**stored, "graph_sha1": "0" * 40}))
    with pytest.raises(ConfigError, match="not the graph of its specification"):
        table(config, out)
    path.unlink()
    assert table(config, out).splitlines()[1] == "28 runs: 23 with a result, 4 without a result, 1 missing"
    with pytest.raises(ConfigError, match="another specification"):
        table(write(tmp_path, "changed.yaml", alpha=0.05), out)


def test_a_run_without_common_targets_has_no_f1(tmp_path):
    # With 5 observations of 3 series no step can be tested given all earlier nodes.
    config, out = write(tmp_path, data={"mechanism": "linear_gaussian", "M": 5}, graphs=[0], methods=[{"name": "Lasso"}]), tmp_path / "out"
    run(config, out)
    row, = rows(load(config), out)
    assert row["scores"]["edges"]["common"] is None and row["values"]["f1"] is None and row["values"]["exact"] in (0.0, 1.0)
    assert table(config, out).splitlines()[-2:] == ["M  method  graphs", "5  Lasso   0"]
    all_targets = write(tmp_path, "all.yaml", data={"mechanism": "linear_gaussian", "M": 5}, graphs=[0], methods=[{"name": "Lasso"}],
                        scoring={"targets": "all"})
    assert rows(load(all_targets), out)[0]["values"]["f1"] == row["scores"]["edges"]["all"]["f1"]


# The scores of the violation tables and of the gene networks.

def counts(score, targets="all") -> tuple:
    return tuple(score[targets][key] for key in ("tp", "fp", "fn"))


def test_violation_scores():
    # Three series, three steps; node = 3 * step + series. Besides the self edges: one lag-1 edge 0 -> 4, the
    # same-step edge 5 -> 4 at step 1, and the same-step chain 6 -> 7 -> 8 at step 2.
    A = np.zeros((9, 9), dtype=bool)
    for u, v in [(0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (0, 4), (5, 4), (6, 7), (7, 8)]:
        A[u, v] = True
    truth = Instance(A, 3, 3, 1, None, graph_hash(A))
    # Markov boundary among the earlier nodes: of 4: 0, 1, and 2 (the parent of 5); of 7: 4 and 3 (parent of 6);
    # of 8: 5, 4 and 3. The self pairs 1 -> 4, 4 -> 7 and 5 -> 8 are not scored.
    boundary = scores.markov_boundary_among_earlier(A, 3)
    assert {v: np.flatnonzero(boundary[:, v]).tolist() for v in (3, 4, 7, 8)} == {3: [0], 4: [0, 1, 2], 7: [3, 4], 8: [3, 4, 5]}
    estimate = np.zeros((9, 9), dtype=bool)
    for u, v in [(0, 4), (2, 4), (3, 8), (1, 4), (0, 7)]:
        estimate[u, v] = True
    # Lag-1 edges: 0 -> 4 is the one true edge; 2 -> 4, 3 -> 8 and 0 -> 7 are false.
    assert counts(scores.lag1_edges(truth, estimate)) == (1, 3, 0)
    # Markov boundary: 0 -> 4, 2 -> 4 and 3 -> 8 are true, 0 -> 7 is false, 3 -> 7 and 4 -> 8 are missed.
    assert counts(scores.markov_boundary(truth, estimate)) == (3, 1, 2)
    # An untested target has no parents.
    assert counts(scores.markov_boundary(truth, estimate, untested=[8])) == (2, 1, 3)
    # With 6 observations the common targets are those of step 1.
    assert counts(scores.lag1_edges(truth, estimate, 6), "common") == (1, 1, 0)
    assert counts(scores.markov_boundary(truth, estimate, 6), "common") == (2, 0, 0)

    # Without an edge inside a step both truths are the graph; with such edges every lag-1 edge is in the boundary.
    plain = violations.make_instance(1, 0, N=4, T=4)
    assert np.array_equal(scores.markov_boundary_among_earlier(plain.A, 4), plain.A)
    assert scores.lag1_edges(plain, plain.A) == scores.markov_boundary(plain, plain.A) == metrics.score(plain.A, plain.A, 4, 4)
    moved = violations.make_instance(1, 0, same_step_share=0.5, N=4, T=4)
    assert moved.info["n_moved_edges"] > 0 and not (moved.info["A_lag1"] & ~scores.markov_boundary_among_earlier(moved.A, 4)).any()
    assert counts(scores.lag1_edges(moved, moved.info["A_lag1"])) == (int(moved.info["A_lag1"].sum()) - 12, 0, 0)
    assert scores.markov_boundary(moved, moved.info["A_lag1"])["all"]["fn"] > 0


def test_gene_network_scores_by_hand():
    """Check gene-network scores by hand.

    Scores twelve estimated edges on network 3 against the counts worked out by
    hand from the two truth tables."""
    truth = gene_networks.load(3, 0, GENE_DATA)
    per_step = {key: int(truth.info[key][:20, 20:40].sum()) for key in ("regulatory", "translation", "indirect")}
    assert per_step == {"regulatory": 15, "translation": 10, "indirect": 18} and int(truth.A[:20, 20:40].sum()) == 45
    other = gene_networks.load(5, 0, GENE_DATA)
    assert [int(other.info[key][:20, 20:40].sum()) for key in ("regulatory", "translation", "indirect")] == [12, 10, 32]

    def m(gene):
        return gene - 1          # series of the mRNA of a gene

    def p(gene):
        return 10 + gene - 1     # series of its protein
    edges = [(0, p(7), 1, m(1)),     # regulatory edge
             (0, p(1), 1, m(2)),     # regulatory edge
             (0, m(1), 1, p(1)),     # translation edge: not in the regulatory score
             (1, p(7), 2, m(1)),     # regulatory edge
             (1, m(7), 2, m(1)),     # indirect pair (through the protein of gene 7): false
             (1, m(2), 2, m(2)),     # self edge: never scored
             (0, p(2), 2, m(3)),     # lag 2: false
             (0, m(4), 1, m(9)),     # no edge and no indirect pair: false
             (11, p(7), 12, m(1)),   # regulatory edge at step 12, not a common target at M = 200
             (1, m(7), 2, p(1)),     # indirect pair: false
             (5, m(3), 6, m(2)),     # indirect pair: false
             (3, p(3), 4, m(4))]     # regulatory edge
    estimate = np.zeros((800, 800), dtype=bool)
    for step, source, later, target in edges:
        estimate[20 * step + source, 20 * later + target] = True
    # (regulatory, false edges without indirect pairs, every edge of the graph), on the common and on all targets.
    for untested, common, every in [((), ((4, 5, 131), 2, (5, 5, 220)), ((5, 5, 580), 2, (6, 5, 969))),
                                    # The mRNA of gene 1 at step 2 is untested: two of its edges leave the estimate.
                                    ([40], ((3, 4, 132), 2, (4, 4, 221)), ((4, 4, 581), 2, (5, 4, 970)))]:
        regulatory = gene_networks.regulatory(truth, estimate, 200, untested)
        ignored = gene_networks.regulatory_ignore_indirect(truth, estimate, 200, untested)
        graph = metrics.score(truth.A, estimate, 20, 40, 200, untested)
        assert regulatory["last_common_step"] == 9
        for targets, (expected, not_indirect, all_edges) in (("common", common), ("all", every)):
            assert counts(regulatory, targets) == expected and counts(graph, targets) == all_edges
            assert counts(ignored, targets) == (expected[0], not_indirect, expected[2])


# The ablations.

CHANGES = [{"name": "ITPD"}, {"name": "ITPD", "label": "Algorithm 1", "switches": {}},
           {"name": "ITPD", "label": "Phase II removed", "switches": {"phase2": False}},
           {"name": "ITPD", "label": "alpha1 = 0.05", "gamma": 0.05, "kappa": 1000}]


def recorded(make, V):
    """Fit `make(test)` on V with a recording Fisher-z test.

    Returns the recorded calls and the fitted model."""
    calls, test = [], FisherZ(V)

    def record(x, y, S):
        calls.append((x, y, S))
        return test(x, y, S)
    return calls, make(record).fit(V)


def test_default_switches_are_algorithm_1():
    V = linear.make_instance(4, 6, 1, 2, 0, stationary=False).sample(150)
    calls, model = recorded(ITPD, V)
    for switches in (Switches(), {}):
        same, ablation = recorded(lambda test: ITPDAblation(test, switches=switches), V)
        assert same == calls and ablation.adjacency_ == model.adjacency_ and ablation.ci_queries_ == model.ci_queries_
    assert sorted(ablation.screened_sizes_) == list(range(4, 24)) and min(ablation.screened_sizes_.values()) >= 1
    assert ITPDAblation(switches={"line9": "X"}).switches == Switches(line9="X")
    for bad in ({"line9": "T"}, {"line9": False}, {"phase2": "off"}, {"ancestor_gate": 1}):
        with pytest.raises(ValueError, match="is not one of"):
            Switches(**bad)


def test_switches_change_single_lines():
    instance = linear.make_instance(4, 6, 2, 2, 0, stationary=False)
    V, truth = instance.sample(150), instance.A.tolist()
    # These changes keep the output exact under the oracle.
    for switches in ({"candidate_order": "ascending"}, {"ancestor_gate": False}, {"marginal_test": False}, {"line9": "X"},
                     {"line9": "off"}, {"dropped_lines": True}, {"markov_test": False}, {"phase2_order": "descending"}):
        model = ITPDAblation(oracle_adjacency=instance.A, switches=switches).fit(n_variables=4)
        assert (model.get_adjacency_matrix() != 0).tolist() == truth, switches
    # Phase II removed: S is returned.
    calls, model = recorded(lambda test: ITPDAblation(test, switches={"phase2": False}), V)
    assert model.ci_queries_["II"] == 0 and all(len(model.adjacency_[y]) == size for y, size in model.screened_sizes_.items())
    # Line 11 with the members of S at the step of X: a target at the second step has all of S as parents.
    model = ITPDAblation("fisherz", switches={"line11": "same_step"}).fit(V)
    assert all(len(model.adjacency_[y]) == model.screened_sizes_[y] for y in range(4, 8))
    # Lines 9' and 10': tests of Z and X. Algorithm 1 never tests two nodes of one step.
    calls, model = recorded(lambda test: ITPDAblation(test, switches={"dropped_lines": True}), V)
    assert any(x // 4 == z // 4 and not S for z, x, S in calls)
    assert not any(y // 4 == z // 4 for z, y, S in recorded(ITPD, V)[0])
    # The levels.
    model = ITPDAblation("fisherz", switches={"phase1_level": "alpha", "phase2_level": "extended_bic"}).fit(V)
    assert model._levels(2) == (0.01, extended_bic_level(150, 11)) and ITPDAblation(switches={"phase2_level": "alpha"}).fit(V)._levels(2)[1] == 0.01
    # P(chi2_1 > log(M m)) = erfc(sqrt(log(M m) / 2)).
    assert extended_bic_level(200, 19) == pytest.approx(0.004091495235204291, rel=1e-12)


def test_exhaustive_check():
    for latent, judged in ((False, 4 * 256), (True, 128)):
        result = exhaustive_check(None, 2, 3, range(256), latent_first_step=latent)
        assert result == exhaustive_check({}, 2, 3, range(256), latent_first_step=latent)
        assert result["graphs"] == result["exact"] == 256 and result["first_failure"] is None and result["wrong_targets"] == 0
        assert result["judged_targets"] == judged and result["requested_phase1"] > result["requested_phase2"] > 0
    # Without Phase II the nonparents in S stay: the smallest failing graph has two optional edges.
    removed = exhaustive_check({"phase2": False}, 2, 3, range(256))
    assert removed["exact"] == 184 and removed["first_failure"][0] == 2 and removed["requested_phase2"] == 0
    assert exhaustive_check(Switches(phase2=False), 2, 3, [removed["first_failure"][1]])["exact"] == 0


def test_changed_runs_have_their_own_files(tmp_path):
    blocks = {"ci_test": "oracle", "data": {"mechanism": "none"}, "graphs": [0], "methods": CHANGES[:3],
              "graph": {"generator": "enumerate", "N": 2, "T": 3, "latent_first_step": [False, True]}}
    report = {"metrics": ["exact", "f1", "screened_size"], "relative_to": "Algorithm 1", "pool": {"by": "M", "cap": 100}}
    config = write(tmp_path, methods=CHANGES, graphs=[0], settings=[{}, blocks], report=report)
    out = tmp_path / "out"
    assert run(config, out) == {"written": 2 * 4 + 2 * 3, "skipped": 0}
    plain, same, removed, level = (read(out, name) for name in ("ITPD", "Algorithm-1", "Phase-II-removed", "alpha1-0-05"))
    assert [r["spec"]["method"] for r in (plain, same, removed, level)] == CHANGES
    assert all(same[key] == plain[key] for key in ("edges", "requested", "unique", "largest_set", "mean_set"))
    assert "screened_size" not in plain and "screened_size" not in level and same["screened_size"] >= 1
    assert len(removed["edges"]) == round(9 * removed["screened_size"]) > len(same["edges"])
    block = read(out, "Phase-II-removed", M=None, ci_test="oracle", setting="enumerate_N2_T3_nonlatent_first_step")
    assert block["edges"] is None and block["graphs"] == 256 > block["exact"] and block["graph_sha1"] is None
    scored = {(row["setting"], row["M"], row["method"]): row["values"] for row in rows(load(config), out)}
    assert scored[("enumerate_N2_T3_nonlatent_first_step", None, "Phase II removed")]["exact"] == block["exact"] / 256
    assert scored[(SETTING, 100, "Algorithm 1")]["screened_size"] == same["screened_size"]
    assert scored[(SETTING, 100, "ITPD")]["screened_size"] is None
    text = table(config, out).splitlines()
    assert text[1] == "14 runs: 14 with a result" and "exact: mean of the cell means by M" in text
    start = text.index("screened_size: mean of the cell means by M")
    assert text[start + 1].split() == ["M", "method", "cells", "mean", "minus", "Algorithm", "1", "higher", "tied", "lower"]
    assert [line.split()[0] for line in text[start + 2:start + 6]] == ["9", "9", ">=", ">="]
    # The same label with other switches in the same folder stops the runner.
    other = [CHANGES[0], {**CHANGES[1], "switches": {"line9": "X"}}]
    with pytest.raises(ConfigError, match="another specification"):
        run(write(tmp_path, "other.yaml", methods=other, graphs=[0]), out)


# The command line.

def test_command_line(tmp_path, capsys):
    config, out = write(tmp_path), tmp_path / "out"
    arguments = ["--out", str(out), "--graphs", "0", "--methods", "ITPD,Lasso", "--M", "100"]
    main(["run", config, *arguments])
    assert capsys.readouterr().out.splitlines()[-1] == "2 runs written, 0 finished runs skipped"
    assert sorted(path.name for path in out.rglob("*.json")) == ["ITPD.json", "Lasso.json"]
    main(["run", config, "--out", str(out), "--graphs", "0-1", "--methods", "ITPD", "--M", "100"])
    assert len(list(out.rglob("ITPD.json"))) == 2
    # Another specification in the same folder: an error message and the exit code 1, no traceback.
    command = [sys.executable, "-B", "-m", "itpd.experiments", "run", write(tmp_path, "changed.yaml", alpha=0.05), *arguments]
    done = subprocess.run(command, capture_output=True, text=True, cwd=tmp_path)
    assert done.returncode == 1 and done.stderr.startswith("error: ") and "another specification" in done.stderr
    assert "Traceback" not in done.stderr
