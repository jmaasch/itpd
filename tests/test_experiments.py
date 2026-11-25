"""Tests of itpd.experiments: the config reader, the runner and its result files, and the table of means."""
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from itpd import ITPD, CICounter, FisherZ, metrics
from itpd.experiments import run, table
from itpd.experiments.aggregate import cells, rows
from itpd.experiments.__main__ import main
from itpd.experiments.config import ConfigError, indices, load, resolve
from itpd.experiments.methods import run_method
from itpd.experiments.runner import run_path, specifications
from itpd.external import run_external
from itpd.simulation import linear, nonlinear, redrawn_edges, stepwise

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
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
    return json.loads((Path(out) / setting / f"g{g:02d}" / f"M{M}" / ci_test / f"{name}.json").read_text())


@pytest.fixture
def no_interpreters(monkeypatch):
    for name in ("ITPD_PYTHON_PCMCI", "ITPD_PYTHON_DFPC", "ITPD_PYTHON_DYNOTEARS", "ITPD_PYTHON_CDNOD"):
        monkeypatch.delenv(name, raising=False)


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
    # An entry with switches has its own file name.
    ({"methods": [{"name": "ITPD", "switches": {}}]}, "label"),
    ({"methods": [{"name": "ITPD", "switches": {}, "label": "ITPD"}]}, "label"),
    ({"methods": [{"name": "ITPD", "label": "plain"}]}, "label"),
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
    # PCMCI searches up to the lag of the graph unless the entry gives max_lag.
    assert first["methods"][-1] == {"name": "PCMCI", "max_lag": 2, "level": "corrected"}
    assert oracle["graph"]["N"] == 2 and oracle["data"] == {"mechanism": "none", "M": [None]} and oracle["graphs"] == [0, 1]
    spec = specifications({"settings": [oracle]})[0]
    assert run_path(spec) == str(Path("standard_N2_T3_lag1_in_degree0.5_stationary", "g00", "oracle", "ITPD.json"))
    assert indices("0-2,7") == [0, 1, 2, 7] and indices(4) == [4]


def test_paths_resolve_against_the_folder_of_the_config(tmp_path, monkeypatch):
    config = load(write(tmp_path / "somewhere"))
    monkeypatch.chdir(tmp_path)
    assert resolve(config, "data/x.csv") == str(tmp_path / "somewhere" / "data" / "x.csv")
    assert resolve(config, "../data") == str(tmp_path / "data")


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
    """Configs that describe the same run give it the same file and the same specification."""
    def runs(name, keep=lambda spec: True):
        return {run_path(spec): spec for spec in specifications(load(CONFIGS / f"{name}.yaml")) if keep(spec)}
    f1 = runs("main_setting_f1")
    for name in ("main_setting_tests", "frontier_tests", "frontier_seconds"):
        assert runs(name) and all(f1.get(path) == spec for path, spec in runs(name).items())
    # Sample size: with 20 series and 200, 500 or 2000 observations, the runs are runs of the main setting.
    shared = runs("sample_size", lambda spec: spec["graph"]["N"] == 20 and spec["data"]["M"] >= 200)
    assert len(shared) == 10 * 3 * 13 and all(f1.get(path) == spec for path, spec in shared.items())
    # Scaling of seconds: the methods that are not timed there are runs of the main setting and of the scalability table.
    others = runs("scaling_time", lambda spec: spec["method"]["name"] not in ("ITPD", "IAMB", "Lasso"))
    known = {**f1, **runs("scalability")}
    assert len(others) == 20 * 4 + 2 * 5 * 2 and all(known.get(path) == spec for path, spec in others.items())


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
        assert {r["metric"] for r in expected} <= set(config["report"]["metrics"])
        assert all(float(r["mean"]) >= 0 and r["better_level"] in ("True", "False") and int(r["graphs"]) for r in expected)


ONE_METRIC = {"settings": [{"scoring": {}}], "report": {"metrics": ["f1"]}}


def cell_rows(f1, method="IAMB", level=None) -> list:
    """Rows of one cell: one per graph with the F1 values f1; None is a run without a result."""
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
    assert text[4].split() == ["M", "method", "level", "graphs", "mean", "minus", "ITPD"]     # No bootstrap: no lo, hi, verdict.
    assert [line.split()[1] for line in text[5:11]] == ["ITPD", "IAMB", "IPaDL", "FullCI", "Lasso", "PCMCI,"]
    assert text[10].split()[4:8] == ["1", "corrected", "0", "-"] and "unique_per_candidate" in text
    main(["table", config, "--runs", str(out)])
    assert capsys.readouterr().out.splitlines()[-len(text):] == text

    # Expected means: configs/expected/<config>.csv beside the config.
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


# The command line.

def test_command_line(tmp_path, no_interpreters):
    config, out = write(tmp_path), tmp_path / "out"
    command = [sys.executable, "-B", "-m", "itpd.experiments", "run", config, "--out", str(out),
               "--graphs", "0", "--methods", "ITPD,Lasso", "--M", "100"]
    done = subprocess.run(command, capture_output=True, text=True, cwd=tmp_path)
    assert done.returncode == 0 and done.stdout.splitlines()[-1] == "2 runs written, 0 finished runs skipped"
    assert sorted(path.name for path in out.rglob("*.json")) == ["ITPD.json", "Lasso.json"]
    main(["run", config, "--out", str(out), "--graphs", "0-1", "--methods", "ITPD", "--M", "100"])
    assert len(list(out.rglob("ITPD.json"))) == 2
    # Another specification in the same folder: an error message and the exit code 1, no traceback.
    done = subprocess.run([*command[:5], write(tmp_path, "changed.yaml", alpha=0.05), *command[6:]],
                          capture_output=True, text=True, cwd=tmp_path)
    assert done.returncode == 1 and done.stderr.startswith("error: ") and "another specification" in done.stderr
    assert "Traceback" not in done.stderr


# The external baselines.

def test_missing_interpreter_gives_skipped(no_interpreters):
    assert run_external("PCMCI", np.zeros((5, 4, 2)), alpha=0.01)["status"] == "skipped"
    instance = linear.make_instance(3, 4, 1, 1, 0, stationary=False)
    for method in ({"name": "PCMCI", "level": "alpha", "max_lag": 1}, {"name": "DF-PC", "level": "corrected", "time_limit": 5},
                   {"name": "DYNOTEARS", "max_lag": 1, "unit_variance": True}, {"name": "CD-NOD", "level": "corrected", "max_lag": 1}):
        result = run_method(method, instance, instance.sample(50), "fisherz", 0.01)
        assert result["status"] == "skipped" and result["edges"] is None and result["untested"] == []
