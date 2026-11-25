"""Read a config file and expand it to settings.

A config is a YAML file that describes the runs behind one table or figure. Its
keys:

    paper_item, description   text
    alpha                     nominal significance level (default 0.01)
    ci_test                   fisherz (default), gcm or oracle
    seed                      base seed of every random stream (default 0)
    graph                     generator and its parameters
    data                      mechanism, and M (one number or a list)
    graphs                    graph indices: "0-19", "1-8,10" or a list
    methods                   list of entries {name: ..., options}
    scoring, report           read when the results are scored (aggregate.py)
    repeat                    timed repetitions of every run (default 1)
    settings                  list of entries; each is merged over the keys
                              above and gives one or more settings

A list as the value of a graph parameter gives one setting per value. In an
entry of `settings`, a `graph` block with a generator, or a `data` block with a
mechanism, replaces the block; otherwise its keys are merged over the block.
The oracle goes with the data mechanism "none" and no M. An unknown key is an
error.

The generator "enumerate" stands for every time graph of N series and T steps.
Its graph index g is block g of 65536 graphs, and a run is the exhaustive check
of ITPD on a block (`itpd.experiments.ablations`).

The generator "gene_networks" reads stored data (data mechanism "stored"). Its
parameter `path` is the data folder, relative to the folder of the config file,
and its graph index is the number of a data set.
"""
import itertools
import os

import yaml

from .instances import GENERATORS
from .methods import expand


class ConfigError(ValueError):
    """A config, or an output folder, that the runner cannot use."""


PER_SETTING = ("alpha", "ci_test", "seed", "graph", "data", "graphs", "methods", "scoring", "repeat")
TOP = PER_SETTING + ("paper_item", "description", "report", "settings")
CI_TESTS = ("fisherz", "gcm", "oracle")
SCORING = ("view", "targets")
REPORT = ("metrics", "level_by", "bootstrap", "relative_to", "pool")


def _check(condition, message):
    if not condition:
        raise ConfigError(message)


def _keys(block, allowed, where):
    """Raise `ConfigError` unless `block` maps only keys in `allowed`; `where` names it."""
    _check(isinstance(block, dict), f"{where} must be a mapping")
    unknown = sorted(set(block) - set(allowed))
    _check(not unknown, f"{where}: unknown key {', '.join(map(str, unknown))}")


def indices(value) -> list:
    """Return the integers of "0-19", "1-8,10", a list of such parts, or one integer."""
    parts = value.split(",") if isinstance(value, str) else value if isinstance(value, list) else [value]
    try:
        out = []
        for part in parts:
            first, _, last = str(part).strip().partition("-")
            out += range(int(first), int(last or first) + 1)
    except ValueError:
        raise ConfigError(f"not a list of indices: {value!r}") from None
    _check(out, f"no index in {value!r}")
    return out


def _settings(merged) -> list:
    """Check one merged block and return one setting per graph parameter combination."""
    for key in ("graph", "data", "graphs", "methods"):
        _check(key in merged, f"missing key: {key}")
    ci_test, graph, data = merged["ci_test"], merged["graph"], merged["data"]
    _check(ci_test in CI_TESTS, f"ci_test must be one of {', '.join(CI_TESTS)}")
    _keys(merged["scoring"], SCORING, "scoring")
    _check(isinstance(graph, dict) and graph.get("generator") in GENERATORS,
           f"graph.generator must be one of {', '.join(GENERATORS)}")
    parameters, mechanisms = GENERATORS[graph["generator"]]
    _keys(graph, ("generator", *parameters), "graph")
    _check(all(key in graph for key in parameters), f"graph needs the keys {', '.join(parameters)}")
    _keys(data, ("mechanism", "M"), "data")
    mechanism = data.get("mechanism")
    _check(mechanism in mechanisms, f"data.mechanism must be one of {', '.join(mechanisms)}")
    _check((ci_test == "oracle") == (mechanism == "none") == ("M" not in data),
           'ci_test "oracle", data.mechanism "none" and no data.M go together; every other CI test needs data.M')
    M = data["M"] if isinstance(data.get("M"), list) else [data.get("M")]
    _check(isinstance(merged["methods"], list) and merged["methods"], "methods must be a list of entries")
    settings = []
    for values in itertools.product(*(graph[key] if isinstance(graph[key], list) else [graph[key]] for key in parameters)):
        one = {"generator": graph["generator"], **dict(zip(parameters, values))}
        runs = [run for entry in merged["methods"] for run in expand(entry, one.get("lag", 1), ci_test)]
        _check(one["generator"] != "enumerate" or all(run["name"] == "ITPD" for run in runs),
               "the generator enumerate checks ITPD only")
        settings.append({**{key: merged[key] for key in ("alpha", "ci_test", "seed", "repeat")}, "graph": one,
                         "data": {"mechanism": mechanism, "M": M}, "graphs": indices(merged["graphs"]),
                         "methods": runs, "scoring": dict(merged["scoring"])})
    return settings


def load(path) -> dict:
    """Read the config file at `path` and return it with its settings expanded.

    The result has the keys `path`, `folder` (of the config file),
    `paper_item`, `description`, `report`, and `settings`. A setting is a dict
    with:
    - `alpha`, `ci_test`, `seed`, `repeat`
    - `graph`: generator and one value per parameter
    - `data`: mechanism and the list M (`[None]` without data)
    - `graphs`: list of indices
    - `methods`: one dict per run (`name`, `level`, options)
    - `scoring`"""
    with open(path) as f:
        raw = yaml.safe_load(f)
    try:
        _keys(raw, TOP, "config")
        _keys(raw.get("report", {}), REPORT, "report")
        top = {"alpha": 0.01, "ci_test": "fisherz", "seed": 0, "repeat": 1, "scoring": {},
               **{key: raw[key] for key in PER_SETTING if key in raw}}
        entries = raw.get("settings", [{}])
        _check(isinstance(entries, list) and entries, "settings must be a list of entries")
        settings = []
        for entry in entries:
            _keys(entry, PER_SETTING, "an entry of settings")
            merged = dict(top)
            for key, value in entry.items():
                merges = isinstance(value, dict) and isinstance(top.get(key), dict)
                if merges and "generator" not in value and "mechanism" not in value:
                    value = {**top[key], **value}
                merged[key] = value
            settings += _settings(merged)
    except ValueError as error:
        raise ConfigError(f"{path}: {error}") from None
    path = os.path.abspath(path)
    return {"path": path, "folder": os.path.dirname(path), "paper_item": raw.get("paper_item"),
            "description": raw.get("description"), "report": raw.get("report", {}), "settings": settings}

