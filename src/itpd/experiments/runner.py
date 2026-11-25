"""Run the methods of a config and write one result file per run.

A run is one method at one level on one graph of one setting with M
observations. Its file is

    OUT/<generator and its parameters>/g<graph index>/M<M>/<ci_test>/<method>[__<level>][__lag<K>][__unit].json

(no M folder with the oracle). The file holds the specification of the run
(key "spec": graph parameters, graph index, seed, data mechanism, M, CI
test, alpha, the method with its level and options, repeat), the SHA-1 of the
true graph and of the data, what the method returned (status, edges,
untested targets, counts of CI tests), the seconds of every repetition, and
the package versions. It holds no score: a score is computed from the edges.

The name of a file does not hold the whole specification. So a file that
exists is compared with the specification of the run before anything
starts: an equal one is a finished run and is skipped, and another one stops
the runner with an error. Runs with the status "skipped" or "error" are run
again.
"""
import json
import os
import platform
from importlib.metadata import version

from threadpoolctl import threadpool_limits

from .. import __version__
from . import instances
from .config import ConfigError, load
from .methods import run_method, slug

FINISHED = ("ok", "partial", "timeout")


def run_path(spec: dict) -> str:
    """Return the path of the result file of a run, relative to the output folder."""
    graph, method, M = spec["graph"], spec["method"], spec["data"].get("M")
    words = [graph["generator"]]
    for key, value in graph.items():
        if key != "generator":
            words.append(("" if value else "non") + key if isinstance(value, bool) else f"{key}{value:g}")
    stem = slug(method.get("label", method["name"]))
    stem += "".join("__" + part for part in (method.get("level"), f"lag{method['max_lag']}" if "max_lag" in method else None,
                                             "unit" if method.get("unit_variance") else None) if part)
    folders = ["_".join(words), f"g{spec['graph_index']:02d}"] + ([f"M{M}"] if M else []) + [spec["ci_test"]]
    return os.path.join(*folders, stem + ".json")


def specifications(config: dict, graphs=None, methods=None, M=None) -> list:
    """Return the specification of every run of a config, in the order of the runs.

    graphs and M (lists of integers) and methods (list of names) keep the
    runs with these graph indices, numbers of observations and methods."""
    known = {method.get("label", method["name"]) for setting in config["settings"] for method in setting["methods"]}
    unknown = sorted(set(methods or ()) - known)
    if unknown:
        raise ConfigError(f"the config has no method {', '.join(unknown)}")
    specs = []
    for setting in config["settings"]:
        for g in setting["graphs"]:
            for m in setting["data"]["M"]:
                for method in setting["methods"]:
                    if (graphs and g not in graphs or M and m is not None and m not in M
                            or methods and method.get("label", method["name"]) not in methods):
                        continue
                    data = {"mechanism": setting["data"]["mechanism"], **({} if m is None else {"M": m})}
                    spec = {"graph": setting["graph"], "graph_index": g, "seed": setting["seed"], "data": data,
                            "ci_test": setting["ci_test"], "alpha": setting["alpha"], "method": method,
                            "repeat": setting["repeat"]}
                    specs.append(json.loads(json.dumps(spec)))
    return specs


def _finished(path: str, spec: dict) -> bool:
    """Return whether path holds a finished run of spec. Stop when it holds a run of another specification."""
    if not os.path.exists(path):
        return False
    with open(path) as f:
        stored = json.load(f)
    if stored["spec"] != spec:
        keys = sorted(key for key in set(stored["spec"]) | set(spec) if stored["spec"].get(key) != spec.get(key))
        raise ConfigError(f"{path} holds a run with another specification (it differs in: {', '.join(keys)}). "
                          "Use another output folder.")
    return stored["status"] in FINISHED


def run(config, out, *, graphs=None, methods=None, M=None) -> dict:
    """Run every run of a config that the folder out does not hold, and return the numbers of runs.

    Parameters
    ----------
    config : str or path
        The config file.
    out : str or path
        The output folder. Several configs can share it: equal runs share a file.
    graphs, methods, M : list or None
        Keep only these graph indices, methods and numbers of observations.

    Returns
    -------
    dict
        written: number of runs made; skipped: number of finished runs found.
    """
    planned = {}
    for spec in specifications(load(config), graphs, methods, M):
        path = os.path.join(out, run_path(spec))
        if planned.setdefault(path, spec) != spec:
            raise ConfigError(f"two runs of the config have the file {path}")
    if not planned:
        raise ConfigError("no run of the config matches the selection")
    pending = [(path, spec) for path, spec in planned.items() if not _finished(path, spec)]
    versions = {"python": platform.python_version(), "itpd": __version__,
                **{name: version(name) for name in ("numpy", "scipy", "scikit-learn", "networkx")}}
    built = draws = None
    # One BLAS thread: the simulated data and the counts must not depend on the number of cores.
    with threadpool_limits(limits=1):
        for path, spec in pending:
            key = (spec["graph"], spec["data"]["mechanism"], spec["seed"], spec["graph_index"])
            if key != built:
                instance, built, draws = instances.make_instance(*key), key, {}
            V = data_sha1 = None
            if "M" in spec["data"]:
                rows = max(instances.ROWS, spec["data"]["M"])
                if rows not in draws:
                    draws[rows] = instances.draw(instance, rows)
                V, data_sha1 = draws[rows][0][:spec["data"]["M"]], draws[rows][1]
            results = [run_method(spec["method"], instance, V, spec["ci_test"], spec["alpha"]) for _ in range(spec["repeat"])]
            record = {"spec": spec, "graph_sha1": instance.graph_sha1, "data_sha1": data_sha1, **results[0],
                      "seconds": [result.get("seconds") for result in results]}
            record["versions"] = record.get("versions") or versions
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path + ".tmp", "w") as f:
                json.dump(record, f)
            os.replace(path + ".tmp", path)
            print(f"{os.path.relpath(path, out)}: {record['status']}", flush=True)
    summary = {"written": len(pending), "skipped": len(planned) - len(pending)}
    print(f"{summary['written']} runs written, {summary['skipped']} finished runs skipped")
    return summary
