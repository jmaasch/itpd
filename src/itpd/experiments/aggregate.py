"""Score the runs of a config and print the table of means.

    python -m itpd.experiments table CONFIG --runs DIR [--compare]

A run file holds the estimated edges. The score of a run is computed here from
the edges and the true graph, which is built again from the specification of
the run and must have the graph hash of the run file.

Keys of a config that are read here:

    scoring.view      a view, or a list of views. "edges": every edge of
                      the true graph (itpd.metrics.score). "lag1_edges" and
                      "markov_boundary": itpd.experiments.scores.
                      "regulatory" and "regulatory_ignore_indirect":
                      itpd.experiments.gene_networks. With a list, a score
                      has one column per view, such as "f1 (lag1_edges)".
    scoring.targets   "common" (default) or "all": the targets of f1,
                      precision and recall. With the oracle and the GCM test
                      every target is a common target.
    report.metrics    list from f1, precision, recall, exact,
                      unique_per_candidate, mean_set, largest_set,
                      screened_size (mean |S| before Phase II, for ITPD
                      with switches), seconds
    report.level_by   {view, targets}: the F1 that chooses the level
                      (default: the first view on scoring.targets)
    report.bootstrap  {resamples, seed}: 95% percentile interval of a mean
    report.relative_to  a method: every other method gets its mean
                      difference to it, paired by graph
    report.pool       {by, cap}: after the cells of a metric, the mean of
                      the cell means for every value of by (M, or a graph
                      parameter); the values from cap on are one group

A cell is one setting with one M; its mean is over the graphs with a result.
Runs with the status `timeout`, `error` or `skipped`, and missing runs, have no
result; they are counted and never filled. A method with two levels is shown at
its better level: the level with the higher mean F1 of the cell, rounded to 12
digits; a tie goes to the corrected level, and a level without a result is not
a candidate. That level is shown for every metric.

The bootstrap draws one index matrix per number of graphs, from a generator
seeded with `report.bootstrap.seed`, and uses it for every mean and every
difference of that number of graphs, so differences are paired by graph. A
difference is "higher" or "lower" when its interval excludes 0, else "tied".

A run of the generator "enumerate" is a block of graphs. Its `exact` is the
share of the graphs of the block with an exact output, and it has no other
score.
"""
import csv
import json
import os
from collections import Counter

import numpy as np

from .. import metrics
from ..counting import n_candidates
from . import instances
from .config import ConfigError, load
from .gene_networks import regulatory, regulatory_ignore_indirect
from .runner import run_path, specifications
from .scores import lag1_edges, markov_boundary

# View -> function (truth, A_hat, M, untested) -> {"common": scores or None, "all": scores}; truth is the instance.
VIEWS = {"edges": lambda truth, A_hat, M, untested: metrics.score(truth.A, A_hat, truth.N, truth.T, M, untested),
         "lag1_edges": lag1_edges, "markov_boundary": markov_boundary,
         "regulatory": regulatory, "regulatory_ignore_indirect": regulatory_ignore_indirect}
SCORES = ("f1", "precision", "recall", "exact")
METRICS = SCORES + ("unique_per_candidate", "mean_set", "largest_set", "screened_size", "seconds")


def _views(scoring) -> list:
    """Return the views of `scoring` as a list (default `["edges"]`)."""
    views = scoring.get("view", "edges")
    return views if isinstance(views, list) else [views]


def _several(config) -> bool:
    """Return whether a setting of `config` has more than one view."""
    return any(len(_views(setting["scoring"])) > 1 for setting in config["settings"])


def columns(config: dict) -> list:
    """Return the columns of the table.

    One column per metric, and per view for a score when a setting has several
    views."""
    views = list(dict.fromkeys(view for setting in config["settings"] for view in _views(setting["scoring"])))
    several = _several(config)
    names = config["report"].get("metrics", ["f1"])
    chooser = config["report"].get("level_by", {}).get("view", "edges")
    unknown = [name for name in names if name not in METRICS] + [view for view in [*views, chooser] if view not in VIEWS]
    if unknown:
        raise ConfigError(f"unknown metric or view: {', '.join(map(str, unknown))}")
    return [name + (f" ({view})" if several else "") for name in names for view in (views if several and name in SCORES else views[:1])]


def score_run(run: dict, truth, view: str = "edges") -> dict:
    """Return the score of a run file on `view` against the instance `truth`."""
    if run["graph_sha1"] != truth.graph_sha1:
        raise ConfigError("the graph of a run file is not the graph of its specification")
    spec, size = run["spec"], truth.N * truth.T
    A_hat = np.zeros(size * size, dtype=bool)
    A_hat[run["edges"]] = True
    # The Fisher-z test sets the common targets; the oracle and the GCM test can test every target.
    M = spec["data"].get("M") if spec["ci_test"] == "fisherz" else None
    return VIEWS[view](truth, A_hat.reshape(size, size), M, run["untested"])


def rows(config: dict, runs) -> list:
    """Return one dict per run of a loaded config, in the order of the runs.

    Keys: `setting` (folder of the setting), `graph`, M, `graph_index`,
    `method` (name and options), `level`, `status` ("missing" without a file).
    A run with a result also has `run` (the file), `scores` (view -> score) and
    `values` (column -> number or `None`; "level_by" is the F1 that chooses the
    level)."""
    out, seen, built, truth = [], set(), None, None
    for setting in config["settings"]:
        views, targets = _views(setting["scoring"]), setting["scoring"].get("targets", "common")
        by = {"view": views[0], "targets": targets, **config["report"].get("level_by", {})}
        for spec in specifications({"settings": [setting]}):
            relative, method = run_path(spec), spec["method"]
            if relative in seen:      # Two entries of a config can give the same run.
                continue
            seen.add(relative)
            label = method.get("label", method["name"]) + (f", max lag {method['max_lag']}" if "max_lag" in method else "")
            row = {"setting": relative.split(os.sep)[0], "graph": spec["graph"], "M": spec["data"].get("M"),
                   "graph_index": spec["graph_index"], "method": label + (", unit variance" if method.get("unit_variance") else ""),
                   "level": method.get("level"), "status": "missing"}
            out.append(row)
            path = os.path.join(runs, relative)
            if not os.path.exists(path):
                continue
            with open(path) as f:
                run = json.load(f)
            if run["spec"] != spec:
                raise ConfigError(f"{path} holds a run with another specification")
            row["status"] = run["status"]
            if run["status"] not in ("ok", "partial"):
                continue
            if run["edges"] is None:      # A block of enumerated graphs: its one score is the share of exact outputs.
                share = run["exact"] / run["graphs"]
                row.update(run=run, scores={}, values={"exact": share, "level_by": share, "largest_set": run["largest_set"]})
                continue
            key = (spec["graph"], spec["seed"], spec["graph_index"])
            if key != built:
                truth, built = instances.make_instance(spec["graph"], "none", spec["seed"], spec["graph_index"], config["folder"]), key
            scores = {view: score_run(run, truth, view) for view in dict.fromkeys([*views, by["view"]])}
            per_candidate = run.get("unique_per_candidate")
            if per_candidate is None and method["name"] == "DF-PC":
                per_candidate = run["unique"] / n_candidates(truth.N, truth.T)
            values = {"level_by": (scores[by["view"]][by["targets"]] or {}).get("f1"), "unique_per_candidate": per_candidate,
                      "mean_set": run.get("mean_set"), "largest_set": run.get("largest_set"),
                      "screened_size": run.get("screened_size"), "seconds": float(np.median(run["seconds"]))}
            for view in views:
                part, suffix = scores[view][targets] or {}, f" ({view})" if _several(config) else ""
                values.update({name + suffix: part.get(name) for name in SCORES[:3]})
                values["exact" + suffix] = float(scores[view]["all"]["fp"] == 0 and scores[view]["all"]["fn"] == 0)
            row.update(run=run, scores=scores, values=values)
    return out


def _mean(values, bootstrap, index):
    """Return mean, lo, hi and count of values, or `None` if empty.

    `None` and NaN are dropped."""
    v = np.array([np.nan if value is None else value for value in values], dtype=float)
    v = v[~np.isnan(v)]
    if not len(v):
        return None
    lo = hi = None
    if bootstrap and len(v) > 1:
        if len(v) not in index:
            index[len(v)] = np.random.default_rng(bootstrap["seed"]).integers(0, len(v), (bootstrap["resamples"], len(v)))
        lo, hi = (float(q) for q in np.percentile(v[index[len(v)]].mean(1), [2.5, 97.5]))
    return {"mean": float(v.mean()), "lo": lo, "hi": hi, "graphs": len(v)}


def cells(config: dict, rows: list) -> list:
    """Return one dict per setting, M, method and level, from `rows`.

    Keys: `setting`, `graph`, M, `method`, `level`, `better_level`, `shown`
    (the better level, or every level of a method without a result), `mean`
    (column -> `{mean, lo, hi, graphs}` or `None`) and, with
    `report.relative_to`, `difference` (column -> the same for the method minus
    the reference on the graphs that both have, with verdict)."""
    report, index, names = config["report"], {}, columns(config)
    bootstrap, reference = report.get("bootstrap"), report.get("relative_to")
    groups = {}
    for row in rows:
        key = (row["setting"], row["M"], row["method"], row["level"])
        group = groups.setdefault(key, {**{k: row[k] for k in ("setting", "graph", "M", "method", "level")}, "values": {}})
        if "values" in row:
            group["values"][row["graph_index"]] = row["values"]
    for group in groups.values():
        by_graph = [group["values"][g] for g in sorted(group["values"])]
        group["mean"] = {name: _mean([values.get(name) for values in by_graph], bootstrap, index) for name in [*names, "level_by"]}
    for group in groups.values():
        levels = [other for other in groups.values() if (other["setting"], other["M"], other["method"]) ==
                  (group["setting"], group["M"], group["method"]) and other["mean"]["level_by"]]
        best = max(levels, key=lambda other: (round(other["mean"]["level_by"]["mean"], 12), other["level"] == "corrected"), default=None)
        group["better_level"], group["shown"] = group is best, best is None or group is best
    for group in groups.values() if reference else ():
        base = [other for other in groups.values() if (other["setting"], other["M"], other["method"], other["better_level"]) ==
                (group["setting"], group["M"], reference, True)]
        group["difference"] = {}
        for name in names if base and base[0] is not group else ():
            both = [(group["values"][g].get(name), base[0]["values"][g].get(name)) for g in sorted(group["values"]) if g in base[0]["values"]]
            d = _mean([None if a is None or b is None else a - b for a, b in both], bootstrap, index)
            if d:
                d["verdict"] = None if d["lo"] is None else "higher" if d["lo"] > 0 else "lower" if d["hi"] < 0 else "tied"
            group["difference"][name] = d
    return list(groups.values())


def _expected(config: dict) -> dict:
    """Read `configs/expected/<config>.csv`.

    Maps `(setting, M, method, level, metric)` to its row."""
    path = os.path.join(config["folder"], "expected", os.path.splitext(os.path.basename(config["path"]))[0] + ".csv")
    if not os.path.exists(path):
        raise ConfigError(f"no expected means for this config: {path} is missing")
    with open(path, newline="") as f:
        return {(r["setting"], r["M"], r["method"], r["level"], r["metric"]): r for r in csv.DictReader(line for line in f if line[0] != "#")}


def _number(x, form=".3f") -> str:
    return "-" if x is None else format(x, form)


def _lines(body: list) -> list:
    """Return `body` as aligned lines.

    A column without an entry is left out (M with the oracle, `lo` and `hi`
    without the bootstrap)."""
    keep = [i for i in range(len(body[0])) if len(body) == 1 or any(line[i] != "-" for line in body[1:])]
    body = [[line[i] for i in keep] for line in body]
    widths = [max(len(line[i]) for line in body) for i in range(len(body[0]))]
    return ["  ".join(word.ljust(width) for word, width in zip(line, widths)).rstrip() for line in body]


def _pooled(groups: list, name: str, pool: dict, reference) -> list:
    """Return the `report.pool` block for metric `name`.

    Gives the mean of the cell means per value of `pool["by"]` and method."""
    found = {}
    for group in groups:
        value = {"M": group["M"], **group["graph"]}.get(pool["by"])
        if group["shown"] and group["mean"][name] and value is not None:
            key = (min(value, pool["cap"]), group["method"], group["level"] or "-")
            found.setdefault(key, []).append((group["mean"][name]["mean"], group.get("difference", {}).get(name)))
    body = [[pool["by"], "method", "level", "cells", "mean"] + ([f"minus {reference}", "higher", "tied", "lower"] if reference else [])]
    for (value, method, level), cells_ in sorted(found.items(), key=lambda item: item[0][0]):
        line = [(">= " if value == pool["cap"] else "") + f"{value:g}", method, level, str(len(cells_)),
                _number(float(np.mean([mean for mean, _ in cells_])))]
        if reference:
            differences = [d for _, d in cells_ if d]
            verdicts = Counter(d["verdict"] for d in differences)
            line += [_number(float(np.mean([d["mean"] for d in differences])) if differences else None, "+.3f"),
                     *(str(verdicts[word]) if differences else "-" for word in ("higher", "tied", "lower"))]
        body.append(line)
    return body


def table(config, runs, *, compare: bool = False) -> str:
    """Return the table of `config` as text, from the run files in `runs`.

    One block per metric, one line per cell and method; a method with two
    levels is shown at its better level. With `report.pool` a second block per
    metric has the means of the cell means. `compare=True` adds the mean of
    `configs/expected/<config>.csv` and the difference to it."""
    config = load(config)
    names, reference = columns(config), config["report"].get("relative_to")
    all_rows = rows(config, runs)
    groups = cells(config, all_rows)
    expected = _expected(config) if compare else {}
    status = Counter("with a result" if "values" in row else "missing" if row["status"] == "missing" else "without a result"
                     for row in all_rows)
    lines = [config["paper_item"] or "", f"{len(all_rows)} runs: " + ", ".join(f"{count} {word}" for word, count in status.items())]
    graph_keys = [key for key in dict.fromkeys(key for group in groups for key in group["graph"])
                  if key != "path" and len({str(group["graph"].get(key)) for group in groups}) > 1]
    differences, other_graphs, other_level = [], 0, 0
    for name in names:
        head = [*graph_keys, "M", "method", "level", "graphs", "mean", "lo", "hi"]
        head += [f"minus {reference}", "lo", "hi", "verdict"] if reference else []
        body = [head + (["expected", "difference"] if compare else [])]
        for group in groups:
            mean = group["mean"][name] or {}
            stored = expected.get((group["setting"], str(group["M"] or ""), group["method"], group["level"] or "", name))
            if stored and mean:
                differences.append(mean["mean"] - float(stored["mean"]))
                other_graphs += mean["graphs"] != int(stored["graphs"])
                other_level += str(group["better_level"]) != stored["better_level"]
            if not group["shown"]:
                continue
            line = [*(str(group["graph"].get(key, "-")) for key in graph_keys), str(group["M"] or "-"), group["method"],
                    group["level"] or "-", str(mean.get("graphs", 0)), *(_number(mean.get(k)) for k in ("mean", "lo", "hi"))]
            if reference:
                d = group["difference"].get(name) or {}
                line += [_number(d.get("mean"), "+.3f"), _number(d.get("lo"), "+.3f"), _number(d.get("hi"), "+.3f"), d.get("verdict") or "-"]
            if compare:
                line += ["-", "-"] if not (stored and mean) else [_number(float(stored["mean"])), format(differences[-1], "+.1e")]
            body.append(line)
        lines += ["", name] + _lines(body)
        if config["report"].get("pool"):
            pool = config["report"]["pool"]
            lines += ["", f"{name}: mean of the cell means by {pool['by']}"] + _lines(_pooled(groups, name, pool, reference))
    if compare:
        lines += ["", f"{len(differences)} means compared with the expected means: largest absolute difference "
                      f"{max(map(abs, differences), default=0):.1e}; {other_graphs} with another number of graphs; "
                      f"{other_level} with another better level"]
    return "\n".join(lines)
