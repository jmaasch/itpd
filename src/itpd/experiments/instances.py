"""Build the instance of a setting: the true time graph and its data.

An instance is a function of the `graph` block of the setting, the data
mechanism, the seed and the graph index. The module of the generator in
`itpd.simulation` lists the random streams. The generator "gene_networks" reads
stored data (`itpd.experiments.gene_networks`): the graph index is the number
of the data set, and the seed is not used.
"""
import os

from ..simulation import Instance, linear, nonlinear, redrawn_edges, stepwise, violations
from . import gene_networks
from .ablations import BLOCK


def _standard(graph, mechanism, seed, g) -> Instance:
    make = nonlinear.make_instance if mechanism == "nonlinear" else linear.make_instance
    options = {"weights": False} if mechanism == "none" else {}
    return make(graph["N"], graph["T"], graph["lag"], graph["in_degree"], g, stationary=graph["stationary"], seed=seed, **options)


def _stepwise(graph, mechanism, seed, g) -> Instance:
    return stepwise.make_instance(graph["N"], graph["T"], graph["in_degree"], g, seed=seed)


def _redrawn_edges(graph, mechanism, seed, g) -> Instance:
    return redrawn_edges.make_instance(graph["N"], graph["T"], graph["in_degree"], graph["share"], g, seed=seed)


def _violations(graph, mechanism, seed, g) -> Instance:
    return violations.make_instance(graph["in_degree"], g, no_self_edge_share=graph["no_self_edge_share"],
                                    same_step_share=graph["same_step_share"], seed=seed, N=graph["N"], T=graph["T"])


def _gene_networks(graph, mechanism, seed, g) -> Instance:
    return gene_networks.load(graph["network"], g, graph["path"])


def _enumerate(graph, mechanism, seed, g) -> Instance:
    # Not one graph: g stands for the masks of block g, and `info` holds the
    # arguments of `exhaustive_check`.
    N, T = graph["N"], graph["T"]
    masks = range(g * BLOCK, min((g + 1) * BLOCK, 1 << (N * N * T * (T - 1) // 2 - N * (T - 1))))
    if not masks:
        raise ValueError(f"{N} series and {T} steps have no block {g} of enumerated graphs")
    return Instance(None, N, T, T - 1, None, None, {"masks": masks, "latent_first_step": graph["latent_first_step"]})


# Generator -> (parameters in folder-name order, data mechanism -> function).
# "none" gives the graph without data, for the oracle and scores.
GENERATORS = {
    "standard": (("N", "T", "lag", "in_degree", "stationary"),
                 {"linear_gaussian": _standard, "nonlinear": _standard, "none": _standard}),
    "stepwise": (("N", "T", "in_degree"), {"linear_gaussian": _stepwise, "none": _stepwise}),
    "redrawn_edges": (("N", "T", "in_degree", "share"), {"linear_gaussian": _redrawn_edges, "none": _redrawn_edges}),
    "violations": (("N", "T", "in_degree", "no_self_edge_share", "same_step_share"),
                   {"linear_gaussian": _violations, "none": _violations}),
    "gene_networks": (("network", "path"), {"stored": _gene_networks, "none": _gene_networks}),
    "enumerate": (("N", "T", "latent_first_step"), {"none": _enumerate}),
}


def make_instance(graph: dict, mechanism: str, seed: int, g: int, folder: str = "") -> Instance:
    """Return graph number g of a setting.

    `graph` holds the generator and its parameters. With the mechanism "none"
    only the graph is used: it is the graph of every other mechanism. The
    parameter `path` (generator "gene_networks") is relative to `folder`, the
    folder of the config file. The generator "enumerate" gives a block of
    `BLOCK` enumerated graphs for the exhaustive check of
    `itpd.experiments.ablations`, and no single graph."""
    if "path" in graph:
        graph = {**graph, "path": os.path.normpath(os.path.join(folder, graph["path"]))}
    return GENERATORS[graph["generator"]][1][mechanism](graph, mechanism, seed, g)


def rows(mechanism: str, M: int) -> int:
    """Return the row count of the draw for M observations.

    Simulated data are the first M rows of a draw of max(linear.ROWS, M) rows.
    Stored data (mechanism "stored") are centered over the M observations in
    use, so every M has its own draw of M rows."""
    return M if mechanism == "stored" else max(linear.ROWS, M)


def draw(instance: Instance, rows: int):
    """Return a draw of `rows` rows, and its SHA-1."""
    X = instance.sample(rows)
    return X, linear.data_hash(X)
