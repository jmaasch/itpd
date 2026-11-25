"""Build the instance of a setting: the true time graph and its data.

An instance is a function of the graph block of the setting, the data
mechanism, the seed and the graph index. The module of the generator in
itpd.simulation lists the random streams.
"""
from ..simulation import Instance, linear, nonlinear, redrawn_edges, stepwise

# The methods get the first M rows of one draw of max(ROWS, M) rows.
ROWS = 2000


def _standard(graph, mechanism, seed, g) -> Instance:
    make = nonlinear.make_instance if mechanism == "nonlinear" else linear.make_instance
    options = {"weights": False} if mechanism == "none" else {}
    return make(graph["N"], graph["T"], graph["lag"], graph["in_degree"], g, stationary=graph["stationary"], seed=seed, **options)


def _stepwise(graph, mechanism, seed, g) -> Instance:
    return stepwise.make_instance(graph["N"], graph["T"], graph["in_degree"], g, seed=seed)


def _redrawn_edges(graph, mechanism, seed, g) -> Instance:
    return redrawn_edges.make_instance(graph["N"], graph["T"], graph["in_degree"], graph["share"], g, seed=seed)


# Generator -> (its parameters, in the order of the folder name; data mechanism -> function).
# The mechanism "none" is the graph without data, for the oracle and for the scores.
GENERATORS = {
    "standard": (("N", "T", "lag", "in_degree", "stationary"),
                 {"linear_gaussian": _standard, "nonlinear": _standard, "none": _standard}),
    "stepwise": (("N", "T", "in_degree"), {"linear_gaussian": _stepwise, "none": _stepwise}),
    "redrawn_edges": (("N", "T", "in_degree", "share"), {"linear_gaussian": _redrawn_edges, "none": _redrawn_edges}),
}


def make_instance(graph: dict, mechanism: str, seed: int, g: int) -> Instance:
    """Return graph number g of a setting.

    graph holds the generator and its parameters. With the mechanism "none"
    only the graph is used: it is the graph of every other mechanism."""
    return GENERATORS[graph["generator"]][1][mechanism](graph, mechanism, seed, g)


def draw(instance: Instance, M: int):
    """Return the draw that holds the data of M observations, and its SHA-1.

    The draw has max(ROWS, M) rows; the data are its first M rows."""
    X = instance.sample(max(ROWS, M))
    return X, linear.data_hash(X)
