"""Build the instance of a setting: the true time graph and its data.

An instance is a function of the graph block of the setting, the data
mechanism, the seed and the graph index. The module of the generator in
itpd.simulation lists the random streams.
"""
from ..simulation import Instance, linear

# The methods get the first M rows of one draw of max(ROWS, M) rows.
ROWS = 2000


def _standard(graph, mechanism, seed, g) -> Instance:
    return linear.make_instance(graph["N"], graph["T"], graph["lag"], graph["in_degree"], g,
                                stationary=graph["stationary"], seed=seed, weights=mechanism != "none")


# Generator -> (its parameters, in the order of the folder name; data mechanism -> function).
# The mechanism "none" is the graph without data, for the oracle.
GENERATORS = {"standard": (("N", "T", "lag", "in_degree", "stationary"), {"linear_gaussian": _standard, "none": _standard})}


def make_instance(graph: dict, mechanism: str, seed: int, g: int) -> Instance:
    """Return graph number g of a setting.

    graph holds the generator and its parameters. With the mechanism "none"
    the instance is the graph alone: it equals the graph of every other
    mechanism, and its sample is None."""
    return GENERATORS[graph["generator"]][1][mechanism](graph, mechanism, seed, g)


def draw(instance: Instance, M: int):
    """Return the draw that holds the data of M observations, and its SHA-1.

    The draw has max(ROWS, M) rows; the data are its first M rows."""
    X = instance.sample(max(ROWS, M))
    return X, linear.data_hash(X)
