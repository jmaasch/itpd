"""Simulated time series with a known time graph.

Every module has a function `make_instance` that returns an `Instance`:
- `linear`: linear-Gaussian data on a stationary or nonstationary graph.
- `stepwise`: the same model built step by step, for many series.
- `redrawn_edges`: a share of the edge slots is redrawn at every step.
- `violations`: series without lag-1 self edge, or edges inside a time step.
- `nonlinear`: tanh-and-sine mechanisms.

All random numbers come from seeded generators, so an instance is a function of
its arguments.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class Instance:
    """A true time graph and the data drawn from it.

    A is the Boolean time graph (T * N, T * N). `sample(M)` returns M
    observations with shape (M, T, N); a smaller M gives the first rows of the
    same draw. `sample` is `None` for an instance without data. `graph_sha1` is
    `itpd.graphs.graph_hash(A)`. `info` holds the weights or parameters, the
    numbers of the weight check, and further truths of the generator."""
    A: np.ndarray
    N: int
    T: int
    lag: int
    sample: Callable[[int], np.ndarray] | None
    graph_sha1: str
    info: dict = field(default_factory=dict)


from . import graphs, linear, nonlinear, redrawn_edges, stepwise, violations  # noqa: E402

__all__ = ["Instance", "graphs", "linear", "nonlinear", "redrawn_edges", "stepwise", "violations"]
