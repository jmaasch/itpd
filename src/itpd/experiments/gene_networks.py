"""Simulated gene regulatory networks: data, graph and regulatory score.

A network has 10 genes and 20 series: the mRNA of gene 1 to 10, then the
protein of gene 1 to 10. A stored array has the shape (observations, 41, 20);
its first time step is the same in every observation and is dropped, so T = 40.
The files are in `data/gene_networks` (see its `README.md`):

    truth/net<k>_edges.csv       edges of the lag-1 graph, with their type
    truth/net<k>_indirect.csv    indirect pairs of series
    net<k>_g<g>.npy              data set g of network k (made by simulate.py)
    prefix/net<k>_g<g>_M200.npy  its first 200 observations

The true time graph has every edge of the lag-1 graph at every step: regulatory
edges (protein of gene j -> mRNA of gene k), translation edges (mRNA of gene
i -> protein of gene i) and the self edges.

The regulatory score takes the regulatory edges as the truth. The self edges
and the translation edges are removed from the estimate; every other estimated
edge, of any lag, is false. With `ignore_indirect`, an estimated lag-1 edge of
an indirect pair is removed as well, so it counts neither as true nor as false.
An indirect pair is a pair of series without an edge whose first series changes
the second within one time step, through a third series.
"""
import csv
import os

import numpy as np

from .. import metrics
from ..graphs import graph_hash
from ..simulation import Instance

N, T = 20, 40
SERIES = [f"mRNA_G{i}" for i in range(1, 11)] + [f"protein_G{i}" for i in range(1, 11)]


def _lag1(path, kinds=None) -> np.ndarray:
    """Return the time graph with lag-1 edges from a CSV file at every step.

    `kinds` selects pairs `(source, target)` by the column `type`."""
    with open(path, newline="") as f:
        pairs = [(SERIES.index(row["source"]), SERIES.index(row["target"])) for row in csv.DictReader(f)
                 if kinds is None or row["type"] in kinds]
    A = np.zeros((T * N, T * N), dtype=bool)
    for t in range(1, T):
        for i, j in pairs:
            A[(t - 1) * N + i, t * N + j] = True
    return A


def load(network: int, dataset: int, root) -> Instance:
    """Return data set `dataset` of `network`, from the folder `root`.

    `Instance.A` is the true time graph. `sample(M)` returns the first M
    observations, centered per series and time step over these M observations;
    it reads the simulated array, or the stored first 200 observations when
    M <= 200 and the simulated array is missing. `info` holds the time graphs
    "regulatory", "translation" and "indirect"."""
    edges = os.path.join(root, "truth", f"net{network}_edges.csv")
    info = {"regulatory": _lag1(edges, ("protein_to_mRNA",)), "translation": _lag1(edges, ("mRNA_to_protein",)),
            "indirect": _lag1(os.path.join(root, "truth", f"net{network}_indirect.csv"))}
    A = _lag1(edges)

    def sample(M):
        whole = os.path.join(root, f"net{network}_g{dataset}.npy")
        first = os.path.join(root, "prefix", f"net{network}_g{dataset}_M200.npy")
        path = whole if os.path.exists(whole) else first
        if not os.path.exists(path) or len(np.load(path, mmap_mode="r")) < M:
            raise FileNotFoundError(f"{M} observations of data set {dataset} of network {network} need the simulated array "
                                    f"{whole}: see README.md in that folder")
        X = np.load(path, mmap_mode="r")[:M, 1:, :]
        return np.ascontiguousarray(X - X.mean(axis=0, keepdims=True))
    return Instance(A, N, T, 1, sample, graph_hash(A), info)


def regulatory(truth, A_hat, M=None, untested=(), *, ignore_indirect: bool = False) -> dict:
    """Score `A_hat` on the regulatory edges of a gene-network instance.

    `truth` is an `Instance` of `load`. M sets the common targets (`None`:
    every target). `untested` lists the untested targets. Returns the dict of
    `itpd.metrics.score`."""
    estimate = (np.asarray(A_hat) != 0) & ~truth.info["translation"]
    if ignore_indirect:
        estimate &= ~truth.info["indirect"]
    return metrics.score(truth.info["regulatory"], estimate, truth.N, truth.T, M, untested)


def regulatory_ignore_indirect(truth, A_hat, M=None, untested=()) -> dict:
    """Score as `regulatory`, ignoring indirect pairs.

    Indirect pairs count neither as true nor as false."""
    return regulatory(truth, A_hat, M, untested, ignore_indirect=True)
