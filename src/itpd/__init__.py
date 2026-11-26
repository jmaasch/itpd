"""Iterative Temporal Parent Discovery (ITPD).

Start reading at `algorithm.py`: it implements Algorithm 1 of the paper.

Exports:
- `ITPD`: the algorithm (`algorithm.py`).
- `FisherZ`, `GCM`, `DSeparation`: CI tests (`ci_tests.py`).
- `CICounter`: counter of CI tests (`counting.py`).
- `NotEvaluable`: exception of a CI test that cannot be evaluated.

Other modules and subpackages:
- `graphs.py`: time graphs as adjacency matrices.
- `metrics.py`: scores.
- `baselines`: IAMB, IPaDL, FullCI, and Lasso.
- `simulation`: simulated data.
- `external`: scripts for PCMCI, DF-PC, DYNOTEARS, and CD-NOD.
- `experiments`: runs the configs and prints the tables."""
from .algorithm import ITPD
from .ci_tests import GCM, DSeparation, FisherZ, NotEvaluable
from .counting import CICounter

__version__ = "0.2.0"
__all__ = ["ITPD", "FisherZ", "GCM", "DSeparation", "CICounter", "NotEvaluable"]
