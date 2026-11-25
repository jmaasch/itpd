# Iterative Temporal Parent Discovery

Iterative Temporal Parent Discovery (ITPD) is a constraint-based causal discovery algorithm for multivariate time series.

## Folders

- `src/itpd/`: the Python package. `src/itpd/algorithm.py` is ITPD, Algorithm 1 of the paper; the comments in `_discover` carry the line numbers of the algorithm.
- `tests/`: tests of the package.
- `requirements/constraints.txt`: the exact versions behind the reference numbers of the paper (Python 3.12).
- `lean/`: a Lean 4 proof that Algorithm 1 returns the true parent set of every target whose parents are all observed, under the paper's assumptions and with a d-separation oracle as the CI test. The proof does not cover the significance levels or the Python code. See `lean/README.md`.
- `legacy/`: earlier code, kept for reference. It is unmaintained and the package does not use it.

## Install

```
pip install -e ".[test]"                                   # Python 3.10 or newer
pip install -c requirements/constraints.txt -e ".[test]"   # Python 3.12, with the exact versions
```

## Use

```python
from itpd import ITPD

model = ITPD(ci_test, alpha=0.01).fit(V)   # V has shape (M, T, N); ci_test(x, y, S) returns a p-value
parents = model.adjacency_                 # node t * N + n -> frozenset of parents
```

## Tests

```
python -m pytest -q
python -m pytest -q -m "not slow"    # without the finite-data runs
```
