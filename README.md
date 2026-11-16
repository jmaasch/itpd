# Iterative Temporal Parent Discovery

Iterative Temporal Parent Discovery (ITPD) is a constraint-based causal discovery algorithm for multivariate time series.

## Repository structure

```
.
├── itpd/                # Instrumented implementation of ITPD, ITPD_naive, ITPD-S (blanket-screened shrink), ITPD-S+ (ITPD-S with a forward-backward second pass) and baselines (package; see itpd/README.md). itpd/experiments and itpd/tables hold the experiment drivers and the collectors.
├── itpd_clear/          # ITPD, Algorithm 1 of the paper, in one file (itpd_clear/itpd.py); see the section below.
├── tests/                # Tests of the package, including parity tests against legacy/.
├── pyproject.toml
├── legacy/
│   ├── itpd.py               # ITPD with CI test reduction.
│   ├── padl_itpd.py          # The version of PaDL used by ITPD.
│   ├── itpd_naive.py         # ITPD_naive (no CI test reduction).
│   ├── padl_naive.py         # The version of PaDL used by ITPD_naive.
│   ├── itpd_test.ipynb       # A notebook comparing ITPD to ITPD_naive with an oracle independence test.
│   └── itpd_naive_test.ipynb # A notebook using ITPD_naive with finite data simulation (finite data have errors; just for ITPD syntax demo).
├── LICENSE
└── README.md
```

## Install

```
pip install -e '.[test]'
```

This installs the package `itpd` and the test dependencies. It needs Python 3.10 or newer. The d-separation oracle uses the optional C extension `fastdsep` when it is installed and a pure-Python implementation otherwise; both give the same answers.

## Quick start

```python
import numpy as np
from itpd import method_runner, sim

graph = sim.sample_time_graph(N=4, T=6, d=2, tau=2, rng=np.random.default_rng(0))   # random time graph, 4 series, 6 steps
for method in ("itpd_naive", "itpd", "full_conditioning"):
    out = method_runner.run_s2_method(method, None, graph=graph, ci_kind="oracle", alpha=0.01)
    print(method, "exact:", out["metrics"]["exact"], "unique tests:", out["tests"]["unique_tests"])
```

The example runs ITPD_naive, ITPD and the full-conditioning baseline with the d-separation oracle on the true graph and prints the number of unique CI tests of each method.

## Tests

```
python -m pytest -q
```

## ITPD, Algorithm 1 of the paper (`itpd_clear/`)

`itpd_clear/itpd.py` is a single-file implementation of ITPD as printed in Algorithm 1 of the paper. The comments in `_discover` carry the line numbers of the algorithm. The file needs only `numpy` and `networkx` (`matplotlib` for `plot_graph`, `causal-learn` or `tigramite` for their CI tests) and does not import the package above.

```python
from itpd import ITPD          # run from inside itpd_clear/, or add that folder to sys.path

model = ITPD(ci_test="fisherz", alpha=0.01, gamma=0.2, kappa=50).fit(V)    # V has shape (M, T, N)
parents = model.adjacency_     # node t * N + n -> frozenset of parents
```

The algorithm tests at two levels. For a target at time step t + 1 (the paper's indexing), let m = N t - 1 be its number of candidates.

| Level | Formula | Used on | Arguments and defaults |
| --- | --- | --- | --- |
| α1 | max(α, min(γ, κ / m)) | Lines 8, 9 and 10 (Phase I) | `alpha=0.01`, `gamma=0.2`, `kappa=50` |
| α2 | α / m | Line 14 (Phase II) | `alpha=0.01` |

- A test answers "independent" if and only if p is larger than the level.
- Line 6 visits the candidates in descending time and Line 13 in ascending time; inside a time step both go by increasing series index.
- A CI test that returns a NaN p-value cannot be evaluated (for example `itpd_aistats.ci.FisherZ` when M - |S| - 3 <= 0; causal-learn's `fisherz`, the default, returns p = 1 at 0 and raises an error below 0). The target of such a test keeps only its lag-1 self edge and is listed in `untested_targets_`, and `fit` emits one warning. Exclude these targets from a score.
- `python itpd_clear/itpd.py` runs an example with the d-separation oracle and checks exact recovery.

Tests of this file:

```
python -m pytest tests/test_itpd_clear.py                 # about 20 seconds
python -m pytest tests/test_itpd_clear.py -m "not slow"   # without the finite-data runs, a few seconds
```

They check exact recovery with the d-separation oracle (all 256 graphs with N = 2 and T = 3, and random graphs with N = 2, T = 4 and N = 3, T = 3), the two levels, and the edge counts and CI test counts of three finite-data runs behind the paper's results, with data regenerated from seeds.

The Lean 4 proof that Algorithm 1 returns the true parent set of every target whose parents are all observed, under the paper's assumptions and with a d-separation oracle as the CI test, is in the folder `lean/`. The proof does not cover the significance levels or this Python code.

## Counting conventions

- Every CI test goes through one counting wrapper, `Recorder` in `itpd/ci.py`, so the counts of different methods are comparable.
- A unique test is a distinct (unordered pair of nodes, sorted conditioning set); it is the headline count. A raw call is every call that reaches the recorder.
- Marginal tests are memoised inside one PaDL call, so raw counts match the original code in `legacy/`.
- Lazy counting issues a test only when its result can change a label; non-lazy counting issues both tests of every step, as the original code does. Both give the same graphs.
- A test that the data cannot evaluate (Fisher-z with n - |S| - 3 <= 0) is never answered "independent": the target is reported as infeasible and is scored apart.

## Experiments

`itpd/README.md` has the module map, the table of method names and the table "Experiment -> command" with the driver and the collector of every experiment. The drivers run with `python -m itpd.experiments <name>`; the collectors turn their JSON output into tables with `python -m itpd.tables <name>`. Put your cluster job scripts in the ignored folder `jobs/`; they are not part of the repository.

## Legacy code

`legacy/` holds the original modules and notebooks. They moved there unchanged, except for three type annotations in each of `itpd.py` and `itpd_naive.py` that Python older than 3.14 could not import. The modules import each other by their flat names, so run the notebooks and the scripts from inside `legacy/`. They import `cdt`, which needs R; the tests replace it with a stub. `tests/test_parity.py` checks that the package `itpd` gives the same results as these modules. The package and the module `legacy/itpd.py` share a name: inside `legacy/`, `import itpd` gives the original module, and the tests load the original modules by file path under the names `legacy_itpd`, `legacy_itpd_naive`, `legacy_padl_itpd` and `legacy_padl_naive`.
