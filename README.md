# Iterative Temporal Parent Discovery (ITPD)

ITPD (Iterative Temporal Parent Discovery) is a constraint-based algorithm for causal structure learning on multivariate time series without assuming causal stationarity. It finds the parent set of every variable in time order with conditional independence (CI) tests. This repository holds the Python package `itpd`, the configuration files of the paper's experiments, and a Lean 4 proof that Algorithm 1 is exact with a d-separation oracle.

## Where things are

- `src/itpd/algorithm.py`: the class `ITPD` (Algorithm 1; read this first). Comments in `_discover` give Lines 5 to 16, and the defaults match the paper (alpha = 0.01, gamma = 0.2, kappa = 50, Fisher-z test).
- `src/itpd/ci_tests.py`: the CI tests `FisherZ` (linear-Gaussian data), `GCM` (nonlinear data) and `DSeparation` (d-separation oracle).
- `src/itpd/counting.py`: `CICounter`, a CI test wrapper that counts every test call in the same way across all methods.
- `src/itpd/baselines/`: the baselines IAMB, IPaDL, FullCI and Lasso, one file per baseline.
- `src/itpd/simulation/`: the generators of the simulated graphs and data, with data shape (M, T, N).
- `src/itpd/external/`: one wrapper script per external baseline (PCMCI, DF-PC, DYNOTEARS and CD-NOD); the files in `requirements/` pin the external tools.
- `src/itpd/experiments/`: the command line `python -m itpd.experiments`. `runner.py` runs the methods of a configuration, `aggregate.py` builds the table of means, and `ablations.py` runs Algorithm 1 with single-line changes for the ablation tables.
- `configs/`: 19 YAML files for the paper's experiments and `configs/expected/` for expected means. See `configs/README.md` for status and what is not reproduced.
- `examples/`: `examples/fit_simulated_data.py` fits ITPD on simulated data, and `examples/oracle_exact_recovery.py` recovers a random graph with the d-separation oracle.
- `tests/`: the package tests, including `tests/test_reference_runs.py` for comparing runs with stored runs of the paper's experiments.
- `data/gene_networks/`: the data for the two tables on simulated gene regulatory networks; see `data/gene_networks/README.md`.
- `lean/`: the Lean 4 proof (`lean/ITPD/Paper.lean` states `itpdPaper` and `itpd_exact`); run `lake build` here, and see `lean/README.md`.
- `legacy/`: earlier code, kept for reference. The package and the configurations do not use it.
- `LICENSE`, `NOTICE.md`: the GPL-3.0 license and the third-party notices.

## Install

```bash
pip install .
pip install -c requirements/constraints.txt ".[test]"
```

The package needs Python 3.10 or newer. On Python 3.12, `requirements/constraints.txt` installs the exact versions behind the paper's numbers, and `test` adds pytest and matplotlib.

## Use

```bash
python examples/fit_simulated_data.py
```

This fits ITPD on 500 observations of 5 series over 8 time steps in a few seconds and prints F1 and CI test counts. In Python, call `ITPD().fit(V)` on an array `V` of shape (M, T, N) and read `adjacency_`.

```bash
python -m itpd.experiments run configs/main_setting.yaml --out out/main_setting --methods ITPD --M 500
python -m itpd.experiments table configs/main_setting.yaml --runs out/main_setting --compare
python -m pytest -q
```

The first command makes the 80 runs of ITPD with 500 observations in about 5 minutes. The second command prints the tables of the main setting with each mean next to its expected mean. It ends with `largest absolute difference 0.0e+00`. The third command runs 194 tests in about one minute. `python -m pytest -q -m slow` runs 7 more tests on stored runs. See `configs/README.md` for all configurations.
