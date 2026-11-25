# Configurations

Each YAML file in `configs/` describes the runs behind one table, one figure or one statement of the paper, named on its first line. `configs/expected/` holds the expected means. The method ITPD is the class `ITPD` with the constants of the paper (alpha = 0.01, gamma = 0.2, kappa = 50).

| Config | Reproduces | Checked against stored runs | Full run |
|---|---|---|---|
| `oracle_exact_recovery.yaml` | Table: unique CI tests per candidate and conditioning set sizes under the exact oracle | 205 | about 13 h |
| `oracle_all_settings.yaml` | Text: exact recovery in all 32 exact-oracle settings | 40 | more than 82 h |
| `main_setting_f1.yaml` | Table: mean F1 in the main finite data setting | 294 | about 5.2 h |
| `main_setting_tests.yaml` | Table: unique CI tests per candidate and conditioning set sizes in the main finite data setting | 256 | about 2.2 h |
| `frontier_seconds.yaml` | Figure: mean F1 against wall-clock time in the main setting (Pareto frontier) | 104 | about 66 min |
| `frontier_tests.yaml` | Figure: mean F1 against unique CI tests per candidate in the main setting | 216 | about 66 min |
| `lag.yaml` | Figure: mean F1 on common targets against the true lag on stationary and nonstationary graphs | 1110 | about 1.5 h |
| `redrawn_edges.yaml` | Figure: mean F1 against the share of edges redrawn at each time step | 376 | about 8.9 h |
| `sample_size.yaml` | Figure: mean F1 on all targets against the number of observations | 123 | more than 3.5 h |
| `nonlinear.yaml` | Table: mean F1 with nonlinear causal mechanisms | 41 | more than 2.7 h |
| `scalability.yaml` | Table: mean F1 and unique CI tests per candidate on larger graphs | 23 | more than 3.6 h |
| `runtime_itpd_iamb.yaml` | Table: seconds per run of ITPD and IAMB | 56 | more than 7.6 h |
| `scaling_time.yaml` | Figure: seconds per run and F1 of ITPD against the number of time-indexed variables | 33 | more than 5.4 h |
| `violation_no_self_edge.yaml` | Table: mean F1 with one assumption violated (no lag-1 self edge) | 235 | more than 2.5 h |
| `violation_same_step_edges.yaml` | Table: mean F1 with edges moved into the same time step | 210 | more than 2.8 h |
| `gene_networks.yaml` | Table: mean F1 on simulated gene regulatory networks (regulatory edges) | 24 | more than 50 min |
| `gene_networks_indirect.yaml` | Table: mean F1 on simulated gene regulatory networks (regulatory edges, indirect pairs left out) | 16 | more than 50 min |
| `extended_bic.yaml` | Table: mean F1 of ITPD with alpha / m and with the extended-BIC level in Phase II | 274 | more than 2.2 h |
| `gamma_kappa.yaml` | Text: choice of gamma and kappa | 458 | more than 2.4 h |
| `larger_in_degree.yaml` | Text: unique CI tests per candidate of ITPD and IAMB at larger in-degrees | 156 | about 1.8 h |
| `dropped_steps.yaml` | Table: steps dropped from Algorithm 1, each added back alone | 240 | about 46 min |
| `ablation.yaml` | Table: one change at a time to ITPD | 190 | about 13 h |

- Checked against stored runs: N runs were rerun from this code. Their true positives, false positives, false negatives, CI test counts and conditioning set sizes equal the stored runs of the paper's experiments.
- The other runs of each config were not rerun because of run time. Full run is the rough time for all runs of the config with the package methods on one core.
- All 435 completely rerun cell means equal the expected means.

## Not reproduced by this repository

1. Wall-clock seconds in `frontier_seconds.yaml`, `runtime_itpd_iamb.yaml` and `scaling_time.yaml` depend on the machine and its load. These three configs were checked on their edges and CI test counts only.
2. The external baselines (PCMCI, DF-PC, DYNOTEARS and CD-NOD) need their own tools, wrapped in `src/itpd/external/` and pinned in `requirements/`. 50 of these runs were rerun with the tools installed separately, and all 50 equal their stored runs.
3. For gene-network data, the repository has only the first 200 observations of data set 0 for each of the two networks. The other data sets and larger M require simulation with GeneNetWeaver (`data/gene_networks/simulate.py`).
4. One appendix table on the exactness and results of each version of the algorithm has no config. Every other table or figure with results has a config.

One baseline run of `sample_size.yaml` (IAMB at the level alpha, 40 series, M = 50, graph 0) has 635 true positives, 3117 false positives and 572 false negatives versus 637, 3115 and 570 in the stored run; the count of CI tests, the set sizes and the hash of the data are equal.

## Rerun a configuration

```bash
python -m itpd.experiments run configs/frontier_tests.yaml --out out/frontier_tests --methods ITPD
python -m itpd.experiments table configs/frontier_tests.yaml --runs out/frontier_tests --compare
```

- `run` writes one result file per run into `--out` and skips finished runs. An interrupted run restarts with the same command.
- `--graphs` (such as `0-2`), `--methods` (such as `ITPD,IAMB`) and `--M` (such as `200,500`) select a subset.
- `table --compare` prints each mean next to its expected mean. A mean matches the paper only when all graphs of the cell ran.
- Each external tool uses its own Python interpreter. See the top of the files in `requirements/`.
