# Configurations

Each YAML file in `configs/` describes the runs behind one table, one figure or one statement of the paper. `main_setting.yaml` describes the two tables and two figures of the main setting from the same runs. `configs/expected/` has the expected means. The method ITPD is the class `ITPD` with the constants of the paper (alpha = 0.01, gamma = 0.2, kappa = 50).

| Config | Reproduces | Checked against stored runs | Full run |
|---|---|---|---|
| `oracle_exact_recovery.yaml` | Table: unique CI tests per candidate and conditioning set sizes under the exact oracle | 205 | about 13 h |
| `oracle_all_settings.yaml` | Text: exact recovery in all 32 exact-oracle settings | 40 | more than 82 h |
| `main_setting.yaml` | Main finite data setting. Tables: mean F1; unique CI tests per candidate and conditioning set sizes. Figures: mean F1 against wall-clock time (Pareto frontier); mean F1 against unique CI tests per candidate | 672 | about 5.2 h |
| `lag.yaml` | Figure: mean F1 on common targets against the true lag on stationary and nonstationary graphs | 1110 | about 1.5 h |
| `redrawn_edges.yaml` | Figure: mean F1 against the share of edges redrawn at each time step | 376 | about 8.9 h |
| `sample_size.yaml` | Figure: mean F1 on all targets against the number of observations | 487 | more than 3.5 h |
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
- 479 of the 481 completely rerun cell means equal the expected means. The other two appear under "Not reproduced by this repository".

## Not reproduced by this repository

1. Wall-clock seconds depend on the machine and its load, and they have no expected means. This concerns the figure of mean F1 against wall-clock time in `main_setting.yaml`, `runtime_itpd_iamb.yaml` and `scaling_time.yaml`. Their runs were checked on edges and CI test counts.
2. The external baselines (PCMCI, DF-PC, DYNOTEARS and CD-NOD) need their own tools, wrapped in `src/itpd/external/` and pinned in `requirements/`. 50 of these runs were rerun with the tools installed separately, and all 50 equal their stored runs.
3. For gene-network data, the repository has only the first 200 observations of data set 0 for each of the two networks. The other data sets and larger M require simulation with GeneNetWeaver (`data/gene_networks/simulate.py`).
4. One appendix table on the exactness and results of each version of the algorithm has no config. Every other table or figure with results has a config.

In `sample_size.yaml`, 29 of the 480 reruns with M = 50 and M = 100 differ from their stored runs. All 29 are IAMB at the level alpha. The stored runs used another implementation of the Fisher-z test, and the two give different p-values on the same data when few degrees of freedom are left. The mean F1 of IAMB at the level alpha is 0.248 versus the expected 0.249 (40 series, M = 50) and 0.097 versus 0.099 (80 series, M = 50). The paper shows IAMB at its better level, the corrected level in these cells, and those means equal the expected means.

## Rerun a configuration

```bash
python -m itpd.experiments run configs/main_setting.yaml --out out/main_setting --methods ITPD --M 500
python -m itpd.experiments table configs/main_setting.yaml --runs out/main_setting --compare
```

- `run` writes one result file per run into `--out` and skips finished runs. An interrupted run restarts with the same command.
- `--graphs` (such as `0-2`), `--methods` (such as `ITPD,IAMB`) and `--M` (such as `200,500`) select a subset.
- `table --compare` prints one table per table or figure of the config, with each mean next to its expected mean. It shows a method with two levels (alpha and corrected) at its better level. A method has a line only where it has a run file. A mean matches the paper only when all graphs of the cell ran.
- The last line gives the largest difference in the lines shown and the largest difference at the levels not shown.
- Each external tool uses its own Python interpreter. See the top of the files in `requirements/`.
