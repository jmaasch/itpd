# Configurations

- Each YAML file in `configs/` describes the runs behind one table, one figure or one statement of the paper. Its first line names the item. A run is one method at one significance level on one graph of one setting with M observations.
- `configs/expected/<config>.csv` has the expected means over graphs from the stored runs of the paper's experiments, for the methods in the package.
- The method ITPD is the class `ITPD` with the constants of the paper (alpha = 0.01, gamma = 0.2, kappa = 50). The CI test comes from the config. An entry with a label is ITPD with other constants (`gamma_kappa.yaml`) or with the ablation switches in `src/itpd/experiments/ablations.py`. The entry "Algorithm 1" sets no switch. `--methods` takes the label.
- PCMCI, DF-PC, DYNOTEARS and CD-NOD are external tools (see item 3 below).

| Config | Reproduces | Methods | Status | Full run |
|---|---|---|---|---|
| `oracle_exact_recovery.yaml` | Table: unique CI tests per candidate and conditioning set sizes under the exact oracle | ITPD, IAMB, IPaDL, FullCI | checked against 205 stored runs; 3795 of 4000 runs not rerun | about 13 h |
| `oracle_all_settings.yaml` | Text: exact recovery in all 32 exact-oracle settings | ITPD, IAMB, IPaDL, FullCI | checked against 40 stored runs; 16920 of 16960 runs not rerun | more than 82 h |
| `main_setting_f1.yaml` | Table: mean F1 in the main finite data setting | ITPD, IAMB, IPaDL, FullCI, Lasso, PCMCI, DF-PC, DYNOTEARS, CD-NOD | checked against 294 stored runs; 3306 of 3600 runs not rerun | about 5.2 h |
| `main_setting_tests.yaml` | Table: unique CI tests per candidate and conditioning set sizes in the main finite data setting | ITPD, IAMB, IPaDL, FullCI, PCMCI, DF-PC | checked against 256 stored runs; 624 of 880 runs not rerun | about 2.2 h |
| `frontier_seconds.yaml` | Figure: mean F1 against wall-clock time in the main setting (Pareto frontier) | ITPD, IAMB, IPaDL, FullCI, Lasso, PCMCI, DF-PC, DYNOTEARS | checked against 104 stored runs; 416 of 520 runs not rerun | about 66 min |
| `frontier_tests.yaml` | Figure: mean F1 against unique CI tests per candidate in the main setting | ITPD, IAMB, IPaDL, FullCI, DF-PC | checked against 216 stored runs; 144 of 360 runs not rerun | about 66 min |
| `lag.yaml` | Figure: mean F1 on common targets against the true lag on stationary and nonstationary graphs | ITPD, IAMB, IPaDL, FullCI, Lasso, DF-PC, PCMCI, DYNOTEARS, CD-NOD | checked against 1110 stored runs; 4890 of 6000 runs not rerun | about 1.5 h |
| `redrawn_edges.yaml` | Figure: mean F1 against the share of edges redrawn at each time step | ITPD, IAMB, IPaDL, FullCI, Lasso, PCMCI, DF-PC, DYNOTEARS, CD-NOD | checked against 376 stored runs; 4424 of 4800 runs not rerun | about 8.9 h |
| `sample_size.yaml` | Figure: mean F1 on all targets against the number of observations | ITPD, IAMB, IPaDL, FullCI, Lasso, PCMCI, DF-PC, DYNOTEARS | checked against 123 stored runs; 1 run differs (item 2 below); 1826 of 1950 runs not rerun | more than 3.5 h |
| `nonlinear.yaml` | Table: mean F1 with nonlinear causal mechanisms | ITPD, IAMB, IPaDL, FullCI, Lasso, DYNOTEARS | checked against 41 stored runs; 40 runs without a stored per-run result; 519 of 600 runs not rerun | more than 2.7 h |
| `scalability.yaml` | Table: mean F1 and unique CI tests per candidate on larger graphs | ITPD, IAMB, IPaDL, FullCI, Lasso, PCMCI, DF-PC, CD-NOD | checked against 23 stored runs; 257 of 280 runs not rerun | more than 3.6 h |
| `runtime_itpd_iamb.yaml` | Table: seconds per run of ITPD and IAMB | ITPD, IAMB | checked against 56 stored runs; 64 of 120 runs not rerun | more than 7.6 h |
| `scaling_time.yaml` | Figure: seconds per run and F1 of ITPD against the number of time-indexed variables | ITPD, IAMB, Lasso, IPaDL, FullCI, PCMCI, DF-PC | checked against 33 stored runs; 172 of 205 runs not rerun | more than 5.4 h |
| `violation_no_self_edge.yaml` | Table: mean F1 with one assumption violated (no lag-1 self edge) | ITPD, IAMB, FullCI, Lasso, PCMCI | checked against 235 stored runs; 1365 of 1600 runs not rerun | more than 2.5 h |
| `violation_same_step_edges.yaml` | Table: mean F1 with edges moved into the same time step | ITPD, IAMB, FullCI, Lasso, PCMCI | checked against 210 stored runs; 750 of 960 runs not rerun | more than 2.8 h |
| `gene_networks.yaml` | Table: mean F1 on simulated gene regulatory networks (regulatory edges) | ITPD, IAMB, IPaDL, FullCI, DF-PC, Lasso, PCMCI, CD-NOD, DYNOTEARS | checked against 24 stored runs; 876 of 900 runs not rerun | more than 50 min |
| `gene_networks_indirect.yaml` | Table: mean F1 on simulated gene regulatory networks (regulatory edges, indirect pairs left out) | ITPD, IAMB, IPaDL, FullCI, DF-PC, Lasso, PCMCI, CD-NOD, DYNOTEARS | checked against 16 stored runs; 884 of 900 runs not rerun | more than 50 min |
| `extended_bic.yaml` | Table: mean F1 of ITPD with alpha / m and with the extended-BIC level in Phase II | ITPD, extended BIC | checked against 274 stored runs; 42 of 316 runs not rerun | more than 2.2 h |
| `gamma_kappa.yaml` | Text: choice of gamma and kappa | ITPD, alpha1 = 0.01, alpha1 = 0.05, alpha1 = 0.1, alpha1 = 0.2 | checked against 458 stored runs; 842 of 1300 runs not rerun | more than 2.4 h |
| `larger_in_degree.yaml` | Text: unique CI tests per candidate of ITPD and IAMB at larger in-degrees | ITPD, IAMB | checked against 156 stored runs; 234 of 390 runs not rerun | about 1.8 h |
| `dropped_steps.yaml` | Table: steps dropped from Algorithm 1, each added back alone | Algorithm 1, With Lines 9' and 10', Line 11 with the members of S at the step of X | checked against 240 stored runs; 480 of 720 runs not rerun | about 46 min |
| `ablation.yaml` | Table: one change at a time to ITPD | Algorithm 1, and 11 entries with one change each | checked against 190 stored runs; 5810 of 6000 runs not rerun | about 13 h |

- Status "checked against N stored runs": N runs were rerun from this code and compared with the stored per-run results of the paper's experiments. Every stored number of these N runs equals the rerun: counts of true positives, false positives and false negatives, CI test counts, conditioning set sizes, graph and data hashes, and run status, where the stored result holds them. Seconds are never compared.
- Status "K of T runs not rerun": the config has T runs, and K were not rerun due to runtime, external tools or missing gene-network data.
- Full run: the rough time on one core of a shared machine for all runs of the config with the package methods. "more than" means that some runs have no timing.
- The 22 configs have 53361 runs, and 4721 were rerun with Python 3.12, `requirements/constraints.txt` and one BLAS thread.
- Of the 4721 reruns, 4681 have a stored per-run result: 4680 equal it and 1 differs (`sample_size.yaml`, item 2 below). The remaining 40 are Lasso on the nonstationary graphs of `nonlinear.yaml`; they have no stored per-run result, and their four cell means equal the expected means.
- A cell is one setting, one M, one method and one level. All 435 completely rerun cell means equal the expected mean to three decimals.

## Not reproduced by this repository

1. Wall-clock seconds in `frontier_seconds.yaml`, `runtime_itpd_iamb.yaml` and `scaling_time.yaml` depend on the machine and its load. `configs/expected` has no seconds, and no seconds were compared. These three configs were checked on their edges and CI test counts only.
2. One run of `sample_size.yaml` differs from its stored run: IAMB at the level alpha, 40 series, M = 50, graph 0. The rerun has 635 true positives, 3117 false positives, and 572 false negatives, versus 637, 3115, and 570 in the stored run. The count of CI tests, the set sizes, and the hash of the data are equal. The stored run used a faster form of the Fisher-z test, and the repository holds the plain form only. Here the conditioning sets reach 46 variables with 50 observations, and the two forms differ by up to 0.04 in the p-value. The stored run files of `sample_size.yaml` and `scalability.yaml` name the faster form or no form, and the other 146 reruns of these two configs equal their stored runs. The runs of the baselines at the level alpha with 40 and 80 series and M = 50 or 100 on graphs 1-9 were not rerun. `configs/expected/sample_size.csv` marks the corrected level as the better level of IAMB in this cell, so the figure does not show the mean of this cell at the level alpha.
3. External tools (PCMCI, DF-PC, DYNOTEARS and CD-NOD) are not in the repository, which has only wrappers in `src/itpd/external/` and pinned versions in `requirements/`. 14 configs have such runs. 50 of these runs were rerun with the tools installed separately, and all 50 equal their stored runs. The nonlinear table in the paper also includes PCMCI and DF-PC with a GCM test, and the package does not run them.
4. For gene-network data, the repository has only the first 200 observations of data set 0 for each of the two networks. The other 9 data sets per network, and M = 500 and M = 2000, require simulation with GeneNetWeaver (`data/gene_networks/simulate.py`). No cell mean of `gene_networks.yaml` or `gene_networks_indirect.yaml`, and none of the gene-network cells of `extended_bic.yaml`, was rerun completely.
5. One appendix table on the exactness and results of each version of the algorithm has no config. Every other table or figure with results has a config.

## Rerun a configuration

```bash
python -m itpd.experiments run configs/frontier_tests.yaml --out out/frontier_tests --methods ITPD
python -m itpd.experiments table configs/frontier_tests.yaml --runs out/frontier_tests --compare
```

- `run` uses one BLAS thread per run and writes one result file per run into `--out`. It skips finished runs, so an interrupted run restarts with the same command.
- `--graphs` (such as `0-2`), `--methods` (such as `ITPD,IAMB`) and `--M` (such as `200,500`) select a subset. Without them, `run` executes all runs of the config.
- `table --compare` prints the graph count, mean, expected mean and difference for every cell. A mean matches the paper only when all graphs of the cell ran; the last line counts means "with another number of graphs".
- The example above makes 40 runs in about 2 minutes, and all 4 compared means have difference `0.0e+00`.
- Each external tool uses its own Python interpreter; see the top of the files in `requirements/` for setup and environment variables such as `ITPD_PYTHON_PCMCI`. Without the variable, those runs get status "skipped".
- Simulated data pass through BLAS, and another CPU or BLAS build can change their last bits. Tests on stored runs then skip with "the data differ on this platform".
