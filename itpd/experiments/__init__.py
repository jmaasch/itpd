"""Experiment drivers. One command line: `python -m itpd.experiments <name> [args]`; `<name> --help` shows the arguments of one
experiment. A module also runs on its own: `python -m itpd.experiments.<name> [args]`.

Every driver is resumable (a task file that exists is never recomputed). The drivers that run many tasks start no new task
after `--budget-sec` seconds (`--start-by-sec` for `oracle_counts large`): repeat the same command until its last line says
`remaining 0`. R is the results directory of the earlier runs, OUT and DIR are output directories.

| name | what it does | reads | writes |
|---|---|---|---|
| `oracle_counts cell` | exactness and test counts of the ITPD family and full conditioning on random S2 graphs with the exact d-separation oracle, one cell (N, T, tau, d) | - | `--out` (one JSON), `--instances-dir/*.npz` |
| `oracle_counts grid` | all cells of an (N, T, tau, d) grid in parallel | - | `OUT/N<N>_T<T>_tau<tau>_d<d>.json`, instances |
| `oracle_counts large` | large cells, one process per (arm, N, T, graph, method) | - | `OUT/<arm>/N<N>_T<T>_g<g>_<method>.json`, `OUT/instances/<arm>/` |
| `finite_data` | Fisher-z runs on linear-Gaussian S2 data, alpha grid, one JSON per (cell, graph, M) | - | `DIR/<cell>/g<idx>_M<M>.json`, `DIR/instances/<cell>/g<idx>.npz` |
| `nonlinear_data` | gradient-boosting GCM test on nonlinear additive-noise data | - | `DIR/<cell>/g<idx>_M<M>.json`, `DIR/instances/<cell>/`, `DIR/memo/` |
| `single_series` | setting S1, one long series: one run on the last window slice against a run on every slice | - | `DIR/N<N>_T<T>_tau<tau>_d<d>/g<idx>.json`, `DIR/instances/` |
| `robustness` | nonstationary replicates (`--exp nonstationary`) and assumption violations (`--exp violations`) | - | `DIR/<exp>/<setting>/g<idx>_M<M>.json`, `DIR/instances/<exp>/<setting>/` |
| `stored_instances known_order oracle` | IAMB per target and ITPD + marginal-first with the oracle, paired with the earlier oracle runs | `R/oracle`, `R/instances/<arm>` | `OUT/<arm>/N<N>_T<T>_tau<tau>_d<d>.json` |
| `stored_instances known_order finite` | IAMB, ITPD + marginal-first and lasso on the stored finite-data instances | `R/finite/window` | `OUT/<cell>/g<idx>_M<M>.json` |
| `stored_instances known_order lasso_ebic_fixed` | extended-BIC and fixed-penalty lasso on the same instances | `R/finite/window` | `OUT/<cell>/g<idx>_M<M>.json` |
| `stored_instances known_order timing` | wall-clock per method with a fresh test object for each method | `R/finite/window` | `OUT/<cell>/g<idx>_M<M>.json` |
| `stored_instances itpd_s oracle` | ITPD-S variants with the oracle on stored instances | `R/oracle`, `R/instances/<arm>` | `OUT/<arm>/N<N>_T<T>_tau<tau>_d2/g<idx>.json` |
| `stored_instances itpd_s finite` | ITPD-S and ITPD-S+ on the stored finite-data instances | `R/finite/window` | `OUT/<cell>/g<idx>_M<M>.json` |
"""
