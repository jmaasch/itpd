"""Collectors: they turn the JSON files of the experiments into markdown tables and csv files. One command line:
`python -m itpd.tables <name> [args]`; `<name> --help` shows the arguments of one table. A module also runs on its own:
`python -m itpd.tables.<name> [args]`.

Numbers only: every table gives a median [Q1, Q3] over graphs, a mean with a bootstrap interval, or a count. A directory DIR is the
output directory of the experiment of the same name (`python -m itpd.experiments <name>`); R is the results directory of the
earlier runs (`--results` where the command has it, else `$ITPD_RESULTS`, else `./results`). Results written before the
method-name rename are read with the new names.

| name | what it does | reads | writes |
|---|---|---|---|
| `oracle_counts cells DIR` | exact recovery, unique tests by method, factor and conditioning-set size, scaling, the tests of an appended time step | `DIR/N*.json` | `DIR/TABLES.md`, `DIR/aggregates.json` (`--out` moves the markdown) |
| `oracle_counts large BIG_DIR --oracle DIR` | large cells next to the small cells at tau 1, d 2: tests per candidate, largest sets, slopes, wall-clock | `BIG_DIR/<arm>/N*_T*_g*_<method>.json`, `DIR/N*_tau1_d2.json` | `BIG_DIR/TABLES-<arm>.md` |
| `oracle_counts slopes DIR` | log-log slopes of the unique tests with bootstrap intervals over graphs | `DIR/N*.json` | `DIR/SLOPES.md`, `DIR/SLOPES.json` |
| `finite_data DIR` | Fisher-z runs: feasibility, accuracy on own-feasible and common targets, matched false positives, paired differences, tests by size, alpha sensitivity, pairing with the oracle runs (`--oracle`) | `DIR/*/g*_M*.json`, `DIR/instances/` | `DIR/TABLES.md` |
| `nonlinear_data DIR` | GCM runs: tests, accuracy, matched false positives, cost per test, comparison with the Fisher-z runs on the same graphs (`--lin`) | `DIR/nonlinear_*/g*_M*.json` | `DIR/TABLES.md` |
| `single_series DIR` | last slice against every slice of one long series | `DIR/N*/g*.json` | `DIR/TABLES.md` |
| `robustness DIR --exp nonstationary` or `violations` | accuracy and test counts per setting, paired differences against the base setting | `DIR/<exp>/*/g*_M*.json` | `DIR/<exp>/TABLES.md` |
| `stored_instances known_order` | IAMB, ITPD + marginal-first and lasso on the stored instances: oracle check, finite-data tables, paired differences, matched false positives, wall-clock | `OUT/{oracle,finite,lasso_ebic_fixed,timing}`, `R/finite/window`, `R/oracle`, `R/blanket_screened_shrink/{single_pass,recheck,oracle}` | `OUT/TABLES.md`, `OUT/aggregates.json` (`--out`, default `R/baselines`) |
| `stored_instances shrink` | blanket-screened shrink against the stored rows of ITPD, ITPD_naive and full conditioning: oracle tables (`--only oracle`), finite-data tables (`--only finite`), pooled recall-FP curves and matched false positives (`--only curves`) | `R/blanket_screened_shrink/{oracle,single_pass,recheck}`, `R/oracle`, `R/finite/window` | `tables_oracle.md`, `tables_finite.md`, `aggregates.json`; `curves.csv`, `matched_fp.csv`, `tables.md`, `aggregates_curves.json` (`--out`, default `R/blanket_screened_shrink`) |
"""
