# itpd

Instrumented implementation of ITPD (Iterative Temporal Parent Discovery), ITPD_naive, the full-conditioning baseline, the blanket-screened shrink and the other
known-order methods, on one CI-test foundation: every method calls the same counting wrapper, so the numbers of tests are comparable.
Conventions: time is 0-based, the process starts at t = 0 (those nodes are roots); node (variable n, time t) has column t * N + n;
`A[i, j] = 1` iff i -> j. Setting S1 = one long series, setting S2 = M replicates of a T-step process.

## Module map (read in this order)

1. `graphs.py` time graph, unrolled graph, d-separation (`DSep`, the reference oracle).
2. `sim.py` graph sampling and data generation (linear and nonlinear, window and time graphs).
3. `instances.py` stored graph instances: structure, weights, seeds, hashes. `observed_data.py` builds the data and the truth from them
   (data seed, optional assumption violation or nonstationarity).
4. `nonlinear.py` nonlinear additive-noise instances.
5. `ci.py` CI tests (`DSepCI`, `FisherZ`, `GCM`) and `Recorder`, the only place where tests are counted. `DSepCI` uses the optional `fastdsep` C extension when it is installed and the pure-Python `DSep` of `graphs.py` otherwise; `ITPD_ORACLE=old` forces `DSep`.
6. `padl.py` PaDL (parents of one target); `itpd.py` ITPD_naive and ITPD (S2 on the unrolled graph, S1 on a window).
7. `baselines/full_conditioning.py`, `iamb.py`, `blanket_screened_shrink.py`, `lasso.py` the other methods (full conditioning, IAMB per target, blanket-screened shrink, lasso).
8. `metrics.py` precision, recall, F1, SHD on directed time-indexed edges (self edges excluded unless asked).
9. `methods_registry.py` the one table of method names (below) and `OLD_TO_NEW`, the names used in result files written before the rename.
10. `method_runner.py` run one method by name and score it; `dataset_eval.py` run a list of methods on one dataset (one shared p-value memo).
11. Drivers, one per kind of experiment (table below): `run_oracle_counts.py`, `run_finite_data.py`, `run_nonlinear.py`,
    `run_single_series.py`, `run_robustness.py`, `run_known_order_baselines.py`, `run_blanket_screened_shrink.py`. Collectors that turn their JSON into tables are in `scripts/`.

## Counting conventions

- raw = every call that reaches the `Recorder`; unique = distinct (unordered pair, sorted conditioning set), the headline count.
- Marginals are memoised inside one PaDL call, so raw counts match the original code in `legacy/`.
- lazy: a test is issued only when its result can change a label; nonlazy: both tests of every step are issued (the evaluation of the original code).
  Both give the same graphs; the lazy count is the headline and the nonlazy count sits beside it as `<name>_nonlazy`.
- A test the data cannot evaluate (Fisher-z with n - |S| - 3 <= 0) is never answered "independent": the target is infeasible, scored apart.

## Methods

Every method name written to a result file is defined once in `methods_registry.py`. Names in files written before the rename are translated with
`OLD_TO_NEW` (for example `itpd_nl` is `itpd_nonlazy`, `order` and `order_based` are `full_conditioning`, `hpv_rc_eq` and `hpv_safe` are `blanket_screened_shrink_recheck`); every collector in `scripts/` reads both.
Blanket-screened shrink names are `blanket_screened_shrink[_recheck][_<screening>][_lenient]`: screening `learned_blanket` and equal alpha are the defaults and are omitted. `screening` is the set a candidate is screened given besides X; the screening set of a candidate Z is this set (for `learned_blanket`, the Markov blanket of Z in the graph learned for earlier steps) plus X.

| name | function | screening | alpha rule | counting | meaning |
|---|---|---|---|---|---|
| `itpd_naive` | `itpd_naive` | - | single | lazy | PaDL once per target, no cross-target skip rules (paper step 4) |
| `itpd_naive_nonlazy` | `itpd_naive` | - | single | nonlazy | itpd_naive with both tests of every step issued (evaluation of the original code) |
| `itpd` | `itpd` | - | single | lazy | ITPD: PaDL with the skip rules adjacency, de_z1, de_z5, de_z4 (paper step 4) |
| `itpd_nonlazy` | `itpd` | - | single | nonlazy | itpd with both tests of every step issued |
| `itpd_repo_variant` | `itpd` | - | single | lazy | ITPD with step 4 of the original code (extra marginal conjunct) |
| `itpd_repo_variant_nonlazy` | `itpd` | - | single | nonlazy | itpd_repo_variant with both tests of every step issued |
| `itpd_adjacency_self` | `itpd_adjacency_self` | - | single | lazy | ITPD plus the optional adjacency_self rule |
| `itpd_adjacency_self_nonlazy` | `itpd_adjacency_self` | - | single | nonlazy | itpd_adjacency_self with both tests of every step issued |
| `itpd_marginal_first` | `itpd_marginal_first` | - | single | lazy | ITPD with the Z8 step never skipped and the Y-marginal tested first |
| `itpd_marginal_first_nonlazy` | `itpd_marginal_first` | - | single | nonlazy | itpd_marginal_first with both tests of every step issued |
| `full_conditioning` | `full_conditioning` | - | single | every_issued_test | full-conditioning baseline: one test per candidate given all other earlier nodes |
| `iamb_known_order` | `iamb_known_order` | - | single | every_issued_test | IAMB per target, self edge known; ties in the grow step go to the latest candidate |
| `iamb_known_order_tie_first` | `iamb_known_order` | - | single | every_issued_test | iamb_known_order with ties going to the first candidate (oracle check of the tie rule) |
| `iamb_known_order_tie_random` | `iamb_known_order` | - | single | every_issued_test | iamb_known_order with ties broken by a fixed random order, seed 7 (oracle check of the tie rule) |
| `blanket_screened_shrink` | `blanket_screened_shrink` | learned_blanket | equal | every_issued_test | blanket-screened shrink: screen each candidate given its learned blanket and X, then shrink the survivors (no re-check) |
| `blanket_screened_shrink_lenient` | `blanket_screened_shrink` | learned_blanket | lenient | every_issued_test | blanket_screened_shrink with a lenient screen (alpha_A = 0.1) |
| `blanket_screened_shrink_recheck` | `blanket_screened_shrink` | learned_blanket | equal | every_issued_test | blanket_screened_shrink plus the re-check of the pruned candidates and a final verification (always-verify) |
| `blanket_screened_shrink_recheck_lenient` | `blanket_screened_shrink` | learned_blanket | lenient | every_issued_test | blanket_screened_shrink_recheck with a lenient screen (alpha_A = 0.1) |
| `blanket_screened_shrink_oracle_blanket` | `blanket_screened_shrink` | oracle_blanket | equal | every_issued_test | blanket_screened_shrink with the blanket of the TRUE graph as the screening set (reads the truth; upper bound) |
| `blanket_screened_shrink_oracle_blanket_lenient` | `blanket_screened_shrink` | oracle_blanket | lenient | every_issued_test | blanket_screened_shrink_oracle_blanket with a lenient screen (alpha_A = 0.1) |
| `blanket_screened_shrink_x_only` | `blanket_screened_shrink` | none | equal | every_issued_test | blanket_screened_shrink with the screening set X only |
| `blanket_screened_shrink_union` | `blanket_screened_shrink` | union | equal | every_issued_test | blanket_screened_shrink with the learned blanket plus the shifted parents as screening set |
| `blanket_screened_shrink_shifted_parents` | `blanket_screened_shrink` | shifted_parents | equal | every_issued_test | blanket_screened_shrink with the learned parents of X shifted one step forward as screening set |
| `lasso_cv` | `lasso.lasso_s2` | - | penalty | no_ci_tests | known-order lasso, penalty by 5-fold cross-validation |
| `lasso_path` | `lasso.lasso_s2` | - | penalty | no_ci_tests | known-order lasso at one of 13 fixed penalties (one point of a recall / false-positive curve) |
| `lasso_ebic` | `lasso.lasso_s2` | - | penalty | no_ci_tests | known-order lasso, penalty by extended BIC (the headline lasso) |
| `lasso_fixed` | `lasso.lasso_s2` | - | penalty | no_ci_tests | known-order lasso, one fixed penalty sqrt(2 log p / M) |
| `lasso_path_all_lambdas` | `lasso.lasso_s2` | - | penalty | no_ci_tests | wall-clock of the whole 13-penalty lasso path (timing rows) |
| `itpd_last_slice` | `itpd.run_s1` | - | single | lazy | single long series: ITPD for each series once, on the last window slice |
| `full_conditioning_last_slice` | `full_conditioning.run_s1` | - | single | every_issued_test | single long series: full-conditioning baseline on the last window slice |
| `itpd_naive_every_slice` | `itpd.run_s2` | - | single | lazy | single long series: itpd_naive on every window slice, outputs unioned |
| `itpd_every_slice` | `itpd.run_s2` | - | single | lazy | single long series: ITPD on every window slice, outputs unioned |

## Experiment -> command

Run from the repository root, after `pip install -e '.[test]'` or with `PYTHONPATH=.`. Put your cluster job scripts in the ignored folder `jobs/`; they are not part of the repository. The drivers and collectors that read earlier runs (layout below) look for them in the directory named by the environment variable `ITPD_RESULTS`, or in `results` when it is not set; the option `--results` overrides this where a script has it.

The drivers write one file per task and skip a task whose file exists, so a stopped run resumes when you run the same command again. `--workers K` sets the number of processes and `--budget-sec S` stops starting new tasks after S seconds. The last line a driver prints ends with `remaining N`; `remaining 0` means that all tasks are finished.

| experiment | command | collector |
|---|---|---|
| oracle exactness and counts | `python -m itpd.run_oracle_counts --N 20 --T 16 --tau 2 --d 2 --graphs 20 --out OUT.json` or the grid `python scripts/oracle_grid.py --out-dir results/oracle/window --graph window --instances-dir results/instances/window` | `scripts/oracle_collect.py DIR`, `scripts/oracle_scaling_slopes.py DIR` |
| large oracle cells | `python scripts/large_oracle_run.py OUT --cells 80x25 --graphs 0-4` | `scripts/large_oracle_collect.py OUT --oracle DIR` |
| finite data (Fisher-z) | `python -m itpd.run_finite_data --out-dir results/finite/window --arm window --N 10,20 --T 8,16 --M 50,100,200,500,2000 --graphs 20` | `scripts/finite_collect.py DIR` |
| nonlinear data (GCM) | `python -m itpd.run_nonlinear --out-dir DIR --N 10 --T 8 --M 500,2000 --graphs 20` | `scripts/nonlinear_collect.py DIR` |
| single series (S1) | `python -m itpd.run_single_series --out-dir DIR` | `scripts/single_series_collect.py DIR` |
| robustness | `python -m itpd.run_robustness --out-dir DIR --exp nonstationary` or `--exp violations` | `scripts/robustness_collect.py DIR --exp nonstationary` or `--exp violations` |
| baselines IAMB, lasso, ITPD + marginal-first | `python -m itpd.run_known_order_baselines oracle\|finite\|lasso_ebic_fixed\|timing --out-dir DIR` | `scripts/known_order_baselines_collect.py` |
| blanket-screened shrink | `python -m itpd.run_blanket_screened_shrink oracle\|finite --out-dir DIR` | `scripts/blanket_screened_shrink_collect.py`, `scripts/blanket_screened_shrink_tables.py` |
| tests | `python -m pytest -q` | |

`run_blanket_screened_shrink`, `run_known_order_baselines` and the collectors `blanket_screened_shrink_collect.py`, `blanket_screened_shrink_tables.py` and `known_order_baselines_collect.py` read the output of earlier runs from the results directory `R`. They expect this layout:

| directory under `R` | written by |
|---|---|
| `oracle/<arm>/N<N>_T<T>_tau<tau>_d<d>.json` | `scripts/oracle_grid.py --out-dir R/oracle/<arm> --graph <arm>` (arm `time` or `window`) |
| `instances/<arm>/*.npz` | the same command with `--instances-dir R/instances/<arm>` |
| `finite/window/<cell>/` and `finite/window/instances/<cell>/` | `run_finite_data --out-dir R/finite/window --arm window` |
| `blanket_screened_shrink/oracle/`, `blanket_screened_shrink/single_pass/`, `blanket_screened_shrink/recheck/` | `run_blanket_screened_shrink oracle`, `run_blanket_screened_shrink finite`, `run_blanket_screened_shrink finite --specs blanket_screened_shrink_recheck,blanket_screened_shrink_recheck_lenient` |
| `baselines/{oracle,finite,lasso_ebic_fixed,timing}/` | `run_known_order_baselines` with the mode of the same name |

Simulated data can differ at the 1e-15 level between CPU generations (for example AMD Zen 3 and Zen 4). `run_blanket_screened_shrink` and `run_known_order_baselines` regenerate the data of a stored instance and assert that its hash equals the hash stored with the instance, so run them on the CPU type that wrote the instances.
