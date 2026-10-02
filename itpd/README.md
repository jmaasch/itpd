# itpd

Instrumented implementation of ITPD (Iterative Temporal Parent Discovery), ITPD_naive, the order-based baseline and the other
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
7. `baselines/order_based.py`, `iamb.py`, `hpv.py`, `lasso.py` the other methods (order-based, IAMB per target, hint-pruned verification, lasso).
8. `metrics.py` precision, recall, F1, SHD on directed time-indexed edges (self edges excluded unless asked).
9. `methods_registry.py` the one table of method names (below) and `OLD_TO_NEW`, the names used in result files written before the rename.
10. `method_runner.py` run one method by name and score it; `dataset_eval.py` run a list of methods on one dataset (one shared p-value memo).
11. Drivers, one per kind of experiment (table below): `run_oracle_counts.py`, `run_finite_data.py`, `run_nonlinear.py`,
    `run_single_series.py`, `run_robustness.py`, `run_known_order_baselines.py`, `run_hpv.py`. Collectors that turn their JSON into tables are in `scripts/`.

## Counting conventions

- raw = every call that reaches the `Recorder`; unique = distinct (unordered pair, sorted conditioning set), the headline count.
- Marginals are memoised inside one PaDL call, so raw counts match the original code in `legacy/`.
- lazy: a test is issued only when its result can change a label; nonlazy: both tests of every step are issued (the evaluation of the original code).
  Both give the same graphs; the lazy count is the headline and the nonlazy count sits beside it as `<name>_nonlazy`.
- A test the data cannot evaluate (Fisher-z with n - |S| - 3 <= 0) is never answered "independent": the target is infeasible, scored apart.

## Methods

Every method name written to a result file is defined once in `methods_registry.py`. Names in files written before the rename are translated with
`OLD_TO_NEW` (for example `itpd_nl` is `itpd_nonlazy`, `order` is `order_based`, `hpv_rc_eq` is `hpv_safe`); every collector in `scripts/` reads both.
HPV names are `hpv_<single_pass|safe>[_<hint>][_lenient]`: hint `learned_blanket` and equal alpha are the defaults and are omitted.

| name | function | hint | alpha rule | counting | meaning |
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
| `order_based` | `order_based` | - | single | every_issued_test | order-based baseline: one test per candidate given all earlier nodes |
| `iamb_known_order` | `iamb_known_order` | - | single | every_issued_test | IAMB per target, self edge known; ties in the grow step go to the latest candidate |
| `iamb_known_order_tie_first` | `iamb_known_order` | - | single | every_issued_test | iamb_known_order with ties going to the first candidate (oracle check of the tie rule) |
| `iamb_known_order_tie_random` | `iamb_known_order` | - | single | every_issued_test | iamb_known_order with ties broken by a fixed random order, seed 7 (oracle check of the tie rule) |
| `hpv_single_pass` | `hpv` | learned_blanket | equal | every_issued_test | HPV: screen each candidate given its learned blanket, then verify the survivors (no re-check) |
| `hpv_single_pass_lenient` | `hpv` | learned_blanket | lenient | every_issued_test | hpv_single_pass with a lenient screen (alpha_A = 0.1) |
| `hpv_safe` | `hpv` | learned_blanket | equal | every_issued_test | HPV-safe: hpv_single_pass plus the re-check of the pruned candidates and a final verification (always-verify) |
| `hpv_safe_lenient` | `hpv` | learned_blanket | lenient | every_issued_test | hpv_safe with a lenient screen (alpha_A = 0.1) |
| `hpv_single_pass_oracle_blanket` | `hpv` | oracle_blanket | equal | every_issued_test | hpv_single_pass with the blanket of the TRUE graph as hint (reads the truth; upper bound) |
| `hpv_single_pass_oracle_blanket_lenient` | `hpv` | oracle_blanket | lenient | every_issued_test | hpv_single_pass_oracle_blanket with a lenient screen (alpha_A = 0.1) |
| `hpv_single_pass_no_hint` | `hpv` | none | equal | every_issued_test | hpv_single_pass with the empty hint: screen given X only |
| `hpv_single_pass_union` | `hpv` | union | equal | every_issued_test | hpv_single_pass with learned blanket plus shifted parents as hint |
| `hpv_single_pass_shifted_parents` | `hpv` | shifted_parents | equal | every_issued_test | hpv_single_pass with the learned parents of X shifted one step forward as hint |
| `lasso_cv` | `lasso.lasso_s2` | - | penalty | no_ci_tests | known-order lasso, penalty by 5-fold cross-validation |
| `lasso_path` | `lasso.lasso_s2` | - | penalty | no_ci_tests | known-order lasso at one of 13 fixed penalties (one point of a recall / false-positive curve) |
| `lasso_ebic` | `lasso.lasso_s2` | - | penalty | no_ci_tests | known-order lasso, penalty by extended BIC (the headline lasso) |
| `lasso_fixed` | `lasso.lasso_s2` | - | penalty | no_ci_tests | known-order lasso, one fixed penalty sqrt(2 log p / M) |
| `lasso_path_all_lambdas` | `lasso.lasso_s2` | - | penalty | no_ci_tests | wall-clock of the whole 13-penalty lasso path (timing rows) |
| `itpd_last_slice` | `itpd.run_s1` | - | single | lazy | single long series: ITPD for each series once, on the last window slice |
| `order_based_last_slice` | `order_based.run_s1` | - | single | every_issued_test | single long series: order-based baseline on the last window slice |
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
| HPV | `python -m itpd.run_hpv oracle\|finite --out-dir DIR` | `scripts/hpv_collect.py`, `scripts/hpv_tables.py` |
| tests | `python -m pytest -q` | |

`run_hpv`, `run_known_order_baselines` and the collectors `hpv_collect.py`, `hpv_tables.py` and `known_order_baselines_collect.py` read the output of earlier runs from the results directory `R`. They expect this layout:

| directory under `R` | written by |
|---|---|
| `oracle/<arm>/N<N>_T<T>_tau<tau>_d<d>.json` | `scripts/oracle_grid.py --out-dir R/oracle/<arm> --graph <arm>` (arm `time` or `window`) |
| `instances/<arm>/*.npz` | the same command with `--instances-dir R/instances/<arm>` |
| `finite/window/<cell>/` and `finite/window/instances/<cell>/` | `run_finite_data --out-dir R/finite/window --arm window` |
| `hpv/oracle/`, `hpv/single_pass/`, `hpv/safe/` | `run_hpv oracle`, `run_hpv finite`, `run_hpv finite --specs hpv_safe,hpv_safe_lenient` |
| `baselines/{oracle,finite,lasso_ebic_fixed,timing}/` | `run_known_order_baselines` with the mode of the same name |

Simulated data can differ at the 1e-15 level between CPU generations (for example AMD Zen 3 and Zen 4). `run_hpv` and `run_known_order_baselines` regenerate the data of a stored instance and assert that its hash equals the hash stored with the instance, so run them on the CPU type that wrote the instances.
