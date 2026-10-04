"""The one table of method names.

Every method name that appears in a result file (`name` of a row, `method` of a runner output) is defined here, with the
function that runs it, its fixed options, its screening set, its alpha rule, its counting mode and a one-line meaning.
Drivers build their (name, alphas) lists from this table (`spec`); readers of old result files translate with `OLD_TO_NEW`.

Row identity is the name: the lazy and the non-lazy counting of the same method are two rows, `itpd` and `itpd_nonlazy`.

Columns
  name        the label written to result files
  function    `method` string of `method_runner.run_s2_method` (or the lasso function, or the S1 / last-slice runner)
  options     fixed options of that call: variant (PaDL step 4: "paper" | "repo"), lazy (bool recorded in the row),
              shrink (options of `itpd_s.run_s2`), tie (IAMB tie rule)
  screening   ITPD-S and ITPD-S+ rows only, the screening conditioning set, the set a candidate is screened given, besides X:
              learned_blanket (estimated Markov blanket in the graph learned so far), true_blanket (blanket in the TRUE graph, an upper
              bound that reads the truth), shifted_parents (the learned parents of X, shifted one step forward), blanket_shifted
              (learned_blanket plus shifted_parents), own_lag (X only, the previous value of the series: an autoregressive,
              Granger-style test); "-" for the other methods
  alpha_rule  single (one alpha for every test), equal (ITPD-S, ITPD-S+: alpha_scr = alpha_shr), screen_clean (ITPD-S, ITPD-S+: a
              liberal screening level alpha_scr = 0.1 and a strict elimination level alpha_shr, swept),
              penalty (lasso: a penalty rule, no alpha)
  counting    lazy (a test is issued only when its result can change a label), nonlazy (both tests of every step are issued, as
              in the original code in legacy/; the same graphs), every_issued_test (the method has no lazy rule: every test it issues
              can change the output), no_ci_tests (lasso)
Counting conventions of the numbers themselves (unique vs raw tests) are in itpd/README.md.

Frozen strings: the violation kinds in `observed_data` ("hidden", "contemp", ...) feed the data seed and are not renamed.
"""
from __future__ import annotations

from typing import NamedTuple


class Method(NamedTuple):
    name: str
    function: str
    options: dict
    screening: str
    alpha_rule: str
    counting: str
    meaning: str


def _itpd(name, function, counting, meaning, **options):
    return Method(name, function, options, "-", "single", counting, meaning)


def _shrink(name, screening, alpha_rule, recheck, meaning):
    opts = {"screening": screening}
    if alpha_rule == "screen_clean":
        opts["alpha_scr"] = 0.1
    if recheck:
        opts["recheck"] = True
    return Method(name, "itpd_s", {"shrink": opts, "lazy": True}, screening, alpha_rule, "every_issued_test", meaning)


METHODS = (
    _itpd("itpd_naive", "itpd_naive", "lazy", "PaDL once per target, no cross-target skip rules (paper step 4)",
          variant="paper", lazy=True),
    _itpd("itpd_naive_nonlazy", "itpd_naive", "nonlazy", "itpd_naive with both tests of every step issued (evaluation of the original code)",
          variant="paper", lazy=False),
    _itpd("itpd", "itpd", "lazy", "ITPD: PaDL with the skip rules adjacency, de_z1, de_z5, de_z4 (paper step 4)",
          variant="paper", lazy=True),
    _itpd("itpd_nonlazy", "itpd", "nonlazy", "itpd with both tests of every step issued", variant="paper", lazy=False),
    _itpd("itpd_repo_variant", "itpd", "lazy", "ITPD with step 4 of the original code (extra marginal conjunct)",
          variant="repo", lazy=True),
    _itpd("itpd_repo_variant_nonlazy", "itpd", "nonlazy", "itpd_repo_variant with both tests of every step issued",
          variant="repo", lazy=False),
    _itpd("itpd_adjacency_self", "itpd_adjacency_self", "lazy", "ITPD plus the optional adjacency_self rule",
          variant="paper", lazy=True),
    _itpd("itpd_adjacency_self_nonlazy", "itpd_adjacency_self", "nonlazy", "itpd_adjacency_self with both tests of every step issued",
          variant="paper", lazy=False),
    _itpd("itpd_marginal_first", "itpd_marginal_first", "lazy",
          "ITPD with the Z8 step never skipped and the Y-marginal tested first", variant="paper", lazy=True),
    _itpd("itpd_marginal_first_nonlazy", "itpd_marginal_first", "nonlazy", "itpd_marginal_first with both tests of every step issued",
          variant="paper", lazy=False),
    Method("full_conditioning", "full_conditioning", {"lazy": False}, "-", "single", "every_issued_test",
           "full-conditioning baseline: one test per candidate given all other earlier nodes"),
    Method("iamb_known_order", "iamb_known_order", {"lazy": False}, "-", "single", "every_issued_test",
           "IAMB per target, self edge known; ties in the grow step go to the latest candidate"),
    Method("iamb_known_order_tie_first", "iamb_known_order", {"lazy": False, "tie": "first"}, "-", "single", "every_issued_test",
           "iamb_known_order with ties going to the first candidate (oracle check of the tie rule)"),
    Method("iamb_known_order_tie_random", "iamb_known_order", {"lazy": False, "tie": "random:7"}, "-", "single", "every_issued_test",
           "iamb_known_order with ties broken by a fixed random order, seed 7 (oracle check of the tie rule)"),
    _shrink("itpd_s", "learned_blanket", "equal", False,
         "ITPD-S: screen each candidate given its estimated blanket and X, then shrink the survivors (single pass, no second pass)"),
    _shrink("itpd_s_screen_clean", "learned_blanket", "screen_clean", False,
         "itpd_s with screen-and-clean levels (screening level alpha_scr = 0.1, elimination level alpha_shr)"),
    _shrink("itpd_s_plus", "learned_blanket", "equal", True,
         "ITPD-S+: itpd_s plus the forward-backward second pass (re-check of the screened-out candidates, then a final shrink)"),
    _shrink("itpd_s_plus_screen_clean", "learned_blanket", "screen_clean", True,
         "itpd_s_plus with screen-and-clean levels (alpha_scr = 0.1)"),
    _shrink("itpd_s_true_blanket", "true_blanket", "equal", False,
         "itpd_s with the blanket of the TRUE graph as the screening conditioning set (an oracle: reads the truth)"),
    _shrink("itpd_s_true_blanket_screen_clean", "true_blanket", "screen_clean", False,
         "itpd_s_true_blanket with screen-and-clean levels (alpha_scr = 0.1)"),
    _shrink("itpd_s_own_lag", "own_lag", "equal", False,
         "itpd_s with the previous value of the series alone as the screening conditioning set (autoregressive, Granger-style test)"),
    _shrink("itpd_s_blanket_shifted", "blanket_shifted", "equal", False,
         "itpd_s with the estimated blanket plus the time-shifted parents as the screening conditioning set"),
    _shrink("itpd_s_shifted_parents", "shifted_parents", "equal", False,
         "itpd_s with the learned parents of X shifted one step forward as the screening conditioning set"),
    Method("lasso_cv", "lasso.lasso_s2", {}, "-", "penalty", "no_ci_tests", "known-order lasso, penalty by 5-fold cross-validation"),
    Method("lasso_path", "lasso.lasso_s2", {}, "-", "penalty", "no_ci_tests",
           "known-order lasso at one of 13 fixed penalties (one point of a recall / false-positive curve)"),
    Method("lasso_ebic", "lasso.lasso_s2", {}, "-", "penalty", "no_ci_tests",
           "known-order lasso, penalty by extended BIC (the headline lasso)"),
    Method("lasso_fixed", "lasso.lasso_s2", {}, "-", "penalty", "no_ci_tests",
           "known-order lasso, one fixed penalty sqrt(2 log p / M)"),
    Method("lasso_path_all_lambdas", "lasso.lasso_s2", {}, "-", "penalty", "no_ci_tests",
           "wall-clock of the whole 13-penalty lasso path (timing rows)"),
    Method("itpd_last_slice", "itpd.run_s1", {}, "-", "single", "lazy",
           "single long series: ITPD for each series once, on the last window slice"),
    Method("full_conditioning_last_slice", "full_conditioning.run_s1", {}, "-", "single", "every_issued_test",
           "single long series: full-conditioning baseline on the last window slice"),
    Method("itpd_naive_every_slice", "itpd.run_s2", {}, "-", "single", "lazy",
           "single long series: itpd_naive on every window slice, outputs unioned"),
    Method("itpd_every_slice", "itpd.run_s2", {}, "-", "single", "lazy",
           "single long series: ITPD on every window slice, outputs unioned"),
)

BY_NAME = {m.name: m for m in METHODS}

# Names in result files written before the renames (rows, runner strings, oracle-run rows) -> current names.
OLD_TO_NEW = {
    "itpd_naive_nl": "itpd_naive_nonlazy",
    "itpd_nl": "itpd_nonlazy",
    "itpd_repo": "itpd_repo_variant",
    "itpd_repo_nl": "itpd_repo_variant_nonlazy",
    "itpd_adjself": "itpd_adjacency_self",
    "itpd_adjself_nl": "itpd_adjacency_self_nonlazy",
    "itpd_mf": "itpd_marginal_first",
    "itpd_mf_nl": "itpd_marginal_first_nonlazy",
    "order": "full_conditioning",
    "iamb": "iamb_known_order",
    "iamb_tie_first": "iamb_known_order_tie_first",
    "iamb_tie_rand": "iamb_known_order_tie_random",
    "hpv_mb_eq": "itpd_s",
    "mb": "itpd_s",
    "hpv_mb_len": "itpd_s_screen_clean",
    "hpv_rc_eq": "itpd_s_plus",
    "hpv_safe_eq": "itpd_s_plus",
    "mb_recheck": "itpd_s_plus",
    "hpv_rc_len": "itpd_s_plus_screen_clean",
    "hpv_omb_eq": "itpd_s_true_blanket",
    "oracle_mb": "itpd_s_true_blanket",
    "hpv_omb_len": "itpd_s_true_blanket_screen_clean",
    "none": "itpd_s_own_lag",
    "union": "itpd_s_blanket_shifted",
    "shift": "itpd_s_shifted_parents",
    "lasso_path_13": "lasso_path_all_lambdas",
    "itpd_last": "itpd_last_slice",
    "order_last": "full_conditioning_last_slice",
    "naive_every": "itpd_naive_every_slice",
    "itpd_every": "itpd_every_slice",
    # names of the previous release of this package
    "hpv": "itpd_s",
    "order_based": "full_conditioning",
    "order_based_last_slice": "full_conditioning_last_slice",
    "hpv_single_pass": "itpd_s",
    "hpv_single_pass_lenient": "itpd_s_screen_clean",
    "hpv_safe": "itpd_s_plus",
    "hpv_safe_lenient": "itpd_s_plus_screen_clean",
    "hpv_single_pass_oracle_blanket": "itpd_s_true_blanket",
    "hpv_single_pass_oracle_blanket_lenient": "itpd_s_true_blanket_screen_clean",
    "hpv_single_pass_no_hint": "itpd_s_own_lag",
    "hpv_single_pass_union": "itpd_s_blanket_shifted",
    "hpv_single_pass_shifted_parents": "itpd_s_shifted_parents",
    # names of the release before ITPD-S and ITPD-S+
    "blanket_screened_shrink": "itpd_s",
    "blanket_screened_shrink_lenient": "itpd_s_screen_clean",
    "blanket_screened_shrink_recheck": "itpd_s_plus",
    "blanket_screened_shrink_recheck_lenient": "itpd_s_plus_screen_clean",
    "blanket_screened_shrink_oracle_blanket": "itpd_s_true_blanket",
    "blanket_screened_shrink_oracle_blanket_lenient": "itpd_s_true_blanket_screen_clean",
    "blanket_screened_shrink_x_only": "itpd_s_own_lag",
    "blanket_screened_shrink_union": "itpd_s_blanket_shifted",
    "blanket_screened_shrink_shifted_parents": "itpd_s_shifted_parents",
    # interim names of the variants of ITPD-S (before the standard-term names)
    "itpd_s_lenient": "itpd_s_screen_clean",
    "itpd_s_plus_lenient": "itpd_s_plus_screen_clean",
    "itpd_s_oracle_blanket": "itpd_s_true_blanket",
    "itpd_s_oracle_blanket_lenient": "itpd_s_true_blanket_screen_clean",
    "itpd_s_x_only": "itpd_s_own_lag",
    "itpd_s_union": "itpd_s_blanket_shifted",
}

# Screening values (argument `screening`) written before the renames -> current values.
OLD_SCREENING_TO_NEW = {"mb": "learned_blanket", "oracle_mb": "true_blanket", "shift": "shifted_parents", "none": "own_lag",
                        "union": "blanket_shifted", "oracle_blanket": "true_blanket"}

# Keys of the `shrink` options (and the key of a row of the finite-data results) written before the renames -> current keys:
# the level of the screening step and the level of the shrink step.
OLD_OPTION_KEYS = {"alpha_A": "alpha_scr", "alpha_B": "alpha_shr"}


def current_shrink_options(opts: dict) -> dict:
    """`shrink` options of `itpd_s.run_s2` as written before the renames (keys `alpha_A`, `alpha_B`, earlier screening strings)
    -> the current keys and strings; the one place that reads the old keys. Current options come back unchanged (a copy)."""
    out = {OLD_OPTION_KEYS.get(k, k): v for k, v in opts.items()}
    if "screening" in out:
        out["screening"] = OLD_SCREENING_TO_NEW.get(out["screening"], out["screening"])
    return out


def spec(name: str, alphas, **shrink_extra) -> tuple:
    """(name, runner method, PaDL variant or shrink options, lazy flag, alphas): the tuple `dataset_eval.evaluate_dataset` runs.
    `shrink_extra` adds diagnostic options (keep_parent_tests) to an ITPD-S or ITPD-S+ row."""
    m = BY_NAME[name]
    if "shrink" in m.options:
        return (name, m.function, {**m.options["shrink"], **shrink_extra}, m.options["lazy"], tuple(alphas))
    return (name, m.function, m.options.get("variant"), m.options["lazy"], tuple(alphas))


def shrink_options(name: str) -> dict:
    """Options of `itpd_s.run_s2` for an ITPD-S or ITPD-S+ row (copy)."""
    return dict(BY_NAME[name].options["shrink"])
