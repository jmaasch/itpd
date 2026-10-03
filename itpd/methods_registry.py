"""The one table of method names.

Every method name that appears in a result file (`name` of a row, `method` of a runner output) is defined here, with the
function that runs it, its fixed options, its screening set, its alpha rule, its counting mode and a one-line meaning.
Drivers build their (name, alphas) lists from this table (`spec`); readers of old result files translate with `OLD_TO_NEW`.

Row identity is the name: the lazy and the non-lazy counting of the same method are two rows, `itpd` and `itpd_nonlazy`.

Columns
  name        the label written to result files
  function    `method` string of `method_runner.run_s2_method` (or the lasso function, or the S1 / last-slice runner)
  options     fixed options of that call: variant (PaDL step 4: "paper" | "repo"), lazy (bool recorded in the row),
              shrink (options of `blanket_screened_shrink.run_s2`), tie (IAMB tie rule)
  screening   blanket-screened shrink only, the set a candidate is screened given, besides X: learned_blanket (Markov blanket in
              the graph learned so far), oracle_blanket (blanket in the TRUE graph, an upper bound that reads the truth),
              shifted_parents (the learned parents of X, shifted one step forward), union (learned_blanket plus
              shifted_parents), none (X only); "-" for the other methods
  alpha_rule  single (one alpha for every test), equal (blanket-screened shrink: alpha_A = alpha_B), lenient (blanket-screened shrink: alpha_A = 0.1, alpha_B swept),
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
    if alpha_rule == "lenient":
        opts["alpha_A"] = 0.1
    if recheck:
        opts["recheck"] = True
    return Method(name, "blanket_screened_shrink", {"shrink": opts, "lazy": True}, screening, alpha_rule, "every_issued_test", meaning)


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
    _shrink("blanket_screened_shrink", "learned_blanket", "equal", False,
         "blanket-screened shrink: screen each candidate given its learned blanket and X, then shrink the survivors (no re-check)"),
    _shrink("blanket_screened_shrink_lenient", "learned_blanket", "lenient", False, "blanket_screened_shrink with a lenient screen (alpha_A = 0.1)"),
    _shrink("blanket_screened_shrink_recheck", "learned_blanket", "equal", True,
         "blanket_screened_shrink plus the re-check of the pruned candidates and a final verification (always-verify)"),
    _shrink("blanket_screened_shrink_recheck_lenient", "learned_blanket", "lenient", True, "blanket_screened_shrink_recheck with a lenient screen (alpha_A = 0.1)"),
    _shrink("blanket_screened_shrink_oracle_blanket", "oracle_blanket", "equal", False,
         "blanket_screened_shrink with the blanket of the TRUE graph as the screening set (reads the truth; upper bound)"),
    _shrink("blanket_screened_shrink_oracle_blanket_lenient", "oracle_blanket", "lenient", False,
         "blanket_screened_shrink_oracle_blanket with a lenient screen (alpha_A = 0.1)"),
    _shrink("blanket_screened_shrink_x_only", "none", "equal", False, "blanket_screened_shrink with the screening set X only"),
    _shrink("blanket_screened_shrink_union", "union", "equal", False, "blanket_screened_shrink with the learned blanket plus the shifted parents as screening set"),
    _shrink("blanket_screened_shrink_shifted_parents", "shifted_parents", "equal", False,
         "blanket_screened_shrink with the learned parents of X shifted one step forward as screening set"),
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
    "hpv_mb_eq": "blanket_screened_shrink",
    "mb": "blanket_screened_shrink",
    "hpv_mb_len": "blanket_screened_shrink_lenient",
    "hpv_rc_eq": "blanket_screened_shrink_recheck",
    "hpv_safe_eq": "blanket_screened_shrink_recheck",
    "mb_recheck": "blanket_screened_shrink_recheck",
    "hpv_rc_len": "blanket_screened_shrink_recheck_lenient",
    "hpv_omb_eq": "blanket_screened_shrink_oracle_blanket",
    "oracle_mb": "blanket_screened_shrink_oracle_blanket",
    "hpv_omb_len": "blanket_screened_shrink_oracle_blanket_lenient",
    "none": "blanket_screened_shrink_x_only",
    "union": "blanket_screened_shrink_union",
    "shift": "blanket_screened_shrink_shifted_parents",
    "lasso_path_13": "lasso_path_all_lambdas",
    "itpd_last": "itpd_last_slice",
    "order_last": "full_conditioning_last_slice",
    "naive_every": "itpd_naive_every_slice",
    "itpd_every": "itpd_every_slice",
    # names of the previous release of this package
    "hpv": "blanket_screened_shrink",
    "order_based": "full_conditioning",
    "order_based_last_slice": "full_conditioning_last_slice",
    "hpv_single_pass": "blanket_screened_shrink",
    "hpv_single_pass_lenient": "blanket_screened_shrink_lenient",
    "hpv_safe": "blanket_screened_shrink_recheck",
    "hpv_safe_lenient": "blanket_screened_shrink_recheck_lenient",
    "hpv_single_pass_oracle_blanket": "blanket_screened_shrink_oracle_blanket",
    "hpv_single_pass_oracle_blanket_lenient": "blanket_screened_shrink_oracle_blanket_lenient",
    "hpv_single_pass_no_hint": "blanket_screened_shrink_x_only",
    "hpv_single_pass_union": "blanket_screened_shrink_union",
    "hpv_single_pass_shifted_parents": "blanket_screened_shrink_shifted_parents",
}

# Screening values (argument `screening`) written before the rename -> current values.
OLD_SCREENING_TO_NEW = {"mb": "learned_blanket", "oracle_mb": "oracle_blanket", "shift": "shifted_parents"}


def spec(name: str, alphas, **shrink_extra) -> tuple:
    """(name, runner method, PaDL variant or shrink options, lazy flag, alphas): the tuple `dataset_eval.evaluate_dataset` runs.
    `shrink_extra` adds diagnostic options (keep_parent_tests) to a blanket-screened shrink row."""
    m = BY_NAME[name]
    if "shrink" in m.options:
        return (name, m.function, {**m.options["shrink"], **shrink_extra}, m.options["lazy"], tuple(alphas))
    return (name, m.function, m.options.get("variant"), m.options["lazy"], tuple(alphas))


def shrink_options(name: str) -> dict:
    """Options of `blanket_screened_shrink.run_s2` for a blanket-screened shrink row (copy)."""
    return dict(BY_NAME[name].options["shrink"])
