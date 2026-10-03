"""The one table of method names.

Every method name that appears in a result file (`name` of a row, `method` of a runner output) is defined here, with the
function that runs it, its fixed options, its hint type, its alpha rule, its counting mode and a one-line meaning.
Drivers build their (name, alphas) lists from this table (`spec`); readers of old result files translate with `OLD_TO_NEW`.

Row identity is the name: the lazy and the non-lazy counting of the same method are two rows, `itpd` and `itpd_nonlazy`.

Columns
  name        the label written to result files
  function    `method` string of `method_runner.run_s2_method` (or the lasso function, or the S1 / last-slice runner)
  options     fixed options of that call: variant (PaDL step 4: "paper" | "repo"), lazy (bool recorded in the row),
              hpv (options of `hpv.run_s2`), tie (IAMB tie rule)
  hint        HPV only, the set a candidate is first screened against: learned_blanket (Markov blanket in the graph learned so
              far), oracle_blanket (blanket in the TRUE graph, an upper bound that reads the truth), shifted_parents (the
              learned parents of X, shifted one step forward), union (learned_blanket plus shifted_parents), none (empty hint);
              "-" for methods without hints
  alpha_rule  single (one alpha for every test), equal (HPV: alpha_A = alpha_B), lenient (HPV: alpha_A = 0.1, alpha_B swept),
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
    hint: str
    alpha_rule: str
    counting: str
    meaning: str


def _itpd(name, function, counting, meaning, **options):
    return Method(name, function, options, "-", "single", counting, meaning)


def _hpv(name, hint, alpha_rule, recheck, meaning):
    opts = {"hint": hint}
    if alpha_rule == "lenient":
        opts["alpha_A"] = 0.1
    if recheck:
        opts["recheck"] = True
    return Method(name, "hpv", {"hpv": opts, "lazy": True}, hint, alpha_rule, "every_issued_test", meaning)


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
    Method("order_based", "order_based", {"lazy": False}, "-", "single", "every_issued_test",
           "order-based baseline: one test per candidate given all earlier nodes"),
    Method("iamb_known_order", "iamb_known_order", {"lazy": False}, "-", "single", "every_issued_test",
           "IAMB per target, self edge known; ties in the grow step go to the latest candidate"),
    Method("iamb_known_order_tie_first", "iamb_known_order", {"lazy": False, "tie": "first"}, "-", "single", "every_issued_test",
           "iamb_known_order with ties going to the first candidate (oracle check of the tie rule)"),
    Method("iamb_known_order_tie_random", "iamb_known_order", {"lazy": False, "tie": "random:7"}, "-", "single", "every_issued_test",
           "iamb_known_order with ties broken by a fixed random order, seed 7 (oracle check of the tie rule)"),
    _hpv("hpv_single_pass", "learned_blanket", "equal", False,
         "HPV: screen each candidate given its learned blanket, then verify the survivors (no re-check)"),
    _hpv("hpv_single_pass_lenient", "learned_blanket", "lenient", False, "hpv_single_pass with a lenient screen (alpha_A = 0.1)"),
    _hpv("hpv_safe", "learned_blanket", "equal", True,
         "HPV-safe: hpv_single_pass plus the re-check of the pruned candidates and a final verification (always-verify)"),
    _hpv("hpv_safe_lenient", "learned_blanket", "lenient", True, "hpv_safe with a lenient screen (alpha_A = 0.1)"),
    _hpv("hpv_single_pass_oracle_blanket", "oracle_blanket", "equal", False,
         "hpv_single_pass with the blanket of the TRUE graph as hint (reads the truth; upper bound)"),
    _hpv("hpv_single_pass_oracle_blanket_lenient", "oracle_blanket", "lenient", False,
         "hpv_single_pass_oracle_blanket with a lenient screen (alpha_A = 0.1)"),
    _hpv("hpv_single_pass_no_hint", "none", "equal", False, "hpv_single_pass with the empty hint: screen given X only"),
    _hpv("hpv_single_pass_union", "union", "equal", False, "hpv_single_pass with learned blanket plus shifted parents as hint"),
    _hpv("hpv_single_pass_shifted_parents", "shifted_parents", "equal", False,
         "hpv_single_pass with the learned parents of X shifted one step forward as hint"),
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
    Method("order_based_last_slice", "order_based.run_s1", {}, "-", "single", "every_issued_test",
           "single long series: order-based baseline on the last window slice"),
    Method("itpd_naive_every_slice", "itpd.run_s2", {}, "-", "single", "lazy",
           "single long series: itpd_naive on every window slice, outputs unioned"),
    Method("itpd_every_slice", "itpd.run_s2", {}, "-", "single", "lazy",
           "single long series: ITPD on every window slice, outputs unioned"),
)

BY_NAME = {m.name: m for m in METHODS}

# Names in result files written before the rename (rows, runner strings, HPV oracle-run rows) -> current names.
OLD_TO_NEW = {
    "itpd_naive_nl": "itpd_naive_nonlazy",
    "itpd_nl": "itpd_nonlazy",
    "itpd_repo": "itpd_repo_variant",
    "itpd_repo_nl": "itpd_repo_variant_nonlazy",
    "itpd_adjself": "itpd_adjacency_self",
    "itpd_adjself_nl": "itpd_adjacency_self_nonlazy",
    "itpd_mf": "itpd_marginal_first",
    "itpd_mf_nl": "itpd_marginal_first_nonlazy",
    "order": "order_based",
    "iamb": "iamb_known_order",
    "iamb_tie_first": "iamb_known_order_tie_first",
    "iamb_tie_rand": "iamb_known_order_tie_random",
    "hpv_mb_eq": "hpv_single_pass",
    "mb": "hpv_single_pass",
    "hpv_mb_len": "hpv_single_pass_lenient",
    "hpv_rc_eq": "hpv_safe",
    "hpv_safe_eq": "hpv_safe",
    "mb_recheck": "hpv_safe",
    "hpv_rc_len": "hpv_safe_lenient",
    "hpv_omb_eq": "hpv_single_pass_oracle_blanket",
    "oracle_mb": "hpv_single_pass_oracle_blanket",
    "hpv_omb_len": "hpv_single_pass_oracle_blanket_lenient",
    "none": "hpv_single_pass_no_hint",
    "union": "hpv_single_pass_union",
    "shift": "hpv_single_pass_shifted_parents",
    "lasso_path_13": "lasso_path_all_lambdas",
    "itpd_last": "itpd_last_slice",
    "order_last": "order_based_last_slice",
    "naive_every": "itpd_naive_every_slice",
    "itpd_every": "itpd_every_slice",
}

# HPV hint values (argument `hint` and stored `hpv.hint`) before the rename -> current values.
OLD_HINT_TO_NEW = {"mb": "learned_blanket", "oracle_mb": "oracle_blanket", "shift": "shifted_parents"}


def spec(name: str, alphas, **hpv_extra) -> tuple:
    """(name, runner method, PaDL variant or HPV options, lazy flag, alphas): the tuple `dataset_eval.evaluate_dataset` runs.
    `hpv_extra` adds diagnostic options (keep_parent_tests) to an HPV row."""
    m = BY_NAME[name]
    if "hpv" in m.options:
        return (name, m.function, {**m.options["hpv"], **hpv_extra}, m.options["lazy"], tuple(alphas))
    return (name, m.function, m.options.get("variant"), m.options["lazy"], tuple(alphas))


def hpv_options(name: str) -> dict:
    """Options of `hpv.run_s2` for an HPV row (copy)."""
    return dict(BY_NAME[name].options["hpv"])
