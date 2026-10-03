"""Method registry: every name is defined once, old names translate, and the spec tables built from the registry equal the tables
that were written out by hand before the rename (copied here as literals, translated with OLD_TO_NEW / OLD_SCREENING_TO_NEW)."""
import os

import numpy as np

from itpd import dataset_eval, method_runner, run_blanket_screened_shrink, run_known_order_baselines, run_oracle_counts, run_nonlinear, run_robustness, sim
from itpd.methods_registry import BY_NAME, METHODS, OLD_SCREENING_TO_NEW, OLD_TO_NEW

README = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "itpd", "README.md")
ALPHAS = dataset_eval.ALPHAS
P = dataset_eval.PRIMARY
ALPHAS_LEN = tuple(a for a in ALPHAS if a <= 0.1 + 1e-12)


def new(name):
    return OLD_TO_NEW.get(name, name)


def new_variant(v):
    """Shrink options written with the old key `hint` and old strings -> current."""
    if isinstance(v, dict) and "hint" in v:
        return {**{k: x for k, x in v.items() if k != "hint"}, "screening": OLD_SCREENING_TO_NEW.get(v["hint"], v["hint"])}
    return v


def test_names_are_unique_and_old_names_translate_to_registry_names():
    names = [m.name for m in METHODS]
    assert len(names) == len(set(names)) == len(BY_NAME)
    assert set(OLD_TO_NEW.values()) <= set(names)
    assert not (set(OLD_TO_NEW) & set(names) - {"none", "union"}), "an old name must not also be a current name"
    assert {m.counting for m in METHODS} <= {"lazy", "nonlazy", "every_issued_test", "no_ci_tests"}
    assert {m.alpha_rule for m in METHODS} <= {"single", "equal", "lenient", "penalty"}


def test_every_registry_name_is_in_the_readme():
    txt = open(README).read()
    missing = [m.name for m in METHODS if f"`{m.name}`" not in txt]
    assert not missing, missing


def test_shrink_rows_carry_screening_and_alpha_rule_consistently():
    for m in METHODS:
        if m.function != "blanket_screened_shrink":
            continue
        o = m.options["shrink"]
        assert o["screening"] == m.screening
        assert (o.get("alpha_A") == 0.1) == (m.alpha_rule == "lenient")
        assert m.name.startswith("blanket_screened_shrink_recheck") == bool(o.get("recheck"))
        assert m.name.endswith("_lenient") == (m.alpha_rule == "lenient")


def test_spec_tables_equal_the_hand_written_tables_before_the_rename():
    old_finite = (("itpd_naive", "itpd_naive", "paper", True, ALPHAS), ("itpd", "itpd", "paper", True, ALPHAS),
              ("itpd_repo", "itpd", "repo", True, ALPHAS), ("order", "order", None, False, ALPHAS),
              ("itpd_naive_nl", "itpd_naive", "paper", False, (P,)), ("itpd_nl", "itpd", "paper", False, (P,)),
              ("itpd_repo_nl", "itpd", "repo", False, (P,)), ("itpd_adjself", "itpd_adjself", "paper", True, (P,)),
              ("itpd_adjself_nl", "itpd_adjself", "paper", False, (P,)))
    old_known_order = (("iamb", "iamb_known_order", None, False, ALPHAS), ("itpd_mf", "itpd_marginal_first", "paper", True, ALPHAS),
              ("itpd_mf_nl", "itpd_marginal_first", "paper", False, (P,)))
    old_f = (("hpv_mb_eq", "hpv", {"hint": "mb", "keep_parent_tests": True}, True, ALPHAS),
             ("hpv_mb_len", "hpv", {"hint": "mb", "alpha_A": 0.1, "keep_parent_tests": True}, True, ALPHAS_LEN),
             ("hpv_omb_eq", "hpv", {"hint": "oracle_mb", "keep_parent_tests": True}, True, (P,)),
             ("hpv_omb_len", "hpv", {"hint": "oracle_mb", "alpha_A": 0.1, "keep_parent_tests": True}, True, (P,)),
             ("hpv_rc_eq", "hpv", {"hint": "mb", "recheck": True}, True, ALPHAS),
             ("hpv_rc_len", "hpv", {"hint": "mb", "alpha_A": 0.1, "recheck": True}, True, ALPHAS_LEN))
    old_robust = tuple(s[:4] + ((P,),) for s in old_finite if s[0] in ("itpd_naive", "itpd", "order", "itpd_naive_nl", "itpd_nl"))

    def translated(table):
        return tuple((new(n), new(f), new_variant(v), lazy, tuple(al)) for n, f, v, lazy, al in table)

    assert dataset_eval.ITPD_AND_ORDER_SPECS == translated(old_finite)
    assert dataset_eval.IAMB_MARGINAL_FIRST_SPECS == translated(old_known_order)
    assert run_blanket_screened_shrink.SHRINK_FINITE_SPECS == translated(old_f)
    assert dataset_eval.PRIMARY_ALPHA_SPECS == run_robustness.ROBUSTNESS_SPECS == translated(old_robust)
    assert run_nonlinear.specs_for((0.01, 0.05)) == (
        ("itpd_naive", "itpd_naive", "paper", True, (0.01, 0.05)), ("itpd", "itpd", "paper", True, (0.01, 0.05)),
        ("full_conditioning", "full_conditioning", None, False, (0.01, 0.05)))
    # oracle runs
    old_oracle = (("itpd_naive", "itpd_naive", "paper", True), ("itpd", "itpd", "paper", True), ("itpd_repo", "itpd", "repo", True),
                  ("itpd_naive_nl", "itpd_naive", "paper", False), ("itpd_nl", "itpd", "paper", False),
                  ("itpd_repo_nl", "itpd", "repo", False), ("order", "order", None, False))
    assert run_oracle_counts.ORACLE_SPECS == tuple((new(n), new(f), v, lz) for n, f, v, lz in old_oracle)
    assert [s[0] for s in run_oracle_counts.EXTRA_ORACLE_SPECS] == ["itpd_adjacency_self", "itpd_adjacency_self_nonlazy"]
    assert run_known_order_baselines.ORACLE_SPECS == (
        ("iamb_known_order", "iamb_known_order", {"lazy": False}), ("iamb_known_order_tie_first", "iamb_known_order", {"lazy": False, "tie": "first"}),
        ("iamb_known_order_tie_random", "iamb_known_order", {"lazy": False, "tie": "random:7"}),
        ("itpd_marginal_first", "itpd_marginal_first", {"lazy": True}), ("itpd_marginal_first_nonlazy", "itpd_marginal_first", {"lazy": False}))
    assert dict(run_blanket_screened_shrink.SHRINK_ORACLE_VARIANTS) == {
        "blanket_screened_shrink": {"screening": "learned_blanket"}, "blanket_screened_shrink_oracle_blanket": {"screening": "oracle_blanket"},
        "blanket_screened_shrink_x_only": {"screening": "none"}, "blanket_screened_shrink_union": {"screening": "union"},
        "blanket_screened_shrink_recheck": {"screening": "learned_blanket", "recheck": True}, "blanket_screened_shrink_shifted_parents": {"screening": "shifted_parents"}}
    assert [n for n, _ in run_blanket_screened_shrink.SHRINK_ORACLE_VARIANTS] == ["blanket_screened_shrink", "blanket_screened_shrink_oracle_blanket", "blanket_screened_shrink_x_only",
                                                           "blanket_screened_shrink_union", "blanket_screened_shrink_recheck", "blanket_screened_shrink_shifted_parents"]


def test_every_name_used_by_a_driver_is_in_the_registry():
    used = {s[0] for s in dataset_eval.ITPD_AND_ORDER_SPECS + dataset_eval.IAMB_MARGINAL_FIRST_SPECS + run_blanket_screened_shrink.SHRINK_FINITE_SPECS}
    used |= {s[0] for s in run_oracle_counts.ORACLE_SPECS + run_oracle_counts.EXTRA_ORACLE_SPECS}
    used |= {s[0] for s in run_known_order_baselines.ORACLE_SPECS + run_known_order_baselines.TIMING_SPECS}
    used |= {n for n, _ in run_blanket_screened_shrink.SHRINK_ORACLE_VARIANTS}
    assert used <= set(BY_NAME), used - set(BY_NAME)


def _same(a, b):
    a, b = dict(a), dict(b)
    for d in (a, b):
        d.pop("seconds", None)
        d["tests"] = {k: v for k, v in d["tests"].items() if not k.startswith("seconds")}
    return a == b


def test_old_and_new_runner_strings_and_screening_strings_give_identical_outputs():
    s = sim.simulate_s2(5, 5, 400, 2, 1, np.random.default_rng(3), graph="window")
    g = s.graph
    for old, new_ in (("order", "full_conditioning"), ("itpd_adjself", "itpd_adjacency_self")):
        a = method_runner.run_s2_method(old, None, graph=g, order="time")
        b = method_runner.run_s2_method(new_, None, graph=g, order="time")
        assert a["method"] == b["method"] == new_ and _same(a, b)
    for old, new_ in (("mb", "learned_blanket"), ("oracle_mb", "oracle_blanket"), ("shift", "shifted_parents")):
        a = method_runner.run_s2_method("hpv", None, graph=g, shrink={"screening": old}, order="time", per_target=True)
        b = method_runner.run_s2_method("blanket_screened_shrink", None, graph=g, shrink={"screening": new_}, order="time", per_target=True)
        assert a["shrink"] == b["shrink"] == {"screening": new_} and a["method"] == b["method"] and _same(a, b)
