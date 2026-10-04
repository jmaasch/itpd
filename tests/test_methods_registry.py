"""Method registry: every name is defined once, old names translate, and the spec tables built from the registry equal the tables
that were written out by hand before the renames (copied here as literals, translated with OLD_TO_NEW / OLD_SCREENING_TO_NEW)."""
import os

import numpy as np

from itpd import dataset_eval, method_runner, sim
from itpd.experiments import nonlinear_data, oracle_counts, robustness, stored_instances
from itpd.methods_registry import BY_NAME, METHODS, OLD_OPTION_KEYS, OLD_SCREENING_TO_NEW, OLD_TO_NEW, current_shrink_options
from itpd.tables.common import rename_rows

README = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "itpd", "README.md")
ALPHAS = dataset_eval.ALPHAS
P = dataset_eval.PRIMARY
ALPHAS_LEN = tuple(a for a in ALPHAS if a <= 0.1 + 1e-12)


def new(name):
    return OLD_TO_NEW.get(name, name)


def new_variant(v):
    """Shrink options written with the old key `hint`, the old strings and the old key `alpha_A` -> current."""
    if isinstance(v, dict) and "hint" in v:
        out = {**{k: x for k, x in v.items() if k != "hint"}, "screening": OLD_SCREENING_TO_NEW.get(v["hint"], v["hint"])}
        return {("alpha_scr" if k == "alpha_A" else k): x for k, x in out.items()}
    return v


def test_names_are_unique_and_old_names_translate_to_registry_names():
    names = [m.name for m in METHODS]
    assert len(names) == len(set(names)) == len(BY_NAME)
    assert set(OLD_TO_NEW.values()) <= set(names)
    assert not (set(OLD_TO_NEW) & set(names) - {"none", "union"}), "an old name must not also be a current name"
    assert {m.counting for m in METHODS} <= {"lazy", "nonlazy", "every_issued_test", "no_ci_tests"}
    assert {m.alpha_rule for m in METHODS} <= {"single", "equal", "screen_clean", "penalty"}


def test_every_registry_name_is_in_the_readme():
    txt = open(README).read()
    missing = [m.name for m in METHODS if f"`{m.name}`" not in txt]
    assert not missing, missing


def test_every_itpd_s_name_is_in_the_readme():
    txt = open(README).read()
    names = [m.name for m in METHODS if m.name.startswith("itpd_s")]
    assert len(names) == 9, names
    assert [n for n in names if f"`{n}`" not in txt] == []


def test_every_generation_of_shrink_names_translates_to_the_itpd_s_names():
    first = {"hpv_mb_eq": "itpd_s", "hpv_mb_len": "itpd_s_screen_clean", "hpv_rc_eq": "itpd_s_plus", "hpv_safe_eq": "itpd_s_plus",
             "hpv_rc_len": "itpd_s_plus_screen_clean", "hpv_omb_eq": "itpd_s_true_blanket", "hpv_omb_len": "itpd_s_true_blanket_screen_clean"}
    second = {"hpv": "itpd_s", "hpv_single_pass": "itpd_s", "hpv_single_pass_lenient": "itpd_s_screen_clean", "hpv_safe": "itpd_s_plus",
              "hpv_safe_lenient": "itpd_s_plus_screen_clean", "hpv_single_pass_oracle_blanket": "itpd_s_true_blanket",
              "hpv_single_pass_oracle_blanket_lenient": "itpd_s_true_blanket_screen_clean", "hpv_single_pass_no_hint": "itpd_s_own_lag",
              "hpv_single_pass_union": "itpd_s_blanket_shifted", "hpv_single_pass_shifted_parents": "itpd_s_shifted_parents"}
    third = {"blanket_screened_shrink": "itpd_s", "blanket_screened_shrink_lenient": "itpd_s_screen_clean",
             "blanket_screened_shrink_recheck": "itpd_s_plus", "blanket_screened_shrink_recheck_lenient": "itpd_s_plus_screen_clean",
             "blanket_screened_shrink_oracle_blanket": "itpd_s_true_blanket",
             "blanket_screened_shrink_oracle_blanket_lenient": "itpd_s_true_blanket_screen_clean",
             "blanket_screened_shrink_x_only": "itpd_s_own_lag", "blanket_screened_shrink_union": "itpd_s_blanket_shifted",
             "blanket_screened_shrink_shifted_parents": "itpd_s_shifted_parents"}
    fourth = {"itpd_s_lenient": "itpd_s_screen_clean", "itpd_s_plus_lenient": "itpd_s_plus_screen_clean",
              "itpd_s_oracle_blanket": "itpd_s_true_blanket", "itpd_s_oracle_blanket_lenient": "itpd_s_true_blanket_screen_clean",
              "itpd_s_x_only": "itpd_s_own_lag", "itpd_s_union": "itpd_s_blanket_shifted"}
    for table in (first, second, third, fourth):
        for old, new_ in table.items():
            assert OLD_TO_NEW[old] == new_ and new_ in BY_NAME, (old, new_)
    assert set(third.values()) == {m.name for m in METHODS if m.name.startswith("itpd_s")}
    assert set(fourth.values()) <= set(third.values())
    assert not any(v.startswith(("hpv", "blanket_screened_shrink")) or v.endswith(("lenient", "x_only", "union", "oracle_blanket"))
                   for v in OLD_TO_NEW.values())


def test_screening_strings_and_option_keys_of_every_generation_translate():
    assert OLD_SCREENING_TO_NEW == {"mb": "learned_blanket", "oracle_mb": "true_blanket", "shift": "shifted_parents", "none": "own_lag",
                                    "union": "blanket_shifted", "oracle_blanket": "true_blanket"}
    assert OLD_OPTION_KEYS == {"alpha_A": "alpha_scr", "alpha_B": "alpha_shr"}
    assert current_shrink_options({"screening": "oracle_blanket", "alpha_A": 0.1, "recheck": True, "cap": "auto"}) == \
        {"screening": "true_blanket", "alpha_scr": 0.1, "recheck": True, "cap": "auto"}
    cur = {"screening": "own_lag", "alpha_scr": 0.1}
    assert current_shrink_options(cur) == cur and current_shrink_options(cur) is not cur
    # a stored finite-data task of an earlier release: names, `shrink` options and the level key of a row are read with the current ones
    task = {"runs": [{"name": "itpd_s_oracle_blanket_lenient", "alpha": 0.01, "alpha_A": 0.1,
                      "shrink": {"screening": "oracle_blanket", "alpha_A": 0.1}},
                     {"name": "itpd_s", "alpha": 0.05, "alpha_A": 0.05, "shrink": {"screening": "learned_blanket"}}]}
    assert rename_rows(task)["runs"] == [
        {"name": "itpd_s_true_blanket_screen_clean", "alpha": 0.01, "alpha_scr": 0.1, "shrink": {"screening": "true_blanket", "alpha_scr": 0.1}},
        {"name": "itpd_s", "alpha": 0.05, "alpha_scr": 0.05, "shrink": {"screening": "learned_blanket"}}]


def test_shrink_rows_carry_screening_and_alpha_rule_consistently():
    for m in METHODS:
        if m.function != "itpd_s":
            continue
        o = m.options["shrink"]
        assert o["screening"] == m.screening
        assert (o.get("alpha_scr") == 0.1) == (m.alpha_rule == "screen_clean")
        assert m.name.startswith("itpd_s_plus") == bool(o.get("recheck"))
        assert m.name.endswith("_screen_clean") == (m.alpha_rule == "screen_clean")


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
    assert stored_instances.SHRINK_FINITE_SPECS == translated(old_f)
    assert dataset_eval.PRIMARY_ALPHA_SPECS == robustness.ROBUSTNESS_SPECS == translated(old_robust)
    assert nonlinear_data.specs_for((0.01, 0.05)) == (
        ("itpd_naive", "itpd_naive", "paper", True, (0.01, 0.05)), ("itpd", "itpd", "paper", True, (0.01, 0.05)),
        ("full_conditioning", "full_conditioning", None, False, (0.01, 0.05)))
    # oracle runs
    old_oracle = (("itpd_naive", "itpd_naive", "paper", True), ("itpd", "itpd", "paper", True), ("itpd_repo", "itpd", "repo", True),
                  ("itpd_naive_nl", "itpd_naive", "paper", False), ("itpd_nl", "itpd", "paper", False),
                  ("itpd_repo_nl", "itpd", "repo", False), ("order", "order", None, False))
    assert oracle_counts.ORACLE_SPECS == tuple((new(n), new(f), v, lz) for n, f, v, lz in old_oracle)
    assert [s[0] for s in oracle_counts.EXTRA_ORACLE_SPECS] == ["itpd_adjacency_self", "itpd_adjacency_self_nonlazy"]
    assert stored_instances.KNOWN_ORDER_ORACLE_SPECS == (
        ("iamb_known_order", "iamb_known_order", {"lazy": False}), ("iamb_known_order_tie_first", "iamb_known_order", {"lazy": False, "tie": "first"}),
        ("iamb_known_order_tie_random", "iamb_known_order", {"lazy": False, "tie": "random:7"}),
        ("itpd_marginal_first", "itpd_marginal_first", {"lazy": True}), ("itpd_marginal_first_nonlazy", "itpd_marginal_first", {"lazy": False}))
    assert dict(stored_instances.SHRINK_ORACLE_VARIANTS) == {
        "itpd_s": {"screening": "learned_blanket"}, "itpd_s_true_blanket": {"screening": "true_blanket"},
        "itpd_s_own_lag": {"screening": "own_lag"}, "itpd_s_blanket_shifted": {"screening": "blanket_shifted"},
        "itpd_s_plus": {"screening": "learned_blanket", "recheck": True}, "itpd_s_shifted_parents": {"screening": "shifted_parents"}}
    assert [n for n, _ in stored_instances.SHRINK_ORACLE_VARIANTS] == ["itpd_s", "itpd_s_true_blanket", "itpd_s_own_lag",
                                                           "itpd_s_blanket_shifted", "itpd_s_plus", "itpd_s_shifted_parents"]


def test_every_name_used_by_a_driver_is_in_the_registry():
    used = {s[0] for s in dataset_eval.ITPD_AND_ORDER_SPECS + dataset_eval.IAMB_MARGINAL_FIRST_SPECS + stored_instances.SHRINK_FINITE_SPECS}
    used |= {s[0] for s in oracle_counts.ORACLE_SPECS + oracle_counts.EXTRA_ORACLE_SPECS}
    used |= {s[0] for s in stored_instances.KNOWN_ORDER_ORACLE_SPECS + stored_instances.TIMING_SPECS}
    used |= {n for n, _ in stored_instances.SHRINK_ORACLE_VARIANTS}
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
    for old, new_ in (("mb", "learned_blanket"), ("oracle_mb", "true_blanket"), ("shift", "shifted_parents"), ("none", "own_lag"),
                      ("union", "blanket_shifted"), ("oracle_blanket", "true_blanket")):
        a = method_runner.run_s2_method("hpv", None, graph=g, shrink={"screening": old}, order="time", per_target=True)
        b = method_runner.run_s2_method("blanket_screened_shrink", None, graph=g, shrink={"screening": new_}, order="time", per_target=True)
        c = method_runner.run_s2_method("itpd_s", None, graph=g, shrink={"screening": new_}, order="time", per_target=True)
        assert a["shrink"] == b["shrink"] == c["shrink"] == {"screening": new_} and a["method"] == b["method"] == c["method"] == "itpd_s"
        assert _same(a, c) and _same(b, c)
    # the keys of the options: the old `alpha_A` and the current `alpha_scr` give the same run and the same stored options
    a = method_runner.run_s2_method("itpd_s", None, graph=g, shrink={"screening": "none", "alpha_A": 0.2}, order="time", per_target=True)
    b = method_runner.run_s2_method("itpd_s", None, graph=g, shrink={"screening": "own_lag", "alpha_scr": 0.2}, order="time", per_target=True)
    assert a["shrink"] == b["shrink"] == {"screening": "own_lag", "alpha_scr": 0.2} and _same(a, b)
    # the finite-data harness reads the same old options: the row carries the current level key and options
    old = dataset_eval.run_dataset(g.A, s.X, 1, (("itpd_s_screen_clean", "itpd_s", {"screening": "none", "alpha_A": 0.1}, True, (0.01,)),), order="time")
    cur = dataset_eval.run_dataset(g.A, s.X, 1, (("itpd_s_screen_clean", "itpd_s", {"screening": "own_lag", "alpha_scr": 0.1}, True, (0.01,)),), order="time")
    ro, rc = old["runs"][0], cur["runs"][0]
    assert ro["alpha_scr"] == rc["alpha_scr"] == 0.1 and ro["shrink"] == rc["shrink"] == {"screening": "own_lag", "alpha_scr": 0.1}
    assert ro["metrics"] == rc["metrics"] and ro["unique"] == rc["unique"]
