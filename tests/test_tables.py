"""Collectors (itpd.tables): the command lines, the shared helpers, and the oracle tables on a tiny grid made by the driver."""
import json

import pytest

from itpd.experiments import oracle_counts as oracle_driver
from itpd.methods_registry import OLD_TO_NEW
from itpd.tables import common, oracle_counts
from itpd.tables.__main__ import NAMES, main

COMMANDS = [(n,) for n in NAMES] + [("oracle_counts", m) for m in ("cells", "large", "slopes")] + \
           [("stored_instances", m) for m in ("known_order", "shrink")]


@pytest.mark.parametrize("command", COMMANDS, ids=[" ".join(c) for c in COMMANDS])
def test_every_command_has_help(command, capsys):
    with pytest.raises(SystemExit) as e:
        main([*command, "--help"])
    assert e.value.code == 0 and "usage" in capsys.readouterr().out


def test_the_dispatcher_lists_the_names_and_rejects_others(capsys):
    main(["--help"])
    out = capsys.readouterr().out
    assert all(f"`{n}" in out for n in NAMES)
    with pytest.raises(SystemExit) as e:
        main(["no_such_table"])
    assert e.value.code != 0


def test_old_row_names_are_read_with_the_new_names(tmp_path):
    (old, new), (old2, new2) = list(OLD_TO_NEW.items())[:2]
    cell = tmp_path / "cell"
    cell.mkdir()
    task = {"runs": [{"name": old, "alpha": 0.01}, {"name": "itpd", "alpha": 0.01}], "lasso": {"rows": [{"name": old2}]}}
    (cell / "g00_M50.json").write_text(json.dumps(task))
    (cell / "g00_M50.json.tmp123").write_text("not json")
    (t,) = common.load_tasks(str(cell / "g*_M*.json*"))
    assert [r["name"] for r in t["runs"]] == [new, "itpd"] and t["lasso"]["rows"][0]["name"] == new2
    assert common.run_of(t, new)["alpha"] == 0.01 and common.run_of(t, "itpd", 0.05) is None


def test_number_formats_and_markdown():
    assert common.q([1, 2, 3, None]) == "2.00 [1.50, 2.50]" and common.q([]) == "-"
    assert common.md(["a", 2], [[1, "x"]]) == "| a | 2 |\n|---|---|\n| 1 | x |"
    assert common.size_bins({"0": 3, "1": 2, "3": 5, "250": 1}) == [3, 2, 5, 0, 0, 1]
    assert common.fmt_ci(common.boot_ci([1.0])) == "-" and common.fmt_ci((0.5, 0.25, 0.75)) == "+0.500 [+0.250, +0.750]"
    assert common.ncand(2, 3) == 2 * ((2 * 1 - 1) + (2 * 2 - 1))
    assert common.slope([1, 10, 100], [2, 20, 200]) == pytest.approx(1.0)


def test_recall_at_matched_false_positives():
    curve = {0.1: [100, 80, 20], 0.01: [10, 50, 50], 0.001: [1, 20, 80]}          # alpha: [fp, tp, fn]
    assert common.interp(curve, 10)[:2] == (0.5, "ok")
    r, status, bracket = common.interp(curve, 30)
    assert status == "ok" and 0.5 < r < 0.8 and bracket == (0.01, 0.1)
    assert common.interp(curve, 1000)[1] == "above" and common.interp(curve, 0)[1] == "below"


def test_oracle_tables_from_a_tiny_grid(tmp_path):
    d = str(tmp_path / "oracle")
    oracle_driver.main(["grid", "--out-dir", d, "--N", "3,4", "--T", "3,4", "--tau", "1", "--d", "2", "--graphs", "3", "--workers", "2"])
    oracle_counts.main(["cells", d])
    oracle_counts.main(["slopes", d, "--boot", "20"])
    text = (tmp_path / "oracle" / "TABLES.md").read_text()
    assert "Exact recovery" in text and "12 graphs" in text
    assert json.loads((tmp_path / "oracle" / "aggregates.json").read_text())["slopes"]
    assert (tmp_path / "oracle" / "SLOPES.md").exists() and (tmp_path / "oracle" / "SLOPES.json").exists()
