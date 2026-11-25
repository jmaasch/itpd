"""Check that the example scripts run."""
import runpy
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def run_script(name, arguments, monkeypatch, capsys) -> list:
    """Run an example as a script and return the lines that it prints."""
    path = str(EXAMPLES / name)
    monkeypatch.setattr(sys, "argv", [path, *arguments])
    runpy.run_path(path, run_name="__main__")
    return capsys.readouterr().out.splitlines()


def test_oracle_exact_recovery(tmp_path, monkeypatch, capsys):
    lines = run_script("oracle_exact_recovery.py", [], monkeypatch, capsys)
    assert lines == ["Exact recovery: 15 nodes, 18 edges (seed=7).",
                     "CI tests: Phase I=115, Phase II=8; largest conditioning set=5."]
    figure = tmp_path / "graph.png"
    lines = run_script("oracle_exact_recovery.py", ["--seed", "3", "--output", str(figure)], monkeypatch, capsys)
    assert lines[0].startswith("Exact recovery: 15 nodes") and lines[-1] == f"Learned graph saved to {figure}"
    assert figure.stat().st_size > 0


@pytest.mark.parametrize("arguments, M, graph", [([], 500, 0), (["--graph", "1", "--M", "200"], 200, 1)])
def test_fit_simulated_data(arguments, M, graph, monkeypatch, capsys):
    lines = run_script("fit_simulated_data.py", arguments, monkeypatch, capsys)
    assert lines[0] == f"Data: {M} observations of 5 series over 8 time steps (graph {graph})."
    assert lines[1].startswith("Edges (lag-1 self edges not scored): ") and lines[2].startswith("CI tests: ")
    assert float(lines[1].rsplit("F1 = ", 1)[1].rstrip(".")) > 0.9
