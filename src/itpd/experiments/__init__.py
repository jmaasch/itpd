"""Experiments of the paper: configs, the runner and the table of means.

    python -m itpd.experiments run CONFIG --out DIR
    python -m itpd.experiments table CONFIG --runs DIR

run makes the runs of a config file (folder configs/) and writes one result
file per run; table scores the runs and prints the means of the config.
"""
from .aggregate import table
from .runner import run

__all__ = ["run", "table"]
