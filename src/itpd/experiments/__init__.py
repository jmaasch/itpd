"""Experiments of the paper: configs, the runner and the table of means.

    python -m itpd.experiments run CONFIG --out DIR
    python -m itpd.experiments table CONFIG --runs DIR

`run` makes the runs of a config file (folder `configs/`) and writes one result
file per run; `table` scores the runs and prints the means of the config.

Where to look:
- `config.py`: reads a config.
- `instances.py`: builds the graph and the data.
- `methods.py`: runs one method.
- `runner.py`: writes one result file per run.
- `aggregate.py`: scores the runs and prints the table.
- `ablations.py`: ITPD with single lines changed.
- `scores.py` and `gene_networks.py`: further scores.
- The functions `run` and `table` are the two commands.
"""
from .aggregate import table
from .runner import run

__all__ = ["run", "table"]
