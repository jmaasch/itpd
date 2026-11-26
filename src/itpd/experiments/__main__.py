"""Command line of the experiments.

    python -m itpd.experiments run CONFIG --out DIR [--graphs 0-2] [--methods ITPD,IAMB] [--M 200,500]
    python -m itpd.experiments table CONFIG --runs DIR [--compare]
"""
import argparse
import sys

from .aggregate import table
from .config import ConfigError, indices
from .runner import run


def main(argv=None) -> None:
    """Parse `argv` (or process arguments) and call `run` or `table`.

    `table` prints its text. On `ConfigError`, print "error: ..." and exit with
    code 1.
    """
    parser = argparse.ArgumentParser(prog="python -m itpd.experiments", description="Run the experiments of a config file.")
    commands = parser.add_subparsers(dest="command", required=True)
    runs = commands.add_parser("run", help="run the methods of a config; one result file per run")
    runs.add_argument("config", help="config file (YAML)")
    runs.add_argument("--out", required=True, help="output folder; finished runs in it are skipped")
    runs.add_argument("--graphs", help='graph indices to run, such as "0-2" or "0,3,5-7" (default: all of the config)')
    runs.add_argument("--methods", help='methods to run, such as "ITPD,IAMB" (default: all of the config)')
    runs.add_argument("--M", help='numbers of observations to run, such as "200,500" (default: all of the config)')
    tables = commands.add_parser("table", help="score the runs of a config and print the table of means")
    tables.add_argument("config", help="config file (YAML)")
    tables.add_argument("--runs", required=True, help="output folder of the runs")
    tables.add_argument("--compare", action="store_true", help="add the difference to the means in configs/expected")
    args = parser.parse_args(argv)
    try:
        if args.command == "table":
            print(table(args.config, args.runs, compare=args.compare))
        else:
            run(args.config, args.out, graphs=args.graphs and indices(args.graphs),
                methods=args.methods and [name.strip() for name in args.methods.split(",")],
                M=args.M and indices(args.M))
    except ConfigError as error:
        sys.exit(f"error: {error}")


if __name__ == "__main__":
    main()
