"""Dispatcher of `python -m itpd.tables <name> [args]`: runs `main(args)` of the module `itpd.tables.<name>`."""
import importlib
import sys

from . import __doc__ as TABLE

NAMES = ("oracle_counts", "finite_data", "nonlinear_data", "single_series", "robustness", "stored_instances")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] in ("-h", "--help"):
        print("usage: python -m itpd.tables {" + ",".join(NAMES) + "} [args]\n")
        print(TABLE)
        return
    if not argv or argv[0] not in NAMES:
        sys.exit("usage: python -m itpd.tables {" + ",".join(NAMES) + "} [args]; --help lists the tables")
    importlib.import_module(f"{__package__}.{argv[0]}").main(argv[1:])


if __name__ == "__main__":
    main()
