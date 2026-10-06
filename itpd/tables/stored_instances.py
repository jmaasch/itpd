"""Tables of the runs on stored instances: the known-order baselines and ITPD-S and ITPD-S+.

    python -m itpd.tables stored_instances known_order [--out DIR] [--cells ...] [--M ...] [--shrink-graphs 20]
    python -m itpd.tables stored_instances itpd_s [--results DIR] [--out DIR] [--only oracle,finite,curves] [--arm ...] [--N ...] [--T ...]
        [--tau ...] [--M ...] [--graphs 20]

`known_order` is in known_order.py and `itpd_s` in itpd_s.py; `known_order --help` and `itpd_s --help` show the arguments and the files.
"""
import argparse

from . import itpd_s, known_order


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m itpd.tables stored_instances", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = ap.add_subparsers(dest="mode", required=True, metavar="{known_order,itpd_s}")
    for name, mod, text in (("known_order", known_order, "tables of the known-order baselines"),
                            ("itpd_s", itpd_s, "tables of the ITPD-S and ITPD-S+ runs")):
        mod.add_args(modes.add_parser(name, help=text, description=mod.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    a = ap.parse_args(argv)
    {"known_order": known_order.run, "itpd_s": itpd_s.run}[a.mode](a)


if __name__ == "__main__":
    main()
