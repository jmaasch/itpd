"""The modules in legacy/ import on Python < 3.14 (their annotations used to fail at definition time), and loading them
leaves the package itpd alone."""
import os
import sys

import legacy_modules


def test_legacy_modules_import():
    assert hasattr(legacy_modules.legacy_itpd, "ITPD")
    assert hasattr(legacy_modules.legacy_itpd_naive, "ITPDNaive")
    assert hasattr(legacy_modules.legacy_padl_itpd, "PaDL")
    assert hasattr(legacy_modules.legacy_padl_naive, "PaDL")
    # the package itpd is not shadowed by legacy/itpd.py
    import itpd
    assert os.path.realpath(legacy_modules.LEGACY) not in [os.path.realpath(p) for p in sys.path if p]
    assert os.path.basename(os.path.dirname(itpd.__file__)) == "itpd"
    assert os.path.basename(itpd.__file__) == "__init__.py"
