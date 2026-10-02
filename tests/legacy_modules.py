"""Load the original modules in legacy/ as legacy_itpd, legacy_itpd_naive, legacy_padl_itpd and legacy_padl_naive.

The package is called itpd, like the module legacy/itpd.py, so legacy/ is not put on sys.path. The original modules import
each other by their plain names (`from padl_itpd import PaDL`): the plain names of the two PaDL modules are registered in
sys.modules before the modules that import them run. The original files are not edited.
"""
import importlib.util
import os
import sys

LEGACY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "legacy")


def _load(alias, filename, plain_name=None):
    """Execute legacy/<filename> as module `alias`; `plain_name` is the name other legacy modules import it by."""
    if alias in sys.modules:
        return sys.modules[alias]
    spec = importlib.util.spec_from_file_location(alias, os.path.join(LEGACY, filename))
    module = importlib.util.module_from_spec(spec)
    names = [alias] + ([plain_name] if plain_name else [])
    for name in names:
        sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        for name in names:
            sys.modules.pop(name, None)
        raise
    return module


legacy_padl_itpd = _load("legacy_padl_itpd", "padl_itpd.py", plain_name="padl_itpd")
legacy_padl_naive = _load("legacy_padl_naive", "padl_naive.py", plain_name="padl_naive")
legacy_itpd = _load("legacy_itpd", "itpd.py")
legacy_itpd_naive = _load("legacy_itpd_naive", "itpd_naive.py")
