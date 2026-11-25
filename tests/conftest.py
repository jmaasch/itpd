"""Shared setup of the tests."""
import os

# Before numpy is imported: simulated data can change in the last bits with the number of BLAS threads.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import pytest  # noqa: E402


@pytest.fixture
def no_interpreters(monkeypatch):
    """Unset the variables that name the interpreters of the external baselines and the DF-PC clone."""
    for name in ("ITPD_PYTHON_PCMCI", "ITPD_PYTHON_DFPC", "ITPD_PYTHON_DYNOTEARS", "ITPD_PYTHON_CDNOD", "ITPD_DFPC_SOURCE"):
        monkeypatch.delenv(name, raising=False)
