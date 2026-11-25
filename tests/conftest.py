"""Shared setup of the tests."""
import os

# Before numpy is imported: simulated data can change in the last bits with the number of BLAS threads.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import pytest  # noqa: E402


@pytest.fixture
def reference_data():
    """Return a function that draws the 2000 observations of an instance and checks their SHA-1.

    Simulated data pass through BLAS, so another CPU type or BLAS build can
    change their last bits. A test of stored numbers is then skipped. With
    ITPD_REQUIRE_REFERENCE_DATA=1 in the environment it fails instead."""
    from itpd.simulation.linear import data_hash

    def draw(instance, sha1):
        X = instance.sample(2000)
        if data_hash(X) != sha1:
            if os.environ.get("ITPD_REQUIRE_REFERENCE_DATA") == "1":
                pytest.fail("the data differ on this platform")
            pytest.skip("the data differ on this platform")
        return X
    return draw
