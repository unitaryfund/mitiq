"""Testing of zero-noise extrapolation methods (factories) with classically generated data."""
import numpy as np
from numpy.testing import assert_array_equal
from mitiq.matrices import npI, npX, npY, npZ


def test_npI() -> AssertionError:
    """Test square of npI is a projector (identity)."""
    assert_array_equal(npI.dot(npI), npI)

def test_npX() -> AssertionError:
    """Test square of npX."""
    assert_array_equal(npX.dot(npX), npI)

def test_npY() -> AssertionError:
    """Test square of npY."""
    assert_array_equal(npY.dot(npY), npI)

def test_npZ() -> AssertionError:
    """Test square of npZ."""
    assert_array_equal(npZ.dot(npZ), npI)

def test_matrices_algebra() -> AssertionError:
    """Test commutation relations for Pauli matrices as in SU(2) algebra."""
    assert_array_equal(npX.dot(npY)-npY.dot(npX), 2*1j*npZ)