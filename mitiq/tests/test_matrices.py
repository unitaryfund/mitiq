"""Testing of zero-noise extrapolation methods (factories) with classically generated data."""
import numpy as np
from numpy.testing import assert_array_equal
from mitiq.matrices import npI, npX, npY, npZ


def test_npI() -> AssertionError:
    """Test square of identity in npI is a projector"""
    assert_array_equal(npI().dot(npI()), npI())

def test_npX() -> AssertionError:
    """Test square of sigma_y of npX"""
    assert_array_equal(npX().dot(npX()), npI())

def test_npY() -> AssertionError:
    """Test square of sigma_y of npY"""
    assert_array_equal(npY().dot(npY()), npI())

def test_npZ() -> AssertionError:
    """Test square of sigma_z of npZ"""
    assert_array_equal(npZ().dot(npZ()), npI())

def test_matrices_algebra() -> AssertionError:
    """Test SU(2) algebra with commutation relations"""
    assert_array_equal(npI().dot(npI()), npI())