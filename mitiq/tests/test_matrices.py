"""Testing of zero-noise extrapolation methods (factories) with classically generated data."""
import numpy as np
from numpy.testing import assert_array_equal
from mitiq.matrices import npI, npX, npY, npZ


def test_npX() -> None:
    """Test square of identity in npI is a projector"""
    assert_array_equal(npI**2, npI)

def test_npX() -> None:
    """Test square of sigma_y of npX"""
    assert_array_equal(npX**2, npI)

def test_npY() -> None:
    """Test square of sigma_y of npY"""
    assert_array_equal(npY**2, npI)

def test_npZ() -> None:
    """Test square of sigma_z of npZ"""
    assert_array_equal(npZ**2, npI)

