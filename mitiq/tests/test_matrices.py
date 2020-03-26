"""Testing of zero-noise extrapolation methods (factories) with classically generated data."""
import numpy as np
from mitiq.matrices import npI, npX, npY, npZ


def test_npX() -> None:
    """Test square of identity in npI is a projector"""
    assert npI**2 == npI

def test_npX() -> None:
    """Test square of sigma_y of npX"""
    assert npX**2 == npI

def test_npY() -> None:
    """Test square of sigma_y of npY"""
    assert npY**2 == npI

def test_npZ() -> None:
    """Test square of sigma_z of npZ"""
    assert npZ**2 == npI

