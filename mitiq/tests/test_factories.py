# test_algorithms.py

from typing import Callable
import numpy as np
from mitiq.factories import Factory, RichardsonFactory, LinearFactory, PolyFactory
from mitiq.zne import run_factory

# Constant parameters for test functions:
A = 1.2
B = 1.5
C = 1.7
X_VALS = [1, 1.4, 1.9]

# Classical test functions (f_lin and f_non_lin):
def f_lin(x: float) -> float:
    """Linear function."""
    return A + B * x

def f_non_lin(x: float) -> float:
    """Non-linear function."""
    return A + B * x + C * x ** 2

def test_richardson_extr():
    """Test of the Richardson's extrapolator."""
    for f in [f_lin, f_non_lin]:
        algo_object = RichardsonFactory(X_VALS)
        run_factory(algo_object, f)
        assert np.isclose(algo_object.reduce(), f(0), atol=1.0e-7)

def test_linear_extr():
    """Test of linear extrapolator."""
    algo_object = LinearFactory(X_VALS)
    run_factory(algo_object, f_lin)
    assert np.isclose(algo_object.reduce(), f_lin(0), atol=1.0e-7)

def test_poly_extr():
    """Test of polynomial extrapolator."""
    # test (order=1)
    algo_object = PolyFactory(X_VALS, order=1)
    run_factory(algo_object, f_lin)
    assert np.isclose(algo_object.reduce(), f_lin(0), atol=1.0e-7)
    # test that, for some non-linear functions,
    # order=1 is bad while ored=2 is better.
    algo_object = PolyFactory(X_VALS, order=1)
    run_factory(algo_object, f_non_lin)
    assert not np.isclose(algo_object.reduce(), f_non_lin(0), atol=1) 
    algo_object = PolyFactory(X_VALS, order=2)
    run_factory(algo_object, f_non_lin)
    assert np.isclose(algo_object.reduce(), f_non_lin(0), atol=1.0e-7)
