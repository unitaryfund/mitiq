# test_algorithms.py

from typing import Callable
import numpy as np
from mitiq.factories import Factory, RichardsonFactory, LinearFactory, PolyFactory


# Constant parameters for test functions:
A = 1.2
B = 1.5
C = 1.7
D = 0.9
X_VALS = [1, 1.4, 1.9]

# two test functions (f_lin and f_non_lin):
def f_lin(x: float) -> float:
    """Linear function."""
    return A + B * x


def f_non_lin(x: float) -> float:
    """Non-linear function."""
    return A + B * x + C * x ** 2


def apply_algorithm(algo_object: Factory, f: Callable[[float], float]) -> float:
    """Applies a generc algorithm for extrapolating a given (classical) test function f(x).
    Returns an estimate of f(0).

    Args:
        factory object: instance of a Factory corresponding to a specific extrapolation method.
        f: test function to be extrapolated.
    """
    y_vals = [f(x) for x in X_VALS]
    algo_object.instack = X_VALS
    algo_object.outstack = y_vals
    return algo_object.reduce()


def test_richardson_extr():
    """Test of the Richardson's extrapolator."""
    for f in [f_lin, f_non_lin]:
        algo_object = RichardsonFactory(X_VALS)
        f_of_zero = apply_algorithm(algo_object, f)
        assert np.isclose(f_of_zero, f(0), atol=1.0e-7)


def test_linear_extr():
    """Test of linear extrapolator."""
    algo_object = LinearFactory(X_VALS)
    f_of_zero = apply_algorithm(algo_object, f_lin)
    assert np.isclose(f_of_zero, f_lin(0), atol=1.0e-7)


def test_poly_extr():
    """Test of polynomial extrapolator."""
    # test (order=1)
    algo_object = PolyFactory(X_VALS, 1)
    f_of_zero = apply_algorithm(algo_object, f_lin)
    assert np.isclose(f_of_zero, f_lin(0), atol=1.0e-7)
    
    # test that, for some non-linear functions,
    # order=1 is bad while ored=2 is better.
    algo_object = PolyFactory(X_VALS, 1)
    f_of_zero = apply_algorithm(algo_object, f_non_lin)
    assert not np.isclose(f_of_zero, f_non_lin(0), atol=1) 
    
    algo_object = PolyFactory(X_VALS, 2)
    f_of_zero = apply_algorithm(algo_object, f_non_lin)
    assert np.isclose(f_of_zero, f_non_lin(0), atol=1.0e-7)
