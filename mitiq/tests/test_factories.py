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
    return A + B*x

def f_non_lin(x: float) -> float:
    """Non-linear function."""
    return A + B*x + C*x**2


def apply_algorithm(algo_class: Factory, f: Callable[[float], float], order: float = None) -> float:
    """Applies a generc algorithm for extrapolating a given test function f(x).
    Returns an estimate of f(0).

    Args:
        algo_class: class of type Factory corresponding to a specific extrapolation method.
        f: test function to be extrapolated.
        order: (optional) extrapolation order.
    """
    y_vals = [f(x) for x in X_VALS]
    algo_object = algo_class(X_VALS, X_VALS, y_vals)
    if order is None:
        return algo_object.reduce(X_VALS, y_vals)
    return algo_object.reduce(X_VALS, y_vals, order)

def test_richardson_extr():
    """Tests the Richardson's extrapolator."""
    for f in [f_lin, f_non_lin]:
        f_of_zero = apply_algorithm(RichardsonFactory, f)
        assert np.isclose(f_of_zero, f(0), atol=1.e-7)

def test_linear_extr():
    """Tests the linear extrapolator."""
    f_of_zero = apply_algorithm(LinearFactory, f_lin)
    assert np.isclose(f_of_zero, f_lin(0), atol=1.e-7)

def test_poly_extr():
    """Tests the polinomial extrapolator."""
    # test linear extrapolation (order=1)
    f_of_zero = apply_algorithm(PolyFactory, f_lin, 1)
    assert np.isclose(f_of_zero, f_lin(0), atol=1.e-7)
    # test that, for some non-linear functions,
    # order=1 is bad while ored=2 is better.
    f_of_zero = apply_algorithm(PolyFactory, f_non_lin, 1)
    assert not np.isclose(f_of_zero, f_non_lin(0), atol=1)
    f_of_zero = apply_algorithm(PolyFactory, f_non_lin, 2)
    assert np.isclose(f_of_zero, f_non_lin(0), atol=1.e-7)
