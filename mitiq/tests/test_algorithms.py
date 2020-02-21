# test_algorithms.py
import numpy as np
from mitiq.algorithms import RichardsonExtr



# Constant parameters for test functions:
A = 1.2
B = 1.5
C = 1.7
D = 0.9

X_VALS = [1, 1.4, 1.9]

# Some classical functions which are used to test extrapolation methods

def f_lin(x: float) -> float:
    """Linear function."""
    return A + B*x

def f_non_lin(x: float) -> float:
    """Non-linear function."""
    return A + B*x + C*x**2
        


def test_richardson_extr():
    """Test the Richardson's extrapolator with different functions."""
    for f in [f_lin, f_non_lin]:
        y_vals = [f(x) for x in X_VALS]
        f_of_zero = RichardsonExtr.reduce(X_VALS, y_vals)
        assert np.isclose(f_of_zero, A, atol=1.e-7)
