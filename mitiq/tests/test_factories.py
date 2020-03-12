"""Testing of zero-noise extrapolation methods (factories) with classically generated data."""

import numpy as np
from mitiq.factories import (
    RichardsonFactory,
    LinearFactory,
    PolyFactory,
    ExpFactory,
    PolyExpFactory,
)
from mitiq.zne import run_factory

# Constant parameters for test functions:
A = 0.5
B = 0.7
C = 0.4
D = 0.3
X_VALS = [1, 1.3, 1.7, 2.2]

STAT_NOISE = 0.0001
CLOSE_TOL = 1.0e-2
NOT_CLOSE_TOL = 1.0e-1

# Classical test functions with statistical error:
def f_lin(x: float, err: float = STAT_NOISE) -> float:
    """Linear function."""
    return A + B * x + np.random.normal(scale=err)


def f_non_lin(x: float, err: float = STAT_NOISE) -> float:
    """Non-linear function."""
    return A + B * x + C * x ** 2 + np.random.normal(scale=err)


def f_exp_down(x: float, err: float = STAT_NOISE) -> float:
    """Exponential decay."""
    return A + B * np.exp(-C * x) + np.random.normal(scale=err)


def f_exp_up(x: float, err: float = STAT_NOISE) -> float:
    """Exponential growth."""
    return A - B * np.exp(-C * x) + np.random.normal(scale=err)


def f_poly_exp_down(x: float, err: float = STAT_NOISE) -> float:
    """Poly-exponential decay."""
    return A + B * np.exp(-C * x - D * x ** 2) + np.random.normal(scale=err)


def f_poly_exp_up(x: float, err: float = STAT_NOISE) -> float:
    """Poly-exponential growth."""
    return A - B * np.exp(-C * x - D * x ** 2) + np.random.normal(scale=err)


def test_richardson_extr():
    """Test of the Richardson's extrapolator."""
    for f in [f_lin, f_non_lin]:
        algo_object = RichardsonFactory(X_VALS)
        run_factory(algo_object, f)
        assert np.isclose(algo_object.reduce(), f(0, err=0), atol=CLOSE_TOL)


def test_linear_extr():
    """Test of linear extrapolator."""
    algo_object = LinearFactory(X_VALS)
    run_factory(algo_object, f_lin)
    assert np.isclose(algo_object.reduce(), f_lin(0, err=0), atol=CLOSE_TOL)


def test_poly_extr():
    """Test of polynomial extrapolator."""
    # test (order=1)
    algo_object = PolyFactory(X_VALS, order=1)
    run_factory(algo_object, f_lin)
    assert np.isclose(algo_object.reduce(), f_lin(0, err=0), atol=CLOSE_TOL)
    # test that, for some non-linear functions,
    # order=1 is bad while ored=2 is better.
    algo_object = PolyFactory(X_VALS, order=1)
    run_factory(algo_object, f_non_lin)
    assert not np.isclose(algo_object.reduce(), f_non_lin(0, err=0), atol=NOT_CLOSE_TOL)
    algo_object = PolyFactory(X_VALS, order=2)
    run_factory(algo_object, f_non_lin)
    assert np.isclose(algo_object.reduce(), f_non_lin(0, err=0), atol=CLOSE_TOL)


def test_exp_factory_with_asympt():
    """Test of exponential extrapolator."""
    for f in [f_exp_down, f_exp_up]:
        algo_object = ExpFactory(X_VALS, asymptote=A)
        run_factory(algo_object, f)
        assert np.isclose(algo_object.reduce(), f(0, err=0), atol=CLOSE_TOL)


def test_poly_exp_factory_with_asympt():
    """Test of (almost) exponential extrapolator."""
    for f in [f_poly_exp_down, f_poly_exp_up]:
        # test that, for a non-linear exponent,
        # order=1 is bad while order=2 is better.
        algo_object = PolyExpFactory(X_VALS, order=1, asymptote=A)
        run_factory(algo_object, f)
        assert not np.isclose(algo_object.reduce(), f(0, err=0), atol=NOT_CLOSE_TOL)
        algo_object = PolyExpFactory(X_VALS, order=2, asymptote=A)
        run_factory(algo_object, f)
        assert np.isclose(algo_object.reduce(), f(0, err=0), atol=CLOSE_TOL)


def test_exp_factory_no_asympt():
    for f in [f_exp_down, f_exp_up]:
        """Test of exponential extrapolator."""
        algo_object = ExpFactory(X_VALS, asymptote=None)
        run_factory(algo_object, f)
        assert np.isclose(algo_object.reduce(), f(0, err=0), atol=CLOSE_TOL)


def test_poly_exp_factory_no_asympt():
    """Test of (almost) exponential extrapolator."""
    for f in [f_poly_exp_down, f_poly_exp_up]:
        # test that, for a non-linear exponent,
        # order=1 is bad while order=2 is better.
        algo_object = PolyExpFactory(X_VALS, order=1, asymptote=None)
        run_factory(algo_object, f)
        assert not np.isclose(algo_object.reduce(), f(0, err=0), atol=NOT_CLOSE_TOL)
        algo_object = PolyExpFactory(X_VALS, order=2, asymptote=None)
        run_factory(algo_object, f)
        assert np.isclose(algo_object.reduce(), f(0, err=0), atol=CLOSE_TOL)
