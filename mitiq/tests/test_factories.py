# test_algorithms.py

from typing import Callable
import numpy as np
from mitiq.factories import (
    Factory, 
    RichardsonFactory, 
    LinearFactory, 
    PolyFactory,
    DecayFactory,
    PolyDecayFactory,
)
from mitiq.zne import run_factory

# Constant parameters for test functions:
A = 1.2
B = 1.5
C = 1.7
D = 0.9
X_VALS = [1, 1.4, 1.9]
X_VALS_MORE = [1, 1.4, 1.9, 2.3]

# Classical test functions:
def f_lin(x: float) -> float:
    """Linear function."""
    return A + B * x


def f_non_lin(x: float) -> float:
    """Non-linear function."""
    return A + B * x + C * x ** 2


def f_decay(x: float) -> float:
    """Exponential decay function."""
    return A + B * np.exp(-C * x)


def f_poly_decay(x: float) -> float:
    """Polynomial decay function."""
    return A + B * np.exp(-C * x - D * x ** 2)


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

def test_decay_factory_with_asympt():
    """Test of exponential decay extrapolator."""
    algo_object = DecayFactory(X_VALS, asymptote=A)
    run_factory(algo_object, f_decay)
    assert np.isclose(algo_object.reduce(), f_decay(0), atol=1.0e-7)

def test_poly_decay_factory_with_asympt():
    """Test of (almost) exponential decay extrapolator."""
    # test that, for a decay with a non-linear exponent,
    # order=1 is bad while ored=2 is better.
    algo_object = PolyDecayFactory(X_VALS, order=1, asymptote=A)
    run_factory(algo_object, f_poly_decay)
    assert not np.isclose(algo_object.reduce(), f_poly_decay(0), atol=1.0)
    algo_object = PolyDecayFactory(X_VALS, order=2, asymptote=A)
    run_factory(algo_object, f_poly_decay)
    assert np.isclose(algo_object.reduce(), f_poly_decay(0), atol=1.0e-7)

# TODO: don't work if asymptote=None
def test_decay_factory_no_asympt():
    """Test of exponential decay extrapolator."""
    algo_object = DecayFactory(X_VALS_MORE, asymptote=None)
    run_factory(algo_object, f_decay)
    assert np.isclose(algo_object.reduce(), f_decay(0), atol=1.0e-7)

# TODO: don't work if asymptote=None
def test_poly_decay_factory_no_asympt():
    """Test of (almost) exponential decay extrapolator."""
    # test that, for a decay with a non-linear exponent,
    # order=1 is bad while ored=2 is better.
    algo_object = PolyDecayFactory(X_VALS_MORE, order=1, asymptote=A)
    run_factory(algo_object, f_poly_decay)
    assert not np.isclose(algo_object.reduce(), f_poly_decay(0), atol=1.0)
    algo_object = PolyDecayFactory(X_VALS_MORE, order=2, asymptote=A)
    run_factory(algo_object, f_poly_decay)
    assert np.isclose(algo_object.reduce(), f_poly_decay(0), atol=1.0e-7)