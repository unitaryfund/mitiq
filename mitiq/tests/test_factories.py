"""
Testing of zero-noise extrapolation methods
(factories) with classically generated data.
"""

from typing import Callable
from pytest import mark
import numpy as np
from mitiq.factories import (
    RichardsonFactory,
    LinearFactory,
    PolyFactory,
    ExpFactory,
    PolyExpFactory,
    AdaExpFactory,
    BatchedFactory,
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
# PolyExp fit is non-linear, so we set a larger tolerance 
POLYEXP_TOL = 2 * CLOSE_TOL 
NOT_CLOSE_TOL = 1.0e-1

# Set a seed.
SEED = 808

# Set a random state for classical noise.
rnd_state = np.random.RandomState(SEED)


def reset_rnd_state(seed):
    """Called in each test to reset the seed."""
    global rnd_state
    rnd_state = np.random.RandomState(seed)

# Classical test functions with statistical error:
def f_lin(x: float, err: float = STAT_NOISE) -> float:
    """Linear function."""
    return A + B * x + rnd_state.normal(scale=err)


def f_non_lin(x: float, err: float = STAT_NOISE) -> float:
    """Non-linear function."""
    return A + B * x + C * x ** 2 + rnd_state.normal(scale=err)


def f_exp_down(x: float, err: float = STAT_NOISE) -> float:
    """Exponential decay."""
    return A + B * np.exp(-C * x) + rnd_state.normal(scale=err)


def f_exp_up(x: float, err: float = STAT_NOISE) -> float:
    """Exponential growth."""
    return A - B * np.exp(-C * x) + rnd_state.normal(scale=err)


def f_poly_exp_down(x: float, err: float = STAT_NOISE) -> float:
    """Poly-exponential decay."""
    return A + B * np.exp(-C * x - D * x ** 2) + rnd_state.normal(scale=err)


def f_poly_exp_up(x: float, err: float = STAT_NOISE) -> float:
    """Poly-exponential growth."""
    return A - B * np.exp(-C * x - D * x ** 2) + rnd_state.normal(scale=err)


@mark.parametrize("test_f", [f_lin, f_non_lin])
def test_noise_seeding(test_f: Callable[[float], float]):
    """Check seeding work as expected."""
    reset_rnd_state(SEED)
    noise_a = test_f(0)
    noise_b = test_f(0)
    reset_rnd_state(SEED)
    noise_c = test_f(0)
    assert noise_a != noise_b
    assert noise_a == noise_c


@mark.parametrize("test_f", [f_lin, f_non_lin])
def test_richardson_extr(test_f: Callable[[float], float]):
    """Test of the Richardson's extrapolator."""
    reset_rnd_state(SEED)
    fac = RichardsonFactory(X_VALS)
    run_factory(fac, test_f)
    assert np.isclose(fac.reduce(), test_f(0, err=0), atol=CLOSE_TOL)


def test_linear_extr():
    """Test of linear extrapolator."""
    reset_rnd_state(SEED)
    fac = LinearFactory(X_VALS)
    run_factory(fac, f_lin)
    assert np.isclose(fac.reduce(), f_lin(0, err=0), atol=CLOSE_TOL)


def test_poly_extr():
    """Test of polynomial extrapolator."""
    reset_rnd_state(SEED)
    # test (order=1)
    fac = PolyFactory(X_VALS, order=1)
    run_factory(fac, f_lin)
    assert np.isclose(fac.reduce(), f_lin(0, err=0), atol=CLOSE_TOL)
    # test that, for some non-linear functions,
    # order=1 is bad while ored=2 is better.
    reset_rnd_state(SEED)
    fac = PolyFactory(X_VALS, order=1)
    run_factory(fac, f_non_lin)
    assert not np.isclose(
        fac.reduce(), f_non_lin(0, err=0), atol=NOT_CLOSE_TOL
    )
    reset_rnd_state(SEED)
    fac = PolyFactory(X_VALS, order=2)
    run_factory(fac, f_non_lin)
    assert np.isclose(
        fac.reduce(), f_non_lin(0, err=0), atol=CLOSE_TOL
    )


@mark.parametrize("test_f", [f_exp_down, f_exp_up])
def test_exp_factory_with_asympt(test_f: Callable[[float], float]):
    """Test of exponential extrapolator."""
    reset_rnd_state(SEED)
    fac = ExpFactory(X_VALS, asymptote=A)
    run_factory(fac, test_f)
    assert np.isclose(fac.reduce(), test_f(0, err=0), atol=CLOSE_TOL)


@mark.parametrize("test_f", [f_poly_exp_down, f_poly_exp_up])
def test_poly_exp_factory_with_asympt(test_f: Callable[[float], float]):
    """Test of (almost) exponential extrapolator."""
    reset_rnd_state(SEED)
    # test that, for a non-linear exponent,
    # order=1 is bad while order=2 is better.
    fac = PolyExpFactory(X_VALS, order=1, asymptote=A)
    run_factory(fac, test_f)
    assert not np.isclose(
        fac.reduce(), test_f(0, err=0), atol=NOT_CLOSE_TOL
    )
    reset_rnd_state(SEED)
    fac = PolyExpFactory(X_VALS, order=2, asymptote=A)
    run_factory(fac, test_f)
    assert np.isclose(fac.reduce(), test_f(0, err=0), atol=POLYEXP_TOL)


@mark.parametrize("test_f", [f_exp_down, f_exp_up])
def test_exp_factory_no_asympt(test_f: Callable[[float], float]):
    """Test of exponential extrapolator."""
    reset_rnd_state(SEED)
    fac = ExpFactory(X_VALS, asymptote=None)
    run_factory(fac, test_f)
    assert np.isclose(fac.reduce(), test_f(0, err=0), atol=CLOSE_TOL)


@mark.parametrize("test_f", [f_poly_exp_down, f_poly_exp_up])
def test_poly_exp_factory_no_asympt(test_f: Callable[[float], float]):
    """Test of (almost) exponential extrapolator."""
    reset_rnd_state(SEED)
    # test that, for a non-linear exponent,
    # order=1 is bad while order=2 is better.
    fac = PolyExpFactory(X_VALS, order=1, asymptote=None)
    run_factory(fac, test_f)
    assert not np.isclose(
        fac.reduce(), test_f(0, err=0), atol=NOT_CLOSE_TOL
    )
    reset_rnd_state(SEED)
    fac = PolyExpFactory(X_VALS, order=2, asymptote=None)
    run_factory(fac, test_f)
    assert np.isclose(fac.reduce(), test_f(0, err=0), atol=POLYEXP_TOL)


@mark.parametrize("test_f", [f_exp_down, f_exp_up])
def test_ada_exp_factory_with_asympt(test_f: Callable[[float], float]):
    """Test of the adaptive exponential extrapolator."""
    reset_rnd_state(SEED)
    fac = AdaExpFactory(steps=3, scale_factor=2.0, asymptote=A)
    run_factory(fac, test_f)
    assert np.isclose(fac.reduce(), test_f(0, err=0), atol=CLOSE_TOL)


@mark.parametrize("test_f", [f_exp_down, f_exp_up])
def test_ada_exp_factory_with_asympt_more_steps(
    test_f: Callable[[float], float]
):
    """Test of the adaptive exponential extrapolator."""
    reset_rnd_state(SEED)
    fac = AdaExpFactory(steps=6, scale_factor=2.0, asymptote=A)
    run_factory(fac, test_f)
    assert np.isclose(fac.reduce(), test_f(0, err=0), atol=CLOSE_TOL)


@mark.parametrize("test_f", [f_exp_down, f_exp_up])
def test_ada_exp_factory_no_asympt(test_f: Callable[[float], float]):
    """Test of the adaptive exponential extrapolator."""
    reset_rnd_state(SEED)
    fac = AdaExpFactory(steps=4, scale_factor=2.0, asymptote=None)
    run_factory(fac, test_f)
    assert np.isclose(fac.reduce(), test_f(0, err=0), atol=CLOSE_TOL)


@mark.parametrize("test_f", [f_exp_down, f_exp_up])
def test_ada_exp_factory_no_asympt_more_steps(
    test_f: Callable[[float], float]
):
    """Test of the adaptive exponential extrapolator."""
    reset_rnd_state(SEED)
    fac = AdaExpFactory(steps=8, scale_factor=2.0, asymptote=None)
    run_factory(fac, test_f)
    assert np.isclose(fac.reduce(), test_f(0, err=0), atol=CLOSE_TOL)
