"""
Testing of zero-noise extrapolation methods
(factories) with classically generated data.
"""

from typing import Callable
from pytest import mark
import numpy as np
from numpy.random import RandomState
from mitiq.factories import (
    RichardsonFactory,
    LinearFactory,
    PolyFactory,
    ExpFactory,
    PolyExpFactory,
    AdaExpFactory,
)

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


def apply_seed_to_func(func: Callable, seed: int) -> Callable:
    """Applies the input seed to the input function by
    using a random state and returns the seeded function."""
    rnd_state = RandomState(seed)
    def seeded_func(x: float, err: float = STAT_NOISE) -> float:
        return func(x, err, rnd_state)
    return seeded_func


# Classical test functions with statistical error:
def f_lin(x: float, err: float = STAT_NOISE,
          rnd_state: RandomState = np.random) -> float:
    """Linear function."""
    return A + B * x + rnd_state.normal(scale=err)


def f_non_lin(x: float, err: float = STAT_NOISE,
              rnd_state: RandomState = np.random) -> float:
    """Non-linear function."""
    return A + B * x + C * x ** 2 + rnd_state.normal(scale=err)


def f_exp_down(x: float, err: float = STAT_NOISE,
               rnd_state: RandomState = np.random) -> float:
    """Exponential decay."""
    return A + B * np.exp(-C * x) + rnd_state.normal(scale=err)


def f_exp_up(x: float, err: float = STAT_NOISE,
             rnd_state: RandomState = np.random) -> float:
    """Exponential growth."""
    return A - B * np.exp(-C * x) + rnd_state.normal(scale=err)


def f_poly_exp_down(x: float, err: float = STAT_NOISE,
                    rnd_state: RandomState = np.random) -> float:
    """Poly-exponential decay."""
    return A + B * np.exp(-C * x - D * x ** 2) + rnd_state.normal(scale=err)


def f_poly_exp_up(x: float, err: float = STAT_NOISE,
                  rnd_state: RandomState = np.random) -> float:
    """Poly-exponential growth."""
    return A - B * np.exp(-C * x - D * x ** 2) + rnd_state.normal(scale=err)


@mark.parametrize("test_f", [f_lin, f_non_lin])
def test_noise_seeding(test_f: Callable[[float], float]):
    """Check that seeding works as expected."""
    seeded_f = apply_seed_to_func(test_f, SEED)
    noise_a = seeded_f(0)
    noise_b = seeded_f(0)
    seeded_f = apply_seed_to_func(test_f, SEED)
    noise_c = seeded_f(0)
    assert noise_a != noise_b
    assert noise_a == noise_c


@mark.parametrize("test_f", [f_lin, f_non_lin])
def test_richardson_extr(test_f: Callable[[float], float]):
    """Test of the Richardson's extrapolator."""
    seeded_f = apply_seed_to_func(test_f, SEED)
    fac = RichardsonFactory(X_VALS)
    fac.iterate(seeded_f)
    assert np.isclose(fac.reduce(), seeded_f(0, err=0), atol=CLOSE_TOL)


def test_linear_extr():
    """Test of linear extrapolator."""
    seeded_f = apply_seed_to_func(f_lin, SEED)
    fac = LinearFactory(X_VALS)
    fac.iterate(seeded_f)
    assert np.isclose(fac.reduce(), seeded_f(0, err=0), atol=CLOSE_TOL)


def test_poly_extr():
    """Test of polynomial extrapolator."""
    seeded_f = apply_seed_to_func(f_lin, SEED)
    # test (order=1)
    fac = PolyFactory(X_VALS, order=1)
    fac.iterate(f_lin)
    assert np.isclose(fac.reduce(), f_lin(0, err=0), atol=CLOSE_TOL)
    # test that, for some non-linear functions,
    # order=1 is bad while ored=2 is better.
    seeded_f = apply_seed_to_func(f_non_lin, SEED)
    fac = PolyFactory(X_VALS, order=1)
    fac.iterate(seeded_f)
    assert not np.isclose(
        fac.reduce(), seeded_f(0, err=0), atol=NOT_CLOSE_TOL
    )
    seeded_f = apply_seed_to_func(f_non_lin, SEED)
    fac = PolyFactory(X_VALS, order=2)
    fac.iterate(seeded_f)
    assert np.isclose(
        fac.reduce(), seeded_f(0, err=0), atol=CLOSE_TOL
    )


@mark.parametrize("test_f", [f_exp_down, f_exp_up])
def test_exp_factory_with_asympt(test_f: Callable[[float], float]):
    """Test of exponential extrapolator."""
    seeded_f = apply_seed_to_func(test_f, SEED)
    fac = ExpFactory(X_VALS, asymptote=A)
    fac.iterate(seeded_f)
    assert np.isclose(fac.reduce(), seeded_f(0, err=0), atol=CLOSE_TOL)


@mark.parametrize("test_f", [f_poly_exp_down, f_poly_exp_up])
def test_poly_exp_factory_with_asympt(test_f: Callable[[float], float]):
    """Test of (almost) exponential extrapolator."""
    # test that, for a non-linear exponent,
    # order=1 is bad while order=2 is better.
    seeded_f = apply_seed_to_func(test_f, SEED)
    fac = PolyExpFactory(X_VALS, order=1, asymptote=A)
    fac.iterate(seeded_f)
    assert not np.isclose(
        fac.reduce(), seeded_f(0, err=0), atol=NOT_CLOSE_TOL
    )
    seeded_f = apply_seed_to_func(test_f, SEED)
    fac = PolyExpFactory(X_VALS, order=2, asymptote=A)
    fac.iterate(seeded_f)
    assert np.isclose(fac.reduce(), seeded_f(0, err=0), atol=POLYEXP_TOL)


@mark.parametrize("test_f", [f_exp_down, f_exp_up])
def test_exp_factory_no_asympt(test_f: Callable[[float], float]):
    """Test of exponential extrapolator."""
    seeded_f = apply_seed_to_func(test_f, SEED)
    fac = ExpFactory(X_VALS, asymptote=None)
    fac.iterate(seeded_f)
    assert np.isclose(fac.reduce(), seeded_f(0, err=0), atol=CLOSE_TOL)


@mark.parametrize("test_f", [f_poly_exp_down, f_poly_exp_up])
def test_poly_exp_factory_no_asympt(test_f: Callable[[float], float]):
    """Test of (almost) exponential extrapolator."""
    seeded_f = apply_seed_to_func(test_f, SEED)
    # test that, for a non-linear exponent,
    # order=1 is bad while order=2 is better.
    fac = PolyExpFactory(X_VALS, order=1, asymptote=None)
    fac.iterate(seeded_f)
    assert not np.isclose(
        fac.reduce(), seeded_f(0, err=0), atol=NOT_CLOSE_TOL
    )
    seeded_f = apply_seed_to_func(test_f, SEED)
    fac = PolyExpFactory(X_VALS, order=2, asymptote=None)
    fac.iterate(seeded_f)
    assert np.isclose(fac.reduce(), seeded_f(0, err=0), atol=POLYEXP_TOL)


@mark.parametrize("test_f", [f_exp_down, f_exp_up])
def test_ada_exp_factory_with_asympt(test_f: Callable[[float], float]):
    """Test of the adaptive exponential extrapolator."""
    seeded_f = apply_seed_to_func(test_f, SEED)
    fac = AdaExpFactory(steps=3, scale_factor=2.0, asymptote=A)
    fac.iterate(seeded_f)
    assert np.isclose(fac.reduce(), seeded_f(0, err=0), atol=CLOSE_TOL)


@mark.parametrize("test_f", [f_exp_down, f_exp_up])
def test_ada_exp_factory_with_asympt_more_steps(
    test_f: Callable[[float], float]
):
    """Test of the adaptive exponential extrapolator."""
    seeded_f = apply_seed_to_func(test_f, SEED)
    fac = AdaExpFactory(steps=6, scale_factor=2.0, asymptote=A)
    fac.iterate(seeded_f)
    assert np.isclose(fac.reduce(), seeded_f(0, err=0), atol=CLOSE_TOL)


@mark.parametrize("test_f", [f_exp_down, f_exp_up])
def test_ada_exp_factory_no_asympt(test_f: Callable[[float], float]):
    """Test of the adaptive exponential extrapolator."""
    seeded_f = apply_seed_to_func(test_f, SEED)
    fac = AdaExpFactory(steps=4, scale_factor=2.0, asymptote=None)
    fac.iterate(seeded_f)
    assert np.isclose(fac.reduce(), seeded_f(0, err=0), atol=CLOSE_TOL)


@mark.parametrize("test_f", [f_exp_down, f_exp_up])
def test_ada_exp_factory_no_asympt_more_steps(
    test_f: Callable[[float], float]
):
    """Test of the adaptive exponential extrapolator."""
    seeded_f = apply_seed_to_func(test_f, SEED)
    fac = AdaExpFactory(steps=8, scale_factor=2.0, asymptote=None)
    fac.iterate(seeded_f)
    assert np.isclose(fac.reduce(), seeded_f(0, err=0), atol=CLOSE_TOL)
