"""
Testing of zero-noise extrapolation methods
(factories) with classically generated data.
"""
from copy import copy
from typing import Callable
from pytest import mark, raises, warns
import numpy as np
from numpy.random import RandomState
from mitiq.factories import (
    ExtrapolationError,
    ExtrapolationWarning,
    ConvergenceWarning,
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
X_VALS = [1, 1.3, 1.7, 2.2, 2.4]

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


@mark.parametrize("avoid_log", [False, True])
@mark.parametrize("test_f", [f_exp_down, f_exp_up])
def test_exp_factory_with_asympt(test_f: Callable[[float], float],
                                 avoid_log: bool):
    """Test of exponential extrapolator."""
    seeded_f = apply_seed_to_func(test_f, SEED)
    fac = ExpFactory(X_VALS, asymptote=A, avoid_log=True)
    fac.iterate(seeded_f)
    assert np.isclose(fac.reduce(), seeded_f(0, err=0), atol=CLOSE_TOL)


@mark.parametrize("test_f", [f_exp_down, f_exp_up])
def test_exp_factory_no_asympt(test_f: Callable[[float], float]):
    """Test of exponential extrapolator."""
    seeded_f = apply_seed_to_func(test_f, SEED)
    fac = ExpFactory(X_VALS, asymptote=None)
    fac.iterate(seeded_f)
    assert np.isclose(fac.reduce(), seeded_f(0, err=0), atol=CLOSE_TOL)


@mark.parametrize("avoid_log", [False, True])
@mark.parametrize("test_f", [f_poly_exp_down, f_poly_exp_up])
def test_poly_exp_factory_with_asympt(test_f: Callable[[float], float],
                                      avoid_log: bool):
    """Test of (almost) exponential extrapolator."""
    # test that, for a non-linear exponent,
    # order=1 is bad while order=2 is better.
    seeded_f = apply_seed_to_func(test_f, SEED)
    fac = PolyExpFactory(X_VALS, order=1, asymptote=A, avoid_log=avoid_log)
    fac.iterate(seeded_f)
    assert not np.isclose(
        fac.reduce(), seeded_f(0, err=0), atol=NOT_CLOSE_TOL
    )
    seeded_f = apply_seed_to_func(test_f, SEED)
    fac = PolyExpFactory(X_VALS, order=2, asymptote=A, avoid_log=avoid_log)
    fac.iterate(seeded_f)
    assert np.isclose(fac.reduce(), seeded_f(0, err=0), atol=POLYEXP_TOL)


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


@mark.parametrize("avoid_log", [False, True])
@mark.parametrize("test_f", [f_exp_down, f_exp_up])
def test_ada_exp_factory_with_asympt(test_f: Callable[[float], float],
                                     avoid_log: bool):
    """Test of the adaptive exponential extrapolator."""
    seeded_f = apply_seed_to_func(test_f, SEED)
    fac = AdaExpFactory(steps=3,
                        scale_factor=2.0,
                        asymptote=A,
                        avoid_log=avoid_log)
    fac.iterate(seeded_f)
    assert np.isclose(fac.reduce(), seeded_f(0, err=0), atol=CLOSE_TOL)


@mark.parametrize("avoid_log", [False, True])
@mark.parametrize("test_f", [f_exp_down, f_exp_up])
def test_ada_exp_fac_with_asympt_more_steps(test_f: Callable[[float], float],
                                            avoid_log: bool):
    """Test of the adaptive exponential extrapolator with more steps."""
    seeded_f = apply_seed_to_func(test_f, SEED)
    fac = AdaExpFactory(steps=6,
                        scale_factor=2.0,
                        asymptote=A,
                        avoid_log=avoid_log)
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


def test_avoid_log_keyword():
    """Test that avoid_log=True and avoid_log=False give different results."""
    fac = ExpFactory(X_VALS, asymptote=A, avoid_log=False)
    fac.iterate(f_exp_down)
    znl_with_log = fac.reduce()
    fac.avoid_log = True
    znl_without_log = fac.reduce()
    assert not znl_with_log == znl_without_log


def test_few_scale_factors_error():
    """Test that a wrong initialization error is raised."""
    with raises(ValueError, match=r"The extrapolation order cannot exceed"):
        _ = PolyFactory(X_VALS, order=10)


def test_few_points_error():
    """Test that the correct error is raised if data is not enough to fit."""
    fac = PolyFactory(X_VALS, order=2)
    fac.instack = [1.0, 2.0]
    fac.outstack = [1.0, 2.0]
    with raises(ValueError, match=r"Extrapolation order is too high."):
        fac.reduce()


def test_failing_fit_error():
    """Test error handling for a failing fit."""
    fac = ExpFactory(X_VALS, asymptote=None)
    fac.instack = X_VALS
    fac.outstack = [1.0, 2.0, 1.0, 2.0, 1.0]
    with raises(ExtrapolationError,
                match=r"The extrapolation fit failed to converge."):
        fac.reduce()


@mark.parametrize("fac", [LinearFactory([1, 1, 1]), ExpFactory([1, 1, 1])])
def test_failing_fit_warnings(fac):
    """Test that the correct warning is raised for an ill-conditioned fit."""
    fac.instack = [1, 1, 1, 1]
    fac.outstack = [1, 1, 1, 1]
    with warns(ExtrapolationWarning,
               match=r"The extrapolation fit may be ill-conditioned."):
        fac.reduce()


def test_iteration_warnings():
    """Test that the correct warning is raised beyond the iteration limit."""
    fac = LinearFactory(X_VALS)
    with warns(ConvergenceWarning,
               match=r"Factory iteration loop stopped before convergence."):
        fac.iterate(lambda scale_factor: 1.0, max_iterations=3)


@mark.parametrize(
    "factory", (LinearFactory, RichardsonFactory, PolyFactory)
)
def test_equal(factory):
    """Tests that copies are factories are equal to the original factories."""
    for iterate in (True, False):
        if factory is PolyFactory:
            fac = factory(scale_factors=[1, 2, 3], order=2)
        else:
            fac = factory(scale_factors=[1, 2, 3])

        if iterate:
            fac.iterate(noise_to_expval=lambda x: np.exp(x) + 0.5)

        copied_factory = copy(fac)
        assert copied_factory == fac
        assert copied_factory is not fac

        if iterate:
            fac.reduce()
            copied_factory = copy(fac)
            assert copied_factory == fac
            assert copied_factory is not fac
