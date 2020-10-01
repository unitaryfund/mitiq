# Copyright (C) 2020 Unitary Fund
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Tests for zero-noise inference and extrapolation methods (factories) with
classically generated data.
"""
from copy import copy
from typing import Callable, List
from pytest import mark, raises, warns

import numpy as np
from numpy.random import RandomState

import cirq
from mitiq.zne.inference import (
    ExtrapolationError,
    ExtrapolationWarning,
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
        return func(x, err=err, rnd_state=rnd_state)

    return seeded_func


# Classical test functions with statistical error:
def f_lin(
    x: float, err: float = STAT_NOISE, rnd_state: RandomState = np.random
) -> float:
    """Linear function."""
    return A + B * x + rnd_state.normal(scale=err)


def f_non_lin(
    x: float, err: float = STAT_NOISE, rnd_state: RandomState = np.random
) -> float:
    """Non-linear function."""
    return A + B * x + C * x ** 2 + rnd_state.normal(scale=err)


def f_exp_down(
    x: float, err: float = STAT_NOISE, rnd_state: RandomState = np.random
) -> float:
    """Exponential decay."""
    return A + B * np.exp(-C * x) + rnd_state.normal(scale=err)


def f_exp_up(
    x: float, err: float = STAT_NOISE, rnd_state: RandomState = np.random
) -> float:
    """Exponential growth."""
    return A - B * np.exp(-C * x) + rnd_state.normal(scale=err)


def f_poly_exp_down(
    x: float, err: float = STAT_NOISE, rnd_state: RandomState = np.random
) -> float:
    """Poly-exponential decay."""
    return A + B * np.exp(-C * x - D * x ** 2) + rnd_state.normal(scale=err)


def f_poly_exp_up(
    x: float, err: float = STAT_NOISE, rnd_state: RandomState = np.random
) -> float:
    """Poly-exponential growth."""
    return A - B * np.exp(-C * x - D * x ** 2) + rnd_state.normal(scale=err)


def f_lin_shot(x: float, shots=1) -> float:
    """Linear function with "shots" argument."""
    return A + B * x + 0.001 / np.sqrt(shots)


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


@mark.parametrize(
    "factory",
    (
        LinearFactory,
        RichardsonFactory,
        PolyFactory,
        ExpFactory,
        PolyExpFactory,
    ),
)
def test_get_scale_factors_static_factories(factory):
    scale_factors = np.linspace(1.0, 10.0, num=20)
    if factory is PolyFactory or factory is PolyExpFactory:
        fac = factory(scale_factors=scale_factors, order=2)
    else:
        fac = factory(scale_factors=scale_factors)

    # Expectation values haven't been computed at any scale factors yet
    assert isinstance(fac.get_scale_factors(), np.ndarray)
    assert len(fac.get_scale_factors()) == 0

    # Compute expectation values at all the scale factors
    fac.run_classical(apply_seed_to_func(f_lin, seed=1))
    assert isinstance(fac.get_scale_factors(), np.ndarray)
    assert np.allclose(fac.get_scale_factors(), scale_factors)


@mark.parametrize("factory", (AdaExpFactory,))
def test_get_scale_factors_adaptive_factories(factory):
    num_steps = 8
    fac = AdaExpFactory(steps=num_steps, scale_factor=2.0, asymptote=None)

    # Expectation values haven't been computed at any scale factors yet
    assert isinstance(fac.get_scale_factors(), np.ndarray)
    assert len(fac.get_scale_factors()) == 0

    # Compute expectation values at all the scale factors
    fac.iterate(apply_seed_to_func(f_exp_up, seed=1))
    assert isinstance(fac.get_scale_factors(), np.ndarray)

    # Given this seeded executor, the scale factors should be as follows
    correct_scale_factors = np.array(
        [
            1.0,
            2.0,
            4.0,
            4.20469548,
            4.20310693,
            4.2054822,
            4.2031916,
            4.2052843,
        ]
    )
    assert len(fac.get_scale_factors()) == num_steps
    assert np.allclose(fac.get_scale_factors(), correct_scale_factors)


@mark.parametrize(
    "factory",
    (
        LinearFactory,
        RichardsonFactory,
        PolyFactory,
        ExpFactory,
        PolyExpFactory,
    ),
)
def test_get_expectation_values_static_factories(factory):
    scale_factors = np.linspace(1.0, 10.0, num=20)
    executor = apply_seed_to_func(f_lin, seed=1)
    expectation_values = np.array([executor(scale) for scale in scale_factors])

    if factory is PolyFactory or factory is PolyExpFactory:
        fac = factory(scale_factors=scale_factors, order=2)
    else:
        fac = factory(scale_factors=scale_factors)

    # Expectation values haven't been computed at any scale factors yet
    assert isinstance(fac.get_expectation_values(), np.ndarray)
    assert len(fac.get_expectation_values()) == 0

    # Compute expectation values at all the scale factors
    fac.run_classical(executor)
    assert isinstance(fac.get_expectation_values(), np.ndarray)
    assert np.allclose(
        fac.get_expectation_values(), expectation_values, atol=1e-3
    )


@mark.parametrize("factory", (AdaExpFactory,))
def test_get_expectation_values_adaptive_factories(factory):
    num_steps = 8
    fac = AdaExpFactory(steps=num_steps, scale_factor=2.0, asymptote=None)
    executor = apply_seed_to_func(f_exp_up, seed=1)

    # Expectation values haven't been computed at any scale factors yet
    assert isinstance(fac.get_expectation_values(), np.ndarray)
    assert len(fac.get_expectation_values()) == 0

    # Compute expectation values at all the scale factors
    fac.iterate(executor)
    assert isinstance(fac.get_scale_factors(), np.ndarray)

    # Given this seeded executor, the scale factors should be as follows
    correct_scale_factors = np.array(
        [
            1.0,
            2.0,
            4.0,
            4.20469548,
            4.20310693,
            4.2054822,
            4.2031916,
            4.2052843,
        ]
    )
    correct_expectation_values = np.array(
        [executor(scale) for scale in correct_scale_factors]
    )
    assert len(fac.get_expectation_values()) == num_steps
    assert np.allclose(
        fac.get_expectation_values(), correct_expectation_values, atol=1e-3
    )


@mark.parametrize(
    "factory",
    (
        LinearFactory,
        RichardsonFactory,
        PolyFactory,
        ExpFactory,
        PolyExpFactory,
    ),
)
@mark.parametrize("batched", (True, False))
def test_run_sequential_and_batched(factory, batched):
    scale_factors = np.linspace(1.0, 10.0, num=20)

    if factory is PolyFactory or factory is PolyExpFactory:
        fac = factory(scale_factors=scale_factors, order=2)
    else:
        fac = factory(scale_factors=scale_factors)

    # Expectation values haven't been computed at any scale factors yet
    assert isinstance(fac.get_expectation_values(), np.ndarray)
    assert len(fac.get_expectation_values()) == 0

    # Compute expectation values at all the scale factors
    if batched:

        def executor(circuits, kwargs) -> List[float]:
            return [1.0] * len(circuits)

    else:

        def executor(circuit):
            return 1.0

    fac.run(
        cirq.Circuit(),
        executor,
        scale_noise=lambda circ, _: circ,
    )
    assert isinstance(fac.get_expectation_values(), np.ndarray)
    assert np.allclose(
        fac.get_expectation_values(), np.ones_like(scale_factors)
    )


@mark.parametrize("test_f", [f_lin, f_non_lin])
def test_richardson_extr(test_f: Callable[[float], float]):
    """Test of the Richardson's extrapolator."""
    seeded_f = apply_seed_to_func(test_f, SEED)
    fac = RichardsonFactory(scale_factors=X_VALS)
    assert fac.opt_params == []
    fac.run_classical(seeded_f)
    zne_value = fac.reduce()
    assert np.isclose(zne_value, seeded_f(0, err=0), atol=CLOSE_TOL)
    assert len(fac.opt_params) == len(X_VALS)
    assert np.isclose(fac.opt_params[-1], zne_value)


def test_linear_extr():
    """Tests extrapolation with a LinearFactory."""
    seeded_f = apply_seed_to_func(f_lin, SEED)
    fac = LinearFactory(X_VALS)
    assert fac.opt_params == []
    fac.run_classical(seeded_f)
    assert np.isclose(fac.reduce(), seeded_f(0, err=0), atol=CLOSE_TOL)
    assert np.allclose(fac.opt_params, [B, A], atol=CLOSE_TOL)


def test_poly_extr():
    """Test of polynomial extrapolator."""
    # test (order=1)
    fac = PolyFactory(X_VALS, order=1)
    fac.run_classical(f_lin)
    assert np.isclose(fac.reduce(), f_lin(0, err=0), atol=CLOSE_TOL)
    # test that, for some non-linear functions,
    # order=1 is bad while order=2 is better.
    seeded_f = apply_seed_to_func(f_non_lin, SEED)
    fac = PolyFactory(X_VALS, order=1)
    fac.run_classical(seeded_f)
    assert not np.isclose(fac.reduce(), seeded_f(0, err=0), atol=NOT_CLOSE_TOL)
    seeded_f = apply_seed_to_func(f_non_lin, SEED)
    fac = PolyFactory(X_VALS, order=2)
    fac.run_classical(seeded_f)
    assert np.isclose(fac.reduce(), seeded_f(0, err=0), atol=CLOSE_TOL)


@mark.parametrize("order", [2, 3, 4, 5])
def test_opt_params_poly_factory(order):
    """Tests that optimal parameters are stored after calling the reduce
    method.
    """
    fac = PolyFactory(scale_factors=np.linspace(1, 10, 10), order=order)
    assert fac.opt_params == []
    fac.run_classical(apply_seed_to_func(f_non_lin, seed=SEED))
    zne_value = fac.reduce()
    assert len(fac.opt_params) == order + 1
    assert np.isclose(fac.opt_params[-1], zne_value)


@mark.parametrize("avoid_log", [False, True])
@mark.parametrize("test_f", [f_exp_down, f_exp_up])
def test_exp_factory_with_asympt(
    test_f: Callable[[float], float], avoid_log: bool
):
    """Test of exponential extrapolator."""
    seeded_f = apply_seed_to_func(test_f, SEED)
    fac = ExpFactory(X_VALS, asymptote=A, avoid_log=True)
    fac.run_classical(seeded_f)
    assert len(fac.opt_params) == 0
    assert np.isclose(fac.reduce(), seeded_f(0, err=0), atol=CLOSE_TOL)

    # There are three parameters to fit in the exponential ansatz
    assert len(fac.opt_params) == 3


@mark.parametrize("test_f", [f_exp_down, f_exp_up])
def test_exp_factory_no_asympt(test_f: Callable[[float], float]):
    """Test of exponential extrapolator."""
    seeded_f = apply_seed_to_func(test_f, SEED)
    fac = ExpFactory(X_VALS, asymptote=None)
    fac.run_classical(seeded_f)
    assert len(fac.opt_params) == 0
    assert np.isclose(fac.reduce(), seeded_f(0, err=0), atol=CLOSE_TOL)

    # There are three parameters to fit in the exponential ansatz
    assert len(fac.opt_params) == 3


@mark.parametrize("avoid_log", [False, True])
@mark.parametrize("test_f", [f_poly_exp_down, f_poly_exp_up])
def test_poly_exp_factory_with_asympt(
    test_f: Callable[[float], float], avoid_log: bool
):
    """Test of (almost) exponential extrapolator."""
    # test that, for a non-linear exponent,
    # order=1 is bad while order=2 is better.
    seeded_f = apply_seed_to_func(test_f, SEED)
    fac = PolyExpFactory(X_VALS, order=1, asymptote=A, avoid_log=avoid_log)
    fac.run_classical(seeded_f)
    assert not np.isclose(fac.reduce(), seeded_f(0, err=0), atol=NOT_CLOSE_TOL)
    seeded_f = apply_seed_to_func(test_f, SEED)
    fac = PolyExpFactory(X_VALS, order=2, asymptote=A, avoid_log=avoid_log)
    fac.run_classical(seeded_f)
    assert len(fac.opt_params) == 0
    assert np.isclose(fac.reduce(), seeded_f(0, err=0), atol=POLYEXP_TOL)

    # There are four parameters to fit for the PolyExpFactory of order 1
    assert len(fac.opt_params) == 4


@mark.parametrize("test_f", [f_poly_exp_down, f_poly_exp_up])
def test_poly_exp_factory_no_asympt(test_f: Callable[[float], float]):
    """Test of (almost) exponential extrapolator."""
    seeded_f = apply_seed_to_func(test_f, SEED)
    # test that, for a non-linear exponent,
    # order=1 is bad while order=2 is better.
    fac = PolyExpFactory(X_VALS, order=1, asymptote=None)
    fac.run_classical(seeded_f)
    assert not np.isclose(fac.reduce(), seeded_f(0, err=0), atol=NOT_CLOSE_TOL)
    seeded_f = apply_seed_to_func(test_f, SEED)
    fac = PolyExpFactory(X_VALS, order=2, asymptote=None)
    fac.run_classical(seeded_f)
    assert np.isclose(fac.reduce(), seeded_f(0, err=0), atol=POLYEXP_TOL)


@mark.parametrize("avoid_log", [False, True])
@mark.parametrize("test_f", [f_exp_down, f_exp_up])
def test_ada_exp_factory_with_asympt(
    test_f: Callable[[float], float], avoid_log: bool
):
    """Test of the adaptive exponential extrapolator."""
    seeded_f = apply_seed_to_func(test_f, SEED)
    fac = AdaExpFactory(
        steps=3, scale_factor=2.0, asymptote=A, avoid_log=avoid_log
    )
    # Note: iterate calls next which calls reduce, so calling fac.iterate with
    # an AdaExpFactory sets the optimal parameters as well. Hence we check that
    # the opt_params are empty before AdaExpFactory.iterate is called.
    assert len(fac.opt_params) == 0
    fac.iterate(seeded_f)
    assert np.isclose(fac.reduce(), seeded_f(0, err=0), atol=CLOSE_TOL)

    # There are three parameters to fit for the (adaptive) exponential ansatz
    assert len(fac.opt_params) == 3


@mark.parametrize("avoid_log", [False, True])
@mark.parametrize("test_f", [f_exp_down, f_exp_up])
def test_ada_exp_fac_with_asympt_more_steps(
    test_f: Callable[[float], float], avoid_log: bool
):
    """Test of the adaptive exponential extrapolator with more steps."""
    seeded_f = apply_seed_to_func(test_f, SEED)
    fac = AdaExpFactory(
        steps=6, scale_factor=2.0, asymptote=A, avoid_log=avoid_log
    )
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
    test_f: Callable[[float], float],
):
    """Test of the adaptive exponential extrapolator."""
    seeded_f = apply_seed_to_func(test_f, SEED)
    fac = AdaExpFactory(steps=8, scale_factor=2.0, asymptote=None)
    fac.iterate(seeded_f)
    assert np.isclose(fac.reduce(), seeded_f(0, err=0), atol=CLOSE_TOL)


def test_avoid_log_keyword():
    """Test that avoid_log=True and avoid_log=False give different results."""
    fac = ExpFactory(X_VALS, asymptote=A, avoid_log=False)
    fac.run_classical(f_exp_down)
    znl_with_log = fac.reduce()
    fac.avoid_log = True
    znl_without_log = fac.reduce()
    assert not znl_with_log == znl_without_log


@mark.parametrize("factory", (LinearFactory, RichardsonFactory))
def test_too_few_scale_factors(factory):
    """Test less than 2 scale_factors."""
    with raises(ValueError, match=r"At least 2 scale factors are necessary"):
        _ = factory([1])


def test_few_scale_factors_error():
    """Test that a wrong initialization error is raised."""
    with raises(ValueError, match=r"The extrapolation order cannot exceed"):
        _ = PolyFactory(X_VALS, order=10)


def test_too_few_points_for_polyfit_warning():
    """Test that the correct warning is raised if data is not enough to fit."""
    fac = PolyFactory(X_VALS, order=2)
    fac._instack = [
        {"scale_factor": 1.0, "shots": 100},
        {"scale_factor": 2.0, "shots": 100},
    ]
    fac._outstack = [1.0, 2.0]
    with warns(
        ExtrapolationWarning,
        match=r"The extrapolation fit may be ill-conditioned.",
    ):
        fac.reduce()
    # test also the static "extrapolate" method.
    with warns(
        ExtrapolationWarning,
        match=r"The extrapolation fit may be ill-conditioned.",
    ):
        PolyFactory.extrapolate([1.0, 2.0], [1.0, 2.0], order=2)


def test_failing_fit_error():
    """Test error handling for a failing fit."""
    fac = ExpFactory(X_VALS, asymptote=None)
    fac._instack = [{"scale_factor": x} for x in X_VALS]
    fac._outstack = [1.0, 2.0, 1.0, 2.0, 1.0]
    with raises(
        ExtrapolationError, match=r"The extrapolation fit failed to converge."
    ):
        fac.reduce()
    # test also the static "extrapolate" method.
    with raises(
        ExtrapolationError, match=r"The extrapolation fit failed to converge."
    ):
        ExpFactory.extrapolate(X_VALS, [1.0, 2.0, 1.0, 2.0, 1.0])


@mark.parametrize("fac", [LinearFactory([1, 1, 1]), ExpFactory([1, 1, 1])])
def test_failing_fit_warnings(fac):
    """Test that the correct warning is raised for an ill-conditioned fit."""
    fac._instack = [{"scale_factor": 1.0} for _ in range(4)]
    fac._outstack = [1, 1, 1, 1]
    with warns(
        ExtrapolationWarning,
        match=r"The extrapolation fit may be ill-conditioned.",
    ):
        fac.reduce()
    # test also the static "extrapolate" method.
    with warns(
        ExtrapolationWarning,
        match=r"The extrapolation fit may be ill-conditioned.",
    ):
        fac.extrapolate([1, 1, 1, 1], [1.0, 1.0, 1.0, 1.0])


@mark.parametrize("factory", (LinearFactory, RichardsonFactory, PolyFactory))
def test_equal(factory):
    """Tests that copies are factories are equal to the original factories."""
    for iterate in (True, False):
        if factory is PolyFactory:
            fac = factory(
                scale_factors=[1, 2, 3], order=2, shot_list=[1, 2, 3]
            )
        else:
            fac = factory(scale_factors=[1, 2, 3], shot_list=[1, 2, 3])
        if iterate:
            fac.run_classical(
                scale_factor_to_expectation_value=lambda x, shots: np.exp(x)
                + 0.5
            )

        copied_factory = copy(fac)
        assert copied_factory == fac
        assert copied_factory is not fac

        if iterate:
            fac.reduce()
            copied_factory = copy(fac)
            assert copied_factory == fac
            assert copied_factory is not fac


@mark.parametrize("fac_class", [LinearFactory, RichardsonFactory])
def test_iterate_with_shot_list(fac_class):
    """Tests factories with (and without) the "shot_list" argument."""
    # first test without shot_list
    fac = fac_class(X_VALS)
    fac.run_classical(f_lin_shot)
    assert np.isclose(fac.reduce(), f_lin_shot(0), atol=CLOSE_TOL)

    # Check instack and outstack are as expected
    SHOT_LIST = [100, 200, 300, 400, 500]
    for j, shots in enumerate(SHOT_LIST):
        assert fac._instack[j] == {"scale_factor": X_VALS[j]}
        assert fac._outstack[j] != f_lin_shot(X_VALS[j], shots=shots)
        assert fac._outstack[j] == f_lin_shot(X_VALS[j])

    # Now pass an arbitrary shot_list as an argument
    fac = fac_class(X_VALS, shot_list=SHOT_LIST)
    fac.run_classical(f_lin_shot)
    assert np.isclose(fac.reduce(), f_lin_shot(0), atol=CLOSE_TOL)

    # Check instack and outstack are as expected
    for j, shots in enumerate(SHOT_LIST):
        assert fac._instack[j] == {"scale_factor": X_VALS[j], "shots": shots}
        assert fac._outstack[j] == f_lin_shot(X_VALS[j], shots=shots)
        assert fac._outstack[j] != f_lin_shot(X_VALS[j])


def test_shot_list_errors():
    """Tests errors related to the "shot_lists" argument."""
    with raises(IndexError, match=r"must have the same length."):
        PolyFactory(X_VALS, order=2, shot_list=[1, 2])
    with raises(TypeError, match=r"valid iterator of integers"):
        PolyFactory(X_VALS, order=2, shot_list=[1.0, 2])


def test_push_after_already_reduced_warning():
    """Tests a warning is raised if new data is pushed in a factory
    which was already reduced."""
    fac = LinearFactory([1, 2])
    fac.push({"scale_factor": 1.0}, 1.0)
    fac.push({"scale_factor": 2.0}, 2.0)
    fac.reduce()
    with warns(
        ExtrapolationWarning,
        match=r"You are pushing new data into a factory object",
    ):
        fac.push({"scale_factor": 3.0}, 3.0)
    # Assert no warning is raised when .reset() is used
    fac.reset()
    fac.push({"scale_factor": 1.0}, 2.0)
    fac.push({"scale_factor": 2.0}, 1.0)
    assert np.isclose(3.0, fac.reduce())


def test_full_output_keyword():
    """Tests the full_output keyword in extrapolate method."""
    zlim = LinearFactory.extrapolate([1, 2], [1, 2])
    assert np.isclose(zlim, 0.0)
    zlim, opt_params = LinearFactory.extrapolate(
        [1, 2], [1, 2], full_output=True
    )
    assert len(opt_params) == 2
    assert np.isclose(zlim, 0.0)
    assert np.isclose(0.0, opt_params[1])
    assert np.isclose(1.0, opt_params[0])
