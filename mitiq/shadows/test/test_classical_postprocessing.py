# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Tests for classical post-processing functions for classical shadows."""

import cirq
import numpy as np

import mitiq
from mitiq.shadows.classical_postprocessing import (
    classical_snapshot,
    shadow_state_reconstruction,
    expectation_estimation_shadow,
)


def test_classical_snapshot():
    b_list = [1, -1]
    u_list = ["X", "Y"]
    expected_result = np.array(
        [
            [0.25 + 0.0j, 0.0 + 0.75j, 0.75 + 0.0j, 0.0 + 2.25j],
            [0.0 - 0.75j, 0.25 + 0.0j, 0.0 - 2.25j, 0.75 + 0.0j],
            [0.75 + 0.0j, 0.0 + 2.25j, 0.25 + 0.0j, 0.0 + 0.75j],
            [0.0 - 2.25j, 0.75 + 0.0j, 0.0 - 0.75j, 0.25 + 0.0j],
        ]
    )
    result = classical_snapshot(b_list, u_list)
    assert isinstance(result, np.ndarray)
    assert result.shape == (
        2 ** len(b_list),
        2 ** len(b_list),
    )
    assert np.allclose(result, expected_result)


def test_shadow_state_reconstruction():
    b_lists = np.array([[1, -1, 1], [1, 1, -1], [1, 1, 1]])
    u_lists = np.array([["X", "Y", "Z"], ["Z", "Y", "X"], ["Y", "X", "Z"]])
    measurement_outcomes = (b_lists, u_lists)

    expected_result = np.array(
        [
            [
                [
                    0.5 + 0.0j,
                    -0.5 + 0.0j,
                    0.5 + 0.0j,
                    0.0 + 1.5j,
                    0.5 - 0.5j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                ],
                [
                    -0.5 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 1.5j,
                    -0.25 - 0.75j,
                    0.0 + 0.0j,
                    -0.25 + 0.25j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                ],
                [
                    0.5 + 0.0j,
                    0.0 - 1.5j,
                    0.5 + 0.0j,
                    -0.5 + 0.0j,
                    0.0 - 3.0j,
                    0.0 + 0.0j,
                    0.5 - 0.5j,
                    0.0 + 0.0j,
                ],
                [
                    0.0 - 1.5j,
                    -0.25 + 0.75j,
                    -0.5 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 1.5j,
                    0.0 + 0.0j,
                    -0.25 + 0.25j,
                ],
                [
                    0.5 + 0.5j,
                    0.0 + 0.0j,
                    0.0 + 3.0j,
                    0.0 + 0.0j,
                    0.25 + 0.0j,
                    0.25 + 0.0j,
                    0.5 + 0.75j,
                    0.0 - 0.75j,
                ],
                [
                    0.0 + 0.0j,
                    -0.25 - 0.25j,
                    0.0 + 0.0j,
                    0.0 - 1.5j,
                    0.25 + 0.0j,
                    -0.25 + 0.0j,
                    0.0 - 0.75j,
                    -0.25 + 0.0j,
                ],
                [
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.5 + 0.5j,
                    0.0 + 0.0j,
                    0.5 - 0.75j,
                    0.0 + 0.75j,
                    0.25 + 0.0j,
                    0.25 + 0.0j,
                ],
                [
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    -0.25 - 0.25j,
                    0.0 + 0.75j,
                    -0.25 + 0.0j,
                    0.25 + 0.0j,
                    -0.25 + 0.0j,
                ],
            ]
        ]
    )

    result = shadow_state_reconstruction(measurement_outcomes)
    num_qubits = measurement_outcomes[0].shape[1]
    assert isinstance(result, np.ndarray)
    assert result.shape == (
        2**num_qubits,
        2**num_qubits,
    )
    assert np.allclose(result, expected_result)


def test_expectation_estimation_shadow():
    b_lists = np.array(
        [
            [1, -1, 1, -1],
            [1, -1, -1, 1],
        ]
    )
    u_lists = np.array(
        [
            ["Z", "Z", "X", "X"],
            ["Z", "Z", "X", "X"],
        ]
    )
    measurement_outcomes = (b_lists, u_lists)
    observable = mitiq.PauliString("ZZ", support=(0, 1))
    k = 1
    expected_result = -9
    print("expected_result", expected_result)

    result = expectation_estimation_shadow(measurement_outcomes, observable, k)
    assert isinstance(result, complex), f"Expected a float, got {type(result)}"
    assert np.isclose(result, expected_result)


def test_expectation_estimation_shadow_no_indices():
    """
    Test expectation estimation for a shadow with no matching indices.
    The result should be 0 as there are no matching
    """
    q0, q1, q2 = cirq.LineQubit.range(3)
    observable = mitiq.PauliString("XYZ", support=(0, 1, 2))
    measurement_outcomes = (
        np.array([[-1, 1, -1], [1, -1, 1], [-1, 1, -1]]),
        np.array([["Z", "X", "Y"], ["Y", "Z", "X"], ["Z", "Z", "Y"]]),
    )
    k_shadows = 1

    result = expectation_estimation_shadow(
        measurement_outcomes, observable, k_shadows
    )

    assert result == 0.0
