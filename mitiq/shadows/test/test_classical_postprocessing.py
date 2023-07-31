# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Tests for classical post-processing functions for classical shadows."""

import cirq
import numpy as np

import mitiq
from mitiq.shadows.shadows_utils import operator_ptm_vector_rep
from mitiq.shadows.classical_postprocessing import (
    get_single_shot_pauli_fidelity,
    get_pauli_fidelity,
    get_normalize_factor,
    classical_snapshot,
    shadow_state_reconstruction,
    expectation_estimation_shadow,
)


def test_get_single_shot_pauli_fidelity():
    b_list = "01"
    u_list = "XY"
    expected_result = 1.0
    assert np.isclose(
        get_single_shot_pauli_fidelity(b_list, u_list), expected_result
    )


def test_get_pauli_fidelity():
    calibration_measurement_outcomes = (
        ["00", "01", "10", "11"],
        ["XX", "YY", "ZZ", "XY"],
    )
    k_calibration = 2
    expected_result = {"00": 0.25, "01": 0.25, "10": 0.0, "11": 0.25}
    result = get_pauli_fidelity(
        calibration_measurement_outcomes, k_calibration
    )
    for key in expected_result.keys():
        assert np.isclose(result[key], expected_result[key])


def test_get_normalize_factor():
    f_est = {"00": 1, "01": 1 / 3, "10": 1 / 3, "11": 1 / 9}
    expected_result = 1
    assert np.isclose(get_normalize_factor(f_est), expected_result)


def test_classical_snapshot_cal():
    b_list_shadow = "01"
    u_list_shadow = "XY"
    pauli_twirling_calibration = True
    f_est = {"00": 1, "01": 1 / 3, "10": 1 / 3, "11": 1 / 9}
    expected_result = operator_ptm_vector_rep(
        np.array(
            [
                [0.25 + 0.0j, 0.0 + 0.75j, 0.75 + 0.0j, 0.0 + 2.25j],
                [0.0 - 0.75j, 0.25 + 0.0j, 0.0 - 2.25j, 0.75 + 0.0j],
                [0.75 + 0.0j, 0.0 + 2.25j, 0.25 + 0.0j, 0.0 + 0.75j],
                [0.0 - 2.25j, 0.75 + 0.0j, 0.0 - 0.75j, 0.25 + 0.0j],
            ]
        )
    )
    np.testing.assert_array_almost_equal(
        classical_snapshot(
            b_list_shadow, u_list_shadow, pauli_twirling_calibration, f_est
        ),
        expected_result,
    )


def test_classical_snapshot():
    b_list = "01"
    u_list = "XY"
    expected_result = np.array(
        [
            [0.25 + 0.0j, 0.0 + 0.75j, 0.75 + 0.0j, 0.0 + 2.25j],
            [0.0 - 0.75j, 0.25 + 0.0j, 0.0 - 2.25j, 0.75 + 0.0j],
            [0.75 + 0.0j, 0.0 + 2.25j, 0.25 + 0.0j, 0.0 + 0.75j],
            [0.0 - 2.25j, 0.75 + 0.0j, 0.0 - 0.75j, 0.25 + 0.0j],
        ]
    )
    result = classical_snapshot(b_list, u_list, False)
    assert isinstance(result, np.ndarray)
    assert result.shape == (
        2 ** len(b_list),
        2 ** len(b_list),
    )
    assert np.allclose(result, expected_result)


def test_shadow_state_reconstruction():
    b_lists = ["010", "001", "000"]
    u_lists = ["XYZ", "ZYX", "YXZ"]

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

    result = shadow_state_reconstruction(measurement_outcomes, False)
    num_qubits = len(measurement_outcomes[0])
    assert isinstance(result, np.ndarray)
    assert result.shape == (
        2**num_qubits,
        2**num_qubits,
    )
    assert np.allclose(result, expected_result)


def test_shadow_state_reconstruction_cal():
    b_lists = ["01", "01"]
    u_lists = ["XY", "XY"]
    measurement_outcomes = (b_lists, u_lists)
    f_est = {"00": 1, "01": 1 / 3, "10": 1 / 3, "11": 1 / 9}
    expected_result_vec = operator_ptm_vector_rep(
        np.array(
            [
                [0.25 + 0.0j, 0.0 + 0.75j, 0.75 + 0.0j, 0.0 + 2.25j],
                [0.0 - 0.75j, 0.25 + 0.0j, 0.0 - 2.25j, 0.75 + 0.0j],
                [0.75 + 0.0j, 0.0 + 2.25j, 0.25 + 0.0j, 0.0 + 0.75j],
                [0.0 - 2.25j, 0.75 + 0.0j, 0.0 - 0.75j, 0.25 + 0.0j],
            ]
        )
    )
    result = shadow_state_reconstruction(measurement_outcomes, True, f_est)
    num_qubits = len(measurement_outcomes[0])
    assert isinstance(result, np.ndarray)
    assert result.shape == (4**num_qubits,)
    assert np.allclose(result, expected_result_vec)


def test_expectation_estimation_shadow():
    b_lists = ["0101", "0110"]
    u_lists = ["ZZXX", "ZZXX"]

    measurement_outcomes = (b_lists, u_lists)
    observable = mitiq.PauliString("ZZ", support=(0, 1))
    k = 1
    expected_result = -9

    result = expectation_estimation_shadow(
        measurement_outcomes, observable, k, False
    )
    assert isinstance(result, float), f"Expected a float, got {type(result)}"
    assert np.isclose(result, expected_result)


def test_expectation_estimation_shadow_cal():
    b_lists = ["0101", "0110"]
    u_lists = ["ZZXX", "ZZXX"]
    f_est = {
        "0000": 1,
        "0001": 1 / 3,
        "0010": 1 / 3,
        "0100": 1 / 3,
        "1000": 1 / 3,
        "0011": 1 / 9,
        "0110": 1 / 9,
        "1100": 1 / 9,
        "0101": 1 / 9,
        "1010": 1 / 9,
        "1001": 1 / 9,
        "0111": 1 / 27,
        "1110": 1 / 27,
        "1011": 1 / 27,
        "1101": 1 / 27,
        "1111": 1 / 81,
    }

    measurement_outcomes = b_lists, u_lists
    observable = mitiq.PauliString("ZZ", support=(0, 1))
    k = 1
    expected_result = -9
    print("expected_result", expected_result)

    result = expectation_estimation_shadow(
        measurement_outcomes, observable, k, True, f_est
    )
    assert isinstance(result, float), f"Expected a float, got {type(result)}"
    assert np.isclose(result, expected_result)


def test_expectation_estimation_shadow_no_indices():
    """
    Test expectation estimation for a shadow with no matching indices.
    The result should be 0 as there are no matching
    """
    q0, q1, q2 = cirq.LineQubit.range(3)
    observable = mitiq.PauliString("XYZ", support=(0, 1, 2))
    measurement_outcomes = ["101", "010", "101"], ["ZXY", "YZX", "ZZY"]
    k_shadows = 1

    result = expectation_estimation_shadow(
        measurement_outcomes, observable, k_shadows, False
    )

    assert result == 0.0
