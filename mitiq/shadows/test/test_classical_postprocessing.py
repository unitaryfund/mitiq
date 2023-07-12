import cirq
import numpy as np

from mitiq.shadows.classical_postprocessing import (
    snapshot_state,
    shadow_state_reconstruction,
    expectation_estimation_shadow,
)


def test_snapshot_state():
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
    result = snapshot_state(b_list, u_list)
    assert isinstance(
        result, np.ndarray
    ), f"Expected a numpy array, got {type(result)}"
    assert result.shape == (2 ** len(b_list), 2 ** len(b_list),), (
        f"Expected shape {(2 ** len(b_list), 2 ** len(b_list))}, "
        f"got {result.shape}"
    )
    assert np.allclose(
        result, expected_result
    ), f"Expected {expected_result}, got {result}"


def test_shadow_state_reconstruction():
    b_lists = np.array([[1, -1, 1], [1, 1, -1], [1, 1, 1]])
    u_lists = np.array([["X", "Y", "Z"], ["Z", "Y", "X"], ["Y", "X", "Z"]])
    measurement_outcomes = (b_lists, u_lists)

    expected_result = np.array(
        [
            # Fill in the expected result based on the specific
            # b_lists and u_lists inputs.
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
    assert isinstance(
        result, np.ndarray
    ), f"Expected a numpy array, got {type(result)}"
    assert result.shape == (
        2**num_qubits,
        2**num_qubits,
    )
    assert np.allclose(
        result, expected_result
    ), f"Expected {expected_result}, but got {result}"


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
    observable = cirq.Z(cirq.LineQubit(0)) * cirq.Z(cirq.LineQubit(1))
    k = 1
    expected_result = -9
    print("expected_result", expected_result)

    result = expectation_estimation_shadow(measurement_outcomes, observable, k)
    assert isinstance(result, float), f"Expected a float, got {type(result)}"
    assert np.isclose(
        result, expected_result
    ), f"Expected {expected_result}, but got {result}"
