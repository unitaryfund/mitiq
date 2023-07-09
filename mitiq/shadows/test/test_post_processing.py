from mitiq.shadows.post_processing import snapshot_state, shadow_state_reconstruction, expectation_estimation_shadow
from mitiq.shadows.shadows_utils import *


# executor = Executor(cirq_simulator_shadow_executor_fn, max_batch_size=int(1e10))

def test_snapshot_state():
    b_list = [1, -1]
    u_list = ['X', 'Y']
    expected_result = np.array(
        # here put the expected result based on the specific b_list and u_list inputs
        [[0.25 + 0.j, 0. + 0.75j, 0.75 + 0.j, 0. + 2.25j],
         [0. - 0.75j, 0.25 + 0.j, 0. - 2.25j, 0.75 + 0.j],
         [0.75 + 0.j, 0. + 2.25j, 0.25 + 0.j, 0. + 0.75j],
         [0. - 2.25j, 0.75 + 0.j, 0. - 0.75j, 0.25 + 0.j]]
    )
    result = snapshot_state(b_list, u_list)
    assert isinstance(result, np.ndarray), f"Expected a numpy array, got {type(result)}"
    assert result.shape == (
    2 ** len(b_list), 2 ** len(b_list)), f"Expected shape {(2 ** len(b_list), 2 ** len(b_list))}, got {result.shape}"
    assert np.allclose(result, expected_result), f"Expected {expected_result}, got {result}"


def test_shadow_state_reconstruction():
    b_lists = np.array([
        [1, -1, 1],
        [1, 1, -1],
        [1, 1, 1]
    ])
    u_lists = np.array([
        ['X', 'Y', 'Z'],
        ['Z', 'Y', 'X'],
        ['Y', 'X', 'Z']
    ])
    measurement_outcomes = (b_lists, u_lists)

    expected_result = np.array([
        # Fill in the expected result based on the specific b_lists and u_lists inputs.
        [[0.5 + 0.j, -0.5 + 0.j, 0.5 + 0.j, 0. + 1.5j, 0.5 - 0.5j, 0. + 0.j,
          0. + 0.j, 0. + 0.j],
         [-0.5 + 0.j, 0. + 0.j, 0. + 1.5j, -0.25 - 0.75j, 0. + 0.j, -0.25 + 0.25j,
          0. + 0.j, 0. + 0.j],
         [0.5 + 0.j, 0. - 1.5j, 0.5 + 0.j, -0.5 + 0.j, 0. - 3.j, 0. + 0.j,
          0.5 - 0.5j, 0. + 0.j],
         [0. - 1.5j, -0.25 + 0.75j, -0.5 + 0.j, 0. + 0.j, 0. + 0.j, 0. + 1.5j,
          0. + 0.j, -0.25 + 0.25j],
         [0.5 + 0.5j, 0. + 0.j, 0. + 3.j, 0. + 0.j, 0.25 + 0.j, 0.25 + 0.j,
          0.5 + 0.75j, 0. - 0.75j],
         [0. + 0.j, -0.25 - 0.25j, 0. + 0.j, 0. - 1.5j, 0.25 + 0.j, -0.25 + 0.j,
          0. - 0.75j, -0.25 + 0.j],
         [0. + 0.j, 0. + 0.j, 0.5 + 0.5j, 0. + 0.j, 0.5 - 0.75j, 0. + 0.75j,
          0.25 + 0.j, 0.25 + 0.j],
         [0. + 0.j, 0. + 0.j, 0. + 0.j, -0.25 - 0.25j, 0. + 0.75j, -0.25 + 0.j,
          0.25 + 0.j, -0.25 + 0.j]]
    ])

    result = shadow_state_reconstruction(measurement_outcomes)
    num_qubits = measurement_outcomes[0].shape[1]
    assert isinstance(result, np.ndarray), f"Expected a numpy array, got {type(result)}"
    assert result.shape == (
    2 ** num_qubits, 2 ** num_qubits), f"Expected shape {(2 ** num_qubits, 2 ** num_qubits)}, got {result.shape}"
    assert np.allclose(result, expected_result), f"Expected {expected_result}, but got {result}"


def test_expectation_estimation_shadow():
    b_lists = np.array([
        [1, -1, 1, -1],
        [1, -1, -1, 1],

    ])
    u_lists = np.array([
        ['Z', 'Z', 'X', 'X'],
        ['Z', 'Z', 'X', 'X'],
    ])
    measurement_outcomes = (b_lists, u_lists)
    observable = cirq.Z(cirq.LineQubit(0)) * cirq.Z(cirq.LineQubit(1))
    k = 1
    expected_result = -9  # This might not be correct. Please replace it with the correct expected result if you have it.
    print('expected_result', expected_result)

    result = expectation_estimation_shadow(measurement_outcomes, observable, k)
    assert isinstance(result, float), f"Expected a float, got {type(result)}"
    assert np.isclose(result, expected_result), f"Expected {expected_result}, but got {result}"
