import cirq
import pytest

from mitiq.shadows.executor_functions import cirq_simulator_shadow_executor_fn, qiskit_simulator_shadow_executor_fn
from mitiq.typing import MeasurementResult


@pytest.fixture
def executor_fn(request):
    return request.param


@pytest.mark.parametrize("executor_fn", [cirq_simulator_shadow_executor_fn, qiskit_simulator_shadow_executor_fn],
                         indirect=True)
# @pytest.mark.parametrize("seed", [0,1,2,3,4,5,6,7,8,9])
def test_executor_fn(executor_fn):
    num_qubits = 3  # Number of qubits in the state

    # # Create the pure state in the Z basis
    # state_0 = np.zeros(2**num_qubits)
    # state_0[0] = 1  # |0> state

    # state_1 = np.zeros(2**num_qubits)
    # state_1[1] = 1  # |1> state

    # Create a Cirq circuit with measurement
    # degree = 2*np.pi*np.random.rand(0, 1)

    qubits = cirq.LineQubit.range(num_qubits)
    circuit_0 = cirq.Circuit(cirq.I(qubits[0]),
                             cirq.measure(qubits[0]),
                             cirq.X(qubits[1]),
                             cirq.measure(qubits[1]),
                             cirq.I(qubits[2]),
                             cirq.measure(qubits[2])
                             )

    qubits = cirq.LineQubit.range(num_qubits)
    circuit_1 = cirq.Circuit(cirq.X(qubits[0]),
                             cirq.measure(qubits[0]),
                             cirq.X(qubits[1]),
                             cirq.measure(qubits[1]),
                             cirq.I(qubits[2]),
                             cirq.measure(qubits[2])
                             )

    # Run the executor function on the circuits
    outcomes = executor_fn([circuit_0, circuit_1])

    # Check the measurement outcomes for the |0> state
    expected_outcome_0 = MeasurementResult([[0, 1, 0]], tuple(range(num_qubits)))

    print('executor_fn', executor_fn)
    print(outcomes[0], expected_outcome_0)
    assert outcomes[0] == expected_outcome_0, "Measurement outcome for |0> state is incorrect."

    # Check the measurement outcomes for the |1> state
    expected_outcome_1 = MeasurementResult([[1, 1, 0]], tuple(range(num_qubits)))
    # if isinstance(executor_fn, qiskit_simulator_shadow_executor_fn):

    print(outcomes[1], expected_outcome_1)
    assert outcomes[1] == expected_outcome_1, "Measurement outcome for |1> state is incorrect."
