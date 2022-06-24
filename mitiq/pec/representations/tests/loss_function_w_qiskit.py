import numpy as np
import qiskit

from mitiq import Executor
from mitiq.interface.mitiq_qiskit.qiskit_utils import (
    execute_with_noise,
    initialized_depolarizing_noise,
)
from mitiq.cdr import generate_training_circuits
from mitiq.observable.observable import Observable
from mitiq.observable.pauli import PauliString
from mitiq.pec.representations.learning import _biased_noise_loss_function


qreg, creg = qiskit.QuantumRegister(2), qiskit.ClassicalRegister(2)
circuit = qiskit.QuantumCircuit(qreg, creg)
circuit.h(0)
circuit.h(1)
circuit.rz(1.75, 0),
circuit.rz(2.31, 1),
circuit.cnot(0, 1),
circuit.rz(-1.17, 1),
circuit.rz(3.23, 0),
circuit.rx(np.pi / 2, 0),
circuit.rx(np.pi / 2, 1),
circuit.measure(qreg, creg)
print(circuit)


backend = qiskit.Aer.get_backend("qasm_simulator")

training_circuits = generate_training_circuits(
    circuit=circuit,
    num_training_circuits=3,
    fraction_non_clifford=0,
    method_select="uniform",
    method_replace="closest",
)


def ideal_executor(circ: qiskit.QuantumCircuit) -> float:
    noise_model = initialized_depolarizing_noise(noise_level=0)
    observable = Observable(PauliString("ZZ"))
    return execute_with_noise(circ, observable.matrix(), noise_model)


ideal_values = np.array([ideal_executor(t) for t in training_circuits])


def biased_noise_loss_compare_ideal_qiskit(operations) -> float:
    """Test that the loss function is zero when the noise strength is zero"""

    pec_kwargs = {"num_samples": 10}

    def noisy_execute(circ: qiskit.QuantumCircuit) -> float:
        noise_model = initialized_depolarizing_noise(0.0)
        observable = Observable(PauliString("ZZ"))
        return execute_with_noise(circ, observable.matrix(), noise_model)

    noisy_executor = Executor(noisy_execute)

    loss = _biased_noise_loss_function(
        params=[0, 0],
        operations_to_mitigate=operations,
        training_circuits=training_circuits,
        ideal_values=ideal_values,
        noisy_executor=noisy_executor,
        pec_kwargs=pec_kwargs,
    )
    return loss


op = qiskit.QuantumCircuit(qreg, creg)
op.rx(np.pi / 2, 0)

print(biased_noise_loss_compare_ideal_qiskit([op]))
