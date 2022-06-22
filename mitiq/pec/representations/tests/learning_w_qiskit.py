import numpy as np
import qiskit

from mitiq import Executor
from mitiq.interface.mitiq_qiskit.qiskit_utils import initialized_depolarizing_noise
from mitiq.cdr import generate_training_circuits
from mitiq.pec.representations.learning import _biased_noise_loss_function


qreg, creg = qiskit.QuantumRegister(2), qiskit.ClassicalRegister(2)
circuit = qiskit.QuantumCircuit(qreg, creg)
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
# Set number of samples used to calculate mitigated value in loss function
pec_kwargs = {"num_samples": 100}


training_circuits = generate_training_circuits(
    circuit=circuit,
    num_training_circuits=3,
    fraction_non_clifford=0,
    method_select="uniform",
    method_replace="closest",
)


def ibmq_executor(circuit: qiskit.QuantumCircuit, shots: int = 1000, noise_level: float = 0.7)  -> float:
    """Returns the expectation value to be mitigated.

    Args:
        circuit: Circuit to run.
        shots: Number of times to execute the circuit to compute the expectation value.
    """
 
    # Simulate the circuit with noise
    noise_model = initialized_depolarizing_noise(noise_level)
    job = qiskit.execute(
        experiments=circuit,
        backend=qiskit.Aer.get_backend("qasm_simulator"),
        noise_model=noise_model,
        basis_gates=noise_model.basis_gates,
        shots=shots,
    )

    # Convert from raw measurement counts to the expectation value
    counts = job.result().get_counts()
    if counts.get("0") is None:
        expectation_value = 0.
    else:
        expectation_value = counts.get("0") / shots
    return expectation_value


def ideal_executor(circ: qiskit.QuantumCircuit) -> float:
    return ibmq_executor(circ, noise_level=0.0)


ideal_values = np.array(
    [ideal_executor(t) for t in training_circuits]
)    

unmitigated = ibmq_executor(circuit)
print(f"Unmitigated result {unmitigated:.3f}")


def test_biased_noise_loss_compare_ideal(operations):
    """Test that the loss function is zero when the noise strength is zero"""

    pec_kwargs = {}
    noisy_executor = Executor(ibmq_executor)
    loss = _biased_noise_loss_function(
        params=[0, 0],
        operations_to_mitigate=operations,
        training_circuits=training_circuits,
        ideal_values=ideal_values,
        noisy_executor=noisy_executor,
        pec_kwargs=pec_kwargs,
    )
    return loss

op = qiskit.QuantumCircuit(qiskit.QuantumRegister(2))
op.rx(np.pi / 2, 0)


print(test_biased_noise_loss_compare_ideal([op]))