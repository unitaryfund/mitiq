from qiskit import QuantumCircuit
from mitiq.interface import convert_to_mitiq
import qiskit.circuit.library as qlib

# Create a quantum circuit with the `ecr` gate
test_qc = QuantumCircuit(2)
test_qc.h(0)  # Add a Hadamard gate
test_qc.h(1)
test_qc.append(qlib.ECRGate(), [0, 1])  # Apply the `ecr` gate
print("Quantum Circuit with `ecr` gate:")
print(test_qc)

# Attempt conversion to Mitiq circuit
try:
    mitiq_circuit = convert_to_mitiq(test_qc)
    print("Conversion successful. Mitiq Circuit:")
    print(mitiq_circuit)
except Exception as e:
    print(f"Error during conversion: {e}")
