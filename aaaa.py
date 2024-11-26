from qiskit import QuantumCircuit
from mitiq.interface import convert_to_mitiq
import qiskit
import qiskit.circuit.library as qlib
import numpy as np

# Create a quantum circuit with the `u` gate
test_qc = QuantumCircuit(1)
test_qc.u(0.1, 0.2, 0.3, 0)  # Apply the `u` gate
print("Quantum Circuit with `u` gate:")
print(test_qc)
decomposed_qc = test_qc.decompose()
mitiq_circuit = convert_to_mitiq(decomposed_qc)
print("Conversion successful. Mitiq Circuit:")
print(mitiq_circuit)