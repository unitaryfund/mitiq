from qiskit import QuantumCircuit
from mitiq.interface import convert_to_mitiq
import qiskit.circuit.library as qlib
import numpy as np

# Create a quantum circuit with the `CU1Gate`
test_qc = QuantumCircuit(2)
test_qc.h(0)  # Add a Hadamard gate
test_qc.h(1)

# Instantiate the CU1Gate with a phase angle (e.g., π/4)
cu1_gate = qlib.CU1Gate(np.pi / 4)

# Append the instantiated CU1Gate to the circuit
test_qc.append(cu1_gate, [0, 1])

print("Quantum Circuit with `CU1Gate`:")
print(test_qc)

# Attempt conversion to Mitiq circuit
try:
    mitiq_circuit = convert_to_mitiq(test_qc)
    print("Conversion successful. Mitiq Circuit:")
    print(mitiq_circuit)
except Exception as e:
    print(f"Error during conversion: {e}")
