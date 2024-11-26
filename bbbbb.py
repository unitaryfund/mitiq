from qiskit import QuantumCircuit
from mitiq.interface import convert_to_mitiq
import numpy as np


# Create a circuit with multiple qubits
test_qc = QuantumCircuit(3)

# Apply unsupported gates (sx, cu1, PhaseGate)
test_qc.h(0)  # Add a Hadamard for complexity
test_qc.sx(1)  # Unsupported gate
test_qc.cx(0, 1)  # Add entanglement
test_qc.p(np.pi / 4, 2)  # Phase gate (potentially unsupported)
# test_qc.cu1(np.pi / 3, 1, 2)  # Controlled-U1 gate

# Add layers of operations
for i in range(3):
    test_qc.rx(np.pi / 2, i)
    test_qc.rz(np.pi / 4, i)

# Add controlled operations with unsupported gates
test_qc.cx(1, 0)
test_qc.sx(2)

# Print the circuit
print("Complicated Quantum Circuit:")
print(test_qc)
mitiq_circuit = convert_to_mitiq(test_qc)
print("Conversion successful. Mitiq Circuit:")
print(mitiq_circuit)
# Attempt conversion to Mitiq circuit

