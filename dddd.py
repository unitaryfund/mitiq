from qiskit import QuantumCircuit
import numpy as np
import qiskit.circuit.library as qlib
from mitiq.interface import convert_to_mitiq
# Circuit with unsupported gates
test_circuit = QuantumCircuit(2)
test_circuit.sx(0)
test_circuit.p(np.pi / 4, 0)
# Instantiate the CU1Gate with a phase angle (e.g., π/4)
cu1_gate = qlib.CU1Gate(np.pi / 4)

# Append the instantiated CU1Gate to the circuit
test_circuit.append(cu1_gate, [0, 1])
test_circuit.u(0.1, 0.2, 0.3, 1)
test_circuit.append(qlib.ECRGate(), [0, 1])

# Convert using the updated function
mitiq_circuit = convert_to_mitiq(test_circuit)
print("Conversion successful. Mitiq Circuit:")
print(mitiq_circuit)
