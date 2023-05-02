import cirq

class Barrier(cirq.Gate):
    def num_qubits(self):
        return 1

    def _decompose_(self, qubits):
        return []

    def _circuit_diagram_info_(self, args):
        return "B"

# # Create a 1-qubit circuit with several gates and custom barrier
# qubit = cirq.LineQubit(0)
# circuit = cirq.Circuit(
#     cirq.PhasedXPowGate(phase_exponent=0.2).on(qubit),
#     Barrier().on(qubit),
#     cirq.S(qubit)**0.1,
#     cirq.XPowGate(exponent=0.2).on(qubit),
#     Barrier().on(qubit),
# )

# print("Cirq Circuit with Custom Barrier:")
# print(circuit)


# print("\nOptimized Cirq Circuit with Custom Barrier:")
# print(cirq.eject_z(circuit))

