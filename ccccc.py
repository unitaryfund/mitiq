from qiskit import QuantumCircuit
from qiskit.transpiler import transpile

# Example circuit with composite gates
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.z(1)
qc.draw('text')

transpiled_unroller = transpile(qc, basis_gates=["u3", "cx"], translation_method="unroller")
transpiled_unroller.draw('text')

transpiled_translator = transpile(qc, basis_gates=["u3", "cx"], translation_method="translator")
transpiled_translator.draw('text')

transpiled_synthesis = transpile(qc, basis_gates=["u3", "cx"], translation_method="synthesis")
transpiled_synthesis.draw('text')
