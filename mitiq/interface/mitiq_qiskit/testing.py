from qiskit import QuantumCircuit, Aer, assemble
import numpy as np

from qiskit import IBMQ, Aer
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor
from qiskit import QuantumCircuit, assemble, transpile
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from qiskit_utils import (
    execute,
    execute_with_shots,
    execute_with_noise,
    execute_with_shots_and_noise,
    initialized_depolarizing_noise,
)

qc = QuantumCircuit(2,2)
# Apply H-gate to each qubit:
qc.h(0)
qc.cx(0,1)
qc.measure(0,0)
# See the circuit:
#qc.draw()

# Let's see the result
# svsim = Aer.get_backend('aer_simulator')
# qc.save_statevector()
# qobj = assemble(qc)
# final_state = svsim.run(qobj).result().get_statevector()

print("\nTesting the working of added code in execute with shots")
print("\naer_simulator")
sim = Aer.get_backend('aer_simulator')
qobj = assemble(qc)  # Assemble circuit into a Qobj that can be run
counts = sim.run(qobj,shots=1024).result().get_counts()  # Do the simulation, returning the state vector
#plot_histogram(counts)  # Display the output on measurement of state vector
print("\nSim result:")
print(counts)

result = execute_with_shots(circuit= qc, obs= None, shots=1024, simulator=True, machine_name="aer_simulator",IBMQ_ACCOUNT_TOKEN=None)

print("Execute_with_shots: ")
print(result)

#Testing Other simulators
# Stabilizer simulation method
print("aer_simulator_stabilizer")
sim_stabilizer = Aer.get_backend('aer_simulator_stabilizer')
job_stabilizer = sim_stabilizer.run(qc, shots=1024)
counts_stabilizer = job_stabilizer.result().get_counts()

print("\nSim result:")
print(counts_stabilizer)

result_new = execute_with_shots(circuit= qc, obs= None, shots=1024, simulator=True, machine_name="aer_simulator_stabilizer",IBMQ_ACCOUNT_TOKEN=None)

print("Execute_with_shots: ")
print(result_new)


# Statevector simulation method
print("\naer_simulator_statevector")
sim_statevector = Aer.get_backend('aer_simulator_statevector')
job_statevector = sim_statevector.run(qc, shots=1024)
counts_statevector = job_statevector.result().get_counts()

print("\nSim result:")
print(counts_statevector)

result_new1 = execute_with_shots(circuit= qc, obs= None, shots=1024, simulator=True, machine_name="aer_simulator_statevector",IBMQ_ACCOUNT_TOKEN=None)

print("Execute_with_shots: ")
print(result_new1)


#Add your account tokens
result_real = execute_with_shots(
    circuit=qc, obs=None, shots=1024, simulator=False, machine_name='ibmq_lima',IBMQ_ACCOUNT_TOKEN=''
)

print(result_real)
