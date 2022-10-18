import warnings

warnings.simplefilter("ignore", UserWarning)
import numpy as np
from mitiq.pec import execute_with_pec
from mitiq.pec.representations import (
    represent_operation_with_local_biased_noise,
)
from mitiq.cdr import generate_training_circuits
from cirq import (
    CXPowGate,
    MixedUnitaryChannel,
    I,
    X,
    Y,
    Z,
    LineQubit,
    Circuit,
    ops,
    unitary,
    InsertStrategy,
)
from mitiq import Executor, Observable, PauliString
from mitiq.interface.mitiq_cirq import compute_density_matrix
from mitiq.cdr import generate_training_circuits
from mitiq.cdr._testing import random_x_z_cnot_circuit


circuit = random_x_z_cnot_circuit(
    LineQubit.range(2), n_moments=5, random_state=np.random.RandomState(1)
)

CNOT_all_ops = list(circuit.findall_operations_with_gate_type(CXPowGate))
CNOT_op = CNOT_all_ops[0]
index = CNOT_op[0]
op = CNOT_op[1]

num_training_circuits = 5
training_circuits = generate_training_circuits(
        circuit=circuit,
        num_training_circuits=num_training_circuits,
        fraction_non_clifford=0.2,
        random_state=np.random.RandomState(1),
    )


def biased_noise_channel(epsilon: float, eta: float) -> MixedUnitaryChannel:
    a = 1 - epsilon
    b = epsilon * (3 * eta + 1) / (3 * (eta + 1))
    c = epsilon / (3 * (eta + 1))

    mix = [
        (a, unitary(I)),
        (b, unitary(Z)),
        (c, unitary(X)),
        (c, unitary(Y)),
    ]
    return ops.MixedUnitaryChannel(mix)

observable = Observable(PauliString("XZ"), PauliString("YY"))

true_epsilon = 0.05
true_eta = 1


# We assume the operation "op" appears just once in the circuit such
# that it's enough to add a single noise channel after that operation.
def noisy_execute(circ: Circuit) -> np.ndarray:
    noisy_circ = circ.copy()
    qubits = op.qubits
    for q in qubits:
        noisy_circ.insert(
            index + 1,
            biased_noise_channel(true_epsilon, true_eta)(q),
            strategy=InsertStrategy.EARLIEST,
        )
    return compute_density_matrix(noisy_circ, noise_level=(0.0,))

noisy_executor = Executor(noisy_execute)
num_epsilons = num_etas = 121
epsilons = np.linspace(true_epsilon - 0.03, true_epsilon + 0.03, num_epsilons)
etas = np.linspace(true_eta * 0.8, true_eta * 1.4, num_etas)
#  2-D array w/ dim1 noise strenth, dim2 noise bia
pec_values = np.zeros([num_epsilons + 1, num_etas + 1], dtype=float)
pec_values[1:, 0] = epsilons
pec_values[0, 1:] = etas


import timeit

start = timeit.default_timer()
for tc in enumerate(training_circuits):
    pec_data = pec_values
    for et in enumerate(etas): 
        for eps in enumerate(epsilons):
            reps = [represent_operation_with_local_biased_noise(
            ideal_operation=Circuit(op), epsilon=eps[1], eta=et[1],
            )]
            pec_values[eps[0] + 1, et[0] + 1] = execute_with_pec(
                circuit=tc[1],
                executor=noisy_executor,
                observable=observable,
                representations=reps,
                num_samples=600,
                random_state=np.random.RandomState(1)
        )
    np.savetxt("./mitiq/pec/representations/tests//learning_pec_data/learning_pec_data_eps_"
        + str(true_epsilon).replace(".", "_")
        + "eta_"
        + str(true_eta)
        + "tc_"
        + str(tc[0])
        + ".txt"
    , pec_values)
    pec_values = pec_data

end = timeit.default_timer()
print(end - start)
