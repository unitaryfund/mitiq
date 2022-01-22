import numpy as np
import warnings
warnings.simplefilter("ignore", np.ComplexWarning)

import cirq
from mitiq import cdr, Observable, PauliString


a, b = cirq.LineQubit.range(2)
circuit = cirq.Circuit(
    cirq.H.on(a), # Clifford
    # cirq.H.on(b), # Clifford
    # cirq.rz(1.75).on(a),
    # cirq.rz(2.31).on(b),
    # cirq.CNOT.on(a, b),  # Clifford
    # cirq.rz(-1.17).on(b),
    # cirq.rz(3.23).on(a),
    # cirq.rx(np.pi / 2).on(a),  # Clifford
    cirq.rx(np.pi / 2).on(b),  # Clifford
)

# CDR works better if the circuit is not too short. So we increase its depth.
circuit =  3* circuit

from mitiq.interface.mitiq_cirq import compute_density_matrix

compute_density_matrix(circuit).round(3)

obs = Observable(PauliString("ZZ"), PauliString("X", coeff=-1.75))

def simulate(circuit: cirq.Circuit) -> np.ndarray:
    return compute_density_matrix(circuit, noise_level=(0.0,))

simulate(circuit).round(3)

ideal_measurement = obs.expectation(circuit, simulate).real
print("ideal_measurement = ",ideal_measurement)

unmitigated_measurement = obs.expectation(circuit, compute_density_matrix).real
print("unmitigated_measurement = ", unmitigated_measurement)


mitigated_measurement = cdr.execute_with_cdr(
    circuit,
    compute_density_matrix,
    # observable=obs,
    simulator=simulate,
    seed=0,
).real
print("mitigated_measurement = ", mitigated_measurement)