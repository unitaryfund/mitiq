# Copyright (C) 2020 Unitary Fund
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""pyQuil utitility functions."""
import numpy as np
from pyquil import Program

# Backend and Noise simulation
from pyquil import get_qc
from pyquil.gates import X, Y, Z, MEASURE
from pyquil.noise import append_kraus_to_gate
from pyquil.simulation.matrices import I as npI, X as npX, Y as npY, Z as npZ

QVM = get_qc("1q-qvm")

# Set the random seeds for testing
QVM.qam.random_seed = 1337
np.random.seed(1001)


def random_identity_circuit(depth=None):
    """Returns a single-qubit identity circuit based on Pauli gates."""

    # initialize a quantum circuit
    prog = Program()

    # index of the (inverting) final gate: 0=I, 1=X, 2=Y, 3=Z
    k_inv = 0

    # apply a random sequence of Pauli gates
    for _ in range(depth):
        # random index for the next gate: 1=X, 2=Y, 3=Z
        k = np.random.choice([1, 2, 3])
        # apply the Pauli gate "k"
        if k == 1:
            prog += X(0)
        elif k == 2:
            prog += Y(0)
        elif k == 3:
            prog += Z(0)

        # update the inverse index according to
        # the product rules of Pauli matrices k and k_inv
        if k_inv == 0:
            k_inv = k
        elif k_inv == k:
            k_inv = 0
        else:
            _ = [1, 2, 3]
            _.remove(k_inv)
            _.remove(k)
            k_inv = _[0]

    # apply the final inverse gate
    if k_inv == 1:
        prog += X(0)
    elif k_inv == 2:
        prog += Y(0)
    elif k_inv == 3:
        prog += Z(0)

    return prog


def run_with_noise(circuit: Program, noise: float, shots: int) -> float:
    """Returns the expectation value of a circuit run several times with noise.

    Args:
        circuit: Quantum circuit as :class:`~pyquil.quil.Program`.
        noise: Noise constant for depolarizing channel.
        shots: Number of shots the circuit is run.

    Returns:
        expval: Expectation value.
    """
    # apply depolarizing noise to all gates
    kraus_ops = [
        np.sqrt(1 - noise) * npI,
        np.sqrt(noise / 3) * npX,
        np.sqrt(noise / 3) * npY,
        np.sqrt(noise / 3) * npZ,
    ]
    circuit.define_noisy_gate("X", [0], append_kraus_to_gate(kraus_ops, npX))
    circuit.define_noisy_gate("Y", [0], append_kraus_to_gate(kraus_ops, npY))
    circuit.define_noisy_gate("Z", [0], append_kraus_to_gate(kraus_ops, npZ))

    # set number of shots
    circuit.wrap_in_numshots_loop(shots)

    # we want to simulate noise, so we run without compiling
    results = QVM.run(circuit)
    expval = (results == [0]).sum() / shots
    return expval


def run_program(pq: Program, shots: int = 500) -> float:
    """Returns the expectation value of a circuit run several times.

    Args:
        pq: Quantum circuit as :class:`~pyquil.quil.Program`.
        shots: (Default: 500) Number of shots the circuit is run.

    Returns:
        expval: Expectation value.
    """
    pq.wrap_in_numshots_loop(shots)
    results = QVM.run(pq)
    expval = (results == [0]).sum() / shots
    return expval


def add_depolarizing_noise(pq: Program, noise: float) -> Program:
    """Returns a quantum program with depolarizing channel noise.

    Args:
        pq: Quantum program as :class:`~pyquil.quil.Program`.
        noise: Noise constant for depolarizing channel.

    Returns:
        pq: Quantum program with added noise.
    """
    pq = pq.copy()
    # apply depolarizing noise to all gates
    kraus_ops = [
        np.sqrt(1 - noise) * npI,
        np.sqrt(noise / 3) * npX,
        np.sqrt(noise / 3) * npY,
        np.sqrt(noise / 3) * npZ,
    ]
    pq.define_noisy_gate("X", [0], append_kraus_to_gate(kraus_ops, npX))
    pq.define_noisy_gate("Y", [0], append_kraus_to_gate(kraus_ops, npY))
    pq.define_noisy_gate("Z", [0], append_kraus_to_gate(kraus_ops, npZ))
    return pq


NATIVE_NOISE = 0.007


def scale_noise(pq: Program, param: float) -> Program:
    """Returns a circuit rescaled by the depolarizing noise parameter.

    Args:
        pq: Quantum circuit as :class:`~pyquil.quil.Program`.
        param: noise scaling.

    Returns:
        Quantum program with added noise.
    """
    noise = param * NATIVE_NOISE
    assert noise <= 1.0, (
        "Noise scaled to {} is out of bounds (<=1.0) for "
        "depolarizing channel.".format(noise)
    )
    return add_depolarizing_noise(pq, noise)


def measure(circuit, qid):
    """Returns a circuit adding a register for readout results.

    Args:
        circuit: Quantum circuit as :class:`~pyquil.quil.Program`.
        qid: position of the measurement in the circuit.

    Returns:
        Quantum program with added measurement.
    """
    ro = circuit.declare("ro", "BIT", 1)
    circuit += MEASURE(qid, ro[0])
    return circuit
