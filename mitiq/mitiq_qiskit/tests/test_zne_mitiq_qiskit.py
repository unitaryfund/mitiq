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

"""Tests for zero-noise extrapolation with Qiskit front-ends and back-ends."""
import pytest
import numpy as np

import qiskit
from qiskit import (
    Aer,
    ClassicalRegister,
    QuantumCircuit,
    QuantumRegister,
    execute,
)


from mitiq import zne
from mitiq._typing import QPROGRAM
from mitiq.benchmarks.randomized_benchmarking import generate_rb_circuits
from mitiq.mitiq_qiskit.conversions import to_qiskit
from mitiq.mitiq_qiskit import (
    execute_with_shots_and_noise,
    initialized_depolarizing_noise,
)

BASE_NOISE = 0.007
TEST_DEPTH = 30
ONE_QUBIT_GS_PROJECTOR = np.array([[1, 0], [0, 0]])
QASM_SIMULATOR = qiskit.Aer.get_backend("qasm_simulator")


def measure(circuit, qid) -> QuantumCircuit:
    """Helper function to measure one qubit."""
    # Ensure that we have a classical register of enough size available
    if len(circuit.clbits) == 0:
        reg = ClassicalRegister(qid + 1, "cbits")
        circuit.add_register(reg)
    circuit.measure(0, qid)
    return circuit


def qiskit_executor(qp: QPROGRAM, shots: int = 500) -> float:
    # initialize a qiskit noise model
    expectation = execute_with_shots_and_noise(
        qp,
        shots=shots,
        obs=ONE_QUBIT_GS_PROJECTOR,
        noise_model=initialized_depolarizing_noise(BASE_NOISE),
        seed=1,
    )
    return expectation


def get_counts(circuit: QuantumCircuit):
    return (
        execute(circuit, Aer.get_backend("qasm_simulator"), shots=100)
        .result()
        .get_counts()
    )


@zne.zne_decorator()
def decorated_executor(qp: QPROGRAM) -> float:
    return qiskit_executor(qp)


def test_execute_with_zne():
    true_zne_value = 1.0

    circuit = measure(
        *generate_rb_circuits(
            n_qubits=1,
            num_cliffords=TEST_DEPTH,
            trials=1,
            return_type="qiskit",
        ),
        0,
    )
    base = qiskit_executor(circuit)
    zne_value = zne.execute_with_zne(circuit, qiskit_executor)

    assert abs(true_zne_value - zne_value) < abs(true_zne_value - base)


def test_mitigate_executor():
    true_zne_value = 1.0

    circuit = measure(
        *generate_rb_circuits(
            n_qubits=1,
            num_cliffords=TEST_DEPTH,
            trials=1,
            return_type="qiskit",
        ),
        0,
    )
    base = qiskit_executor(circuit)

    mitigated_executor = zne.mitigate_executor(qiskit_executor)
    zne_value = mitigated_executor(circuit)
    assert abs(true_zne_value - zne_value) < abs(true_zne_value - base)


def test_zne_decorator():
    true_zne_value = 1.0

    circuit = measure(
        *generate_rb_circuits(
            n_qubits=1,
            num_cliffords=TEST_DEPTH,
            trials=1,
            return_type="qiskit",
        ),
        0,
    )
    base = qiskit_executor(circuit)

    zne_value = decorated_executor(circuit)
    assert abs(true_zne_value - zne_value) < abs(true_zne_value - base)


def test_run_factory_with_number_of_shots():
    true_zne_value = 1.0

    scale_factors = [1.0, 2.0, 3.0]
    shot_list = [1_000, 2_000, 3_000]

    fac = zne.inference.ExpFactory(
        scale_factors=scale_factors, shot_list=shot_list
    )

    circuit = measure(
        *generate_rb_circuits(
            n_qubits=1,
            num_cliffords=TEST_DEPTH,
            trials=1,
            return_type="qiskit",
        ),
        0,
    )
    base = qiskit_executor(circuit)
    zne_value = fac.run(
        circuit, qiskit_executor, scale_noise=zne.scaling.fold_gates_at_random,
    ).reduce()

    assert abs(true_zne_value - zne_value) < abs(true_zne_value - base)

    for i in range(len(fac._instack)):
        assert fac._instack[i] == {
            "scale_factor": scale_factors[i],
            "shots": shot_list[i],
        }


def test_mitigate_executor_with_shot_list():
    true_zne_value = 1.0

    scale_factors = [1.0, 2.0, 3.0]
    shot_list = [1_000, 2_000, 3_000]

    fac = zne.inference.ExpFactory(
        scale_factors=scale_factors, shot_list=shot_list
    )
    mitigated_executor = zne.mitigate_executor(qiskit_executor, fac)

    circuit = measure(
        *generate_rb_circuits(
            n_qubits=1,
            num_cliffords=TEST_DEPTH,
            trials=1,
            return_type="qiskit",
        ),
        0,
    )
    base = qiskit_executor(circuit)
    zne_value = mitigated_executor(circuit)

    assert abs(true_zne_value - zne_value) < abs(true_zne_value - base)

    for i in range(len(fac._instack)):
        assert fac._instack[i] == {
            "scale_factor": scale_factors[i],
            "shots": shot_list[i],
        }


@pytest.mark.parametrize("order", [(0, 1), (1, 0), (0, 1, 2), (1, 2, 0)])
def test_measurement_order_is_preserved_single_register(order):
    """Tests measurement order is preserved when folding, i.e., the dictionary
    of counts is the same as the original circuit on a noiseless simulator.
    """
    qreg, creg = QuantumRegister(len(order)), ClassicalRegister(len(order))
    circuit = QuantumCircuit(qreg, creg)

    circuit.x(qreg[0])
    for i in order:
        circuit.measure(qreg[i], creg[i])

    folded = zne.scaling.fold_gates_at_random(circuit, scale_factor=1.0)

    assert get_counts(folded) == get_counts(circuit)


def test_measurement_order_is_preserved_two_registers():
    """Tests measurement order is preserved when folding, i.e., the dictionary
    of counts is the same as the original circuit on a noiseless simulator.
    """
    n = 4
    qreg = QuantumRegister(n)
    creg1, creg2 = ClassicalRegister(n // 2), ClassicalRegister(n // 2)
    circuit = QuantumCircuit(qreg, creg1, creg2)

    circuit.x(qreg[0])
    circuit.x(qreg[2])

    # Some order of measurements.
    circuit.measure(qreg[0], creg2[1])
    circuit.measure(qreg[1], creg1[0])
    circuit.measure(qreg[2], creg1[1])
    circuit.measure(qreg[3], creg2[1])

    folded = zne.scaling.fold_gates_at_random(circuit, scale_factor=1.0)

    assert get_counts(folded) == get_counts(circuit)
