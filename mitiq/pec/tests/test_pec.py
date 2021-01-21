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

"""Unit tests for PEC."""

from itertools import product
from typing import List, Optional
from functools import partial
import pytest

import numpy as np
import cirq
import pyquil
import qiskit

from mitiq import QPROGRAM
from mitiq.utils import _equal
from mitiq.conversions import convert_to_mitiq, convert_from_mitiq
from mitiq.benchmarks.utils import noisy_simulation

from mitiq.pec import execute_with_pec, NoisyOperation, OperationRepresentation
from mitiq.pec.pec import LargeSampleWarning


# Noisy representations of Pauli operations for testing.
def get_pauli_representations(
    base_noise: float,
    qubits: Optional[List[cirq.Qid]] = None,
) -> List[OperationRepresentation]:
    if qubits is None:
        qreg = cirq.LineQubit.range(2)
    else:
        qreg = qubits
    pauli_ops = [cirq.I, cirq.X, cirq.Y, cirq.Z]

    # Single-qubit representation coefficients.
    epsilon = base_noise * 4 / 3
    c_neg = -(1 / 4) * epsilon / (1 - epsilon)
    c_pos = 1 - 3 * c_neg

    # This does
    #  X = c_neg I + c_pos X + c_neg Y + c_neg Z
    #  Y = c_neg I + c_neg X + c c_pos + c_neg Z
    #  Z = c_neg I + c_neg X + c_neg Y + c_pos Z
    #  for both qubits.
    representations = []
    for q in qreg:
        paulis = [cirq.Circuit(p.on(q)) for p in pauli_ops]
        for p in paulis[1:]:
            representations.append(
                OperationRepresentation(
                    ideal=p,
                    basis_expansion={
                        NoisyOperation(op): c_pos
                        if _equal(op, p) else c_neg for op in paulis
                    }
                )
            )

    # Two-qubit representation coefficients (assuming local noise).
    c_pos_pos = c_pos * c_pos
    c_pos_neg = c_neg * c_pos
    c_neg_neg = c_neg * c_neg

    # TODO: Add equation of what this code is doing.
    cnot_circuit = cirq.Circuit(cirq.CNOT.on(qreg[0], qreg[1]))
    cd = {NoisyOperation(cnot_circuit): c_pos_pos}

    for p in [cirq.Circuit(p.on(q)) for p in pauli_ops[1:] for q in qreg]:
        cd.update({NoisyOperation(cnot_circuit + p): c_pos_neg})

    for (p0, p1) in product(pauli_ops[1:], repeat=2):
        circ = cnot_circuit + cirq.Circuit(p0.on(qreg[0]), p1.on(qreg[1]))
        cd.update({NoisyOperation(circ): c_neg_neg})

    return representations + [OperationRepresentation(cnot_circuit, cd)]


BASE_NOISE = 0.02
pauli_representations = get_pauli_representations(base_noise=BASE_NOISE)
noiseless_pauli_representations = get_pauli_representations(base_noise=0.0)


def serial_executor(circuit: QPROGRAM, noise: float = BASE_NOISE) -> float:
    """A noisy executor function which executes the input circuit with `noise`
    depolarizing noise and returns the expectation value of the ground state
    projector. Simulation will be slow for "large circuits" (> a few qubits).
    """
    circuit, _ = convert_to_mitiq(circuit)

    # Ground state projector.
    d = 2 ** len(circuit.all_qubits())
    obs = np.zeros(shape=(d, d), dtype=np.float32)
    obs[0, 0] = 1.0

    return noisy_simulation(circuit, noise, obs)


def batched_executor(circuits) -> np.ndarray:
    return np.array([serial_executor(circuit) for circuit in circuits])


def noiseless_serial_executor(circuit: QPROGRAM) -> float:
    return serial_executor(circuit, noise=0.0)


def fake_executor(circuit: cirq.Circuit, random_state: np.random.RandomState):
    """A fake executor which just samples from a normal distribution."""
    return random_state.randn()


# Simple circuits for testing.
q0, q1 = cirq.LineQubit.range(2)
oneq_circ = cirq.Circuit(cirq.Z.on(q0), cirq.Z.on(q0))
twoq_circ = cirq.Circuit(cirq.Y.on(q1), cirq.CNOT.on(q0, q1), cirq.Y.on(q1))


def test_execute_with_pec_cirq_trivial_decomposition():
    circuit = cirq.Circuit(cirq.H.on(cirq.LineQubit(0)))
    rep = OperationRepresentation(
        circuit, basis_expansion={NoisyOperation(circuit): 1.0}
    )

    unmitigated = serial_executor(circuit)
    mitigated = execute_with_pec(
        circuit,
        serial_executor,
        representations=[rep],
        force_run_all=False,
        num_samples=100,
        random_state=1,
    )

    assert np.isclose(unmitigated, mitigated)


def test_execute_with_pec_pyquil_trivial_decomposition():
    circuit = pyquil.Program(pyquil.gates.H(0))
    rep = OperationRepresentation(
        circuit, basis_expansion={NoisyOperation(circuit): 1.0}
    )
    unmitigated = serial_executor(circuit)

    mitigated = execute_with_pec(
        circuit,
        serial_executor,
        representations=[rep],
        num_samples=100,
        force_run_all=False,
        random_state=1,
    )

    assert np.isclose(unmitigated, mitigated)


def test_execute_with_pec_qiskit_trivial_decomposition():
    qreg = qiskit.QuantumRegister(1)
    circuit = qiskit.QuantumCircuit(qreg)
    _ = circuit.x(qreg)
    rep = OperationRepresentation(
        circuit, basis_expansion={NoisyOperation(circuit): 1.0}
    )
    unmitigated = serial_executor(circuit)

    mitigated = execute_with_pec(
        circuit,
        serial_executor,
        representations=[rep],
        num_samples=100,
        force_run_all=False,
        random_state=1,
    )

    assert np.isclose(unmitigated, mitigated)


@pytest.mark.parametrize("circuit", [oneq_circ, twoq_circ])
def test_execute_with_pec_cirq_noiseless_decomposition(circuit):
    unmitigated = noiseless_serial_executor(circuit)

    mitigated = execute_with_pec(
        circuit,
        noiseless_serial_executor,
        representations=noiseless_pauli_representations,
        force_run_all=False,
        num_samples=100,
        random_state=1,
    )

    assert np.isclose(unmitigated, mitigated)


@pytest.mark.parametrize("nqubits", [1, 2, 5])
def test_pyquil_noiseless_decomposition_multiqubit(nqubits):
    circuit = pyquil.Program(pyquil.gates.H(q) for q in range(nqubits))

    # Decompose H(q) for each qubit q into Paulis.
    representations = []
    for q in range(nqubits):
        representation = OperationRepresentation(
            ideal=pyquil.Program(pyquil.gates.H(q)),
            basis_expansion={
                NoisyOperation(ideal=pyquil.Program(pyquil.gates.X(q))): 0.5,
                NoisyOperation(ideal=pyquil.Program(pyquil.gates.Z(q))): 0.5,
            }
        )
        representations.append(representation)

    exact = noiseless_serial_executor(circuit)
    pec_value = execute_with_pec(
        circuit,
        noiseless_serial_executor,
        representations=representations,
        num_samples=500,
        random_state=1,
    )
    assert np.isclose(pec_value, exact, atol=0.1)


@pytest.mark.skip(reason="Slow test.")
@pytest.mark.parametrize("nqubits", [1, 2])
def test_qiskit_noiseless_decomposition_multiqubit(nqubits):
    qreg = [qiskit.QuantumRegister(1) for _ in range(nqubits)]
    circuit = qiskit.QuantumCircuit(*qreg)
    for q in qreg:
        circuit.h(q)

    # Decompose H(q) for each qubit q into Paulis.
    representations = []
    for q in qreg:
        opcircuit = qiskit.QuantumCircuit(q)
        opcircuit.h(q)

        xcircuit = qiskit.QuantumCircuit(q)
        xcircuit.x(q)

        zcircuit = qiskit.QuantumCircuit(q)
        zcircuit.z(q)

        representation = OperationRepresentation(
            ideal=opcircuit,
            basis_expansion={
                NoisyOperation(ideal=xcircuit): 0.5,
                NoisyOperation(ideal=zcircuit): 0.5,
            }
        )
        representations.append(representation)

    exact = noiseless_serial_executor(circuit)
    pec_value = execute_with_pec(
        circuit,
        noiseless_serial_executor,
        representations=representations,
        num_samples=500,
        random_state=1,
    )
    assert np.isclose(pec_value, exact, atol=0.1)


@pytest.mark.parametrize("circuit", [oneq_circ, twoq_circ])
@pytest.mark.parametrize("executor", [serial_executor, batched_executor])
@pytest.mark.parametrize("circuit_type", ["cirq", "qiskit", "pyquil"])
def test_execute_with_pec_mitigates_noise(circuit, executor, circuit_type):
    """Tests that execute_with_pec mitigates the error of a noisy
    expectation value.
    """
    circuit = convert_from_mitiq(circuit, circuit_type)

    true_noiseless_value = 1.0
    unmitigated = serial_executor(circuit)

    if circuit_type == "qiskit":
        # Note this is an important subtlety necessary because of conversions.
        reps = get_pauli_representations(
            base_noise=BASE_NOISE,
            qubits=[cirq.NamedQubit(name) for name in ("q_0", "q_1")]
        )
    else:
        reps = pauli_representations

    mitigated = execute_with_pec(
        circuit,
        executor,
        representations=reps,
        force_run_all=False,
        random_state=101,
    )
    error_unmitigated = abs(unmitigated - true_noiseless_value)
    error_mitigated = abs(mitigated - true_noiseless_value)

    assert error_mitigated < error_unmitigated
    assert np.isclose(mitigated, true_noiseless_value, atol=0.1)


@pytest.mark.parametrize("circuit", [oneq_circ, twoq_circ])
@pytest.mark.parametrize("seed", (2, 3))
def test_execute_with_pec_with_different_samples(circuit, seed):
    """Tests that, on average, the error decreases as the number of samples is
    increased.
    """
    errors_few_samples = []
    errors_more_samples = []
    for _ in range(10):
        mitigated = execute_with_pec(
            circuit,
            serial_executor,
            representations=pauli_representations,
            num_samples=10,
            force_run_all=True,
            random_state=seed,
        )
        errors_few_samples.append(abs(mitigated - 1.0))
        mitigated = execute_with_pec(
            circuit,
            serial_executor,
            representations=pauli_representations,
            num_samples=100,
            force_run_all=True,
            random_state=seed,
        )
        errors_more_samples.append(abs(mitigated - 1.0))

    assert np.average(errors_more_samples) < np.average(errors_few_samples)


@pytest.mark.parametrize("num_samples", [100, 1000])
def test_execute_with_pec_error_scaling(num_samples: int):
    """Tests that the error associated to the PEC value scales as
    1/sqrt(num_samples).
    """
    _, error_pec = execute_with_pec(
        oneq_circ,
        partial(fake_executor, random_state=np.random.RandomState(0)),
        representations=pauli_representations,
        num_samples=num_samples,
        force_run_all=True,
        full_output=True,
    )
    # The error should scale as 1/sqrt(num_samples)
    print(error_pec * np.sqrt(num_samples))
    assert np.isclose(error_pec * np.sqrt(num_samples), 1.0, atol=0.1)


@pytest.mark.parametrize("precision", [0.1, 0.05])
def test_precision_option_in_execute_with_pec(precision: float):
    """Tests that the 'precision' argument is used to deduce num_samples."""
    # For a noiseless circuit we expect num_samples = 1/precision^2:
    _, pec_error = execute_with_pec(
        oneq_circ,
        partial(fake_executor, random_state=np.random.RandomState(0)),
        representations=pauli_representations,
        precision=precision,
        force_run_all=True,
        full_output=True,
    )
    # The error should scale as precision
    assert np.isclose(pec_error / precision, 1.0, atol=0.1)

    # If num_samples is given, precision is ignored.
    nsamples = 1000
    _, pec_error = execute_with_pec(
        oneq_circ,
        partial(fake_executor, random_state=np.random.RandomState(0)),
        representations=pauli_representations,
        precision=precision,
        num_samples=nsamples,
        full_output=True,
    )
    # The error should scale as 1/sqrt(num_samples)
    assert not np.isclose(pec_error / precision, 1.0, atol=0.1)
    assert np.isclose(pec_error * np.sqrt(nsamples), 1.0, atol=0.1)


@pytest.mark.parametrize("bad_value", (0, -1, 2))
def test_bad_precision_argument(bad_value: float):
    """Tests that if 'precision' is not within (0, 1] an error is raised."""
    with pytest.raises(ValueError, match="The value of 'precision' should"):
        execute_with_pec(
            oneq_circ,
            serial_executor,
            pauli_representations,
            precision=bad_value
        )


@pytest.mark.skip(reason="Slow test.")
def test_large_sample_size_warning():
    """Tests whether a warning is raised when PEC sample size
    is greater than 10 ** 5.
    """
    with pytest.warns(
        LargeSampleWarning, match=r"The number of PEC samples is very large.",
    ):
        execute_with_pec(
            oneq_circ,
            partial(fake_executor, random_state=np.random.RandomState(0)),
            pauli_representations,
            num_samples=100001
        )
