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

"""Unit tests for zero-noise extrapolation."""
import functools

import numpy as np
import pytest
import cirq
import qiskit

from mitiq.zne import (
    inference,
    scaling,
    execute_with_zne,
    mitigate_executor,
    zne_decorator,
)
from mitiq.zne.inference import (
    AdaExpFactory,
    LinearFactory,
    PolyFactory,
    RichardsonFactory,
)
from mitiq.zne.scaling import (
    fold_gates_from_left,
    fold_gates_from_right,
    fold_gates_at_random,
)
from mitiq._typing import QPROGRAM
from mitiq.benchmarks.randomized_benchmarking import generate_rb_circuits

from mitiq.interface.mitiq_qiskit import (
    execute_with_shots_and_noise,
    initialized_depolarizing_noise,
)

from mitiq._typing import SUPPORTED_PROGRAM_TYPES
from mitiq.interface import convert_from_mitiq, accept_any_qprogram_as_input
from mitiq.interface.mitiq_cirq import (
    sample_bitstrings,
    compute_density_matrix,
)
from mitiq.observable import Observable, PauliString


BASE_NOISE = 0.007
TEST_DEPTH = 30
ONE_QUBIT_GS_PROJECTOR = np.array([[1, 0], [0, 0]])

npX = np.array([[0, 1], [1, 0]])
"""Defines the sigma_x Pauli matrix in SU(2) algebra as a (2,2) `np.array`."""

npZ = np.array([[1, 0], [0, -1]])
"""Defines the sigma_z Pauli matrix in SU(2) algebra as a (2,2) `np.array`."""

# Default qubit register and circuit for unit tests
qreg = cirq.GridQubit.rect(2, 1)
circ = cirq.Circuit(cirq.ops.H.on_each(*qreg), cirq.measure_each(*qreg))


@accept_any_qprogram_as_input
def generic_executor(circuit, noise_level: float = 0.1) -> float:
    """Executor that simulates a circuit of any type and returns
    the expectation value of the ground state projector."""
    noisy_circuit = circuit.with_noise(cirq.depolarize(p=noise_level))
    result = cirq.DensityMatrixSimulator().simulate(noisy_circuit)
    return result.final_density_matrix[0, 0].real


# Default executor for unit tests
def executor(circuit) -> float:
    wavefunction = circuit.final_state_vector()
    return np.real(wavefunction.conj().T @ np.kron(npX, npZ) @ wavefunction)


@pytest.mark.parametrize(
    "executor", (sample_bitstrings, compute_density_matrix)
)
def test_with_observable_batched_factory(executor):
    observable = Observable(PauliString(spec="Z"))
    circuit = cirq.Circuit(cirq.H.on(cirq.LineQubit(0))) * 20

    noisy_value = observable.expectation(circuit, sample_bitstrings)
    zne_value = execute_with_zne(
        circuit,
        executor=functools.partial(executor, noise_model=cirq.depolarize),
        observable=observable,
        factory=PolyFactory(scale_factors=[1, 3, 5], order=2),
    )
    true_value = observable.expectation(
        circuit, functools.partial(compute_density_matrix, noise_level=(0,))
    )

    assert abs(zne_value - true_value) <= abs(noisy_value - true_value)


@pytest.mark.parametrize(
    "executor", (sample_bitstrings, compute_density_matrix)
)
def test_with_observable_adaptive_factory(executor):
    observable = Observable(PauliString(spec="Z"))
    circuit = cirq.Circuit(cirq.H.on(cirq.LineQubit(0))) * 20

    noisy_value = observable.expectation(circuit, sample_bitstrings)
    zne_value = execute_with_zne(
        circuit,
        executor=functools.partial(executor, noise_model=cirq.amplitude_damp),
        observable=observable,
        factory=AdaExpFactory(steps=4, asymptote=0.5),
    )
    true_value = observable.expectation(
        circuit, functools.partial(compute_density_matrix, noise_level=(0,))
    )

    assert abs(zne_value - true_value) <= abs(noisy_value - true_value)


@pytest.mark.parametrize(
    "executor", (sample_bitstrings, compute_density_matrix)
)
def test_with_observable_two_qubits(executor):
    observable = Observable(
        PauliString(spec="XX", coeff=-1.21), PauliString(spec="ZZ", coeff=0.7)
    )
    circuit = cirq.Circuit(
        cirq.H.on(cirq.LineQubit(0)), cirq.CNOT.on(*cirq.LineQubit.range(2))
    )
    circuit += [circuit, cirq.inverse(circuit)] * 20

    noisy_value = observable.expectation(circuit, sample_bitstrings)
    zne_value = execute_with_zne(
        circuit,
        executor=functools.partial(executor, noise_model=cirq.depolarize),
        observable=observable,
        factory=PolyFactory(scale_factors=[1, 3, 5], order=2),
    )
    true_value = observable.expectation(
        circuit, functools.partial(compute_density_matrix, noise_level=(0,))
    )

    assert abs(zne_value - true_value) <= 3 * abs(noisy_value - true_value)


@pytest.mark.parametrize(
    "fold_method",
    [fold_gates_from_left, fold_gates_from_right, fold_gates_at_random],
)
@pytest.mark.parametrize("factory", [LinearFactory, RichardsonFactory])
@pytest.mark.parametrize("num_to_average", [1, 2, 5])
def test_execute_with_zne_no_noise(fold_method, factory, num_to_average):
    """Tests execute_with_zne with noiseless simulation."""
    zne_value = execute_with_zne(
        circ,
        executor,
        num_to_average=num_to_average,
        scale_noise=fold_method,
        factory=factory([1.0, 2.0, 3.0]),
    )
    assert np.isclose(zne_value, 0.0)


@pytest.mark.parametrize("factory", [LinearFactory, RichardsonFactory])
@pytest.mark.parametrize(
    "fold_method",
    [fold_gates_from_left, fold_gates_from_right, fold_gates_at_random],
)
def test_averaging_improves_zne_value_with_fake_noise(factory, fold_method):
    """Tests that averaging with Gaussian noise produces a better ZNE value
    compared to not averaging with several folding methods.

    For non-deterministic folding, the ZNE value with average should be better.
    For deterministic folding, the ZNE value should be the same.
    """
    for seed in range(5):
        rng = np.random.RandomState(seed)

        def noisy_executor(circuit) -> float:
            return executor(circuit) + rng.randn()

        zne_value_no_averaging = execute_with_zne(
            circ,
            noisy_executor,
            num_to_average=1,
            scale_noise=fold_gates_at_random,
            factory=factory([1.0, 2.0, 3.0]),
        )

        zne_value_averaging = execute_with_zne(
            circ,
            noisy_executor,
            num_to_average=10,
            scale_noise=fold_method,
            factory=factory([1.0, 2.0, 3.0]),
        )

        # True (noiseless) value is zero. Averaging should ==> closer to zero.
        assert abs(zne_value_averaging) <= abs(zne_value_no_averaging)


def test_execute_with_zne_bad_arguments():
    """Tests errors are raised when execute_with_zne is called with bad
    arguments.
    """
    with pytest.raises(TypeError, match="Argument `factory` must be of type"):
        execute_with_zne(circ, executor, factory=RichardsonFactory)

    with pytest.raises(TypeError, match="Argument `scale_noise` must be"):
        execute_with_zne(circ, executor, scale_noise=None)


def test_error_zne_decorator():
    """Tests that the proper error is raised if the decorator is
    used without parenthesis.
    """
    with pytest.raises(TypeError, match="Decorator must be used with paren"):

        @zne_decorator
        def test_executor(circuit):
            return 0


def test_doc_is_preserved():
    """Tests that the doc of the original executor is preserved."""

    def first_executor(circuit):
        """Doc of the original executor."""
        return 0

    mit_executor = mitigate_executor(first_executor)
    assert mit_executor.__doc__ == first_executor.__doc__

    @zne_decorator()
    def second_executor(circuit):
        """Doc of the original executor."""
        return 0

    assert second_executor.__doc__ == first_executor.__doc__


def qiskit_measure(circuit, qid) -> qiskit.QuantumCircuit:
    """Helper function to measure one qubit."""
    # Ensure that we have a classical register of enough size available
    if len(circuit.clbits) == 0:
        reg = qiskit.ClassicalRegister(qid + 1, "cbits")
        circuit.add_register(reg)
    circuit.measure(0, qid)
    return circuit


def qiskit_executor(qp: QPROGRAM, shots: int = 10000) -> float:
    # initialize a qiskit noise model
    expectation = execute_with_shots_and_noise(
        qp,
        shots=shots,
        obs=ONE_QUBIT_GS_PROJECTOR,
        noise_model=initialized_depolarizing_noise(BASE_NOISE),
        seed=1,
    )
    return expectation


def get_counts(circuit: qiskit.QuantumCircuit):
    return (
        qiskit.execute(
            circuit, qiskit.Aer.get_backend("aer_simulator"), shots=100
        )
        .result()
        .get_counts()
    )


def test_qiskit_execute_with_zne():
    true_zne_value = 1.0

    circuit = qiskit_measure(
        *generate_rb_circuits(
            n_qubits=1,
            num_cliffords=TEST_DEPTH,
            trials=1,
            return_type="qiskit",
        ),
        0,
    )
    base = qiskit_executor(circuit)
    zne_value = execute_with_zne(circuit, qiskit_executor)
    assert abs(true_zne_value - zne_value) < abs(true_zne_value - base)


@zne_decorator()
def qiskit_decorated_executor(qp: QPROGRAM) -> float:
    return qiskit_executor(qp)


def test_qiskit_mitigate_executor():
    true_zne_value = 1.0

    circuit = qiskit_measure(
        *generate_rb_circuits(
            n_qubits=1,
            num_cliffords=TEST_DEPTH,
            trials=1,
            return_type="qiskit",
        ),
        0,
    )
    base = qiskit_executor(circuit)

    mitigated_executor = mitigate_executor(qiskit_executor)
    zne_value = mitigated_executor(circuit)
    assert abs(true_zne_value - zne_value) < abs(true_zne_value - base)


def test_qiskit_zne_decorator():
    true_zne_value = 1.0

    circuit = qiskit_measure(
        *generate_rb_circuits(
            n_qubits=1,
            num_cliffords=TEST_DEPTH,
            trials=1,
            return_type="qiskit",
        ),
        0,
    )
    base = qiskit_executor(circuit)

    zne_value = qiskit_decorated_executor(circuit)
    assert abs(true_zne_value - zne_value) < abs(true_zne_value - base)


def test_qiskit_run_factory_with_number_of_shots():
    true_zne_value = 1.0

    scale_factors = [1.0, 3.0]
    shot_list = [10_000, 30_000]

    fac = inference.ExpFactory(
        scale_factors=scale_factors, shot_list=shot_list, asymptote=0.5,
    )

    circuit = qiskit_measure(
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
        circuit, qiskit_executor, scale_noise=scaling.fold_gates_at_random,
    ).reduce()

    assert abs(true_zne_value - zne_value) < abs(true_zne_value - base)

    for i in range(len(fac._instack)):
        assert fac._instack[i] == {
            "scale_factor": scale_factors[i],
            "shots": shot_list[i],
        }


def test_qiskit_mitigate_executor_with_shot_list():
    true_zne_value = 1.0

    scale_factors = [1.0, 3.0]
    shot_list = [10_000, 30_000]

    fac = inference.ExpFactory(
        scale_factors=scale_factors, shot_list=shot_list, asymptote=0.5,
    )
    mitigated_executor = mitigate_executor(qiskit_executor, factory=fac)

    circuit = qiskit_measure(
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
def test_qiskit_measurement_order_is_preserved_single_register(order):
    """Tests measurement order is preserved when folding, i.e., the dictionary
    of counts is the same as the original circuit on a noiseless simulator.
    """
    qreg, creg = (
        qiskit.QuantumRegister(len(order)),
        qiskit.ClassicalRegister(len(order)),
    )
    circuit = qiskit.QuantumCircuit(qreg, creg)

    circuit.x(qreg[0])
    for i in order:
        circuit.measure(qreg[i], creg[i])

    folded = scaling.fold_gates_at_random(circuit, scale_factor=1.0)

    assert get_counts(folded) == get_counts(circuit)


def test_qiskit_measurement_order_is_preserved_two_registers():
    """Tests measurement order is preserved when folding, i.e., the dictionary
    of counts is the same as the original circuit on a noiseless simulator.
    """
    n = 4
    qreg = qiskit.QuantumRegister(n)
    creg1, creg2 = (
        qiskit.ClassicalRegister(n // 2),
        qiskit.ClassicalRegister(n // 2),
    )
    circuit = qiskit.QuantumCircuit(qreg, creg1, creg2)

    circuit.x(qreg[0])
    circuit.x(qreg[2])

    # Some order of measurements.
    circuit.measure(qreg[0], creg2[1])
    circuit.measure(qreg[1], creg1[0])
    circuit.measure(qreg[2], creg1[1])
    circuit.measure(qreg[3], creg2[1])

    folded = scaling.fold_gates_at_random(circuit, scale_factor=1.0)

    assert get_counts(folded) == get_counts(circuit)


@pytest.mark.parametrize("circuit_type", SUPPORTED_PROGRAM_TYPES.keys())
def test_execute_with_zne_with_supported_circuits(circuit_type):
    # Define a circuit equivalent to the identity
    qreg = cirq.LineQubit.range(2)
    cirq_circuit = cirq.Circuit(
        cirq.H.on_each(qreg),
        cirq.CNOT(*qreg),
        cirq.CNOT(*qreg),
        cirq.H.on_each(qreg),
    )
    # Convert to one of the supported program types
    circuit = convert_from_mitiq(cirq_circuit, circuit_type)
    expected = generic_executor(circuit, noise_level=0.0)
    unmitigated = generic_executor(circuit)
    # Use odd scale factors for deterministic results
    fac = RichardsonFactory([1.0, 3.0, 5.0])
    zne_value = execute_with_zne(circuit, generic_executor, factory=fac)
    # Test zero noise limit is better than unmitigated expectation value
    assert abs(unmitigated - expected) > abs(zne_value - expected)


def test_execute_with_zne_transpiled_qiskit_circuit():
    """Tests ZNE when transpiling to a Qiskit device. Note transpiling can
    introduce idle (unused) qubits to the circuit.
    """
    from qiskit.test.mock import FakeSantiago

    santiago = FakeSantiago()
    backend = qiskit.providers.aer.AerSimulator.from_backend(santiago)

    def execute(circuit: qiskit.QuantumCircuit, shots: int = 8192) -> float:
        job = qiskit.execute(circuit, backend, shots=shots)
        return job.result().get_counts().get("00", 0.0) / shots

    qreg = qiskit.QuantumRegister(2)
    creg = qiskit.ClassicalRegister(2)
    circuit = qiskit.QuantumCircuit(qreg, creg)
    for _ in range(10):
        circuit.x(qreg)

    circuit.measure(qreg, creg)
    circuit = qiskit.transpile(circuit, backend, optimization_level=0)

    true_value = 1.0
    zne_value = execute_with_zne(circuit, execute)

    # Note: Unmitigated value is also (usually) within 10% of the true value.
    # This is more to test usage than effectiveness.
    assert abs(zne_value - true_value) < 0.1
