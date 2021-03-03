.. _guide-executors:

*********************************************
Back-end Plug-ins: Executor Examples
*********************************************

Mitiq uses "executor" functions to abstract different backends.
Executors always accept a quantum program, sometimes accept other
arguments, and always return an expectation value as a float. If your
quantum programming interface of choice can be used
to make a Python function with this type, then it can be used with Mitiq.

These example executors as especially flexible as they
accept an arbitrary observable. You can instead hardcode your choice of
observable in any way you like. All that matters from Mitiq's perspective
is that your executor accepts a quantum program and returns a float.


Cirq Executors
======================================

This section includes noisy and noiseless simulator executor examples using
Cirq.

Cirq: Wavefunction Simulation
---------------------------------

This executor can be used for noiseless simulation. Note that this executor
can be :ref:`wrapped using partial function application <partial-note>`
to be used in Mitiq.

.. testcode::

    import numpy as np
    from cirq import Circuit

    def execute(circ: Circuit, obs: np.ndarray) -> float:
        """Simulates noiseless wavefunction evolution and returns the
        expectation value of some observable.

        Args:
            circ: The input Cirq circuit.
            obs: The observable to measure as a NumPy array.

        Returns:
            The expectation value of obs as a float.
        """
        final_wvf = circ.final_wavefunction()
        return np.real(final_wvf.conj().T @ obs @ final_wvf)

.. testcode::
    :hide:

    from cirq import LineQubit
    import cirq

    qc = Circuit()
    qc += [cirq.X(LineQubit(0)), cirq.CNOT(LineQubit(0), LineQubit(1))]

    print(execute(qc, obs=np.diag([1, 0, 0, 0])))

.. testoutput::
   :hide:

   0.0

.. testcode::
    :hide:

    print(execute(qc, obs=np.diag([0, 0, 0, 1])))

.. testoutput::
    :hide:

    1.0

Cirq: Wavefunction Simulation with Sampling
-----------------------------------------------

We can add in functionality that takes into account some finite number of
samples (aka shots). Here we will use Cirq's ``PauliString`` methods to
construct our observable. You can read more about these methods in the Cirq
documentation `here <https://quantumai.google/reference/python/cirq/ops/PauliString>`_.

.. testcode::

    def execute(circ: Circuit, obs: cirq.PauliString, shots: int) -> float:
        """Simulates noiseless wavefunction evolution and returns the
        expectation value of a PauliString observable.

        Args:
            circ: The input Cirq circuit.
            obs: The observable to measure as a cirq.PauliString.
            shots: The number of measurements.

        Returns:
            The expectation value of obs as a float.
        """

        # Do the sampling
        psum = cirq.PauliSumCollector(circ, obs, samples_per_term=shots)
        psum.collect(sampler=cirq.Simulator())

        # Return the expectation value
        return psum.estimated_energy()

.. testcode::
    :hide:

    ham = cirq.PauliString(cirq.ops.Z.on(LineQubit(0)), cirq.ops.Z.on(LineQubit(1)))
    qc = Circuit()
    qc += [cirq.X(LineQubit(0)), cirq.CNOT(LineQubit(0), LineQubit(1))]

    assert np.isclose(execute(qc, ham, 10000), 1.0)


Cirq: Density-matrix Simulation with Depolarizing Noise
------------------------------------------------------------

This executor can be used for noisy depolarizing simulation.

.. testcode::

    import numpy as np
    from cirq import Circuit, depolarize
    from cirq import DensityMatrixSimulator

    SIMULATOR = DensityMatrixSimulator()

    def execute(circ: Circuit, obs: np.ndarray, noise: float) -> float:
        """Simulates a circuit with depolarizing noise at level noise.

        Args:
            circ: The input Cirq circuit.
            obs: The observable to measure as a NumPy array.
            noise: The depolarizing noise as a float, i.e. 0.001 is 0.1% noise.

        Returns:
            The expectation value of obs as a float.
        """
        circuit = circ.with_noise(depolarize(p=noise))
        rho = SIMULATOR.simulate(circuit).final_density_matrix
        expectation = np.real(np.trace(rho @ obs))
        return expectation

.. testcode::
    :hide:

    qc = Circuit()
    for _ in range(100):
        qc += cirq.X(LineQubit(0))

    assert execute(qc, np.diag([0, 1]), 0.0) == 0.0
    assert np.isclose(execute(qc, np.diag([0, 1]), 0.5), 0.5)
    assert np.isclose(execute(qc, np.diag([0, 1]), 0.001), 0.062452)

Other noise models can be used by substituting the ``depolarize`` channel with
any other channel available in Cirq, for example ``cirq.amplitude_damp``.
More details can be found in the Cirq
`noise documentation <https://quantumai.google/cirq/noise>`__

Cirq: Density-matrix Simulation with Depolarizing Noise and Sampling
------------------------------------------------------------------------

You can also include both noise models and finite sampling in your executor.

.. testcode::

    import numpy as np
    from cirq import Circuit, depolarize
    from cirq import DensityMatrixSimulator

    SIMULATOR = DensityMatrixSimulator()

    def execute(circ: Circuit, obs: cirq.PauliString, noise: float, shots: int) -> float:
        """Simulates a circuit with depolarizing noise at level noise.

        Args:
            circ: The input Cirq circuit.
            obs: The observable to measure as a NumPy array.
            noise: The depolarizing noise strength as a float, i.e. 0.001 is 0.1%.
            shots: The number of measurements.

        Returns:
            The expectation value of obs as a float.
        """
        # add the noise
        noisy = circ.with_noise(depolarize(p=noise))

        # Do the sampling
        psum = cirq.PauliSumCollector(noisy, obs, samples_per_term=shots)
        psum.collect(sampler=cirq.DensityMatrixSimulator())

        # Return the expectation value
        return psum.estimated_energy()

.. testcode::
    :hide:

    qc = Circuit()
    for _ in range(4):
        qc += cirq.X(LineQubit(0))
    qc += cirq.measure(LineQubit(0))
    qc = qc.with_noise(depolarize(p=0.02))
    ham = cirq.PauliString(cirq.ops.Z.on(LineQubit(0)))
    noisy_output = execute(qc, ham, 0.01, 200)
    assert 0.5 < noisy_output < 1.0


.. _pyquil_executors:

PyQuil Executors
================

This section contains executors for use with `pyQuil <https://github.com/rigetti/pyquil>`__.

PyQuil: Quantum Cloud Services
------------------------------

This executor can be used to run on `Quantum Cloud Services <https://arxiv.org/abs/2001.04449>`__
(QCS), the hardware platform provided by Rigetti Computing, and requires a QCS account and
reservation on a quantum processor (QPU).

.. note::
    The module :mod:`mitiq.mitiq_pyquil` has a function ``generate_qcs_executor`` for
    easily generating a QCS executor of this form.

Note that you will have to replace the string in ``get_qc`` with the name of an actual
Rigetti QPU, and will need to have a QCS account and reservation, in order to run on
real quantum hardware.

.. testcode::

    from pyquil import Program, get_qc
    from pyquil.gates import MEASURE, RESET, X

    from mitiq.mitiq_pyquil.compiler import basic_compile
    from mitiq.mitiq_pyquil.pyquil_utils import ground_state_expectation

    # replace with qpu = get_qc("Aspen-8") to run on the Aspen-8 QPU
    qpu = get_qc("2q-pyqvm")

    def execute(program: Program, shots: int = 1000) -> float:
        p = Program()

        # add reset
        p += RESET()

        # add main body program
        p += program.copy()

        # add memory declaration
        qubits = p.get_qubits()
        ro = p.declare("ro", "BIT", len(qubits))

        # add measurements
        for idx, q in enumerate(qubits):
            p += MEASURE(q, ro[idx])

        # add numshots
        p.wrap_in_numshots_loop(shots)

        # nativize the circuit
        p = basic_compile(p)

        # compile the circuit
        b = qpu.compiler.native_quil_to_executable(p)

        # run the circuit, collect bitstrings
        qpu.reset()
        results = qpu.run(b)

        # compute ground state expectation value
        return ground_state_expectation(results)

    # prepare state |11>
    program = Program()
    program += X(0)
    program += X(1)

    # should give 0.0 with a noiseless backend
    execute(program)

.. testcode::
    :hide:

    assert execute(program) == 0.0

.. _qiskit_executors:

Qiskit Executors
======================================

This section includes noisy and noiseless simulator executor examples using
Qiskit.

Qiskit: Wavefunction Simulation
---------------------------------

This executor can be used for noiseless simulation. Note that this executor
can be :ref:`wrapped using partial function application <partial-note>`
to be used in Mitiq.

.. testcode::

    import numpy as np
    import qiskit
    from qiskit import QuantumCircuit

    wvf_simulator = qiskit.Aer.get_backend('statevector_simulator')

    def execute(circ: QuantumCircuit, obs: np.ndarray) -> float:
        """Simulates noiseless wavefunction evolution and returns the
        expectation value of some observable.

        Args:
            circ: The input Qiskit circuit.
            obs: The observable to measure as a NumPy array.

        Returns:
            The expectation value of obs as a float.
        """
        result = qiskit.execute(circ, wvf_simulator).result()
        final_wvf = result.get_statevector()
        return np.real(final_wvf.conj().T @ obs @ final_wvf)

.. testcode::
    :hide:

    qc = QuantumCircuit(2)
    qc.x(0)
    qc.cnot(0, 1)
    assert np.isclose(execute(qc, obs=np.diag([1, 0, 0, 0])), 0.0)
    assert np.isclose(execute(qc, obs=np.diag([0, 0, 0, 1])), 1.0)


Qiskit: Wavefunction Simulation with Sampling
-----------------------------------------------

The above executor can be modified to still perform exact wavefunction simulation,
but to also include finite sampling of measurements. Note that this executor
can be :ref:`wrapped using partial function application <partial-note>`
to be used in Mitiq.

Note that this executor implementation measures arbitrary observables by using
a change of basis into the computational basis. More information behind the math
in this example can be found `here <https://quantumcomputing.stackexchange.com/a/6944>`__.

.. testcode::

    import copy

    QISKIT_SIMULATOR = qiskit.Aer.get_backend("qasm_simulator")

    def execute(circ: QuantumCircuit, obs: np.ndarray, shots: int) -> float:
        """Simulates the evolution of the circuit and returns
        the expectation value of the observable.

        Args:
            circ: The input Qiskit circuit.
            obs: The observable to measure as a NumPy array.
            shots: The number of measurements.

        Returns:
            The expectation value of obs as a float.

        """
        if len(circ.clbits) > 0:
            raise ValueError("This executor only works on programs with no classical bits.")

        circ = copy.deepcopy(circ)
        # we need to modify the circuit to measure obs in its eigenbasis
        # we do this by appending a unitary operation
        eigvals, U = np.linalg.eigh(obs) # obtains a U s.t. obs = U diag(eigvals) U^dag
        circ.unitary(np.linalg.inv(U), qubits=range(circ.num_qubits))

        circ.measure_all()

        # execution of the experiment
        job = qiskit.execute(
            circ,
            backend=QISKIT_SIMULATOR,
            # we want all gates to be actually applied,
            # so we skip any circuit optimization
            optimization_level=0,
            shots=shots
        )
        results = job.result()
        counts = results.get_counts()
        expectation = 0
        # classical bits are included in bitstrings with a space
        # this is what breaks if you have them
        for bitstring, count in counts.items():
            expectation += eigvals[int(bitstring, 2)] * count / shots
        return expectation


.. testcode::
    :hide:

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cnot(0, 1)
    out = execute(qc, obs=np.diag([0, 0, 0, 1]), shots=50)
    assert 0.0 < out < 1.0
    out = execute(qc, obs=np.diag([0, 0, 0, 1]), shots=int(1e5))
    assert abs(out - 0.5) < 0.1

    qc_zero = QuantumCircuit(1)
    out = execute(qc_zero, obs=np.diag([1, 0]), shots=50)
    assert np.isclose(out, 1.0)


Qiskit: Density-matrix Simulation with Depolarizing Noise
-----------------------------------------------------------

Analogous to the Cirq implementation, this executor can be used for noisy depolarizing
simulation.

.. testcode::

    import qiskit
    from qiskit import QuantumCircuit
    import numpy as np
    import copy

    # Noise simulation packages
    from qiskit.providers.aer.noise import NoiseModel
    from qiskit.providers.aer.noise.errors.standard_errors import depolarizing_error
    from qiskit.providers.aer.extensions import snapshot_density_matrix

    QISKIT_SIMULATOR = qiskit.Aer.get_backend("qasm_simulator")

    def execute(circ: QuantumCircuit, obs: np.ndarray, noise: float) -> float:
        """Simulates the evolution of the noisy circuit and returns
        the expectation value of the observable.

        Args:
            circ: The input Cirq circuit.
            obs: The observable to measure as a NumPy array.
            noise: The depolarizing noise strength as a float, i.e. 0.001 is 0.1%.

        Returns:
            The expectation value of obs as a float.
        """
        if len(circ.clbits) > 0:
            raise ValueError("This executor only works on programs with no classical bits.")

        circ = copy.deepcopy(circ)
        # we need to modify the circuit to measure obs in its eigenbasis
        # we do this by appending a unitary operation
        eigvals, U = np.linalg.eigh(obs) # obtains a U s.t. obs = U diag(eigvals) U^dag
        circ.unitary(np.linalg.inv(U), qubits=range(circ.num_qubits))
        circ.snapshot_density_matrix('final')
        circ.measure_all()

        # initialize a qiskit noise model
        noise_model = NoiseModel()

        # we assume the same depolarizing error for each
        # gate of the standard IBM basis
        noise_model.add_all_qubit_quantum_error(depolarizing_error(noise, 1), ["u1", "u2", "u3"])
        noise_model.add_all_qubit_quantum_error(depolarizing_error(noise, 2), ["cx"])

        # execution of the experiment
        job = qiskit.execute(
            circ,
            backend=QISKIT_SIMULATOR,
            backend_options={'method':'density_matrix'},
            noise_model=noise_model,
            # we want all gates to be actually applied,
            # so we skip any circuit optimization
            basis_gates=noise_model.basis_gates,
            optimization_level=0,
            shots=1,
        )
        result = job.result()
        rho = result.data()['snapshots']['density_matrix']['final'][0]['value']

        expectation = np.real(np.trace(rho @ obs))
        return expectation

.. testcode::
    :hide:

    qc = QuantumCircuit(1)
    for _ in range(100):
        qc.u1(0, 0)
    assert 0.1 < execute(qc, np.diag([1, 0]), 0.02) < 1.0

    qc = QuantumCircuit(1)
    for _ in range(100):
        qc.x(0)

    assert execute(qc, np.diag([0, 1]), 0.0) == 0.0
    assert np.isclose(execute(qc, np.diag([0, 1]), 0.5), 0.5)
    assert np.isclose(execute(qc, np.diag([0, 1]), 0.001), 0.0480563)

Qiskit: Density-matrix Simulation with Depolarizing Noise and Sampling
------------------------------------------------------------------------

You can also include both noise models and finite sampling in your executor.

.. testcode::

    import qiskit
    from qiskit import QuantumCircuit
    import numpy as np
    import copy

    # Noise simulation packages
    from qiskit.providers.aer.noise import NoiseModel
    from qiskit.providers.aer.noise.errors.standard_errors import depolarizing_error

    QISKIT_SIMULATOR = qiskit.Aer.get_backend("qasm_simulator")

    def execute(circ: QuantumCircuit, obs: np.ndarray, noise: float, shots: int) -> float:
        """Simulates the evolution of the noisy circuit and returns
        the expectation value of the observable.

        Args:
            circ: The input Cirq circuit.
            obs: The observable to measure as a NumPy array.
            noise: The depolarizing noise strength as a float, i.e. 0.001 is 0.1%.
            shots: The number of measurements.

        Returns:
            The expectation value of obs as a float.
        """
        if len(circ.clbits) > 0:
            raise ValueError("This executor only works on programs with no classical bits.")

        circ = copy.deepcopy(circ)
        # we need to modify the circuit to measure obs in its eigenbasis
        # we do this by appending a unitary operation
        eigvals, U = np.linalg.eigh(obs) # obtains a U s.t. obs = U diag(eigvals) U^dag
        circ.unitary(np.linalg.inv(U), qubits=range(circ.num_qubits))

        circ.measure_all()

        # initialize a qiskit noise model
        noise_model = NoiseModel()

        # we assume the same depolarizing error for each
        # gate of the standard IBM basis
        noise_model.add_all_qubit_quantum_error(depolarizing_error(noise, 1), ["u1", "u2", "u3"])
        noise_model.add_all_qubit_quantum_error(depolarizing_error(noise, 2), ["cx"])

        # execution of the experiment
        job = qiskit.execute(
            circ,
            backend=QISKIT_SIMULATOR,
            backend_options={'method':'density_matrix'},
            noise_model=noise_model,
            # we want all gates to be actually applied,
            # so we skip any circuit optimization
            basis_gates=noise_model.basis_gates,
            optimization_level=0,
            shots=shots,
        )
        results = job.result()
        counts = results.get_counts()
        expectation = 0
        # classical bits are included in bitstrings with a space
        # this is what breaks if you have them
        for bitstring, count in counts.items():
            expectation += eigvals[int(bitstring, 2)] * count / shots
        return expectation

.. testcode::
    :hide:

    qc = QuantumCircuit(1)
    for _ in range(10):
        qc.u1(0, 0)
    assert 0.1 < execute(qc, np.diag([1, 0]), 0.02, 1000) < 1.0

    qc_zero = QuantumCircuit(2)
    out = execute(qc_zero, obs=np.diag([1, 0, 0, 0]), noise=0.0, shots=10)
    assert np.isclose(out, 1.0)
    out = execute(qc_zero, obs=np.diag([1, 0, 0, 0]), noise=1.0, shots=10 ** 5)
    assert np.isclose(out, 0.25, atol=0.1)

Other noise models can be defined using any functionality available in Qiskit.
More details can be found in the
`Qiskit simulator documentation <https://qiskit.org/documentation/tutorials/simulators/index.html>`__

Qiskit: Hardware
------------------------------------------------------------

An example of an executor that runs on IBMQ hardware is given
:ref:`here <high_level_usage>`.
