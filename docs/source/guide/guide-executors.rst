.. _guide-executors:

*********************************************
Back-end Plug-ins: Executor Examples
*********************************************

``Mitiq`` uses ``executor`` functions to abstract different backends.
``Executors`` always accept a quantum program, sometimes accept other
arguments, and always return an expectation value as a float. If your
quantum programming interface of choice can be used
to make a Python function with this type, then it can be used with mitiq.

These example executors as especially flexible as they
accept an arbitrary observable. You can instead hardcode your choice of
observable in any way you like. All that matters from ``mitiq``'s perspective
is that your executor accepts a quantum program and returns a float.


Cirq Executors
======================================

This section includes noisy and noiseless simulator executor examples using
``cirq``.

Cirq: Wavefunction Simulation
---------------------------------
This executor can be used for noiseless simulation. Note that this executor
can be :ref:`wrapped using partial function application <partial-note>`
to be used in ``mitiq``.

.. testcode::

    import numpy as np
    from cirq import Circuit

    def wvf_sim(circ: Circuit, obs: np.ndarray) -> float:
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

    print(wvf_sim(qc, obs=np.diag([1, 0, 0, 0])))

.. testoutput::
   :hide:

   0.0

.. testcode::
    :hide:

    print(wvf_sim(qc, obs=np.diag([0, 0, 0, 1])))

.. testoutput::
    :hide:

    1.0

Cirq: Wavefunction Simulation with Sampling
-----------------------------------------------
We can add in functionality that takes into account some finite number of
samples (aka shots). Here we will use ``cirq``'s `PauliString` methods to
construct our observable. You can read more about these methods in the ``cirq``
documentation `here <https://cirq.readthedocs.io/en/master/generated/cirq.PauliString.html?highlight=paulistring>`_.

.. testcode::

    def wvf_sampling_sim(circ: Circuit, obs: cirq.PauliString, shots: int) -> float:
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

    assert np.isclose(wvf_sampling_sim(qc, ham, 10000), 1.0)


Cirq: Density-matrix Simulation with Depolarizing Noise
------------------------------------------------------------
This executor can be used for noisy depolarizing simulation.

.. testcode::

    import numpy as np
    from cirq import Circuit, depolarize
    from cirq import DensityMatrixSimulator

    SIMULATOR = DensityMatrixSimulator()

    def noisy_sim(circ: Circuit, obs: np.ndarray, noise: float) -> float:
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

    assert noisy_sim(qc, np.diag([0, 1]), 0.0) == 0.0
    assert np.isclose(noisy_sim(qc, np.diag([0, 1]), 0.5), 0.5)
    assert np.isclose(noisy_sim(qc, np.diag([0, 1]), 0.001), 0.062452)

Other noise models can be used by substituting the ``depolarize`` channel with
any other channel available in ``cirq``, for example ``cirq.amplitude_damp``.
More details can be found in the ``cirq``
`noise documentation <https://cirq.readthedocs.io/en/stable/noise.html>`_

Cirq: Density-matrix Simulation with Depolarizing Noise and Sampling
------------------------------------------------------------------------
You can also include both noise models and finite sampling in your executor.

.. testcode::

    import numpy as np
    from cirq import Circuit, depolarize
    from cirq import DensityMatrixSimulator

    SIMULATOR = DensityMatrixSimulator()

    def noisy_sample_sim(circ: Circuit, obs: cirq.PauliString, noise: float, shots: int) -> float:
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
    noisy_output = noisy_sample_sim(qc, ham, 0.01, 200)
    assert noisy_output < 1.0
    assert noisy_output > 0.5

.. _pyquil_executors:

PyQuil Executors
================

This section contains executors for use with `pyQuil <https://github.com/rigetti/pyquil>`_.

PyQuil: Quantum Cloud Services
------------------------------

This executor can be used to run on `Quantum Cloud Services <https://arxiv.org/abs/2001.04449>`_
(QCS), the hardware platform provided by Rigetti Computing. Requires a QCS account and
reservation on a quantum processor (QPU).

In addition, ``mitiq_pyquil/executors.py`` has a function ``generate_qcs_executor`` for
easily generating a QCS executor of this form from a template.

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

    def executor(program: Program, shots: int = 1000) -> float:
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
    executor(program)


.. _qiskit_executors:

Qiskit Executors
======================================

This section includes noisy and noiseless simulator executor examples using
``qiskit``.


Qiskit: Wavefunction Simulation
---------------------------------
This executor can be used for noiseless simulation. Note that this executor
can be :ref:`wrapped using partial function application <partial-note>`
to be used in ``mitiq``.

.. testcode::

    import numpy as np
    import qiskit
    from qiskit import QuantumCircuit

    wvf_simulator = qiskit.Aer.get_backend('statevector_simulator')

    def qs_wvf_sim(circ: QuantumCircuit, obs: np.ndarray) -> float:
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
    assert np.isclose(qs_wvf_sim(qc, obs=np.diag([1, 0, 0, 0])), 0.0)
    assert np.isclose(qs_wvf_sim(qc, obs=np.diag([0, 0, 0, 1])), 1.0)


Qiskit: Wavefunction Simulation with Sampling
-----------------------------------------------
The above executor can be modified to still perform exact wavefunction simulation,
but to also include finite sampling of measurements. Note that this executor
can be :ref:`wrapped using partial function application <partial-note>`
to be used in ``mitiq``.

Note that this executor implementation measures arbitrary observables by using
a change of basis into the computational basis. More information about the math
behind how this example is available `here <https://quantumcomputing.stackexchange.com/a/6944>`_.

.. testcode::

    import copy

    QISKIT_SIMULATOR = qiskit.Aer.get_backend("qasm_simulator")

    def qs_wvf_sampling_sim(circ: QuantumCircuit, obs: np.ndarray, shots: int) -> float:
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
        circ.unitary(np.linalg.inv(U), qubits=range(circ.n_qubits))

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
    out = qs_wvf_sampling_sim(qc, obs=np.diag([0, 0, 0, 1]), shots=50)
    assert 0.0 < out < 1.0
    out = qs_wvf_sampling_sim(qc, obs=np.diag([0, 0, 0, 1]), shots=int(1e5))
    assert abs(out - 0.5) < 0.1

    qc_zero = QuantumCircuit(1)
    out = qs_wvf_sampling_sim(qc_zero, obs=np.diag([1, 0]), shots=50)
    assert np.isclose(out, 1.0)


Qiskit: Density-matrix Simulation with Depolarizing Noise
-----------------------------------------------------------
    TODO

Qiskit: Density-matrix Simulation with Depolarizing Noise and Sampling
------------------------------------------------------------------------
This executor can be used for noisy depolarizing simulation.

.. testcode::

    import qiskit
    from qiskit import QuantumCircuit
    import numpy as np
    import copy

    # Noise simulation packages
    from qiskit.providers.aer.noise import NoiseModel
    from qiskit.providers.aer.noise.errors.standard_errors import depolarizing_error

    QISKIT_SIMULATOR = qiskit.Aer.get_backend("qasm_simulator")

    def qs_noisy_sampling_sim(circ: QuantumCircuit, obs: np.ndarray, noise: float, shots: int) -> float:
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
        circ.unitary(np.linalg.inv(U), qubits=range(circ.n_qubits))

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
    assert 0.1 < qs_noisy_sampling_sim(qc, np.diag([1, 0]), 0.02, 1000) < 1.0

    qc_zero = QuantumCircuit(2)
    out = qs_noisy_sampling_sim(qc_zero, obs=np.diag([1, 0, 0, 0]), noise=0.0, shots=10)
    assert np.isclose(out, 1.0)
    out = qs_noisy_sampling_sim(qc_zero, obs=np.diag([1, 0, 0, 0]), noise=1.0, shots=10 ** 5)
    assert np.isclose(out, 0.25, atol=0.1)

Other noise models can be defined using any functionality available in ``qiskit``.
More details can be found in the ``qiskit``
`simulator documentation <https://qiskit.org/documentation/tutorials/simulators/index.html>`_

Qiskit: Hardware
------------------------------------------------------------
An example of an executor that runs on IBMQ hardware is given
:ref:`here <high_level_usage>`.

.. _tfq_executors:

TensorFlow Quantum Executors
==========================================

This section provides an example of how to use `TensorFlow Quantum <https://github.com/tensorflow/quantum>`_
as an executor. Note that at the time of this writing, TensorFlow Quantum is limited to

  1. ``Cirq`` ``Circuits`` that use  ``Cirq`` ``GridQubit`` instances
  2. Unitary circuits only, so non-unitary errors need to use Monte Carlo simulations

Despite this latter limitation, there is a crossover point where Monte Carlo using
Tensorflow evaluates faster than the exact density matrix simulation using ``Cirq``.

Below is an example to use TensorFlow Quantum to simulate a bit-flip channel:

.. code-block::

    import numpy as np
    import sympy
    # tensorflow-quantum 0.4.0 is unavailable on Windows
    try:
        import tensorflow as tf
        import tensorflow_quantum as tfq
        tfq_exists = True
    except ImportError:
        tfq_exists = False
    from cirq import Circuit


    def stochastic_bit_flip_simulation(circ: Circuit, p: float, num_monte_carlo: int = 100) -> float:
        """
        Simulates a circuit with random bit flip (X(\pi)) errors
        Args:
            circ: The quantum program as a cirq object
            p: probability of an X(\pi) gate on each qubit after each gate in circ
            num_monte_carlo: number of random trajectories to average over
        Returns:
            The expectation value of the 0 state as a float
        """
        nM = len(circ.moments)
        nQ = len(circ.all_qubits())

        # Create array of symbolic variables and reshape to natural circuit parameterization
        h = sympy.symbols(''.join(['h_{0} '.format(i) for i in range(nM * nQ)]), positive=True)
        h_array = np.asarray(h).reshape((nQ, nM))

        # Symbolicly add X gates to the input circuit
        noisy_circuit = Circuit()
        for i, moment in enumerate(circ.moments):
            noisy_circuit.append(moment)
            for j, q in enumerate(circ.all_qubits()):
                noisy_circuit.append(cirq.rx(h_array[j, i]).on(q))

        # rotations will be pi w/ prob p, 0 w/ prob 1-p
        vals = [np.reshape((np.random.rand(nQ, nM) < p) * np.pi, (1, nQ * nM)) for _ in range(num_monte_carlo)]

        # needs to be a rank 2 tensor
        vals = np.squeeze(vals)
        if num_monte_carlo == 1:
            vals = [vals]

        # Instantiate tfq layer for computing state vector
        state = tfq.layers.State()

        # Execute monte carlo sim with symbolic values specified by vals
        out = state(noisy_circuit, symbol_names=h, symbol_values=vals).to_tensor()

        # Fancy way of computing and summing individual density operators, follwed by averaging
        dm = tf.tensordot(tf.transpose(out), tf.math.conj(out), axes=[[1], [0]]).numpy() / num_monte_carlo

        # return measurement of 0 state
        return np.real(dm[0, 0])

.. code-block::
    :hide:

    if tfq_exists:
        import cirq
        from mitiq.benchmarks import randomized_benchmarking

        circ = randomized_benchmarking.rb_circuits(1, [20], 1)[0]

        # Need to make sure the qubits are cirq.GridQubit
        circ=circ.transform_qubits(lambda q: cirq.GridQubit.rect(1, 1)[0])

        out = stochastic_bit_flip_simulation(circ, 0.001)
        assert 0.5 < out < 1
