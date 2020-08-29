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
