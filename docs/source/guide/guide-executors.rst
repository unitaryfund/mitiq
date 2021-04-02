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
        final_wvf = circ.final_state_vector()
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
See  :py:mod:`mitiq.mitiq_qiskit.qiskit_utils`.

This section includes noisy and noiseless simulator executor examples you can use on Qiskit circuits.

Qiskit: Wavefunction Simulation
---------------------------------
See :py:func:`mitiq.mitiq_qiskit.qiskit_utils.execute`.

This executor can be used for noiseless simulation. Note that this executor
can be :ref:`wrapped using partial function application <partial-note>`
to be used in Mitiq.

Qiskit: Wavefunction Simulation with Sampling
-----------------------------------------------
See :py:func:`mitiq.mitiq_qiskit.qiskit_utils.execute_with_shots`.

The noiseless simulation executor can be modified to still perform exact wavefunction
simulation, but to also include finite sampling of measurements. Note that this executor
can be :ref:`wrapped using partial function application <partial-note>`
to be used in Mitiq.

Note that this executor implementation measures arbitrary observables by using
a change of basis into the computational basis. More information behind the math
in this example can be found `here <https://quantumcomputing.stackexchange.com/a/6944>`__.

Qiskit: Density-matrix Simulation with Noise
-----------------------------------------------------------
See :py:func:`mitiq.mitiq_qiskit.qiskit_utils.execute_with_noise`.

This executor can be used to simulate a circuit with noise and to return the exact expectation
value of an observable (without the shot noise typical of a real experiment).

See :py:func:`mitiq.mitiq_qiskit.qiskit_utils.initialized_depolarizing_noise` for an example depolarizing noise
model you can use.

Qiskit: Density-matrix Simulation with Noise and Sampling
------------------------------------------------------------------------
See :py:func:`mitiq.mitiq_qiskit.qiskit_utils.execute_with_shots_and_noise`.

This executor can be used to simulate a circuit with noise. The expectation value is estimated
with a finite number of measurements and so it is affected by statistical noise.

Noise models can be defined using any functionality available in Qiskit.
More details can be found in the
`Qiskit simulator documentation <https://qiskit.org/documentation/tutorials/simulators/index.html>`__.

Qiskit: Hardware
------------------------------------------------------------

An example of an executor that runs on IBMQ hardware is given
:ref:`here <high_level_usage>`.
