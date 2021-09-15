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

See :mod:`mitiq_cirq.cirq_utils` for several Cirq executor functions.


.. _pyquil_executors:

PyQuil Executors
================

This section contains executors for use with `pyQuil <https://github.com/rigetti/pyquil>`__.

PyQuil: Quantum Cloud Services
------------------------------

This executor can be used to run on `Quantum Cloud Services <https://arxiv.org/abs/2001.04449>`__
(QCS), the hardware platform provided by Rigetti Computing, and requires a QCS account and
reservation on a quantum processor (QPU).

Note that you will have to replace the string in ``get_qc`` with the name of an actual
Rigetti QPU, and will need to have a QCS account and reservation, in order to run on
real quantum hardware.

.. testcode::

    import numpy as np

    from pyquil import Program, get_qc
    from pyquil.gates import MEASURE, RESET, X

    from mitiq.interface.mitiq_pyquil.compiler import basic_compile

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
        return (
            shots - np.count_nonzero(np.count_nonzero(results, axis=1))
        ) / shots

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
See  :py:mod:`~mitiq.interface.mitiq_qiskit.qiskit_utils`.

This section includes noisy and noiseless simulator executor examples you can use on Qiskit circuits.

Qiskit: Wavefunction Simulation
---------------------------------
See :py:func:`~mitiq.interface.mitiq_qiskit.qiskit_utils.execute`.

This executor can be used for noiseless simulation. Note that this executor
can be :ref:`wrapped using partial function application <partial-note>`
to be used in Mitiq.

Qiskit: Wavefunction Simulation with Sampling
-----------------------------------------------
See :py:func:`~mitiq.interface.mitiq_qiskit.qiskit_utils.execute_with_shots`.

The noiseless simulation executor can be modified to still perform exact wavefunction
simulation, but to also include finite sampling of measurements. Note that this executor
can be :ref:`wrapped using partial function application <partial-note>`
to be used in Mitiq.

Note that this executor implementation measures arbitrary observables by using
a change of basis into the computational basis. More information behind the math
in this example can be found `here <https://quantumcomputing.stackexchange.com/a/6944>`__.

Qiskit: Density-matrix Simulation with Noise
-----------------------------------------------------------
See :py:func:`~mitiq.interface.mitiq_qiskit.qiskit_utils.execute_with_noise`.

This executor can be used to simulate a circuit with noise and to return the exact expectation
value of an observable (without the shot noise typical of a real experiment).

See :py:func:`~mitiq.interface.mitiq_qiskit.qiskit_utils.initialized_depolarizing_noise` for an example depolarizing noise
model you can use.

Qiskit: Density-matrix Simulation with Noise and Sampling
------------------------------------------------------------------------
See :py:func:`~mitiq.interface.mitiq_qiskit.qiskit_utils.execute_with_shots_and_noise`.

This executor can be used to simulate a circuit with noise. The expectation value is estimated
with a finite number of measurements and so it is affected by statistical noise.

Noise models can be defined using any functionality available in Qiskit.
More details can be found in the
`Qiskit simulator documentation <https://qiskit.org/documentation/tutorials/simulators/index.html>`__.

Qiskit: Hardware
------------------------------------------------------------

An example of an executor that runs on IBMQ hardware is given in the examples.
