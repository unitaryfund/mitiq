.. _guide-getting-started:

*********************************************
Getting Started
*********************************************

Improving the performance of your quantum programs is only a few lines of
code away.

This getting started shows examples using cirq
`cirq <https://cirq.readthedocs.io/en/stable/index.html>`_ and
`qiskit <https://qiskit.org/>`_. We'll first test ``mitiq`` by running
against the noisy simulator built into ``cirq``. The qiskit example works
similarly as you will see in :ref:`Qiskit Mitigation <qiskit_getting_started>`.


Multi-platform Framework
----------------------------------------------
In ``mitiq``, a "back-end" is a function that executes quantum programs. A
"front-end" is a library/language that constructs quantum programs. ``mitiq``
lets you mix and match these. For example, you could write a quantum program in
``qiskit`` and then execute it using a ``cirq`` backend, or vice versa.

Back-ends are abstracted to functions called ``executors`` that always accept
a quantum program, sometimes accept other arguments, and always
return an expectation value as a float. You can see some examples of different
executors for common packages :ref:`here <guide-executors>` and in this
getting started. If your quantum programming interface of choice can be used
to make a Python function with this type, then it can be used with mitiq.


Error Mitigation with Zero-Noise Extrapolation
----------------------------------------------

We define some functions that make it simpler to simulate noise in
``cirq``. These don't have to do with ``mitiq`` directly.

.. testcode::

    import numpy as np
    from cirq import Circuit, depolarize
    from cirq import LineQubit, X, DensityMatrixSimulator

    SIMULATOR = DensityMatrixSimulator()
    # 0.1% depolarizing noise
    NOISE = 0.001

    def noisy_simulation(circ: Circuit) -> float:
        """ Simulates a circuit with depolarizing noise at level NOISE.
        Args:
            circ: The quantum program as a cirq object.

        Returns:
            The expectation value of the |0> state.
        """
        circuit = circ.with_noise(depolarize(p=NOISE))
        rho = SIMULATOR.simulate(circuit).final_density_matrix
        # define the computational basis observable
        obs = np.diag([1, 0])
        expectation = np.real(np.trace(rho @ obs))
        return expectation

Now we can look at our example. We'll test single qubit circuits with even
numbers of X gates. As there are an even number of X gates, they should all
evaluate to an expectation of 1 in the computational basis if there was no
noise.

.. testcode::

    from cirq import Circuit, LineQubit, X

    qbit = LineQubit(0)
    circ = Circuit(X(qbit) for _ in range(80))
    unmitigated = noisy_simulation(circ)
    exact = 1
    print(f"Error in simulation is {exact - unmitigated:.{3}}")

.. testoutput::

    Error in simulation is 0.0506

This shows the impact the noise has had. Let's use ``mitiq`` to improve this
performance.

.. testcode::

    from mitiq import execute_with_zne

    mitigated = execute_with_zne(circ, noisy_simulation)
    print(f"Error in simulation is {exact - mitigated:.{3}}")

.. testoutput::

    Error in simulation is 0.000519

.. testcode::

    print(f"Mitigation provides a {(exact - unmitigated) / (exact - mitigated):.{3}} factor of improvement.")

.. testoutput::

    Mitigation provides a 97.6 factor of improvement.

You can also use ``mitiq`` to wrap your backend execution function into an
error-mitigated version.

.. testcode::

    from mitiq import mitigate_executor

    run_mitigated = mitigate_executor(noisy_simulation)
    mitigated = run_mitigated(circ)
    print(round(mitigated,5))

.. testoutput::

    0.99948

.. _partial-note:

.. note::
   As shown here, ``mitiq`` wraps executor functions that have a specific type:
   they take quantum programs as input and return expectation values. However,
   one often has an execution function with other arguments such as the number of
   shots, the observable to measure, or the noise level of a noisy simulation.
   It is still easy to use these with mitiq by using partial function application.
   Here's a pseudo-code example:

   .. code-block::

      from functools import partial

      def shot_executor(qprogram, n_shots) -> float:
          ...
      # we partially apply the n_shots argument to get a function that just
      # takes a quantum program
      mitigated = execute_with_zne(circ, partial(shot_executor, n_shots=100))

   You can read more about functools partial application
   `here <https://docs.python.org/3/library/functools.html#functools.partial>`_.


The default implementation uses Richardson extrapolation to extrapolate the
expectation value to the zero noise limit :cite:`Temme_2017_PRL`. ``Mitiq``
comes equipped with other extrapolation methods as well. Different methods of
extrapolation are packaged into ``Factory`` objects. It is easy to try
different ones.

.. testcode::

    from mitiq import execute_with_zne
    from mitiq.zne.inference import LinearFactory

    fac = LinearFactory(scale_factors=[1.0, 2.0, 2.5])
    linear = execute_with_zne(circ, noisy_simulation, factory=fac)
    print(f"Mitigated error with the linear method is {exact - linear:.{3}}")

.. testoutput::

    Mitigated error with the linear method is 0.00638

You can read more about the ``Factory`` objects that are built into ``mitiq``
and how to create your own :ref:`here <guide-factories>`.

Another key step in zero-noise extrapolation is to choose how your circuit is
transformed to scale the noise. You can read more about the noise scaling
methods built into ``mitiq`` and how to create your
own :ref:`here <guide-folding>`.

.. _qiskit_getting_started:

Qiskit Mitigation
--------------------------

``Mitiq`` is designed to be agnostic to the stack that you are using. Thus for
``qiskit`` things work in the same manner as before. Since we are now using ``qiskit``,
we want to run the error mitigated programs on a qiskit backend. Let's define
the new backend that accepts ``qiskit`` circuits. In this case it is a simulator,
but you could also use a QPU.

.. testcode::

    import qiskit
    from qiskit import QuantumCircuit

    # Noise simulation packages
    from qiskit.providers.aer.noise import NoiseModel
    from qiskit.providers.aer.noise.errors.standard_errors import depolarizing_error

    # 0.1% depolarizing noise
    NOISE = 0.001

    QISKIT_SIMULATOR = qiskit.Aer.get_backend("qasm_simulator")

    def qs_noisy_simulation(circuit: QuantumCircuit, shots: int = 4096) -> float:
        """Runs the quantum circuit with a depolarizing channel noise model at
        level NOISE.

        Args:
            circuit (qiskit.QuantumCircuit): Ideal quantum circuit.
            shots (int): Number of shots to run the circuit
                         on the back-end.

        Returns:
            expval: expected values.
        """
        # initialize a qiskit noise model
        noise_model = NoiseModel()

        # we assume a depolarizing error for each
        # gate of the standard IBM basis
        noise_model.add_all_qubit_quantum_error(depolarizing_error(NOISE, 1), ["u1", "u2", "u3"])

        # execution of the experiment
        job = qiskit.execute(
            circuit,
            backend=QISKIT_SIMULATOR,
            basis_gates=["u1", "u2", "u3"],
            # we want all gates to be actually applied,
            # so we skip any circuit optimization
            optimization_level=0,
            noise_model=noise_model,
            shots=shots
        )
        results = job.result()
        counts = results.get_counts()
        expval = counts["0"] / shots
        return expval

We can then use this backend for our mitigation.

.. testcode::

    from qiskit import QuantumCircuit
    from mitiq import execute_with_zne

    circ = QuantumCircuit(1, 1)
    for __ in range(120):
         _ = circ.x(0)
    _ = circ.measure(0, 0)

    unmitigated = qs_noisy_simulation(circ)
    mitigated = execute_with_zne(circ, qs_noisy_simulation)
    exact = 1
    # The mitigation should improve the result.
    print(abs(exact - mitigated) < abs(exact - unmitigated))

.. testoutput::

    True

Note that we don't need to even redefine factories for different stacks. Once
you have a ``Factory`` it can be used with different front and backends.
