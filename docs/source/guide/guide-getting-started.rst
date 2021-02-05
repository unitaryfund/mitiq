.. _guide-getting-started:

*********************************************
Getting Started
*********************************************

Improving the performance of your quantum programs is only a few lines of
code away.

This getting started shows examples using
`Cirq <https://cirq.readthedocs.io/en/stable/index.html>`_ and
`Qiskit <https://qiskit.org/>`_. We'll first test Mitiq by running
against the noisy simulator built into Cirq. The Qiskit example works
similarly as you will see in :ref:`Zero-Noise Extrapolation with Qiskit <qiskit_getting_started>`.


.. _multi_platform_framework:

Multi-platform Framework
------------------------

In Mitiq, a "back-end" is a function that executes quantum programs. A
"front-end" is a library/language that constructs quantum programs. Mitiq
lets you mix and match these. For example, you could write a quantum program in
Qiskit and then execute it using a Cirq backend, or vice versa.

Back-ends are abstracted to user-defined functions called *executors* that
always accept a quantum program, sometimes accept other arguments, and always
return an expectation value as a float. You can see some examples of different
executors for common packages :ref:`here <guide-executors>` and in this
getting started. If your quantum programming interface of choice can be used
to make a Python function with this type, then it can be used with Mitiq.

Let us define a simple ``executor`` function which simulates a Cirq circuit
with depolarizing noise and returns the expectation value of
:math:`|00...\rangle \langle00...|`.

.. testcode::

    import numpy as np
    from cirq import Circuit, depolarize, DensityMatrixSimulator

    SIMULATOR = DensityMatrixSimulator()
    # 1% depolarizing noise
    NOISE_LEVEL = 0.01

    def executor(circ: Circuit) -> float:
        """ Simulates a circuit with depolarizing noise at level NOISE_LEVEL.
        Args:
            circ: The quantum program as a Cirq object.

        Returns:
            The expectation value of the ground state projector.
        """
        circuit = circ.with_noise(depolarize(p=NOISE_LEVEL))
        rho = SIMULATOR.simulate(circuit).final_density_matrix
        return np.real(rho[0,0])

Now we consider a simple example: a single-qubit circuit with an even
number of X gates. By construction, the ideal expectation value should be
1, but the noisy expectation value will be slightly different.

.. testcode::

    from cirq import Circuit, LineQubit, X

    qubit = LineQubit(0)
    circuit = Circuit(X(qubit) for _ in range(6))
    noisy_result = executor(circuit)
    exact_result = 1
    print(f"Error in noisy simulation: {abs(exact_result - noisy_result):.{3}}")

.. testoutput::

    Error in noisy simulation: 0.0387

This shows the impact of noise on the final expectation value (without error mitigation).
Now let's use Mitiq to improve this performance.

Error Mitigation with Zero-Noise Extrapolation
----------------------------------------------

Zero-noise extrapolation can be easily implemented by importing the function
:func:`~mitiq.zne.zne.execute_with_zne` from the :mod:`~mitiq.zne.zne` module.

.. testcode::

    from mitiq import execute_with_zne

    mitigated_result = execute_with_zne(circuit, executor)
    
    print(f"Error without mitigation: {abs(exact_result - noisy_result):.{3}}")
    print(f"Error with mitigation (ZNE): {abs(exact_result - mitigated_result):.{3}}")


.. testoutput::

    Error without mitigation: 0.0387
    Error with mitigation (ZNE): 0.000232

You can also use Mitiq to wrap your backend execution function into an
error-mitigated version.

.. testcode::

    from mitiq import mitigate_executor

    run_mitigated = mitigate_executor(executor)
    mitigated_result = run_mitigated(circuit)
    print(round(mitigated_result, 5))

.. testoutput::

    0.99977

.. _partial-note:

.. note::
   As shown here, Mitiq wraps executor functions that have a specific type:
   they take quantum programs as input and return expectation values. However,
   one often has an execution function with other arguments such as the number of
   shots, the observable to measure, or the noise level of a noisy simulation.
   It is still easy to use these with Mitiq by using partial function application.
   Here's a pseudo-code example:

   .. code-block::

      from functools import partial

      def shot_executor(qprogram, n_shots) -> float:
          ...
      # we partially apply the n_shots argument to get a function that just
      # takes a quantum program
      mitigated = execute_with_zne(circ, partial(shot_executor, n_shots=100))

   You can read more about ``functools`` partial application
   `here <https://docs.python.org/3/library/functools.html#functools.partial>`_.


The default implementation uses Richardson extrapolation to extrapolate the
expectation value to the zero noise limit :cite:`Temme_2017_PRL`. Mitiq
comes equipped with other extrapolation methods as well. Different methods of
extrapolation are packaged into :class:`~mitiq.zne.inference.Factory` objects.
It is easy to try different ones.

.. testcode::

    from mitiq import execute_with_zne
    from mitiq.zne.inference import LinearFactory

    fac = LinearFactory(scale_factors=[1.0, 2.0, 2.5])
    linear_zne_result = execute_with_zne(circuit, executor, factory=fac)
    abs_error = abs(exact_result - linear_zne_result)
    print(f"Mitigated error with linear ZNE: {abs_error:.{3}}")

.. testoutput::

    Mitigated error with linear ZNE: 0.00769

You can use bult-in methods from factories like :meth:`~mitiq.zne.inference.Factory.plot_data`
and :meth:`~mitiq.zne.inference.Factory.plot_fit` to plot the noise scale factors v. the expectation
value returned by the executor.

.. testcode::

   fac.plot_fit()

.. image:: ../img/factory-plot_fit.png
    :width: 600
    :alt: factory data from executor.

You can read more about the :class:`~mitiq.zne.inference.Factory` objects that are built into Mitiq
and how to create your own :ref:`here <guide-factories>`.

Another key step in zero-noise extrapolation is to choose how your circuit is
transformed to scale the noise. You can read more about the noise scaling
methods built into Mitiq and how to create your
own :ref:`here <guide-folding>`.

.. _qiskit_getting_started:

Zero-Noise Extrapolation with Qiskit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Mitiq is designed to be agnostic to the stack that you are using. Thus for
Qiskit things work in the same manner as before. Since we are now using Qiskit,
we want to run the error mitigated programs on a Qiskit backend. Let's define
the new backend that accepts Qiskit circuits. In this case it is a simulator,
but you could also use a QPU.

.. testcode::

    import qiskit
    from qiskit import QuantumCircuit

    # Noise simulation packages
    from qiskit.providers.aer.noise import NoiseModel
    from qiskit.providers.aer.noise.errors.standard_errors import depolarizing_error

    # 0.1% depolarizing noise
    QISKIT_NOISE = 0.001

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
        noise_model.add_all_qubit_quantum_error(
            depolarizing_error(QISKIT_NOISE, 1),
            ["u1", "u2", "u3"],
        )

        # execution of the experiment
        job = qiskit.execute(
            circuit,
            backend=QISKIT_SIMULATOR,
            basis_gates=["u1", "u2", "u3"],
            # we want all gates to be actually applied,
            # so we skip any circuit optimization
            optimization_level=0,
            noise_model=noise_model,
            shots=shots,
            seed_transpiler=1,
            seed_simulator=1
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
    for _ in range(100):
         _ = circ.x(0)
    _ = circ.measure(0, 0)

    exact = 1
    unmitigated = qs_noisy_simulation(circ)
    mitigated = execute_with_zne(circ, qs_noisy_simulation)

    # The mitigation should improve the result.
    assert abs(exact - mitigated) < abs(exact - unmitigated)

Note that we don't need to even redefine factories for different stacks. Once
you have a :class:`~mitiq.zne.inference.Factory` it can be used with different front and backends.

Error Mitigation with Probabilistic Error Cancellation
------------------------------------------------------

In *Mitiq*, it is very easy to switch between different error mitigation methods.

For example, we can implement Probabilistic Error Cancellation (PEC) by using the same execution function (``executor``)
and the same Cirq circuit (``circuit``) that we have already defined in the section
:ref:`Multi-platform Framework <multi_platform_framework>`.

PEC requires a good knowledge of the noise model and of the noise strength acting on the system.
In particular for each operation of the circuit, we need to build a quasi-probability representation of the 
ideal unitary gate expanded in a basis of noisy implementable operations. For more details behind the theory of PEC see
the :ref:`Probabilistic Error Cancellation <guide_qem_pec>` section.

In our simple case, ``circuit`` corresponds to the repetition of the same X gate,
whose representation in the presence of depolarizing noise can be obtained as follows:

.. testcode::

    from mitiq.pec.representations import represent_operation_with_local_depolarizing_noise

    x_representation = represent_operation_with_local_depolarizing_noise(
        ideal_operation=Circuit(X(qubit)), 
        noise_level=NOISE_LEVEL,
    )

    print(x_representation)

.. testoutput::

    0: ───X─── = 1.010*0: ───X───-0.003*0: ───X───X───-0.003*0: ───X───Y───-0.003*0: ───X───Z───

The result above is an :class:`~mitiq.pec.types.types.OperationRepresentation` object which contains
the information for representing the ideal operation X (left-hand-side of the printed output)
as a linear combination of noisy operations (right-hand-side of the printed output). 

We can now implement PEC by importing the function :func:`~mitiq.pec.pec.execute_with_pec` from the 
:mod:`~mitiq.pec.pec` module.

.. testcode::

    from mitiq.pec import execute_with_pec

    SEED = 0
    exact_result = 1
    noisy_result = executor(circuit)
    pec_result = execute_with_pec(
        circuit,
        executor,
        representations=[x_representation],
        random_state=SEED,
    )

    print(f"Error without mitigation: {abs(exact_result - noisy_result):.{3}}")
    print(f"Error with mitigation (PEC): {abs(exact_result - pec_result):.{3}}")

.. testoutput::

    Error without mitigation: 0.0387
    Error with mitigation (PEC): 0.00364
