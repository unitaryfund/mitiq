.. mitiq documentation file

.. _guide-ibmq-backends:

*********************************************
Error mitigation on IBMQ backends
*********************************************

This tutorial shows an example of how to mitigate noise on IBMQ backends, broken down in the following steps.

* :ref:`setup`
* :ref:`high_level_usage`
* :ref:`cirq_frontend`
* :ref:`low_level_usage`

.. _setup:

Setup: Defining a circuit
#########################

First we import Qiskit and mitiq.

.. testcode:: python

    import qiskit
    import mitiq

For simplicity, we'll use a random single-qubit circuit with ten gates that compiles to the identity, defined below.

.. testcode:: python

    qreg, creg = qiskit.QuantumRegister(1), qiskit.ClassicalRegister(1)
    circuit = qiskit.QuantumCircuit(qreg, creg)
    for _ in range(10):
        circuit.x(qreg)
    circuit.measure(qreg, creg)

We will use the probability of the ground state as our observable to mitigate, the expectation value of which should
evaluate to one in the noiseless setting.

.. _high_level_usage:

High-level usage
################

To use ``mitiq`` with just a few lines of code, we simply need to define a function which inputs a circuit and outputs
the expectation value to mitigate. This function will:

1. [Optionally] Add measurement(s) to the circuit.
2. Run the circuit.
3. Convert from raw measurement statistics (or a different output format) to an expectation value.

We define this function in the following code block. Because we are using IBMQ backends, we first load our account.

.. note::
    The following code requires a valid IBMQ account. See https://quantum-computing.ibm.com/ for instructions.

.. testcode:: python

    provider = qiskit.IBMQ.load_account()

    def armonk_executor(circuit: qiskit.QuantumCircuit, shots: int = 1024) -> float:
        """Returns the expectation value to be mitigated.

        Args:
            circuit: Circuit to run.
            shots: Number of times to execute the circuit to compute the expectation value.
        """
        # Run the circuit
        job = qiskit.execute(
            experiments=circuit,
            # Change backend=provider.get_backend("ibmq_armonk") to run on hardware
            backend=provider.get_backend("ibmq_qasm_simulator"),
            optimization_level=0,  # Important!
            shots=shots
        )

        # Convert from raw measurement counts to the expectation value
        counts = job.result().get_counts()
        if counts.get("0") is None:
            expectation_value = 0.
        else:
            expectation_value = counts.get("0") / shots
        return expectation_value

At this point, the circuit can be executed to return a mitigated expectation value by running ``mitiq.execute_with_zne``,
as follows.

.. testcode:: python

    mitigated = mitiq.execute_with_zne(circuit, armonk_executor)


As long as a circuit and a function for executing the circuit are defined, the ``mitiq.execute_with_zne`` function can
be called as above to return zero-noise extrapolated expectation value(s).

.. _options:

Options
*******

Different options for noise scaling and extrapolation can be passed into the ``mitiq.execute_with_zne`` function.
By default, noise is scaled by locally folding gates at random, and the default extrapolation is Richardson.

To specify a different extrapolation technique, we can pass a different ``Factory`` object to ``execute_with_zne``. The
following code block shows an example of using linear extrapolation with five different (noise) scale factors.

.. testcode:: python

    linear_factory = mitiq.zne.inference.LinearFactory(scale_factors=[1.0, 1.5, 2.0, 2.5, 3.0])
    mitigated = mitiq.execute_with_zne(circuit, armonk_executor, factory=linear_factory)

To specify a different noise scaling method, we can pass a different function for the argument ``scale_noise``. This
function should input a circuit and scale factor and return a circuit. The following code block shows an example of
scaling noise by folding gates starting from the left (instead of at random, the default behavior for
``mitiq.execute_with_zne``).

.. testcode:: python

    mitigated = mitiq.execute_with_zne(circuit, armonk_executor, scale_noise=mitiq.zne.scaling.fold_gates_from_left)

Any different combination of noise scaling and extrapolation technique can be passed as arguments to
``mitiq.execute_with_zne``.

.. _cirq_frontend:

Cirq frontend
*************

It isn't necessary to use Qiskit frontends (circuits) to run on IBM backends. We can use conversions in
``mitiq`` to use any supported frontend with any supported backend. Below, we show how to run a Cirq circuit on an
IBMQ backend.

First, we define the Cirq circuit.

.. testcode:: python

    import cirq

    qbit = cirq.LineQubit(0)
    cirq_circuit = cirq.Circuit([cirq.X(qbit)] * 10, cirq.measure(qbit))

Now, we simply add a line to our executor function which converts from a Cirq circuit to a Qiskit circuit.

.. testcode:: python

    from mitiq.mitiq_qiskit.conversions import to_qiskit

    def cirq_armonk_executor(cirq_circuit: cirq.Circuit, shots: int = 1024) -> float:
        qiskit_circuit = to_qiskit(cirq_circuit)
        return armonk_executor(qiskit_circuit, shots)

After this, we can use ``mitiq.execute_with_zne`` in the same way as above.

.. testcode:: python

    mitigated = mitiq.execute_with_zne(cirq_circuit, cirq_armonk_executor)

As above, different noise scaling or extrapolation methods can be used.

.. _low_level_usage:

Lower-level usage
#################

Here, we give more detailed usage of the ``mitiq`` library which mimics what happens in the call to
``mitiq.execute_with_zne`` in the previous example. In addition to showing more of the ``mitiq`` library, this
example explains the code in the previous section in more detail.

First, we define factors to scale the circuit length by and fold the circuit using the ``fold_gates_at_random``
local folding method.

.. testcode:: python

    scale_factors = [1., 1.5, 2., 2.5, 3.]
    folded_circuits = [
            mitiq.zne.scaling.fold_gates_at_random(circuit, scale)
            for scale in scale_factors
    ]


For a noiseless simulation, the expectation of this observable should be 1.0 because our circuit compiles to the identity.
For noisy simulation, the value will be smaller than one. Because folding introduces more gates and thus more noise,
the expectation value will decrease as the length (scale factor) of the folded circuits increase. By fitting this to
a curve, we can extrapolate to the zero-noise limit and obtain a better estimate.

In the code block below, we setup our connection to IBMQ backends.

.. note::
    The following code requires a valid IBMQ account. See https://quantum-computing.ibm.com/ for instructions.

.. doctest:: python

    provider = qiskit.IBMQ.load_account()
    print("Available backends:", *provider.backends(), sep="\n")

Depending on your IBMQ account, this print statement will display different available backend names. Shown below is an
example of executing the folded circuits using the IBMQ Armonk single qubit backend. Depending on what backends are
available, you may wish to choose a different backend by changing the ``backend_name`` below.

.. testcode:: python

    shots = 8192
    backend_name = "ibmq_armonk"

    job = qiskit.execute(
       experiments=folded_circuits,
       # Change backend=provider.get_backend(backend_name) to run on hardware
       backend=provider.get_backend("ibmq_qasm_simulator"),
       optimization_level=0,  # Important!
       shots=shots
    )


.. note::
    We set the ``optimization_level=0`` to prevent any compilation by Qiskit transpilers.


Once the job has finished executing, we can convert the raw measurement statistics to observable values by running the
following code block.

.. testcode:: python

    all_counts = [job.result().get_counts(i) for i in range(len(folded_circuits))]
    expectation_values = [counts.get("0") / shots for counts in all_counts]

We can now see the unmitigated observable value by printing the first element of ``expectation_values``. (This value
corresponds to a circuit with scale factor one, i.e., the original circuit.)

.. code-block:: python

    >>> print("Unmitigated expectation value:", round(expectation_values[0], 3))
    Unmitigated expectation value: 0.945

Now we can use the ``reduce`` method of ``mitiq.Factory`` objects to extrapolate to the zero-noise limit. Below we use
a linear fit (order one polynomial fit) and print out the extrapolated zero-noise value.

.. code-block:: python

    >>> fac = mitiq.zne.inference.LinearFactory(scale_factors)
    >>> fac.instack, fac.outstack = scale_factors, expectation_values
    >>> zero_noise_value = fac.reduce()
    >>> print(f"Extrapolated zero-noise value:", round(zero_noise_value, 3))
    Extrapolated zero-noise value: 0.961

For this example, we indeed see that the extrapolated zero-noise value (0.961) is closer to the true value (1.0) than
the unmitigated expectation value (0.945).
