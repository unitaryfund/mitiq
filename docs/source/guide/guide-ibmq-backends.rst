.. mitiq documentation file

.. _guide-ibmq-backends:

*********************************************
Error mitigation on IBMQ backends
*********************************************

This tutorial shows an example of how to mitigate noise on IBMQ backends. First we import Qiskit and mitiq.

.. code-block:: python

    >>> import qiskit
    >>> import mitiq
    >>> from mitiq.mitiq_qiskit.qiskit_utils import random_identity_circuit

For simplicity, we'll use a random single-qubit circuit with ten gates that compiles to the identity, defined below.

.. code-block:: python

    >>> depth = 10
    >>> circuit = random_identity_circuit(depth=depth)
    >>> print(circuit)
            ┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐
    q_0: |0>┤ Y ├┤ Y ├┤ X ├┤ Z ├┤ Z ├┤ Z ├┤ Z ├┤ X ├┤ X ├┤ Z ├┤ Y ├
            └───┘└───┘└───┘└───┘└───┘└───┘└───┘└───┘└───┘└───┘└───┘
     c_0: 0 ═══════════════════════════════════════════════════════

We now define factors to scale the circuit length by and fold the circuit using the ``fold_gates_at_random``
local folding method.

.. code-block:: python

    >>> scale_factors = [1., 1.5, 2., 2.5, 3.]
    >>> folded_circuits = [
            mitiq.folding.fold_local(
                circuit, scale, method=mitiq.folding.fold_gates_at_random
            ) for scale in scale_factors
        ]

We now add the observables we want to measure to the circuit. Here we use a single observable
:math:`\Pi_0 \equiv |0\rangle \langle0|` -- i.e., the probability of measuring the ground state -- but other observables
can be used.

.. code-block:: python

    >>> for folded_circuit in folded_circuits:
    >>>     folded_circuit.measure_all()


For a noiseless simulation, the expectation of this observable should be 1.0 because our circuit compiles to the identity.
For noisy simulation, the value will be smaller than one. Because folding introduces more gates and thus more noise,
the expectation value will decrease as the length (scale factor) of the folded circuits increase. By fitting this to
a curve, we can extrapolate to the zero-noise limit and obtain a better estimate.

In the code block below, we setup our connection to IBMQ backends.

.. note::
    The following code requires a valid IBMQ account. See https://quantum-computing.ibm.com/ for instructions.

.. code-block:: python

    >>> provider = qiskit.IBMQ.load_account()
    >>> print("Available backends:", *provider.backends(), sep="\n")

Depending on your IBMQ account, this print statement will display different available backend names. Shown below is an
example of executing the folded circuits using the IBMQ Armonk single qubit backend. Depending on what backends are
available, you may wish to choose a different backend by changing the ``backend_name`` below.

.. code-block:: python

    >>> shots = 8192
    >>> backend_name = "ibmq_armonk"

    >>> job = qiskit.execute(
    >>>    experiments=folded_circuits,
    >>>    backend=provider.get_backend(backend_name),
    >>>    optimization_level=0,
    >>>    shots=shots
    >>> )


.. note::
    We set the ``optimization_level=0`` to prevent any compilation by Qiskit transpilers.


Once the job has finished executing, we can convert the raw measurement statistics to observable values by running the
following code block.

.. code-block:: python

    >>> all_counts = [job.result().get_counts(i) for i in range(len(folded_circuits))]
    >>> expectation_values = [counts.get("0") / shots for counts in all_counts]

We can now see the unmitigated observable value by printing the first element of ``expectation_values``. (This value
corresponds to a circuit with scale factor one, i.e., the original circuit.)

.. code-block:: python

    >>> print("Unmitigated expectation value:", round(expectation_values[0], 3))
    Unmitigated expectation value: 0.945

Now we can using the ``reduce`` method of ``mitiq.Factory`` objects to extrapolate to the zero-noise limit. Below we use
a linear fit (order one polynomial fit) and print out the extrapolated zero-noise value.

.. code-block:: python

    >>> order = 1
    >>> zero_noise_value = mitiq.factories.PolyFactory.static_reduce(
    >>>     scale_factors[1:], expectation_values[1:], order=order
    >>> )
    >>> print(f"Extrapolated zero-noise value:", round(zero_noise_value, 3))
    Extrapolated zero-noise value: 0.961

For this example, we indeed see that the extrapolated zero-noise value (0.961) is closer to the true value (1.0) than
the unmitigated expectation value (0.945).