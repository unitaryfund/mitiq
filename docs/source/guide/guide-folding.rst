.. mitiq documentation file

*********************************************
Unitary Folding
*********************************************

At the gate level, noise is amplified by mapping gates (or groups of gates) `G` to

.. math::
  G \mapsto G G^\dagger G .

This makes the circuit longer (adding more noise) while keeping its effect unchanged (because
:math:`G^\dagger = G^{-1}` for unitary gates).  We refer to this process as
*unitary folding*. If `G` is a subset of the gates in a circuit, we call it `local folding`.
If `G` is the entire circuit, we call it `global folding`.

In ``mitiq``, folding functions input a circuit and a *stretch* (or *stretch factor*), i.e., a floating point value
which corresponds to (approximately) how much the length of the circuit is scaled.
The minimum stretch is one (which corresponds to folding no gates), and the maximum stretch is three
(which corresponds to folding all gates).

=============================================
Local folding methods
=============================================

For local folding, there is a degree of freedom for which gates to fold first. As such,
``mititq`` defines several local folding methods.

We introduce three folding functions:

    1. ``mitiq.folding.fold_gates_from_left``
    2. ``mitiq.folding.fold_gates_from_right``
    3. ``mitiq.folding.fold_gates_at_random``

The ``mitiq`` function ``fold_gates_from_left`` will fold gates from the left (or start) of the circuit
until the desired stretch factor is reached.


.. doctest:: python

    >>> import cirq
    >>> from mitiq.folding import fold_gates_from_left

    # Get a circuit to fold
    >>> qreg = cirq.LineQubit.range(2)
    >>> circ = cirq.Circuit(cirq.ops.H.on(qreg[0]), cirq.ops.CNOT.on(qreg[0], qreg[1]))
    >>> print("Original circuit:", circ, sep="\n")
    Original circuit:
    0: ───H───@───
              │
    1: ───────X───

    # Fold the circuit
    >>> folded = fold_gates_from_left(circ, stretch=2.)
    >>> print("Folded circuit:", folded, sep="\n")
    Folded circuit:
    0: ───H───H───H───@───
                      │
    1: ───────────────X───

In this example, we see that the folded circuit has the first (Hadamard) gate folded.

.. note::
    ``mitiq`` folding functions do not modify the input circuit.

Because input circuits are not modified, we can reuse this circuit for the next example. In the following code,
we use the ``fold_gates_from_right`` function on the same input circuit.

.. doctest:: python

    >>> from mitiq.folding import fold_gates_from_right

    # Fold the circuit
    >>> folded = fold_gates_from_right(circ, stretch=2.)
    >>> print("Folded circuit:", folded, sep="\n")
    Folded circuit:
    0: ───H───@───@───@───
              │   │   │
    1: ───────X───X───X───

We see the second (CNOT) gate in the circuit is folded, as expected when we start folding from the right (or end) of
the circuit instead of the left (start).

Finally, we mention ``fold_gates_at_random`` which folds gates according to the following rules.

    1. Gates are selected at random and folded until the input stretch factor is reached.
    2. No gate is folded more than once.
    3. "Virtual gates" (i.e., gates appearing from folding) are never folded.

=============================================
Any supported circuits can be folded
=============================================

Any program types supported by ``mitiq`` can be folded. The interface for all folding functions is the same. In the
following example, we fold a Qiskit circuit.

.. note::
    This example assumes you have Qiskit installed. ``mitiq`` can interface with Qiskit, but Qiskit is not
    a core ``mitiq`` requirement and is not installed by default.

.. doctest:: python

    >>> import qiskit
    >>> from mitiq.folding import fold_gates_from_left

    # Get a circuit to fold
    >>> qreg = qiskit.QuantumRegister(2)
    >>> circ = qiskit.QuantumCircuit(qreg)
    >>> _ = circ.h(qreg[0])
    >>> _ = circ.cnot(qreg[0], qreg[1])
    >>> # print("Original circuit:", circ, sep="\n")

This code (when the print statement is uncommented) should display something like:

.. code-block:: python

    Original circuit:
             ┌───┐
    q0_0: |0>┤ H ├──■──
             └───┘┌─┴─┐
    q0_1: |0>─────┤ X ├
                  └───┘

We can now fold this circuit as follows.

    # Fold the circuit. Specify keep_input_type=True to return a Qiskit circuit.
    >>> folded = fold_gates_from_left(circ, stretch=2., keep_input_type=True)
    >>> # print("Folded circuit:", folded, sep="\n")

This code (when the print statement is uncommented) should display something like:

.. code-block:: python

    Folded circuit:
            ┌───┐┌──────────┐┌─────────┐┌───────────┐┌───┐
    q_0: |0>┤ H ├┤ Ry(pi/4) ├┤ Rx(-pi) ├┤ Ry(-pi/4) ├┤ H ├──■──
            └───┘└──────────┘└─────────┘└───────────┘└───┘┌─┴─┐
    q_1: |0>──────────────────────────────────────────────┤ X ├
                                                          └───┘

Notice that we specify ``keep_input_type=True`` to return a circuit of the same type as the input. If this
is not specified, the internal ``mitiq`` representation of a circuit (using Cirq by default) will be returned.


.. note::

    Compared to the previous example which input a Cirq circuit, we see that this folded circuit has more gates. In
    particular, the inverse Hadamard gate is expressed differently (but equivalently) as a product of three
    rotations. This behavior occurs because circuits are first converted to ``mitiq``'s internal
    representation (Cirq circuits), then folded, then converted back to the input circuit type.
    Because different circuits decompose gates differently, some gates (or their inverses)
    may be expressed differently (but equivalently) across different circuits.

=============================================
Global folding
=============================================

As mentioned, global folding methods fold the entire circuit instead of individual gates. An example using the same Cirq
circuit above is shown below.


.. doctest:: python

    >>> import cirq
    >>> from mitiq.folding import fold_global

    # Get a circuit to fold
    >>> qreg = cirq.LineQubit.range(2)
    >>> circ = cirq.Circuit(cirq.ops.H.on(qreg[0]), cirq.ops.CNOT.on(qreg[0], qreg[1]))
    >>> print("Original circuit:", circ, sep="\n")
    Original circuit:
    0: ───H───@───
              │
    1: ───────X───

    # Fold the circuit
    >>> folded = fold_global(circ, stretch=2.)
    >>> print("Folded circuit:", folded, sep="\n")
    Folded circuit:
    0: ───H───@───@───H───H───@───
              │   │           │
    1: ───────X───X───────────X───

Notice that this circuit is still logically equivalent to the input circuit, but the global folding strategy folds
the entire circuit until the input stretch factor is reached.


=============================================
Folding with larger stretches
=============================================

The three local folding methods introduced require that the stretch factor be between one and three (inclusive). To fold
circuits with larger stretch factors, the function ``mitiq.folding.fold_local`` can be used. This function inputs a
circuit, an arbitrary stretch factor, and a local folding method, as in the following example.

.. doctest:: python

    >>> import cirq
    >>> from mitiq.folding import fold_local, fold_gates_from_left

    # Get a circuit to fold
    >>> qreg = cirq.LineQubit.range(2)
    >>> circ = cirq.Circuit(cirq.ops.H.on(qreg[0]), cirq.ops.CNOT.on(qreg[0], qreg[1]))
    >>> print("Original circuit:", circ, sep="\n")
    Original circuit:
    0: ───H───@───
              │
    1: ───────X───

    # Fold the circuit
    >>> folded = fold_local(circ, stretch=5., fold_method=fold_gates_from_left)
    >>> print("Folded circuit:", folded, sep="\n")
    Folded circuit:
    0: ───H───H───H───H───H───H───H───@───@───@───
                                      │   │   │
    1: ───────────────────────────────X───X───X───

=============================================
Local folding with a custom strategy
=============================================

The ``fold_local`` method from the previous example can input custom folding functions. The signature
of this function must be as follows.

.. doctest:: python

    import cirq

    def my_custom_folding_function(circuit: cirq.Circuit, stretch: float) -> cirq.Circuit:
        # Implements the custom folding strategy
        return folded_circuit

This function can then be used with ``fold_local`` as in the previous example via

.. doctest:: python

    # Variables circ and stretch are a circuit to fold and a stretch factor, respectively
    folded = fold_local(circ, stretch, fold_method=my_custom_folding_function)
