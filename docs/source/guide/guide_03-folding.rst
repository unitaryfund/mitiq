.. mitiq documentation file

.. _guide-folding:

*********************************************
Unitary Folding
*********************************************
Zero noise extrapolation has two main components: noise scaling and then extrapolation.
Unitary folding is a method for noise scaling that operates directly at the gate level.
This makes it easy to use it across platforms. It is especially appropriate when
your underlying noise should scale with the depth and/or the number of gates in your
quantum program. More details can be found in :cite:`Giurgica_Tiron_2020_arXiv`
where the unitary folding framework was introduced.

At the gate level, noise is amplified by mapping gates (or groups of gates) `G` to

.. math::
  G \mapsto G G^\dagger G .

This makes the circuit longer (adding more noise) while keeping its effect unchanged (because
:math:`G^\dagger = G^{-1}` for unitary gates).  We refer to this process as
*unitary folding*. If `G` is a subset of the gates in a circuit, we call it `local folding`.
If `G` is the entire circuit, we call it `global folding`.

In ``mitiq``, folding functions input a circuit and a *scale factor* (or simply *scale*), i.e., a floating point value
which corresponds to (approximately) how much the length of the circuit is scaled.
The minimum scale factor is one (which corresponds to folding no gates). A scale factor of three corresponds to folding
all gates locally. Scale factors beyond three begin to fold gates more than once.

=============================================
Local folding methods
=============================================

For local folding, there is a degree of freedom for which gates to fold first. The order in which gates are folded can
have an important effect on how the noise is caled. As such, ``mititq`` defines several local folding methods.

We introduce three folding functions:

    1. ``mitiq.folding.fold_gates_from_left``
    2. ``mitiq.folding.fold_gates_from_right``
    3. ``mitiq.folding.fold_gates_at_random``

The ``mitiq`` function ``fold_gates_from_left`` will fold gates from the left (or start) of the circuit
until the desired scale factor is reached.


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
    >>> folded = fold_gates_from_left(circ, scale_factor=2.)
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
    >>> folded = fold_gates_from_right(circ, scale_factor=2.)
    >>> print("Folded circuit:", folded, sep="\n")
    Folded circuit:
    0: ───H───@───@───@───
              │   │   │
    1: ───────X───X───X───

We see the second (CNOT) gate in the circuit is folded, as expected when we start folding from the right (or end) of
the circuit instead of the left (or start).

Finally, we mention ``fold_gates_at_random`` which folds gates according to the following rules.

    1. Gates are selected at random and folded until the input scale factor is reached.
    2. No gate is folded more than once for any ``scale_factor <= 3``.
    3. "Virtual gates" (i.e., gates appearing from folding) are never folded.

All of these local folding methods can be called with any ``scale_factor >= 1``.

=============================================
Any supported circuits can be folded
=============================================

Any program types supported by ``mitiq`` can be folded, and the interface for all folding functions is the same. In the
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

    # Fold the circuit
    >>> folded = fold_gates_from_left(circ, scale_factor=2.)
    >>> # print("Folded circuit:", folded, sep="\n")

This code (when the print statement is uncommented) should display something like:

.. code-block:: python

    Folded circuit:
            ┌───┐┌──────────┐┌─────────┐┌───────────┐┌───┐
    q_0: |0>┤ H ├┤ Ry(pi/4) ├┤ Rx(-pi) ├┤ Ry(-pi/4) ├┤ H ├──■──
            └───┘└──────────┘└─────────┘└───────────┘└───┘┌─┴─┐
    q_1: |0>──────────────────────────────────────────────┤ X ├
                                                          └───┘

By default, the folded circuit has the same type as the input circuit. To return an internal ``mitiq`` representation
of the folded circuit (a Cirq circuit), one can use the keyword argument ``return_mitiq=True``.


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
        >>> folded = fold_global(circ, scale_factor=3.)
        >>> print("Folded circuit:", folded, sep="\n")
        Folded circuit:
        0: ───H───@───@───H───H───@───
                  │   │           │
        1: ───────X───X───────────X───

Notice that this circuit is still logically equivalent to the input circuit, but the global folding strategy folds
the entire circuit until the input scale factor is reached. As with local folding methods, global folding can be called
with any ``scale_factor >= 3``.


=============================================
Custom folding methods
=============================================

Custom folding methods can be defined and used with ``mitiq`` (e.g., with ``mitiq.execute_with_zne``. The signature
of this function must be as follows.

.. doctest:: python

    import cirq
    from mitiq.folding import converter

    @converter
    def my_custom_folding_function(circuit: cirq.Circuit, scale_factor: float) -> cirq.Circuit:
        # Insert custom folding method here
        return folded_circuit

.. note::

    The ``converter`` decorator makes it so ``my_custom_folding_function`` can be used with any supported circuit type,
    not just Cirq circuits. The body of the ``my_custom_folding_function`` should assume the input circuit is a Cirq
    circuit, however.

This function can then be used with ``mitiq.execute_with_zne`` as an option to scale the noise:

.. doctest:: python

    # Variables circ and scale are a circuit to fold and a scale factor, respectively
    zne = mitiq.execute_with_zne(circuit, executor, scale_noise=my_custom_folding_function)
