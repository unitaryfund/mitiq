.. mitiq documentation file

*********************************************
Getting Started
*********************************************

Improving the performance of your quantum programs is only a few lines of
code away.

Here's an example using
`cirq <https://cirq.readthedocs.io/en/stable/index.html>`_. We'll test
``mitiq`` by running against the noisy simulator built into ``cirq``.

We first define some functions that make it simpler to simulate noise in
``cirq``. These don't have to do with ``mitiq`` directly.

.. doctest:: python

    >>> import numpy as np
    >>> from cirq import Circuit, depolarize
    >>> from cirq import LineQubit, X, DensityMatrixSimulator
    >>> SIMULATOR = DensityMatrixSimulator()
    >>> # 0.1% depolarizing noise
    >>> NOISE = 0.001
    >>> def noisy_simulation(circ: Circuit, shots=None) -> float:
    ...     """ Simulates a circuit with depolarizing noise at level NOISE.
    ...     Args:
    ...         circ: The quantum program as a cirq object.
    ...         shots: This unused parameter is needed to match mitiq's expected type
    ...                signature for an executor function.
    ...
    ...     Returns:
    ...         The observable's measurements as as
    ...         tuple (expectation value, variance).
    ...     """
    ...     circuit = circ.with_noise(depolarize(p=NOISE))
    ...     rho = SIMULATOR.simulate(circuit).final_density_matrix
    ...     # define the computational basis observable
    ...     obs = np.diag([1, 0])
    ...     expectation = np.real(np.trace(rho @ obs))
    ...     return expectation

Now we can look at our example. We'll test single qubit circuits with even
numbers of X gates. As there are an even number of X gates, they should all
evaluate to an expectation of 1 in the computational basis if there was no
noise.



.. doctest:: python

    >>> from cirq import Circuit, LineQubit, X
    >>> qbit = LineQubit(0)
    >>> circ = Circuit(X(qbit) for _ in range(80))
    >>> unmitigated = noisy_simulation(circ)
    >>> exact = 1
    >>> print(f"Error in simulation is {exact - unmitigated:.{3}}")
    Error in simulation is 0.0506

This shows the impact the noise has had. Let's use ``mitiq`` to improve this
performance.

.. doctest:: python

    >>> from mitiq import execute_with_zne
    >>> mitigated = execute_with_zne(circ, noisy_simulation)
    >>> print(f"Error in simulation is {exact - mitigated:.{3}}")
    Error in simulation is 0.000519
    >>> print(f"Mitigation provides a {(exact - unmitigated) / (exact - mitigated):.{3}} factor of improvement.")
    Mitigation provides a 97.6 factor of improvement.

The variance in the mitigated expectation value is now stored in ``var``.

You can also use ``mitiq`` to wrap your backend execution function into an
error-mitigated version.

.. doctest:: python

    >>> from mitiq import mitigate_executor

    >>> run_mitigated = mitigate_executor(noisy_simulation)
    >>> mitigated = run_mitigated(circ)
    >>> round(mitigated,5)
    0.99948

The default implementation uses Richardson extrapolation to extrapolate the
expectation value to the zero noise limit [1]. ``Mitiq`` comes equipped with other
extrapolation methods as well. Different methods of extrapolation are packaged
into ``Factory`` objects. It is easy to try different ones.

.. doctest:: python

    >>> from mitiq.factories import LinearFactory

    >>> fac = LinearFactory(scale_factors=[1.0, 2.0, 2.5])
    >>> linear = execute_with_zne(circ, noisy_simulation, fac=fac)
    >>> print(f"Mitigated error with the linear method is {exact - linear:.{3}}")
    Mitigated error with the linear method is 0.00638

You can read more about the ``Factory`` objects that are built into ``mitiq`` and
how to create your own `here <factories.html>`_.

Another key step in zero-noise extrapolation is to choose how your circuit is
transformed to scale the noise. You can read more about the noise scaling
methods built into ``mitiq`` and how to create your
own `here <noise-scaling.html>`_.

.. [1] `Error mitigation for short-depth quantum circuits <https://arxiv.org/abs/1612.02058>`_
