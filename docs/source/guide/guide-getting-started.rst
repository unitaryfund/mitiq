.. mitiq documentation file

*********************************************
Getting Started
*********************************************

Improving the performance of your quantum programs is only a few lines of
code away.

Here's an example using
`cirq <https://cirq.readthedocs.io/en/stable/index.html>`_. We'll test `mitiq`
by running against the noisy simulator built into `cirq`.

We first define some functions that make it simpler to simulate noise in
`cirq`. These don't have to do with `mitiq` directly.

.. code-block:: python

    import numpy as np
    from typing import Tuple
    from cirq import Circuit, depolarize, LineQubit, X, DensityMatrixSimulator

    SIMULATOR = DensityMatrixSimulator()

    def meas_observable(rho: np.ndarray, obs: np.ndarray) -> Tuple[float, float]:
        """Measures a density matrix rho against observable obs.

        Args:
            rho: A density matrix.
            obs: A Hermitian observable.

        Returns:
            The tuple (expectation value, variance).
        """
        obs_avg = np.real(np.trace(rho @ obs))
        obs_delta = np.sqrt(np.real(np.trace(rho @ obs @ obs)) - obs_avg ** 2)
        return obs_avg, obs_delta


    # 0.1% depolarizing noise
    NOISE = 0.001


    def noisy_simulation(circ: Circuit, shots=None) -> Tuple[float, float]:
        """ Simulates a circuit with depolarizing noise at level NOISE.

        Args:
            circ: The quantum program as a cirq object.
            shots: This unused parameter is needed to match mitiq's expected type
                   signature for an executor function.

        Returns:
            The observable's measurements as as
            tuple (expectation value, variance).
        """
        A = np.diag([1, 0])
        circuit = circ.with_noise(depolarize(p=NOISE))
        rho = SIMULATOR.simulate(circuit).final_density_matrix
        A_avg, A_delta = meas_observable(rho, obs=A)
        return A_avg, A_delta

Now we can look at our example. We'll test single qubit circuits with even
numbers of X gates. As there are an even number of X gates, they should all
evaluate to an expectation of 1 in the computational basis if there was no
noise.

.. code-block:: python

    >>> qbit = LineQubit(0)
    >>> circ = Circuit()
    >>> for _ in range(80):
    >>>    circ += X(qbit)

    >>> unmitigated, _ = noisy_simulation(circ)
    >>> exact = 1
    >>> print(f"Error in simulation is {exact - unmitigated:.{3}}")

    Error in simulation is 0.0506

This shows the impact the noise has had. Let's use `mitiq to improve this
performance.

.. code-block:: python

    >>> from mitiq import execute_with_zne

    >>> mitigated, var = execute_with_zne(circ, noisy_simulation)
    >>> print(f"Error in simulation is {exact - mitigated:.{3}}")
    >>> print("Mitigation provides " \
    >>>  f"a {(exact - unmitigated) / (exact - mitigated):.{3}}" \
    >>>  "factor of improvement.")

    Error in simulation is 0.000519
    Mitigation provides a 97.6factor of improvement.

..

The variance in the mitigated expectation value is now stored in `var`.

The default implementation uses Richardson extrapolation to extrapolate the
expectation value to the zero noise limit. `Mitiq` comes equipped with other
extrapolation methods as well. Different methods of extrapolation are packaged
into `Factory` objects. It is easy to try different ones.

.. code-block:: python
    >>> from mitiq.factories import LinearFactory

    >>> fac = LinearFactory([1.0, 2.0, 2.5])
    >>> linear, _ = execute_with_zne(circ, noisy_simulation, fac=fac)
    >>> print("Mitigated error with the linear method" \
              f"is {exact - linear:.{3}}")

    Mitigated error with the linear methodis 0.00638

You can read more about the `Factory` objects that are built into `mitiq` and
how to create your own `here <factories.html>`_.

Another key step in zero-noise extrapolation is to choose how your circuit is
transformed to scale the noise. You can read more about the noise scaling
methods built into `mitiq` and how to create your
own `here <noise-scaling.html>`_.