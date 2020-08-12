.. mitiq documentation file

.. _guide-factories:

============================================================
Classical fitting and extrapolation: Factory Objects
============================================================
A :class:`.Factory` object is a self-contained representation of an error mitigation method.

This representation is not just hardware-agnostic, it is even *quantum-agnostic*,
in the sense that it mainly deals with classical data: the classical input and the classical output of a
noisy computation. Nonetheless, a factory can easily interact with a quantum system via its ``self.run`` method
which is the only interface between the "classical world" of a factory and the "quantum world" of a circuit.

The typical tasks of a factory are:

1. Record the result of the computation executed at the chosen noise level;

2. Determine the noise scale factor at which the next computation should be run;

3. Given the history of noise scale factors and results
   (respectively stored in the object attributes ``self.instack``
   and ``self.outstack``), evaluate the associated zero-noise extrapolation.

The structure of the :class:`.Factory` class is adaptive by construction, since the choice of the next noise
level can depend on the history of ``self.instack`` and ``self.outstack``. Obviously, non-adaptive
methods are supported too and they actually represent the most common choice.

Specific classes derived from the abstract class :class:`.Factory`, like :class:`.LinearFactory`,
:class:`.RichardsonFactory`, etc., represent different zero-noise extrapolation methods.
All the built-in factories can be found in the module :py:mod:`mitiq.factories` and
are summarized in the following table.

.. _built-in-factories:

   .. autosummary::
      :nosignatures:

      mitiq.factories.LinearFactory
      mitiq.factories.RichardsonFactory
      mitiq.factories.PolyFactory
      mitiq.factories.ExpFactory
      mitiq.factories.PolyExpFactory
      mitiq.factories.AdaExpFactory


Once instantiated, a factory can be passed as an argument to the high-level functions contained in the module :py:mod:`mitiq.zne`.
Alternatively, a factory can be directly used to implement a zero-noise extrapolation procedure in a fully self-contained way.

To clarify this aspect, we now perform the same zero-noise extrapolation with both methods.

----------------------------------------------------------
Using a factory object with the :py:mod:`mitiq.zne` module
----------------------------------------------------------

Let us consider an ``executor`` function which is similar to the one used in
the :ref:`getting started <guide-getting-started>` section.

.. testcode::

   import numpy as np
   from cirq import Circuit, depolarize, DensityMatrixSimulator

   # initialize a backend
   SIMULATOR = DensityMatrixSimulator()
   # 5% depolarizing noise
   NOISE = 0.05

   def executor(circ: Circuit) -> float:
      """Executes a circuit with depolarizing noise and
      returns the expectation value of the projector |0><0|."""
      circuit = circ.with_noise(depolarize(p=NOISE))
      rho = SIMULATOR.simulate(circuit).final_density_matrix
      obs = np.diag([1, 0])
      expectation = np.real(np.trace(rho @ obs))
      return expectation

.. note::

   In this example we used *Cirq* but other quantum software platforms can be used,
   as shown in the :ref:`getting started <guide-getting-started>` section.

We also define a simple quantum circuit whose ideal expectation value is by construction equal to
``1.0``.

.. testcode::

   from cirq import LineQubit, X, H

   qubit = LineQubit(0)
   circuit = Circuit(X(qubit), H(qubit), H(qubit), X(qubit))
   expval = executor(circuit)
   exact = 1.0
   print(f"The ideal result should be {exact}")
   print(f"The real result is {expval:.4f}")
   print(f"The abslute error is {abs(exact - expval):.4f}")

.. testoutput::

   The ideal result should be 1.0
   The real result is 0.8794
   The abslute error is 0.1206


Now we are going to initialize three factory objects, each one encapsulating a different
zero-noise extrapolation method.

.. testcode::

   from mitiq.factories import LinearFactory, RichardsonFactory, PolyFactory

   # method: scale noise by 1 and 2, then extrapolate linearly to the zero noise limit.
   linear_fac = LinearFactory(scale_factors=[1.0, 2.0])

   # method: scale noise by 1, 2 and 3, then evaluate the Richardson extrapolation.
   richardson_fac = RichardsonFactory(scale_factors=[1.0, 2.0, 3.0])

   # method: scale noise by 1, 2, 3, and 4, then extrapolate quadratically to the zero noise limit.
   poly_fac = PolyFactory(scale_factors=[1.0, 2.0, 3.0, 4.0], order=2)

The previous factory objects can be passed as arguments to the high-level functions
in ``mitiq.zne``. For example:

.. testcode::

   from mitiq.zne import execute_with_zne

   zne_expval = execute_with_zne(circuit, executor, factory=linear_fac)
   print(f"Error with linear_fac: {abs(exact - zne_expval):.4f}")

   zne_expval = execute_with_zne(circuit, executor, factory=richardson_fac)
   print(f"Error with richardson_fac: {abs(exact - zne_expval):.4f}")

   zne_expval = execute_with_zne(circuit, executor, factory=poly_fac)
   print(f"Error with poly_fac: {abs(exact - zne_expval):.4f}")

.. testoutput::

   Error with linear_fac: 0.0291
   Error with richardson_fac: 0.0070
   Error with poly_fac: 0.0110


---------------------------------------------
Directly using a factory for error mitigation
---------------------------------------------

Zero-noise extrapolation can also be implemented by directly using the methods ``self.run``
and ``self.reduce`` of a :class:`.Factory` object.

The method ``self.run`` evaluates different expectation values at different noise levels
until a sufficient amount of data is collected.

The method ``self.reduce`` instead returns the final zero-noise extrapolation which, in practice,
corresponds to a statistical inference based on the measured data.

.. testcode::

   # we import one of the built-in noise scaling function
   from mitiq.folding import fold_gates_at_random

   linear_fac.run(circuit, executor, scale_noise=fold_gates_at_random)
   zne_expval = linear_fac.reduce()
   print(f"Error with linear_fac: {abs(exact - zne_expval):.4f}")

   richardson_fac.run(circuit, executor, scale_noise=fold_gates_at_random)
   zne_expval = richardson_fac.reduce()
   print(f"Error with richardson_fac: {abs(exact - zne_expval):.4f}")

   poly_fac.run(circuit, executor, scale_noise=fold_gates_at_random)
   zne_expval = poly_fac.reduce()
   print(f"Error with poly_fac: {abs(exact - zne_expval):.4f}")

.. testoutput::

   Error with linear_fac: 0.0291
   Error with richardson_fac: 0.0070
   Error with poly_fac: 0.0110

---------------------------
Advanced usage of a factory
---------------------------

.. note::
   This section can be safely skipped by all the readers who are interested
   in a standard usage of ``mitiq``.
   On the other hand, more experienced users and ``mitiq`` contributors
   may find this content useful to understand how a factory object actually
   works at a deeper level.

In this advanced section we present a *low-level usage* and a *very-low-level usage* of a factory.
Again, for simplicity, we solve the same zero-noise extrapolation problem that we have just considered
in the previous sections.

Eventually we will also discuss how the user can easily define a custom factory class.


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Low-level usage: the ``iterate`` method.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``self.run`` method takes as arguments a circuit and other "quantum" objects.
On the other hand, the core computation performed by any factory corresponds to
a some classical computation applied to the measurement results.

At a lower level, it is possible to clearly separate the quantum and the
classical steps of a zero-noise extrapolation procedure.
This can be done by defining a function which maps a noise scale factor to the
corresponding expectation value.

.. testcode::

   def noise_to_expval(scale_factor: float) -> float:
      """Function returning an expectation value for a given scale_factor."""
      # apply noise scaling
      scaled_circuit = fold_gates_at_random(circuit, scale_factor)
      # return the corresponding expectation value
      return executor(scaled_circuit)

.. note::
   The body of the previous function contains the execution of a quantum circuit.
   However, if we see it as a "black-box", it is just a classical function mapping real
   numbers to real numbers.

The function ``noise_to_expval`` encapsulate the "quantum part" of the problem. The "classical
part" of the problem can be solved by passing ``noise_to_expval``
to the ``self.iterate`` method of a factory object.
This method will repeatedly call ``noise_to_expval`` for different
noise levels until a sufficient amount of data is collected.
So, one can view ``self.iterate`` as the classical counterpart of the quantum method ``self.run``.

.. testcode::

   linear_fac.iterate(noise_to_expval)
   zne_expval = linear_fac.reduce()
   print(f"Error with linear_fac: {abs(exact - zne_expval):.4f}")

   richardson_fac.iterate(noise_to_expval)
   zne_expval = richardson_fac.reduce()
   print(f"Error with richardson_fac: {abs(exact - zne_expval):.4f}")

   poly_fac.iterate(noise_to_expval)
   zne_expval = poly_fac.reduce()
   print(f"Error with poly_fac: {abs(exact - zne_expval):.4f}")

.. testoutput::

   Error with linear_fac: 0.0291
   Error with richardson_fac: 0.0070
   Error with poly_fac: 0.0110

.. note::
   With respect to ``self.run`` the ``self.iterate`` method is much more flexible and
   can be applied whenever the user is able to autonomously scale the noise level associated
   to an expectation value. Indeed, the function ``noise_to_expval`` can represent any experiment
   or any simulation in which noise can be artificially increased. The scenario
   is therefore not restricted to quantum circuits but can be easily extended to
   annealing devices or to gates which are controllable at a pulse level. In principle,
   one could even use the ``self.iterate`` method to mitigate experiments which are
   unrelated to quantum computing.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Very low-level usage of a factory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is also possible to emulate the action of the ``self.iterate`` method
by manually measuring individual expectation values and saving them, one by one, into the factory.

.. note::
   In a typical situation, such a deep level of control is likely unnecessary.
   It is anyway instructive to understand the internal structure of the
   :class:`.Factory` class, especially if one is interested in defining a custom factory.


.. testcode::

   zne_list = []
   # loop over different factories
   for fac in [linear_fac, richardson_fac, poly_fac]:
      # loop until enough expectation values are measured
      while not fac.is_converged():
         # Get the next noise scale factor from the factory
         next_scale_factor = fac.next()
         # Evaluate the expectation value
         expval = noise_to_expval(next_scale_factor)
         # Save the noise scale factor and the result into the factory
         fac.push(next_scale_factor, expval)
      # evaluate the zero-noise limit and append it to zne_list
      zne_list.append(fac.reduce())

   print(f"Error with linear_fac: {abs(exact - zne_list[0]):.4f}")
   print(f"Error with richardson_fac: {abs(exact - zne_list[1]):.4f}")
   print(f"Error with poly_fac: {abs(exact - zne_list[2]):.4f}")

.. testoutput::

   Error with linear_fac: 0.0291
   Error with richardson_fac: 0.0070
   Error with poly_fac: 0.0110


In the previous code block we used the some core methods of a :class:`.Factory` object:

   - ``self.next`` to get the next noise scale factor;
   - ``self.push`` to save the measured data into the factory;
   - ``self.is_converged`` to know if enough data has been collected.


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Defining a custom factory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If necessary, the user can modify an existing extrapolation methods by subclassing
one of the :ref:`built-in factories <built-in-factories>`.

Alternatively, a new adaptive extrapolation method can be derived from the abstract class :class:`.Factory`.
In this case its core methods must be implemented:
``self.next``, ``self.push``, ``self.is_converged``, ``self.reduce``, etc.
Typically, the ``self.__init__`` method must be overridden.

A new non-adaptive method can instead be derived from the abstract :class:`.BatchedFactory` class.
In this case it is usually sufficient to override only the ``self.__init__`` and
the ``self.reduce`` methods, which are responsible for the initialization and for the
final zero-noise extrapolation, respectively.

---------------------------------------------
Example: a simple custom factory
---------------------------------------------

Assume that, from physical considerations, we know that the ideal expectation value
(measured by some quantum circuit) must always be within two limits: ``min_expval`` and ``max_expval``.
For example, this is a typical situation whenever the measured observable has a bounded
spectrum.

We can define a linear non-adaptive factory which takes into account this information
and clips the result if it falls outside its physical domain.

.. testcode::

   from typing import Iterable
   from mitiq.factories import BatchedFactory, mitiq_polyfit
   import numpy as np

   class MyFactory(BatchedFactory):
      """Factory object implementing a linear extrapolation taking
      into account that the expectation value must be within a given
      interval. If the zero-noise limit falls outside the
      interval, its value is clipped.
      """

      def __init__(
            self,
            scale_factors: Iterable[float],
            min_expval: float,
            max_expval: float,
         ) -> None:
         """
         Args:
            scale_factors: The noise scale factors at which
                           expectation values should be measured.
            min_expval: The lower bound for the expectation value.
            min_expval: The upper bound for the expectation value.
         """
         super(MyFactory, self).__init__(scale_factors)
         self.min_expval = min_expval
         self.max_expval = max_expval

      def reduce(self) -> float:
         """
         Fit a linear model and clip its zero-noise limit.

         Returns:
            The clipped extrapolation to the zero-noise limit.
         """
         # Fit a line and get the intercept
         scale_factors = [params["scale_factor"] for params in self.instack]
         _, intercept = mitiq_polyfit(scale_factors, self.outstack, deg=1)

         # Return the clipped zero-noise extrapolation.
         return np.clip(intercept, self.min_expval, self.max_expval)

.. testcleanup::

   fac = MyFactory([1, 2, 3], min_expval=0.0, max_expval=2.0)
   fac.iterate(noise_to_expval)
   assert np.isclose(fac.reduce(), 1.0, atol=0.1)
   # Linear model with a large zero-noise limit
   noise_to_large_expval = lambda x : noise_to_expval(x) + 10.0
   fac.iterate(noise_to_large_expval)
   # assert the output is clipped to 2.0
   assert np.isclose(fac.reduce(), 2.0)

This custom factory can be used in exactly the same way as we have
shown in the previous section. By simply replacing ``LinearFactory``
with ``MyFactory`` in all the previous code snippets, the new extrapolation
method will be applied.

---------------------------------------------
Regression tools in :py:mod:`mitiq.factories`
---------------------------------------------

In the body of the previous ``MyFactory`` example, we imported and used the :py:func:`.mitiq_polyfit` function.
This is simply a wrap of :py:func:`numpy.polyfit`, slightly adapted to the notion and to the error types
of ``mitiq``. This function can be used to fit a polynomial ansatz to the measured expectation values. This function performs
a least squares minimization which is **linear** (with respect to the coefficients) and therefore admits an algebraic solution.

Similarly, from :py:mod:`mitiq.factories` one can also import :py:func:`.mitiq_curve_fit`,
which is instead a wrap of :py:func:`scipy.optimize.curve_fit`. Differently from :py:func:`.mitiq_polyfit`,
:py:func:`.mitiq_curve_fit` can be used with a generic (user-defined) ansatz.
Since the fit is based on a numerical **non-linear** least squares minimization, this method may fail to converge
or could be subject to numerical instabilities.
