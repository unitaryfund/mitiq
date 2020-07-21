.. mitiq documentation file

.. _guide-factories:

*********************************************
Factory Objects
*********************************************

*Factories* are important elements of the ``mitiq`` library.

A ``Factory`` class is a self-contained representation of an error mitigation method.
This representation not just hardware-agnostic, it is even *quantum-agnostic*,
in the sense that it mainly deals with classical data: the classical input and the classical output of a
noisy computation. Nonetheless, a factory object can easily interact with a quantum system via its ``self.run`` method
which is the only interface between the *"classical world"* of a factory and the *"quantum world"* of a circuit.

Specific classes derived from the abstract class ``Factory``, like ``LinearFactory``, ``RichardsonFactory``, etc., represent
different zero-noise extrapolation methods.

The typical tasks of a factory are:

1. Record the result of the computation executed at the chosen noise level;

2. Determine the noise scale factor at which the next computation should be run;

3. Given the history of noise scale factors (``self.instack``) and results (``self.outstack``),
   evaluate the associated zero-noise extrapolation.

The structure of the ``Factory`` class is adaptive by construction, since the choice of the next noise
level can depend on the history of ``self.instack`` and ``self.outstack``.

The abstract class of a non-adaptive extrapolation method is ``BatchedFactory``.
The main feature of ``BatchedFactory`` is that all the noise scale factors are determined
*a priori* by the initialization argument ``scale_factors``.
All non-adaptive methods are derived from ``BatchedFactory``.

Once instantiated, a factory object can be passed as an argument to the high-level functions contained in the module ``mitiq.zne``.
Alternatively, a factory object can also be used to implement an entire zero-noise extrapolation procedure in an self-contained way, i.e., without 
using the high-level tools in ``mitiq.zne``. This approach may require more lines of code but allows for a low-level control on 
the zero-noise extrapolation method.

In the following sections we are going to perform the same zero-noise extrapolation task at different levels of abstraction: high-level,
intermediate-level and low-level. The user is free to chose one of the three alternative methods, depending on the particular problem of interest 
and on the desired level of flexibility.

=============================================
High level usage of factory objects
=============================================


Let us consider an ``executor`` function which is similar to the one used in the :ref:`getting started <guide-getting-started>` section.


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
   
   In this example we used *Cirq* but other quantum software platforms can be used as well,
   as discussed in :ref:`getting started <guide-getting-started>` section.

We define a single-qubit circuit whose ideal expectation value is by construction equal to
``1.0``. 

.. testcode::

   from cirq import LineQubit, X, H

   qubit = LineQubit(0)
   circuit = Circuit(X(qubit), H(qubit), H(qubit), X(qubit))
   expval = executor(circuit)
   exact = 1
   print(f"The ideal result should be {exact}")
   print(f"The real result is {expval:.4}")
   print(f"The abslute error is {abs(exact - expval):.4}")

.. testoutput::

   The ideal result should be 1
   The real result is 0.8794
   The abslute error is 0.1206


Now we are going to instantiate different factory objects, each encapsulating a different zero-noise extrapolation method. 

.. testcode::

   from mitiq.factories import LinearFactory, RichardsonFactory, PolyFactory

   # method: scale noise 1 and 2, then extrapolate linearly to the zero noise limit.
   linear_fac = LinearFactory(scale_factors = [1.0, 2.0])

   # method: scale noise by 1, 2 and 3, then evaluate the Richardson extrapolation.
   richardson_fac = RichardsonFactory(scale_factors = [1.0, 2.0, 3.0])

   # method: scale noise by 1, 2, 3, and 4, then extrapolate quadratically to the zero noise limit.
   poly_fac = PolyFactory(scale_factors = [1.0, 2.0, 3.0, 4.0], order=2)

The previous factory objects can be passed as arguments to the high-level functions in ``mitiq.zne``. For example:

.. testcode::

   from mitiq.zne import execute_with_zne

   zne_expval = execute_with_zne(circuit, executor, factory = linear_fac)
   print(f"Error with linear_fac: {abs(exact - zne_expval):.4}")

   zne_expval = execute_with_zne(circuit, executor, factory = richardson_fac)
   print(f"Error with richardson_fac: {abs(exact - zne_expval):.4}")

   zne_expval = execute_with_zne(circuit, executor, factory = poly_fac)
   print(f"Error with poly_fac: {abs(exact - zne_expval):.4}")

.. testoutput::

   Error with linear_fac: 0.02908
   Error with richardson_fac: 0.007013
   Error with poly_fac: 2.37e-07


=============================================
Intermediate-level usage of a factory
=============================================

Zero-noise extrapolation can also be implemented by directly using the methods ``self.qrun`` and ``self.reduce`` of factory object.
The method ``self.run`` *runs* the factory object until convergence, i.e., it evaluates different expectation values at different
noise levels until a sufficient amount of data is collected. The ``self.reduce`` method instead returns the final
zero-noise extrapolation and corresponds to a classical inference based on the measured data.

.. testcode::

   # we use fold_global but other noise scale functions could be used
   from mitiq.folding import fold_global

   linear_fac.run(circuit, executor, scale_noise = fold_global)
   zne_expval = linear_fac.reduce()
   print(f"Error with linear_fac: {abs(exact - zne_expval):.4}")

   richardson_fac.run(circuit, executor, scale_noise = fold_global)
   zne_expval = richardson_fac.reduce()
   print(f"Error with richardson_fac: {abs(exact - zne_expval):.4}")

   poly_fac.run(circuit, executor, scale_noise = fold_global)
   zne_expval = poly_fac.reduce()
   print(f"Error with poly_fac: {abs(exact - zne_expval):.4}")


=============================================
Low-level usage of a factory
=============================================

To make an example, let us assume that the result of our quantum computation is an expectation
value which has a linear dependance on the noise.
Since our aim is to understand the usage of a factory, instead of actually running quantum experiments,
we simply simulate an effective classical model which returns the expectation value as a function of the
noise scale factor.

.. testcode::

   def noise_to_expval(scale_factor: float) -> float:
      """A simple linear model for the expectation value."""
      ZERO_NOISE_LIMIT = 0.5
      NOISE_ERROR = 0.7
      return ZERO_NOISE_LIMIT + NOISE_ERROR * scale_factor

In this case the zero-noise limit is ``0.5`` and we would like to deduce it by evaluating
the function only for values of ``scale_factor`` which are larger than or equal to 1.

.. note::

   For implementing zero-noise extrapolation, it is not necessary to know the details of the
   noise model. It is also not necessary to control the absolute strength of the noise
   acting on the physical system. The only key assumption is that we can artificially scale the noise
   with respect to its normal level by a dimensionless ``scale_factor``.
   A practical approach for scaling the noise is discussed in the :ref:`guide-folding` section.


In this example, we plan to measure the expectation value at 3 different noise scale
factors: ``SCALE_FACTORS = [1.0, 2.0, 3.0]``.

To get the zero-noise limit, we are going to use a ``LinearFactory`` object, run it until convergence
(in this case until 3 expectation values are measured and saved) and eventually perform the zero-noise extrapolation.

.. testcode::

   from mitiq.factories import LinearFactory

   # Some fixed noise scale factors
   SCALE_FACTORS = [1.0, 2.0, 3.0]

   # Instantiate a LinearFactory object
   fac = LinearFactory(SCALE_FACTORS)

   # Run the factory until convergence
   while not fac.is_converged():
      # Get the next noise scale factor from the factory
      next_scale_factor = fac.next()
      # Evaluate the expectation value
      expval = noise_to_expval(next_scale_factor)
      # Save the noise scale factor and the result into the factory
      fac.push(next_scale_factor, expval)

   # Evaluate the zero-noise extrapolation.
   zn_limit = fac.reduce()
   print(f"{zn_limit:.3}")

.. testoutput::

   0.5

In the previous code block we used the main methods of a typical ``Factory`` object:

   - **self.next** to get the next noise scale factor;
   - **self.push** to save data into the factory;
   - **self.is_converged** to know if enough data has been pushed;
   - **self.reduce** to get the zero-noise extrapolation.

Since our idealized model ``noise_to_expval`` is linear and noiseless,
the extrapolation will exactly match the true zero-noise limit ``0.5``:

.. testcode::

   print(f"The zero-noise extrapolation is: {zn_limit:.3}")

.. testoutput::

   The zero-noise extrapolation is: 0.5

.. note::

   In a real scenario, the quantum expectation value can be determined only up to some statistical uncertainty
   (due to a finite number of measurement shots). This makes the zero-noise extrapolation less trivial.
   Moreover the expectation value could depend non-linearly on the noise level. In this case
   factories with higher extrapolation *order* (``PolyFactory``, ``RichardsonFactory``, etc.)
   could be more appropriate.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The ``Factory().iterate`` method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Running a factory until convergence is a typical step of the zero-noise extrapolation
workflow. For this reason, every factory can be run to convergence using an
``iterate`` method. The previous example can be simplified to the following
equivalent code:

.. testcode::

   from mitiq.factories import LinearFactory

   # Some fixed noise scale factors
   SCALE_FACTORS = [1.0, 2.0, 3.0]
   # Instantiate a LinearFactory object
   fac = LinearFactory(SCALE_FACTORS)
   # Run the factory until convergence
   fac.iterate(noise_to_expval)
   # Evaluate the zero-noise extrapolation.
   zn_limit = fac.reduce()
   print(f"The zero-noise extrapolation is: {zn_limit:.3}")

.. testoutput::

   The zero-noise extrapolation is: 0.5

=============================================
Built-in factories
=============================================

All the built-in factories of ``mitiq`` can be found in the submodule ``mitiq.factories``.

.. autosummary::
   :nosignatures:

   mitiq.factories.LinearFactory
   mitiq.factories.RichardsonFactory
   mitiq.factories.PolyFactory
   mitiq.factories.ExpFactory
   mitiq.factories.PolyExpFactory
   mitiq.factories.AdaExpFactory

=============================================
Defining a custom Factory
=============================================

If necessary, the user can modify an existing extrapolation method by subclassing
the corresponding factory.

A new adaptive extrapolation method can be derived from the abstract class ``Factory``.
In this case its core methods must be implemented:
``self.next``, ``self.push``, ``self.is_converged``, and ``self.reduce``.
Moreover ``self.__init__`` can also be overridden if necessary.

A new non-adaptive method can instead be derived from the ``BatchedFactory`` class.
In this case it is usually sufficient to override only ``self.__init__`` and
``self.reduce``, which are responsible for the initialization and for the
final zero-noise extrapolation, respectively.

=============================================
Example: a simple custom factory
=============================================

Assume that, from physical considerations, we know that the ideal expectation value
(measured by some quantum circuit) must always be within two limits: ``min_expval`` and ``max_expval``.
For example, this is a typical situation whenever the measured observable has a bounded
spectrum.

We can define a linear non-adaptive factory which takes into account this information
and clips the result if it falls outside its physical domain.

.. testcode::

   from typing import Iterable
   from mitiq.factories import BatchedFactory
   import numpy as np

   class MyFactory(BatchedFactory):
      """Factory object implementing a linear extrapolation taking
      into account that the expectation value must be within a given
      interval. If the zero-noise extrapolation falls outside the
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
         Fits a line to the data with a least squared method.
         Extrapolates and, if necessary, clips.

         Returns:
            The clipped extrapolation to the zero-noise limit.
         """
         # Fit a line and get the intercept
         _, intercept = np.polyfit(self.instack, self.outstack, 1)

         # Return the clipped zero-noise extrapolation.
         return np.clip(intercept, self.min_expval, self.max_expval)

.. testcleanup::

   fac = MyFactory(SCALE_FACTORS, min_expval=-1.0, max_expval=1.0)
   fac.iterate(noise_to_expval)
   assert np.isclose(fac.reduce(), 0.5)
   # Linear model with a large zero-noise limit
   noise_to_large_expval = lambda x : noise_to_expval(x) + 10.0
   fac.iterate(noise_to_large_expval)
   assert np.isclose(fac.reduce(), 1.0)

This custom factory can be used in exactly the same way as we have
shown in the previous section. By simply replacing ``LinearFactory``
with ``MyFactory`` in all the previous code snippets, the new extrapolation
method will be applied.
