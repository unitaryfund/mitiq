.. mitiq documentation file

*********************************************
Factory objects
*********************************************

*Factories* are important elements of the ``mitiq`` library.

The abstract class ``Factory`` is a high-level representation of a generic error mitigation method. 
A factory is not just hardware-agnostic, it is even *quantum-agnostic*,
in the sense that it only deals with classical data: the classical input and the classical output of a
noisy computation.

Specific classes derived from ``Factory``, like ``LinearFactory``, ``RichardsonFactory``, etc., represent   
different zero-noise extrapolation methods. 

The main tasks of a factory are:
    
1. Record the result of the computation executed at the chosen noise level;

2. Determine the noise level at which the next computation should be run;

3. Given the history of noise levels (``self.instack``) and results (``self.outstack``), 
   evaluate the associated zero-noise extrapolation.

The structure of the ``Factory`` class is adaptive by construction, since the choice of the next noise
level can depend on the history of ``self.instack`` and ``self.outstack``.

The abstract class of a non-adaptive extrapolation method is ``BatchedFactory``. 
The main feature of ``BatchedFactory`` is that all the noise levels are determined
*a priori* by the initialization argument ``scalars``.
All non-adaptive methods are derived from ``BatchedFactory``.  


=============================================
Example: basic usage of a factory.
=============================================

To make an example, let us assume that the result of our quantum computation is an expectation 
value which has a linear dependance on the noise.
Since our aim to understand the usage of a factory, instead of actually running quantum experiments, 
we simply simulate an effective classical model which returns the expectation value as a function of the noise level:

.. code-block:: python

   def noise_to_expval(noise_level: float) -> float:
      """A simple linear model for the expectation value."""
      A = 0.5
      B = 0.7
      return A + B * noise_level

In this case the zero noise limit is ``A = 0.5`` and we would like to deduce it by evaluating
the function only for values of ``noise_level`` which are larger than or equal to 1.


In this example, we plan to measure the expectation value at 3 different noise levels: ``NOISE_LEVELS = [1.0, 2.0, 3.0]``.

To get the zero-noise limit, we are going to use a ``LinearFactory`` object, run it until convergence 
(in this case until 3 expectation values are measured and saved) and eventually perform the zero noise extrapolation.

.. code-block:: python

   from mitiq.factories import LinearFactory

   # Some fixed noise levels
   NOISE_LEVELS = [1.0, 2.0, 3.0]

   # Instantiate a LinearFactory object
   fac = LinearFactory(NOISE_LEVELS)

   # Run the factory until convergence
   while not fac.is_converged():
      # Get the next noise level from the factory
      next_noise_level = fac.next()
      # Evaluate the expectation value
      expval = noise_to_expval(next_noise_level)
      # Save the noise level and the result into the factory
      fac.push(next_noise_level, expval)
   
   # Evaluate the zero-noise extrapolation.
   zn_limit = fac.reduce()


In the previous code block we used the main methods of a typical ``Factory`` object:

   - **self.next** to get the next noise level;
   - **self.push** to save data into the factory;
   - **self.is_converged** to know if enough data has been pushed;
   - **self.reduce** to get the zero-noise extrapolation.   

Since our idealized model ``noise_to_expval`` is linear and noiseless, 
the extrapolation will exactly match the true zero-noise limit ``A = 0.5``:

>>> print(f"The zero-noise extrapolation is: {zn_limit:.3}")
The zero-noise extrapolation is: 0.5

.. note::
   
   In a real scenario, the quantum expectation value can be determined only up to some statistical uncertainty  
   (due to a finite number of measurement shots). This makes the zero-noise extrapolation less trivial.
   Moreover the expectation value could depend non-linearly on the noise level. In this case
   factories with higher extrapolation *order* (``PolyFactory``, ``RichardsonFactory``, etc.)
   could be more appropriate.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The ``run_factory`` function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Running a factory until convergence is a typical step of the zero-noise extrapolation
workflow. For this reason, in ``mitiq.zne`` there is a built-in function for this task: ``run_factory``.
The previous example can be reduced to the following equivalent code:

.. code-block:: python

   from mitiq.factories import LinearFactory
   from mitiq.zne import run_factory

   # Some fixed noise levels
   NOISE_LEVELS = [1.0, 2.0, 3.0]
   # Instantiate a LinearFactory object
   fac = LinearFactory(NOISE_LEVELS)
   # Run the factory until convergence
   run_factory(fac, noise_to_expval)
   # Evaluate the zero-noise extrapolation.
   zn_limit = fac.reduce()

=============================================
Built-in factories
=============================================

All the built-in factories of ``mitiq`` can be found in the submodule ``mitiq.factories``.
m
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
In this case its 4 core methods must be implemented:
``self.next``, ``self.push``, ``self.is_converged``, and ``self.reduce``.
Moreover ``self.__init__`` can also be overridden if necessary.

A new non-adaptive method can instead be derived from the ``BatchedFactory`` class.
In this case it is usually sufficient to override only ``self.__init__`` and 
``self.reduce``, which are responsible for the initialization and for the
final zero-noise extrapolation, respectively.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Example: a simple custom factory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Assume that, from physical considerations, we know that the true expectation
value must always be within two limits: ``min_expval`` and ``max_expval``.
For example, this is a typical situation whenever the measured observable has a bounded
spectrum.

We can define a linear non-adaptive factory which takes into account this information
and clips the result if it falls outside its physical domain.

.. code-block:: python
 
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
            scalars: Iterable[float],
            min_expval: float, 
            max_expval: float,
         ) -> None:
         """
         Args:
            scalars: The noise scale factors at which expectation 
                     values should be measured.
            min_expval: The lower bound for the expectation value.
            min_expval: The upper bound for the expectation value.
         """
         super(MyFactory, self).__init__(scalars)
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

This custom factory can be used in exactly the same way as we have
shown in the previous section. By simply replacing ``LinearFactory``
with ``MyFactory`` in all the previous code snippets, the new extrapolation 
method will be applied.
