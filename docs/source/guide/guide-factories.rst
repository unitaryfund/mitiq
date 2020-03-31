.. mitiq documentation file

*********************************************
Factory objects
*********************************************

*Factories* are key elements of the ``mitiq`` library.

The abstract class ``Factory`` is a high-level representation of a generic error mitigation method. 
A factory is not just hardware-agnostic, it is even *quantum-agnostic*,
in the sense that it only deals with classical data: the classical input and the classical output of a
noisy computation.

Specific classes derived from ``Factory``, like ``LinearFactory``, ``RichardsonFactory``, etc., represent   
different zero-noise extrapolation methods. 

The main tasks of a factory are:
    
1. Determine the noise level at which the next computation should be run.

2. Record the result of the computation executed at the chosen noise level.

3. Given the history of noise levels (``self.instack``) and results (``self.outstack``), 
   evaluate the associated zero-noise extrapolation.

The structure of ``Factory`` is adaptive by construction, since the choice of the next noise
level can depend on the history of ``self.instack`` and ``self.outstack``.

The abstract class of a non-adaptive ``Factory`` is ``BatchedFactory``. 
The main feature of ``BatchedFactory`` is that all the noise levels are determined
*a priori* by the initialization argument ``scalars``.
All non-adaptive methods are derived from ``BatchedFactory``.  


=============================================
Example
=============================================

To make an example, let us assume that the result of our quantum computation is an expectation 
value which has a linear dependance on the noise.
Since our aim to understand the usage of a factory, instead of actually running quantum quantum experiments, 
we simply simulate an effective classical model for the expectation value as a function of the noise level:

.. code-block:: python

   def noise_to_expval(noise_level):
      A = 0.5
      B = 0.7
      return A + B * noise_level

In this case the zero noise limit is ``A = 0.5`` and we would like to extrapolate its value evaluating
the function only at ``noise_level`` larger then 1.


We plan to measure the expectation value at 3 different noise levels: ``NOISE_LEVELS = [1, 1.3, 1.7]``.

To get the zero-noise limit, we are going to use a ``LinearFactory`` object, run it until convergence 
(in this case until 3 expectation values are saved) and eventually perform the zero noise extrapolation.

.. code-block:: python

   from mitiq.factories import LinearFactory

   # Some fixed noise levels
   NOISE_LEVELS = [1, 1.3, 1.7]

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
   zero_noise_limit = fac.reduce()


In the previous code block we used the main methods of a typical ``Factory`` object:

   - **self.next()** to get the next noise level;
   - **self.push()** to save data into the factory;
   - **self.is_converged()** to know if enough data has been pushed.
   - **self.reduce()** to get the zero-noise extrapolation.   

Since our idealized model ``noise_to_expval`` is linear and noiseless, 
the extrapolation will exactly match the true zero-noise limit ``A = 0.5``:

.. code-block:: python

   >>> print(f"The zero-noise extrapolation is: {zero_noise_limit:.3}")
   The zero-noise extrapolation is: 0.5

.. note::
   
   In a real scenario, the quantum expectation value can be determined only up to some statistical uncertainty  
   (due to a finite number of measurement shots). This makes the zero-noise extrapolation less trivial.
   Moreover the expectation value could depend non-linearly on the noise level. In this case
   factories with higher extrapolation *order* (``PolyFactory``, ``RichardsonFactory``, etc.)
   could be more appropriate.

=============================================
Defining a custom Factory
=============================================

Todo...