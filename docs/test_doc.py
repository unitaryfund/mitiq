def noise_to_expval(noise_level):
      A = 0.5
      B = 0.7
      return A + B * noise_level


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
print(f"The zero-noise extrapolation is: {zero_noise_limit:.3}")