---
jupytext:
  text_representation:
    extension: .myst
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# What happens when I use ZNE?

In Mitiq, ZNE is clearly divided into two steps, noise scaling and extrapolation. They are shown in the Figure below.
The corresponding sub-modules in the codebase are {mod}`mitiq.zne.scaling.folding` and {mod}`mitiq.zne.inference`.

```{figure} ../img/zne_workflow2_steps.png
---
width: 400
name: figzne
---
The diagram shows the workflow of the zero noise extrapolation (ZNE) technique in Mitiq.
```

**The first step** involves generating and executing noise-scaled quantum circuits.
  - The user provides a `QPROGRAM`, i.e., a quantum circuit defined via any of the supported frontends.
  - Mitiq generates a set of noise-scaled circuits by applying unitary folding with different scale factors.
  - The noise-scaled circuits are executed on the noisy backend obtaining a set of noise-scaled expectation values.

**The second step** involves inferring the zero-noise value from the measured results.
  - A parametrized curve is fit to the noise-scaled expectation values.
  - The curve is extrapolated to the zero-noise limit, obtaining an error mitigated expectation value.

As demonstrated in [How do I use ZNE ?](zne-1-intro.myst), the function {func}`.execute_with_zne()` applies both steps behind the scenes.
In the next sections instead, we show how one can apply ZNE at a lower level, i.e., by applying each step independently.

Moreover, we will also show how the user can customize noise scaling methods and factory objects.

+++

## First step: generating and executing noise-scaled circuits

+++

### Problem setup
We define a circuit and an executor, as shown in [How do I use ZNE?](zne-1-intro.myst).

```{code-cell} ipython3
from mitiq import benchmarks
from cirq import DensityMatrixSimulator, depolarize

circuit = benchmarks.generate_rb_circuits(
  n_qubits=1, num_cliffords=2, return_type="cirq",
)[0]


def execute(circuit, noise_level=0.01):
    """Returns Tr[ρ |0⟩⟨0|] where ρ is the state prepared by the circuit
    executed with depolarizing noise.
    """
    noisy_circuit = circuit.with_noise(depolarize(p=noise_level))
    rho = DensityMatrixSimulator().simulate(noisy_circuit).final_density_matrix
    return rho[0, 0].real
```

### Evaluate noise-scaled expectation values

We first apply local unitary folding to generate a sequence of noise-scaled circuits.

```{code-cell} ipython3
from mitiq import zne

# Choose a list of scale factors
scale_factors = [1.0, 3.0, 5.0]
# Generate a list of folded circuits
noise_scaled_circuits = [zne.scaling.fold_gates_at_random(circuit, s) for s in scale_factors]
```

For each noise-scaled circuit we evaluate the associated expectation value.

```{code-cell} ipython3
expectation_values = [execute(circ) for circ in noise_scaled_circuits]
```

## Second step: extrapolating the zero-noise limit

+++

The simplest way to extrapolate the zero-noise limit of an expectation value is to use the static method {meth}`.Factory.extrapolate()` of a {class}`.Factory` object. For example, an exponential extrapolation (assuming an infinite noise limit of $0.5$) can be obtained as follows.

```{code-cell} ipython3
zne.ExpFactory.extrapolate(scale_factors, expectation_values, asymptote=0.5)
```

Alternatively, one can also instantiate a `Factory` object, which can be useful for additional analysis and visualization of the measured data.
```{code-cell} ipython3
# Initialize a factory
fac = zne.inference.ExpFactory(scale_factors, asymptote=0.5)

# Load data:
for s, e in zip(scale_factors, expectation_values):
    fac.push({"scale_factor": s}, e)

# Evaluate the extrapolation
fac.reduce()
```

```{code-cell} ipython3
# Plot the extrapolation fit
_ = fac.plot_fit()
```

## Custom noise-scaling methods

Custom folding methods can be defined and used with Mitiq, e.g., with {py:func}`.execute_with_zne`. The signature of this function must be as follows:

```{code-cell} ipython3
import cirq
from mitiq.interface.conversions import atomic_converter

@atomic_converter
def my_custom_folding_function(circuit: cirq.Circuit, scale_factor: float) -> cirq.Circuit:
    # Insert custom folding method here
    folded_circuit = circuit  # Trivial example for testing
    return folded_circuit
```

:::{note}
The {func}`.atomic_converter` decorator makes it so `my_custom_folding_function`
can be used with any supported circuit type, not just Cirq circuits.
The body of the `my_custom_folding_function` should assume the input
circuit is a Cirq circuit, however.
:::

This function can then be used with `.zne.execute_with_zne` as an option to scale the noise:

```{code-cell} ipython3
zne.execute_with_zne(circuit, execute, scale_noise=my_custom_folding_function)
```

## Custom extrapolation methods

+++

If necessary, the user can modify existing extrapolation methods by subclassing
one of the [built-in factories](zne-3-options.myst#extrapolation-methods-factory-objects).

Alternatively, a custom adaptive extrapolation method can be derived from the abstract class {class}`.AdaptiveFactory`.
In this case its core methods must be implemented:
{meth}`.AdaptiveFactory.__init__`, {meth}`.AdaptiveFactory.next`, {meth}`.AdaptiveFactory.is_converged`, {meth}`.AdaptiveFactory.reduce`.



A new non-adaptive method can instead be derived from the abstract {class}`.BatchedFactory` class.
In this case it is usually sufficient to override only the {meth}`.BatchedFactory.__init__` and
the {meth}`.BatchedFactory.extrapolate` methods, which are responsible for the initialization and for the
final zero-noise extrapolation, respectively. An example of a simple custom non-adaptive {class}`.Factory`
is given in the next code cell.

```{code-cell} ipython3
from mitiq.zne.inference import BatchedFactory, LinearFactory
import numpy as np

class MyFactory(BatchedFactory):
    """Factory object implementing a linear extrapolation taking
    into account that the expectation value must be within a given
    interval. If the zero-noise limit falls outside the
    interval, its value is clipped.
    """

    def __init__(self, scale_factors, min_expval, max_expval):
       """
       Args:
          scale_factors: The noise scale factors at which
                         expectation values should be measured.
          min_expval: The lower bound for the expectation value.
          min_expval: The upper bound for the expectation value.
       """
       super(MyFactory, self).__init__(scale_factors)
       self._options = {"min_expval": min_expval, "max_expval": max_expval}

    @staticmethod
    def extrapolate(
       scale_factors, exp_values, min_expval, max_expval, full_output = False,
    ):
        """Fit a linear model and clip its zero-noise limit."""
        # Perform standard linear extrapolation
        result = LinearFactory.extrapolate(scale_factors, exp_values, full_output)
        # Return the clipped zero-noise extrapolation.
        if not full_output:
           return np.clip(result, min_expval, max_expval)
        else:
           # In this case "result" is a tuple of extrapolation data
           zne_limit = np.clip(result[0], min_expval, max_expval)
           return (zne_limit, *result[1:])
```

Using `MyFactory` as an option for {func}`.execute_with_zne`, the customized extrapolation is applied.

```{code-cell} ipython3
# Test MyFactory clips the result as expected
fac = MyFactory([1, 2, 3], min_expval=0.0, max_expval=0.5)
zne_limit_clipped = zne.execute_with_zne(circuit, execute, factory=fac)
assert zne_limit_clipped == 0.5
```

After defining a custom {class}`.Factory`, we suggest to check that all the methods inherited from the parent class run without errors.

```{code-cell} ipython3
# Test parent methods run without errors
fac.get_expectation_values()
fac.get_extrapolation_curve()
fac.get_optimal_parameters()
fac.get_parameters_covariance()
fac.get_scale_factors()
fac.get_zero_noise_limit_error()
zne_limit_clipped = fac.get_zero_noise_limit()
```

### Regression tools in `zne.inference`

In the body of the previous `MyFactory` example, we imported and used the {py:func}`.mitiq_polyfit` function.
This is simply a wrap of {py:func}`numpy.polyfit`, slightly adapted to the notion and to the error types
of Mitiq. This function can be used to fit a polynomial ansatz to the measured expectation values. This function performs
a least squares minimization which is **linear** (with respect to the coefficients) and therefore admits an algebraic solution.

Similarly, from {py:mod}`mitiq.zne.inference` one can also import {py:func}`.mitiq_curve_fit`,
which is instead a wrap of {py:func}`scipy.optimize.curve_fit`. Differently from {py:func}`.mitiq_polyfit`,
{py:func}`.mitiq_curve_fit` can be used with a generic (user-defined) ansatz.
Since the fit is based on a numerical **non-linear** least squares minimization, this method may fail to converge
or could be subject to numerical instabilities.

+++

## Low-level usage of a factory

In this section we present a low-level usage of a {class}`.Factory` . In typical use-cases, the following information is not necessary but it can be useful for understanding how {class}`.Factory` objects work under the hood.

+++

### The `run` method.

+++

The {meth}`.Factory.run` method can be used to run all the data acquisition steps associated to a {class}`.Factory`, until enough data is collected for the extrapolation.

```{code-cell} ipython3
fac = zne.inference.AdaExpFactory(steps=5, asymptote=0.5)
# Run the factory to collect data
fac.run(circuit, execute)
# Extrapolate
fac.reduce()
```

### The `run_classical` method.

Instead of {meth}`.Factory.run`, the {meth}`.Factory.run_classical` method can be used if we have at disposal a function which directly
maps a noise scale factor to the corresponding expectation value.

```{code-cell} ipython3
def noise_to_expval(scale_factor: float) -> float:
    """Function returning an expectation value for a given scale_factor."""
    scaled_circuit = zne.scaling.fold_gates_at_random(circuit, scale_factor)
    return execute(scaled_circuit)
```

```{code-cell} ipython3
# Remove internal data, if present.
fac.reset()
# Run the factory to collect data
fac.run_classical(noise_to_expval)
# Extrapolate
fac.reduce()
```

The {meth}`.Factory.run_classical` applies ZNE as a fully classical inference problem.
Indeed, all the quantum aspects of the problem (circuit, observable, backend, etc.) are
wrapped into the `noise_to_expval()` function.
