---
jupytext:
 text_representation:
   extension: .md
   format_name: myst
   format_version: 0.13
   jupytext_version: 1.11.1
kernelspec:
 display_name: Python 3
 language: python
 name: python3
---


# What happens when I use LRE?

As shown in the figure below, LRE works in two steps, layerwise noise scaling and extrapolation.

The noise-scaled circuits are
created through the functions in {mod}`mitiq.lre.multivariate_scaling.layerwise_folding` while the error-mitigated expectation value is estimated by using the functions in {mod}`mitiq.lre.inference.multivariate_richardson`.

```{figure} ../img/lre_workflow_steps.png
---
width: 700px
name: lre-overview2
---
The diagram shows the workflow of the layerwise Richardson extrapolation (LRE) in Mitiq.
```


**The first step** involves generating and executing layerwise noise-scaled quantum circuits.
  - The user provides a `QPROGRAM` i.e. a frontend supported quantum circuit .

  - Mitiq generates a set of layerwise noise-scaled circuits by applying unitary folding based on a set of pre-determined scale factor vectors. 
  - The noise-scaled circuits are executed on the noisy backend obtaining a set of noisy expectation values.

**The second step** involves inferring the error mitigated expectation value from the measured results through multivariate richardson extrapolation.

The function {func}`.execute_with_lre` accomplishes both steps behind the scenes to  estimate the error mitigate expectation value. Additional information is available in [](lre-1-intro.md).

If one were interested in applying each step individually, the following sections demonstrate how a user can do so.

## First step: generating and executing noise-scaled circuits

### Problem setup

We'll first define a quantum circuit, and a method of executing circuits for demonstration purposes.

For simplicity, we define a circuit whose unitary compiles to the identity operation.
Here we will use a randomized benchmarking circuit on a single qubit, visualized below.

```{code-cell} ipython3
from cirq import LineQubit, Circuit, CZ, CNOT, H


q0, q1, q2, q3  = LineQubit.range(4)
circuit = Circuit(
   H(q0),
   CNOT.on(q0, q1),
   CZ.on(q1, q2),
   CNOT.on(q2, q3),
)


print(circuit)
```

We define an [executor](executors.md) which simulates the input circuit subjected to depolarizing noise, and returns the probability of measuring the ground state.
By altering the value for `noise_level`, ideal and noisy expectation values can be obtained.

```{code-cell} ipython3
from cirq import DensityMatrixSimulator, depolarize


def execute(circuit, noise_level=0.025):
    noisy_circuit = circuit.with_noise(depolarize(p=noise_level))
    rho = DensityMatrixSimulator().simulate(noisy_circuit).final_density_matrix
    return rho[0, 0].real
```

Compare the noisy and ideal expectation values:

```{code-cell} ipython3
noisy = execute(circuit)
ideal = execute(circuit, noise_level=0.0)
print(f"Error without mitigation: {abs(ideal - noisy) :.5f}")
```

### Create noise-scaled circuits

We start by creating a number of noise-scaled circuits which we will pass to the executor. {func}`.get_scale_factor_vectors` determines the scale factor vectors while {func}`.multivariate_layer_scaling` creates the noise scaled circuits. 

Each noise-scaled circuit is scaled based on some pre-determined values of the scale factor vectors. These vectors depend
on the degree of polynomial chosen for multivariate extrapolation and the fold multiplier for unitary folding.

```{code-cell} ipython3
import numpy as np
from mitiq.lre.multivariate_scaling import get_scale_factor_vectors

degree = 2
fold_multiplier = 2

scale_factors = get_scale_factor_vectors(circuit, degree, fold_multiplier)

# print a randomly chosen scale factor vector

print(f"Example scale factor vector: {scale_factors[2]}")

```
Each value in the scale factor vector corresponds to how unitary folding is applied to a layer in the circuit. If a scale factor value at some position in the scale factor vector is

- greater than 1, then unitary folding is applied to the layer at that position in the circuit.
- 1, then the layer at that position in the circuit is not scaled.

```{code-cell} ipython3
from mitiq.lre.multivariate_scaling import multivariate_layer_scaling

noise_scaled_circuits = multivariate_layer_scaling(circuit, degree, fold_multiplier)
num_scaled_circuits = len(noise_scaled_circuits)

print(f"Total number of noise-scaled circuits for LRE = {num_scaled_circuits}")
```

An example noise-scaled circuit is shown below:

```{code-cell} ipython3

print("Example noise-scaled circuit:", noise_scaled_circuits[2], sep="\n")
```

### Evaluate noise-scaled expectation values

With the many noise-scaled circuits in hand, we can run them through our executor to obtain the expectation values.

```{code-cell} ipython3
noise_scaled_exp_values = [
    execute(circuit) for circuit in noise_scaled_circuits
]
```


## Second step: Multivariate Extrapolation for the error-mitigated expectation value

The penultimate step here is to fetch the coefficients we'll use to combine the noisy data we obtained above. Each noise scaled circuit has a coefficient of linear combination and a noisy expectation value associated with it. 

```{code-cell} ipython3
from mitiq.lre.inference import multivariate_richardson_coefficients


coefficients = multivariate_richardson_coefficients(
    circuit,
    fold_multiplier=fold_multiplier,
    degree=degree,
)
```
These coefficients are calculated through solving a system of linear equations $\mathbf{A} c = z$, where each row of the sample matrix $\mathbf{A}$ is formed by the [monomial terms](lre-3-options.md) of the multivariate polynommial evaluated using the values in the scale factor vectors, $z$ is the vector of expectation values and $c$ is the coefficients vector.

{func}`.sample_matrix` is used by {func}`.multivariate_richardson_coefficients` behind the scenes to calculate these coefficients as discussed in [](lre-5-theory.md).

For example, if the terms in the monomial basis are given by the following:

$$\{1, λ_1, λ_2, λ_3, λ_4, λ_1^2, λ_1 λ_2, λ_1 λ_3, λ_1 λ_4, λ_2^2, λ_2 λ_3, λ_2 λ_4, λ_3^2, λ_3 λ_4, λ_4^2\}$$

Each row of the sample matrix is defined by these monomial basis terms. Let one of the scale factor vectors be $(1, 5, 1, 1)$. To get to the sample matrix, each row is then evaluated using the scale factor vectors. For the example scale factor vector, a row of the sample matrix will be evaluated using $λ_1=1, λ_2=5, λ_3=1, λ_4=1$.



### Combine the results

The error mitigated expectation value is described as a linear combination of the noisy expectation values where the
coefficients of linear combination were calculated in the preceding section.

```{code-cell} ipython3
mitigated = sum(
    exp_val * coeff
    for exp_val, coeff in zip(noise_scaled_exp_values, coefficients)
)
print(f"Error with mitigation (LRE): {abs(ideal - mitigated):.{3}}")
```