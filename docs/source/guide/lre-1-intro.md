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


# How do I use LRE?


LRE works in two steps: generate noise-scaled circuits and apply inference to results from executed circuits.


A user has the choice to either use {func}`.execute_with_lre` to combine both steps into one if they are
only interested in obtaining the mitigated expectation value or splitting the process into two using
{func}`.multivariate_layer_scaling` and {func}`.multivariate_richardson_coefficients`.


```{warning}
LRE is currently compatible with quantum programs written using `cirq`. Work on making this technique compatible with other frontends is ongoing. 🚧
```


## Problem Setup


To use {func}`.execute_with_lre` without any additional options the following are required:


- a quantum circuit
- a method of returning an expectation value from a circuit
- the degree of the multivariate polynomial extrapolation
- fold multiplier AKA the scaling gap which is used to generate the scale factor vectors


### Define the circuit of interest


For simplicity, we define a simple circuit whose ideal execution is identical to the identity operation.


```{code-cell} ipython3
from mitiq import benchmarks


circuit = benchmarks.generate_rb_circuits(n_qubits=1, num_cliffords=3)[0]


print(circuit)
```


### Define executor for ideal and noisy executions


We define an [executor](executors.md) which executes the input circuit subjected to depolarizing noise, and returns the probability of the ground state. By altering the value for `noise_level`, ideal and noisy expectation
values can be obtained.


```{code-cell} ipython3
import numpy as np
from cirq import DensityMatrixSimulator, depolarize


def execute(circuit, noise_level=0.025):
   """Default executor for all unit tests."""
   noisy_circuit = circuit.with_noise(depolarize(p=noise_level))
   rho = DensityMatrixSimulator().simulate(noisy_circuit).final_density_matrix
   return rho[0, 0].real
```


Compare the noisy and ideal expectation values:


```{code-cell} ipython3
# Compute the expectation value of the |0><0| observable.
noisy_value = execute(circuit)
ideal_value = execute(circuit, noise_level=0.0)
print(f"Error without mitigation: {abs(ideal_value - noisy_value) :.5f}")
```


## Apply LRE directly


With the circuit, and executor defined, we just need to choose the polynomial extrapolation degree as well as the fold multiplier.


```{code-cell} ipython3
from mitiq.lre import execute_with_lre


input_degree = 2
input_fold_multiplier = 3


mitigated_result = execute_with_lre(
    circuit,
    execute,
    degree = input_degree,
    fold_multiplier = input_fold_multiplier,
)


print(f"Error with mitigation (ZNE): {abs(ideal_value - mitigated_result):.{3}}")
```


## Step by step application of LRE

In this section we will walk through what happens in each of the two stages of LRE.


### Create noise-scaled circuits


We start with creating a number of noise-scaled circuits which we will pass to the executor.


```{code-cell} ipython3
from mitiq.lre import multivariate_layer_scaling


noise_scaled_circuits = multivariate_layer_scaling(circuit, input_degree, input_fold_multiplier)


print(f"total number of noise-scaled circuits for LRE = {len(noise_scaled_circuits)}")
```
An example noise-scaled circuit is shown below:


```{code-cell} ipython3
noise_scaled_circuits[3]
```


### Classical inference


Based on the choice of input parameters, a sample matrix created using {func}`.sample_matrix` is used to find the
coefficients of linear combination required for multivariate Richardson extrapolation (**link theory section here**).


```{code-cell} ipython3
from mitiq.lre import multivariate_richardson_coefficients

coeffs_of_linear_comb = multivariate_richardson_coefficients(
    circuit,
    fold_multiplier = input_fold_multiplier,
    degree = input_degree,
)


print(f"total number of noise-scaled circuits for LRE = {len(noise_scaled_circuits)}")
print(f"total number of coefficients of linear combination for LRE = {len(coeffs_of_linear_comb)}")
```
Each noise scaled circuit has a coefficient of linear combination and a noisy expectation value associated with it.


### Combine the results


```{code-cell} ipython3
## execute each noise scaled circuit

noise_scaled_exp_values = []


for i in noise_scaled_circuits:
 noise_scaled_exp_values.append(execute(i))


calculated_mitigated_result = np.dot(noise_scaled_exp_values, coeffs_of_linear_comb)
print(f"Error with mitigation (ZNE): {abs(ideal_value - calculated_mitigated_result):.{3}}")
```
The section [](lre-3-options.md) contains more information on other options available in LRE in addition to how to
control the hyperparameters associated with the LRE options.
