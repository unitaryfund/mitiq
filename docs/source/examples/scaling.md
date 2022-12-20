---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Noise Scaling Methods

In this tutorial we will compare two noise scaling methods available for use in [Zero-Noise Extrapolation](https://mitiq.readthedocs.io/en/stable/guide/zne.html) (ZNE): identity insertion and unitary folding.
ZNE works by running multiple versions of the desired circuit, each intended to scale the noise up from the base-level achieved by the hardware.
Experimentally these experiments are often performed by pulse stretching, but as a quantum programmer, we typically do not have access to such low-level control.
For this reason, we use "digital" methods that allow us to scale the noise using gate-based methods.
To this end, we will study circuit folding and identity insertion as methods to increase the amount of noise present in our computation.

These techniques are summarized by the following equations, and can be performed in Mitiq with the associated functions.

|                | Folding              | Identity Insertion |
| -------------- | -------------------- | ------------------ |
| Equation       | $G \to GG^\dagger G$ | $G \to I G$        |
| Mitiq Function | [`fold_global`](https://mitiq.readthedocs.io/en/stable/apidoc.html#mitiq.zne.scaling.folding.fold_global) | [`insert_id_layers`](https://mitiq.readthedocs.io/en/stable/apidoc.html#mitiq.zne.scaling.identity_insertion.insert_id_layers) |

## Comparison

To get started, we can demo what these two functions do to a small GHZ circuit.
Each function (`fold_global` and `insert_id_layers`) will take a circuit, and a specified scale factor as inputs.
The argument `scale_factor` controls how much to increase the depth of the input circuit so that the achieved scale factor is exactly equal, or very close, to the specified scale factor.

```{code-cell} ipython3
from mitiq.benchmarks import generate_ghz_circuit
from mitiq.zne.scaling import insert_id_layers, fold_global

demo = generate_ghz_circuit(3)
scale_factor = 3

print("-----ORIGINAL-----")
print(demo)
print("\n-----------------FOLDING------------------")
print(fold_global(demo, scale_factor))
print("\n-----------------SCALING------------------")
print(insert_id_layers(demo, scale_factor))
```

Theoretically, these circuits should give the same result when measured, but due to noise, this is almost never the case.
Both methods work by extending the duration of the circuit, but do so in different ways that might be beneficial for different scenarios.
When using folding, noise is amplified by applying additional gates, and in particular inverse gates.
Scaling amplifies noise by letting the qubits idle for longer _between_ computation.

```{warning}
Unitary folding scales noise by applying an additional layer $G^\dagger G$ to the circuit.
For non-hermitian gates $G$ and $G^\dagger$ may not have the same noise model, and hence noise is potentially scaled in an non-linear way.

Similarly, the noise that predominantly scaled in identity insertion is that of idle qubit noise/decoherence.
```

Let's now look at how much the depth of these circuit increase with different scale factors.

```{code-cell} ipython3
print(
    "{: >12}  {: ^14} {: ^14} {: ^15}".format(
        "scale factor", "original depth", "folded depth", "id insertion depth"
    )
)
for scale_factor in range(1, 10):
    folded_depth = len(fold_global(demo, scale_factor))
    id_insert_depth = len(insert_id_layers(demo, scale_factor))
    print(
        "{: >12}  {: ^14} {: ^14} {: ^15}".format(
            scale_factor, len(demo), folded_depth, id_insert_depth
        )
    )
```

As expected, we have $\mathtt{depth} * \mathtt{scale\_factor} = \mathtt{scaled\_depth}$ when the scale factor is an integer.
The scale factor can also take on non-integer values, where this equation will hold approximately.

## Using noise scaling methods

Here, we demo how you can use these noise-scaling technique in ZNE.
First, we define an [executor](https://mitiq.readthedocs.io/en/stable/guide/executors.html) which is needed to tell Mitiq how to run the circuit.
We choose depolarizing noise via a simple density matrix simulation to act between every circuit layer.

```{code-cell} ipython3
from mitiq.zne import execute_with_zne
import cirq

def execute(circuit, noise_level=0.05):
    noisy_circuit = circuit.with_noise(cirq.depolarize(p=noise_level))
    return (
        cirq.DensityMatrixSimulator()
        .simulate(noisy_circuit)
        .final_density_matrix[0, 0]
        .real
    )
```

We can then pass the desired noise-scaling method into `execute_with_zne` using the `scale_noise` keyword argument.

```{code-cell} ipython3
execute_with_zne(generate_ghz_circuit(6), execute, scale_noise=insert_id_layers)
```

To give a slightly more systematic understanding of the differences between these two methods, we will perform a small experiment on the following variational circuit.

```{code-cell} ipython3
def variational_circuit(gamma):
    q0, q1 = cirq.LineQubit.range(2)

    return cirq.Circuit(
        [
            cirq.rx(gamma)(q0),
            cirq.CNOT(q0, q1),
            cirq.rx(gamma)(q1),
            cirq.CNOT(q0, q1),
            cirq.rx(gamma)(q0),
        ]
    )
```

We can run this circuit many times, varying $\gamma$ each time, and compute ideal and noisy expectation values to compare to the mitigated versions.
We do this comparison by using an improvement factor (IF) which is defined as
```{math}
\left|\frac{\text{ideal} - \text{noisy}}{\text{ideal} - \text{mitigated}}\right|.
```

```{code-cell} ipython3
from random import uniform
import numpy as np

results = {"fold": [], "id": []}
for _ in range(100):
    gamma = uniform(0, 2 * np.pi)
    circuit = variational_circuit(gamma)

    ideal_expval = execute(circuit, noise_level=0.0)
    noisy_expval = execute(circuit)
    folded_expval = execute_with_zne(circuit, execute, scale_noise=fold_global)
    id_expval = execute_with_zne(circuit, execute, scale_noise=insert_id_layers)

    noisy_error = abs(ideal_expval - noisy_expval)
    folded_IF = noisy_error / abs(ideal_expval - folded_expval)
    scaled_IF = noisy_error / abs(ideal_expval - id_expval)

    results["fold"].append(folded_IF)
    results["id"].append(scaled_IF)

print("Avg improvement factor (`fold_global`):      ", round(np.average(results["fold"]), 4))
print("Avg improvement factor (`insert_id_layers`): ", round(np.average(results["id"]), 4))
```

As we can see, both techniques offer an improvement.


## Conclusion

In this tutorial, we've shown how to use both folding and identity insertion as noise scaling methods for Zero-Noise Extrapolation.
If you're interested in finding out more about these techniques, check out our [Noise Scaling Functions](https://mitiq.readthedocs.io/en/stable/guide/zne-3-options.html#noise-scaling-functions) section of our users guide!
