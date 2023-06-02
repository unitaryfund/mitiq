---
jupytext:
  text_representation:
    extension: .myst
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# How do I use PT?

As with all techniques, PT is compatible with any frontend supported by Mitiq:

```{code-cell} ipython3
import mitiq

mitiq.SUPPORTED_PROGRAM_TYPES.keys()
```


## Problem setup
We first define the circuit of interest. In this example, the circuit has 
two CNOT gates and a CZ gate. We can see that when we apply Pauli Twirling,
we will generate 

```{code-cell} ipython3
from cirq import LineQubit, Circuit, CZ, CNOT

a, b, c, d  = LineQubit.range(4)
circuit = Circuit(
    CNOT.on(a, b),
    CZ.on(b, c),
    CNOT.on(c, d),
)

print(circuit)
```

Next we define a simple executor function which inputs a circuit, executes
the circuit on a noisy simulator, and returns the probability of the ground
state. See the [Executors](executors.md) section for more information on
how to define more advanced executors.

```{code-cell} ipython3
import numpy as np
from cirq import DensityMatrixSimulator, amplitude_damp
from mitiq.interface import convert_to_mitiq

def execute(circuit, noise_level=0.1):
    """Returns Tr[ρ |0⟩⟨0|] where ρ is the state prepared by the circuit
    executed with amplitude damping noise.
    """
    # Replace with code based on your frontend and backend.
    mitiq_circuit, _ = convert_to_mitiq(circuit)
    noisy_circuit = mitiq_circuit.with_noise(amplitude_damp(gamma=noise_level))
    rho = DensityMatrixSimulator().simulate(noisy_circuit).final_density_matrix
    return rho[0, 0].real
```

The [executor](executors.md) can be used to evaluate noisy (unmitigated)
expectation values.

```{code-cell} ipython3
# Compute the expectation value of the |0><0| observable.
noisy_value = execute(circuit)
ideal_value = execute(circuit, noise_level=0.0)
print(f"Error without mitigation: {abs(ideal_value - noisy_value) :.3}")
```

## Apply PT
Pauli Twirling can be easily implemented with the function
{func}`.execute_with_pt()`.

```{code-cell} ipython3
from mitiq import pt
mitigated_result = pt.execute_with_pt(
    circuit=circuit,
    executor=execute,
)
```

```{code-cell} ipython3
print(f"Error with mitigation (PT): {abs(ideal_value - mitigated_result) :.3}")
```

Here we observe that the application of PT reduces the estimation error when compared
to the unmitigated result.

```{admonition} Note:
PT is designed to transform the noise simulated in this example,
but it should nnot be expected to always be a positive effect.
In this sense, it is more of a noise tailoring technique, designed
to be composed with other techniques rather than an error mitigation
technique in and of itself.
```

+++

The section
[What additional options are available when using PT?](pt-3-options.md)
contains information on more advanced ways of applying PT with Mitiq.
