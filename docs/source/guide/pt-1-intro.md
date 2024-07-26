---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# How do I use PT?

```{admonition} Warning:
Pauli Twirling in Mitiq is still under construction. This users guide will change in the future
after some utility functions are introduced. 
```

As with all techniques, PT is compatible with any frontend supported by Mitiq:

```{code-cell} ipython3
import mitiq

mitiq.SUPPORTED_PROGRAM_TYPES.keys()
```

## Problem setup
We first define the circuit of interest, which contains Hadamard, C-NOT, and C-Z gates.

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

Next we define a simple executor function which inputs a circuit, executes
the circuit on a noisy simulator, and returns the probability of the ground
state. See the [Executors](executors.md) section for more information on
how to define more advanced executors.

As per the noise model that is applied during execution by the simulator, 
we choose a depolarizing channel applied to the output of each 2-qubit gates.
Admittedly, this is not necessarily the most realistic noise,
but for the sake of this tutorial, it is going to be useful for highlighting 
the effect of Pauli Twirling.  

```{code-cell} ipython3
from cirq import CZPowGate, CXPowGate, CircuitOperation, depolarize, DensityMatrixSimulator
from cirq.devices.noise_model import GateSubstitutionNoiseModel

def get_noise_model(noise_level: float) -> GateSubstitutionNoiseModel:
    """Substitute each C-Z and C-NOT gate in the circuit 
    with the gate itself followed by a depolarizing channel
    """
    def noisy_c_gate(op):
        if isinstance(op.gate, (CZPowGate, CXPowGate)):
            return CircuitOperation(
                Circuit(
                    op.gate.on(*op.qubits), 
                    depolarize(p=noise_level, n_qubits=2).on_each(op.qubits)
                ).freeze())
        return op

    return GateSubstitutionNoiseModel(noisy_c_gate)

def execute(circuit: Circuit, noise_level: float):
    """Returns Tr[ρ |0⟩⟨0|] where ρ is the state prepared by the circuit"""
    return (
        DensityMatrixSimulator(noise=get_noise_model(noise_level=noise_level))
        .simulate(circuit)
        .final_density_matrix[0, 0]
        .real
    )
```

The [executor](executors.md) can be used to evaluate noisy (unmitigated)
expectation values.

```{code-cell} ipython3
# Set the parameter of the depolarizing channel
NOISE_LEVEL = 0.1

# Compute the expectation value of the |0><0| observable.
ideal_value = execute(circuit, noise_level=0.0)
noisy_value = execute(circuit, noise_level=NOISE_LEVEL)

print(f"Error without twirling: {abs(ideal_value - noisy_value) :.3}")
```

## Apply PT
Pauli Twirling can be applied with the function
{func}`.pauli_twirl_circuit()` from the `mitiq.pt` module.

```{code-cell} ipython3
from functools import partial
import numpy as np
from mitiq.executor.executor import Executor
from mitiq.pt import generate_pauli_twirl_variants

# Generate twirled circuits
twirled_circuits = generate_pauli_twirl_variants(circuit)

# Average results executed over twirled circuits
pt_vals = Executor(partial(execute, noise_level=NOISE_LEVEL)).evaluate(twirled_circuits)
mitigated_result = np.average(pt_vals)

print(f"Error with mitigation (PT): {abs(ideal_value - mitigated_result) :.3}")
```

Here we observe that the application of PT does not reduce the estimation error when compared
to the unmitigated result. The intended effect was to only tailor the noise. 

```{admonition} Note:
PT is designed to transform the noise simulated in this example,
but it should not be expected to always be a positive effect.
In this sense, it is more of a noise tailoring technique, designed
to be composed with other techniques rather than an error mitigation
technique in and of itself.
```

+++

The section
[What additional options are available when using PT?](pt-3-options.md)
contains information on more advanced ways of applying PT with Mitiq.
