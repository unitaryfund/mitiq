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

```{tags} qibo, zne, basic
```

# Error mitigation with Qibo using noisy simulation

In this tutorial we will cover how to use Mitiq to apply [Zero-Noise Extrapolation](../guide/zne.md) (ZNE) to a quantum program written using [Qibo](https://qibo.science/).
We will demonstrate the use of these two tools by implementing a noisy simulation as random Pauli gates probabilistically applied after each ideal gate.

## Defining a circuit

For simplicity, we will use a single-qubit circuit with ten Pauli _X_ gates that compiles to the identity.
This will give us a simple circuit whose probability of measuring the $|0\rangle$ state at the end, ideally, should be 1.

```{code-cell} ipython3
import os 
os.environ['QIBO_LOG_LEVEL'] = '3' #Supress Qibo INFO messages

from qibo import Circuit, gates

c = Circuit(1)
for _ in range(10):
    c.add(gates.X(0))
c.add(gates.M(0))
```

We will use the probability of measuring the $|0\rangle$ state as our observable to mitigate.

## Defining the executor

To use Mitiq, an [executor](../guide/executors.md) function is required to be defined.
This function abstracts away the logic of _how_ a circuit is executed, allowing Mitiq to only see the circuit, and the value received from the executor (which, in the case of applying ZNE, must be a float).
In the executor, we create a noise map and apply it to the circuit.
We then simulate the noisy circuit and return the measured observable.
For more detailed information about the noise map features see Qibo's documentation on [noisy simulation](https://qibo.science/qibo/stable/code-examples/advancedexamples.html#how-to-perform-noisy-simulation).

```{code-cell} ipython3
def executor(circuit, shots=100):
    """Returns the expectation value to be mitigated.
    In this case the expectation value is the probability to get the |0⟩ state.

    Args:
        circuit: Circuit to run.
        shots: Number of times to execute the circuit to compute the expectation value.
    """
    # noise_map = {qubit_id: [(gate, probability)]}
    noise_map = {0: [('X', 0.03), ('Z', 0.03)]}
    noisy_c = circuit.with_pauli_noise(noise_map)

    result = noisy_c(nshots=shots)
    result_freq = result.frequencies(binary=True)
    counts_0 = result_freq.get('0', 0)

    return counts_0 / shots
```

## Applying ZNE

We can now test the mitigated version of the circuit against the unmitigated one to ensure it is working as expected.
We apply ZNE with scale factors 1, 2 and 3 and using `LinearFactory` (which extrapolates the measured expectation values using a linear extrapolation).
For each scaling factor we average over three circuits.

```{code-cell} ipython3
from mitiq import zne
from mitiq.zne.inference import LinearFactory

unmitigated = executor(c)
print(f"Unmitigated result {unmitigated:.3f}")

scale_factors = [1, 2, 3]
factory = LinearFactory(
    scale_factors=scale_factors
)  # default ZNE configuration
mitigated = zne.execute_with_zne(c, executor, factory=factory, num_to_average=3)
print(f"Mitigated result {mitigated:.3f}")
```

The mitigated result is noticeably closer to the noiseless result compared to the result without mitigation.
In addition, Mitiq offers the capability to view the extrapolation directly from the `RichardsonFactory` object.

```{code-cell} ipython3
import matplotlib.pyplot as plt
factory.plot_fit()
plt.show()
```
