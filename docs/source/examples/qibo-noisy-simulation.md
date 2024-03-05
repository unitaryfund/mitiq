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

# Error mitigation with Qibo using noisy simulation

In this tutorial we will cover how to use Mitiq in conjunction with [Qibo](https://qibo.science/). Besides, we will simulate a noisy channel by adding a Pauli noise channel after each gate.

- [Error mitigation with Qibo using noisy simulation ](#error-mitigation-with-qibo-using-noisy-simulation)
  - [Setup: Defining a circuit](#setup-defining-a-circuit)
  - [Setup: Defining the executor](#setup-defining-the-executor)
  - [Applying ZNE](#applying-zne)

## Setup: Defining a circuit

For simplicity, we will use a single-qubit circuit with ten Pauli _X_ gates that compiles to the identity, defined below.

```{code-cell} ipython3
from qibo import Circuit,gates

c = Circuit(1) 
for _ in range(10): 
    c.add(gates.X(0))
c.add(gates.M(0))
```

In this example, we will use the probability of obtaining the |0⟩ state as our observable to mitigate, the expectation value of which should evaluate to one in the noiseless setting.

## Setup: Defining the executor 

We define the executor function in the following code block. In the executor, we create a noise map and apply it to the circuit. Finally we simulate the noisy circuit and obtain the desired observable as output of the executor function. For more detailed information about the noise map features see [Qibo noisy simulation](<https://qibo.science/qibo/stable/code-examples/advancedexamples.html#adding-noise-after-every-gate>).  

```{code-cell} ipython3
def executor(circuit, shots = 100):
    """Returns the expectation value to be mitigated. 
    In this case the expectation value is the probability to get the |0> state. 

    Args:
        circuit: Circuit to run.
        shots: Number of times to execute the circuit to compute the expectation value.
    """
    # Apply noisy map (simulate noisy backend)
    noise_map = {0: list(zip(["X", "Z"], [0.03, 0.03]))}
    noisy_c = circuit.with_pauli_noise(noise_map)
    
    result = noisy_c(nshots=shots)
    result_freq = result.frequencies(binary=True)
    counts_0 = result_freq.get(0)
    if counts_0 is None:
        expectation_value = 0.
    else:
        expectation_value = counts_0 / shots  
    return expectation_value
```

## Applying ZNE

We can now test the mitigated version of the circuit against the unmitigated one to ensure it is working as expected. We apply ZNE using 
as scale factors 1, 2 and 3 and using RichardsonFactory. For each scaling factor we average over three circuits. 

```{code-cell} ipython3
from mitiq import zne
from mitiq.zne.inference import RichardsonFactory

unmitigated = executor(c) 
print(f"Unmitigated result {unmitigated:.3f}")
scale_factors = [1.0,2.0,3.0]
factory = RichardsonFactory(scale_factors=scale_factors) #default ZNE configuration
mitigated = zne.execute_with_zne(c, executor, factory = factory, num_to_average = 3)
print(f"Mitigated result {mitigated:.3f}")
```

The mitigated result is noticeably closer to the noiseless result compared to the result without mitigation.
In addition, we can show the interpolation performed: 
```{code-cell} ipython3
import matplotlib.pyplot as plt
factory.plot_fit()
plt.show()
```
