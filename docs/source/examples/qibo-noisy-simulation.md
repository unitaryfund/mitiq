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

For simplicity, we will use a single-qubit circuit with ten Pauli $X$ gates that compiles to the identity, defined below.

```{code-cell} ipython3
from qibo import Circuit,gates

c = Circuit(1) 
for _ in range(10): 
    c.add(gates.X(0))
```

In this example, we will use the probability of obtaining the $|0\rangle$ state as our observable to mitigate, the expectation value of which should evaluate to one in the noiseless setting.

## Setup: Defining the executor 

We define the executor function in the following code block. In the executor, we initially make a deep copy of the input circuit and modify it to add the measurement gates. We then create a noise map and apply it to the circuit. Finally we simulate the noisy circuit and obtain the desired observable as output of the executor function. For more detailed information about the noise map features see [Qibo noisy simulation](<https://qibo.science/qibo/stable/code-examples/advancedexamples.html#adding-noise-after-every-gate>).  

```{code-cell} ipython3
import copy 

def executor(circuit, shots = 1000):
    """Returns the expectation value to be mitigated. 
    In this case the expectation value is the probability to get the |0> state. 

    Args:
        circuit: Circuit to run.
        shots: Number of times to execute the circuit to compute the expectation value.
    """
    circuit_copy = copy.deepcopy(circuit) 
    circuit_copy.add(gates.M(0)) 

    # Apply noisy map (simulate noisy backend)
    noise_map = {0: list(zip(["X", "Z"], [0.03, 0.03]))}
    noisy_c = circuit_copy.with_pauli_noise(noise_map)
    result = noisy_c(nshots=shots)
    result_freq = result.frequencies(binary=True)
    counts_0 = result_freq.get("0")
     
    if counts_0 is None:
        expectation_value = 0.
    else:
        expectation_value = counts_0 / shots  
    return expectation_value
```

## Applying ZNE

We can now test the mitigated version of the circuit against the unmitigated one to ensure it is working as expected.

```{code-cell} ipython3
from mitiq import zne

unmitigated = executor(c) 
print(f"Unmitigated result {unmitigated:.3f}")
mitigated = zne.execute_with_zne(c, executor)
print(f"Mitigated result {mitigated:.3f}")
```
