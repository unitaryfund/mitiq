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


In this tutorial we will cover how to use Mitiq in conjunction with [Qibo](https://qibo.science/). Besides, we will simulate a noisy channel by adding pauli gates with a certain probability after each gate.

- [Error mitigation with Qibo using noisy simulation ](#error-mitigation-with-qibo-using-noisy-simulation)
  - [Setup: Defining a circuit](#setup-defining-a-circuit)
  - [Setup: Defining the executor](#setup-defining-the-executor)
  - [Applying ZNE](#applying-zne)
  - [Decorator usage](#decorator-usage)

+++

(examples/qibo-noisy-simulation/setup-defining-a-circuit)=
## Setup: Defining a circuit

+++

We'll use a two-qubit circuit which creates the maximally entangled state $\frac{|00\rangle+|11\rangle}{\sqrt{2}}$ and then adds four  $X$ gates to the first qubit that are equivalent to the identity. The circuit is defined below: 

```{code-cell} ipython3
from qibo import Circuit,gates

c = Circuit(2) 
c.add(gates.H(0)) 
c.add(gates.CNOT(0,1))
for _ in range(4): 
    c.add(gates.X(0))
```

In this example, we will use the probability of obtaining the $|00\rangle$ state as our observable to mitigate, the expectation value of which should evaluate to one half in the noiseless setting.

+++

## Setup: Defining the executor 

We define the executor function in the following code block. In the executor, we initially make a deep copy of the input circuit and modify it to add the measurement gates. We then create a noise map and apply it to the circuit. Finally we simulate the noisy circuit and obtain the desired observable as output of the executor function. For more detailed information about the noise map features see [Qibo noisy simulation](<https://qibo.science/qibo/stable/code-examples/advancedexamples.html#adding-noise-after-every-gate>).  

```{code-cell} ipython3
import copy

def executor(circuit, shots = 1000):
    """Returns the expectation value to be mitigated. 
    In this case the expectation value is the probability to get the |00> state. 

    Args:
        circuit: Circuit to run.
        shots: Number of times to execute the circuit to compute the expectation value.
    """
    circuit_copy=copy.deepcopy(circuit) 
    circuit_copy.add(gates.M(0, 1)) 

    # Apply noisy map (simulate noisy backend)
    noise_map = {0: list(zip(["X", "Z"], [0.02, 0.02])), 1: list(zip(["X", "Z"], [0.02, 0.02]))}
    noisy_c = circuit_copy.with_pauli_noise(noise_map)

    result = noisy_c(nshots=shots)
    result_freq=result.frequencies(binary=True)
    counts_00 = result_freq.get("00")
     
    if counts_00 is None:
        expectation_value = 0.
    else:
        expectation_value = counts_00 / shots  
    return expectation_value
```

## Applying ZNE

We can now test the mitigated version of the circuit against the unmitigated to ensure it is working as expected.

```{code-cell} ipython3
unmitigated = device_circuit()
mitigated = error_mitigated_device_circuit()
print(f"Unmitigated result {unmitigated:.3f}")
print(f"Mitigated result   {mitigated:.3f}")
```

As the ideal, desired result is `1.000`, the mitigated result performs much better than unmitigated.

In this section we will discuss the many different options for noise scaling and extrapolation that can be passed into PennyLane's `mitigate_with_zne` function.

The following code block shows an example of using linear extrapolation with five different (noise) scale factors.

```{code-cell} ipython3
from mitiq.zne.inference import LinearFactory

scale_factors = [1.0, 1.5, 2.0, 2.5, 3.0]
noise_scale_method = fold_global
mitigated = qml.transforms.mitigate_with_zne(
    device_circuit,
    scale_factors,
    noise_scale_method,
    LinearFactory.extrapolate, 
)()

print(f"Mitigated result {mitigated:.3f}")
```

To specify a different noise scaling method, we can pass a different function for the argument ``scale_noise``.
This function should input a circuit and scale factor and return a circuit.
The following code block shows an example of scaling noise by local folding instead of global folding.

```{code-cell} ipython3
from mitiq.zne.scaling import fold_gates_at_random

scale_factors = [1.0, 1.5, 2.0, 2.5, 3.0]
noise_scale_method = fold_gates_at_random
mitigated = qml.transforms.mitigate_with_zne(
    device_circuit,
    scale_factors,
    noise_scale_method,
    LinearFactory.extrapolate, 
)()

print(f"Mitigated result {mitigated:.3f}")
```

Further options are described and elaborated in our article on [additional options](../guide/zne-3-options.md) in ZNE.


## Decorator usage

Finally, it is perhaps more common to define a circuit using a decorator when you know in advance you would like an error mitigated value.
For this our circuit will be defined as above, but we will use decorators to indicate which device we would like to run it on, and that we would like to error-mitigate it.

```{code-cell} ipython3
from mitiq.zne.scaling import fold_gates_from_left

@qml.transforms.mitigate_with_zne([1, 2, 3], fold_gates_from_left, RichardsonFactory.extrapolate)
@qml.qnode(dev)
def circuit():
    for _ in range(10):
        qml.PauliX(wires=0)
    return qml.expval(qml.PauliZ(0))

print(f"Zero-noise extrapolated value: {circuit():.3f}")
```

Finally, more information about using PennyLane together with Mitiq can be found in PennyLane's [tutorial](https://pennylane.ai/qml/demos/tutorial_error_mitigation.html) on error mitigation.
