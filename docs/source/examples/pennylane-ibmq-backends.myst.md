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

# Error mitigation with Pennylane on IBMQ backends


In this tutorial we will cover how to use Mitiq in conjunction with [PennyLane](https://pennylane.ai/), and further how to run error-mitigated circuits on IBMQ backends.

- [](#setup-defining-a-circuit)
- [](#high-level-usage)
- [](#options)
- [](#decorator-usage)

+++

(examples/ibmq-backends-pennylane/setup-defining-a-circuit)=
## Setup: Defining a circuit

+++

For simplicity, we'll use a single-qubit circuit with ten Pauli $X$ gates that compiles to the identity, defined below.

```{code-cell} ipython3
import pennylane as qml

def circuit():
    for _ in range(10):
        qml.PauliX(wires=0)
    return qml.expval(qml.PauliZ(0))
```

In this example, we will use the probability of the ground state as our observable to mitigate, the expectation value of which should evaluate to one in the noiseless setting.

+++

## High-level usage

As of version `0.19` of PennyLane, and `0.11` of Mitiq, PennyLane comes with out of the box support for error mitigation.
This makes it very easy to use zero-noise extrapolation when working with PennyLane, regardless of where the circuit is being executed.

We define the executor function in the following code block.
As we are using IBMQ backends, we first load our account.
    
**Note:** Using an IBM quantum computer requires a valid IBMQ account.
See <https://quantum-computing.ibm.com/> for instructions to create an account, save credentials, and get access to online quantum computers.

First, we get our devices set up depending on whether we would like to use real hardware, or a simulator.

```{code-cell} ipython3
import qiskit

USE_REAL_HARDWARE = False

if qiskit.IBMQ.stored_account() and USE_REAL_HARDWARE:
    provider = qiskit.IBMQ.load_account()
    dev = qml.device(
        "qiskit.ibmq",
        wires=1,
        backend="ibmq_qasm_simulator",
        provider=provider
    )
else:
    noise_strength = 0.05
    dev_noise_free = qml.device("default.mixed", wires=1)
    dev = qml.transforms.insert(
        qml.AmplitudeDamping,
        noise_strength
    )(dev_noise_free)
```

With `dev` set to the desired device, we can now use PennyLane's [`mitigate_with_zne`](https://pennylane.readthedocs.io/en/stable/code/api/pennylane.transforms.mitigate_with_zne.html) function, in conjuction with a noise scaling method, and inference technique from Mitiq.

```{code-cell} ipython3
from mitiq.zne.scaling import fold_global
from mitiq.zne.inference import RichardsonFactory

scale_factors = [1, 2, 3]
noise_scale_method = fold_global

device_circuit = qml.QNode(circuit, dev)
error_mitigated_device_circuit = qml.transforms.mitigate_with_zne(
    device_circuit,
    scale_factors,
    noise_scale_method,
    RichardsonFactory.extrapolate, 
)
```

We can now test the mitigated version of the circuit against the unmitigated to ensure it is working as expected.

```{code-cell} ipython3
unmitigated = device_circuit()
mitigated = error_mitigated_device_circuit()
print(f"Unmitigated result {unmitigated:.3f}")
print(f"Mitigated result   {mitigated:.3f}")
```

As the ideal, desired result is `1.000`, the mitigated result performs much better than unmitigated.


## Options

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

Further options are described and elaborated in our article on [additional options](../guide/zne-3-options.myst) in ZNE.


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
