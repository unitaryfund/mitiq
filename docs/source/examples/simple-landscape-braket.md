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

# Using ZNE to compute the energy landscape of a variational circuit with Braket

This tutorial shows an example in which the energy landscape for a two-qubit variational circuit is explored with and without error mitigation, using Amazon's [Braket](https://amazon-braket-sdk-python.readthedocs.io/en/latest/) as our frontend.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np

from braket.devices import LocalSimulator
from braket.circuits import Circuit, Noise, Observable

from mitiq import zne
from mitiq.zne.inference import RichardsonFactory
```

## Defining the ideal variational circuit in Braket

+++

We define a function which returns a simple two-qubit variational circuit depending on a single parameter $\gamma$ ("gamma").

```{code-cell} ipython3
def variational_circuit(gamma: float) -> Circuit:
    """Returns a two-qubit circuit for a given variational parameter.

    Args:
        gamma: The variational parameter.

    Returns:
        The two-qubit circuit with a fixed gamma.
    """
    my_circuit = Circuit()
    my_circuit.rx(0, gamma)
    my_circuit.cnot(0, 1)
    my_circuit.rx(1, gamma)
    my_circuit.cnot(0, 1)
    my_circuit.rx(0, gamma)
    
    return my_circuit
```

We can visualize the circuit for a particular $\gamma$ as follows.

```{code-cell} ipython3
circuit = variational_circuit(gamma=np.pi)
print(circuit)
```

## Defining the executor functions with and without noise

To use error mitigation methods in Mitiq, we define an executor function which computes the expectation value of a simple Hamiltonian $H=Z \otimes Z$, i.e., Pauli-$Z$ on each qubit.
To compare to the noiseless result, we define both a noiseless and a noisy executor below.
More information about executors can be found [here](../guide/executors.md).

```{code-cell} ipython3
# Observable to measure
Z = np.diag([1, -1])
hamiltonian = np.kron(Z, Z)

def noiseless_executor(circ: Circuit) -> float:
    """Simulates the execution of a circuit without noise.

    Args:
        circ: The input circuit.

    Returns:
        The expectation value of the ZZ observable.
    """
    device = LocalSimulator('braket_dm')
    # Evaluate the ZZ expectation value
    circ.expectation(observable=Observable.Z() @ Observable.Z(), target=range(2))
    
    task = device.run(circ)
    result = task.result()
    return result.values

# Strength of noise channel
noise_level = 0.04

def executor_with_noise(circ: Circuit) -> float:
    """Simulates the execution of a circuit with depolarizing noise.

    Args:
        circ: The input circuit.

    Returns:
        The expectation value of the ZZ observable.
    """
    # Add depolarizing noise to the circuit
    noise = Noise.Depolarizing(probability=noise_level)
    circ.apply_gate_noise(noise)
    # Use the noiseless_executor function to return the expectation value of the ZZ observable for the noisy circuit
    return noiseless_executor(circ)
```

```{note}
The above code block uses depolarizing noise, but any Braket [`Noise`](https://amazon-braket-sdk-python.readthedocs.io/en/latest/_apidoc/braket.circuits.noise.html) channel can be substituted in.
```

+++

## Computing the landscape without noise

We now compute the energy landscape $\langle H \rangle(\gamma) =\langle Z \otimes Z \rangle(\gamma)$ on the noiseless simulator.

```{note}
The remaining code in this tutorial is generic and does not depend on a particular frontend.
```

```{code-cell} ipython3
gammas = np.linspace(0, 2 * np.pi, 50)
noiseless_expectations = [noiseless_executor(variational_circuit(g)) for g in gammas]
```

The following code plots the values for visualization.

```{code-cell} ipython3
plt.figure(figsize=(8, 6))
plt.plot(gammas, noiseless_expectations, color="g", linewidth=3, label="Noiseless")
plt.title("Energy landscape", fontsize=16)
plt.xlabel(r"Ansatz angle $\gamma$", fontsize=16)
plt.ylabel(r"$\langle H \rangle(\gamma)$", fontsize=16)
plt.legend(fontsize=14)
plt.ylim(-1, 1);
plt.show()
```

## Computing the unmitigated landscape

We now compute the unmitigated energy landscape $\langle H \rangle(\gamma) =\langle Z \otimes Z \rangle(\gamma)$
in the following code block.

```{code-cell} ipython3
expectations = [executor_with_noise(variational_circuit(g)) for g in gammas]
```

The following code plots these values for visualization along with the noiseless landscape.

```{code-cell} ipython3
plt.figure(figsize=(8, 6))
plt.plot(gammas, noiseless_expectations, color="g", linewidth=3, label="Noiseless")
plt.scatter(gammas, expectations, color="r", label="Unmitigated")
plt.title(rf"Energy landscape", fontsize=16)
plt.xlabel(r"Ansatz angle $\gamma$", fontsize=16)
plt.ylabel(r"$\langle H \rangle(\gamma)$", fontsize=16)
plt.legend(fontsize=14)
plt.ylim(-1, 1);
plt.show()
```

## Computing the mitigated landscape

We now repeat the same task but use Mitiq to mitigate errors.
We initialize a [RichardsonFactory](https://mitiq.readthedocs.io/en/stable/apidoc.html#mitiq.zne.inference.RichardsonFactory) with scale factors `[1, 3, 5]` and we get a mitigated executor as follows.

```{code-cell} ipython3
fac = RichardsonFactory(scale_factors=[1, 3, 5])
mitigated_executor = zne.mitigate_executor(executor_with_noise, factory=fac)
```

We then run the same code above to compute the energy landscape, but this time use the ``mitigated_executor`` instead of just the executor.

```{code-cell} ipython3
mitigated_expectations = [mitigated_executor(variational_circuit(g)) for g in gammas]
```

Let us visualize the mitigated landscape alongside the unmitigated and noiseless landscapes.

```{code-cell} ipython3
plt.figure(figsize=(8, 6))
plt.plot(gammas, noiseless_expectations, color="g", linewidth=3, label="Noiseless")
plt.scatter(gammas, expectations, color="r", label="Unmitigated")
plt.scatter(gammas, mitigated_expectations, color="b", label="Mitigated")
plt.title(rf"Energy landscape", fontsize=16)
plt.xlabel(r"Variational angle $\gamma$", fontsize=16)
plt.ylabel(r"$\langle H \rangle(\gamma)$", fontsize=16)
plt.legend(fontsize=14)
plt.ylim(-1.5, 1.5);
plt.show()
```

Noise usually tends to flatten expectation values towards a constant. Therefore error mitigation
can be used to increase the visibility the landscape and this fact can simplify the energy minimization
which is required in most variational algorithms such as VQE or QAOA.

We also observe that the minimum of mitigated energy approximates well the theoretical ground state which is equal to $-1$. Indeed:

```{code-cell} ipython3
print(f"Minimum of the noisy landscape: {round(min([exp[0] for exp in expectations]), 3)}")
print(f"Minimum of the mitigated landscape: {round(min(mitigated_expectations), 3)}")
print(f"Theoretical ground state energy: {min(np.linalg.eigvals(hamiltonian))}")
```
