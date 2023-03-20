---
jupytext:
  text_representation:
    extension: .myst
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---


# Digital dynamical decoupling (DDD) with Qiskit on GHZ Circuits

In this notebook DDD is applied to improve the success rate of the computation on a real hardware backend. 
A similar approach can be taken on a simulated backend, by setting the ``USE_REAL_HARDWARE`` option to ``False``
and specifying a simulated backend from `qiskit.providers.fake_provider`, which includes a noise model that approximates the noise of the
real device.

In DDD, sequences of gates are applied to slack windows, i.e. single-qubit idle windows, in a quantum circuit. 
Applying such sequences can reduce the coupling between the qubits and the environment, mitigating the effects of noise.
While the DDD module includes some built-in sequences, the user may choose to define others best suited to their application.
For more information on DDD, see the section [DDD section of the user guide](../guide/ddd.md).


## Setup

We begin by importing the relevant modules and libraries that we will require
for the rest of this tutorial.

```{code-cell} ipython3
from typing import List, Callable
import numpy as np
from matplotlib import pyplot as plt

import cirq
import qiskit

from mitiq.interface.mitiq_qiskit import to_qiskit
from mitiq import ddd, QPROGRAM
```


## Define DDD rules
We now use DDD _rule_ from Mitiq, i. e., a function that generates DDD sequences of different length.
In this example, we test the performance of repeated I, repeated IXIX, repeated XX, and XX sequences.

```{code-cell} ipython3
def rep_i_rule(window_length: int) -> Callable[[int], QPROGRAM]:
    """This is the trivial sequence and corresponds the unmitigated case."""
    seq = ddd.rules.repeated_rule(window_length, [cirq.I])
    return cirq.Circuit(seq)

def rep_ixix_rule(window_length: int) -> Callable[[int], QPROGRAM]:
    return ddd.rules.repeated_rule(
        window_length, [cirq.I, cirq.X, cirq.I, cirq.X]
    )

def rep_xx_rule(window_length: int) -> Callable[[int], QPROGRAM]:
    return ddd.rules.repeated_rule(window_length, [cirq.X, cirq.X])

# Set DDD sequences to test.
rules = [rep_i_rule, rep_ixix_rule, rep_xx_rule, ddd.rules.xx]

# Test the sequence insertion
for rule in rules:
    print(rule(10))
```


## Set parameters for the experiment

```{code-cell} ipython3
# Total number of shots to use.
shots = 10000

# Qubits to use on the experiment.
num_qubits = 2

# Test at multiple depths.
depths = [10, 30, 50, 100]
```


## Define the circuit

We use Greenberger-Horne-Zeilinger (GHZ) circuits to benchmark the performance of the device.
GHZ circuits are designed such that only two bitstrings $|00...0 \rangle$ and $|11...1 \rangle$
should be sampled, with $P_0 = P_1 = 0.5$.
As noted in *Mooney et al. (2021)* {cite}`Mooney_2021`, when GHZ circuits are run on a device, any other measured bitstrings are due to noise.
In this example the GHZ sequence is applied first, followed by a long idle window of identity gates and finally the inverse of the GHZ
sequence.
Therefore $P_0 = 1$ and the frequency of the $|00...0 \rangle$ bitstring is our target metric.

```{code-cell} ipython3
def get_circuit_with_sequence(depth: int, rule: Callable[[int], QPROGRAM]):
    """Returns a circuit composed of a GHZ sequence, idle windows with or
        without DDD sequences, and finally an inverse GHZ sequence.

    Args:
        depth: The depth of the idle window in the circuit.
        rule: A function determining the sequence to insert in the idle window.
            In the unmitigated case this generates a sequence of identity
            gates. In the DDD mitigated case it generates a sequence of non-
            identity gates or a combination of identity and non-identity gates.
    """
    circuit = qiskit.QuantumCircuit(num_qubits, num_qubits)
    circuit.h(0)
    circuit.cx(0, 1)
    
    sequence = rule(depth)
    sequence_qiskit = to_qiskit(sequence)    
    circuit = circuit.compose(sequence_qiskit)
    
    circuit.cx(0, 1)
    circuit.h(0)
    circuit.measure(0, 0)
    return circuit

def get_circuit(depth):
    return get_circuit_with_sequence(depth, rep_i_rule)
```

Test the circuit output for depth 4, unmitigated
```{code-cell} ipython3
ibm_circ= get_circuit(4)
print(ibm_circ)
```

Test the circuit output for depth 4, with IX sequences inserted 
```{code-cell} ipython3
ibm_circ= get_circuit_with_sequence(4, rep_ixix_rule)
print(ibm_circ)
```


## Define the executor

Now that we have a circuit, we define the `execute` function which inputs a circuit and returns an expectation value -
here, the frequency of sampling the correct bitstring.

```{code-cell} ipython3
USE_REAL_HARDWARE = True
```

```{code-cell} ipython3
:tags: ["remove-cell"]
# hidden settings to allow efficient docs build
USE_REAL_HARDWARE = False
depths = [2, 4, 6, 8]
```

```{code-cell} ipython3
if USE_REAL_HARDWARE:
    provider = qiskit.IBMQ.load_account()
    backend = provider.get_backend("ibmq_lima")
else:
    from qiskit.providers.fake_provider import FakeLima as FakeLima
    backend = FakeLima()


def ibm_executor(
    circuit: qiskit.QuantumCircuit,
    shots: int,
    correct_bitstring: List[int],
    noisy: bool = True,
) -> float:
    """Executes the input circuit(s) and returns ⟨A⟩, where 
    A = |correct_bitstring⟩⟨correct_bitstring| for each circuit.

    Args:
        circuit: Circuit to run.
        shots: Number of times to execute the circuit to compute the
            expectation value.
        correct_bitstring: Bitstring the circuit is expected to return, in the
            absence of noise.
    """
    if noisy:
        transpiled = qiskit.transpile(circuit, backend=backend, optimization_level=0)
        job = backend.run(transpiled, optimization_level=0, shots=shots)
    else:
        ideal_backand = qiskit.Aer.get_backend("qasm_simulator")
        job = ideal_backand.run(circuit, optimization_level=0, shots=shots)

    # Convert from raw measurement counts to the expectation value
    all_counts = job.result().get_counts()
    print("Counts:", all_counts)
    prob_zero = all_counts.get("".join(map(str, correct_bitstring)), 0.0) / shots
    return prob_zero
```


## Run circuits with and without DDD

```{code-cell} ipython3
:tags: ["remove-output"]
data = []
for depth in depths:
    for rule in rules:
        print(f"DDD sequence: {rule}.")
        circuit = get_circuit_with_sequence(depth, rule)
        noisy_value = ibm_executor(
            circuit, shots=shots, correct_bitstring=[0]
        )
        print("Result:", noisy_value)
        data.append((depth, rule, noisy_value))
```

Now we can visualize the results.

```{code-cell} ipython3
:tags: ["remove-output"]
# Plot unmitigated
x, y = [], []
for res in data:
    if res[1].__name__ == "rep_i_rule":
        x.append(res[0])
        y.append(res[2])
plt.plot(x, y, "--*", label="Unmitigated")

# Plot xx
x, y = [], []
for res in data:
    if res[1].__name__ == "rep_xx_rule":
        x.append(res[0])
        y.append(res[2])
plt.plot(x, y, "--*", label="rep_xx_rule")

# Plot ixix
x, y = [], []
for res in data:
    if res[1].__name__ == "rep_ixix_rule":
        x.append(res[0])
        y.append(res[2])
plt.plot(x, y, "--*", label="rep_ixix_rule")

# Plot xx
x, y = [], []
for res in data:
    if res[1].__name__ == "xx":
        x.append(res[0])
        y.append(res[2])
plt.plot(x, y, "--*", label="xx")


plt.legend()
```

```{figure} ../_thumbnails/ddd_qiskit_ghz_plot.png
---

name: ddd-qiskit-ghz-plot-ibmq
---
Plot of the unmitigated and DDD-mitigated expectation values obtained from executing the corresponding circuits.
```


We can see that DDD improves the expectation value at each circuit depth, and the repeated XX sequence is the best at mitigating the errors
occurring during idle windows. 
