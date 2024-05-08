---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Composing techniques: Digital Dynamical Decoupling and Zero Noise Extrapolation

Noise in quantum computers can arise from a variety of sources, and sometimes applying multiple error mitigation techniques can be more beneficial than applying a single technique alone. 

Here we apply a combination of Digital Dynamical Decoupling (DDD) and Zero Noise Extrapolation (ZNE) on a GHZ state.

In [DDD](../guide/ddd.md), the input quantum circuit is modified by inserting gate sequences at regular intervals designed to reduce interaction between (i.e., decouple) the qubits from their environment. 

In [ZNE](../guide/zne.md), the expectation value of the observable of interest is computed at different noise levels, and subsequently the ideal expectation value is inferred by extrapolating the measured results to the zero-noise
limit. 


## Setup

We begin by importing the relevant modules and libraries required for the rest of this tutorial.

```python
import cirq
import numpy as np
from mitiq import MeasurementResult, Observable, PauliString
```

## Task

We will demonstrate quantum error mitigation on a [GHZ state](https://en.wikipedia.org/wiki/Greenberger%E2%80%93Horne%E2%80%93Zeilinger_state), entangling multiple qubits together. We can create a short function to do this for convenience, including an optional argument `idle_steps` which will give us a way to insert additional time steps in which to let our qubits idle. This is to more clearly simulate the effect of time correlated noise.

```python
def ghz(num_qubits, idle_steps=0):
    # Create  qubit registers
    qubits = cirq.LineQubit.range(num_qubits)

    # Create a quantum circuit
    circuit = cirq.Circuit()
    # Add a Hadamard gate to the first qubit
    circuit.append(cirq.H(qubits[0]))

    # Add CNOT gates to entangle the first qubit with each of the other qubits
    for i in range(1, num_qubits):
        circuit.append(cirq.CNOT(qubits[0], qubits[i]))
        # Set qubits to idle for specfied number of steps
        # in the form of Identity gates
        for step in range(idle_steps):
            circuit.append(cirq.I(q) for q in qubits)

    return circuit
```

```python
num_qubits = 6
circuit = ghz(num_qubits, 2)
print(circuit)
```

## Noise model and executor

**Importantly**, since DDD is designed to mitigate time-correlated (non-Markovian) noise, we simulate systematic $R_z$ rotations and depolarising noise applied to each qubit after each time step. This corresponds to noise which is strongly time-correlated and, therefore, likely to be mitigated by DDD.

We use an [executor function](../guide/executors.md) to run the quantum circuit with the noise model applied.

```python
def execute(
    circuit: cirq.Circuit, 
    rz_noise: float = 0.01,
    depolar_noise: float = 0.0005
    ) -> MeasurementResult:
    """
    Execute a circuit with R_z dephasing noise of strength ``rz_noise`` and depolarizing noise ``depolar_noise``
    """
    # Simulate systematic dephasing (coherent RZ) on each qubit for each moment.
    circuit = circuit.with_noise(cirq.rz(rz_noise))

    # Simulate systematic depolarizing on each qubit for each moment.
    circuit = circuit.with_noise(cirq.bit_flip(depolar_noise))

    # Measure out all qubits
    circuit += cirq.measure(*sorted(circuit.all_qubits()), key="m")

    # Use a noise simulator
    simulator = cirq.DensityMatrixSimulator()

    # Run the circuit 1000 times
    result = simulator.run(circuit, repetitions=1000)

    # Retrieve the measurement results in bitstring form
    bitstrings = result.measurements["m"]

    return MeasurementResult(bitstrings)
```

Let's see what the execute function returns when we call it on our GHZ circuit, just leaving the default noise levels:

```python
res = execute(circuit)
res.to_dict() # Dictionary for more convenient visualization
```

The `'counts'` key represents the number of times a particular measurement bitstring occurred. 

Since the default execute function includes noise in this case, we see that most of the results are all ones or all zeros, corresponding to perfect entanglement between all qubits.

A few others however are one or more bitflips away from either of these states. It is these types of errors we want to mitigate. 


## Observable

In this example, we just want to check if we have achieved entanglement across all qubits. In this case, we will measure the observable $⨂_{i=1}^n​X_i$ or measuring `X` on all qubits. 

This corresponds to projecting the state onto either the $∣+⟩^{⊗n}$ or $∣−⟩^{⊗n}$ basis. For a perfect GHZ state, this observable will have an expectation value of 1, which corresponds to all qubits being in the same state.

```python
obs = Observable(PauliString("X" * num_qubits))
print(obs)
```

For the circuit defined above, the ideal (noiseless) expectation value of the observable is 1, as we will see though, the unmitigated (noisy) result is impacted by depolarizing and readout errors.

```python
from functools import partial

ideal_exec = partial(execute, rz_noise = 0.0, depolar_noise = 0.0)

ideal = obs.expectation(circuit, ideal_exec)
print("Ideal value:", "{:.5f}".format(ideal.real))
```

```python
noisy_exec = partial(execute, rz_noise = 0.01, depolar_noise = 0.001)
noisy = obs.expectation(circuit, noisy_exec) 
print("Unmitigated noisy value:", "{:.5f}".format(noisy.real))
```

Next we choose our gate sequences to be used in the digital dynamical decoupling routine (DDD). 
More information on choosing appropriate sequences can be found in the [DDD theory](../guide/ddd-5-theory.md#common-examples-of-ddd-sequences) section of the user guide.

To do this, we will insert DDD sequences into our circuit itself and then compare the original circuit with the modified one.

```python
from mitiq.ddd import rules, insert_ddd_sequences

print("Original circuit \n", circuit)

rule = rules.yy

ddd_circuit = insert_ddd_sequences(circuit, rule)
print("DDD modified circuit \n", ddd_circuit)
```

Now we execute our function on the `ddd_circuit`:

```python
ddd_noisy = obs.expectation(ddd_circuit, noisy_exec)

print("Unmitigated expectation value:", "{:.5f}".format(noisy.real))

print("Expectation value with DDD:", "{:.5f}".format(ddd_noisy.real))
```

For comparison, we then apply ZNE one our original circuit (without DDD sequences inserted).

```python
from mitiq import zne

zne_executor = zne.mitigate_executor(noisy_exec, observable=obs, scale_noise=zne.scaling.folding.fold_global)
zne_result = zne_executor(circuit)
print("Mitigated value obtained with ZNE:", "{:.5f}".format(zne_result.real))
```

Finally, we apply a combination of DDD and ZNE.
DDD is applied first to apply the control pulses to each circuit which ZNE runs to do its extrapolation.

```python
combined_executor = zne.mitigate_executor(execute, observable=obs, scale_noise=zne.scaling.folding.fold_global)

combined_result = combined_executor(ddd_circuit)
print("Mitigated value obtained with DDD + ZNE:", "{:.5f}".format(combined_result.real))
```

From this example we can see that each technique affords some improvement, and for this specific noise model, the combination of DDD and ZNE is more effective in mitigating errors than either technique alone.
