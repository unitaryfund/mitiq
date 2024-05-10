---
jupytext:
  formats: md:myst
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

# Composing techniques: Digital Dynamical Decoupling and Zero Noise Extrapolation

Noise in quantum computers can arise from a variety of sources, and sometimes applying multiple error mitigation techniques can be more beneficial than applying a single technique alone.

Here we apply a combination of Digital Dynamical Decoupling (DDD) and Zero Noise Extrapolation (ZNE) on a GHZ state.

In [DDD](../guide/ddd.md), the input quantum circuit is modified by inserting gate sequences at regular intervals designed to reduce interaction between (i.e., decouple) the qubits from their environment. 

In [ZNE](../guide/zne.md), the expectation value of the observable of interest is computed at different noise levels, and subsequently the ideal expectation value is inferred by extrapolating the measured results to the zero-noise
limit. 

+++

## Setup

We begin by importing the relevant modules and libraries required for the rest of this tutorial.

```{code-cell}
import cirq
import numpy as np
from mitiq import MeasurementResult, Observable, PauliString
```

## Task

We will demonstrate quantum error mitigation on a [GHZ state](https://en.wikipedia.org/wiki/Greenberger%E2%80%93Horne%E2%80%93Zeilinger_state), entangling multiple qubits together. We can create a short function to do this for convenience. We will also define a utility function `idle_qubits`, which will give us a way to insert additional time steps in which to let our qubits idle. 


We will explain this further in the noise model section, but suffice it to say that these are intended to amplify the effect of time correlated noise. 

```{code-cell}
def idle_qubits(circuit, qubits, idle_steps):
    """Set qubits to idle for specfied number of steps 
    in by inserting Identity gates in each `moment`
    """
    for step in range(idle_steps):
        circuit.append(cirq.I(q) for q in qubits)

    return circuit
```

```{code-cell}
def ghz(num_qubits, idle_steps=0):
    # Create  qubit registers
    qubits = cirq.LineQubit.range(num_qubits)

    # Create a quantum circuit
    circuit = cirq.Circuit()
    
    # Add CNOT gates to entangle the first qubit with each of the other qubits
    for i in range(num_qubits):
        if i == 0: 
            # Add a Hadamard gate to the first qubit
            circuit.append(cirq.H(qubits[0]))
        else:
            circuit.append(cirq.CNOT(qubits[0], qubits[i]))
            # Set qubits to idle for specfied number of steps
            # in the form of Identity gates
            other_qubits = qubits[1:i] + qubits[i+1:]
            circuit = idle_qubits(circuit, other_qubits, idle_steps)
    
    return circuit
```

### Defining the circuit

For this example we will create a 6 qubit GHZ state with two idle steps after each moment:

```{code-cell}
num_qubits = 6
circuit = ghz(num_qubits, idle_steps=3)

print(circuit)
```

In the diagram above, the horizontal brakets above and below specific gates represent the [cirq](https://quantumai.google/cirq) notion of [moments](https://quantumai.google/reference/python/cirq/Moment), which are time-slices intended to help with qubit scheduling (e.g. making sure qubits arrive at a gate at the same time). 

In the our noise model, we will apply errors after each moment. Adding the additional identity gates within each "moment" should amplify the effect of these time-correlated errors.

**Note:** Due to cirq's definition of moments, the first two gates of the circuit (`cirq.H(qubits[0])` and `cirq.CNOT(qubits[0], qubits[i])`) are in the same moment. This means the idle steps cannot be added between these gates, and therefore the time correlated noise will have less of an effect for the start of the circuit. 

+++

## Noise model and executor

**Importantly**, since DDD is designed to mitigate time-correlated (non-Markovian) noise, we simulate systematic $R_z$ rotations and depolarising noise applied to each qubit after each time step. This corresponds to noise which is strongly time-correlated and, therefore, likely to be mitigated by DDD. We 

We use an [executor function](../guide/executors.md) to run the quantum circuit with the noise model applied.

```{code-cell}
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

```{code-cell}
res = execute(circuit)
res.to_dict() # Dictionary for more convenient visualization
```

The `'counts'` key represents the number of times a particular measurement bitstring occurred. 

Since the default execute function includes noise in this case, we see that most of the results are all ones or all zeros, corresponding to perfect entanglement between all qubits.

A few others, like `'110111'` however are one or more bitflips away from either of these states. It is these types of errors we want to mitigate. 

+++

## Observable

In this example, we just want to check if we have achieved entanglement across all qubits. In this case, we will measure the observable $⨂_{i=1}^n​X_i$ or measuring `X` on all qubits. 

This corresponds to projecting the state onto either the $∣+⟩^{⊗n}$ or $∣−⟩^{⊗n}$ basis. For a perfect GHZ state, this observable will have an expectation value of 1, which corresponds to all qubits being in the same state.

```{code-cell}
obs = Observable(PauliString("X" * num_qubits))
print(obs)
```

For the circuit defined above, the ideal (noiseless) expectation value of the observable is 1:

```{code-cell}
from functools import partial

ideal_exec = partial(execute, rz_noise = 0.0, depolar_noise = 0.0)

ideal = obs.expectation(circuit, ideal_exec)
print("Ideal value:", "{:.5f}".format(ideal.real))
```

The unmitigated (noisy) result however, is impacted by the `Rz` and depolarizing errors. In fact, we are not even halfway to the correct expectation value:

```{code-cell}
noisy_exec = partial(execute, rz_noise = 0.02, depolar_noise = 0.005)
noisy = obs.expectation(circuit, noisy_exec) 
print("Unmitigated noisy value:", "{:.5f}".format(noisy.real))
```

### Applying Digital Dynamical Decoupling

Next we choose our gate sequences to be used in the digital dynamical decoupling routine (DDD). 
More information on choosing appropriate sequences can be found in the [DDD theory](../guide/ddd-5-theory.md#common-examples-of-ddd-sequences) section of the user guide.

To do this, we will insert DDD sequences into our circuit itself and then compare the original circuit with the modified one.

```{code-cell}
from mitiq.ddd import insert_ddd_sequences, rules

print("Original circuit \n", circuit)

rule = rules.yy

ddd_circuit = insert_ddd_sequences(circuit, rule)
print("DDD modified circuit \n", ddd_circuit)
```

Now we execute our function on the `ddd_circuit`:

```{code-cell}
ddd_noisy = obs.expectation(ddd_circuit, noisy_exec)

print("Unmitigated expectation value:", "{:.5f}".format(noisy.real))

print("Expectation value with DDD:", "{:.5f}".format(ddd_noisy.real))
```

### Zero Noise Extrapolation alone

+++

For comparison, we then apply ZNE one our original circuit (without DDD sequences inserted).

```{code-cell}
from mitiq import zne

zne_executor = zne.mitigate_executor(noisy_exec, observable=obs, scale_noise=zne.scaling.folding.fold_global)
zne_result = zne_executor(circuit)
print("Mitigated value obtained with ZNE:", "{:.5f}".format(zne_result.real))
```

In this case, we see that ZNE by itself actually _makes things worse_ than the unmitigated expectation value!

+++

### Digital Dynamical Decoupling + Zero noise extrapolation

+++

Finally, we apply a combination of DDD and ZNE.
DDD is applied first to apply the control pulses to each circuit which ZNE runs to do its extrapolation.

```{code-cell}
combined_executor = zne.mitigate_executor(execute, observable=obs, scale_noise=zne.scaling.folding.fold_global)

combined_result = combined_executor(ddd_circuit)
print("Mitigated value obtained with DDD + ZNE:", "{:.5f}".format(combined_result.real))
```

From this example we can see that each technique affords some improvement, and for this specific noise model, the combination of DDD and ZNE is more effective in mitigating errors than either technique alone.

We encourage users to experiment with different circuits and noise models to see where they can find the best advantage to using these techniques in conjunction!
