---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Composing techniques: Readout Error Mitigation and Zero Noise Extrapolation

Noise in quantum computers can arise from a variety of sources, and sometimes applying multiple error mitigation techniques can be more beneficial than applying a single technique alone. Here we apply a combination of readout error mitigation (REM) and zero noise extrapolation (ZNE) to a randomized benchmarking (RB) task. In REM, the inverse transition / confusion matrices are generated and applied to the noisy measurement results. In ZNE, the expectation value of the observable of interest is computed at different noise levels, and subsequently the ideal expectation value is inferred by extrapolating the measured results to the zero-noise limit. More information on the [REM](../guide/rem.md) and [ZNE](../guide/zne.md) techniques can be found in the corresponding sections of the user guide.

+++

## Setup

We begin by importing the relevant modules and libraries required for the rest of this tutorial.

```{code-cell} ipython3
import cirq
import numpy as np
from mitiq.benchmarks import generate_rb_circuits
from mitiq import MeasurementResult, Observable, PauliString, raw
```

## Task

We will demonstrate using REM + ZNE on RB circuits, which are generated using Mitiq's built-in benchmarking function, `generate_rb_circuits`. More information on the RB protocol is available [here](https://learn.qiskit.org/course/quantum-hardware/randomized-benchmarking). In this example we use a two-qubit RB circuit with a Clifford depth (number of Clifford groups) of 5.

```{code-cell} ipython3
circuit = generate_rb_circuits(2, 5)[0]
```

## Noise model and executor

The noise in this example is a combination of depolarizing and readout errors, the latter of which are modeled as bit flips immediately prior to measurement. We use an [executor function](../guide/executors.md) to run the quantum circuit with the noise model applied.

```{code-cell} ipython3
def execute(circuit: cirq.Circuit, noise_level: float = 0.005, p0: float = 0.05) -> MeasurementResult:
    """Execute a circuit with depolarizing noise of strength ... and readout errors ...
    measurements = circuit[-1]
    circuit =  circuit[:-1]
    circuit = circuit.with_noise(cirq.depolarize(noise_level))
    circuit.append(cirq.bit_flip(p0).on_each(circuit.all_qubits()))
    circuit.append(measurements)

    simulator = cirq.DensityMatrixSimulator()

    result = simulator.run(circuit, repetitions=10000)
    bitstrings = np.column_stack(list(result.measurements.values()))
    return MeasurementResult(bitstrings)
```

## Observable

We then obtain a unmitigated expectation value of the `ZI` + `ZZ` observable. In the ideal case the result is 2, but as we will see, the unmitigated result is impacted by depolarizing and readout errors.

```{code-cell} ipython3
obs = Observable(PauliString("ZI"), PauliString("IZ"))
noisy = raw.execute(circuit, execute, obs)
```

```{code-cell} ipython3
from functools import partial

ideal = raw.execute(circuit, partial(execute, noise_level=0, p0=0), obs)
f"Error without mitigation: {abs(ideal - noisy) :.5f}"
```

Next we generate the inverse confusion matrix and apply readout error mitigation (REM). More information on generating the inverse confusion matrix is available in the [REM theory](../guide/rem-5-theory.md) section of the user guide. 

```{code-cell} ipython3
from mitiq import rem

p0 = p1 = 0.05
icm = rem.generate_inverse_confusion_matrix(2, p0, p1)
rem_executor = rem.mitigate_executor(execute, inverse_confusion_matrix=icm)

rem_result = obs.expectation(circuit, rem_executor)
f"Error with REM: {abs(ideal - rem_result) :.5f}"
```

We can see that REM improves the results, but errors remain. For comparison, we then apply ZNE (without REM).

```{code-cell} ipython3
from mitiq import zne

zne_executor = zne.mitigate_executor(execute, observable=obs)
zne_result = zne_executor(circuit)
f"Error with ZNE: {abs(ideal - zne_result) :.5f}"
```

Finally, we apply a combination of REM and ZNE. REM is applied first to minimize the impact of measurement errors on the extrapolated result in ZNE.

```{code-cell} ipython3
combined_executor = zne.mitigate_executor(rem_executor, observable=obs)

combined_result = combined_executor(circuit)
f"Error with REM + ZNE: {abs(ideal - combined_result) :.5f}"
```

We can see that, in this example, the combination of REM and ZNE is more effective in mitigating errors than either technique alone.
