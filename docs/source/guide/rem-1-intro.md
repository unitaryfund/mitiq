---
jupytext:
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

# How do I use REM?

As with all techniques, REM is compatible with any frontend supported by Mitiq:

```{code-cell} ipython3
import mitiq

mitiq.SUPPORTED_PROGRAM_TYPES.keys()
```

## Problem setup
In this example we will simulate a noisy device to demonstrate the capabilities of REM. This method requires an 
{ref}`observable <guide/observables/observables>` to be defined, and we use
$Z_0 + Z_1$ as an example. Since the circuit includes an $X$ gate on each qubit, 
the noiseless expectation value should be $-2$.

```{code-cell} ipython3
from cirq import LineQubit, Circuit, X, measure_each

from mitiq.observable.observable import Observable
from mitiq.observable.pauli import PauliString

qreg = [LineQubit(i) for i in range(2)]
circuit = Circuit(X.on_each(*qreg))
observable = Observable(PauliString("ZI"), PauliString("IZ"))

print(circuit)
```

Next we define a simple noisy readout executor function which takes a 
circuit as input, executes the circuit on a noisy simulator, and 
returns the raw measurement results. See the [Executors](executors.md) 
section for more information on how to define more advanced executors.

```{warning}
REM executors require bitstrings as output since the technique applies
to raw measurements results.
```

```{code-cell} ipython3
from functools import partial

import numpy as np
from cirq.experiments.single_qubit_readout_calibration_test import (
    NoisySingleQubitReadoutSampler,
)

from mitiq import MeasurementResult

def noisy_readout_executor(circuit, p0, p1, shots=8192) -> MeasurementResult:
    # Replace with code based on your frontend and backend.
    simulator = NoisySingleQubitReadoutSampler(p0, p1)
    result = simulator.run(circuit, repetitions=shots)
    bitstrings = np.column_stack(list(result.measurements.values()))
    return MeasurementResult(bitstrings, qubit_indices = (0, 1))
```

The [executor](executors.md) can be used to evaluate noisy (unmitigated)
expectation values.

```{code-cell} ipython3
from mitiq.raw import execute as raw_execute

# Compute the expectation value of the observable.
# Use a noisy executor that has a 25% chance of bit flipping
p_flip = 0.25
noisy_executor = partial(noisy_readout_executor, p0=p_flip, p1=p_flip)
noisy_value = raw_execute(circuit, noisy_executor, observable)

ideal_executor = partial(noisy_readout_executor, p0=0, p1=0)
ideal_value = raw_execute(circuit, ideal_executor, observable)
error = abs((ideal_value - noisy_value)/ideal_value)
print(f"Error without mitigation: {error:.3}")
```

## Apply Postselection
The simplest version of readout error mitigation that we could do is to postselect specific bitstrings using 
{mod}`mitiq.rem.post_select`. This strategy is only applicable if, from the structure of the problem, there is some kind symmetry that we can assume and therefore enforce by post-selection. For example, we observe that the circuit in our example is symmetric with respect to interchanging the
two qubits. Assuming we are given the promise that the ideal result must be a unique bitstring, such a bitstring must
be symmetric with respect to a bit swap. Therefore, a possible error mitigation strategy is to keep only the results
where the bits match.

```{code-cell} ipython3
from mitiq.rem import post_select

circuit_with_measurements = circuit.copy()
circuit_with_measurements.append(measure_each(*qreg))
noisy_measurements = noisy_executor(circuit_with_measurements)
print(f"Before postselection: {noisy_measurements.get_counts()}")
postselected_measurements = post_select(noisy_measurements, lambda bits: bits[0] == bits[1])
print(f"After postselection: {postselected_measurements.get_counts()}")
total_measurements = len(noisy_measurements.result)
discarded_measurements = total_measurements - len(postselected_measurements.result)
print(f"Discarded measurements: {discarded_measurements} ({discarded_measurements/total_measurements:.0%} of total)")

mitigated_result = observable._expectation_from_measurements([postselected_measurements])
error = abs((ideal_value - mitigated_result)/ideal_value)
print(f"Error with mitigation (PS): {error:.3}")
```

So, if we used these postselected results, then we'd get closer to the expected noiseless expectation value. However, that comes at the cost of throwing away a fraction of our measurements. 

## Apply REM
A more elaborate readout-error mitigation technique can be easily applied with the function
{func}`.execute_with_rem()`.

```{code-cell} ipython3
from mitiq.rem import generate_inverse_confusion_matrix
from mitiq import rem

# We use a utility method to generate a simple inverse confusion matrix, but
# you can supply your own confusion matrices and invert them using the helper
# function generate_tensored_inverse_confusion_matrix().
inverse_confusion_matrix = generate_inverse_confusion_matrix(2, p_flip, p_flip)

mitigated_result = rem.execute_with_rem(
    circuit,
    noisy_executor,
    observable,
    inverse_confusion_matrix=inverse_confusion_matrix,
)
```

```{code-cell} ipython3
error = abs((ideal_value - mitigated_result)/ideal_value)
print(f"Error with mitigation (REM): {error:.3}")
```

Here we observe that the application of REM reduces the readout error when compared
to the unmitigated result.

```{note} 
It is necessary to supply the inverse confusion matrix to the REM technique.
There are various approaches that can be used to generate the inverse 
confusion matrix with some being more costly than others.
```

+++

The section [What additional options are available when using REM?](rem-3-options.md) contains more information on
generating inverse confusion matrices in order to apply REM with Mitiq.
