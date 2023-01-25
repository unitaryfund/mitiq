---
jupytext:
  text_representation:
    extension: .myst
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# What happens when I use REM?

The workflow of Readout-Error Mitigation (REM) in Mitiq is represented in the figure below.

```{figure} ../img/rem_workflow.svg
---
width: 700px
name: rem-workflow
---
Workflow of the REM technique in Mitiq.
```

- The user provides a `QPROGRAM`, (i.e. a quantum circuit defined via any of the supported [frontends](frontends-backends.myst)).
- Mitiq leaves the input circuit unmodified.
- The unmodified circuit is executed via a user-defined [Executor](executors.myst).
- REM modifies the measurement results by transforming by an inverse confusion matrix.
- The error mitigated expectation value is returned to the user.

With respect to the workflows of other error-mitigation techniques (e.g. [ZNE](zne-4-low-level.myst) or [PEC](pec-4-low-level.myst)),
REM does not require the modification of the circuit. The results of the execution of the _single_ unmodified circuit are what get modified.
For this reason, the circuit generation step is trivial and the work of the technique happens in the final inference step.

As shown in [How do I use REM?](rem-1-intro.myst), the function {func}`.execute_with_rem()` applies REM behind the scenes 
and directly returns the error-mitigated expectation value.
In the next sections instead, we show how one can apply REM at a lower level, i.e., by by applying each step independently:

- Obtaining measurement results directly from the executor;
- Mitigating the measurements with an inverse confusion matrix;
- Taking the expectation from the mitigated measurements.

## Obtaining measurements results directly from the executor

Let's use the same example from [How do I use REM?](rem-1-intro.myst):

```{code-cell} ipython3
from cirq import LineQubit, Circuit, X, measure_each

from mitiq.observable.observable import Observable
from mitiq.observable.pauli import PauliString

qreg = [LineQubit(i) for i in range(2)]
circuit = Circuit(X.on_each(*qreg), measure_each(*qreg))
observable = Observable(PauliString("ZI"), PauliString("IZ"))

print(circuit)
```

Let's create a noisy readout executor to have erroneous values to work with:

```{code-cell} ipython3
from functools import partial

import numpy as np
from cirq.experiments.single_qubit_readout_calibration_test import (
    NoisySingleQubitReadoutSampler,
)

from mitiq import MeasurementResult
from mitiq.raw import execute as raw_execute

def noisy_readout_executor(
    circuit, p0: float = 0.01, p1: float = 0.01, shots: int = 8192
) -> MeasurementResult:
    # Replace with code based on your frontend and backend.
    simulator = NoisySingleQubitReadoutSampler(p0, p1)
    result = simulator.run(circuit, repetitions=shots)

    return MeasurementResult(
        result=np.column_stack(list(result.measurements.values())),
        qubit_indices=tuple(
            # q[2:-1] is necessary to convert "q(number)" into "number"
            int(q[2:-1])
            for k in result.measurements.keys()
            for q in k.split(",")
        ),
    )

# Use a noisy executor that has a 25% chance of flipping
p_flip = 0.25
noisy_executor = partial(noisy_readout_executor, p0=p_flip, p1=p_flip)
```

With our noisy executor we can obtain the raw measurement results with {func}`Executor._run()`:

```{code-cell} ipython3
from mitiq.executor.executor import Executor

executor = Executor(noisy_executor)
result = executor._run([circuit])
noisy_result = result[0]
print(noisy_result)
```

## Mitigating the measurements with an inverse confusion matrix

For this example we use a simple, uncorrelated readout error model where each qubit has a separate probability of 
flipping from $|0\rangle \rightarrow |1\rangle$ and $|1\rangle \rightarrow |0\rangle$ during readout. More details
of constructing this inverse confusion matrix are presented in 
[What additional options are available when using REM?](rem-3-options.myst).

```{code-cell} ipython3
from mitiq.rem import generate_inverse_confusion_matrix

inverse_confusion_matrix = generate_inverse_confusion_matrix(2, p_flip, p_flip)
```

Next, we utilize our noisy results and the inverse confusion matrix in order to obtain mitigated results:

```{code-cell} ipython3
from mitiq.rem.inverse_confusion_matrix import mitigate_measurements

mitigated_result = mitigate_measurements(noisy_result, inverse_confusion_matrix)
```

The details of how the measurements are mitigated are presented in 
[What is the theory behind REM?](rem-5-theory.myst).

## Taking the expectation from the mitigated measurements

Finally, once we have our mitigated results we can obtain the expectation value from the observable:

```{code-cell} ipython3
print(observable._expectation_from_measurements([mitigated_result]))
```

Based on the observable we defined, $Z_0 + Z_1$, and the circuit we are close to the expectation value $-2$.