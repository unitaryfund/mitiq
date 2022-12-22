---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: dev
  language: python
  name: python3
---

# Calibration

The `mitiq.calibration.Calibrator` class provides a workflow for users to run a set of experiments to automatically determine an error mitigation strategy.
This gives the user freedom to work on more important parts of their algorithm/quantum program, and allows them to spend less time tuning error mitigation parameters.


## Workflow

To begin, we will need to define an executor which tells mitiq how to run circuits.
In order to use the calibration capabilities of Mitiq, we will need to define an executor which returns bitstrings, rather than an expectation value as might be common.
This allows the calibration experiment to extract more fine-grained data from each circuit experiment it will run

```{code-cell} ipython3
import cirq
import numpy as np
from mitiq import MeasurementResult
```

```{code-cell} ipython3
def execute(circuit, noise_level=0.001):
    circuit = circuit.with_noise(cirq.amplitude_damp(noise_level))

    result = cirq.DensityMatrixSimulator().run(circuit, repetitions=100)
    return MeasurementResult(
        result=np.column_stack(list(result.measurements.values())).tolist(),
        qubit_indices=tuple(
            # q[2:-1] is necessary to convert "q(number)" into "number"
            int(q[2:-1])
            for k in result.measurements.keys()
            for q in k.split(",")
        ),
    )
```

We can now import the required objects and functions required for calibration.

```{code-cell} ipython3
from mitiq.calibration import Calibrator, ZNESettings, execute_with_mitigation
```

To instantiate a `Calibrator` we need to pass it an executor (as defined above), and a `Settings` object.
You are free to define your own, but we provide `ZNESettings` as a starting point to test basic performance.
Finally, the `execute_with_mitigation` function allows us to pass the calibration results directly to Mitiq and have it pick the strategy that performed best.

## Calibration Experiments

Before running any experiments, we can call the `get_cost` function to ensure the experiments will not be too costly.
Once instantiated, we call the `run` method to run the set of experiments, and the results of such experiments are stored internal to the class in `cal.results`.

```{code-cell} ipython3
cal = Calibrator(execute, ZNESettings)
print(cal.get_cost())
cal.run()
```

```{code-cell} ipython3
from pprint import pprint
pprint(cal.results)
print(len(cal.results))
```

## Using Calibrator Object with Mitigation

To demonstrate `execute_with_mitigation` we define a randomized benchmarking circuit to test it on.
All circuits that are run through an executor returning bitstrings must contains measurements.

```{code-cell} ipython3
from mitiq.benchmarks import generate_rb_circuits
circuit = generate_rb_circuits(2, 10)[0]
circuit.append(cirq.measure(circuit.all_qubits()))
```

We can now pass the randomized benchmarking circuit, along with the calibrator object, and a final argument used for specifying a bitstring that we would like to measure.

```{code-cell} ipython3
execute_with_mitigation(circuit, cal, bitstring='00')
```
