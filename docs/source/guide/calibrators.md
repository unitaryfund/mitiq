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

# Calibration

The `mitiq.calibration.Calibrator` class provides a workflow for users to run a set of experiments to automatically determine an error mitigation strategy.
This gives the user freedom to work on more important parts of their algorithm/quantum program, and allows them to spend less time tuning error mitigation parameters.


## Workflow

To begin, we will need to define an [executor](executors.md) which tells Mitiq how to run circuits.
In order to use the calibration capabilities of Mitiq, we will need to define an executor which returns all the measured bitstrings, rather than an expectation value.
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
    bitstrings = np.column_stack(list(result.measurements.values()))
    return MeasurementResult(bitstrings)
```

We can now import the required objects and functions required for calibration.

```{code-cell} ipython3
from mitiq.calibration import Calibrator, ZNESettings, execute_with_mitigation
```

To instantiate a `Calibrator` we need to pass it an executor (as defined above), and a `Settings` object.
You are free to define your own `Settings`, but we provide `ZNESettings` as a simple starting point based on different zero-noise extrapolation strategies.
Finally, the `execute_with_mitigation` function allows us to pass the calibration results directly to Mitiq and have it pick the strategy that performed best.

## Calibration Experiments

Before running any experiments, we can call the `get_cost` function to ensure the experiments will not be too costly.
Once instantiated, we call the `run` method to run the set of experiments, and the results of such experiments are stored internal to the class in `cal.results`.

```{code-cell} ipython3
cal = Calibrator(execute, ZNESettings, frontend="cirq")
print(cal.get_cost())
cal.run()
```

## Applying the optimal error mitigation strategy

We first define randomized benchmarking circuit to test the effect of error mitigation.

```{code-cell} ipython3
from mitiq.benchmarks import generate_rb_circuits

circuit = generate_rb_circuits(2, 10)[0]
# circuits passed to an executor returning bitstrings must contain measurements
circuit.append(cirq.measure(circuit.all_qubits()))
```

Instead of deciding what error mitigation technique and what options to use, we can ask Mitiq to determine the optimal error mitigation strategy based on the previously performed calibration. 
We can obtain this by calling the `execute_with_mitigation` function and passing the `circuit`, `Calibrator` object, and a new expectation value executor.

```{code-cell} ipython3
def execute(circuit, noise_level=0.001):
    circuit = circuit.with_noise(cirq.amplitude_damp(noise_level))

    rho = (
        cirq.DensityMatrixSimulator()
        .simulate(circuit)
        .final_density_matrix
    )
    return rho[0, 0].real

execute_with_mitigation(circuit, execute, calibrator=cal)
```
