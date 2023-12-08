---
jupytext:
  formats: ipynb,md:myst
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

# Breaking into error mitigation with Mitiq's calibration module

<img src="../_thumbnails/calibration.png" width="400">

This tutorial helps answer the question: "What quantum error mitigation technique should I use for my problem?".
The newly introduced `mitiq.calibration` module helps answer that in an optimized way, through `Benchmarks` and `Strategies`.

More specifically, this tutorial covers:

- Getting started with Mitiq's calibration module with ZNE
- Use Qiskit noisy simulator with `FakeJakarta` as backend
- Run calibration with some special settings, `RBSettings`, and logging the results

## Getting started with Mitiq

```{code-cell} ipython3
from mitiq.benchmarks import generate_rb_circuits
from mitiq.zne import execute_with_zne
from mitiq.zne.scaling import (
    fold_gates_at_random,
    fold_global,
    fold_all
)
from mitiq.zne.inference import LinearFactory, RichardsonFactory
from mitiq import (
    Calibrator,
    Settings,
    execute_with_mitigation,
    MeasurementResult,
)

from qiskit.providers.fake_provider import FakeJakarta  # Fake (simulated) QPU
```

### Define the circuit to study

#### Global variables

Define global variables for the quantum circuit of interest: number of qubits, depth of the quantum circuit and number of shots.

```{code-cell} ipython3
n_qubits = 2
depth_circuit = 100
shots = 10 ** 4
```

#### Quantum circuit: Randomized benchmarking (RB)

We now use Mitiq's built-in `generate_rb_circuits` from the `mitiq.benchmarks` module to define the quantum circuit.

```{code-cell} ipython3
circuit = generate_rb_circuits(n_qubits, depth_circuit,return_type="qiskit")[0]
circuit.measure_all()
print(len(circuit))
```

We define a function that executes the quantum circuits and returns the expectation value.
This is consumed by Mitiq's `execute_with_zne`. In this example, the expectation value is the probability of measuring the ground state, which is what one would expect from an ideal randomized benchmarking circuit.

```{code-cell} ipython3
def execute_circuit(circuit):
    """Execute the input circuit and return the expectation value of |00..0><00..0|"""
    noisy_backend = FakeJakarta()
    noisy_result = noisy_backend.run(circuit, shots=shots).result()
    noisy_counts = noisy_result.get_counts(circuit)
    noisy_expectation_value = noisy_counts[n_qubits * "0"] / shots
    return noisy_expectation_value
```

```{code-cell} ipython3
mitigated = execute_with_zne(circuit, execute_circuit, factory=LinearFactory([1, 3, 5]))
unmitigated = execute_circuit(circuit)
ideal = 1 #property of RB circuits

print("ideal = \t \t",ideal)
print("unmitigated = \t \t", "{:.5f}".format(unmitigated))
print("mitigated = \t \t", "{:.5f}".format(mitigated))
```

## Using calibration to improve the results

Let's consider a noisy backend using the Qiskit noisy simulator, `FakeJakarta`. Note that the executor passed to the `Calibrator` object must return counts, as opposed to expectation values.

```{code-cell} ipython3
def execute_calibration(qiskit_circuit):
    """Execute the input circuits and return the measurement results."""
    noisy_backend = FakeJakarta()
    noisy_result = noisy_backend.run(qiskit_circuit, shots=shots).result()
    noisy_counts = noisy_result.get_counts(qiskit_circuit)
    noisy_counts = { k.replace(" ",""):v for k, v in noisy_counts.items()}
    measurements = MeasurementResult.from_counts(noisy_counts)
    return measurements
```

We import from the calibration module the key ingredients to use `mitiq.calibration`: the `Calibrator` class, the `mitiq.calibration.settings.Settings` class and the `execute_with_mitigation` function.

Currently `mitiq.calibration` supports ZNE as a technique to calibrate from, tuning different scale factors, extrapolation methods and circuit scaling methods.

Let's run the calibration using an ad-hoc RBSettings while logging the results for comparison.

- benchmarks: Circuit type: "rb"
- strategies: use various "zne" strategies, testing various "scale_noise" methods
  (such as `mitiq.zne.scaling.folding.fold_global`, `mitiq.zne.scaling.folding.fold_gates_at_random`, and `mitiq.zne.scaling.folding.fold_all`),
  and ZNE factories for extrapolation (such as `mitiq.zne.inference.RichardsonFactory` and `mitiq.zne.inference.LinearFactory`)

```{code-cell} ipython3
RBSettings = Settings(
    benchmarks=[
        {
            "circuit_type": "rb",
            "num_qubits": 2,
            "circuit_depth": int(depth_circuit / 2),
        },
    ],
    strategies=[
        {
            "technique": "zne",
            "scale_noise": fold_global,
            "factory": RichardsonFactory([1.0, 2.0, 3.0]),
        },
        {
            "technique": "zne",
            "scale_noise": fold_global,
            "factory": RichardsonFactory([1.0, 3.0, 5.0]),
        },
        {
            "technique": "zne",
            "scale_noise": fold_global,
            "factory": LinearFactory([1.0, 2.0, 3.0]),
        },
        {
            "technique": "zne",
            "scale_noise": fold_global,
            "factory": LinearFactory([1.0, 3.0, 5.0]),
        },

        {
            "technique": "zne",
            "scale_noise": fold_gates_at_random,
            "factory": RichardsonFactory([1.0, 2.0, 3.0]),
        },
        {
            "technique": "zne",
            "scale_noise": fold_gates_at_random,
            "factory": RichardsonFactory([1.0, 3.0, 5.0]),
        },
        {
            "technique": "zne",
            "scale_noise": fold_gates_at_random,
            "factory": LinearFactory([1.0, 2.0, 3.0]),
        },
        {
            "technique": "zne",
            "scale_noise": fold_gates_at_random,
            "factory": LinearFactory([1.0, 3.0, 5.0]),
        },

        {
            "technique": "zne",
            "scale_noise": fold_all,
            "factory": RichardsonFactory([1.0, 2.0, 3.0]),
        },
        {
            "technique": "zne",
            "scale_noise": fold_all,
            "factory": RichardsonFactory([1.0, 3.0, 5.0]),
        },
        {
            "technique": "zne",
            "scale_noise": fold_all,
            "factory": LinearFactory([1.0, 2.0, 3.0]),
        },
        {
            "technique": "zne",
            "scale_noise": fold_all,
            "factory": LinearFactory([1.0, 3.0, 5.0]),
        },

    ],
)
```

```{code-cell} ipython3
cal = Calibrator(execute_calibration, frontend="qiskit", settings=RBSettings)
cal.run(log="flat")
```

As you can see above, several experiments were run, and each one has either a cross (✘) or a check (✔) to signal whether the error mitigation experiment obtained an expectation value that is better than the non-mitigated one.

```{code-cell} ipython3
calibrated_mitigated=execute_with_mitigation(circuit, execute_circuit, calibrator=cal)
mitigated=execute_with_zne(circuit, execute_circuit, factory=LinearFactory([1, 3, 5]))
unmitigated=execute_circuit(circuit)

print("ideal = \t \t",ideal)
print("unmitigated = \t \t", "{:.5f}".format(unmitigated))
print("mitigated = \t \t", "{:.5f}".format(mitigated))
print("calibrated_mitigated = \t", "{:.5f}".format(calibrated_mitigated))
```

We can see that the mitigated and calibrated-mitigated values show improvement over the unmitigated value, and that the calibrated value shows a larger improvement, achieving the objective of the calibration process.
