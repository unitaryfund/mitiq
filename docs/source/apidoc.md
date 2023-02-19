# API-doc

```{eval-rst}
.. automodule:: mitiq
   :members:

.. modules: alphabetical order
```

## Benchmarks

### Mirror Circuits

```{eval-rst}
.. automodule:: mitiq.benchmarks.mirror_circuits
   :members:
```

### Randomized Benchmarking Circuits

```{eval-rst}
.. automodule:: mitiq.benchmarks.randomized_benchmarking
   :members:
```

### GHZ Circuits
```{eval-rst}
.. automodule:: mitiq.benchmarks.ghz_circuits
   :members:
```

### Quantum Volume Circuits
```{eval-rst}
.. automodule:: mitiq.benchmarks.quantum_volume_circuits
   :members:
```

## Circuit types and result types

```{eval-rst}
.. autoclass:: mitiq.typing.QPROGRAM
```

```{eval-rst}
.. autoclass:: mitiq.typing.QuantumResult
```

```{eval-rst}
.. autoclass:: mitiq.typing.Bitstring
```

```{eval-rst}
.. autoclass:: mitiq.typing.MeasurementResult
   :members:
```


## Clifford Data Regression

### Clifford Data Regression (High-Level Tools)
```{eval-rst}
.. automodule:: mitiq.cdr.cdr
   :members:
```

### Clifford Training Data
```{eval-rst}
.. automodule:: mitiq.cdr.clifford_training_data
   :members:
```

### Data Regression
```{eval-rst}
.. automodule:: mitiq.cdr.data_regression
   :members:
```

See Ref. :cite:`Czarnik_2021_Quantum` for more details on these methods.


## Mitiq - Braket

### Conversions
```{eval-rst}
.. automodule:: mitiq.interface.mitiq_braket.conversions
   :members:
```

## Mitiq - Cirq

### Cirq Utils
```{eval-rst}
.. automodule:: mitiq.interface.mitiq_cirq.cirq_utils
   :members:
```

## Mitiq - PyQuil

### Conversions
```{eval-rst}
.. automodule:: mitiq.interface.mitiq_pyquil.conversions
   :members:
```

## Mitiq - Qiskit

### Conversions
```{eval-rst}
.. automodule:: mitiq.interface.mitiq_qiskit.conversions
   :members:
```

### Qiskit Utils
```{eval-rst}
.. automodule:: mitiq.interface.mitiq_qiskit.qiskit_utils
   :members:
```

## Digital Dynamical Decoupling

### Digital Dynamical Decoupling (High-Level Tools)
```{eval-rst}
.. automodule:: mitiq.ddd.ddd
   :members:
```

### Insertion
```{eval-rst}
.. automodule:: mitiq.ddd.insertion
   :members:
```

### Rules
```{eval-rst}
.. automodule:: mitiq.ddd.rules.rules
   :members:
```

## Executors
```{eval-rst}
.. automodule:: mitiq.executor.executor
   :members:
```

## Observables

### Observable
```{eval-rst}
.. automodule:: mitiq.observable.observable
   :members:
```

### Pauli
```{eval-rst}
.. automodule:: mitiq.observable.pauli
   :members:
```

## Probabilistic Error Cancellation

### Probabilistic Error Cancellation (High-Level Tools)
```{eval-rst}
.. automodule:: mitiq.pec.pec
   :members:
```

### Quasi-Probability Representations
```{eval-rst}
.. automodule:: mitiq.pec.representations.optimal
   :members:

.. automodule:: mitiq.pec.representations.damping
   :members:

.. automodule:: mitiq.pec.representations.depolarizing
   :members:
```

### Learning-based PEC
```{eval-rst}
.. automodule:: mitiq.pec.representations.biased_noise
   :members:

.. automodule:: mitiq.pec.representations.learning
   :members:
```

### Sampling from a Noisy Decomposition of an Ideal Operation
```{eval-rst}
.. automodule:: mitiq.pec.sampling
   :members:
```

### Probabilistic Error Cancellation Types
```{eval-rst}
.. automodule:: mitiq.pec.types.types
   :members:
```

### Utilities for Quantum Channels
```{eval-rst}
.. automodule:: mitiq.pec.channels
   :members:
```

## Raw

### Run experiments without error mitigation (raw results)
```{eval-rst}
.. automodule:: mitiq.raw.raw
```

## Readout-Error Mitigation

### Postselection
```{eval-rst}
.. automodule:: mitiq.rem.post_select
   :members:
```

### REM Technique
```{eval-rst}
.. automodule:: mitiq.rem.rem
   :members:
```

## Zero Noise Extrapolation

### Zero Noise Extrapolation (High-Level Tools)
```{eval-rst}
.. automodule:: mitiq.zne.zne
   :members:
```

### Inference and Extrapolation: Factories
```{eval-rst}
.. automodule:: mitiq.zne.inference
   :members:
```

### Noise Scaling: Unitary Folding
```{eval-rst}
.. automodule:: mitiq.zne.scaling.folding
   :members:
```

### Noise Scaling: Identity Insertion Scaling
```{eval-rst}
.. automodule:: mitiq.zne.scaling.identity_insertion
   :members:
```

### Noise Scaling: Parameter Calibration
```{eval-rst}
.. automodule:: mitiq.zne.scaling.parameter
   :members:
```

## Calibration

```{eval-rst}
.. automodule:: mitiq.calibration.calibrator
   :members:

.. automodule:: mitiq.calibration.settings
   :members:
```