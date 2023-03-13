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

# Composing techniques: REM + ZNE

1. Define unmitigated executor
2. compute unmitigated expval
3. mitigate with REM
4. mitigate with ZNE
5. mitigate with REM + ZNE
6. compare results


Composing techniques

```{code-cell} ipython3
import cirq
import numpy as np
from mitiq.benchmarks import generate_rb_circuits
from mitiq import MeasurementResult, Observable, PauliString, raw
```

Randomized benchmarking circuits in Mitiq

```{code-cell} ipython3
circuit = generate_rb_circuits(2, 5)[0]
```

Define the executor (standard verbiage and link)

```{code-cell} ipython3
def execute(circuit: cirq.Circuit, noise_level: float = 0.005, p0: float = 0.05) -> MeasurementResult:
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

Obtain unmitigated expectation value of the ZI + ZZ observable. In the ideal case the result should be 2, but the unmitigated result is impacted by depolarizing and readout errors.

```{code-cell} ipython3
# step 2
obs = Observable(PauliString("ZI"), PauliString("IZ"))
result = raw.execute(circuit, execute, obs)
print(result)
```

```{code-cell} ipython3
from functools import partial
ideal = raw.execute(circuit, partial(execute, noise_level=0, p0=0), obs)
```

Next we apply readout error mitigation (REM). (Brief explanation + link to guide)

```{code-cell} ipython3
# step 3 (REM)
from mitiq import rem
p0 = p1 =0.05
icm = rem.generate_inverse_confusion_matrix(2, p0, p1)
rem_executor = rem.mitigate_executor(execute, inverse_confusion_matrix=icm)

obs.expectation(circuit, rem_executor)
```

Now apply zero noise extrapolation (ZNE). (Brief explanation + link to guide)

```{code-cell} ipython3
# step 4 (ZNE)
from mitiq import zne

zne_executor = zne.mitigate_executor(execute, observable=obs)

zne_executor(circuit)
```

Finally, we apply a combination of REM and ZNE. REM is applied first to minimize the impact of measurement errors on the extrapolated result in ZNE.

```{code-cell} ipython3
# step 5 (REM + ZNE)

both_executor = zne.mitigate_executor(rem_executor, observable=obs)

both_executor(circuit)
```
