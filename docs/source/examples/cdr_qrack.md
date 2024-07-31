---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# CDR with Qrack as Near-Clifford Simulator

In this tutorial, Clifford Data Regression (CDR) is used with [Qrack](https://qrack.readthedocs.io/en/latest/) and a Qiskit fake backend. For more detailed information about CDR, check the [Users Guide](https://mitiq.readthedocs.io/en/stable/guide/cdr.html) 


+++

## Setup

To start, relevant modules and libraries are imported. Please ensure that the following Python modules are installed: `mitiq`, `numpy`, `pyqrack`, `cirq`, `qiskit`

+++

```{note}
In the code below the environmental variable, `QRACK_MAX_CPU_QB`, is set to `-1`. This enviroment variable sets the maximum on how many qubits can be allocated on a single QEngineCPU instance. More information can be found on the [Qrack README page](https://github.com/unitaryfund/qrack?tab=readme-ov-file#maximum-allocation-guard).
```

```{code-cell} ipython3
import numpy as np
import collections

import os
import warnings
warnings.simplefilter("ignore", np.ComplexWarning)

from pyqrack import QrackSimulator, QrackCircuit
os.environ["QRACK_MAX_CPU_QB"]="-1"

import mitiq.interface.mitiq_qiskit
from mitiq.interface.mitiq_cirq import compute_density_matrix
from mitiq import cdr, Observable, PauliString

import cirq

from qiskit.providers.fake_provider import Fake5QV1
```

## Sample Circuit

This sample circuit includes Clifford gates (`H`, `CNOT`, `RX`) and non-Clifford gates (`RZ`).

```{code-cell} ipython3
a, b = cirq.LineQubit.range(2)
circuit = cirq.Circuit(
    cirq.H.on(a),  # Clifford
    cirq.H.on(b),  # Clifford
    cirq.rz(1.75).on(a),
    cirq.rz(2.31).on(b),
    cirq.CNOT.on(a, b),  # Clifford
    cirq.rz(-1.17).on(b),
    cirq.rz(3.23).on(a),
    cirq.rx(np.pi / 2).on(a),  # Clifford
    cirq.rx(np.pi / 2).on(b),  # Clifford
)

# CDR works better if the circuit is not too short. So we increase its depth.
circuit = 5 * circuit
print(circuit)
```

## Qrack Near-Clifford Simulator

Especially when using CDR at scale, it is important to use an efficient Near-Clifford circuit simulator. In this example, Qrack will be configured and used as the Near-Clifford Simulator. The `qrack_simulate` method accepts a Cirq circuit and the number of shots as parameters. The Qrack simulator is then called and the exepectation value for `00` is returned.

```{code-cell} ipython3
def qrack_simulate(circuit: cirq.Circuit, shots=1000) -> float:
    """Returns the expectation value of 00 from the state prepared by the circuit
    executed without noise by Qrack configured as a near-Clifford simulator.
    """

    # Cirq -> Qiskit circuit
    qiskit_circ = mitiq.interface.mitiq_qiskit.to_qiskit(circuit)

    # Qiskit -> Qrack circuit
    qcircuit = QrackCircuit.in_from_qiskit_circuit(qiskit_circ) 

    # Setup the Qrack simulator and run it
    qsim = QrackSimulator(qiskit_circ.width(), isStabilizerHybrid=True, isTensorNetwork=False, isSchmidtDecomposeMulti=False, isSchmidtDecompose=False, isOpenCL=False)
    qcircuit.run(qsim)

    # Use shot measurements to return the expectation value of 00
    results = qsim.measure_shots(q=list(range(qiskit_circ.width())), s=shots)
    results = dict(collections.Counter(results))
    for key, value in results.items():
        results[key] = value / shots
    return results[0]
```

## Qiskit Fake Backend

CDR requires the use of a quantum device or a noisy simulator. In this example, a Qiskit 5 Qubit Fake Backend is used. The [Fake5QV1](https://docs.quantum.ibm.com/api/qiskit/qiskit.providers.fake_provider.Fake5QV1) uses configurations and noise settings taken previously from the 5 qubit IBM Quantum Yorktown device. The `qiskit_noisy` function takes the Cirq circuit and uses Mitiq to change it into a Qiskit circuit. After adding measurements, the circuit is run on the fake backend. The expectation value for `00` is then returned.

```{code-cell} ipython3
# Use Qiskit's Fave5QV1 as a noisy simulator
def qiskit_noisy(circuit: cirq.Circuit, shots=1000):
    """Execute the input circuit and return the expectation value of |00..0><00..0|"""

    # Cirq -> Qiskit circuit
    qiskit_circ = mitiq.interface.mitiq_qiskit.to_qiskit(circuit)

    # Add measurement gates to the circuit
    qiskit_circ.measure_all()

    # Setup the fake backend and run the circuit
    noisy_backend = Fake5QV1()
    job = noisy_backend.run(qiskit_circ, shots=shots)

    # Use the resulting counts to return the expectation value of 00
    counts = job.result().get_counts()
    ret_val =  counts[qiskit_circ.num_qubits * "0"] / shots
    return ret_val
```

## Cirq Simulator for exact result

The `compute_density_matrix` is the Cirq density matrix simulator with a Mitiq wrapper. It is used to obtain the exact `00` expectation value. This is then used to determine the accuracy of the mitigated and unmitigated reuslts.

```{code-cell} ipython3
def cirq_simulate(circuit: cirq.Circuit) -> np.ndarray:
    """Returns Tr[ρ |0⟩⟨0|] where ρ is the state prepared by the circuit 
    executed without depolarizing noise.
    """
    res = compute_density_matrix(circuit, noise_level=(0.0,))
    return res[0, 0].real
```

## Executing CDR

With the different executor functions defined for running the Qrack, Qiskit, and Cirq simulators, the `mitiq.cdr.execute_with_cdr` function can now be called.

Before getting the results from using CDR, `cirq_simulate` is called to get the ideal expectation value from the circuit. Next, the circuit is run on the fake Qiskit backend, using `qiskit_noisy` in order to obtain an unmitigated expectation value for `00`. Finally, `execute_with_cdr` is called to obtain the mitigated expectation value.

+++

```{note}
Since we are dealing with expectation values, the `observable` parameter is set to `None` in the `execute_with_cdr` call.
```

```{code-cell} ipython3
ideal_expval = cirq_simulate(circuit).round(5)
print(f"Ideal expectation value from Cirq Simulator: {ideal_expval:.3f}")

unmitigated_expval = qiskit_noisy(circuit)
print(f"Unmitigated expectation value from Qiskit Fake backend: {unmitigated_expval:.3f}")

mitigated_expval = cdr.execute_with_cdr(
    circuit,
    qiskit_noisy,
    observable=None,
    simulator=qrack_simulate,
    seed=0,
)
print(f"Mitigated expectation value with Mitiq CDR: {mitigated_expval:.3f}\n")

unmitigated_error = abs(unmitigated_expval - ideal_expval)
mitigated_error = abs(mitigated_expval - ideal_expval)

print(f"Unmitigated Error:           {unmitigated_error:.3f}")
print(f"Mitigated Error:             {mitigated_error:.3f}")

improvement_factor = unmitigated_error / mitigated_error
print(f"Improvement factor with CDR: {improvement_factor:.2f}")
```
