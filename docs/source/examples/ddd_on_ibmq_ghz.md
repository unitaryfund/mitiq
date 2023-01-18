---
jupytext:
  text_representation:
    extension: .myst
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Digital dynamical decoupling (DDD) with Qiskit on GHZ Circuits

In this notebook DDD is applied to improve the success rate of the computation. 
In DDD, sequences of gates are applied to slack windows, i.e. single-qubit idle windows, in a quantum circuit. 
Applying such sequences can reduce the coupling between the qubits and the environment, mitigating the effects of noise. 
For more information on DDD, see the section [DDD section of the user guide](../guide/ddd.myst).

## Setup

We begin by importing the relevant modules and libraries that we will require
for the rest of this tutorial.

```{code-cell} ipython3
import functools
from typing import List, Tuple, Dict

# Plotting imports.
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.size": 15})
%matplotlib inline

# Third-party imports.
import qiskit
import cirq

import numpy as np

# Mitiq imports.
from mitiq.benchmarks.ghz_circuits import generate_ghz_circuit
from mitiq.interface.conversions import convert_to_mitiq, convert_from_mitiq
from mitiq import ddd
```

```{code-cell} ipython3
USE_REAL_HARDWARE = False
```

## Define parameters

```{code-cell} ipython3
# Random seed for circuit generation.
seed = 1

# Total number of shots to use.
shots: int = 10000

# Qubits to use on the experiment.
num_qubits = [4, 6, 8]

# Average results over this many trials (circuit instances) at each depth.
trials = 3
```

## Define the circuit

We use Greenberger-Horne-Zeilinger (GHZ) circuits to benchmark the performance of the device.
GHZ circuits are designed such that only two bitstrings |00...0> and |11...1>
should be sampled, with $P_0 = P_1 = 0.5$.
As noted in *Mooney et al. (2021)* {cite}`Mooney_2021`, when GHZ circuits are run on a device, any other measured bitstrings are due to noise.
In this example, the GHZ circuit is applied, followed by its inverse. 
Therefore the frequency of the |00...0> bitstring is our target metric.

```{code-cell} ipython3
def get_circuit(num_qubits) -> Tuple[qiskit.QuantumCircuit, List[int]]:
    ghz_circuit = generate_ghz_circuit(num_qubits, "qiskit")
    inverted = ghz_circuit.inverse()
    circuit = ghz_circuit.compose(inverted)
    circuit.measure_all()

    correct_bitstring = [0] * num_qubits

    return circuit, correct_bitstring
```

## Define the executor

Now that we have a circuit, we define the `execute` function which inputs a circuit and returns an expectation value - here, the
frequency of sampling the correct bitstring.

**Importantly**, since DDD is designed to mitigate time-correlated (non-Markovian) noise, we simulate a particular noise model consisting of
systematic $R_Z$ rotations applied to each qubit after each moment. 
This corresponds to a dephasing noise which is strongly time-correlated and, therefore, likely to be mitigated by DDD.

```{code-cell} ipython3
if qiskit.IBMQ.stored_account() and USE_REAL_HARDWARE:
    provider = qiskit.IBMQ.load_account()
    backend = provider.get_backend("ibmq_qasm_simulator")  # Set quantum computer here!
else:
    # Default to a simulator.
    backend = (qiskit.Aer.get_backend("qasm_simulator"),)


def ibmq_executor(
    circuit: qiskit.QuantumCircuit,
    shots: int,
    correct_bitstring: List[int],
    noisy_sim: bool = True,
) -> float:
    """Returns the expectation value to be mitigated.

    Args:
        circuit: Circuit to run.
        shots: Number of times to execute the circuit to compute the expectation value.
    """

    if USE_REAL_HARDWARE:
        # Run the circuit on hardware
        job = qiskit.execute(
            experiments=circuit,
            backend=backend,
            optimization_level=0,  # Important to preserve folded gates.
            shots=shots,
        )

    else:
        circuit_to_run = circuit.copy()

        # Simulate the circuit with noise
        if noisy_sim:
            converted, circuit_type = convert_to_mitiq(circuit)
            noisy = converted.with_noise(cirq.rz(0.05))
            circuit_to_run = convert_from_mitiq(noisy, circuit_type)
            job = qiskit.execute(
                experiments=circuit_to_run,
                backend=qiskit.Aer.get_backend("qasm_simulator"),
                optimization_level=0,
                shots=shots,
            )
            all_counts = job.result().get_counts()
            P0 = (
                all_counts.get(
                    ("".join(map(str, correct_bitstring))).replace("", " ")[1:-1], 0.0
                )
                / shots
            )
            return P0

        else:
            job = qiskit.execute(
                experiments=circuit_to_run,
                backend=qiskit.Aer.get_backend("qasm_simulator"),
                optimization_level=0,
                shots=shots,
            )

    # Convert from raw measurement counts to the expectation value
    all_counts = job.result().get_counts()
    P0 = all_counts.get("".join(map(str, correct_bitstring)), 0.0) / shots
    return P0
```

## Select the DDD sequences to be applied
We now import a DDD _rule_ from Mitiq, i. e., a function that generates DDD sequences of different length.
In this example, we opt for YY sequences (pairs of Pauli Y operations).

```{code-cell} ipython3
from mitiq import ddd

rule = ddd.rules.yy
```

## Sample bitstrings from GHZ circuits

```{code-cell} ipython3
true_values, noisy_values, ddd_values = [], [], []

for nq in num_qubits:
    true_nqubits_values, noisy_nqubits_values, ddd_nqubits_values = [], [], []
    circuit, correct_bitstring = get_circuit(num_qubits=nq)

    for trial in range(trials):

        true_nqubits_values.append(
            ibmq_executor(circuit, shots, correct_bitstring, noisy_sim=False)
        )

        noisy_nqubits_values.append(
            ibmq_executor(circuit, shots, correct_bitstring, noisy_sim=True)
        )

        noisy_executor = functools.partial(
            ibmq_executor,
            shots=shots,
            correct_bitstring=correct_bitstring,
        )

        ddd_nqubits_values.append(
            ddd.execute_with_ddd(
                circuit,
                noisy_executor,
                rule=rule,
            )
        )

    true_values.append(true_nqubits_values)
    noisy_values.append(noisy_nqubits_values)
    ddd_values.append(ddd_nqubits_values)
```

Now we can visualize the results.

```{code-cell} ipython3
avg_true_values = np.average(true_values, axis=1)
avg_noisy_values = np.average(noisy_values, axis=1)

std_true_values = np.std(true_values, axis=1, ddof=1)
std_noisy_values = np.std(noisy_values, axis=1, ddof=1)

avg_ddd_values = np.average(ddd_values, axis=1)
std_ddd_values = np.std(ddd_values, axis=1, ddof=1)

plt.figure(figsize=(9, 5))

plt.plot(num_qubits, avg_true_values, "--", label="True", lw=2)
eb = plt.errorbar(
    num_qubits, avg_noisy_values, yerr=std_noisy_values, label="Raw", ls="-."
)
eb[-1][0].set_linestyle("-.")
plt.errorbar(num_qubits, avg_ddd_values, yerr=std_ddd_values, label="DDD")

plt.title(
    f"""Simulator with GHZ circuits using DDD {num_qubits} \nqubits, {trials} trials."""
)
plt.xlabel("Number of Qubits")
plt.ylabel("Expectation value")
_ = plt.legend()
```

We can see that on average DDD improves the expectation value at each circuit width. The improvement slightly increases with circuit size, which is expected given the strongly time-correlated dephasing noise applied in this example. In general, real hardware would exhibit a different noise model from what is shown here, but real devices usually have some time-correlated noise that can be mitigated by dynamical decoupling.
