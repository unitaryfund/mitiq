---
jupytext:
  text_representation:
    extension: .myst
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# How Do I Use Classical Shadows Estimation?


The `mitiq.shadows` module facilitates the application of the classical shadows protocol on quantum circuits, designed for tasks like quantum state tomography or expectation value estimation. In addition this module integrates a robust shadow estimation protocol that's tailored to counteract noise. The primary objective of the classical shadow protocol is to extract information from a quantum state using repeated measurements.

The procedure can be broken down as follows:

1. `shadow_quantum_processing`: 
   - Purpose: Execute quantum processing on the provided quantum circuit.
   - Outcome: Measurement results from the processed circuit.

2. `classical_post_processing`: 
   - Purpose: Handle classical processing of the measurement results.
   - Outcome: Estimation based on user-defined inputs.

For users aiming to employ the robust shadow estimation protocol, An initial step is needed which entails characterizing the noisy quantum channel. This is done by:

0. `pauli_twirling_calibration`
   - Purpose: Characterize the noisy quantum channel.
   - Outcome: A dictionary of `calibration_results`.

1. `shadow_quantum_processing`: same as above.

2. `classical_post_processing`
   - Args: `rshadow` = True, 
           `calibration_results` = output of `pauli_twirling_calibration`
   - Outcome: Error mitigated estimation based on user-defined inputs.

Notes:
   - The calibration process is specifically designed to mitigate noise encountered during the classical shadow protocol, such as rotation and computational basis measurements. It does not address noise that occurs during state preparation.
   - Do not need to redo the calibration stage (0. `pauli_twirling_calibration`) if:
       1. The input circuit has a consistent number of qubits.
       2. The estimated observables have the same or fewer qubit support.
   - The current version supports local Pauli measurements. Upcoming documentation will offer guidance on estimating Renyi entropy and other quantum information metrics.


## Protocol Overview

The classical shadow protocol aims to create an approximate classical representation of a quantum state using minimal measurements. This approach not only characterizes and mitigates noise effectively but also retains sample efficiency and demonstrates noise resilience. For more details, see the section ([Classical Shadow Protocol and its Robust Estimation](shadows-5-theory.md)).

One can use the mitiq.shadows following the steps below:

### User-defined inputs 
Define a quantum circuit, e.g., with the output state the GHZ state, with number of qubits $n$ = `n_qubits`, 

```{code-cell} ipython3
import numpy as np
#fix random seed
np.random.seed(1)
```

```{code-cell} ipython3
import cirq

qubits = cirq.LineQubit.range(3)
num_qubits = len(qubits)
circuit = cirq.Circuit(
    cirq.H(qubits[0]),
    cirq.CNOT(qubits[0], qubits[1]),
    cirq.CNOT(qubits[1], qubits[2]),
)
print(circuit)
```

Define a executor to run the circuit on a quantum computer or noisy simulator. Noted that the robust shadow calibration can only calibrate the noisy rotation and measurement channels of classical shadow protocal, so we we define an executor satisfies

1. Noise channel added to circuit before measurement, i.e. noisy gate satisfies $U_{\Lambda_U}(M_z)_{\Lambda_{\mathcal{M}_Z}}\equiv U\Lambda\mathcal{M}_Z$.

2. Take one shot of measurement for each circuit as classical shadow protocal requires.

We define the executor function as follows as an example:


```{code-cell} ipython3
from mitiq import MeasurementResult


def cirq_executor(
    circuit: cirq.Circuit,
    noise_model_function=cirq.depolarize,
    noise_level=(0.2,),
    sampler=cirq.Simulator(),
) -> MeasurementResult:
    """
    This function returns the measurement outcomes of a circuit with noisy channel added before measurements.
    Args:
        circuit: The circuit to execute.
    Returns:
        A one shot MeasurementResult object containing the measurement outcomes.
    """
    circuit = circuit.copy()
    qubits = sorted(list(circuit.all_qubits()))
    if noise_level[0] > 0:
        noisy_circuit = cirq.Circuit()
        operations = list(circuit)
        n_ops = len(operations)
        for i, op in enumerate(operations):
            if i == n_ops - 1:
                noisy_circuit.append(
                    cirq.Moment(
                        *noise_model_function(*noise_level).on_each(*qubits)
                    )
                )
            noisy_circuit.append(op)
        circuit = noisy_circuit
    # circuit.append(cirq.Moment(*noise_model_function(*noise_level).on_each(*qubits)))
    executor = cirq_sample_bitstrings(
        circuit,
        noise_model_function=None,
        noise_level=(0,),
        shots=1,
        sampler=sampler,
    )
    return executor
```

with the above definition of excutor, one can define a noisy executor as followings, for example, with bif_flip probability of 0.1:


```{code-cell} ipython3
from functools import partial

noisy_executor = partial(
    cirq_executor,
    noise_level=(0.1,),
    noise_model_function=cirq.bit_flip,
)
```

### 0. Calibration Stage

One can simply skip this stage if one just want to perform classical shadow protocal without calibration or the calibration data is already available from previous runs.

Together with the imput of total calibration rounds $R$ = `num_total_measurements_calibration` and the Number of groups of "median of means" used for calibration $K$ = `k_calibration`, one can characterize the noisy quantum channel (see [rshadow tutorial for cirq](../examples/rshadows_tutorial.md) for clear explaination) by running the following code:

```{code-cell} ipython3
from mitiq.shadows import *
from mitiq.interface.mitiq_cirq.cirq_utils import (
    sample_bitstrings as cirq_sample_bitstrings,
)

f_est = pauli_twirling_calibrate(
    k_calibration=1,
    locality=2,
    qubits=qubits,
    executor=noisy_executor,
    num_total_measurements_calibration=5000,
)
f_est
```

the varible `locality` is the maximum number of qubits that our interested operators acting on. e.g. if our interested operator is a sequence of two point correlation function $\{\langle Z_iZ_{i+1}\rangle\}_{0\leq i\leq n-1}$, then `locality` = 2. Noted here, one could also split the calibration process into two stage:

01. `shadow_quantum_processing`
   - Outcome: Get quantum measurement result of the calibration circuit $|0\rangle^{\otimes n}$ `zero_state_shadow_outcomes`.

02. `pauli_twirling_calibration`
   - Outcome: A dictionary of `calibration_results`.
For more details, please refer to the see [rshadow tutorial for cirq](../examples/rshadows_tutorial.md)

### 1. Quantum Processing
In this step, we obtain classical shadow snapshots (before apply the invert channel) from the input state

#### 1.1 Add Rotation Gate and Meausure the Rotated State in Computational Basis
At present, the implementation supports random Pauli measurement. This is equivalent to randomly sampling $U$ from the local Clifford group $Cl_2^n$, followed by a $Z$-basis measurement (see [shadow tutorial for cirq](../examples/shadows_tutorial.md) for clear explaination).

#### 1.2 Get the Classical Shadows
One can obtain the list of measurement results of local Pauli measurement in terms of bitstrings, and the related Pauli-basis measured in terms of strings by calling the `shadow_quantum_processing` function with given the total number of snapshots `num_total_measurements_shadow`:


```{code-cell} ipython3
shadow_measurement_output = shadow_quantum_processing(
    circuit,
    noisy_executor,
    num_total_measurements_shadow=5000,
)
```
                                                                     

for example, print out one of those measurement outcome and meausurement basis:


```{code-cell} ipython3
print("one snapshot measurement reslut = ", shadow_measurement_output[0][0])
print("one snapshot measurement basis = ", shadow_measurement_output[1][0])
```

### 2. Classical Post-Processing
In this step, we estimate our interested object by classical shadow protocol. 

#### 2.1 Example: Operator Expectation Value Esitimation
For example, if we want to estimate the expectation value of a sequence of two point correlation functions $\{\langle Z_iZ_{i+1}\rangle\}_{0\leq i\leq n-1}$, we will define the operators by Mitiq Paulistrings:

```{code-cell} ipython3
from mitiq import PauliString

two_pt_correlations = [
    PauliString("ZZ", support=(i, i + 1), coeff=1)
    for i in range(0, num_qubits - 1)
]
for i in range(0, num_qubits - 1):
    print(two_pt_correlations[i]._pauli)
```

One can estimation the expectation value of the observables with the previous classical shadows. One can get the estimation of expectation values without/with calibration:


```{code-cell} ipython3
est_corrs = classical_post_processing(
    shadow_outcomes=shadow_measurement_output,
    use_calibration=False,
    observables=two_pt_correlations,
    k_shadows=1,
)
cal_est_corrs = classical_post_processing(
    shadow_outcomes=shadow_measurement_output,
    use_calibration=True,
    calibration_results=f_est,
    observables=two_pt_correlations,
    k_shadows=1,
)
```

and compare the results with the exact values:


```{code-cell} ipython3
expval_exact = []
state_vector = circuit.final_state_vector()
for i, pauli_string in enumerate(two_pt_correlations):
    exp = pauli_string._pauli.expectation_from_state_vector(
        state_vector, qubit_map={q: i for i, q in enumerate(qubits)}
    )
    expval_exact.append(exp.real)
```


```{code-cell} ipython3
print("classical shadow estimation:", est_corrs)
print("Robust shadow estimation   :", cal_est_corrs)
print(
    "Exact expectation values:",
    "'Z(q(0))*Z(q(1))':",
    expval_exact[0],
    "'Z(q(1))*Z(q(2))':",
    expval_exact[1],
)
```


#### 2.2 Example: GHZ State Reconstruction
Let's test the shadow state reconstruction and its calibrated version on a simple example. We will use the 3-qubit GHZ state, previously defined, as an initial state. Similarly to the previous example, we will use the same circuit both to generate the shadow state and to reconstruct it. Let's calculate the Pauli fidelity $f_b$ used to characterize the noisy quantum channel $\mathcal{M}=\sum_{b\in\{0,1\}^n}f_b\Pi_b$:



```{code-cell} ipython3
noisy_executor = partial(
    cirq_executor,
    noise_level=(0.2,),
    noise_model_function=cirq.bit_flip,
)

f_est = pauli_twirling_calibrate(
    k_calibration=1,
    qubits=qubits,
    executor=noisy_executor,
    num_total_measurements_calibration=50000,
)
f_est
```


Same as the previous case, the quantum processing of the shadow protocol is done by the following function:


```{code-cell} ipython3
shadow_measurement_output = shadow_quantum_processing(
    circuit,
    noisy_executor,
    num_total_measurements_shadow=50000,
)
```

```{code-cell} ipython3
est_corrs = classical_post_processing(
    shadow_outcomes=shadow_measurement_output,
    use_calibration=False,
    state_reconstruction=True,
)
cal_est_corrs = classical_post_processing(
    shadow_outcomes=shadow_measurement_output,
    use_calibration=True,
    calibration_results=f_est,
    state_reconstruction=True,
)
```


```{code-cell} ipython3
from mitiq.shadows.shadows_utils import operator_ptm_vector_rep

ghz_state = circuit.final_state_vector().reshape(-1, 1)
ghz_true = ghz_state @ ghz_state.conj().T
ptm_ghz_state = operator_ptm_vector_rep(ghz_true)
```


```{code-cell} ipython3
from mitiq.shadows.shadows_utils import fidelity

fidelity_shadow = fidelity(ghz_true, est_corrs["reconstructed_state"])
fidelity_shadow_calibrated = fidelity(
    ptm_ghz_state, cal_est_corrs["reconstructed_state"]
)
print(
    f"fidelity between true state and shadow reconstruced state {fidelity_shadow}"
)
print(
    f"fidelity between true state and rshadow reconstruced state {fidelity_shadow_calibrated}"
)
```
<!-- 
    fidelity between true state and shadow reconstruced state 0.3598774999999995
    fidelity between true state and rshadow reconstruced state 0.9994545201218017 -->
