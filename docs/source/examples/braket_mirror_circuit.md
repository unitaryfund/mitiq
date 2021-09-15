---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.12.0
kernelspec:
  display_name: conda_braket
  language: python
  name: conda_braket
---

+++ {"id": "cm2KQsakEAay"}

# Mitiq with Braket

+++ {"id": "UmvMUUILKQT3"}

This notebook shows improved performance on a mirror circuit benchmark with zero-noise extrapolation on Rigetti Aspen-9 via Amazon Braket.

> Note: This notebook is intended to be run through the Amazon Web Services (AWS) console - that is, by uploading the `.ipynb` file to https://console.aws.amazon.com/braket/ and running from there. This requires an AWS account. **Without an AWS account, you can still run the notebook on a simulator, but results will be noiseless and so zero-noise extrapolation will have no effect**.

+++ {"id": "T8z_3iTaQEP5"}

## Setup

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: m8Vs4oIY1cct
outputId: 06803114-6d80-49bd-907b-49d2cb9dc0af
---
try:
    import mitiq
except ImportError:
    !pip install git+https://github.com/unitaryfund/mitiq --quiet
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: Rw6FhzvDQKT8
outputId: f354b901-2c2e-4242-b1fc-bd1c8e6c3b70
---
!pip install amazon-braket-sdk --quiet
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: FNwVmD481d7n
outputId: 574adb14-d5c3-49f9-cbdb-7be5566fabcc
---
from typing import Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from braket.aws import AwsDevice
from braket.circuits import Circuit, gates
from mitiq import benchmarks, zne
```

+++ {"id": "_W0o8Ho-QEP6"}

## Choose a device to run on

+++ {"id": "9uaLZYRgQEP6"}

We first choose a device to run on.

> Note: Verbatim compiling in Braket - a necessary feature to perform zero-noise extrapolation - is currently only available on Rigetti devices.

```{code-cell} ipython3
:id: fP58xeHuQEP7

try:
    aws_device = AwsDevice("arn:aws:braket:::device/qpu/rigetti/Aspen-9")
except:
    from braket.devices import LocalSimulator

    aws_device = LocalSimulator()

on_aws = aws_device.name != "StateVectorSimulator"
```

+++ {"id": "eYQ1yPOIGQcq"}

## Define the circuit

+++ {"id": "rkCiSuTrQEP7"}

We use mirror circuits to benchmark the performance of the device. Mirror circuits, introduced in https://arxiv.org/abs/2008.11294, are designed such that only one bitstring should be sampled. When run on a device, any other measured bitstrings are due to noise. The frequency of the correct bitstring is our target metric.

> Note: Mirror circuits build on Loschmidt echo circuits - i.e., circuits of the form $U U^\dagger$ for some unitary $U$. Loschmidt echo circuits are good benchmarks but have shortcomings - e.g., they are unable to detect coherent errors. Mirror circuits add new features to account for these shortcomings. For more background, see https://arxiv.org/abs/2008.11294.

To define a mirror circuit, we need the device graph. We will use a subgraph of the device, and our first step is picking a subgraph with good qubits.

+++ {"id": "YuRaURxlQEP7"}

### Pick good qubits

+++ {"id": "gc4bleLkQEP8"}

The full device graph is shown below.

```{code-cell} ipython3
:id: 1sAgIojiQEP8

if on_aws:
    device_graph = aws_device.topology_graph
    nx.draw_kamada_kawai(device_graph, with_labels=True)
```

+++ {"id": "HzhVB2veQEP8"}

To pick good qubits, we pull the latest calibration report in the next two cells. The first cell shows two-qubit calibration data sorted by best `CZ` fidelity.

```{code-cell} ipython3
:id: Gva30wUjQEP8

if on_aws:
    twoq_data = pd.DataFrame.from_dict(aws_device.properties.provider.specs["2Q"]).T
    twoq_data.sort_values(by=["fCZ"], ascending=False).head()
```

+++ {"id": "22MuXvWJQEP9"}

And the next cell shows single-qubit calibration data sorted by best readout (RO) fidelity.

```{code-cell} ipython3
:id: b2Chcm-uQEP9

if on_aws:
    oneq_data = pd.DataFrame.from_dict(aws_device.properties.provider.specs["1Q"]).T
    oneq_data.sort_values(by=["fRO"], ascending=False).head()
```

+++ {"id": "v78rTlqLQEP9"}

Using this calibration data as a guide, we pick good qubits and visualize the device subgraph that we will run on.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 319
id: qRSvHoPrQEP9
outputId: 80181afa-ec13-4ac0-d0ac-e02008e89f0c
---
connectivity_graph = device_graph.subgraph((20, 21)) if on_aws else nx.complete_graph(2)
nx.draw(connectivity_graph, with_labels=True)
```

+++ {"id": "MB7XtN4HQEP-"}

### Generate mirror circuit

+++ {"id": "HQJn-R5yQEP-"}

Now that we have the device (sub)graph, we can generate a mirror circuit and the bitstring it should sample as follows.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: 4Oilf6zxBEpu
outputId: 9cb75f30-ef46-48c4-c19e-ea2233386809
---
circuit, correct_bitstring = benchmarks.generate_mirror_circuit(
    nlayers=1, 
    two_qubit_gate_prob=1.0,
    two_qubit_gate_name="CZ",
    connectivity_graph=connectivity_graph,
    seed=1,
    return_type="braket",
)
print(circuit)
print("\nShould sample:", correct_bitstring)
```

+++ {"id": "xknaNhr5QEP-"}

### Compilation

+++ {"id": "cjMJSWpsQEP_"}

When using verbatim compiling on Braket, every gate must be a native hardware gate. Some single-qubit gates in the above circuit are not natively supported by Rigetti. We account for this with the quick-and-dirty compiler below.

```{code-cell} ipython3
:id: n09dK_I5QEP_

def compile_to_rigetti_gateset(circuit: Circuit) -> Circuit:
    compiled = Circuit()

    for instr in circuit.instructions:
        if isinstance(instr.operator, gates.Vi):
            compiled.add_instruction(gates.Instruction(gates.Rx(-np.pi / 2), instr.target))
        elif isinstance(instr.operator, gates.V):
            compiled.add_instruction(gates.Instruction(gates.Rx(np.pi / 2), instr.target))
        elif isinstance(instr.operator, gates.Ry):
            compiled.add_instruction(gates.Instruction(gates.Rx(-np.pi / 2), instr.target))
            compiled.add_instruction(gates.Instruction(gates.Rz(instr.operator.angle), instr.target))
            compiled.add_instruction(gates.Instruction(gates.Rx(np.pi / 2), instr.target))
        elif isinstance(instr.operator, gates.Y):
            compiled.add_instruction(gates.Instruction(gates.Rx(-np.pi / 2), instr.target))
            compiled.add_instruction(gates.Instruction(gates.Rz(np.pi), instr.target))
            compiled.add_instruction(gates.Instruction(gates.Rx(np.pi / 2), instr.target))
        elif isinstance(instr.operator, gates.X):
            compiled.add_instruction(gates.Instruction(gates.Rx(np.pi / 2), instr.target))
            compiled.add_instruction(gates.Instruction(gates.Rx(np.pi / 2), instr.target))
        elif isinstance(instr.operator, gates.Z):
            compiled.add_instruction(gates.Instruction(gates.Rz(np.pi), instr.target))
        elif isinstance(instr.operator, gates.S):
            compiled.add_instruction(gates.Instruction(gates.Rz(np.pi / 4), instr.target))
        elif isinstance(instr.operator, gates.Si):
            compiled.add_instruction(gates.Instruction(gates.Rz(-np.pi / 4), instr.target))
        else:
            compiled.add_instruction(instr)
    
    return compiled
```

+++ {"id": "o3xJMevgGSdU"}

## Define the executor

+++ {"id": "557hZ6A2QEQA"}

Now that we have a circuit, we define the `execute` function which inputs a circuit and returns an expectation value - here, the frequency of sampling the correct bitstring.

```{code-cell} ipython3
:id: l3jYN2ZVxPAf

def execute(
    circuit: Circuit,
    shots: int = 1_000, 
    s3_folder: Tuple[str, str] = ("bucket", "folder/"),
) -> float:
    # Add verbatim compiling so that zero-noise extrapolation can be used.
    if on_aws:
        circuit = Circuit().add_verbatim_box(compile_to_rigetti_gateset(circuit))
    
    # Run the circuit and return the frequency of sampling the correct bitstring.
    if on_aws:
        aws_task = aws_device.run(circuit, s3_folder, disable_qubit_rewiring=True, shots=shots)
    else:
        aws_task = aws_device.run(circuit, shots=shots)
    return aws_task.result().measurement_probabilities.get("".join(map(str, correct_bitstring)), 0.0)
```

+++ {"id": "_CavaAt6GUpE"}

## Noisy value

+++ {"id": "L7oXjOKqQEQA"}

The result of running the example mirror circuit without zero-noise extrapolation is shown below.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: VqOdtHKmE2qh
outputId: 95ae5a02-2f20-4e9d-8c5a-e2e199d207ba
---
noisy_value = execute(circuit)
print("Noisy value:", noisy_value)
```

+++ {"id": "gYjhZGTIGWiG"}

## Mitigated value

+++ {"id": "Xl19JfqEQEQB"}

The result of running the example mirror circuit with zero-noise extrapolation is shown below.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: sEgBcVjR0_du
outputId: d8e45bef-bcd8-4948-b88e-521c0821651e
---
zne_value = zne.execute_with_zne(
    circuit, 
    execute, 
    scale_noise=zne.scaling.fold_global, 
    factory=zne.inference.PolyFactory(scale_factors=[1, 3, 5], order=2)
)
print("ZNE value:", zne_value)
```

+++ {"id": "R7VjkjFYQEQB"}

In this simple example, we see that zero-noise extrapolation improves the result. (Recall that the noiseless value is `1.0`.)

+++ {"id": "syCmo4dnQEQB"}

## Survival probability vs. depth

+++ {"id": "nIbjK6heQEQC"}

Now we run the same experiment above but varying the depth (`nlayers`) of the mirror circuit. We also average over several mirror circuits at each depth.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: NvewPDsxQEQC
outputId: 80b9c9c5-c7a6-4c28-b178-b8286caa9044
---
# Experiment parameters.
nlayers_values = list(range(1, 20, 2))
ntrials = 4

# To store results.
noisy_values = []
zne_values = []

# Run the experiment and store results.
for nlayers in nlayers_values:
    for i in range(ntrials):
        print(f"Status: nlayers = {nlayers}, trial = {i + 1}.", end="\r")
        circuit, correct_bitstring = benchmarks.generate_mirror_circuit(
            nlayers=nlayers, 
            two_qubit_gate_prob=1.0,
            two_qubit_gate_name="CZ",
            connectivity_graph=connectivity_graph,
            seed=i,
            return_type="braket",
        )
    
        noisy_values.append(execute(circuit))
        zne_values.append(
            zne.execute_with_zne(
                circuit, 
                execute, 
                scale_noise=zne.scaling.fold_global, 
                factory=zne.inference.PolyFactory(scale_factors=[1, 3, 5], order=2)),
        )
```

+++ {"id": "DHHGF1cJQEQC"}

Now we can visualize the results.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 348
id: AVst1g2oQEQC
outputId: 8d5ed654-25bb-479c-ceb4-4b5326ebb3fd
---
average_zne_values = np.average(np.array(zne_values).reshape((len(nlayers_values), ntrials)), axis=1)
average_noisy_values = np.average(np.array(noisy_values).reshape((len(nlayers_values), ntrials)), axis=1)

plt.rcParams.update({"font.family": "serif", "font.size": 16})
plt.figure(figsize=(9, 5))

plt.plot(nlayers_values, average_zne_values, "--o", label="ZNE")
plt.plot(nlayers_values, average_noisy_values, "--o", label="Raw")

plt.xlabel("Circuit depth")
plt.ylabel("Survival probability")

plt.legend()
plt.show();
```

+++ {"id": "SCFiTcYMQEQC"}

We see that zero-noise extrapolation on average improves the survival probability at each depth.
