---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

+++ {"id": "wjnIQWX8d5UY"}

# Mitiq paper codeblocks

Codeblocks from the main text of the [Mitiq paper](https://arxiv.org/abs/2009.04417) [1].

+++ {"id": "DRplhovffVfk"}

**Codeblock 1**

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: YTOaB_3tdymW
outputId: 38c2ce43-463f-4ab8-e402-fbc1d5877ce3
---
# !pip install mitiq --quiet
```

+++ {"id": "Pghlsxv0fXA2"}

**Codeblock 2**

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: Qi1qvROzd0nT
outputId: b0419328-2fba-42eb-b27b-a8b20ed24f13
---
import mitiq


mitiq.about()
```

+++ {"id": "lK7iAVVShWu3"}

**Codeblock 4**

+++ {"id": "IXykFfZ6hYW7"}

> Note: The paper just shows the signature of an executor function, but here we explicitly define one to use in examples.

```{code-cell} ipython3
:id: PyPWG95HhYfj

import cirq
from mitiq.interface import accept_any_qprogram_as_input


@accept_any_qprogram_as_input
def executor(circuit: mitiq.QPROGRAM) -> float:
    return cirq.DensityMatrixSimulator().simulate(
        circuit.with_noise(cirq.depolarize(p=0.01))
    ).final_density_matrix[0, 0].real
```

+++ {"id": "LxQh9Vehh_2Q"}

**Codeblock 5**

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: eLIxAhYwiC0m
outputId: e9f253a9-0777-46c7-b586-48bef61db5d2
---
from mitiq import zne


circuit = cirq.Circuit([cirq.X.on(cirq.LineQubit(0))] * 50)

zne_value = zne.execute_with_zne(circuit, executor)
print("ZNE value:", zne_value)
```

+++ {"id": "3Q2MyAxBiW0C"}

**Codeblock 6**

+++ {"id": "CqiFB4q0ib06"}

> Note: The paper shows pseudocode; here we show an example.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: aHvZAKymicS4
outputId: 1e89cdf9-e7ba-416c-9d4c-16f6f5a7ccf5
---
zne_value = zne.execute_with_zne(
    circuit,
    executor,
    scale_noise=zne.scaling.fold_global,
    factory=zne.inference.ExpFactory(scale_factors=[1.0, 3.0, 5.0, 7.0, 9.0])
)
print("ZNE value:", zne_value)
```

+++ {"id": "VAGU7qZAjKzD"}

**Codeblock 7**

+++ {"id": "-jbnFbQkjN5J"}

> Note: The paper shows pseudocode; here we show an example.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: kMS4r4iAjNL3
outputId: 461f6d11-b45a-4f99-e8e4-d74f20651e9e
---
from mitiq import pec

representation = pec.represent_operation_with_local_depolarizing_noise(
    ideal_operation=cirq.Circuit(circuit[0]), noise_level=0.01
)
print("Representation of ideal operation:", representation, sep="\n\n")

pec_value = pec.execute_with_pec(
    circuit,
    executor,
    representations=[representation],
    num_samples=10,  # Remove argument or increase for better accuracy.
)
print("\n\nPEC value:", pec_value)
```

+++ {"id": "UvTIzr0ZfYuI"}

**Codeblock 8**

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: ULlw67NGecNd
outputId: bc1dd527-4e32-4252-e2cd-3496a46387f7
---
qreg = cirq.LineQubit.range(2)
circ = cirq.Circuit(
    cirq.ops.H.on(qreg[0]),
    cirq.ops.CNOT.on(qreg[0] , qreg[1])
)
print("Original circuit:", circ, sep="\n")
```

+++ {"id": "_5Q3P1DJfcar"}

**Codeblock 9**

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: tSWVDcJIeuck
outputId: e9d0c874-8814-4da0-c54b-a6881d3e614a
---
folded = zne.scaling.fold_gates_from_left(
    circ, scale_factor=2
)
print("Folded circuit:", folded, sep="\n")
```

+++ {"id": "4ocoSlNIfnV0"}

**Codeblock 10**

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: k8gLF_fPfneX
outputId: ac0ee6c0-4ad4-4524-dc3d-7c9a413fe2ba
---
folded = zne.scaling.fold_gates_from_right(
    circ, scale_factor=2
)
print("Folded circuit:", folded, sep="\n")
```

+++ {"id": "4vZ5JMpVfnyO"}

**Codeblock 11**

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: Nsb4lxglfn4W
outputId: 162b8dda-3473-44d0-eae3-0cc7a70e146a
---
folded = zne.scaling.fold_global(circ, scale_factor=3.)
print("Folded circuit:", folded, sep="\n")
```

+++ {"id": "97Jtd6hNk2cg"}

**Codeblock 12**

+++ {"id": "FYJbOqK3k3zT"}

> Note: The paper shows pseudocode; here we show an example.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: Ey4vYYR1k4ul
outputId: d8cea421-0902-442e-e4c6-02cb24b9fd9a
---
q = cirq.LineQubit(0)
circuit2 = cirq.Circuit(cirq.X.on(q) ** (3 / 5), cirq.Z.on(q) ** (4 / 5))
print("Circuit:", circuit2, sep="\n")

scaled = zne.scaling.scale_parameters(circuit2, scale_factor=2.0, base_variance=0.01)
print("\nScaled circuit:", scaled, sep="\n")
```

+++ {"id": "ye9I9WLNrGxU"}

**Codeblock 13**

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: 1cOf4CCHrG8t
outputId: 3798e296-070c-4410-d3af-c54dc2756f36
---
zne_value = zne.execute_with_zne(
    circuit,
    executor,
    scale_noise=zne.scaling.fold_global,
)
print("ZNE value:", zne_value)
```

+++ {"id": "vm3EwufPfn_B"}

**Codeblock 14**

```{code-cell} ipython3
:id: AhnilvXffoF2

linear_factory = zne.inference.LinearFactory(scale_factors=[1.0, 2.0, 3.0])
```

+++ {"id": "3ubB9KrvfoLo"}

**Codeblock 15**

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: BFGHYLfHfoRT
outputId: db11bbc7-9ac3-4e93-a4cf-3613145f70bc
---
zne_value = zne.execute_with_zne(
    circuit,
    executor,
    factory=linear_factory,
)
print("ZNE value:", zne_value)
```

+++ {"id": "doyAREdJfoWC"}

**Codeblock 16**

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: PI5NPdolfob1
outputId: d9616b42-d25b-4e96-fe4e-116f78901d04
---
zne_value = zne.execute_with_zne(
    circuit,
    executor,
    factory=zne.inference.PolyFactory(
        scale_factors=[1.0, 2.0, 3.0], order=2
    ),
)
print("ZNE value:", zne_value)
```

+++ {"id": "zvIc-aLuGAut"}

**Codeblock 17**

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: TdquccfPGB3Z
outputId: 2c86fd20-67a9-451d-b8b9-d835d3b3a511
---
zne_value = zne.execute_with_zne(
    circuit,
    executor,
    factory=zne.inference.AdaExpFactory(
        scale_factor=2.0, steps=5
    ),
)
print("ZNE value:", zne_value)
```

+++ {"id": "bbaOzdc-GLeM"}

**Codeblock 18**

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: rh5iUFPNGLmU
outputId: 8de68b02-8c79-46d0-d384-2331de29658a
---
from mitiq.zne.inference import BatchedFactory, PolyFactory
import numpy as np


class MyFactory(BatchedFactory):
    @staticmethod
    def extrapolate(scale_factors, exp_values, full_output):
        zne_result, *extras = PolyFactory.extrapolate(
            scale_factors, exp_values, order=2, full_output=True
        )
        zne_result = np.clip(zne_result, -1.0, 1.0)
        return zne_result if not full_output else (zne_result, *extras)


# Example usage.
zne_value = zne.execute_with_zne(
    circuit,
    executor,
    factory=MyFactory(scale_factors=[1, 3, 5])
)
print("ZNE value:", zne_value)
```

+++ {"id": "t3o-Xc5rGLsR"}

**Codeblock 19**

```{code-cell} ipython3
:id: qqndfKFIGLy6

from mitiq import pec


noisy_x = pec.NoisyOperation(
    circuit=cirq.Circuit(cirq.X.on(q)),
    channel_matrix=np.random.randn(4, 4),
)

noisy_z = pec.NoisyOperation(
    circuit=cirq.Circuit(cirq.Z.on(q)),
    channel_matrix=np.random.randn(4, 4),
)
```

+++ {"id": "J0rb6P6XGL42"}

**Codeblock 20 & 21**

```{code-cell} ipython3
:id: UsxGtCrAGL-h

h_rep = pec.OperationRepresentation(
    ideal=cirq.Circuit(cirq.H.on(q)),
    basis_expansion = {noisy_x: 0.52, noisy_z: -0.48}
)
```

+++ {"id": "D2NVsYKVGMJ-"}

**Codeblock 22**

```{code-cell} ipython3
:id: Civ6U2OSGMQV

noisy_op, sign, coeff = h_rep.sample()
```

+++ {"id": "F8Ryo8cGGMWu"}

**Codeblock 23**

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: JY2lL4jXGMce
outputId: 9c7227f6-9a4e-4028-a932-a892542062b2
---
circuit3 = cirq.Circuit([cirq.H.on(q)] * 5)

sampled, sign, norm = pec.sample_circuit(circuit3, representations=[h_rep])
print("Sampled circuit:", sampled, sep="\n")  # Run many times to see different sampled circuits!
```

+++ {"id": "RYh92EfKGMix"}

**Codeblock 24 & 25**

+++ {"id": "C6K0VzUCLkcw"}

See https://mitiq.readthedocs.io/en/stable/examples/cdr_api.html.

+++ {"id": "T73iRSXLGMud"}

**Codeblock 26**

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: r0yT4jI7GM05
outputId: 59a2f5c0-ec34-4591-cdcb-b50198e95b39
---
mitigated_executor = zne.mitigate_executor(
    executor, scale_noise=zne.scaling.fold_global, factory=zne.inference.ExpFactory(scale_factors=[1, 3, 5, 7])
)
zne_value = mitigated_executor(circuit)
print("ZNE value:", zne_value)
```

+++ {"id": "042uqn-fMiYj"}

**Codeblock 27**

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: 2tHbd1x7MiiX
outputId: 007938cd-10ea-40d1-cd26-b6bc1f7b0ea6
---
@zne.zne_decorator(factory=zne.inference.ExpFactory(scale_factors=[1, 3, 5, 7]), scale_noise=zne.scaling.fold_gates_at_random)
def execute(circuit: cirq.Circuit) -> float:
    return cirq.DensityMatrixSimulator().simulate(
        circuit.with_noise(cirq.depolarize(p=0.01))
    ).final_density_matrix[0, 0].real


zne_value = execute(circuit)
print("ZNE value:", zne_value)
```

+++ {"id": "P5HEyCXwMio8"}

**Codeblock 28**

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: -AHS0pcLMixV
outputId: 0c26b4a3-77d0-4668-c799-efbd746f00fc
---
@pec.pec_decorator(representations=[h_rep], num_samples=10)
@zne.zne_decorator(
    factory=zne.inference.ExpFactory(scale_factors=[1, 3, 5, 7]),
    scale_noise=zne.scaling.fold_gates_at_random
)
def execute(circuit: cirq.Circuit) -> float:
    return cirq.DensityMatrixSimulator().simulate(
        circuit.with_noise(cirq.depolarize(p=0.01))
    ).final_density_matrix[0, 0].real


zne_then_pec_value = execute(circuit3)
print("ZNE then PEC value:", zne_then_pec_value)  # Note this is not accurate (bad representation).
```

+++ {"id": "LQYPqUazMi3b"}

**Codeblock 29**

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: vWUzgR6EMi9n
outputId: f2f56f66-9d37-4a17-f923-18c47a12ac5e
---
!pip install qiskit==0.24.0 --quiet
```

```{code-cell} ipython3
:id: dyk7Hj1nPB48


import qiskit


provider = qiskit.BasicAer  # Use of a simulator as backend.
# provider = qiskit.IBMQ.load_account()  # Alternative way to run the blocks, with saved credentials.

def execute(
    circuit: qiskit.QuantumCircuit,
    backend_name: str = "qasm_simulator",
    shots: int = 1024
) -> float:
    job = qiskit.execute(
        experiments=circuit,
        backend=provider.get_backend(backend_name),
        optimization_level=0,
        shots=shots,
    )

    counts = job.result().get_counts()
    return counts.get("00", 0.0) / shots
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: BJHEuASKQp4T
outputId: 3e5597ec-470f-48a9-cf3a-9e6fe551a06d
---
# Example usage.
qreg = qiskit.QuantumRegister(2)
creg = qiskit.ClassicalRegister(2)
circ = qiskit.QuantumCircuit(qreg, creg)
circ.h(qreg[0])
circ.cx(*qreg)
circ.measure(qreg, creg)
print("Circuit:")
print(circ)

execute(circ)
```

## References
[1] *Mitiq: A software package for error mitigation on noisy quantum computers*,
Ryan  LaRose, Andrea  Mari, Sarah  Kaiser, Peter  J.  Karalekas, Andre  A. Alves, Piotr Czarnik, Mohamed El Mandouh, Max H. Gordon, Yousef Hindy, Aaron Robertson, Purva Thakre, Nathan Shammah, and William J. Zeng, [https://arxiv.org/abs/2009.04417](https://arxiv.org/abs/2009.04417)

```{code-cell} ipython3

```

```{code-cell} ipython3

```
