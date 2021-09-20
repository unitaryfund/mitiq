---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.2
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

```{code-cell} ipython3
import pennylane as qml
from pennylane import numpy as np

dev = qml.device('default.mixed', wires=2)


@qml.qnode(dev)
def circuit():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))


print(f"QNode output = {circuit():.4f}")
```

```{code-cell} ipython3
@qml.qnode(dev)
def depolarizing_circuit(p):
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.DepolarizingChannel(p, wires=0)
    qml.DepolarizingChannel(p, wires=1)
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))


ps = [0.001, 0.01, 0.1, 0.2]
for p in ps:
    print(f"QNode output for depolarizing probability {p} is {depolarizing_circuit(p):.4f}")
```

# Option 1: subclassing an existing device

```{code-cell} ipython3
from pennylane import QubitDevice
from pennylane.devices import DefaultMixed

class MyDevice(DefaultMixed):
    """Default mixed with noise"""
    
    def expval(self, observable, shot_range=None, bin_size=None):

        if observable.name == "Projector":
            # branch specifically to handle the projector observable
            idx = int("".join(str(i) for i in observable.parameters[0]), 2)
            probs = self.probability(
                wires=observable.wires, shot_range=shot_range, bin_size=bin_size
            )
            return probs[idx]

        
        if self.shots is None:
            # exact expectation value
            eigvals = self._asarray(observable.eigvals, dtype=self.R_DTYPE)
            
            # Generate folded circuits
            
            # Extrapolate 
            
            
            prob = self.probability(wires=observable.wires)
            return self._dot(eigvals, prob)

        # estimate the ev
        samples = self.sample(observable, shot_range=shot_range, bin_size=bin_size)
        return np.squeeze(np.mean(samples, axis=0))
```

# Option 2: using QNodes

```{code-cell} ipython3
def circuit():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])


@qml.qnode(dev)
def executor_node(p, input_circuit):   # circuit must be considered as "non-trainable"
    input_circuit
    qml.DepolarizingChannel(p, wires=0)
    qml.DepolarizingChannel(p, wires=1)
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
```

```{code-cell} ipython3
from functools import partial

from mitiq import zne


zne.execute_with_zne(circuit, partial(executor_node, p=0.1))
```

```{code-cell} ipython3
from qiskit import QuantumCircuit
import numpy as np

dev = qml.device('default.mixed', wires=2)

theta = 0.5


qc = QuantumCircuit(1)
qc.rx(theta, 0)


@qml.qnode(dev)
def executor_qiskit(input_circuit = None):
    qml.from_qiskit(input_circuit)()
    #qml.DepolarizingChannel(p, wires=0)
    #qml.DepolarizingChannel(p, wires=1)
    return qml.expval(qml.PauliZ(0))
```

```{code-cell} ipython3
executor_qiskit(q_circuit)
```

```{code-cell} ipython3
from qiskit import QuantumCircuit
import numpy as np

dev = qml.device('default.mixed', wires=2)

theta = 0.5


qc = QuantumCircuit(1)
qc.rx(theta, 0)
qc.rx(theta, 0)
qc.rx(theta, 0)
qc.rx(theta, 0)
qc.rx(theta, 0)

def executor(input_circuit, p=0.1):
    
    @qml.qnode(dev)
    def node_qiskit():
        qml.from_qiskit(input_circuit)()
        qml.DepolarizingChannel(p, wires=0)
        qml.DepolarizingChannel(p, wires=1)
        return qml.expval(qml.PauliZ(0))
    
    return node_qiskit()
```

```{code-cell} ipython3
executor(qc, p=0.5)
```

```{code-cell} ipython3
zne.execute_with_zne(qc, partial(executor, p=0.5))
```

```{code-cell} ipython3
mitigated_executor = zne.mitigate_executor(partial(executor, p=0.5))
```

```{code-cell} ipython3
mitigated_executor(qc)
```

```{code-cell} ipython3

```
