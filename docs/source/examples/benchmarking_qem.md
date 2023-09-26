---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3
  name: python3
---

# Benchmarking ZNE on random circuits with expecation values other than 0 or 1

## Issue Description

Most examples and most benchmarks in Mitiq are based on benchmark circuits whose ideal expectation value is 1.
For example:
 https://github.com/unitaryfund/mitiq/blob/9a13b3ec476cd2cd978e572d12081af4d73536cb/mitiq/benchmarks/randomized_benchmarking.py#L23

Typically we test how good a QEM technique is, by checking how  close a mitigated expectation value is to 1 (ideal result) compared to the noisy unmitigated result.


This is very convenient and intuitive. However, it is also useful to have more general benchmarks in which the expectation value can be any number in a continuous interval e.g.  

$$ E_{\rm ideal} \in  [-1, 1] $$

This benchmarking method enables testing QEM techniques in more general scenarios, closer to real-world applications in which expectation values can take arbitrary values.


+++

## Setup

+++

```{code-cell} ipython3
import numpy as np

import cirq

import mitiq
```

### Circuit


Here we generate "rotated" RB circuits in which we insert a $R_z(\theta)$ rotation in the middle of an RB circuit, such that

$$ C(\theta) =    G_n \dots G_{n/2 +1} R_z(\theta)G_{n/2} \dots G_2 G_1 $$

where $G_j$ are Clifford elements or Clifford gates.

This should generate expectation values which are sinusoidal functions of $\theta$, so something varying in with a continuous range of ideal expectation values.

At the same time since, up to factors of 2, we have $R_z(\theta) =cos(\theta) I +  i \ sin(\theta) Z$,  the rotated Clifford circuit $C(\theta)$ can be written as a  linear combination of just two Clifford circuits and, therefore, it is still easy to classically simulate.
 


```{code-cell} ipython3
n_qubits = 2
trials = 10
n_cliffords = 10
circuits = mitiq.benchmarks.generate_rb_circuits(
    n_qubits, n_cliffords, trials
)
composed_circs = []

rng = np.random.default_rng()

for c in range(5):
    circ = circuits[c]
    q = circ.all_qubits()
    rotated_circ = circ.append(cirq.Rz(rads=rng.random() * np.pi)(q[0]))
    rotated_circ = circ.append(cirq.Rz(rads=rng.random() * np.pi)(q[1]))
    composed_circs.append(rotated_circ.append(circuits[c + 5]))

```


```{code-cell} ipython3
def execute(circuit, noise_level=0.01):
    noisy_circuit = cirq.Circuit()
    for op in circuit.all_operations():
        noisy_circuit.append(op)
        if len(op.qubits) == 2:
            noisy_circuit.append(
                cirq.depolarize(p=noise_level, n_qubits=2)(*op.qubits)
            )

    rho = (
        cirq.DensityMatrixSimulator()
        .simulate(noisy_circuit)
        .final_density_matrix
    )
    return rho[0, 0].real
```



```{code-cell} ipython3
ideal_values = []
unmitigated_values = []
mitigated_values = []

for circuit in composed_circs:
    ideal_values.append(execute(circuit, 0))
    unmitigated_values.append(execute(circuit))
    mitigated_values.append(mitiq.zne.execute_with_zne(circuit, execute))
    
```

### TODO: Plot per trial
