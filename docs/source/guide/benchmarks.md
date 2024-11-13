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

# Benchmarks

Mitiq benchmarks error mitigation techniques by running quantum circuits with and without mitigation, measuring improvements in accuracy, fidelity, and error rates. The process involves executing various circuit types—like GHZ, Mirror, Quantum Volume, and Randomized Benchmarking circuits—and comparing mitigated results against ideal outcomes. Analysis of these benchmarking results produces performance metrics, comparing mitigated and unmitigated outputs to quantify error reduction. This helps assess Mitiq’s effectiveness across diverse circuits, highlighting strengths and limitations in noise reduction.

The benchmark circuits can be used using the following workflow.

```{code-cell} ipython3
import cirq
from mitiq import benchmarks, zne

def execute(circuit, noise_level=0.005):
    """Returns Tr[ρ |0⟩⟨0|] where ρ is the state prepared by the circuit
    with depolarizing noise."""
    noisy_circuit = circuit.with_noise(cirq.depolarize(p=noise_level))
    return (
        cirq.DensityMatrixSimulator()
        .simulate(noisy_circuit)
        .final_density_matrix[0, 0]
        .real
    )

circuit = benchmarks.generate_ghz_circuit(n_qubits=10) #Call the required benchmark circuit function here

true_value = execute(circuit, noise_level=0.0)      # Ideal quantum computer
noisy_value = execute(circuit)                      # Noisy quantum computer
zne_value = zne.execute_with_zne(circuit, execute)  # Noisy quantum computer + Mitiq

print(f"Error w/o  Mitiq: {abs((true_value - noisy_value) / true_value):.3f}")
print(f"Error w Mitiq:    {abs((true_value - zne_value) / true_value):.3f}")
```


## GHZ Circuits

The {func}`mitiq.generate_ghz_circuit` create the GHZ states at are highly sensitive to noise. Thus, they make it easy to test error rates in entanglement creation and preservation, which is central for many quantum algorithms.

```{code-cell} ipython3
circuit = benchmarks.generate_ghz_circuit(n_qubits=10)
```

## Mirror Circuits



## Mirror Quantum Volume Circuits

```{code-cell} ipython3
circuit= benchmarks.mirror_circuits.random_cliffords(connectivity_graph=connectivity_graph, random_state = rs)
```

## Quantum Phase Estimation Circuits

```{code-cell} ipython3
circuit = benchmarks.qpe_circuits.generate_qpe_circuit(evalue_reg=3)
```

## Quantum Volume Circuits

```{code-cell} ipython3
circuit,_ = benchmarks.quantum_volume_circuits.generate_quantum_volume_circuit(num_qubits=4, depth=10)
```

## Randomized Benchmarking Circuits

```{code-cell} ipython3
circuits = benchmarks.randomized_benchmarking.generate_rb_circuits(n_qubits=1, num_cliffords=5)

circuit=circuits[0]
```

## Rotated Randomized Benchmarking Circuits

```{code-cell} ipython3
circuits = benchmarks.rotated_randomized_benchmarking.generate_rotated_rb_circuits(n_qubits=1, num_cliffords=5)

circuit=circuits[0]
```

## Randomized Clifford+T Circuits

```{code-cell} ipython3
circuit = benchmarks.randomized_clifford_t_circuit.generate_random_clifford_t_circuit(num_qubits=10, num_oneq_cliffords=2, num_twoq_cliffords=2, num_t_gates=2)
```

## W State Circuits

```{code-cell} ipython3
circuit = benchmarks.w_state_circuits.generate_w_circuit(n_qubits=10)
```
