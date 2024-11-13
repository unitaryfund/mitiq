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

circuit = benchmarks.generate_ghz_circuit(n_qubits=10) # Call the required benchmark circuit function here

true_value = execute(circuit, noise_level=0.0)      # Ideal quantum computer
noisy_value = execute(circuit)                      # Noisy quantum computer
zne_value = zne.execute_with_zne(circuit, execute)  # Noisy quantum computer + Mitiq

print(f"Error w/o  Mitiq: {abs((true_value - noisy_value) / true_value):.3f}")
print(f"Error w Mitiq:    {abs((true_value - zne_value) / true_value):.3f}")
```


## GHZ Circuits

The {func}`.generate_ghz_circuit` create the GHZ states that are highly sensitive to noise. Thus, they make it easy to test error rates in entanglement creation and preservation, which is central for many quantum algorithms.

```{code-cell} ipython3
circuit = benchmarks.generate_ghz_circuit(n_qubits=10)
```

## Mirror Circuits

The {func}`.generate_mirror_circuit` involves running a quantum circuit forward and then “mirroring” it (applying the reverse operations). Ideally, this results in returning the system to the initial state, so they’re great for testing if the noise mitigation is effective in preserving information through complex sequences.

## Mirror Quantum Volume Circuits

The {func}`mitiq.mirror_qv_circuits.generate_mirror_qv_circuit` is designed to test `quantum volume`, a metric combining circuit depth, number of qubits, and fidelity. These circuits check whether error mitigation techniques help achieve higher effective quantum volumes on noisy devices.

```{code-cell} ipython3
circuit= benchmarks.mirror_circuits.random_cliffords(connectivity_graph=connectivity_graph, random_state = rs)
```

## Quantum Phase Estimation Circuits

The {func}`mitiq.qpe_circuits.generate_qpe_circuit` is used to the measure eigenvalues of unitary operators. Since accurate phase estimation requires precise control over operations, these circuits test the mitigation techniques’ ability to handle small noise effects over multiple gate sequences.

```{code-cell} ipython3
circuit = benchmarks.qpe_circuits.generate_qpe_circuit(evalue_reg=3)
```

## Quantum Volume Circuits

The {func}`mitiq.quantum_volume_circuits.generate_quantum_volume_circuit` tests the maximum achievable "volume" or computational capacity of a quantum processor. Running these circuits with error mitigation tests if mitiq’s techniques improve the effective quantum volume.

```{code-cell} ipython3
circuit,_ = benchmarks.quantum_volume_circuits.generate_quantum_volume_circuit(num_qubits=4, depth=10)
```

## Randomized Benchmarking Circuits

The {func}`mitiq.randomized_benchmarking.generate_rb_circuits` are sequences of random gates (generally Clifford gates), to estimate an average error rate. They’re standard in benchmarking for evaluating how well mitiq’s error mitigation reduces this error rate across different levels of noise.

```{code-cell} ipython3
circuits = benchmarks.randomized_benchmarking.generate_rb_circuits(n_qubits=1, num_cliffords=5)

circuit=circuits[0]
```

## Rotated Randomized Benchmarking Circuits

The {func}`mitiq.rotated_randomized_benchmarking.generate_rotated_rb_circuits` are sequences of random gates similar to {func}`mitiq.randomized_benchmarking.generate_rb_circuits`, but with rotations added, that allows assessment of errors beyond just the standard Clifford gates. They’re useful to check how well Mitiq handles noise in scenarios with more diverse gates.

```{code-cell} ipython3
circuits = benchmarks.rotated_randomized_benchmarking.generate_rotated_rb_circuits(n_qubits=1, num_cliffords=5)

circuit=circuits[0]
```

## Randomized Clifford+T Circuits

The {func} `mitiq.randomized_clifford_t_circuit.generate_random_clifford_t_circuit` add the T gate to the standard Clifford set, adding more complex operations to the random benchmarking. This type evaluates Mitiq’s performance with gate sets that go beyond the Clifford gates, crucial for fault-tolerant computing.

```{code-cell} ipython3
circuit = benchmarks.randomized_clifford_t_circuit.generate_random_clifford_t_circuit(num_qubits=10, num_oneq_cliffords=2, num_twoq_cliffords=2, num_t_gates=2)
```

## W State Circuits

The {func}`mitiq.w_state_circuits.generate_w_circuit` are entangled circuits that distribute the entanglement across qubits differently than GHZ states. Testing with W state circuits can help explore how well a device maintains distributed entanglement in noisy environments.

```{code-cell} ipython3
circuit = benchmarks.w_state_circuits.generate_w_circuit(n_qubits=10)
```
