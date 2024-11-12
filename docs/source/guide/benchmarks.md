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

## GHZ Circuits

The GHZ (Greenberger–Horne–Zeilinger) circuits create the GHZ states at are highly sensitive to noise. Thus, they make it easy to test error rates in entanglement creation and preservation, which is central for many quantum algorithms.

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

circuits = benchmarks.generate_ghz_circuit(n_qubits=10)

true_value = execute(circuits, noise_level=0.0)      # Ideal quantum computer
noisy_value = execute(circuits)                      # Noisy quantum computer
zne_value = zne.execute_with_zne(circuits, execute)  # Noisy quantum computer + Mitiq

print(f"Error w/o  Mitiq: {abs((true_value - noisy_value) / true_value):.3f}")
print(f"Error w Mitiq:    {abs((true_value - zne_value) / true_value):.3f}")
```

## Mirror Circuits

## Mirror Quantum Volume Circuits

## Quantum Phase Estimation Circuits

## Quantum Volume Circuits

## Randomized Benchmarking Circuits

## Rotated Randomized Benchmarking Circuits

## Randomized Clifford+T Circuits

## W State Circuits

