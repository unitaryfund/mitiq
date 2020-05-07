# random_circ.py
"""
Contains methods used for testing mitiq's performance
"""
from typing import Tuple, Callable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

import cirq

from mitiq import QPROGRAM
from mitiq.folding import fold_gates_from_left, fold_global
from mitiq.factories import PolyFactory, LinearFactory


def rand_benchmark_zne(
    nqubits: int,
    depth: int,
    trials: int,
    depolarizing_noise_strength: float,
    scale_factors: Sequence[float],
    scale_noise: Callable[[QPROGRAM, float], QPROGRAM] = fold_gates_from_left,
    op_density: float = 0.99,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Benchmarks a zero-noise extrapolation method and noise scaling executor
    by running on randomly sampled quantum circuits.

    Args:
        nqubits: The number of qubits.
        depth: The depth in moments of the random circuits.
        trials: The number of random circuits to average over.
        depolarizing_noise_strength: Strength of the depolarizing channel.
        scale_factors: Factors to scale noise by.
        scale_noise: The method for scaling noise, e.g. fold_gates_at_random.
        op_density: The expected proportion of qubits that are acted on in
                    any moment.
        verbose: Print out status messages.

    Returns:
        The tuple (unmitigated_error, mitigated_error) where each is a list
        whose values are the errors of that trial in the unmitigated or
        mitigated cases.
    """
    unmitigated_errors = []
    mitigated_errors = []

    for circuit_number in range(trials):
        if verbose:
            print(f"Status: On circuit {circuit_number + 1} / {trials}.")

        # Get this random circuit
        circuit = cirq.testing.random_circuit(
            qubits=nqubits, n_moments=depth, op_density=op_density
        )

        # Get the noiseless state and probabilities
        noiseless_state = circuit.final_wavefunction()
        noiseless_probs = np.abs(noiseless_state)**2

        # Get the "hardware" state (base noise values)
        dsim = cirq.DensityMatrixSimulator()
        hardware_exp = np.diag(dsim.simulate(
            circuit.with_noise(cirq.depolarize(p=depolarizing_noise_strength))
        ).final_density_matrix)
        hardware_probs = hardware_exp.real

        # Get the noisy states and probabilities
        folded_circuits = [
            scale_noise(circuit, scale_factor) for scale_factor in scale_factors
        ]
        noisy_circuits = [
            folded.with_noise(cirq.depolarize(p=depolarizing_noise_strength))
            for folded in folded_circuits
        ]
        noisy_probs = [
            np.diag(dsim.simulate(noisy).final_density_matrix).real
            for noisy in noisy_circuits
        ]

        # Do the zero noise extrapolation
        to_mitigate = np.array(noisy_probs).T
        zne_probs = []
        for scaled_probabilies in to_mitigate:
            zero_noise_value = PolyFactory.static_reduce(
                np.array(scale_factors), # * depolarizing_noise_strength,
                scaled_probabilies,
                order=1
            )
            zne_probs.append(zero_noise_value)
        zne_probs = np.array(zne_probs)

        # Compare the hardware probs and the extrapolated probs to the true vals
        unmitigated_errors.append(
            np.linalg.norm(hardware_probs - noiseless_probs, ord=2)
        )
        mitigated_errors.append(
            np.linalg.norm(zne_probs - noiseless_probs, ord=2)
        )

    return np.array(unmitigated_errors), np.array(mitigated_errors)


if __name__ == "__main__":
    # Set parameters
    DEPTH = 100
    TRIALS = 30
    DEPO_NOISE_STRENGTH = 0.05

    # Do the benchmark
    unmitigated_errors, mitigated_errors = rand_benchmark_zne(
        nqubits=5,
        depth=DEPTH,
        trials=TRIALS,
        depolarizing_noise_strength=DEPO_NOISE_STRENGTH,
        scale_factors=[1., 1.5, 2.0],
        scale_noise=fold_global,
    )

    # Plot the results
    plt.rcParams.update({"font.size": 16, "font.weight": "bold"})
    plt.hist(unmitigated_errors,
             alpha=0.5,
             bins=25,
             label="Raw")
    plt.hist(mitigated_errors,
             alpha=0.5,
             bins=25,
             label="ZNE")
    plt.xlabel("Distance from true distribution")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()
