from typing import List

from cirq import Circuit

from mitiq import QPROGRAM
from mitiq.pea.amplifications.depolarizing import (
    amplify_operations_in_circuit_with_global_depolarizing_noise,
    amplify_operations_in_circuit_with_local_depolarizing_noise,
)
from mitiq.pec.sampling import sample_circuit


def scale_circuit_amplifications(
    ideal_circuit: Circuit,
    scale_factors: List[float],
    noise_model: str,
    epsilon: float,
) -> List[QPROGRAM]:
    """Returns a list of noise-amplified circuits, corrsponding to each scale
    factor multiplied by the baseline noise level."""

    if noise_model == "local-depolarizing":
        amp_fn = amplify_operations_in_circuit_with_local_depolarizing_noise
        # TODO add other existing noise models from Mitiq
    elif noise_model == "global-depolarizing":
        amp_fn = amplify_operations_in_circuit_with_global_depolarizing_noise
    else:
        raise ValueError("Must specify supported noise model")
        # TODO allow use of custom noise model

    scaled_circuits = []
    for s in scale_factors:
        if s == 1:
            scaled_circuits.append(ideal_circuit)
        else:
            scaled_circuits.append(ideal_circuit.with_noise(amp_fn((s - 1) * epsilon)))
    return scaled_circuits


def sample_circuit_amplifications(
    ideal_circuit: QPROGRAM,
    scale_factors: List[float],
    noise_model: str,
    epsilon: float,
) -> List[QPROGRAM]:
    """Returns a list of expectation values, evaluated at each noise scale
    factor times the baseline noise level."""

    return [
        sample_circuit(circuit)
        for circuit in scale_circuit_amplifications(
            ideal_circuit, scale_factors, noise_model, epsilon
        )
    ]
