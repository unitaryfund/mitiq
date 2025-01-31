from typing import List, Sequence

from cirq import Circuit

from mitiq.pea.amplifications.depolarizing import (
    amplify_operations_in_circuit_with_global_depolarizing_noise,
    amplify_operations_in_circuit_with_local_depolarizing_noise)
from mitiq.pec import OperationRepresentation
from mitiq.pec.sampling import sample_circuit


def scale_circuit_amplifications(
    ideal_circuit: Circuit,
    scale_factors: List[float],
    noise_model: str,
    epsilon: float,
) -> List[Sequence[OperationRepresentation]]:
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

    amplified_circuits = []
    for s in scale_factors:
        if s == 1:
            amplified_circuits.append(ideal_circuit)
        else:
            amplified_circuits.append(
                (amp_fn(ideal_circuit, (s - 1) * epsilon))
        )
    return amplified_circuits


def sample_circuit_amplifications(
    ideal_circuit: Circuit,
    scale_factors: List[float],
    noise_model: str,
    epsilon: float,
) -> List[QPROGRAM]:
    """Returns a list of expectation values, evaluated at each noise scale
    factor times the baseline noise level."""

    return [
        sample_circuit(ideal_circuit, amplifications)
        for amplifications in scale_circuit_amplifications(
            ideal_circuit, scale_factors, noise_model, epsilon
        )
    ]
