# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Defines utility functions for classical shadows protocol."""
from typing import Tuple, List, Any, Union
import mitiq
import cirq
import numpy as np
from cirq.ops.pauli_string import PauliString
from numpy.typing import NDArray
import re


def n_measurements_tomography_bound(epsilon: float, num_qubits: int) -> int:
    """
    This function returns the minimum number of classical shadows required
    for state reconstruction for achieving the desired accuracy.

    Args:
        epsilon: The error on the estimator.
        num_qubits: The number of qubits in the system.

    Returns:
        An integer that gives the number of snapshots required to satisfy the
        shadow bound.
    """
    return int(34 * (4**num_qubits) * epsilon ** (-2))


def local_clifford_shadow_norm(opt: PauliString[Any]) -> float:
    """
    Calculate shadow norm of an operator with random unitary sampled from local
    Clifford group.
    Args:
        opt: a self-adjoint operator
    Returns:
        Shadow norm when unitary ensemble is local Clifford group.
    """
    norm = (
        np.linalg.norm(
            cirq.unitary(opt)
            - np.trace(cirq.unitary(opt))
            / 2 ** int(np.log2(cirq.unitary(opt).shape[0])),
            ord=np.inf,
        )
        ** 2
    )
    return float(norm)


def n_measurements_opts_expectation_bound(
    error: float,
    observables: List[PauliString[Any]],
    failure_rate: float,
) -> Tuple[int, int]:
    """
    This function returns the minimum number of classical shadows required and
    the number of groups "k" into which we need to split the shadows for
    achieving the desired accuracy and failure rate in operator expectation
    value estimation.

    Args:
        error: The error on the estimator.
        observables: List of cirq.PauliString corresponding to the
            observables we intend to measure.
        failure_rate: Rate of failure for the bound to hold.

    Returns:
        Integers quantifying the number of snapshots required to satisfy
        the shadow bound and the chunk size required to attain the specified
        failure rate.
    """
    M = len(observables)
    K = 2 * np.log(2 * M / failure_rate)

    N = (
        34
        * max(local_clifford_shadow_norm(o) for o in observables)
        / error**2
    )
    return int(np.ceil(N * K)), int(K)


def fidelity(
    state_vector: NDArray[np.complex64],
    rho: NDArray[np.complex64],
) -> float:
    """
    Calculate the fidelity between a state vector and a density matrix.
    Args:
        state_vector: The vector whose norm we want to calculate.
        rho: The operator whose norm we want to calculate.

    Returns:
        Scalar corresponding to the fidelity.
    """
    return np.reshape(state_vector.conj().T @ rho @ state_vector, -1).real[0]


def transform_to_cirq_paulistring(
    pauli_str: Union[str, mitiq.PauliString, cirq.PauliString[Any]]
) -> Tuple[float, cirq.PauliString[Any]]:
    """Transforms mitiq.PauliString, cirq.PauliString, cirq.PauliSum or string
    to a cirq.PauliString class.

    Args:
      pauli_str: A mitiq.PauliString, cirq.PauliString, cirq.PauliSum
        or string.

    Returns:
      A tuple containing the coefficient and a cirq.PauliString.
    """
    if isinstance(pauli_str, cirq.PauliString):
        # Extract coefficient and return cirq.PauliString without coefficient
        coeff = np.real(pauli_str.coefficient)  # type: ignore
        new_pauli_str: cirq.PauliString[Any] = cirq.PauliString(
            {qubit: op for qubit, op in pauli_str.items()}
        )
        return coeff, new_pauli_str

    elif isinstance(pauli_str, mitiq.PauliString):
        qubit_pauli_map = {
            cirq.LineQubit(i): gate
            for i, gate in enumerate(pauli_str._pauli.values())
        }
        return np.real(pauli_str.coeff), cirq.PauliString(
            qubit_pauli_map=qubit_pauli_map
        )
    elif isinstance(pauli_str, str):
        # Extract the coefficient from the start of the string
        matched_results = re.match(
            r"([+-]?(\d+(\.\d*)?|\.\d+))?(.*)", pauli_str
        )
        if matched_results is None:
            raise ValueError("pauli_str should be of format '0.5IXXY'")
        coeff_str, _, _, pauli_str = matched_results.groups()
        coefficient = float(coeff_str) if coeff_str else 1.0

        # Create a mapping for Pauli gates
        pauli_map = {"X": cirq.X, "Y": cirq.Y, "Z": cirq.Z}
        # Create a dictionary where the keys are qubits and the values are
        # corresponding Pauli operations
        qubit_pauli_map = {}
        for i, pauli_name in enumerate(pauli_str):  # X, Y, Z, I
            if pauli_name == "I":
                continue
            qubit_pauli_map[cirq.LineQubit(i)] = pauli_map[pauli_name]

        return coefficient, cirq.PauliString(qubit_pauli_map=qubit_pauli_map)

    else:
        raise ValueError(
            "pauli_str must be cirq.PauliString, mitiq.PauliString, "
            "cirq.PauliSum or string."
        )
