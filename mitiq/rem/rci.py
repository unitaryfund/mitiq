# Copyright (C) 2021 Unitary Fund
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Readout Confusion Inversion."""

from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from mitiq import Executor, QPROGRAM
from mitiq.rem import MeasurementResult

from cirq import DensityMatrixSimulator
from cirq.experiments.readout_confusion_matrix import measure_confusion_matrix

MatrixLike = Union[
    np.ndarray,
    Iterable[np.ndarray],
    List[np.ndarray],
    Sequence[np.ndarray],
    Tuple[np.ndarray],
]


def execute_with_rci(
    circuit: QPROGRAM,
    executor: Union[Executor, Callable[[QPROGRAM], MeasurementResult]],
    inverse_confusion_matrix: Optional[MatrixLike] = None,
):
    """Returns the readout error mitigated expectation value utilizing an
    inverse confusion matrix that is computed by running the quantum program
    `circuit` with the executor function.

    Args:
        circuit: Quantum program to execute with error mitigation.
        executor: A ``mitiq.Executor`` or a function which inputs a (list
            of) quantum circuits and outputs a (list of)
            ``mitiq.rem.MeasurementResult``s.
        inverse_confusion_matrix: The inverse confusion matrix to apply to the
            probability vector that represents the noisy measurement results.
            If None, then an inverse confusion matrix will be generated
            utilizing the same `executor`.
    """

    if not isinstance(executor, Executor):
        executor = Executor(executor)

    if inverse_confusion_matrix is None:
        print("No inverse confusion matrix")
        qubits = set()
        for m in circuit.moments:
            for qubit in m.qubits:
                qubits.add(qubit)

        print(list(qubits))
        inverse_confusion_matrix = measure_confusion_matrix(
            DensityMatrixSimulator(), list(qubits)
        )

    result = executor.evaluate(circuit)
    assert isinstance(result, MeasurementResult)

    print(result)

