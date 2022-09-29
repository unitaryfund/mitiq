# Copyright (C) 2022 Unitary Fund
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

"""Run experiments without error mitigation."""
from typing import Callable, Optional, Union

from mitiq import Executor, Observable, QPROGRAM, QuantumResult


def execute(
    circuit: QPROGRAM,
    executor: Union[Executor, Callable[[QPROGRAM], QuantumResult]],
    observable: Optional[Observable] = None,
) -> float:
    """Evaluates the expectation value associated to the input circuit
    without using error mitigation.

    The only purpose of this function is to provide the same interface for
    non-error-mitigated values as the rest of the techniques in Mitiq. This
    is useful when comparing error-mitigated results to non-error-mitigated
    results.

    Args:
        circuit: The circuit to run.
        executor: Executes a circuit and returns a `QuantumResult`.
        observable: Observable to compute the expectation value of. If None,
            the `executor` must return an expectation value. Otherwise,
            the `QuantumResult` returned by `executor` is used to compute the
            expectation of the observable.
    """
    if not isinstance(executor, Executor):
        executor = Executor(executor)

    return executor.evaluate(circuit, observable)[0]
