# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Functions to convert between Mitiq's internal circuit representation and
pyQuil's circuit representation (Quil programs).
"""

from cirq import Circuit, LineQubit
from cirq_rigetti import circuit_from_quil
from cirq_rigetti.quil_output import QuilOutput
from pyquil import Program

QuilType = str


def to_quil(circuit: Circuit) -> QuilType:
    """Returns a Quil string representing the input Mitiq circuit.

    Args:
        circuit: Mitiq circuit to convert to a Quil string.

    Returns:
        QuilType: Quil string equivalent to the input Mitiq circuit.
    """
    max_qubit = max(circuit.all_qubits())
    # if we are using LineQubits, keep the qubit labeling the same
    if isinstance(max_qubit, LineQubit):
        qubit_range = max_qubit.x + 1
        return str(
            QuilOutput(circuit.all_operations(), LineQubit.range(qubit_range))
        )
    # otherwise, use the default ordering (starting from zero)
    return str(
        QuilOutput(circuit.all_operations(), sorted(circuit.all_qubits()))
    )


def to_pyquil(circuit: Circuit) -> Program:
    """Returns a pyQuil Program equivalent to the input Mitiq circuit.

    Args:
        circuit: Mitiq circuit to convert to a pyQuil Program.

    Returns:
        pyquil.Program object equivalent to the input Mitiq circuit.
    """
    return Program(to_quil(circuit))


def from_pyquil(program: Program) -> Circuit:
    """Returns a Mitiq circuit equivalent to the input pyQuil Program.

    Args:
        program: PyQuil Program to convert to a Mitiq circuit.

    Returns:
        Mitiq circuit representation equivalent to the input pyQuil Program.
    """
    return from_quil(program.out())


def from_quil(quil: QuilType) -> Circuit:
    """Returns a Mitiq circuit equivalent to the input Quil string.

    Args:
        quil: Quil string to convert to a Mitiq circuit.

    Returns:
        Mitiq circuit representation equivalent to the input Quil string.
    """
    return circuit_from_quil(quil)
