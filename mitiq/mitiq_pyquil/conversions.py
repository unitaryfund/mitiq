"""
Functions to convert between Mitiq's internal circuit representation
and pyQuil's circuit representation (Quil programs).
"""

from cirq import Circuit
from pyquil import Program

from mitiq.mitiq_pyquil.quil import circuit_from_quil

QuilType = str


# TODO: to_quil needs cirq-unstable or v0.9.0 to be released
def to_quil(circuit: Circuit) -> QuilType:
    """Returns a Quil string representing the input Mitiq circuit.

    Args:
        circuit: Mitiq circuit to convert to a Quil string.

    Returns:
        QuilType: Quil string equivalent to the input Mitiq circuit.
    """
    return circuit.to_quil()


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


# TODO: eventually circuit_from_quil will be moved to Cirq
def from_quil(quil: QuilType) -> Circuit:
    """Returns a Mitiq circuit equivalent to the input Quil string.

    Args:
        quil: Quil string to convert to a Mitiq circuit.

    Returns:
        Mitiq circuit representation equivalent to the input Quil string.
    """
    return circuit_from_quil(quil)
