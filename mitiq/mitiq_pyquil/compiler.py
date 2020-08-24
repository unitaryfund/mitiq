# Copyright (C) 2020 Unitary Fund
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

"""Provides a utility basic_compile to allow for running on QCS without
using quilc, as quilc will optimize away unitary folding.

NB: Copied in large part from rigetti/forest-benchmarking (Apache-2.0)
and modified to support a larger gateset (e.g. CPHASE).
"""

from math import pi
from typing import Tuple

import numpy as np

from pyquil.gates import RX, RZ, CZ, I, XY
from pyquil.quil import Program
from pyquil.quilbase import Gate


def _CCNOT(q0: int, q1: int, q2: int) -> Program:
    """
    A CCNOT in terms of RX(+-pi/2), RZ(theta), and CZ

    .. note:
        Don't control this gate.
    """
    p = Program()
    p.inst(_H(q2))
    p.inst(_CNOT(q1, q2))
    p.inst(_T(q2, dagger=True))
    p.inst(_SWAP(q1, q2))
    p.inst(_CNOT(q0, q1))
    p.inst(_T(q1))
    p.inst(_CNOT(q2, q1))
    p.inst(_T(q1, dagger=True))
    p.inst(_CNOT(q0, q1))
    p.inst(_SWAP(q1, q2))
    p.inst(_T(q1))
    p.inst(_T(q2))
    p.inst(_CNOT(q0, q1))
    p.inst(_H(q2))
    p.inst(_T(q0))
    p.inst(_T(q1, dagger=True))
    p.inst(_CNOT(q0, q1))
    return p


def _CNOT(q0: int, q1: int) -> Program:
    """
    A CNOT in terms of RX(+-pi/2), RZ(theta), and CZ

    .. note:
        This uses two of :py:func:`_H`, so it picks up twice the global phase.
        Don't control this gate.
    """
    p = Program()
    p.inst(_H(q1))
    p.inst(CZ(q0, q1))
    p.inst(_H(q1))
    return p


def _CPHASE(angle: float, q0: int, q1: int) -> Program:
    """
    from quilc:

    (define-compiler CPHASE-to-CNOT ((CPHASE-gate ("CPHASE" (alpha) p q)))
        (inst "CNOT" ()                            p q)
        (inst "RZ"   (list (param-* alpha -0.5d0)) q)
        (inst "CNOT" ()                            p q)
        (inst "RZ"   (list (param-* alpha  0.5d0)) q)
        (inst "RZ"   (list (param-* alpha  0.5d0)) p))
    """
    p = Program()
    p.inst(_CNOT(q0, q1))
    p.inst(RZ(-0.5 * angle, q1))
    p.inst(_CNOT(q0, q1))
    p.inst(RZ(0.5 * angle, q1))
    p.inst(RZ(0.5 * angle, q0))
    return p


def _H(q: int) -> Program:
    """
    A Hadamard in terms of RX(+-pi/2) and RZ(theta)

    .. note:
        This introduces a different global phase! Don't control this gate.
    """
    p = Program()
    p.inst(_RY(-np.pi / 2, q))
    p.inst(RZ(np.pi, q))
    return p


def _ISWAP(q0: int, q1: int) -> Program:
    """
    An ISWAP as an XY(pi). Of course, assumes XY is available.
    """
    p = Program()
    p += XY(np.pi, q0, q1)
    return p


def _PHASE(angle: float, q: int) -> Program:
    """
    from quilc:

    (define-compiler PHASE-to-RZ ((phase-gate ("PHASE" (alpha) q)))
       (inst "RZ" `(,alpha) q))
    """
    p = Program()
    p += RZ(angle, q)
    return p


def _RX(angle: float, q: int) -> Program:
    """
    A RX in terms of native RX(+-pi/2) and RZ gates.
    """
    p = Program()
    p += RZ(pi / 2, q)
    p += RX(pi / 2, q)
    p += RZ(angle, q)
    p += RX(-pi / 2, q)
    p += RZ(-pi / 2, q)
    return p


def _RY(angle: float, q: int) -> Program:
    """
    A RY in terms of RX(+-pi/2) and RZ(theta)
    """
    p = Program()
    p += RX(pi / 2, q)
    p += RZ(angle, q)
    p += RX(-pi / 2, q)
    return p


def _S(q: int, dagger: bool = False) -> Program:
    """
    An S in terms of RZ(theta)
    """
    if dagger:
        return Program(RZ(-np.pi / 2, q))
    else:
        return Program(RZ(np.pi / 2, q))


def _SWAP(q0: int, q1: int) -> Program:
    """
    A SWAP in terms of _CNOT

    .. note:
        This uses :py:func:`_CNOT`, so it picks up a global phase.
        Don't control this gate.

    """
    p = Program()
    p.inst(_CNOT(q0, q1))
    p.inst(_CNOT(q1, q0))
    p.inst(_CNOT(q0, q1))
    return p


def _T(q: int, dagger: bool = False) -> Program:
    """
    A T in terms of RZ(theta)
    """
    if dagger:
        return Program(RZ(-np.pi / 4, q))
    else:
        return Program(RZ(np.pi / 4, q))


def _X(q: int) -> Program:
    """
    An X in terms of RX(pi)

    .. note:
        This introduces a global phase! Don't control this gate.
    """
    p = Program()
    p += RX(np.pi, q)
    return p


def _Y(q: int) -> Program:
    """
    A Y in terms of _RY
    """
    p = Program()
    p += _RY(np.pi, q)
    return p


def _Z(q: int) -> Program:
    """
    A Z in terms of RZ
    """
    p = Program()
    p += RZ(np.pi, q)
    return p


# This function is taken from cirq. License: apache 2.
def match_global_phase(
    a: np.ndarray, b: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Phases the given matrices so that they agree on the phase of one entry.
    To maximize precision, the position with the largest entry from one of the
    matrices is used when attempting to compute the phase difference between
    the two matrices.
    :param a: The first matrix
    :param b: The second matrix
    :return: A tuple (a', b') where a' == b' implies a == b*exp(i t) for some t.
    """

    # Not much point when they have different shapes.
    if a.shape != b.shape:
        return a, b

    # Find the entry with the largest magnitude in one of the matrices.
    k = max(np.ndindex(*a.shape), key=lambda t: abs(b[t]))

    def dephase(v):
        r = np.real(v)
        i = np.imag(v)

        # Avoid introducing floating point error when axis-aligned.
        if i == 0:
            return -1 if r < 0 else 1
        if r == 0:
            return 1j if i < 0 else -1j

        return np.exp(-1j * np.arctan2(i, r))

    # Zero the phase at this entry in both matrices.
    return a * dephase(a[k]), b * dephase(b[k])


def is_magic_angle(angle: float) -> bool:
    return (
        np.isclose(np.abs(angle), pi / 2)
        or np.isclose(np.abs(angle), pi)
        or np.isclose(angle, 0.0)
    )


def basic_compile(program: Program) -> Program:
    """
    A rudimentary but predictable compiler.

    No rewiring or optimization is done by this compilation step. There may be some gates that
    are not yet supported. Gates defined in the input program are included without change in the
    output program.

    :param program: a program to be compiled to native quil with simple replacements.
    :return: a program with some of the input non-native quil gates replaced with basic native quil
        gate implementations.
    """
    new_prog = Program()
    new_prog.num_shots = program.num_shots
    new_prog.inst(program.defined_gates)

    daggered_defgates = []

    for inst in program:
        if isinstance(inst, Gate):
            # TODO: this is only a stopgap while the noisy QVM does not support modifiers.
            # dagger this gate if odd number of daggers. Ignore controlled for now.
            needs_dagger = inst.modifiers.count("DAGGER") % 2 == 1
            angle_param = None
            if len(inst.params) > 0:
                angle_param = inst.params[0]
                if needs_dagger:
                    angle_param = -angle_param

            if "CONTROLLED" in inst.modifiers:
                raise ValueError(
                    "Controlled gates are not currently supported."
                )

            if inst.name == "CCNOT":
                new_prog += _CCNOT(*inst.qubits)
            elif inst.name == "CNOT":
                new_prog += _CNOT(*inst.qubits)
            # NB: we haven't implemented CPHASE00/01/10
            elif inst.name == "CPHASE":
                new_prog += _CPHASE(angle_param, *inst.qubits)
            elif inst.name == "CZ":
                new_prog += CZ(*inst.qubits)  # remove dag modifiers
            elif inst.name == "H":
                new_prog += _H(inst.qubits[0])
            elif inst.name == "I":
                new_prog += I(inst.qubits[0])  # remove dag modifiers
            elif inst.name == "ISWAP":
                new_prog += _ISWAP(*inst.qubits)  # remove dag modifiers
            elif inst.name == "PHASE":
                new_prog += _PHASE(angle_param, inst.qubits[0])
            elif inst.name == "RX":
                if is_magic_angle(inst.params[0]):
                    # in case dagger
                    new_prog += RX(angle_param, inst.qubits[0])
                else:
                    new_prog += _RX(angle_param, inst.qubits[0])
            elif inst.name == "RY":
                new_prog += _RY(angle_param, inst.qubits[0])
            elif inst.name == "RZ":
                # in case dagger
                new_prog += RZ(angle_param, inst.qubits[0])
            elif inst.name == "S":
                new_prog += _S(inst.qubits[0], needs_dagger)
            # NB: we haven't implemented CSWAP or PSWAP
            elif inst.name == "SWAP":
                new_prog += _SWAP(*inst.qubits)
            elif inst.name == "T":
                new_prog += _T(inst.qubits[0], needs_dagger)
            elif inst.name == "X":
                new_prog += _X(inst.qubits[0])
            elif inst.name == "XY":
                new_prog += XY(angle_param, *inst.qubits)
            elif inst.name == "Y":
                new_prog += _Y(inst.qubits[0])
            elif inst.name == "Z":
                new_prog += _Z(inst.qubits[0])
            elif inst.name in [gate.name for gate in new_prog.defined_gates]:
                if needs_dagger and inst.name not in daggered_defgates:
                    new_prog.defgate(inst.name + "DAG", inst.matrix.T.conj())
                    daggered_defgates.append(inst.name)
                new_prog += inst
            else:
                raise ValueError(f"Unknown gate instruction {inst}")

        else:
            new_prog += inst

    new_prog.native_quil_metadata = {
        "final_rewiring": None,
        "gate_depth": None,
        "gate_volume": None,
        "multiqubit_gate_depth": None,
        "program_duration": None,
        "program_fidelity": None,
        "topological_swaps": 0,
    }
    return new_prog
