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
from typing import Any, cast, Union

import numpy as np

from pyquil.gates import RX, RZ, CZ, I, XY
from pyquil.quil import Program, Qubit, QubitPlaceholder, FormalArgument
from pyquil.quilbase import Gate, Expression


QubitLike = Union[Qubit, QubitPlaceholder, FormalArgument]
AngleLike = Union[Expression, Any, complex]


def _CCNOT(q0: QubitLike, q1: QubitLike, q2: QubitLike) -> Program:
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


def _CNOT(q0: QubitLike, q1: QubitLike) -> Program:
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


def _CPHASE(angle: AngleLike, q0: QubitLike, q1: QubitLike) -> Program:
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


def _H(q: QubitLike) -> Program:
    """
    A Hadamard in terms of RX(+-pi/2) and RZ(theta)

    .. note:
        This introduces a different global phase! Don't control this gate.
    """
    p = Program()
    p.inst(_RY(-np.pi / 2, q))
    p.inst(RZ(np.pi, q))
    return p


def _ISWAP(q0: QubitLike, q1: QubitLike) -> Program:
    """
    An ISWAP as an XY(pi). Of course, assumes XY is available.
    """
    p = Program()
    p += XY(np.pi, q0, q1)
    return p


def _PHASE(angle: AngleLike, q: QubitLike) -> Program:
    """
    from quilc:

    (define-compiler PHASE-to-RZ ((phase-gate ("PHASE" (alpha) q)))
       (inst "RZ" `(,alpha) q))
    """
    p = Program()
    p += RZ(angle, q)
    return p


def _RX(angle: AngleLike, q: QubitLike) -> Program:
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


def _RY(angle: AngleLike, q: QubitLike) -> Program:
    """
    A RY in terms of RX(+-pi/2) and RZ(theta)
    """
    p = Program()
    p += RX(pi / 2, q)
    p += RZ(angle, q)
    p += RX(-pi / 2, q)
    return p


def _S(q: QubitLike) -> Program:
    """
    An S in terms of RZ(theta)
    """
    return Program(RZ(np.pi / 2, q))


def _SWAP(q0: QubitLike, q1: QubitLike) -> Program:
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


def _T(q: QubitLike, dagger: bool = False) -> Program:
    """
    A T in terms of RZ(theta)
    """
    if dagger:
        return Program(RZ(-np.pi / 4, q))
    else:
        return Program(RZ(np.pi / 4, q))


def _X(q: QubitLike) -> Program:
    """
    An X in terms of RX(pi)

    .. note:
        This introduces a global phase! Don't control this gate.
    """
    p = Program()
    p += RX(np.pi, q)
    return p


def _Y(q: QubitLike) -> Program:
    """
    A Y in terms of _RY
    """
    p = Program()
    p += _RY(np.pi, q)
    return p


def _Z(q: QubitLike) -> Program:
    """
    A Z in terms of RZ
    """
    p = Program()
    p += RZ(np.pi, q)
    return p


def is_magic_angle(angle: AngleLike) -> bool:
    """
    Checks to see if an angle is 0, +/-pi/2, or +/-pi.
    """
    return bool(
        np.isclose(np.abs(cast(float, angle)), pi / 2)
        or np.isclose(np.abs(cast(float, angle)), pi)
        or np.isclose(cast(float, angle), 0.0)
    )


def basic_compile(program: Program) -> Program:
    """
    A rudimentary but predictable compiler.

    No rewiring or optimization is done by this compilation step. There may be
    some gates that are not yet supported. Gates defined in the input program
    are included without change in the output program.

    :param program: A program to be compiled to native quil with simple
        replacements.
    :return: A program with some of the input non-native quil gates replaced
        with basic native quil gate implementations.
    """
    new_prog = Program()
    new_prog.num_shots = program.num_shots
    new_prog.inst(program.defined_gates)

    for inst in program:
        if isinstance(inst, Gate):
            if inst.name == "CCNOT":
                new_prog += _CCNOT(*inst.qubits)
            elif inst.name == "CNOT":
                new_prog += _CNOT(*inst.qubits)
            # NB: we haven't implemented CPHASE00/01/10
            elif inst.name == "CPHASE":
                angle_param = inst.params[0]
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
                angle_param = inst.params[0]
                new_prog += _PHASE(angle_param, inst.qubits[0])
            elif inst.name == "RX":
                angle_param = inst.params[0]
                if is_magic_angle(inst.params[0]):
                    # in case dagger
                    new_prog += RX(angle_param, inst.qubits[0])
                else:
                    new_prog += _RX(angle_param, inst.qubits[0])
            elif inst.name == "RY":
                angle_param = inst.params[0]
                new_prog += _RY(angle_param, inst.qubits[0])
            elif inst.name == "RZ":
                # in case dagger
                angle_param = inst.params[0]
                new_prog += RZ(angle_param, inst.qubits[0])
            elif inst.name == "S":
                new_prog += _S(inst.qubits[0])
            # NB: we haven't implemented CSWAP or PSWAP
            elif inst.name == "SWAP":
                new_prog += _SWAP(*inst.qubits)
            elif inst.name == "T":
                new_prog += _T(inst.qubits[0])
            elif inst.name == "X":
                new_prog += _X(inst.qubits[0])
            elif inst.name == "XY":
                angle_param = inst.params[0]
                new_prog += XY(angle_param, *inst.qubits)
            elif inst.name == "Y":
                new_prog += _Y(inst.qubits[0])
            elif inst.name == "Z":
                new_prog += _Z(inst.qubits[0])
            elif inst.name in [gate.name for gate in new_prog.defined_gates]:
                new_prog += inst
            else:
                raise ValueError(f"Unknown gate instruction {inst}")
        else:
            new_prog += inst

    new_prog.native_quil_metadata = {  # type: ignore[assignment]
        "final_rewiring": None,
        "gate_depth": None,
        "gate_volume": None,
        "multiqubit_gate_depth": None,
        "program_duration": None,
        "program_fidelity": None,
        "topological_swaps": 0,
    }
    return new_prog
