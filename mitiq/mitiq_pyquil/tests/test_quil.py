import numpy as np

from cirq import Circuit, LineQubit
from cirq.ops import (CCNOT, CNOT, CSWAP, CZ, H, I, ISWAP,
                      MeasurementGate, S, SWAP, T, X, Y, Z,
                      rx, ry, rz)

from mitiq.mitiq_pyquil.quil import (circuit_from_quil, cphase, cphase00,
                                     cphase01, cphase10, pswap, xy)


QUIL_PROGRAM = """
DECLARE ro BIT[3]
I 0
I 1
I 2
X 0
Y 1
Z 2
H 0
S 1
T 2
PHASE(pi/8) 0
PHASE(pi/8) 1
PHASE(pi/8) 2
RX(pi/2) 0
RY(pi/2) 1
RZ(pi/2) 2
CZ 0 1
CNOT 1 2
CPHASE(pi/2) 0 1
CPHASE00(pi/2) 1 2
CPHASE01(pi/2) 0 1
CPHASE10(pi/2) 1 2
ISWAP 0 1
PSWAP(pi/2) 1 2
SWAP 0 1
XY(pi/2) 1 2
CCNOT 0 1 2
CSWAP 0 1 2
MEASURE 0 ro[0]
MEASURE 1 ro[1]
MEASURE 2 ro[2]
"""


def test_circuit_from_quil():
    q0, q1, q2 = LineQubit.range(3)
    cirq_circuit = Circuit([I(q0),
                            I(q1),
                            I(q2),
                            X(q0),
                            Y(q1),
                            Z(q2),
                            H(q0),
                            S(q1),
                            T(q2),
                            Z(q0)**(1/8),
                            Z(q1)**(1/8),
                            Z(q2)**(1/8),
                            rx(np.pi/2)(q0),
                            ry(np.pi/2)(q1),
                            rz(np.pi/2)(q2),
                            CZ(q0, q1),
                            CNOT(q1, q2),
                            cphase(np.pi/2)(q0, q1),
                            cphase00(np.pi/2)(q1, q2),
                            cphase01(np.pi/2)(q0, q1),
                            cphase10(np.pi/2)(q1, q2),
                            ISWAP(q0, q1),
                            pswap(np.pi/2)(q1, q2),
                            SWAP(q0, q1),
                            xy(np.pi/2)(q1, q2),
                            CCNOT(q0, q1, q2),
                            CSWAP(q0, q1, q2),
                            MeasurementGate(1, key="ro[0]")(q0),
                            MeasurementGate(1, key="ro[1]")(q1),
                            MeasurementGate(1, key="ro[2]")(q2)])

    quil_circuit = circuit_from_quil(QUIL_PROGRAM)
    assert cirq_circuit == quil_circuit
