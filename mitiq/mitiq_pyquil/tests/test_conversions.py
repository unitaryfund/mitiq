import numpy as np
from pyquil import Program
from pyquil.gates import CNOT, H, RZ

# from mitiq.mitiq_pyquil.conversions import from_pyquil, to_pyquil


def maxcut_qaoa_program(gamma: float) -> Program:
    """
    Generates a 2Q MAXCUT QAOA circuit with beta = pi/8 and with the provided gamma.

    Args:
        gamma: One of the two variational parameters (the other is fixed).
    Returns:
        A 2Q MAXCUT QAOA circuit with fixed beta and gamma.
    """
    q0, q1 = (0, 1)
    p = Program()
    p += H(q0)
    p += H(q1)
    p += CNOT(q0, q1)
    p += RZ(2 * gamma, q1)
    p += CNOT(q0, q1)
    p += H(q0)
    p += H(q1)
    p += RZ(np.pi / 4, q0)
    p += RZ(np.pi / 4, q1)
    p += H(q0)
    p += H(q1)
    return p


# TODO: bug in to_quil, need to write a Cirq PR to fix
# def test_to_pyquil_from_pyquil():
#     p = maxcut_qaoa_program(np.pi)
#     assert p.out() == to_pyquil(from_pyquil(p)).out()
