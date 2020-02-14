import numpy as np
import random
from pyquil import Program
from pyquil.gates import X, Y, Z
from mitiq.folding_pyquil import local_folding, unitary_folding

def random_circuit(depth):
    """Returns a single-qubit random circuit based on Pauli gates."""
    prog = Program()
    for _ in range(depth):
        prog += random.choice([X(0), Y(0), Z(0)])
    return prog

strech_vals = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.5, 4.5, 5.0]
depth = 10

def test_unitary_folding():
    print("Testing unitary folding...")
    for c in strech_vals:
        circ = random_circuit(depth)
        out = unitary_folding(circ, c)
        actual_c = len(out) / len(circ)
        print("Input stretch: {:}    Real stretch: {}".format(c, actual_c))
        assert np.isclose(c, actual_c, atol=1.0e-1)

def test_local_folding_nosamp():
    print("Testing local folding (no sampling)...")
    for c in strech_vals:
        circ = random_circuit(depth)
        out = local_folding(circ, c, sampling=False)
        actual_c = len(out) / len(circ)
        print("Input stretch: {:}    Real stretch: {}".format(c, actual_c))
        assert np.isclose(c, actual_c, atol=1.0e-1)

def test_local_folding_withsamp():
    print("Testing local folding (random sampling)...")
    for c in strech_vals:
        circ = random_circuit(depth)
        out = local_folding(circ, c, sampling=True)
        actual_c = len(out) / len(circ)
        print("Input stretch: {:}    Real stretch: {}".format(c, actual_c))
        assert np.isclose(c, actual_c, atol=1.0e-1)
