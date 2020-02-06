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

# Test unitary folding
print("Testing unitary folding...")
for c in strech_vals:
    circ = random_circuit(10)
    out = unitary_folding(circ, c)
    actual_c = len(out) / len(circ)
    print("Input stretch: {:}    Real stretch: {}".format(c, actual_c))
    assert np.isclose(c, actual_c, atol=1.e-1)
    # Uncomment to print input and output circuits
    #print("input \n", circ)
    #print("output \n", out)

# Test local_folding
print("Testing local folding...")
for c in strech_vals:
    circ = random_circuit(10)
    out = local_folding(circ, c)
    actual_c = len(out) / len(circ)
    print("Input stretch: {:}    Real stretch: {}".format(c, actual_c))
    assert np.isclose(c, actual_c, atol=1.e-1)
    # Uncomment to print input and output circuits
    #print("input \n", circ)
    #print("output \n", out)