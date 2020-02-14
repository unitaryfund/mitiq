import numpy as np
import random
import cirq
from mitiq.folding_cirq import local_folding, unitary_folding

def random_circuit(depth):
    """Returns a single-qubit random circuit based on Pauli gates."""
    # defines qubit
    q = cirq.GridQubit(0, 0)
    circuit = cirq.Circuit()
    for _ in range(depth):
        circuit += random.choice([cirq.X(q), cirq.Y(q), cirq.Z(q)])
    return circuit

strech_vals = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.5, 4.5, 5.0]
depth = 100

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
        out = local_folding(circ, c)
        actual_c = len(out) / len(circ)
        print("Input stretch: {:}    Real stretch: {}".format(c, actual_c))
        assert np.isclose(c, actual_c, atol=1.0e-1)

def test_local_folding_withsamp():
    print("Testing local folding (random sampling)...")
    for c in strech_vals:
        circ = random_circuit(depth)
        out = local_folding(circ, c)
        actual_c = len(out) / len(circ)
        print("Input stretch: {:}    Real stretch: {}".format(c, actual_c))
        assert np.isclose(c, actual_c, atol=1.0e-1)
