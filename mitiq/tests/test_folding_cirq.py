import numpy as np
import random
from cirq import Circuit, GridQubit, X, Y, Z
from mitiq.folding_cirq import local_folding, unitary_folding


def random_circuit(depth: int):
    """Returns a single-qubit random circuit based on Pauli gates."""
    # defines qubit
    q = GridQubit(0, 0)
    circuit = Circuit()
    for _ in range(depth):
        circuit += random.choice([X(q), Y(q), Z(q)])
    return circuit


STRETCH_VALS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.5, 4.5, 5.0]
DEPTH = 100


def test_unitary_folding():
    for c in STRETCH_VALS:
        circ = random_circuit(DEPTH)
        out = unitary_folding(circ, c)
        actual_c = len(out) / len(circ)
        assert np.isclose(c, actual_c, atol=1.0e-1)


def test_local_folding_nosamp():
    for c in STRETCH_VALS:
        circ = random_circuit(DEPTH)
        out = local_folding(circ, c)
        actual_c = len(out) / len(circ)
        assert np.isclose(c, actual_c, atol=1.0e-1)


def test_local_folding_withsamp():
    for c in STRETCH_VALS:
        circ = random_circuit(DEPTH)
        out = local_folding(circ, c)
        actual_c = len(out) / len(circ)
        assert np.isclose(c, actual_c, atol=1.0e-1)

