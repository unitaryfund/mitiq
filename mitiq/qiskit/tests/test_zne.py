# test_adaptive_zne.py
import numpy as np

from mitiq.factories import LinearFactory 
from mitiq.zne import mitigate
import mitiq.qiskit.qiskit_utils as qs


def test_adaptive_zne_qiskit():
    rand_circ = qs.random_identity_circuit(depth=30)
    rand_circ.measure(0, 0)

    fac = LinearFactory([1.0, 1.5, 2.0])
    instack, outstack = mitigate(rand_circ, fac, qs.scale_noise, qs.run_program)
    xx = fac.reduce(instack, outstack)

    assert np.isclose(xx, 1.0, atol=1.e-1)
