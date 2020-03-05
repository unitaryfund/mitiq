# test_adaptive_zne.py
import numpy as np

from mitiq.factories import LinearFactory 
from mitiq.zne import qrun_factory
import mitiq.qiskit.qiskit_utils as qs


def test_qrun_factory_qiskit():
    rand_circ = qs.random_identity_circuit(depth=30)
    rand_circ.measure(0, 0)
    fac = LinearFactory([1.0, 1.5, 2.0])
    qrun_factory(fac, rand_circ, qs.run_program, qs.scale_noise)
    result = fac.reduce()
    assert np.isclose(result, 1.0, atol=1.e-1)
