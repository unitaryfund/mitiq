# test_adaptive_zne.py
import numpy as np

from mitiq.adaptive_zne import BatchedGenerator, mitigate
import mitiq.qiskit.qiskit_utils as qs


def test_adaptive_zne_qiskit():
    rand_circ = qs.random_identity_circuit(depth=30)
    rand_circ.measure(0, 0)

    gen = BatchedGenerator([1.0, 2.0, 3.0])
    params, expects = mitigate(rand_circ, gen, qs.scale_noise, qs.run_program)
    xx = gen.reduce(expects)

    assert np.isclose(xx, 1.0, atol=1.e-1)
