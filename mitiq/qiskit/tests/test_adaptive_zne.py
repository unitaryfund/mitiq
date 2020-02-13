# test_adaptive_zne.py
import numpy as np

from mitiq.adaptive_zne import BatchedGenerator, Mitigator, reduce, zne, zne_factory
import mitiq.qiskit.qiskit_utils as qs


def test_adaptive_zne_qiskit():
    rand_circ = qs.random_identity_circuit(depth=30)
    rand_circ.measure(0, 0)

    gen = BatchedGenerator([1.0, 2.0, 3.0])
    mitigator = Mitigator(gen, qs.run_program)
    params, expects = mitigator.mitigate(rand_circ, qs.scale_noise)
    xx = reduce(expects)
    assert np.isclose(xx, 1.0, atol=1.e-1)
