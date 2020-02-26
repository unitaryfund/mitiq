# test_adaptive_zne.py
import numpy as np

<<<<<<< HEAD
from mitiq.adaptive_zne import BatchedGenerator, Mitigator, reduce
=======
from mitiq.adaptive_zne import BatchedFactory, mitigate
>>>>>>> origin/master
import mitiq.qiskit.qiskit_utils as qs


def test_adaptive_zne_qiskit():
    rand_circ = qs.random_identity_circuit(depth=30)
    rand_circ.measure(0, 0)

<<<<<<< HEAD
    gen = BatchedGenerator([1.0, 2.0, 3.0])
    mitigator = Mitigator(gen, qs.run_program)
    params, expects = mitigator.mitigate(rand_circ, qs.scale_noise)
    xx = reduce(expects)
=======
    fac = BatchedFactory([1.0, 2.0, 3.0])
    params, expects = mitigate(rand_circ, fac, qs.scale_noise, qs.run_program)
    xx = fac.reduce(expects)

>>>>>>> origin/master
    assert np.isclose(xx, 1.0, atol=1.e-1)
