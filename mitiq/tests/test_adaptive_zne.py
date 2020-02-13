# test_wz_zne.py
import numpy as np

from mitiq.adaptive_zne import BatchedGenerator, Mitigator, reduce, zne, zne_factory
from mitiq.pyquil.pyquil_utils import random_identity_circuit, measure, run_program, scale_noise, Program, QVM
import mitiq.qiskit.qiskit_utils as qs


def test_adaptive_zne_easy():
    rand_circ = random_identity_circuit(depth=30)
    pq = measure(rand_circ, qid=0)

    xx = zne(run_program)(pq)
    print(xx)
    assert np.isclose(xx, 1.0, atol=1.e-1)


@zne_factory()
def basic_zne_run(pq: Program, shots: int = 100) -> float:
    pq.wrap_in_numshots_loop(shots)
    results = QVM.run(pq)
    expval = (results == [0]).sum() / shots
    return expval


def test_adaptive_zne_easy_decorator():
    rand_circ = random_identity_circuit(depth=30)
    pq = measure(rand_circ, qid=0)

    xx = basic_zne_run(pq)
    assert np.isclose(xx, 1.0, atol=1.e-1)


def test_adaptive_zne_pyquil():
    rand_circ = random_identity_circuit(depth=30)
    pq = measure(rand_circ, qid=0)

    gen = BatchedGenerator([1.0, 2.0, 3.0])
    mitigator = Mitigator(gen, run_program)
    params, expects = mitigator.mitigate(pq, scale_noise)
    xx = reduce(expects)
    assert np.isclose(xx, 1.0, atol=1.e-1)


def test_adaptive_zne_qiskit():
    rand_circ = qs.random_identity_circuit(depth=30)
    rand_circ.measure(0, 0)

    gen = BatchedGenerator([1.0, 2.0, 3.0])
    mitigator = Mitigator(gen, qs.run_program)
    params, expects = mitigator.mitigate(rand_circ, qs.scale_noise)
    xx = reduce(expects)
    assert np.isclose(xx, 1.0, atol=1.e-1)
