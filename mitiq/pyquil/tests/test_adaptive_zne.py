# test_adaptive_zne.py
import numpy as np

from mitiq.factories import LinearFactory
from mitiq.zne import qrun_factory, execute_with_zne, zne_decorator
from mitiq.pyquil.pyquil_utils import random_identity_circuit, measure, run_program, scale_noise, Program, QVM


def test_execute_with_zne_easy():
    rand_circ = random_identity_circuit(depth=30)
    qp = measure(rand_circ, qid=0)
    result = execute_with_zne(qp, run_program)
    assert np.isclose(result, 1.0, atol=1.e-1)



def basic_executor(qp: Program, shots: int = 100) -> float:
    noisy_qp = scale_noise(qp, 0.5)
    qp.wrap_in_numshots_loop(shots)
    results = QVM.run(noisy_qp)
    expval = (results == [0]).sum() / shots
    return expval

@zne_decorator()
def decorated_executor(qp: Program, shots: int = 100) -> float:
    return basic_executor(qp, shots)


def test_zne_decorator():
    rand_circ = random_identity_circuit(depth=30)
    qp = measure(rand_circ, qid=0)
    bad_result = basic_executor(qp)
    good_result = decorated_executor(qp)
    assert not np.isclose(bad_result, 1.0, atol=1.e-1)
    assert np.isclose(good_result, 1.0, atol=1.e-1)


def test_qrun_factory():
    rand_circ = random_identity_circuit(depth=30)
    qp = measure(rand_circ, qid=0)

    fac = LinearFactory([1.0, 2.0, 3.0])
    qrun_factory(fac, qp, run_program, scale_noise)
    result = fac.reduce()
    assert np.isclose(result, 1.0, atol=1.e-1)
