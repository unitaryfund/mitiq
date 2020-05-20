import pytest
import numpy as np

from mitiq.benchmarks.rand_bench import rb_circuits


def test_rb_circuits():
    depths = range(2, 10, 2)

    # test single qubit RB
    for trials in [2, 3]:
        circuits = rb_circuits(n_qubits=1, num_cfds=depths, trials=trials)
        for qc in circuits:
            # we check the ground state population to ignore any global phase
            wvf = qc.final_wavefunction()
            zero_prob = abs(wvf[0]**2)
            assert np.isclose(zero_prob, 1)

    # test two qubit RB
    for trials in [2, 3]:
        circuits = rb_circuits(n_qubits=2, num_cfds=depths, trials=trials)
        for qc in circuits:
            # we check the ground state population to ignore any global phase
            wvf = qc.final_wavefunction()
            zero_prob = abs(wvf[0]**2)
            assert np.isclose(zero_prob, 1)
