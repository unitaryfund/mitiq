# Copyright (C) 2020 Unitary Fund
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Tests for zne.py with Qiskit backend."""
# from mitiq._typing import QPROGRAM
# import numpy as np
#
# from qiskit import ClassicalRegister, QuantumCircuit
#
# from mitiq.zne.inference import RichardsonFactory, ExpFactory
# from mitiq.zne import (
#     execute_with_zne,
#     mitigate_executor,
#     zne_decorator,
# )
# from mitiq.mitiq_qiskit.qiskit_utils import (
#     random_one_qubit_identity_circuit,
#     run_program,
#     scale_noise,
# )
#
# TEST_DEPTH = 30
# CIRCUIT_SEED = 1
# QISKIT_SEED = 1337
#
#
# def measure(circuit, qid) -> QuantumCircuit:
#     """Apply the measure method on the first qubit of a quantum circuit
#     given a classical register.
#
#     Args:
#         circuit: Quantum circuit.
#         qid: classical register.
#
#     Returns:
#         circuit: circuit after the measurement.
#     """
#     # Ensure that we have a classical register of enough size available
#     if len(circuit.clbits) == 0:
#         reg = ClassicalRegister(qid + 1, "creg")
#         circuit.add_register(reg)
#     circuit.measure(0, qid)
#     return circuit
#
#
# def basic_executor(qp: QPROGRAM, shots: int = 500) -> float:
#     """Runs a program.
#
#         Args:
#         qp: quantum program.
#         shots: number of executions of the program.
#
#     Returns:
#         A float.
#     """
#     return run_program(qp, shots, QISKIT_SEED)
#

# def test_run_factory():
#     """Tests qrun of a Richardson Factory."""
#     qp = random_one_qubit_identity_circuit(num_cliffords=TEST_DEPTH)
#     qp = measure(qp, 0)
#     fac = RichardsonFactory([1.0, 2.0, 3.0])
#     fac.run(qp, basic_executor, scale_noise)
#     result = fac.reduce()
#     assert np.isclose(result, 1.0, atol=1.0e-1)
#
#
# def test_execute_with_zne():
#     """Tests a random identity circuit with zero-noise extrapolation.
#     """
#     rand_circ = random_one_qubit_identity_circuit(num_cliffords=TEST_DEPTH)
#     qp = measure(rand_circ, qid=0)
#     result = execute_with_zne(qp, basic_executor, scale_noise=scale_noise)
#     assert np.isclose(result, 1.0, atol=1.0e-1)
#
#
# def test_mitigate_executor():
#     """Tests a random identity circuit executor."""
#     rand_circ = random_one_qubit_identity_circuit(num_cliffords=TEST_DEPTH)
#     qp = measure(rand_circ, qid=0)
#     new_executor = mitigate_executor(basic_executor, scale_noise=scale_noise)
#     # bad_result is computed with native noise (scale = 1)
#     bad_result = basic_executor(scale_noise(qp, 1))
#     good_result = new_executor(qp)
#     assert not np.isclose(bad_result, 1.0, atol=1.0e-1)
#     assert np.isclose(good_result, 1.0, atol=1.0e-1)
#
#
# @zne_decorator(scale_noise=scale_noise)
# def decorated_executor(qp: QPROGRAM) -> float:
#     return basic_executor(qp)
#
#
# def test_zne_decorator():
#     """Tests a zne decorator."""
#     rand_circ = random_one_qubit_identity_circuit(num_cliffords=TEST_DEPTH)
#     qp = measure(rand_circ, qid=0)
#     # bad_result is computed with native noise (scale = 1)
#     bad_result = basic_executor(scale_noise(qp, 1))
#     good_result = decorated_executor(qp)
#     assert not np.isclose(bad_result, 1.0, atol=1.0e-1)
#     assert np.isclose(good_result, 1.0, atol=1.0e-1)
#
#
# def test_run_factory_with_number_of_shots():
#     """Tests "run" method of an ExpFactory with shot_list."""
#     qp = random_one_qubit_identity_circuit(num_cliffords=TEST_DEPTH)
#     qp = measure(qp, 0)
#     fac = ExpFactory([1.0, 2.0, 3.0], shot_list=[10 ** 4, 10 ** 5, 10 ** 6])
#     fac.run(qp, basic_executor, scale_noise=scale_noise)
#     result = fac.reduce()
#     assert np.isclose(result, 1.0, atol=1.0e-1)
#     assert fac._instack[0] == {"scale_factor": 1.0, "shots": 10 ** 4}
#     assert fac._instack[1] == {"scale_factor": 2.0, "shots": 10 ** 5}
#     assert fac._instack[2] == {"scale_factor": 3.0, "shots": 10 ** 6}
#
#
# def test_mitigate_executor_with_shot_list():
#     """Tests the mitigation of an executor using different shots
#     for each noise scale factor.
#     """
#     rand_circ = random_one_qubit_identity_circuit(num_cliffords=TEST_DEPTH)
#     qp = measure(rand_circ, qid=0)
#
#     fac = RichardsonFactory(
#         [1.0, 2.0, 3.0], shot_list=[10 ** 4, 10 ** 5, 10 ** 6]
#     )
#     new_executor = mitigate_executor(
#         basic_executor, scale_noise=scale_noise, factory=fac
#     )
#     # bad_result is computed with native noise (scale = 1)
#     bad_result = basic_executor(scale_noise(qp, 1))
#     good_result = new_executor(qp)
#     assert not np.isclose(bad_result, 1.0, atol=1.0e-1)
#     assert np.isclose(good_result, 1.0, atol=1.0e-1)
#     assert fac._instack[0] == {"scale_factor": 1.0, "shots": 10 ** 4}
#     assert fac._instack[1] == {"scale_factor": 2.0, "shots": 10 ** 5}
#     assert fac._instack[2] == {"scale_factor": 3.0, "shots": 10 ** 6}
