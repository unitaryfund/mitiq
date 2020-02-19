import numpy as np
import random
from pyquil import Program
from pyquil.gates import X, Y, Z, CNOT
from pyquil.unitary_tools import program_unitary
from mitiq.folding_pyquil import local_folding, unitary_folding, sampling_stretcher, left_stretcher
from copy import deepcopy


STRETCH_VALS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
DEPTH = 50
NUM_SHOTS = 3


def random_circuit(depth: int):
    """Returns a 2-qubit random circuit based on a simple gate set."""
    prog = Program()
    gate_set = [X(0), Y(0), Z(0), X(1), Y(1), Z(1), CNOT(0, 1),  CNOT(1, 0)]
    for _ in range(depth):
        prog += random.choice(gate_set)
    prog.num_shots = NUM_SHOTS
    return prog


def test_unitary_folding():
    for c in STRETCH_VALS:
        circ = random_circuit(DEPTH)
        circ_copy = deepcopy(circ)
        out = unitary_folding(circ, c)
        actual_c = len(out) / len(circ)
        # test lenght scaling
        assert np.isclose(c, actual_c, atol=1.0e-1)
        # test unitaries are equal
        u_in = program_unitary(circ, 2)
        u_out = program_unitary(out, 2)
        assert (u_in == u_out).all
        # test input is not mutated up to "_synthesized_instructions"
        vars_a = vars(circ_copy)
        vars_b = vars(circ)
        vars_a.pop("_synthesized_instructions")
        vars_b.pop("_synthesized_instructions")
        assert vars_a == vars_b
        # test input properties are passed to output
        # up to "_synthesized_instructions" and "_instructions"
        # we also need to ignore "num_shots" (PyQuil bug?)
        vars_a.pop("_instructions")
        vars_a.pop("num_shots")
        vars_b = vars(out)
        vars_b.pop("_synthesized_instructions")
        vars_b.pop("_instructions")
        vars_b.pop("num_shots")
        assert vars_a == vars_b


def test_local_folding_nosamp():
    for c in STRETCH_VALS:
        circ = random_circuit(DEPTH)
        circ_copy = deepcopy(circ)
        out = local_folding(circ, c, stretcher=sampling_stretcher)
        actual_c = len(out) / len(circ)
        # test lenght scaling
        assert np.isclose(c, actual_c, atol=1.0e-1)
        # test unitaries are equal
        u_in = program_unitary(circ, 2)
        u_out = program_unitary(out, 2)
        assert (u_in == u_out).all
        # test input is not mutated up to "_synthesized_instructions"
        vars_a = vars(circ_copy)
        vars_b = vars(circ)
        vars_a.pop("_synthesized_instructions")
        vars_b.pop("_synthesized_instructions")
        assert vars_a == vars_b
        # test input properties are passed to output
        # up to "_synthesized_instructions" and "_instructions"
        # we also need to ignore "num_shots" (PyQuil bug?)
        vars_a.pop("_instructions")
        vars_a.pop("num_shots")
        vars_b = vars(out)
        vars_b.pop("_synthesized_instructions")
        vars_b.pop("_instructions")
        vars_b.pop("num_shots")
        assert vars_a == vars_b

def test_local_folding_withsamp():
    for c in STRETCH_VALS:
        circ = random_circuit(DEPTH)
        circ_copy = deepcopy(circ)
        out = local_folding(circ, c, stretcher=left_stretcher)
        actual_c = len(out) / len(circ)
        # test lenght scaling
        assert np.isclose(c, actual_c, atol=1.0e-1)
        # test unitaries are equal
        u_in = program_unitary(circ, 2)
        u_out = program_unitary(out, 2)
        assert (u_in == u_out).all
         # test input is not mutated up to "_synthesized_instructions"
        vars_a = vars(circ_copy)
        vars_b = vars(circ)
        vars_a.pop("_synthesized_instructions")
        vars_b.pop("_synthesized_instructions")
        assert vars_a == vars_b
        # test input properties are passed to output
        # up to "_synthesized_instructions" and "_instructions"
        # we also need to ignore "num_shots" (PyQuil bug?)
        vars_a.pop("_instructions")
        vars_a.pop("num_shots")
        vars_b = vars(out)
        vars_b.pop("_synthesized_instructions")
        vars_b.pop("_instructions")
        vars_b.pop("num_shots")
        assert vars_a == vars_b