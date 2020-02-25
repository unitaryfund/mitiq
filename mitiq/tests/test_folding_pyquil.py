import numpy as np
import random
from pyquil import Program
from pyquil.gates import X, Y, Z, CNOT
from pyquil.unitary_tools import program_unitary
from mitiq.folding_pyquil import (
    fold_local,
    unitary_folding,
    fold_gates_at_random,
    fold_gates_from_left,
)
from copy import deepcopy


STRETCH_VALS = [1.0, 1.3, 2.0, 2.6, 3.0, 3.5, 4.0]
DEPTH = 50
NUM_SHOTS = 7
GATE = [[0, 1], [1, 0]]
GATE_NAME = "X_test"


def random_circuit(depth: int):
    """Returns a 2-qubit random circuit based on a simple gate set."""
    prog = Program()
    gate_set = [X(0), Y(0), Z(0), X(1), Y(1), Z(1), CNOT(0, 1), CNOT(1, 0)]
    for _ in range(depth):
        prog += random.choice(gate_set)

    # we add some metadata to the program since we want to test that
    # properties are correctly passed to the output of folding functions.
    prog.wrap_in_numshots_loop(NUM_SHOTS)
    prog.defgate(GATE_NAME, GATE)
    return prog


def apply_folding_tests(circ: Program, circ_copy: Program, out: Program, stretch: float):
    """Performs a sequece of tests associated to a generic folding function.

    Args:
        circ: Input circuit of the generic folding function.
        circ_copy: Deep copy of the input circuit performed before folding.
        out: Output circuit returned by the generic folding function.
        stretch: Stretch factor used by the generic folding function.
    """
    actual_c = len(out) / len(circ)
    # test length scaling
    assert np.isclose(stretch, actual_c, atol=1.0e-1)
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
    vars_a.pop("_instructions")
    vars_c = vars(out)
    vars_c.pop("_synthesized_instructions")
    vars_c.pop("_instructions")
    assert vars_a == vars_c
    

def test_unitary_folding():
    for c in STRETCH_VALS:
        circ = random_circuit(DEPTH)
        circ_copy = deepcopy(circ)
        out = unitary_folding(circ, c)
        apply_folding_tests(circ, circ_copy, out, c)


def test_fold_local_from_left():
    for c in STRETCH_VALS:
        circ = random_circuit(DEPTH)
        circ_copy = deepcopy(circ)
        out = fold_local(circ, c, fold_gates_from_left)
        apply_folding_tests(circ, circ_copy, out, c)


def test_fold_local_at_random_no_seed():
    for c in STRETCH_VALS:
        circ = random_circuit(DEPTH)
        circ_copy = deepcopy(circ)
        # test random folding without seed
        out = fold_local(circ, c, fold_gates_at_random)
        apply_folding_tests(circ, circ_copy, out, c)
        

def test_fold_local_at_random_with_seed():
    for c in STRETCH_VALS:
        circ = random_circuit(DEPTH)
        circ_copy = deepcopy(circ)
        # test random folding with seed
        out = fold_local(circ, c, fold_gates_at_random, (42,))
        apply_folding_tests(circ, circ_copy, out, c)
