from typing import List, Callable, Optional, Tuple, Any
import numpy as np
from pyquil import Program




# Gate level folding
def fold_gates_at_random(circuit: Program, stretch: float, seed: Optional[int] = None) -> Program:
    """Returns a folded circuit by applying the map G -> G G^dag G to a random subset of gates in the input circuit.

    The folded circuit has a number of gates approximately equal to stretch * n where n is the number of gates in
    the input circuit.

    Args:
        circuit: Circuit to fold.
        stretch: Factor to stretch the circuit by. Any real number in the interval [1, 3].
        seed: [Optional] Integer seed for random number generator.

    Note:
        Folding a single gate adds two gates to the circuit, hence the maximum stretch factor is 3.
    """

    if not (1 <= stretch <= 3):
        raise ValueError("The stretch factor must be a real number between 1 and 3.")

    if seed is not None:
        np.random.seed(seed)

    ngates = len(circuit)
    num_to_fold = int(ngates * (stretch - 1) / 2)
    sub_indices = np.random.choice(range(ngates), num_to_fold, replace=False)
    return fold_gates(circuit, sub_indices)

def fold_gates_from_left(circuit: Program, stretch: float) -> Program:
    """Returns a new folded circuit by applying the map G -> G G^dag G to a subset of gates corresponding to the
    initial part of the circuit.

    The folded circuit has a number of gates approximately equal to stretch * n where n is the number of gates in
    the input circuit.

    Args:
        circuit: Circuit to fold.
        stretch: Factor to stretch the circuit by. Any real number in the interval [1, 3].

    Note:
        Folding a single gate adds two gates to the circuit, hence the maximum stretch factor is 3.
    """

    if not ((stretch >= 1) and (stretch <= 3)):
        raise ValueError("The stretch factor must be a real number within 1 and 3.")

    ngates = len(circuit)
    num_to_fold = int(ngates * (stretch - 1) / 2)
    sub_indices = list(range(num_to_fold))
    return fold_gates(circuit, sub_indices)


def fold_gates(circuit: Program, sub_indices: List[int]) -> Program:
    """Applies the map G -> G G^dag G to a subset of gates of the input circuit
    determined by sub_indices."""

    out = circuit.copy_everything_except_instructions()
    for j, gate in enumerate(circuit):
        out += gate
        if j in sub_indices:
            out += circuit[j : j + 1].dagger()  # trick to avoid mutating the gate
            out += gate
    return out

def fold_local(
        circuit: Program,
        stretch: float,
        fold_method: Callable[[Program, float, Tuple[Any]], Program] = fold_gates_from_left,
        fold_method_args: Tuple[Any] = ()
        ) -> Program:
    """Returns a folded circuit by folding gates according to the input fold method.

    Args:
        circuit: Circuit to fold.
        stretch: Factor to stretch the circuit by.
        fold_method: Function which defines the method for folding gates.
                    (e.g., Randomly selects gates to fold, folds gates starting from left of circuit, etc.)
                    Must have signature
                    def fold_method(circuit: Circuit, stretch: float, **kwargs):
                        ...
                    and return a circuit.
        fold_method_args: Any additional input arguments for the fold_method.
                          The method is called with fold_method(circuit, stretch, *fold_method_args).
            Example:
                fold_method = fold_gates_at_random
                fold_method_args = (1,)

                > Uses a seed of 1 for the fold_gates_at_random method.
    """
    if not (stretch >= 1):
        raise ValueError("The stretch factor must be a real number greater than 1.")

    if stretch <= 3:
        return fold_method(circuit, stretch, *fold_method_args)
    else:
        # recursive iterations for stretch > 3
        _ = fold_local(circuit, 3, fold_method)
        return fold_local(_, stretch / 3, fold_method)


# Circuit level folding
def unitary_folding(circuit: Program, stretch: float) -> Program:
    """Applies global unitary folding and a final partial folding of the input circuit.
    Returns a circuit of depth approximately equal to stretch*len(circuit).
    The stretch factor can be any real number >= 1."""

    if not (stretch >= 1):
        raise ValueError("The stretch factor must be a real number >= 1.")

    # determine the number of integer foldings and the final fractional_stretch
    num_foldings, fractional_stretch = divmod(stretch - 1, 2)

    # integer circuit folding
    out = circuit.copy()
    for _ in range(int(num_foldings)):
        # [:] is used to select only instructions since all the
        # other properties of the input circuit have been already copied.
        out += circuit.dagger()[:] + circuit[:]

    # partial circuit folding.
    ngates = len(circuit)
    num_to_fold = int(ngates * fractional_stretch / 2)
    if num_to_fold != 0:
        out += circuit[-num_to_fold:].dagger() + circuit[-num_to_fold:]

    return out
