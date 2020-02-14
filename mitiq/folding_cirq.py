from typing import List, Callable
import random
from cirq import Circuit, inverse


# ==================================================
# Gate level folding
# ==================================================


def sampling_stretcher(circuit: Circuit, stretch: float) -> Circuit:
    """Applies the map G -> G G^dag G to a random subset of gates 
    of the input circuit.
    Returns a circuit of depth approximately equal to stretch*len(circuit).
    The stretch factor can be any real number within 1 and 3."""

    if not ((stretch >= 1) and (stretch <= 3)):
        raise ValueError("The stretch factor must be a real number within 1 and 3.")

    depth = len(circuit)
    reduced_depth = int(depth * (stretch - 1) / 2)
    sub_indices = random.sample(range(depth), reduced_depth)
    return fold_gates(circuit, sub_indices)


def left_stretcher(circuit: Circuit, stretch: float) -> Circuit:
    """Applies the map G -> G G^dag G to a subset of gates of the input circuit (first part of it).
    Returns a circuit of depth approximately equal to stretch*len(circuit).
    The stretch factor can be any real number within 1 and 3."""

    if not ((stretch >= 1) and (stretch <= 3)):
        raise ValueError("The stretch factor must be a real number within 1 and 3.")

    depth = len(circuit)
    reduced_depth = int(depth * (stretch - 1) / 2)
    sub_indices = list(range(reduced_depth))
    return fold_gates(circuit, sub_indices)


def fold_gates(circuit: Circuit, sub_indices: List[int]) -> Circuit:
    """Applies the map G -> G G^dag G to a subset of gates of the input circuit
    determined by sub_indices."""

    out = Circuit()
    for j, gate in enumerate(circuit):
        out += gate
        if j in sub_indices:
            out += inverse(gate)
            out += gate
    return out


def local_folding(circuit: Circuit, stretch: float, stretcher: Callable=left_stretcher) -> Circuit:
    """Applies the map G -> G G^dag G to a subset of gates of the input circuit.
    Returns a circuit of depth approximately equal to stretch*len(circuit).
    The stretch factor can be any real number >= 1."""

    if not (stretch >= 1):
        raise ValueError("The stretch factor must be a real number >= 1.")

    if stretch <= 3:
        return stretcher(circuit, stretch)
    else:
        # recursive iterations for stretch > 3
        _ = local_folding(circuit, 3, stretcher)
        return local_folding(_, stretch / 3, stretcher)


# ==================================================
# Circuit level folding
# ==================================================


def unitary_folding(circuit: Circuit, stretch: float) -> Circuit:
    """Applies global unitary folding and a final partial folding of the input circuit.
    Returns a circuit of depth approximately equal to stretch*len(circuit).
    The stretch factor can be any real number >= 1."""

    if not (stretch >= 1):
        raise ValueError("The stretch factor must be a real number >= 1.")

    # determine the number of integer foldings and the final fractional_stretch
    num_foldings, fractional_stretch = divmod(stretch - 1, 2)

    # integer circuit folding
    eye = Circuit()
    for _ in range(int(num_foldings)):
        eye += inverse(circuit) + circuit

    # partial circuit folding.
    depth = len(circuit)
    fractional_depth = int(depth * fractional_stretch / 2)
    if fractional_depth != 0:
        eye += inverse(circuit[-fractional_depth:]) + circuit[-fractional_depth:]

    return circuit + eye
   
