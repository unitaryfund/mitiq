import random
import cirq

# ==================================================
# Gate level folding
# ==================================================

def sampling_stretcher(circuit, stretch):
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

def start_stretcher(circuit, stretch):
    """Applies the map G -> G G^dag G to a subset of gates of the input circuit (sequentially
    starting from the beginning).
    Returns a circuit of depth approximately equal to stretch*len(circuit).
    The stretch factor can be any real number within 1 and 3."""
    
    if not ((stretch >= 1) and (stretch <= 3)):
        raise ValueError("The stretch factor must be a real number within 1 and 3.")
    
    depth = len(circuit)
    reduced_depth = int(depth * (stretch - 1) / 2)
    sub_indices = list(range(reduced_depth))
    return fold_gates(circuit, sub_indices)

def fold_gates(circuit, sub_indices):
    out = cirq.Circuit()
    # sequentially append gates to out, folding only if j is in sub_indices
    for j, gate in enumerate(circuit):
        out += gate
        if j in sub_indices:
            out += cirq.inverse(gate)
            out += gate
    return out

def local_folding(circuit, stretch, stretcher = start_stretcher):
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

def unitary_folding(circuit, stretch):
    """Applies global unitary folding and a final partial folding of the input circuit.
    Returns a circuit of depth approximately equal to stretch*len(circuit).
    The stretch factor can be any real number >= 1."""

    if not (stretch >= 1):
        raise ValueError("The stretch factor must be a real number >= 1.")

    num_foldings, fractional_stretch = divmod(stretch - 1, 2)

    # global folding
    eye = cirq.Circuit()
    for _ in range(int(num_foldings)):
        eye += cirq.inverse(circuit) + circuit

    # partial folding
    depth = len(circuit)
    fractional_depth = int(depth * fractional_stretch / 2)
    if fractional_depth != 0:
        eye += cirq.inverse(circuit[-fractional_depth:]) + circuit[-fractional_depth:]

    return circuit + eye