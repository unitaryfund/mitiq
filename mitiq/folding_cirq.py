import random
import cirq

def local_folding(circuit, stretch, sampling=False):
    """Applies the map G -> G G^dag G to ((stretch-1)*len(circuit)//2) to a subset of gates 
    of the input circuit. For large stretch factors the function is recursively repeated. 
    Returns a circuit of depth approximately equal to stretch*len(circuit).
    The stretch factor can be any real number >= 1."""

    if not (stretch >= 1):
        raise ValueError("The stretch factor must be a real number >= 1.")

    out = cirq.Circuit()

    if stretch <= 3:
        # select a fraction of subindices
        d = len(circuit)
        f = int(d * (stretch - 1) / 2)
        if sampling == True:
            # indices of f random gates
            sub_indices = random.sample(range(d), f)
        else:
            # indices of the first f gates
            sub_indices = list(range(f))

        # sequentially append gates to out, folding only if j is in sub_indices
        for j, gate in enumerate(circuit):
            out += gate
            if j in sub_indices:
                out += cirq.inverse(gate)
                out += gate

        return out
    else:
        # recursive application for large stretching
        _ = local_folding(circuit, 3, sampling)

        return local_folding(_, stretch / 3, sampling)


def unitary_folding(circuit, stretch):
    """Applies global unitary folding and a final partial folding of the input circuit.
    Returns a circuit of depth approximately equal to stretch*len(circuit).
    The stretch factor can be any real number >= 1."""

    if not (stretch >= 1):
        raise ValueError("The stretch factor must be a real number >= 1.")

    d, r = divmod(stretch - 1, 2)

    # global folding
    eye = cirq.Circuit()
    for j in range(int(d)):
        eye += cirq.inverse(circuit) + circuit

    # partial folding
    partial = int(len(circuit) * r / 2)
    if partial != 0:
        eye += cirq.inverse(circuit[-partial:]) + circuit[-partial:]

    return circuit + eye