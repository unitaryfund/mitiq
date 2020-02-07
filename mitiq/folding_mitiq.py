import random


def deduce_platform(circuit):
    """Deduce platform from circuit."""

    pkg_name = circuit.__module__.split(".")[0]
    if pkg_name == 'cirq':
        from cirq import Circuit, inverse
        my_Circuit = Circuit
        my_inverse = inverse
    if pkg_name == 'pyquil':
        from  pyquil import Program
        my_Circuit = Program
        def _inverse(prog):
            return prog.dagger()
        my_inverse = _inverse
    
    return my_Circuit, my_inverse

def local_folding(circuit, stretch, sampling=False):
    """Applies the map G -> G G^dag G to ((stretch-1)*len(circuit)//2) to a subset of gates 
    of the input circuit. For large stretch factors the function is recursively repeated. 
    Returns a circuit of depth approximately equal to stretch*len(circuit).
    The stretch factor can be any real number >= 1."""

    if not (stretch >= 1):
        raise ValueError("The stretch factor must be a real number >= 1.")

    my_Circuit, my_inverse = deduce_platform(circuit)

    out = my_Circuit()

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
                out += my_inverse(circuit[j:j+1]) # don't put gate if using pyquil
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

    my_Circuit, my_inverse = deduce_platform(circuit)

    d, r = divmod(stretch - 1, 2)

    # global folding
    eye = my_Circuit()
    for j in range(int(d)):
        eye += my_inverse(circuit) + circuit

    # partial folding
    partial = int(len(circuit) * r / 2)
    if partial != 0:
        eye += my_inverse(circuit[-partial:]) + circuit[-partial:]

    return circuit + eye