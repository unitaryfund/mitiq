
import random
# PyQuil
from pyquil import Program
#from pyquil.gates import X,Y,Z, MEASURE

def local_folding(prog, stretch, sampling=False): 
    """Applies the map G -> G G^dag G to ((stretch-1)*len(prog)//2) to a subset of gates 
    of the input circuit. For large stretch factors the function is recursively repeated. 
    Returns a circuit of depth approximately equal to stretch*len(prog).
    The stretch factor can be any real number >= 1."""
    
    if not (stretch > 1):
        raise ValueError("The stretch factor must be a real number >= 1.")
    
    out = Program()

    if stretch <= 3:
        # select a fraction of subindices
        d = len(prog)
        f = int(d * (stretch - 1) / 2)
        if sampling == True:
            # indices of f random gates
            sub_indices = random.sample(range(d), f)
        else:
            # indices of the first f gates
            sub_indices = list(range(f))

        # sequentially append gates to _prog, folding only if j is in sub_indices
        for j, gate in enumerate(prog):
            out += gate
            if j in sub_indices:
                out += prog[j: j + 1].dagger() # this trick avoids affecting the gate
                out += gate

        return out
    else:
        # recursive application for large stretching
        _ = local_folding(prog, 3, sampling)

        return local_folding( _ , stretch / 3, sampling)

def unitary_folding(prog, stretch):
    """Applies global unitary folding and a final partial folding of the input circuit.
    Returns a circuit of depth approximately equal to stretch*len(prog).
    The stretch factor can be any real number >= 1."""
    
    if not (stretch > 1):
        raise ValueError("The stretch factor must be a real number >= 1.")
    
    d, r = divmod(stretch - 1, 2)

    # global folding
    eye = Program()
    for j in range(int(d)):
        eye += prog.dagger() + prog
    
    # partial folding
    partial = int(len(prog) * r / 2)
    if partial != 0:
        eye += prog[-partial:].dagger() + prog[-partial:]

    return prog + eye