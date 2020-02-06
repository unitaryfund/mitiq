import numpy as np

from mitiq.folding_pyquil import local_folding, unitary_folding


def random_circuit(depth):
    """Returns a single-qubit random circuit based on Pauli gates."""
    prog = Program()  
    for _ in range(depth):
        prog += np.random.choice([X(0), Y(0), Z(0)])
    return prog


# Test unitary folding
circ = random_circuit(20)
out = unitary_folding(circ, 3.6)
print("Real stretch:", len(out)/len(circ))
# Uncomment to see input and output circuits
#print("input \n", circ)
#print("output \n", out)

# Test local folding
circ = random_circuit(20)
out = local_folding(circ, 3.6, sampling=True)
print("Real stretch:", len(out)/len(circ))
# Uncomment to see input and output circuits
#print("input \n", circ)
#print("output \n", out)