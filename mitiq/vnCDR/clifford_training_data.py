from cirq import Circuit

def generate_training_set(
    circuit: cirq.Circuit,
    num_training_circuits: int,
    fraction_non_clifford: float,
    method_select: str = 'random',
    method_replace: str = 'nearest',
    can_compile: bool = True,
    )-> List[cirq.Circuit]:

    '''Returns a list of near-Clifford circuits to act as training data.

    Args:
        circuit: A circuit which computes an observable of interest.
        num_training_circuits: Number of circuits in the returned training set.
        fraction_non_clifford: The (approximate) fraction of non-Clifford gates in each returned circuit.
        can_compile: If True, the circuit is first compiled into the Clifford + Rz gate set, then Rz are mapped to Clifford gates.
         Else, the function attempts to project arbitrary gates into Clifford gates.

    Returns:
        List: [map_to_near_clifford(circuit, fraction_non_clifford, method, can_compile) for _ in range(num_training_circuits)]
                
    '''
