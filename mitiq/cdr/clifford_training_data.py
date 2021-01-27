from cirq import Circuit
from random import sample
import numpy as np

def make_training_circuits(
    data: np.ndarray,
    size: int ,
    frac_N: float,
)-> np.ndarray:
    """Function that takes the original circuit and creates a training set with a given number of non-cliffords and total size.
    
    Args:
        data: data of the circuit of interst.
        size: number of Clifford projected circuits to return (size of training set).
        frac_N: fraction of non-Clifford gates to leave in each ciruit in the trianing set.
    
    Returns:
        circ_data: list of data arrays for projected circuits.
    """
    circ_data = []
    mask_rz = data[1,:] == 'rz'
    rz_circ_data = data[:, mask_rz]
    mask_not_rz = data[1, :] != 'rz'
    not_rz_circ_data = data[:, mask_not_rz]
    mask_non_cliff = is_clifford_angle(rz_circ_data[2, :]) == False
    rz_non_cliff = rz_circ_data[:, mask_non_cliff]
    mask_cliff = is_clifford_angle(rz_circ_data[2, :]) == True
    rz_cliff = rz_circ_data[:, mask_cliff]
    tot_n_non_cliff = len(rz_non_cliff[0])
    #print(tot_n_non_cliff)
    N = int(frac_N * tot_n_non_cliff)
    all_cliff = np.column_stack((not_rz_circ_data, rz_cliff))
    #print(rz_non_cliff)
    for s in range(size):
        rz_non_cliff_copy = rz_non_cliff.copy()
        random_columns = sample(np.arange(0, tot_n_non_cliff,1).tolist(), tot_n_non_cliff-N)
        rand_rz_non_cliff_proj = rz_non_cliff_copy[:, random_columns]
        #print(rz_non_cliff_copy)
        rz_non_cliff_copy = np.delete(rz_non_cliff_copy, random_columns, axis=1)
        rand_rz_non_cliff_proj[2,:] = closest_clifford(rand_rz_non_cliff_proj[2, :])
        #print(rz_non_cliff_copy)
        new_circ = np.column_stack((all_cliff, rand_rz_non_cliff_proj))
        new_circ = np.column_stack((new_circ, rz_non_cliff_copy))
        i = np.argsort(new_circ[0, :])
        final = new_circ[:, i]
        circ_data.append(final)
    return(circ_data)

def generate_training_circuits(
    circuit: cirq.Circuit,
    num_training_circuits: int,
    fraction_non_clifford: float,
    method_select: str = 'random',
    method_replace: str = 'nearest',
    )-> List[cirq.Circuit]:

    '''Returns a list of near-Clifford circuits to act as training data.

    Args:
        circuit: A circuit of interest.
        num_training_circuits: Number of circuits in the returned training set, assumes already compiled into gate set
                               (Rz, Rx, Z, X, CNOT)
        fraction_non_clifford: The (approximate) fraction of non-Clifford gates in each returned circuit.

    Returns:
        List: [map_to_near_clifford(circuit, fraction_non_clifford, method, can_compile) for _ in range(num_training_circuits)]
                
    '''


