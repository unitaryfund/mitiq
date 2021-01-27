from cirq import Circuit
from random import sample, choice, randint
import numpy as np
from mitiq._typing import QPROGRAM

def generate_training_circuits(
    circuit: cirq.Circuit,
    num_training_circuits: int,
    fraction_non_clifford: float,
    method_select: str = 'random',
    method_replace: str = 'nearest',
    sigma_select: float = 0.5,
    sigma_replace: float = 0.5,
)-> List[cirq.Circuit]:

    '''Returns a list of near-Clifford circuits to act as training data.

    Args:
        circuit: A circuit of interest.
        num_training_circuits: Number of circuits in the returned training set, assumes already compiled into gate set
                               (Rz, Rx, Z, X, CNOT)
        fraction_non_clifford: The (approximate) fraction of non-Clifford gates in each returned circuit.
        method_select: option to define the way in which the non-Clifford gates to replace with Cliffords are chosen.
        method_replace: str = option to define the way the chosen non-Clifford gates are replace with a Clifford gate.
        sigma_select: variable definined width of probability distribution used in choosing which non-Cliffords to replace.
        sigma_replace: variable definined width of probability distribution used in choosing which Clifford to replace the non
                       -Clifford with.
    Returns:
        List: [map_to_near_clifford(circuit, fraction_non_clifford, method, can_compile) for _ in range(num_training_circuits)]
    '''
    circuits_list = []
    # First turn circuit into an data array which is easier to deal with, empty circuit is used to store qubit layout:
    data, empty_circuit = circuit_to_array(circuit)
    mask_rz = data[1, :] == 'rz'
    rz_circ_data = data[:, mask_rz]
    mask_not_rz = data[1, :] != 'rz'
    not_rz_circ_data = data[:,mask_not_rz]
    mask_non_cliff = is_clifford_angle(rz_circ_data[2, :]) == False
    rz_non_cliff = rz_circ_data[:, mask_non_cliff]
    mask_cliff = is_clifford_angle(rz_circ_data[2, :]) == True
    rz_cliff = rz_circ_data[:, mask_cliff]
    total_non_cliff = len(rz_non_cliff[0])

    # define N:
    N = int(fraction_non_clifford * total_non_cliff)

    # find all the non-Clifford gates:
    all_cliff = np.column_stack((not_rz_circ_data, rz_cliff))

    for n in range(num_training_circuits):
        rz_non_cliff_copy = rz_non_cliff.copy()

        # Choose non Clifford gates to change according to selection methods:
        if method_select == 'random':
            columns_to_change = sample(np.arange(0, total_non_cliff,1).tolist(), total_non_cliff-N)

        elif method_select == 'probabilistic':
            non_cliff_angles = rz_non_cliff_copy[2]
            # form a probability distribution: 
            probabilities = angle_to_probabilities(non_cliff_angles, sigma_select)
            prob_choose_gate = [k / sum(probabilities) for k in probabilities]
            columns_to_change = np.random.choice(np.arange(0, total_non_cliff,1).tolist(), total_non_cliff - N, replace=False,
                                                p=prob_choose_gate)

        rz_non_cliff_selected = rz_non_cliff_copy[:, columns_to_change]
        # Remove these columns from the circuit data (as they are to be changed below):
        rz_non_cliff_copy = np.delete(rz_non_cliff_copy, columns_to_change, axis=1)
        # Now the non Clifford gates have been selected, we need to decide which Clifford gate to replace them with.
        if method_replace == 'closest':
            rz_non_cliff_selected[2,:] = closest_clifford(rz_non_cliff_selected[2, :])

        elif method_replace == 'random':
            rz_non_cliff_selected[2,:] = random_clifford(rz_non_cliff_selected[2, :])

        elif method_replace == 'probabilistic':
            rz_non_cliff_selected[2,:] = probabilistic_angle_to_clifford(rz_non_cliff_selected[2,:], sigma_replace)

        # Add back into rest of data and re-order instructions:
        new_circ = np.column_stack((all_cliff, rz_non_cliff_selected))
        new_circ = np.column_stack((new_circ, rz_non_cliff_copy))
        i = np.argsort(new_circ[0, :])
        projected_circuit_data = new_circ[:,i]

        # Convert data arry into cirq circuit and append it to the storage array:
        projected_circuit = array_to_circuit(projected_circuit_data, empty_circuit)
        circuits_list.append(projected_circuit_data)

    return(circuits_list)

def count_non_cliffords(
    circuit: QPROGRAM,
) -> float:
    '''Function to check how many non-Clifford gates are in a give circuit.
    
    Args: 
        circuit: some quantum circuit.
        
    Returns: 
        number of non-Clifford gates in the given circuit.
    '''
    data, _ = circuit_to_array(circuit)
    mask_rz = data[1, :] == 'rz'
    rz_circ_data = data[:, mask_rz]
    mask_not_rz = data[1, :] != 'rz'
    not_rz_circ_data = data[:, mask_not_rz]
    mask_non_cliff = is_clifford_angle(rz_circ_data[2, :]) == False
    rz_non_cliff = rz_circ_data[:,mask_non_cliff]
    return(len(rz_non_cliff[0]))

def circuit_to_array(
    circuit: QPROGRAM
)-> (np.ndarray, QPROGRAM):
    '''Function to return the order of gates, their names and paramters in a more managable data structure than a Qiskit
    quantum circuit.
    
    Args:
        circ (QPROGRAM): cirq circuit (decomposed).
        
    Returns:
        data (np.ndarray): np.array([order], [names], [parameters], [qubits])
        where order is the order of the gates from 0 to depth, names are the names of the gates, parameters are the paramters 
        specifying the gates and qubits and cbits are the qubits and classical bits on which they act.
        qubits (int): number of qubits.
        QPROGRAM: empty circuit with same qubit layout as original. 
    '''
    order=[]
    gates_list = []
    qubits_list = []
    operation_list = []
    parameters_list = []
    depth = len(circuit)
    nqubits = len(circuit.all_qubits())
    operations = circuit.all_operations()
    #loop through and construct arrays:
    for i, operation in enumerate(operations):
        operation_list.append(operation)
        order.append(i)
        qubits = operation.qubits
        gate = operation.gate
        if isinstance(gate, cirq.ops.common_gates.CXPowGate)==True:
            qubit = [qubits[0], qubits[1]]
            parameters =None
            name = 'cx'
            gates_list.append(name)
            qubits_list.append(qubit)
            #cbit_list.append(None)
            parameters_list.append(parameters)
        elif isinstance(gate, cirq.ops.common_gates.ZPowGate)==True:
            #print(gate.exponent())
            name = 'rz'
            parameters = float(gate.exponent)*np.pi
            gates_list.append(name)
            qubit = qubits[0]
            qubits_list.append(qubit)
            #cbit_list.append(None)
            parameters_list.append(float(parameters))
        elif isinstance(gate, cirq.ops.common_gates.XPowGate)==True and gate != cirq.X:
            name = 'rx'
            parameters = np.pi/2
            gates_list.append(name)
            qubit = qubits[0]
            qubits_list.append(qubit)
            #cbit_list.append(None)
            parameters_list.append(parameters)
        elif gate == cirq.X:
            parameters = None
            name = 'x'
            gates_list.append(name)
            qubit = qubits[0]
            qubits_list.append(qubit)
            #cbit_list.append(None)
            parameters_list.append(parameters)
        elif isinstance(gate, cirq.ops.MeasurementGate) == True:
            parameters = None
            name='measure'
            gates_list.append(name)
            qubit = qubits[0]
            qubits_list.append(qubit)
            parameters_list.append(parameters)
    
    circuit_empty = circuit.copy()[0:0]
    data = np.array([order, gates_list,  parameters_list, qubits_list, operation_list])
    
    return data, circuit_empty

def array_to_circuit(
    data: np.ndarray,
    empty_circuit: QPROGRAM
) -> QPROGRAM:
    ''' Function that takes the data array containing all the circuit data and turns it into a quantum circuit.
    
    Args:
        data: array containing circuit data np.array([order], [names], [parameters], [qubits], [operations]).
        empty_cricuit: cirq object containing circuit structure. (empty circuit object)
        
    Returns:
        circ: QPROGRAM (cirq quantum circuit)
    
    '''
    name_list= data[1]
    parameters_list = data[2]
    qubits_list = data[3]
    circuit = empty_circuit
    operation_list = data[4]
    #print(circuit)
    #print('new circuit')
    for i in range(len(name_list)):
        name = name_list[i]
        parameter = parameters_list[i]
        qubit = qubits_list[i]
        operation = operation_list[i]

        #print(operation)
        if name =='rz':
            gate = cirq.ops.rz(parameter)
            circuit.append(gate(qubit))
        elif name =='rx':
            gate = cirq.ops.rx(parameter)
            circuit.append(gate(qubit))
        elif name =='cx':
            circuit.append(cirq.ops.CNOT(qubit[0], qubit[1]))
        elif name =='x':
            circuit.append(cirq.X(qubit))
        elif name =='measure':
            circuit.append([operation])
    return(circuit)

def is_clifford_angle(
    ang: float,
    tol: float = 10 ** -5,
)-> bool:
    CLIFFORD_ANGLES = (np.pi / 2, np.pi, np.pi * 3 / 2, 2 * np.pi)
    '''Function to check if a given angle is Clifford.
    
    Args:
        ang: rotation angle in the Rz gate.
        
    Returns:
        bool: True / False for Clifford or not. 
    '''
    diff_list = []
    for cliff_ang in CLIFFORD_ANGLES:
        diff_list.append(abs(abs(ang) - cliff_ang) % (2 * np.pi))
    if any(np.array(diff_list) <= tol):
        return True
    else:
        return False
    
# Vectorize function so it can take arrays of angles as its input.
is_clifford_angle = np.vectorize(is_clifford_angle)

def closest_clifford(
    ang: float
)-> float:
    '''Function to take angle and return the nearest Clifford angle note the usage of this function is vectorized
    so it takes and returns arrays.
    
    Args:
        ang: angle in Rz gate.
        
    Returns:
        Clifford angle: closest clifford angle.
    '''
    cliff_angs = [np.pi/2, np.pi, np.pi*3/2, 2*np.pi]
    diff_list=[]
    for cliff_ang in cliff_angs:
        diff_list.append(abs((ang)%(2*np.pi)-cliff_ang))
    min_1 = min(diff_list)
    # need to check that there are not two min values in list
    diff_list_copy = diff_list.copy()
    diff_list_copy.remove(min_1)
    min_2 = min(diff_list_copy)
    # if just one min value, return the corresponding nearest cliff.
    if abs(min_1-min_2) > 10**(-8):
        return (cliff_angs[diff_list.index(min(diff_list))])
    # if two min values (ie two cliff gates equidistant) randomly choose the cliff gate to return.
    else:
        index_list = [diff_list.index(min_1), diff_list_copy.index(min_2)]
        index = choice(index_list)
        return(cliff_angs[index])

# vectorize so function can take array of angles.
closest_clifford = np.vectorize(closest_clifford)

def random_clifford(
    ang: float
)-> float:
    '''Function to take angle and return the random Clifford angle note the usage of this function is vectorized
    so it takes and returns arrays.
    
    Args:
        ang: angle in Rz gate.
        
    Returns:
        Clifford angle: closest clifford angle.
    '''
    cliff_angs = [np.pi/2, np.pi, np.pi*3/2, 2*np.pi]
    random_index = randint(0,3)
    return(cliff_angs[random_index])

# vectorize so function can take array:
random_clifford = np.vectorize(random_clifford)

def angle_to_probabilities(
    angle: float,
    sigma: float
)-> float:
    """Function to return probability disribtuion based on distance from angles to Clifford gates.
    
    Args:
        angle: angle to form probability distribution. 
    
    Returns:
        discrete value of probability distribution calucalted from Prob_project = exp(-(dist/sigma)^2) where dist = sum(dists)
        is the sum of distances from each Clifford gate.
    """
    S = np.array([[1, 0.0], [0.0, 1j]])
    Rz = np.array([[1, 0.0], [0.0, np.exp(angle*1j)]])
    dists = []
    for i in range(4):
        i += 1
        diff = np.linalg.norm(Rz - S ** (i))
        dists.append(np.exp(-(diff / sigma) ** 2))
    return(sum(dists))

# vectorize so function can take array of angles.
angle_to_probabilities = np.vectorize(angle_to_probabilities)

def probabilistic_angle_to_clifford(
    ang: float,
    sigma: float,
)-> float:
    '''Function to take angle and return the Clifford angle according to the probability distirbution:
                        
                        prob = exp(-(dist/sigma)^2)
                        
    where dist is the frobenius norm from the 4 clifford angles and the gate of interest. Note the usage of this function
    is vectorized so it takes and returns arrays.
    
    Args:
        ang: angle in Rz gate.
        sigma: width of probability distribution.
        
    Returns:
        Clifford angle: clifford angle to replace gate angle, calculated probabilistically.
    '''
    CLIFFORD_ANGLES = (np.pi / 2, np.pi, np.pi * 3 / 2, 2 * np.pi)
    S = np.array([[1, 0.0],[0.0, 1j]])
    Rz = np.array([[1, 0.0],[0.0, np.exp(ang*1j)]])
    dists=[]
    for i in range(4):
        i += 1
        diff = np.linalg.norm(Rz - S** (i))
        dists.append(np.exp(-(diff/sigma) ** 2))
    prob_gate = [i/sum(dists) for i in dists]
    cliff_ang = np.random.choice(CLIFFORD_ANGLES, 1, replace=False, p=prob_gate)
    return(cliff_ang)

# vectorize so function can take array of angles.
probabilistic_angle_to_clifford = np.vectorize(probabilistic_angle_to_clifford)


