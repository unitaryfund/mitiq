import sys
sys.path.insert(0, '/home/chris/Desktop/cpython/mitiq')

from mitiq.vd.vd_utils import _copy_circuit_parallel, _apply_diagonalizing_gate, _apply_cyclic_system_permutation, _apply_symmetric_observable
import cirq
import numpy as np
from mitiq import QPROGRAM, Executor, Observable, QuantumResult, MeasurementResult
from mitiq.executor.executor import DensityMatrixLike, MeasurementResultLike
from typing import Callable, Optional, Union, Sequence, List


def execute_with_vd(
        circuit: QPROGRAM, 
        executor: Union[Executor, Callable[[QPROGRAM], QuantumResult]], 
        M: int=2, 
        K: int=100, 
        observable: Optional[Observable] = None,
    ) -> list[float]:
    '''
    Given a circuit rho that acts on N qubits, this function returns the expectation values of a given observable for each qubit i. 
    The expectation values are corrected using the virtual distillation algorithm.

    Args:
        circuit: The input circuit of N qubits to execute with VD.
        executor: A Mitiq executor that executes a circuit and returns the
                    unmitigated ``QuantumResult`` (e.g. an expectation value).
                    The executor must either return a single measurement (bitstring or list),
                    a list of measurements
        M: The number of copies of rho. Only M=2 is implemented at this moment.
        K: The number of iterations of the algorithm. Only used if the executor returns a single measurement.
        observable: The one qubit observable for which the expectation values are computed. 
                    The default observable is the Pauli Z matrix.
                    At the moment using different observables is not supported.

    Returns:
        A list of expectation values for each qubit i in the circuit. Estimated with VD.
    '''

    # input rho is an N qubit circuit
    N = len(circuit.all_qubits())
    rho = _copy_circuit_parallel(circuit, M)

    Ei = np.array(list(0 for _ in range(N)))
    D = 0
    
    # Forcing odd K, this is a workaround so that D (see end of the function) cannot be 0 accidentally
    if K%2 == 0:
        K -= 1

    if not isinstance(executor, Executor):
        executor = Executor(executor)
    
    if executor._executor_return_type in DensityMatrixLike:
        # do density matrix treatment
        rho_tensorM = executor.run(rho)
       
        # two_system_swap = create_S2_N_matrix(N)
        rho_tensorM_swapped = _apply_cyclic_system_permutation(rho_tensorM, N)

        rho_tensorM_swapped_observabled = _apply_symmetric_observable(rho_tensorM_swapped, N, observable)

        Z_i_corrected = np.trace(rho_tensorM_swapped_observabled, axis1=1, axis2=2) / np.trace(rho_tensorM_swapped, axis1=1, axis2=2)
        
    elif executor._executor_return_type in MeasurementResultLike:

        rho = _apply_diagonalizing_gate(rho, M)
    

        #  3) apply measurements
        # The measurements are only added when the executor returns measurement values
        # the measurement keys are applied in accordance with the SWAPS that are applied in the pseudo code in the paper.
        # The SWAP operations are omitted here since they are hardware specific.
        for i in range(M*N):
            rho.append(cirq.measure(cirq.LineQubit(i), key=f"{i}"))

        # this comment we should keep in!
        # if executor._executor_return_type ==  MeasurementResult: # !!!!!!!!!!!!! aaaaaaaaaasagagahgahghaaagaagahaahahgahggggggghghgghghggh
        
        res = executor.run(rho, force_run_all=True, reps=K) # TODO make this reps a **kwargs to allow any executor

        self_packed = True
        if isinstance(res, str):
            res = [res]
            self_packed = False

        elif isinstance(res[0], int):
            res = [res]
            self_packed = False

        if len(res) == 1: # if the executor only returns a single measurement
            for _ in range(K-1): # then we measure K times in total
                if not self_packed:                    
                    res.append( executor.run(rho, force_run_all=True))
                else:
                    res.append( executor.run(rho, force_run_all=True)[0] )

        # post processing measurements
        for bitStr in res:
            
            # This maps 0/1 measurements to 1/-1 measurements, the eigenvalues of the Z observable
            Z_base_mesurement = 1 - 2* np.array(list(bitStr))

            # Separate the two systems
            z1 = Z_base_mesurement[:N]
            z2 = Z_base_mesurement[N:]

            # Implementing the sum and product from the paper
            # Note that integer division prevents floating point errors here, 
            # since each factor in the product or the Ei sum will be either +1 or -1.
            # only in case of pauli Z
            product_term = 1
            for j in range(N):
                product_term *= ( 1 + z1[j] - z2[j] + z1[j]*z2[j] )//2

            D += product_term 

            for i in range(N):
                Ei[i] += (z1[i] + z2[i])//2 * product_term // (( 1 + z1[i] - z2[i] + z1[i]*z2[i] )//2) # undo the j=i term in the product


        # Elementwise division by D, since we are working with numpy arrays
        Z_i_corrected = Ei / D
    

    else:
        raise ValueError("Executor must have a return type of DensityMatrixLike or MeasurementResultLike")

    if not np.allclose(Z_i_corrected.real, Z_i_corrected, atol=1e-6):
        print("Warning: The expectation value contains a significant imaginary part. This should never happen.")
        return Z_i_corrected
    else:
        return Z_i_corrected.real
