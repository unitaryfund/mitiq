# Variational Quantum Eigensolver improved with Mitiq
In this example we investigate how Mitiq can help reduce errors 
and improve convergence of a simple VQE problem executed 
on a simulated noisy backend. 


The PyQuil VQE example shown here is adapted from 
https://grove-docs.readthedocs.io/en/latest/vqe.html#
and the VQE function in Grove.

```{code-cell} 
from pyquil import get_qc, Program
from pyquil.gates import RX, RY, MEASURE, RESET
from typing import List, Union
from collections import Counter
import numpy as np
import mitiq
from mitiq import zne, execute_with_zne
from pyquil.paulis import PauliTerm, PauliSum
```
+++

Use the get_qc command to initialize the simulated noisy device where the PyQuil program will run

```{code-cell} 
# initialize quantum device
qpu = get_qc("2q-noisy-qvm")

# set up quantum circuit
def ansatz(params):
    return Program(RX(params[0], 0), RX(params[1], 0))
```

+++
 
 Compute expectation value of the Hamiltonian over the over the distribution generated from the quantum program
```{code-cell} 
def expectation(qc, samples: int, 
                pauli_sum: Union[PauliSum, PauliTerm, np.ndarray], 
                pyquil_prog: Program) -> float:
    """
    Compute the expectation value of pauli_sum over the distribution generated from pyquil_prog.

    :param pyquil_prog: The state preparation Program to calculate the expectation value of.
    :param pauli_sum: PauliSum representing the operator of which to calculate the expectation
            value
    :param samples: The number of samples used to calculate the expectation value. 
    :param qc: The QuantumComputer object.
    :return: A float representing the expectation value of pauli_sum given the distribution
            generated from quil_prog.
        """ 
    expectation = 0.0
    pauli_sum = PauliSum([pauli_sum])
    for j, term in enumerate(pauli_sum.terms):
        meas_basis_change = Program()
        qubits_to_measure = []
        for index, gate in term:
            qubits_to_measure.append(index)
            if gate == 'X':
               meas_basis_change.inst(RY(-np.pi / 2, index))
            elif gate == 'Y':
               meas_basis_change.inst(RX(np.pi / 2, index))
            meas_outcome = expectation_from_sampling(
                                program + meas_basis_change,
                                qubits_to_measure, qc, samples)
            expectation += term.coefficient * meas_outcome
    return expectation.real
```

+++

Calculates the parity of elements at indexes in marked_qubits
   
```{code-cell} 
def parity_even_p(state, marked_qubits):
"""
    Parity is relative to the binary representation of the integer state.

    :param state: The wavefunction index that corresponds to this state.
    :param marked_qubits: The indexes to be considered in the parity sum.
    :returns: A boolean corresponding to the parity.
    """
    mask = 0
    for q in marked_qubits:
        mask |= 1 << q
    return bin(mask & state).count("1") % 2 == 0
```

+++

calculate the expectation value of the Zi operator at marked_qubits
```{code-cell} 
def expectation_from_sampling(pyquil_program: Program,
                              marked_qubits: List[int], qc,
                              samples: int) -> float:
    """

    Given a wavefunctions, this calculates the expectation value of the Zi
    operator where i ranges over all the qubits given in marked_qubits.

    :param pyquil_program: pyQuil program generating some state
    :param marked_qubits: The qubits within the support of the Z pauli
                          operator whose expectation value is being calculated
    :param qc: A QuantumComputer object.
    :param samples: Number of bitstrings collected to calculate expectation
                    from sampling.
    :returns: The expectation value as a float.
    """
    program = Program()
    ro = program.declare('ro', 'BIT', max(marked_qubits) + 1)
    program += pyquil_program
    program += [MEASURE(qubit, r) 
                for qubit, r in zip(list(range(max(marked_qubits) + 1)), ro)]
    program.wrap_in_numshots_loop(samples)
    executable = qc.compile(program)
    bitstring_samples = qc.run(executable)
    bitstring_tuples = list(map(tuple, bitstring_samples))

    freq = Counter(bitstring_tuples)

    # perform weighted average
    expectation = 0
    for bitstring, count in freq.items():
        bitstring_int = int("".join([str(x) for x in bitstring[::-1]]), 2)
        if parity_even_p(bitstring_int, marked_qubits):
            expectation += float(count) / samples
        else:
            expectation -= float(count) / samples
    return expectation
```
+++

Hamiltonian in this example is just sigma_z on the zeroth qubit

```{code-cell} 
from pyquil.paulis import sZ
hamiltonian = sZ(0)
```
+++

define objective function mapping the set of experimental parameters to the expectation value
```{code-cell} 
def objective(params, program):
   
    program += ansatz(params)
    samples = 10000 
    return expectation(qpu, samples, hamiltonian, program)
```

+++

run VQE routine without noise mitigation
```{code-cell} 
from scipy.optimize import minimize
```

+++

define initial experimental parameters
```{code-cell} 
initial_angle = [0.000, 0.000]
program = Program()
```

+++

calculate expectation value from initial parameters
```{code-cell} 
print(objective(initial_angle, program))
```

+++

use the Nelder-Mead method in scipy minimizer to find the minimimum of the expectation value 
```{code-cell}
minimum = minimize(
          objective, initial_angle, args = (program), 
          method = 'Nelder-Mead',  options = {'disp': True, 'maxiter': 100,'xatol': 1.0e-2})
```

+++

re-run VQE, with noise mitigation technique Zero Noise Extrapolation
```{code-cell}
# add noise scaling 
fac = zne.inference.LinearFactory(scale_factors = [1.0, 7.0]) 

# re-initialize program 
program = Program()
```

+++

update objective function to calculate expecation value w/ noise mitigation
```{code-cell} 
from functools import partial
def objective_zne(params, program):
    program += ansatz(params)    
    samples = 10000
    return execute_with_zne(program, 
                            partial(expectation, qpu, samples, hamiltonian), 
                            fac)
    
print(objective_zne(initial_angle, program))
min_zne = minimize(objective_zne, initial_angle, args = (program), 
                   method = 'Nelder-Mead', 
                   options = {'disp': True, 'maxiter': 100, 'xatol': 1.0e-2})
```

+++

## References
[1] [VQE tutorial in PyQuil.] 
https://grove-docs.readthedocs.io/en/latest/vqe.html#

[2] Rigetti Computing (2018) Grove (Version 1.7.0) [Source code].https://github.com/rigetti/grove


+++

display information about Mitiq, packages, and Python version/platform
```{code-cell} 
mitiq.about()
```
