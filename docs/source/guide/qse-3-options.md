---
jupytext:
  text_representation:
    extension: .myst
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---


# What additional options are available in QSE?
In addition to the necessary ingredients already discussed in [How do I use QSE?](qse-1-intro.md), there are a few additional options included in the implementation. 

## Caching Pauli Strings to Expectation Values

Specifically, in order to save runtime, the QSE implementation supports the use of a cache that maps pauli strings to their expectation values. This is taken as an additional parameter in the “execute_with_qse” function reproduced below. 

```
def execute_with_qse(
    circuit: QPROGRAM,
    executor: Union[Executor, Callable[[QPROGRAM], QuantumResult]],
    check_operators: Sequence[PauliString],
    code_hamiltonian: Observable,
    observable: Observable,
    pauli_string_to_expectation_cache: Dict[PauliString, complex] = {},
) -> float:
    """Function for the calculation of an observable from some circuit of
        interest to be mitigated with Quantum Subspace Expansion
    Args:
        circuit: Quantum program to execute with error mitigation.
        executor: Executes a circuit and returns a `QuantumResult`.
        check_operators: List of check operators that define the
        stabilizer code space.
        code_hamiltonian: Hamiltonian of the code space.
        observable: Observable to compute the mitigated expectation value of.
        pauli_string_to_expectation_cache: Cache for expectation values of
        Pauli strings used to compute the projector and the observable.
    Returns:
        The expectation value estimated with QSE.
```
The inclusion of the cache significantly speeds up the runtime and avoids the need for recomputation of already computed values. Furthermore, it is important to note that the cache gets modified in place. Specifically, the cache be reused between different executions of the program. 

## Requirements for Check Operators

It is also important to note that when specifying the check (or excitation) operators for the execution, it is not necessary to specify the full exponential number of operators. As many or as few operators can be specified. The tradeoff is the fidelity of the projected state.  
