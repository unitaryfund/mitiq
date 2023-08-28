# What happens when I use QSE?
A figure for the typical workflow for QSE is shown below:
```{figure} ../img/qse-data-flow-diagram.png
---
width: 700px
name: qse-workflow-overview-2
---
Workflow of the QSE technique in Mitiq, detailed in the [What happens when I use QSE?](qse-4-low-level.md) section.
```

The QSE workflow in Mitiq is divided into two steps: generating the code hamiltonian and the overlap hamiltonian, and then performing a classical minimization problem in order to retrieve the error-mitigated expectation value. 

Similar to the workflow of other mitigation techniques on Mitiq, a user will have to provide an input circuit, an executor and an observable for which to compute the error-mitigated expectation value. Furthermore, this technique relies on choosing a basis of expansion operators (the check operators), and a ‘Hamiltonian’ which defines the state with least amount of errors.

This new method will perform the necessary computation in order to project to a state that minimizes the energy of the state with respect to the hamiltonian. It will then return the error-mitigated expected value by solving the generalized eigenvalue problem. 
