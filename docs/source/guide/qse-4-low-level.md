---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# What happens when I use QSE?

A figure for the typical workflow for QSE is shown below:

```{figure} ../img/qse-data-flow-diagram-v2.png
---
width: 700px
name: qse_workflow_overview
---
Workflow of the QSE technique in Mitiq, detailed in the [What happens when I use QSE?](qse-4-low-level.md) section.
```

The QSE workflow in Mitiq is divided into two steps

1. Generate the code and overlap Hamiltonian
2. Perform a classical minimization problem in order to retrieve the error-mitigated expectation value.

Similar to the workflow of other mitigation techniques on Mitiq, a user will have to provide an input circuit, an executor and an observable for which to compute the error-mitigated expectation value. Furthermore, this technique relies on choosing a basis of expansion operators (the check operators), and a Hamiltonian which defines the state with least amount of errors.

This method will perform the necessary computation in order to project to a state that minimizes the energy of the state with respect to the Hamiltonian. It will then return the error-mitigated expected value by solving the [generalized eigenvalue problem](https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix#Generalized_eigenvalue_problem).
