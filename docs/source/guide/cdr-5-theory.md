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

# What is the theory behind CDR?

Clifford Data Regression is a quantum error mitigation technique introduced in {cite}`Czarnik_2021_Quantum` and extended to variable-noise CDR in {cite}`Lowe_2021_PRR`.
This error mitigation strategy is designed for application at the gate level and is relatively straightforward to apply on gate-based quantum computers.
CDR primarily consists of creating a training data set $\{(X_{\phi_i}^{\text{error}}, X_{\phi_i}^{\text{exact}})\}$, where $X_{\phi_i}^{\text{error}}$ and $X_{\phi_i}^{\text{exact}}$ are the expectation values of an observable $X$ for a state $|\phi_i\rangle$ under error and error-free conditions, respectively.

This method includes the following steps:

1. **Choose Near-Clifford Circuits for Training.** Near-Clifford circuits are selected due to their capability to be efficiently simulated classically, and are denoted by $S_\psi=\{|\phi_i\rangle\}_i$.
2. **Construct the Training Set.** The training set $\{(X_{\phi_i}^{\text{error}}, X_{\phi_i}^{\text{exact}})\}_i$ is constructed by calculating the expectation values of $X$ for each state $|\phi_i\rangle$ in $S_\psi$, on both a quantum computer (to obtain $X_{\phi_i}^{\text{error}}$) and a classical computer (to obtain $X_{\phi_i}^{\text{exact}}$).
3. **Learn the Error Mitigation Model.** A model $f(X^{\text{error}}, a)$ for $X^{exact}$ is defined and learned. Here, $a$ is the set of parameters to be determined. This is achieved by minimizing the distance between the training set, as expressed by the following optimization problem: $a_{opt} = \underset{a}{\text{argmin}} \sum_i \left| X_{\phi_i}^{\text{exact}} - f(X_{\phi_i}^{\text{error}},a) \right|^2.$ In this expression, $a_{opt}$ are the parameters that minimize the cost function.
4. **Apply the Error Mitigation Model.** Finally, the learned model $f(X^{\text{error}}, a_{opt})$ is used to correct the expectation values of $X$ for new quantum states, expressed as $X_\psi^{\text{exact}} = f(X_\psi^{\text{error}}, a_{opt})$.

The effectiveness of this method has been demonstrated on circuits with up to 64 qubits and for tasks such as estimating ground-state energies.
However, its performance is dependent on the task, the system, the quality of the training data, and the choice of model.
