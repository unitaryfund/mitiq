---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# What is the theory behind VD?

Virtual Distillation (VD) is a quantum error mitigation technique introduced in the paper {cite}`Huggins_2021`. This method is designed to suppress noise in NISQ devices by using multiple noisy copies of a quantum state to estimate expectation values with respect to a "virtually distilled" state, \( $\rho^M / \text{Tr}(\rho^M)$ \). By doing so, VD effectively reduces the influence of non-dominant eigenvectors in the density matrix. The resulting estimator converges exponentially quickly towards the closest pure state to ρ as M is increased. Crucially, VD does not require the explicit preparation of the purified state, making it computationally efficient for near-term quantum devices.

The VD protocol consists of the following steps:

1. **Obtain Multiple Copies of the Noisy State.**  
   The method begins by preparing M independent noisy copies of the quantum state ⍴, with M = 2 being a common starting point for balancing resource requirements with error suppression. The copies are assumed to experience similar noise characteristics.

2. **Perform Collective Measurements.**  
   Collective measurements are applied across the \( M \) copies to estimate expectation values of an observable \( X \) for the virtually distilled state \( $\rho^M / \text{Tr}(\rho^M)$ \). These measurements rely on evaluating quantities such as \( $\text{Tr}(X \rho^M) $\) and \( $\text{Tr}(\rho^M) $\), which involve operators like the cyclic shift operator and symmetrized observables.

3. **Suppress Noise Exponentially.**  
   The measurement outcomes from the collective operations are used to suppress noise. The method exponentially reduces the contributions of non-dominant eigenvalues of \( $\rho$ \) by a factor proportional to \( M \). This brings the virtually distilled state closer to the dominant eigenvector, which often corresponds to the ideal noise-free state.

4. **Output Error-Mitigated Expectation Values.**  
   The final step involves using the computed values to estimate the expectation value of the observable \( X \) with respect to the virtually distilled state. 
   \[$
   \langle X \rangle_\text{corrected} \approx \frac{\text{Tr}(X \rho^2)}{\text{Tr}(\rho^2)}$
   \]


VD has been shown to reduce errors by orders of magnitude in numerical simulations for a variety of quantum circuits and noise models. Its simplicity, effectiveness, and compatibility with NISQ devices make it a powerful addition to the suite of error mitigation techniques available today. However, its performance depends on factors such as the quality of the noisy state copies, the observables being measured, and the underlying noise model.
