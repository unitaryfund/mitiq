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

Virtual Distillation (VD) is a quantum error mitigation technique introduced and explained in much greater detail in the paper {cite}`Huggins_2021`. This method is designed to suppress noise in NISQ devices by using multiple noisy copies of a quantum state to estimate expectation values with respect to a "virtually distilled" state, $ \frac{\rho^M} {\text{Tr}(\rho^M)}$.

The error-free observable is approximated as $\langle O\rangle_{\text{corrected}} := \frac{\text{Tr}(O \rho^M)}{\text{Tr}(\rho^M)} $. By doing so, VD effectively reduces the influence of non-dominant eigenvectors in the density matrixand the estimator converges exponentially quickly towards the closest pure state to ρ as M is increased. Crucially, VD does not require the explicit preparation of the purified state, making it computationally efficient for near-term quantum devices.

To evaluate the numerator and denominators of the $\langle O\rangle_{\text{corrected}}$, the equality below can be used:

$$\text{Tr}(O \rho^M) = \text{Tr}(O^i S^{(M)} \rho^{\otimes M})$$

$O^i$ represents the observable acting upon one of the M subsystems, and $S^{(M)}$ is the cyclic shift operator performend on M subsystems.

To better explain the exact mechanics of the VD protocol, a specific of example for M=2 is expanded upon below:

## When M = 2 with observable Z

### Obtain Multiple Copies of the Noisy State. 
The method begins by preparing M independent noisy copies of the quantum state ⍴, with M = 2 being a common starting point for balancing resource requirements with error suppression. The copies are assumed to experience similar noise characteristics.

### Diagonalizing Gate
 First a symetrized version of the observable is defined as $O^{(M)} = \frac{1}{M} \sum_{i=0}^{M} O^i$. 
 
 For the example of M=2 and O=Z, this equalities cqan be rewritten as: $Z^{(2)}=\frac{1}{2}(Z^1+Z^2)$. This can be used to rewrite the corrected observable as:
 
 $$\langle O\rangle_{\text{corrected}} = \frac{\text{Tr}(Z^{(2)} S^{(2)} \rho^{\otimes 2})}{\text{Tr}(S^{(2)} \rho^{\otimes 2})}$$

 
 
 Apply the two-qubit gate $B^{(2)}$ between each qubit i in the first copy and the corresponding qubit in the second copy.  


3. **Perform Collective Measurements.**  
  Collective measurements are applied across the M copies to estimate expectation values of an observable X for the virtually distilled state $ \frac{\rho^M}{\text{Tr}(\rho^M)}$. These measurements rely on evaluating quantities such as \( $\text{Tr}(X \rho^M) $\) and \( $\text{Tr}(\rho^M) $\), which involve operators like the cyclic shift operator and symmetrized observables.

4. **Output Error-Mitigated Expectation Values.**  
  The final step involves using the computed values to estimate the expectation value of the observable \( X \) with respect to the virtually distilled state. 
  \[
  $ \langle X \rangle_\text{corrected} \approx \frac{\text{Tr}(X \rho^2)}{\text{Tr}(\rho^2)} $
  \]


VD has been shown to reduce errors by orders of magnitude in numerical simulations for a variety of quantum circuits and noise models. Its simplicity, effectiveness, and compatibility with NISQ devices make it a powerful addition to the suite of error mitigation techniques available today. However, its performance depends on factors such as the quality of the noisy state copies, the observables being measured, and the underlying noise model.
