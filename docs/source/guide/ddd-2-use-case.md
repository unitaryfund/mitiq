---
jupytext:
  text_representation:
    extension: .myst
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# When should I use DDD?

## Advantages

Dynamical decoupling is a technique that has been originally devised to shield a quantum system from quantum noise coming from an environment, using open-loop control techniques {cite}`Viola_1999_PRL`. The crucial idea behind dynamical decoupling is to intervene with control pulses on timescales that are faster than those of the system-environment interaction.

It is thus particularly effective against structured baths, colored noise or any type of noise with some level of memory effects. More details on the theory of digital dynamical decoupling are given in the section [What is the theory behind DDD?](ddd-5-theory.myst).

Dynamical decoupling is also effective for single-shot quantum computing algorithms, i.e., it finds application beyond algorithms that just require expectation values, like variational quantum algorithms.




## Disadvantages

Dynamical decoupling is generally applied at the pulse level. Mitiq provides it at the gate-level. For this reason, it may be difficult to know and control what decoupling sequences are actually run on the quantum processor. A way to partially alleviate this issue, is to use DDD sequences built from the native gate set of the quantum backend.

Another limitation of dynamical decoupling is that it cannot improve results against completely symmetric noise effects (symmetrical with respect to gates applied in the decoupling sequence). In particular, digital dynamical decoupling is ineffective against depolarizing Markovian noise.
