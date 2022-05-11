---
jupytext:
  text_representation:
    extension: .myst
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# When should I use CDR?

## Advantages

The main advantage of CDR is that it can be applied without knowing the specific details of the noise
model. Indeed, in CDR, the effects of noise are indirectly _learned_ through the execution of an appropriate
set of test circuits. In this way, the final error mitigation inference tends to self-tune with respect
to the used backend.

This self-tuning property is even stronger in the case of _variable-noise-CDR_, i.e., when using the _scale_factors_ option
in {func}`.execute_with_cdr`. In this case, the final error mitigated expectation value is obtained
as a linear combination of noise-scaled expectation values. This is similar to the [ZNE approach](zne-5-theory.myst) but, in CDR, 
the coefficients of the linear combination are learned instead of being fixed by the extrapolation model.


## Disadvantages

The main disadvantage of CDR is that the learning process is performed on a suite of test circuits which
only _resemble_ the original circuit of interest. Indeed, test circuits are _near-Clifford approximations_
of the original one. Only when the approximation is justified, the application of CDR can produce meaningful
results.
Increasing the `fraction_non_clifford` option in {func}`.execute_with_cdr` can alleviate this problem
to some extent. Note that, the larger `fraction_non_clifford` is, the larger the classical computation overhead is.

Another relevant aspect to consider is that, to apply CDR in a scalable way, a valid near-Clifford simulator
is necessary. Note that the computation cost of a valid near-Clifford simulator should scale with the number of non-Clifford
gates, independently from the circuit depth. Only in this case, the learning phase of CDR can be applied efficiently.
