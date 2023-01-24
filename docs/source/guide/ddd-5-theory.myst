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

# What is the theory behind DDD?

Dynamical decoupling (DD) {cite}`Viola_1998_PRA, Viola_1999_PRL, Zhang_2014_PRL`
is a quantum control technique to effectively reduce the interaction of a quantum system with its environment.
The protocol works by driving a quantum system with rapid sequences of periodic control pulses.

The application of DD sequences can have two effects depending on the correlation time {cite}`Breuer_2007_Oxford` of the environment:

1. For Markovian noise, DD can make the overall quantum channel more symmetric (analogous to quantum twirling {cite}`Wallman_2016_PRA`)
but cannot actually decouple the system from the environment;

2. For non-Markovian noise, DD can effectively decouple the system from the environment.
In theory, ideal sequences of infinitely quick and strong pulses, can result in complete noise suppression.

In practice, due to the finite frequency and finite amplitude of DD sequences,
both effects are possible but only as imperfect approximations.

In the context of quantum computing, DD can be considered as an error mitigation method.
With respect to other error mitigation techniques, DD has very peculiar features:

- It maps a noisy quantum computation to a _single_ error-mitigated computation (no need to take linear combinations
of noisy results as in [ZNE](zne-5-theory.myst), [PEC](pec-5-theory.myst) and [CDR](cdr-5-theory.myst)).

- As a consequence of the previous point, there is not a fundamental error mitigation overhead or
increase in statistical uncertainty in the final result.

- If noise is time-correlated, it can suppress real errors at the physical level instead of applying a virtual noise
reduction via classical post-processing.




## Digital dynamical decoupling

In a quantum computing device based on the circuit model, sequences of DD pulses can be mapped to sequences
of discrete quantum gates (typically Pauli gates). We refer to this gate-level formulation as _digital dynamical decoupling_ (DDD)
to distinguish it from the standard pulse-level formulation.



```{note}
This type of gate-level approach is very similar to the gate-level abstraction used in Mitiq to implement
_digital zero-noise extrapolation_ via _unitary folding_ (see [What is the theory behind ZNE?](zne-5-theory.myst)).
```
Experimental evidence showing the practical utility gate-level decoupling sequences is given in several publications {cite}`Pokharel_2018_PRL, Jurcevic_2021_arxiv, GoogleQuantum_2021_nature, Smith_2021_arxiv, Das_2021_ACM`.


```{warning}
Gate-level DDD can only be considered as an approximation of the ideal (pulse-level) DD technique. Moreover, quantum backends 
may internally optimize and schedule gates in unpredictable ways such that, in practice, DDD sequences may not be physically applied
as expected.
```

A significant advantage of DDD with respect to pulse-level DD is the possibility of defining it in a backend-independent way, 
via simple transformations of abstract quantum circuits. For this reason, DDD is particularly suitable for a multi-platform library like Mitiq.



## Common examples of DDD sequences

Common dynamical decoupling sequences are arrays of (evenly spaced) Pauli gates. In particular:
- The _XX_ sequence is typically appropriate for mitigating (time-correlated) dephasing noise;
- The _YY_ sequence is typically appropriate for mitigating (time-correlated) amplitude damping noise;
- The _XYXY_ sequence is typically appropriate for mitigating generic single-qubit noise.

```{note}
A general property of DDD sequences is that, if executed on a noiseless backend, they are equivalent to the identity operation.
```

All the above examples of DDD sequences are supported in Mitiq and more general ones can be defined and customized by users.
For more details on how to define DDD sequences in Mitiq see [What additional options are available for DDD?](ddd-3-options.myst).

In a practical scenario it is hard to characterize the noise model and the noise spectrum of a quantum device and the
choice of the optimal sequence is not obvious _a priori_. A possible strategy is to run a few circuits whose noiseless
results are theoretically known, such that one can empirically determine what sequence is optimal for a specific backend.

It may happen that, for some sequences, the final error of the quantum computation is actually increased.
As with all other error-mitigation techniques, one should always take into account that an improvement of performances is not guaranteed.

