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

# When should I use PT?

```{admonition} Warning:
Pauli Twirling in Mitiq is still under construction. This users guide will change in the future
after some utility functions are introduced. 
```

## Advantages

Pauli Twirling is a technique devised to tailor noise towards Pauli channels.

More details on the theory of Pauli Twirling are given in the section [What is the theory behind PT?](pt-5-theory.md).

Pauli Twirling is agnostic to our knowledge on the type of noise, easy to implement, and useful to better understand and minimize the benchmarking vs performance gap.



## Disadvantages

Pauli Twirling is generally combined with a compilation pass to maintain circuit depth. Mitiq thus far does not provide this compilation, and so circuit depth is increased by the additional single-qubit gates.

Though the noise is tailored towards a more mitigable channel, it's possible for this channel to be entirely noisy (i.e. a completely depolarizing channel). In this way, it should not be expected to reduce noise on its own.
