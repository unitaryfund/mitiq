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

# When should I use PEC?

## Advantages

The main advantage of PEC is that, under the assumption of perfect gate tomography,
it provides an unbiased estimation of expectation values.
This means that, in the limit of many samples, the error mitigated expectation values
converge to the ideal expectation values.


## Disadvantages

PEC is a noise-aware technique that converges to the ideal noiseless results only if we have the full
tomographic knowledge of the hardware gates. Indeed, in order to represent
ideal gates as linear combination of noisy gates, one typically needs to know the super-operator
matrix associated to each noisy gate.

Another practical problem of PEC is that it involves the execution of many different circuits.
This typically requires more clock time compared to the repeated execution of equal circuits.
Batched execution of circuits, if supported by the hardware provider, can alleviate this problem to some extent.

```{code-cell} ipython3

```
