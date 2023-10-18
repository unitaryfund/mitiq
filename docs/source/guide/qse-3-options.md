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


# What additional options are available in QSE?
In addition to the necessary ingredients already discussed in [How do I use QSE?](qse-1-intro.md), there are a few additional options included in the implementation. 

## Caching Pauli Strings to Expectation Values
```{warning}
The cache object is modified in place.
```

Specifically, in order to save runtime, the QSE implementation supports the use of a cache that maps pauli strings to their expectation values. This is taken as an additional parameter in the [`execute_with_qse`](https://mitiq.readthedocs.io/en/stable/apidoc.html#mitiq.qse.qse.execute_with_qse) function.

The inclusion of the cache significantly speeds up the runtime and avoids the need for re-computation of already computed values. 
Furthermore, it is important to note that the cache gets modified in place, so the user can pass the same cache object to `execute_with_qse` and avoid regenerating the cache unnecessarily , e.g. if the noise model is the same from one execution to another.

## Requirements for Check Operators

It is also important to note that when specifying the check (or excitation) operators for the execution, it is not necessary to specify the full exponential number of operators. As many or as few operators can be specified. The tradeoff is the fidelity of the projected state.  
