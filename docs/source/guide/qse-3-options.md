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

Specifically, in order to save runtime, the QSE implementation supports the use of a cache that maps pauli strings to their expectation values. This is taken as an additional parameter in the [`execute_with_qse`](https://mitiq.readthedocs.io/en/stable/apidoc.html#mitiq.qse.qse.execute_with_qse) function.

```{warning}
The cache object is modified in place when passing it to `execute_with_qse`.
```

The inclusion of the cache significantly speeds up the runtime and avoids the need for re-computation of already computed values.
Furthermore, since the cache is modified in place, it can be reused as long as the noise model remains the same.

## Requirements for Check Operators

When specifying the check operators, it is **not** necessary to specify the full exponential number of operators.
As many or as few operators can be specified.
The tradeoff is the fidelity of the projected state.
