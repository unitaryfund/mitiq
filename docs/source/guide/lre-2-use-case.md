---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# When should I use LRE?

## Advantages

Just as in ZNE, LRE can also be applied without a detailed knowledge of the underlying noise model as the effectiveness of the technique depends on the choice of scale factors.
Thus, LRE is useful in scenarios where tomography is impractical.

The sampling overhead is flexible wherein the cost can be reduced by using larger values for the fold multiplier (used to
create the noise-scaled circuits) or by chunking a larger circuit to fold groups of layers of circuits instead of each one individually.

## Disadvantages

When using a large circuit, the number of noise scaled circuits grows polynomially such that the execution time rises because we require the sample matrix to be a square matrix (more details in the [theory](lre-5-theory.md) section).

When reducing the sampling cost by using a larger fold multiplier, the bias for polynomial extrapolation increases as one moves farther away from the zero-noise limit.

Chunking a large circuit with a lower number of chunks to reduce the sampling cost can reduce the performance of LRE.
In ZNE parlance, this is equivalent to local folding faring better than global folding in LRE when we use a higher number of chunks in LRE.

```{attention}
We are currently investigating the issue related to the performance of chunking large circuits.
```
