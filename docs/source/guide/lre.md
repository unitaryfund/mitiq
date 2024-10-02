
```{warning}
The user guide for LRE in Mitiq is currently under construction.
```

# Layerwise Richardson Extrapolation

Layerwise Richardson Extrapolation (LRE), an error mitigation technique, introduced in
{cite}`Russo_2024_LRE` works by creating multiple noise-scaled variations of the input
circuit such that the noiseless expectation value is extrapolated from the execution of each
noisy circuit (see the section [What is the theory behind LRE?](lre-5-theory.md)). Compared to
unitary folding, the technique treats the noise in each layer of the input circuit as an
independent variable.
 

You can get started with LRE in Mitiq with the following sections of the user guide:

```{toctree}
---
maxdepth: 1
---
lre-5-theory.md
```