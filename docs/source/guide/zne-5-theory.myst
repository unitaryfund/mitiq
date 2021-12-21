---
jupytext:
  text_representation:
    extension: .myst
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---


# What is the theory behind ZNE?


Zero noise extrapolation (ZNE) {cite}`Li_2017_PRX, Temme_2017_PRL, Kandala_2019_Nature` is an error
mitigation technique used to extrapolate the noiseless expectation value of an
observable from a range of expectation values computed at different noise levels.
This process works in two steps:

- **Step 1: Intentionally scale noise**. This can be done with different methods.
*Pulse-stretching* {cite}`Temme_2017_PRL` can be used to increase the noise level
of a quantum computation. Similar results can be obtained, at a gate-level, with *unitary folding*
{cite}`Li_2017_PRX, Giurgica_Tiron_2020_arXiv`.

- **Step 2: Extrapolate to the noiseless limit**. This can be done by
fitting a curve (often called *extrapolation model*) to the expectation values measured at different noise levels
to extrapolate the noiseless expectation value.

## Step 1: Intentionally scale noise.

A technique to increase the noise level of a circuit is to intentionally increase its depth.
This can be obtained using the *unitary folding* mapping $G \mapsto G G^\dagger G$. 
This mapping can be applied *globally* or *locally* as shown in the diagrams below.

```{figure} ../img/zne_global_folding.png
---
width: 500
name: figzne_global
---
The diagram demonstrates how gates are inserted in a circuit when *global folding* is applied.
```

```{figure} ../img/zne_local_folding.png
---
width: 500
name: figzne_local
---
The diagram demonstrates how gates are inserted in a circuit when *local folding* is applied.
```

More details on the theory of unitary folding can be found in {cite}`Giurgica_Tiron_2020_arXiv`.
More details on its practical implementation in Mitiq can be found in 
[What additional options are available for ZNE?](zne-3-options.myst).

A noise scaling technique similar to unitary folding is *pulse-stretching*: a method that only applies to 
devices with pulse-level access {cite}`Temme_2017_PRL, Kandala_2019_Nature`.
The noise of the device can be altered by increasing the time over which pulses
are implemented, as shown in the following diagram:

```{figure} ../img/zne_pulse_stretching.png
---
width: 500
name: figzne_pulse
---
The diagram demonstrates how pulse stretching is used to increase noise in a physical device for ZNE.
```

## Step 2: Extrapolate to the noiseless limit

In both gate-model and pulse-model scenarios, let $\tau$ be a parameter quantifying
noise level in the circuit and let $\tau^\prime = \lambda \tau$ the scaled noise level.

For $\lambda = 1$, the input circuit remains unchanged
since the noise level $\tau^\prime = \tau$ is the same as the noise of the physical device without
any scaling.

Let $\rho(\tau')$ be the state prepared by a noise scaled quantum circuit. The expectation
value of an observable $A$ can be described as a function of the noise scaling parameter as follows:

$$
\langle E(\lambda) \rangle = \text{Tr}[\rho(\tau') A] = \text{Tr}[\rho(\lambda \tau) A]
$$

The idea of ZNE is that one can estimate the ideal expectation value $\langle E(\lambda=0) \rangle$, by measuring a range of different
expectation values $\langle E(\lambda) \rangle$ for different values of $\lambda \ge 1$ and extrapolating
to the zero-noise limit.

In practice the extrapolation can be done as follows:
  1) Assume $E(\lambda)\simeq f(\lambda; p_1, p_2, ... p_m)$, where $f$ is an *extrapolation model*, i.e., a function
  of $\lambda$ depending on a set of real parameters $p_1, p_2, \dots, p_m$.
  2) Fit the function $f$ to the measured noise-scaled expectation values, obtaining an optimal set of 
  parameters $\tilde p_1, \tilde p_2, \dots \tilde p_m$.
  3) Evaluate the corresponding zero-noise limit, i.e., $f(0; \tilde p_1, \tilde p_2, \dots \tilde p_m)$.

Different choices of $f$, produce different extrapolations. Typical choices for $f$ are: a linear function, a polynomial, an exponential.
For example, Richardson extrapolation, corresponds to the following polynomial model:


$$
f(\lambda; p_1, p_2, ... p_m) = p_1 + p_2 \lambda + p_3 \lambda^2 + \dots p_m \lambda^{m-1},
$$

where $m$ is equal to the number of data points in the fit (i.e. the number of noise scaled expectation values).
 
More details on how to apply different extrapolation methods with Mitiq can be found in [What additional options
are available in ZNE?](zne-3-options.myst).
