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

# What is the theory behind REM?

Readout error mitigation (REM), is one of the most general and earliest studied error mitigation techniques, which can encompass a variety of specific approaches.

A simple version of readout error mitigation is postselection of bitstrings. For
example, if one knows that the measured bitstrings should preserve some
symmetry, bitstrings that do not preserve it can be discarded. Such capability 
is indeed available in {mod}`mitiq.rem.post_select`.

With regards to the more elaborate technique of confusion matrix inversion,
also [supported](rem-1-intro) in Mitiq, some relevant references are Refs. {cite}`Maciejewski_2020,Bravyi_2021,Garion_2021,Geller_2021`. The technique is based on two main ideas:

- Generating a confusion matrix for a specific device;

- Computing the psuedoinverse of this confusion matrix and applying it to the raw measurement (or "readout") results.

## What is a confusion matrix?

A device's readout-error confusion matrix $A$ is a square matrix that encodes, for each pair of measurement basis states $|u\rangle$ and $|v\rangle$, the probability that the device will report $|u\rangle$ as the measurement outcome when the true state being measured was $|v\rangle$. On an ideal, noise-free device, $|u\rangle$ would always equal $|v\rangle$, so the corresponding confusion matrix would have ones on the diagonal and zeros elsewhere. For simplicity of exposition, we will assume throughout that the measurement basis (i.e. the eigenbasis of the observable being measured) is the computational or $Z$ basis. For a two qubit device, the general picture of a confusion matrix to have in mind is:

$$
\begin{bmatrix}
Pr(00|00) & Pr(00|01) & Pr(00|10) & Pr(00|11) \\
Pr(01|00) & Pr(01|01) & Pr(01|10) & Pr(01|11) \\
Pr(10|00) & Pr(10|01) & Pr(10|10) & Pr(10|11) \\
Pr(11|00) & Pr(11|01) & Pr(11|10) & Pr(11|11)
\end{bmatrix}
$$


where $Pr(ij|kl)$ is the probability of observing state $|ij\rangle$ when measuring true state $|kl\rangle$. 

The most straightforward way to empirically estimate a device's full confusion matrix is to go through all the measurement basis states, and for each one $|u\rangle$, repeatedly prepare-then-measure $|u\rangle$ and record the histogram of observed outcomes. This histogram, normalized to give a probability distribution, is an estimate for the $u$th column of the confusion matrix A (i.e. the distribution of measurement outcomes when the true state is $|u\rangle$). Since the number of basis states scales exponentially with the number of qubits $n$, estimating the full confusion matrix in this way requires $O(2^n)$ samples and is therefore only practical for small devices. 

## Computing the pseudoinverse

Note that the estimated confusion matrix $A$ is circuit-independent---it characterizes the readout noise of the device regardless of what circuit is being executed. So in principle (assuming the noise characteristics of the device do not shift over time) $A$ only needs to be estimated once, and its [Moore-Penrose](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse) [pseudoinverse](https://numpy.org/doc/stable/reference/generated/numpy.linalg.pinv.html) $A^{+}$ only needs to be computed once. One can then perform REM for any particular circuit on the device by applying $A^{+}$ to the measurement outcomes from repeated runs of that circuit. In practice, the noise characteristics of devices tend to drift, which necessitates a recalibration effort that results in an updated confusion matrix.

## Applying the inverse confusion matrix to results

With our raw measurement results we convert our bitstrings into a probability vector, $p$, representing our empirical
probability distribution. After obtaining the psuedoinverse matrix $A^{+}$, we can apply it to our empirical probability
distribution to obtain an adjusted *quasi*-probability distribution, $p' = A^{+} p$, which could possibly be 
non-positive. As such, we want to find the closest *positive* probability distribution {cite}`Bravyi_2021` to our
empirical probability distribution:

 $$ p'' = \min_{p_{\rm positive}} \|p' - p_{\rm positive}\|_1$$

Finally, we can draw samples from this new probability distribution, $x \sim p''$ , and return those as our mitigated results.