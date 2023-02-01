---
jupytext:
  text_representation:
    extension: .myst
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# What additional options are available when using REM?

## Overview
The main options the user has when using REM concern how to specify the inverse
confusion matrix that is used to mitigate errors in the raw measurement (or "readout") results. Currently Mitiq does not implement methods for estimating readout-error confusion matrices (which is a form of measurement noise calibration and therefore a device specific task), so the user must provide enough information to allow Mitiq to construct one. As described below, Mitiq's options support the differing levels of information a user may have about the readout-error characteristics of their device. After the confusion matrix has been constructed, the remaining steps of standard REM are straightforward (compute the pseudoinverse of the confusion matrix and then apply this to the raw measurement results). 



## What is a Confusion Matrix?

A device's readout-error confusion matrix $A$ is a square matrix that encodes, for each pair of measurement basis states $|u\rangle$ and $|v\rangle$, the probability that the device will report $|u\rangle$ as the measurement outcome when the true state being measured was $|v\rangle$. On an ideal, noise-free device, $|u\rangle$ would always equal $|v\rangle$, so the corresponding confusion matrix would have ones on the diagonal and zeros elsewhere. For simplicity of exposition, we will assume throughout that the measurement basis (i.e. the eigenbasis of the observable being measured) is the computational or $Z$ basis. For a two qubit device, the general picture of a confusion matrix to have in mind is:

$$
\begin{bmatrix}
Pr(00|00) & Pr(01|00) & Pr(10|00) & Pr(11|00) \\
Pr(00|01) & Pr(01|01) & Pr(10|01) & Pr(11|01) \\
Pr(00|10) & Pr(01|10) & Pr(10|10) & Pr(11|10) \\
Pr(00|11) & Pr(01|11) & Pr(10|11) & Pr(11|11)
\end{bmatrix}
$$


where $Pr(ij|kl)$ is the probability of observing state $|ij\rangle$ when measuring true state $|kl\rangle$. 

The most straightforward way to empirically estimate a device's full confusion matrix is to go through all the measurement basis states, and for each one $|u\rangle$, repeatedly prepare-then-measure $|u\rangle$ and record the histogram of observed outcomes. This histogram, normalized to give a probability distribution, is an estimate for the $u$th column of the confusion matrix A (i.e. the distribution of measurement outcomes when the true state is $|u\rangle$). Since the number of basis states scales exponentially with the number of qubits $n$, estimating the full confusion matrix in this way requires $O(2^n)$ samples and is therefore only practical for small devices. 

Note that the estimated confusion matrix $A$ is circuit-independent---it characterizes the readout noise of the device regardless of what circuit is being executed. So in principle (assuming the noise characteristics of the device do not shift over time) $A$ only needs to be estimated once, and its [Moore-Penrose](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse) [pseudoinverse](https://numpy.org/doc/stable/reference/generated/numpy.linalg.pinv.html) $A^{+}$ only needs to be computed once. One can then perform REM for any particular circuit on the device by applying $A^{+}$ to the measurement outcomes from repeated runs of that circuit. For more details, see [What is the theory behind REM?](rem-5-theory.md).


## Options for specifying the inverse confusion matrix  

Mitiq provides two utility functions for constructing an inverse confusion matrix from user-provided information about a device's confusion matrix. We describe these functions below and illustrate with toy examples. (Note that everything that follows is circuit-agnostic; it concerns how to represent a device's noise model in the form required by REM). 

### Inverse confusion matrix from single-qubit noise model 
The function `generate_inverse_confusion_matrix(num_qubits, p0, p1)` embodies the simplest noise model possible, where one assumes that noise affects the measurement of each qubit independently and with the same confusion probabilities, specified by $p_0 = Pr(1|0)$, the probability $|0\rangle$ gets flipped to $|1\rangle$ when measured, and $p_1 = Pr(0|1)$, the probability $|1\rangle$ gets flipped to $|0\rangle$. The $2 \times 2$ confusion matrix $A_1$ for the $1$st qubit (and every other qubit) is then 

$$
\begin{bmatrix}
1-p_0 & p_1 \\
p_0 & 1-p_1
\end{bmatrix}
$$

and the joint $2^n \times 2^n$ confusion matrix $A$ for all $n$ qubits is just $n$ copies of $A_1$ tensored together: $A = A_1 \otimes  \dots \otimes A_1 = A_1^{\otimes n}$. 


To construct an inverse confusion matrix with `generate_inverse_confusion_matrix()` the user supplies the number of qubits and the single-qubit confusion matrix parameters $p_0$ and $p_1$. Here is an example with two qubits.

```{code-cell} ipython3
from functools import reduce

import numpy as np
from mitiq.rem import generate_inverse_confusion_matrix

# Confusion matrix for qubit 1
A1_entries = [
    0.9, 0.2,
    0.1, 0.8
]
A1 = np.array(A1_entries).reshape(2,2)

# Overall 2-qubit confusion matrix (tensor two copies of A1)
A = np.kron(A1, A1)

# Generate inverse confusion matrix.
# For this simple error model the user only has to
# specify the single qubit flip probabilities p0 and p1 
A_pinv = generate_inverse_confusion_matrix(2, p0=0.1, p1=0.2)

print(f"Confusion matrix:\n{A}\n")
print(f"Column-wise sums of confusion matrix:\n{A.sum(axis=0)}\n")
print(f"Inverse confusion matrix:\n{A_pinv}")
```

```{note}
In each code example we show an explicit computation of the full $2^n \times 2^n$ confusion matrix $A$ from the smaller, local confusion matrices supplied by the user, but this is solely for expository purposes. When applying REM in practice only the pseudoinverse $A^{+}$ needs to be computed: the user supplies one or more local confusion matrices, and Mitiq's utility functions can directly compute $A^{+}$ from these. 
```

### Inverse confusion matrix from $k$ local confusion matrices 
The function `generate_tensored_inverse_confusion_matrix(num_qubits, confusion_matrices)` can be applied to any $n$-qubit confusion matrix $A$ which is factorized into the tensor product of $k$ smaller, local confusion matrices (supplied by the user in `confusion_matrices`), one for each subset in a partition of the $n$ qubits into $k$ smaller subsets. The factorization encodes the assumption that there are $k$ independent/uncorrelated noise processes affecting the $k$ disjoint subsets of qubits (possibly of different sizes), but within each subset noise may be correlated between qubits in that subset. This model includes the simplest noise model above as the special case where $k=n$ and each of the $n$ single-qubit subsets has the same confusion matrix $A_1$:

$$
A = A^{(1)}_1 \otimes \dots \otimes A^{(n)}_1.
$$

For a slightly more nuanced model, one could still assume independent noise across qubits, but specify different $2 \times 2$ confusion matrices for each qubit:

$$
A = A^{(1)}_1 \otimes \dots \otimes A^{(n)}_n.  
$$

Here is an example with two qubits.

```{code-cell} ipython3
from mitiq.rem import generate_tensored_inverse_confusion_matrix

# Confusion matrix for qubit 1 (same as above)
A1_entries = [
    0.9, 0.2,
    0.1, 0.8
]
A1 = np.array(A1_entries).reshape(2,2)

# A different confusion matrix for qubit 2
A2_entries = [
    0.7, 0.4,
    0.3, 0.6
]
A2 = np.array(A2_entries).reshape(2,2)

# Overall 2-qubit confusion matrix (A1 tensor A2)
A = np.kron(A1, A2) 

# Generate inverse confusion matrix.
A_pinv = generate_tensored_inverse_confusion_matrix(2, confusion_matrices=[A1, A2]) 

print(f"Confusion matrix:\n{A}\n")
print(f"Column-wise sums of confusion matrix:\n{A.sum(axis=0)}\n")
print(f"Inverse confusion matrix:\n{A_pinv}")
```


More generally, one can provide `generate_tensored_inverse_confusion_matrix()` with a list of $k$ confusion matrices of any size (for any $k$, $1\leq k \leq n$),
as long as their dimensions when tensored together give an overall confusion matrix of the correct dimension $2^{n} \times 2^{n}$. For instance, the first confusion matrix might apply to qubits $1$ and $2$ while the $k$th applies to qubits $n-2, n-1, n$:

$$
A = A^{(1)}_{1,2} \otimes \dots \otimes A^{(k)}_{n-2, n-1, n}. 
$$

Here is an example with three qubits. We represent a stochastic noise model in which errors on
qubit $1$ are independent of errors on qubits $2$ and $3$, but errors on qubits $2$ and $3$ are correlated with
each other. So the confusion matrix factorizes into two differently sized sub-matrices $A = A^{(1)}_1 \otimes A^{(2)}_{2,3}$.

```{code-cell} ipython3
# Confusion matrix for qubit 1 (same as above)
A1_entries = [
    0.9, 0.2,
    0.1, 0.8
]
A1 = np.array(A1_entries).reshape(2,2)

# Generate a random 4x4 confusion matrix (square
# with columns summing to 1) to represent a
# a correlated error model on qubits 2 and 3
matrix = np.random.rand(4,4)
A23 = matrix/(matrix.sum(axis=0)[None,:])

# Overall 3-qubit confusion matrix (A1 tensor A23)
A = np.kron(A1, A23)

# Generate inverse confusion matrix.
A_pinv = generate_tensored_inverse_confusion_matrix(3, [A1, A23])

print(f"Confusion matrix:\n{A}\n")
print(f"Column-wise sums of confusion matrix:\n{A.sum(axis=0)}\n")
print(f"Inverse confusion matrix:\n{A_pinv}")
```

