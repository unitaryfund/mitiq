---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# What is the theory behind LRE?

Similar to [ZNE](zne.md), LRE works in two steps:

- **Step 1:** Intentionally create multiple noise-scaled but logically equivalent circuits by scaling each layer or chunk of the input circuit through unitary folding.

- **Step 2:** Extrapolate to the noiseless limit using multivariate richardson extrapolation.

The noise-scaled circuits in ZNE are scaled by the user choosing which layers of the input circuit to fold whereas in LRE
each noise-scaled circuit scales the layers in the input circuit in a specific pattern.
LRE leverages the flexible configuration space of layerwise unitary folding, allowing for a more nuanced mitigation of errors by treating the noise level of each layer of the quantum circuit as an independent variable.

## Step 1: Create noise-scaled circuits

The goal is to create noise-scaled circuits of different depths where the layers in each circuit are scaled in a specific pattern as a result of [unitary folding](zne-5-theory.md).
This pattern is described by the vector of scale factor vectors which are generated after the fold multiplier and degree for multivariate Richardson extrapolation are chosen.

Suppose we're interested in the value of some observable of a circuit $C$ that has $l$ layers.
For each layer $0 \leq L \leq l$ we can choose a scale factor for how much to scale that particular layer.
Thus a vector $\lambda \in \mathbb{R}^l_+$ corresponds to a folding configuration where $\lambda_0$ corresponds to the scale factor for the first layer, and $\lambda_{l - 1}$ is the scale factor to apply on the circuits final layer.

Fix the number of noise-scaled circuits we wish to generate at $M\in\mathbb{N}$.
Define $\Lambda = (λ_1, λ_2, \ldots, λ_M)^T$ to be the collection of scale factors and let $(C_{λ_1}, C_{λ_2}, \ldots, C_{λ_M})^T$ denote the noise-scaled circuits corresponding to each scale factor.

After $d$ is fixed as the degree of the multivariate polynomial, we define $M_j(λ_i, d)$ to be the terms in the polynomial arranged in increasing order.
In general, the number of monomial terms with $l$ variables up to degree $d$ can be determined
through the [stars and bars method](https://en.wikipedia.org/wiki/Stars_and_bars_%28combinatorics%29).

For example, if $C$ has 2 layers, the degree of the extrapolating polynomial is 2, the basis of monomials contains 6 terms: $\{1, λ_1, λ_2, {λ_1}^2, λ_1 \cdot λ_2, {λ_2}^2 \}$.

$$
\text{total number of terms in the monomial basis with max degree } d = \binom{d + l}{d}
$$

As the choice for the degree of the extrapolating polynomial is 2, we search for the number of terms with total degree 2 using the following formula:

$$
\text{number of terms in the monomial basis with total degree } d = \binom{d + l - 1}{d}
$$

Terms with total degree 2 are 3 calculated by $\binom{2 + 2 -1}{2} = 3$ and correspond to $\{{λ_1}^2, λ_1 \cdot λ_2, {λ_2}^2 \}$.

Similarly, number of terms with total degree 1 and 0 can be calculated as $\binom{1 + 2 -1}{1} = 2:\{λ_1, λ_2\}$ and $\binom{0 + 2 -1}{0}= 1: \{1\}$ respectively.

These terms in the monomial basis define the rows of the square sample matrix as shown below:

$$
\mathbf{A}(\Lambda, d) =
\begin{bmatrix}
    M_1(λ_1, d) & M_2(λ_1, d) & \cdots & M_N(λ_1, d) \\
    M_1(λ_2, d) & M_2(λ_2, d) & \cdots & M_N(λ_2, d) \\
    \vdots & \vdots & \ddots & \vdots \\
    M_1(λ_N, d) & M_2(λ_N, d) & \cdots & M_N(λ_N, d)
\end{bmatrix}
$$

For our example circuit of $l=2$ and $d=2$, each row defined by the generic monomial terms $\{M_1(λ_i, d), M_2(λ_i, d), \ldots, M_N(λ_i, d)\}$ in the sample matrix $\mathbf{A}$ will instead be replaced by $\{1, λ_1, λ_2, {λ_1}^2, λ_1 \cdot λ_2, {λ_2}^2 \}$.

Here, each monomial term in the sample matrix $\mathbf{A}$ is then evaluated using the values in the scale factor vectors. In Step 2, this sample matrix will be utilized to obtain our mitigated expectation value.

## Step 2: Extrapolate to the noiseless limit

Each noise scaled circuit $C_{λ_i}$ has an expectation value $\langle O(λ_i) \rangle$ associated with it such that we can define a vector of the noisy expectation values $z = (\langle O(λ_1) \rangle, \langle O(λ_2) \rangle, \ldots, \langle O(λ_M)\rangle)^T$.
These values can then be combined via a linear combination to estimate the ideal value $variable$.

$$
O_{\mathrm{LRE}} = \sum_{i=1}^{M} \eta_i \langle O(λ_i) \rangle.
$$

Finding the coefficients in the linear combination becomes a problem solvable through a system of linear equations $\mathbf{A} c = z$ where $c$ is the coefficients vector $(\eta_1, \eta_2, \ldots, \eta_N)^T$, $z$ is the vector of the noisy expectation values and $\mathbf{A}$ is the sample matrix evaluated using the values in the scale factor vectors.

The [general multivariate Lagrange interpolation polynomial](https://www.siam.org/media/wkvnvame/a_simple_expression_for_multivariate.pdf) is defined by a new matrix $\mathbf{B}_i$ obtained by replacing the $i$-th row of the sample matrix $\mathbf{A}$ with monomial terms evaluated using the generic variable λ. Thus, matrix $\mathbf{B}_i$ represents an interpolating polynomial in variable λ of degree $d$. As we only need to find the noiseless expectation value, we can skip calculating the full vector of linear combination coefficients if we use the [Lagrange interpolation formula](https://files.eric.ed.gov/fulltext/EJ1231189.pdf) evaluated at $λ = 0$ i.e. the zero-noise limit.

To get the matrix $\mathbf{B}_i(\mathbf{0})$, replace the $i$-th row of the sample matrix $\mathbf{A}$ by $\mathbf{e}_i=(1, 0, \ldots, 0)$ where except $M_1(0, d) = 1$ all the other monomial terms are zero when $λ=0$.

$$
O_{\rm LRE} = \sum_{i=1}^M \langle O (\boldsymbol{\lambda}_i)\rangle  \frac{\det \left(\mathbf{B}_i (\boldsymbol{0}) \right)}{\det \left(\mathbf{A}\right)}
$$

To summarize, based on a user's choice of degree of extrapolating polynomial for some circuit, expectation values from noise scaled circuits created in a specific pattern along with multivariate Lagrange interpolation of the sample matrix evaluated using the scale factor vectors are used to find error mitigated expectation value.

Additional details on the LRE functionality are available in the [API-doc](https://mitiq.readthedocs.io/en/stable/apidoc.html#module-mitiq.lre.multivariate_scaling.layerwise_folding).
