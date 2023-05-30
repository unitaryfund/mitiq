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

# What is the theory behind Pauli Twirling?

Pauli Twirling (PT) {cite}`Wallman_2016_PRA, Hashim_2021_PRX, Urbanek_2021_PRL, Saki_2023_arxiv`
is a quantum error mitigation technique designed to tailor noise from a Markovian
channel to a more manageable stochastic Pauli channel. This tailoring is achieved
by randomly applying a series of Pauli operations to the quantum system, and in doing
so reduces the complexity of the noise channel.

The application of DD sequences can have two effects depending on the correlation time {cite}`Breuer_2007_Oxford` of the environment:

1. For Markovian noise, PT can make the overall quantum channel more symmetric (analogous to dynamical decoupling {cite}`Viola_1998_PRA, Viola_1999_PRL, Zhang_2014_PRL`)

2. For non-Markovian noise, PT's performance might be degraded due to the inability to completely decouple the system from the environment in the presence of memory effects

Pauli Twirling (PT) can be a powerful tool for noise management in quantum systems. By twirling over the Pauli gates, PT transforms complex noise channels into simpler stochastic Pauli noise channels.

The success of PT is contingent on various factors, such as the nature of the noise and the specific characteristics of the quantum system. It's worth noting that, while PT generally simplifies the noise channel, there are circumstances where it could transform the noise negatively, for example into a completely depolarizing channel with a corresponding total loss of quantum information.

For optimal results, Pauli Twirling should be implemented with an understanding of the underlying noise dynamics, and ideally, should be complemented with other error mitigation techniques to ensure robust quantum computation.

In the context of quantum error mitigation, PT is closer to [DDD](ddd-5-theory.md) in that they have very peculiar features with respect to other error mitigation techniques. PT's peculiarities include:

- It constructs a _single_ randomly-modified circuit, and subsequently averages over its executions.
With a single circuit, both the computational cost and complexity are reduced, making the final average of results a relatively straightforward task. That is, there is no need to take a linear combinations of noisy results as in [ZNE](zne-5-theory.md) [PEC](pec-5-theory.md) and [CDR](cdr-5-theory.md).

- As a consequence of the previous point, the fundamental error mitigation overhead is minimized,
such that there is no increase in statistical uncertainty in the final result, assuming optimal executions.

- If noise is time-correlated, PT should not be expected to yield positive (noise-reducing) results on its own,
though in this case composing with other techniques may yield the desired mitigation.
