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

Pauli Twirling (PT) {cite}`Wallman_2016_PRA, Hashim_2021_PRX, Urbanek_2021_PRL`
is a quantum error mitigation technique designed to tailor noise from a Markovian
channel to a more manageable stochastic Pauli channel. This tailoring is achieved
by randomly applying a series of Pauli operations to the quantum system, and in doing
so reduces the complexity of the noise channel.

The application of DD sequences can have two effects depending on the correlation time {cite}`Breuer_2007_Oxford` of the environment:

1. For Markovian noise, PT can make the overall quantum channel more symmetric (analogous to dynamical decoupling {cite}`Viola_1998_PRA, Viola_1999_PRL, Zhang_2014_PRL`)

2. For non-Markovian noise, PT's performance might be degraded due to the inability to completely decouple the system from the environment in the presence of memory effects

Pauli Twirling (PT) can be a powerful tool for noise management in quantum systems. By twirling over the Pauli gates, PT transforms complex noise channels into simpler stochastic Pauli noise channels. Yet, as with all powerful tools, careful handling is required.

The success of PT is contingent on various factors, such as the nature of the noise and the specific characteristics of the quantum system. It's worth noting that, while PT generally simplifies the noise channel, there are circumstances where it could transform the noise negatively, for example into a completely depolarizing channel with a corresponding total loss of quantum information.

For optimal results, Pauli Twirling should be implemented with an understanding of the underlying noise dynamics, and ideally, should be complemented with other error mitigation techniques to ensure robust quantum computation.

In the context of quantum error mitigation, PT and DDD stand apart in that they have very peculiar features with respect to other error mitigation techniques. PT's peculiarities include:

- It maps a noisy quantum computation to a _single_ error-mitigated computation (no need to take linear combinations
of noisy results as in [ZNE](zne-5-theory.md [PEC](pec-5-theory.md) and [CDR](cdr-5-theory.md)).

- As a consequence of the previous point, there is not a fundamental error mitigation overhead or
increase in statistical uncertainty in the final result.

- If noise is time-correlated, PT should not be expected to yield positive (noise-reducing) results on its own

