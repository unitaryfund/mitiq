.. mitiq documentation file

.. _guide overview:

Overview of mitiq
=================
Welcome to the `mitiq` Users Guide.


What is mitiq for?
##################

Today's quantum computers have a lot of noise. This is a problem for quantum
programmers everywhere. `Mitiq` is an open source Python library
currently under development by `Unitary Fund <https://unitary.fund/>`_. It
helps solve this problem by compiling your programs to be more robust to noise.

`Mitiq` helps you do more quantum programming with less quantum compute.

Today's `mitiq` library is based around the zero-noise extrapolation technique.
These references [1][2] give background on the technique. The implementation
in mitiq is an optimized, extensible framework for zero-noise extrapolation.
In the future other error-mitigating techniques will be added to `mitiq`.

`Mitiq` is a framework agnostic library with a long term vision to be useful
for quantum programmers using any quantum programming framework and any quantum
backend. Today we support `cirq` and `qiskit` inputs and backends.

Check out more in our `getting started <guide-getting-started.html>`_ section.


.. [1] `Error mitigation for short-depth quantum circuits <https://arxiv.org/abs/1612.02058>`_
.. [2] `Extending the computational reach of a noisy superconducting quantum processor <https://arxiv.org/abs/1805.04492>`_