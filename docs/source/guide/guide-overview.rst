.. mitiq documentation file

.. _guide overview:

Overview of mitiq
=================
Welcome to the `mitiq` Users Guide.


What is mitiq for?
##################

Today's quantum computers have a lot of noise. This is a problem for quantum
programmers everywhere. Mitiq is an open source Python library
currently under development by `Unitary Fund <https://unitary.fund/>`_. It
helps solve this problem by compiling your programs to be more robust to noise.

Mitiq helps you do more quantum programming with less quantum compute.

The current Mitiq library is based around the zero-noise extrapolation technique.
These references :cite:`Temme_2017_PRL,Kandala_2019_Nature` give background on
the technique. The implementation in mitiq is an optimized, extensible
framework for zero-noise extrapolation.
In the future other error-mitigating techniques will be added to Mitiq.

Mitiq is a framework agnostic library with a long term vision to be useful
for quantum programmers using any quantum programming framework and any quantum
backend. Today we support `Cirq <https://cirq.readthedocs.io/en/stable/>`_,
`Qiskit <https://qiskit.org/>`_, and `PyQuil <https://pyquil-docs.rigetti.com/en/stable/>`_
frontends and backends.

Check out more in our `getting started <guide_02-getting-started.html>`_ section.
