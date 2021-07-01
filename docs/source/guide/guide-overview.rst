.. mitiq documentation file

.. _guide overview:

Overview of Mitiq
=================

Quantum computers are noisy, and executed quantum programs have errors.
Mitiq is an open-source Python library for mitigating errors when executing
quantum programs. For an introduction to the error mitigation techniques implemented in Mitiq, see the
:ref:`Getting Started <guide-getting-started>` guide.

Developed by `Unitary Fund <https://unitary.fund/>`_, Mitiq is a framework-agnostic
library with a long-term vision to be useful for quantum programmers using any quantum programming
framework and any quantum backend. Today we support `Cirq <https://quantumai.google/cirq/>`_,
`Braket <https://github.com/aws/amazon-braket-sdk-python>`_, `PyQuil <https://pyquil-docs.rigetti.com/en/stable/>`_,
and `Qiskit <https://qiskit.org/>`_ frontends. Any backends that (i) you have access to and (ii) accept quantum programs
written in one of the supported frontends can be used with Mitiq.