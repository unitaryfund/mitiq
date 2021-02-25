.. mitiq documentation file

.. _guide overview:

Overview of Mitiq
=================

Quantum computers are noisy, and executed quantum programs have errors.
Mitiq is an open-source Python library for mitigating errors when executing
quantum programs. For an introduction to the error mitigation techniques implemented in Mitiq, see the
:ref:`Getting Started <guide-getting-started>` guide.

Developed by `Unitary Fund <https://unitary.fund/>`_, Mitiq is a framework agnostic
library with a long-term vision to be useful for quantum programmers using any quantum programming
framework and any quantum backend. Today we support `Cirq <https://cirq.readthedocs.io/en/stable/>`_,
`Qiskit <https://qiskit.org/>`_, and `PyQuil <https://pyquil-docs.rigetti.com/en/stable/>`_
frontends and backends.

The guide contains information of how :ref:`zero noise extrapolation <guide_zne>` can be implemented with the software and examples of how it can be applied to any simulator or quantum processor running :ref:`Back-end Plug-ins <guide-executors>`.

A brief overview with more information about error mitigation techniques from the research literature can be found :ref:`here <guide_qem>` (instead, the :ref:`examples` section shows how Mitiq can be used in your :ref:`research`.)

