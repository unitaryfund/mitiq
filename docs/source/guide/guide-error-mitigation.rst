.. _guide_qem:

*********************************************
Quantum error mitigation
*********************************************

This is intended as a primer on quantum error mitigation, providing a
collection of up-to-date resources from the academic literature, as well as
other external links framing this topic in the open-source software ecosystem.

* :ref:`guide_qem_noise`
* :ref:`guide_qem_what`
* :ref:`guide_qem_what_not`
* :ref:`guide_qem_why`
* :ref:`guide_qem_references`

.. _guide_qem_noise:

--------------------------------
Noise in quantum devices
--------------------------------

A series of issues arise when someone wants to perform a calculation on a
quantum computer.

This is due to the fact that quantum computers are devices that are embedded in
an environment and interact with it. This means that stored information can be
corrupted or that during calculations, the protocols are not faithful.

Errors occur for a series of reasons in quantum computers and the microscopic
description at the physical level can vary broadly, depending on the quantum
computing platform that is used, as well as the computing architecture.

For example, superconducting-circuit-based quantum computers are more prone to
cross-talk noise, while ion-based quantum computers need to counteract
inhomogeneous broadening noise.


.. _guide_qem_what:

--------------------------------
What is quantum error mitigation
--------------------------------

Quantum error mitigation refers to a series of modern techniques aimed at
reducing (*mitigating*) the error that occur in quantum computing algorithms.

Unlike code bugs affecting code in usual computers, the nature of errors
corrected by quantum mitigation is due to the hardware.

Quantum error mitigation techniques try to *reduce* the impact of noise in
quantum computations. They generally do not completely remove it.

Among the ideas that have been developed so far for quantum error mitigation,
some of the most recognizable are:

* Zero-noise extrapolation

* (Quasi-)Probabilistic error cancellation

* (Quasi-)Probabilistic error cancellation

(randomized compiling by gate twirling

.. _guide_qem_zne:

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Zero-noise extrapolation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _guide_qem_pec:

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Quasi-)Probabilistic error cancellation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _guide_qem_uf:

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Unitary folding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. _guide_qem_what_not:

--------------------------------------
What quantum error mitigation *is not*
--------------------------------------

Quantum error mitigation is connected to quantum error correction and quantum
optimal control, two fields of study that also aim at reducing the impact of
errors in quantum information processing in quantum computers. While these are
fluid boundaries, it can be useful to point out some differences among these.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
About quantum error correction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Quantum error correction is different from quantum error mitigation, as it
introduces a series of techniques that generally aim at completely *removing*
the impact of errors on quantum computations. In particular, if errors
occurs below a certain threshold, the robustness of the quantum computation can
 be preserved, and fault tolerance is reached.

The main issue of quantum error correction techniques are that generally they
require a large overhead in terms of additional qubits on top of those required
 for the quantum computation. Current quantum computing devices have been able
 to demonstrate quantum error correction only with a very small number of
 qubits.

What is now referred quantum error mitigation is generally a series of
techniques that stemmed as more practical quantum error correction.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
About quantum optimal control
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Optimal control theory is a very versatile set of techniques that can be
applied for many scopes. It entails many fields, and it is generally based on a
feedback loop between an agent and a target system.
Optimal control is applied to several quantum technologies


.. _guide_qem_why:

-----------------------------------------
Why is quantum error mitigation important
-----------------------------------------

^^^^^^^^^^^^
NISQ devices
^^^^^^^^^^^^
* Noise + quantum computing
* Prospects: increasing the usability of short-depth quantum circuits
* Connections: bringing together quantum optics tools (modeling noise and open
quantum systems) and quantum computing community.


.. _guide_qem_references:

---------------------
Additional References
---------------------

Here is a list of useful external resources on quantum error mitigation,
including software tools that provide the possibility of studying quantum
circuits.

^^^^^^^^^^^^^^^^^
Research articles
^^^^^^^^^^^^^^^^^

A list of research articles and PhD theses is this one:

:cite:`Wallman_2016_PRA` https://doi.org/10.1103/PhysRevA.94.052325
:cite:`Temme_2017_PRL`
:cite:`Endo_2018_PRX`
:cite:`Kandala_2019_Nature`

Suguru Endo, *Hybrid quantum-classical algorithms and error mitigation*, PhD
Thesis, 2019, Oxford University (`Link`_).

.. _Link: https://ora.ox.ac.uk/objects/uuid:6733c0f6-1b19-4d12-a899-18946aa5df85

.. _PyGSTi_article: https://arxiv.org/abs/2002.12476

Ball, H., Biercuk, M. J., Carvalho, A., Chakravorty, R., Chen, J., de Castro, L. A., ... & Love, R. (2020). Software tools for quantum control: Improving quantum computer performance through noise and error suppression. arXiv preprint arXiv:2001.04060.

Pokharel, B., Anand, N., Fortman, B., & Lidar, D. A. (2018). Demonstration of fidelity improvement using dynamical decoupling with superconducting qubits. Physical review letters, 121(22), 220502.

^^^^^^^^
Software
^^^^^^^^

Here is a (non-comprehensive) list of open-source software libraries:

IBM Q's `Qiskit`_ provides a stack for quantum computing simulation and
execution on real devices from the cloud. In particular, `qiskit.aer` contains
noise models, integrated with `mitiq` tools. Qiskit's OpenPulse provides
pulse-level control of qubit operations in some of the superconducting circuit
devices.

Goole AI Quantum's `Cirq`_ offers quantum simulation of quantum circuits. It is
integrated with  `mitiq` algorithms.

Rigetti Computing's `PyQuil`_ is a library for quantum programming. Rigetti's
stack offers the execution of quantum circuits on superconducting circuits
devices from the cloud, as well as their simulation on a quantum virtual
machine (QVM), integrated with `mitiq` tools.

`QuTiP`_, the quantum toolbox in Python, contains a quantum information processing
module that allows to simulate quantum circuits, their implementation on
devices, as well as the simulation of pulse-level control and time-dependent
density matrix evolution with the `qutip.qip.noise` module.

`Krotov`_ is a package implementing Krotov method for optimal control,
interfacing with QuTiP for noisy density-matrix quantum evolution.

`PyGSTi`_ allows to characterize quantum circuits by implementing techniques
such as gate set tomography (GST) and randomized benchmarking.

This is just a selection of open-source projects related to quantum error
mitigation. A more comprehensinve collection of software on quantum computing
can be found `here`_.


.. _Qiskit: https://qiskit.org

.. _Cirq: http://cirq.readthedocs.io/

.. _PyQuiL: https://github.com/rigetti/pyquil

.. _Krotov: http://krotov.readthedocs.io/

.. _PyGSTi: https://www.pygsti.info/

.. _here: https://github.com/qosf/awesome-quantum-software

