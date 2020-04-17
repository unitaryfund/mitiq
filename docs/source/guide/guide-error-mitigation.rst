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

* zero-noise extrapolation

* twisting

* (quasi-)probabilistic error cancellation

.. _guide_qem_what_not:

--------------------------------------
What quantum error mitigation *is not*
--------------------------------------


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Differences from quantum error correction
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
Differences from quantum optimal control
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Optimal control theory is a very versatile set of techniques that can be
applied for many scopes. It entails many fields, and it is generally based on a


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

^^^^^^^^
Software
^^^^^^^^
IBM Q's Qiskit provides a stack for quantum computing simulation and execution
on real devices from the cloud. It is integrated with `mitiq` algorithms.

Goole AI Quantum's Cirq offers quantum simulation of quantum circuits. It is
 integrated with  `mitiq` algorithms.

Rigetti Computing's Pyquil also offers execution of quantum circuits and their
 simulation on a quantum virtual machine (QVM) and is integrated with `mitiq`
 algorithms.

Rigetti Computing's Pyquil also offers execution of quantum circuits and their
 simulation on a quantum virtual machine (QVM) and is integrated with `mitiq`
 algorithms.


A comprehensinve collection of software on quantum computing can be found
`here`_.


.. _here: https://github.com/qosf/awesome-quantum-software
