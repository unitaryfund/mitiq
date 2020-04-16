.. _guide_qem:

*********************************************
Quantum error mitigation
*********************************************

This is intended as a primer on quantum error mitigation, providing a
collection of up-to-date resources from the academic literature, as well as
other external links framing this topic in the open-source software ecosystem.

* :ref:`guide_qem_noise`
* :ref:`guide_qem_what`
* :ref:`guide_qem_why`
* :ref:`guide_qem_references`

.. _guide_qem_noise:

--------------------------------
Noise in quantum devices
--------------------------------

Quantum error mitigation refers to a series of modern techniques aimed at
reducing (*mitigating*) the error that occur in quantum computing algorithms.

Unlike code bugs affecting code in usual computers, the nature of errors
corrected by quantum mitigation is due to the hardware.

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

Quantum error mitigation techniques try to *reduce* the impact of noise in
quantum computations. They generally do not completely remove it.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Differences from quantum error correction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Quantum error correction is different from quantum error mitigation, as it
introduces a series of techniques that generally aim

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Differences from quantum optimal control
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

^^^^^^^^^^^^^^^^^
Research articles
^^^^^^^^^^^^^^^^^

^^^^^^^^
Software
^^^^^^^^
A collection of software on quantum computing can be found here.

