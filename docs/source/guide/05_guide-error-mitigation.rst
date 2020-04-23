.. _guide_qem:

*********************************************
About Error Mitigation
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
the most recognizable one is zero-noise extrapolation.

.. _guide_qem_zne:

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Zero-noise extrapolation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The key idea behind zero-noise extrapolation is that it is possible to make
some general assumptions on the kind of noise that affects, with error, the
results of a quantum computation.

In an ideal device, the time evolution is unitary, and as such it is modeled in
the intermediate representation of a quantum circuit,

.. math::

   \begin{eqnarray}
   |\psi\rangle (t)&=&U(t)|\psi\rangle
   =e^{-i\int_0^t H(t')/\hbar dt'}|\psi\rangle,
     \end{eqnarray}

where :math:`|\psi\rangle` is the initial state of the system (e.g., the qubits
involved in the operation) and :math:`U(t)` the unitary
time evolution set by a time-dependent Hamiltonian, H(t).


In the simplest scenario for the system-environment interaction, it is still
possible to describe the time evolution in terms of operators acting on the
system only, at the cost of losing the unitarity of the evolution.


The first required condition to develop such framework, is that the system
interacts less strongly with the environment than within its own
sub-constituents.  This allows to proceed with a perturbative approach to solve
the problem, with a coupling constant :math:`\lambda` taking care of the e.

In this case, it is possible to write the time evolution of the density matrix
associated to the state, :math:`\hat{\rho}=|\psi\rangle\langle \psi|`, as

.. math::

   \begin{eqnarray}
   \frac{\partial d}{ \partial t}\hat{\rho}&=&
   \frac{i}{\hbar}\lbrack H(t), \hat{\rho}\rbrack+\lambda \mathcal{L}
   \lbrack\hat{\rho}\rbrack,
   \end{eqnarray}

where :math:`mathcal{L}` is a super-operator acting on the Hilber space.

The subsequent most straightforward set of sensible approximations includes
assuming that at time zero the system and environment are not entangled, that
the environment is memoryless, and that there is a dominant scale of times set
by the interactions, wich allows to cut off high-frequency perturbations.

These so-called, respectively, Born, Markov and Rotating-Wave approximations,
lead to a so-called Lindblad form of the *dissipation*, i.e. to a special
structure of the system-environment interaction that can be represented with
a linear superoperator that always admits the Lindblad form

.. math::

   \begin{eqnarray}
   \mathcal{L}\lbrack\hat{\rho}\rbrack&=&\mathcal{L}\hat{\rho}
   =\sum_{i=1}^{N^2-1} \gamma_i \left( A_i\hat{\rho} A_i^\dagger
   - \frac{1}{2}( A_i^\dagger A_i\hat{\rho}+ \hat{\rho}A_i^\dagger A_i )\right)
   ,
   \end{eqnarray}
where :math:`\gamma_i` are constants that set the strenghts of the dissipation
mechanisms defined by the jump operators, :math:`A_i`.

The crucial idea behind zero-noise extrapolation is that, while some minimum
strength of noise is unavoidable in the system, it is still possible to
*increase* it to a value :math:`\lambda'=c\lambda`, with :math:`c>1`, so that
it is then possible to extrapolate the zero-noise limit.

This is done in practice by running a quantum circuit (simulation) and
calculating a given expectation variable, :math:`\langle X\rangle_\lambda`,
then rerunning the calculation (which is indeed a time evolution) for
:math:`\langle X\rangle_{\lambda'}`, and then extracting
:math:`\langle X\rangle_{0}`.

The extraction for :math:`\langle X\rangle_{0}` can occur with several
statistical fitting models, which can be linear or non-linear. These methods
are contained in the :mod:`mitiq.zne` module.


.. _guide_qem_uf:

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Unitary folding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Other examples of error mitigation techniques include injecting noisy gates
and perform a probabilistic error cancellation and inserting identity gates, or
unitary folding, in the time evolution, as a way to stretch time with respect
to noise processes. Here are some examples of :ref:`guide-folding`.


.. _guide_qem_what_not:

--------------------------------------
What quantum error mitigation *is not*
--------------------------------------

Quantum error mitigation is connected to quantum error correction and quantum
optimal control, two fields of study that also aim at reducing the impact of
errors in quantum information processing in quantum computers. While these are
fluid boundaries, it can be useful to point out some differences among these
two well-established fields and the emerging field of quantum error mitigation.
It is fair to say that even the terminology of "quantum error mitigation" or
"error mitigation" has only recently coalesced (from ~2015 onward), while even
in the previous decade similar concepts or techniques were scattered across
these and other fields. Suggestions for additional references are `welcome`_.

.. _welcome: https://github.com/unitaryfund/mitiq/issues/new

.. _guide_qem_qec:

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
techniques that stemmed as more practical quantum error correction solutions.

.. _guide_qem_qoc:

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
About quantum optimal control
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Optimal control theory is a very versatile set of techniques that can be
applied for many scopes. It entails many fields, and it is generally based on a
feedback loop between an agent and a target system.
Optimal control is applied to several quantum technologies,
including in the pulse shaping of gate design in quantum circuits calibration
against noisy devices.

Examples of specific application of optimal control to quantum dynamics and
quantum computing is in dynamical decoupling, a technique that employs fast
control pulses to effectively decouple a system and its environment, with
techniques partly borrowed from the nuclear magnetic resonance community.


.. _guide_qem_why:

-----------------------------------------
Why is quantum error mitigation important
-----------------------------------------

The noisy intermediate scale quantum computing (NISQ) era is charactherized by
short or medium-depth circuits and noise affecting operations, state
preparation, and measurement :cite:`Preskill_2018_Quantum`.

Current short-depth quantum circuits are noisy, at at the same time it is not
possible to implement on them quantum error correcting codes, which are more
demanding both in terms of necessary qubits and of circuit depths.

Error mitigation offers the prospects of writing more compact quantum circuits
that can estimate observables with more precision, i.e. increase the
performance of quantum computers.

By implementing quantum optics tools (such as the modeling noise and open
quantum systems), standard as well as cutting-edge statistics and inference
techniques, and tweaking them for the needs of the quantum computing community,
`mitiq` aims at providing the most comprehensive toolchain for error
mitigation.


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

- J. Wallman *et al.*, *Phys. Rev. A*, 2016 :cite:`Wallman_2016_PRA`
- K. Temme *et al.*, *Phys. Rev. Lett.*, 2017 :cite:`Temme_2017_PRL`
- S. Endo *et al.*, *Phys. Rev. X*, 2018 :cite:`Endo_2018_PRX`
- A. Kandala *et al.*, *Nature*, 2019 :cite:`Kandala_2019_Nature`
- Suguru Endo, *Hybrid quantum-classical algorithms and error mitigation*, PhD
Thesis, 2019, Oxford University (`Link`_).

.. _Link: https://ora.ox.ac.uk/objects/uuid:6733c0f6-1b19-4d12-a899-18946aa5df85


^^^^^^^^
Software
^^^^^^^^

Here is a (non-comprehensive) list of open-source software libraries related to
quantum computing, noisy quantum dynamics and error mitigation:

**IBM Q's `Qiskit`_** provides a stack for quantum computing simulation and
execution on real devices from the cloud. In particular, `qiskit.aer` contains
noise models, integrated with `mitiq` tools. Qiskit's OpenPulse provides
pulse-level control of qubit operations in some of the superconducting circuit
devices.

**Goole AI Quantum's `Cirq`_** offers quantum simulation of quantum circuits. It is
integrated with  `mitiq` algorithms.

**Rigetti Computing's `PyQuil`_** is a library for quantum programming. Rigetti's
stack offers the execution of quantum circuits on superconducting circuits
devices from the cloud, as well as their simulation on a quantum virtual
machine (QVM), integrated with `mitiq` tools.

**`QuTiP`_**, the quantum toolbox in Python, contains a quantum information processing
module that allows to simulate quantum circuits, their implementation on
devices, as well as the simulation of pulse-level control and time-dependent
density matrix evolution with the `qutip.qip.noise` module.

**`Krotov`_** is a package implementing Krotov method for optimal control,
interfacing with QuTiP for noisy density-matrix quantum evolution.

**`PyGSTi`_** allows to characterize quantum circuits by implementing techniques
such as gate set tomography (GST) and randomized benchmarking.

This is just a selection of open-source projects related to quantum error
mitigation. A more comprehensinve collection of software on quantum computing
can be found `here`_.


.. _QuTiP: http://qutip.org

.. _Qiskit: https://qiskit.org

.. _Cirq: http://cirq.readthedocs.io/

.. _PyQuiL: https://github.com/rigetti/pyquil

.. _Krotov: http://krotov.readthedocs.io/

.. _PyGSTi: https://www.pygsti.info/

.. _here: https://github.com/qosf/awesome-quantum-software

