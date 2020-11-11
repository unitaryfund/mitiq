.. _guide_qem:

*********************************************
About Error Mitigation
*********************************************

This is intended as a primer on quantum error mitigation, providing a
collection of up-to-date resources from the academic literature, as well as
other external links framing this topic in the open-source software ecosystem.
This recent review article :cite:`Endo_2020_arXiv` summarizes the theory behind many error-mitigating
techniques.

* :ref:`guide_qem_what`
* :ref:`guide_qem_why`
* :ref:`guide_qem_related`
* :ref:`guide_qem_references`

.. _guide_qem_what:

--------------------------------
What quantum error mitigation is
--------------------------------

Quantum error mitigation refers to a series of modern techniques aimed at
reducing (*mitigating*) the errors that occur in quantum computing algorithms.
Unlike software bugs affecting code in usual computers, the errors which we
attempt to reduce with mitigation are due to the hardware.

Quantum error mitigation techniques try to *reduce* the impact of noise in
quantum computations. They generally do not completely remove it. Alternative nomenclature refers to error mitigation as (approximate) error
suppression or approximate quantum error correction, but it is worth noting
that it is :ref:`different from error correction<guide_qem_what_not>`. Among the ideas that have been developed so far for quantum error mitigation,
a leading candidate is zero-noise extrapolation.

.. _guide_qem_zne:

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Zero-noise extrapolation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The crucial idea behind zero-noise extrapolation is that, while some minimum
strength of noise is unavoidable in the system, quantified by a quantity :math:`\lambda`,  it is still possible to
*increase* it to a value :math:`\lambda'=c\lambda`, with :math:`c>1`, so that
it is then possible to extrapolate the zero-noise limit. This is done in practice by running a quantum circuit (simulation) and
calculating a given expectation variable, :math:`\langle X\rangle_\lambda`,
then re-running the calculation (which is indeed a time evolution) for
:math:`\langle X\rangle_{\lambda'}`, and then extracting
:math:`\langle X\rangle_{0}`.
The extraction for :math:`\langle X\rangle_{0}` can occur with several
statistical fitting models, which can be linear or non-linear. These methods
are contained in the :mod:`mitiq.zne.inference` and :mod:`mitiq.zne` modules.

In theory, one way zero-noise extrapolation can be simulated, also with ``mitiq``,
is by picking an underlying noise model, e.g., a memoryless bath such that the system dissipates with Lindblad dynamics. Likewise, zero-noise extrapolation can be applied also to non-Markovian noise models :cite:`Temme_2017_PRL`. However, it is important to point out that zero-noise extrapolation is a very general method in which one is free to scale and extrapolate almost whatever parameter one wishes to, even if the underlying noise model is unknown.

In experiments, zero-noise extrapolation has been performed with pulse
stretching :cite:`Kandala_2019_Nature`. In this way, a difference between the effective time that a gate is affected by decoherence during its execution on the hardware
was introduced by controlling only the gate-defining pulses. The effective noise of a quantum circuit can be scaled also at a gate-level, i.e., without requiring a direct control of the  physical hardware. For example this can be achieved with the :ref:`unitary folding<guide-folding>` technique, a method which is present in the ``mitiq`` toolchain.


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Limitations of zero-noise extrapolation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Zero-noise extrapolation is a useful error mitigation technique but it is not without limitations. First and foremost,
the zero-noise estimate is extrapolated, meaning that performance guarantees are quite difficult in general. If a
reasonable estimate of how increasing the noise affects the observable is known, then ZNE can produce good zero-noise
estimates. This is the case for simple noise models such as depolarizing noise, but noise in actual quantum systems is
more complicated and can produce different behavior than expected. In this case the performance of ZNE is tied to the
performance of the underlying hardware. If expectation values do not produce a smooth curve as noise is increased, the
zero-noise estimate may be poor and certain inference techniques may fail. In particular, one has to take into account
that any initial error in the measured expectation values will propagate to the zero-noise extrapolation value. This
fact can significantly amplify the final estimation uncertainty. In practice, this implies that the evaluation of a
mitigated expectation value requires more measurement shots with respect to the unmitigated one.

Additionally, zero-noise extrapolation cannot increase the performance of arbitrary circuits. If the circuit is large
enough such that the expectation of the observable is almost constant as noise is increased (e.g., if the state is
maximally mixed), then extrapolation will of course not help the zero-noise estimate. The regime in which ZNE is
applicable thus depends on the performance of the underlying hardware as well as the circuit. A detailed description
of when zero-noise extrapolation is effective, and how effective it is, is the subject of ongoing research.

In Mitiq, this technique is implemented in the module :mod:`mitiq.zne`.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Probabilistic error cancellation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Probabilistic error cancellation uses a quasi-probability representation :cite:`Temme_2017_PRL` to express an ideal quantum 
channel as a linear combination of noisy operations. Given a set of noisy but implementable operations :math:`\Omega = \{O_1, \dots, O_m\}`, an ideal unitary gate 
can be expressed as :math:`\mathcal{G} = \sum_{\alpha} \eta_{\alpha} \mathcal{O}_\alpha = \gamma \sum_{\alpha} P(\alpha) \sigma(\alpha) \mathcal{O}_\alpha`, where
:math:`\eta_\alpha` are real coefficients, :math:`\gamma = \sum_{\alpha} |\eta_\alpha|`, :math:`P(\alpha)=|\eta_\alpha | /\gamma` is a probability 
distribution, and :math:`\sigma(\alpha)={\rm sign}(\eta_\alpha)`.

In this setting, we would like to estimate the ideal expectation value of some observable of interest :math:`\langle X\rangle_{\text{ideal}}`, 
after the action of an ideal circuit given by a sequence of ideal quantum gates :math:`\{\mathcal{\mathcal G}_i\}_{i=1}^L`. This can be achieved by 
sampling for each ideal gate :math:`\mathcal{G}_i` a noisy operation :math:`\mathcal{O}_{\alpha}` with probability 
:math:`P_i(\alpha)`. This random sampling will produce a noisy circuit (given by the sequence of sampled operations :math:`\{\mathcal{O}_{\alpha_i}\}_{i=1}^L`)
whose execution produces the final mixed state :math:`\rho_f`.
Then, by measuring the observable :math:`X`, setting :math:`\gamma_{\text{tot}} := \prod_{i}^L \gamma_i` and 
:math:`\sigma_{\text{tot}} = \prod_{i=1}^L \sigma_i(\alpha)`, one can obtain an unbiased estimate of the ideal expectation value as :math:`\langle 
X\rangle_{\text{ideal}} =  \mathbb E \left[ \gamma_{\text{tot}} \sigma_{\text{tot}} X_{\rm noisy} \right]`, where :math:`X_{\rm noisy}` is
the experimental estimate of :math:`{\rm tr}[\rho_f X]` and :math:`\mathbb E` is the sample average over many repetitions of the previous procedure.

In Mitiq, this technique is implemented in the module :mod:`mitiq.pec`.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Limitations of probabilistic error cancellation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The number of samples required to estimate the ideal expectation value with error :math:`\delta` and probability :math:`1-\epsilon` scales as 
:math:`\left(2 \gamma_{\text{tot}}^{2} / \delta^{2}\right) \log (2 / \epsilon)`  :cite:`Takagi2020optimal`. Thus, the sampling overhead is determined 
by :math:`\gamma_{\text{tot}}` which grows exponentially in the number of gates. It is then crucial to find a linear decomposition that minimizes :math:`\gamma_{\text{tot}}`. 
In addition, a full characterization of the noisy operations up to a good precision is required, which can be costly depending on the implementation.
 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Other error mitigation techniques
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Other examples of error mitigation techniques include injecting noisy gates for randomized compiling or the use of subspace reductions and symmetries. A collection of references on this cutting-edge implementations can be found in the :ref:`guide_qem_articles` subsection.

.. _guide_qem_why:

-----------------------------------------
Why quantum error mitigation is important
-----------------------------------------

The noisy intermediate scale quantum computing (NISQ) era is characterized by
short or medium-depth circuits in which noise affects state
preparation, gate operations, and measurement :cite:`Preskill_2018_Quantum`. Current short-depth quantum circuits are noisy, and at the same time it is not
possible to implement quantum error correcting codes on them due to the
needed qubit number and circuit depth required by these codes.

Error mitigation offers the prospects of writing more compact quantum circuits
that can estimate observables with more precision, i.e. increase the
performance of quantum computers. By implementing quantum optics tools (such as the modeling noise and open quantum systems) :cite:`Carmichael_1999_Springer,Carmichael_2007_Springer,Gardiner_2004_Springer,Breuer_2007_Oxford`, standard as well as cutting-edge statistics and inference
techniques, and tweaking them for the needs of the quantum computing community,
``mitiq`` aims at providing the most comprehensive toolchain for error
mitigation.

.. _guide_qem_related:

--------------------------------------------------
Related fields
--------------------------------------------------

Quantum error mitigation is connected to quantum error correction and quantum
optimal control, two fields of study that also aim at reducing the impact of
errors in quantum information processing in quantum computers. While these are
fluid boundaries, it can be useful to point out some differences among these
two well-established fields and the emerging field of quantum error mitigation.

It is fair to say that even the terminology of "quantum error mitigation" or
"error mitigation" has only recently coalesced (from ~2015 onward), while even
in the previous decade similar concepts or techniques were scattered across
these and other fields. Suggestions for additional references are `welcome`_.

.. _welcome: https://github.com/unitaryfund/mitiq/issues/new/choose

.. _guide_qem_qec:

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Quantum error correction
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
qubits. What is now referred quantum error mitigation is generally a series of
techniques that stemmed as more practical quantum error correction solutions
:cite:`Knill_2005_Nature`.

.. _guide_qem_qoc:

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Quantum optimal control
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Optimal control theory is a very versatile set of techniques that can be
applied for many scopes. It entails many fields, and it is generally based on a
feedback loop between an agent and a target system.
Optimal control is applied to several quantum technologies,
including in the pulse shaping of gate design in quantum circuits calibration
against noisy devices :cite:`Brif_2010_NJP`.

A key difference between some quantum error mitigation techniques and quantum
optimal control is that the former can be implemented in some instances with
post-processing techniques, while the latter relies on an active feedback loop. An example of a specific application of optimal control to quantum dynamics that can be seen as a quantum error mitigation technique, is in dynamical decoupling :cite:`Viola_1999_PRL`. This technique employs fast control pulses to effectively decouple a system from its environment, with techniques pioneered in the nuclear magnetic resonance
community.

.. _guide_qem_noise:

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Open quantum systems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

More in general, quantum computing devices can be studied in the framework of
open quantum systems :cite:`Carmichael_1999_Springer,Carmichael_2007_Springer,Gardiner_2004_Springer,Breuer_2007_Oxford`, that is, systems that exchange
energy and information with the surrounding environment. On the one hand, the qubit-environment exchange can be controlled, and this feature is actually fundamental to extract information and process it.
On the other hand, when this interaction is not controlled — and at the fundamental level it cannot be completely suppressed — noise eventually kicks in, thus introducing errors that are disruptive for the *fidelity* of the information-processing protocols.


Indeed, a series of issues arise when someone wants to perform a calculation on a
quantum computer. This is due to the fact that quantum computers are devices that are embedded in an environment and interact with it. This means that stored information can be corrupted, or that, during calculations, the protocols are not faithful.

Errors occur for a series of reasons in quantum computers and the microscopic
description at the physical level can vary broadly, depending on the quantum
computing platform that is used, as well as the computing architecture. For example, superconducting-circuit-based quantum computers have chips that
are prone to cross-talk noise, while qubits encoded in trapped ions need to be
shuttled with electromagnetic pulses, and solid-state artificial atoms, including quantum dots, are heavily affected by inhomogeneous broadening :cite:`Buluta_2011_RPP`.

.. _guide_qem_references:

---------------------
External References
---------------------

Here is a list of useful external resources on quantum error mitigation,
including software tools that provide the possibility of studying quantum
circuits.

.. _guide_qem_articles:

^^^^^^^^^^^^^^^^^
Research articles
^^^^^^^^^^^^^^^^^

A list of research articles academic resources on error mitigation:

- On **zero-noise extrapolation**:
   - Theory, Y. Li and S. Benjamin, *Phys. Rev. X*, 2017 :cite:`Li_2017_PRX` and K. Temme *et al.*, *Phys. Rev. Lett.*, 2017 :cite:`Temme_2017_PRL`
   - Experiment on superconducting circuit chip, A. Kandala *et al.*, *Nature*, 2019 :cite:`Kandala_2019_Nature`
   
- On **probabilistic error cancellation**:
   - Theory, Y. Li and S. Benjamin, *Phys. Rev. X*, 2017 :cite:`Li_2017_PRX` and K. Temme *et al.*, *Phys. Rev. Lett.*, 2017 :cite:`Temme_2017_PRL`
   - Resource analysis for probabilistic error cancellation, Ryuji Takagi, arxiv, 2020 :cite:`Takagi2020optimal`

- On **randomization methods**:
   - Randomized compiling with twirling gates, J. Wallman *et al.*, *Phys. Rev. A*, 2016 :cite:`Wallman_2016_PRA`
   - Porbabilistic error correction, K. Temme *et al.*, *Phys. Rev. Lett.*, 2017 :cite:`Temme_2017_PRL`
   - Practical proposal, S. Endo *et al.*, *Phys. Rev. X*, 2018 :cite:`Endo_2018_PRX`
   - Experiment on trapped ions, S. Zhang  *et al.*, *Nature Comm.* 2020 :cite:`Zhang_2020_NatComm`
   - Experiment with gate set tomography on a supeconducting circuit device, J. Sun *et al.*, 2019 arXiv :cite:`Sun_2020_arXiv`

- On **subspace expansion**:
   - By hybrid quantum-classical hierarchy introduction, J. McClean *et al.*, *Phys. Rev. A*, 2017 :cite:`McClean_2017_PRA`
   - By symmetry verification, X. Bonet-Monroig *et al.*, *Phys. Rev. A*, 2018 :cite:`Bonet_2018_PRA`
   - With a stabilizer-like method, S. McArdle *et al.*, *Phys. Rev. Lett.*, 2019, :cite:`McArdle_2019_PRL`
   - Exploiting molecular symmetries, J. McClean *et al.*, *Nat. Comm.*, 2020 :cite:`McClean_2020_NatComm`
   - Experiment on a superconducting circuit device, R. Sagastizabal *et al.*, *Phys. Rev. A*, 2019 :cite:`Sagastizabal_2019_PRA`

- On **other error-mitigation techniques** such as:
   - Approximate error-correcting codes in the generalized amplitude-damping channels, C. Cafaro *et al.*, *Phys. Rev. A*, 2014 :cite:`Cafaro_2014_PRA`:
   - Extending the variational quantum eigensolver (VQE) to excited states, R. M. Parrish *et al.*, *Phys. Rev. Lett.*, 2017 :cite:`Parrish_2019_PRL`
   - Quantum imaginary time evolution, M. Motta *et al.*, *Nat. Phys.*, 2020 :cite:`Motta_2020_NatPhys`
   - Error mitigation for analog quantum simulation, J. Sun *et al.*, 2020, arXiv :cite:`Sun_2020_arXiv`

- For an extensive introduction: S. Endo, *Hybrid quantum-classical algorithms and error mitigation*, PhD Thesis, 2019, Oxford University (`Link`_), or :cite:`Endo_2020_arXiv`.

.. _Link: https://ora.ox.ac.uk/objects/uuid:6733c0f6-1b19-4d12-a899-18946aa5df85

^^^^^^^^
Software
^^^^^^^^

Here is a (non-comprehensive) list of open-source software libraries related to
quantum computing, noisy quantum dynamics and error mitigation:

- **IBM Q**'s `Qiskit`_ provides a stack for quantum computing simulation and execution on real devices from the cloud. In particular, ``qiskit.Aer`` contains the :class:`~qiskit.providers.aer.noise.NoiseModel` object, integrated with ``mitiq`` tools. Qiskit's OpenPulse provides pulse-level control of qubit operations in some of the superconducting circuit devices. ``mitiq`` is integrated with ``qiskit``, in the :mod:`~mitiq.mitiq_qiskit.qiskit_utils` and :mod:`~mitiq.mitiq_qiskit.conversions` modules.

- **Goole AI Quantum**'s `Cirq`_ offers quantum simulation of quantum circuits. The :class:`cirq.Circuit` object is integrated in  ``mitiq`` algorithms as the default circuit.

- **Rigetti Computing**'s `PyQuil`_ is a library for quantum programming. Rigetti's stack offers the execution of quantum circuits on superconducting circuits devices from the cloud, as well as their simulation on a quantum virtual machine (QVM), integrated with ``mitiq`` tools in the :mod:`~mitiq.mitiq_pyquil.pyquil_utils` module.

- `QuTiP`_, the quantum toolbox in Python, contains a quantum information processing module that allows to simulate quantum circuits, their implementation on devices, as well as the simulation of pulse-level control and time-dependent density matrix evolution with the :class:`qutip.Qobj` object and the :class:`~qutip.qip.device.Processor` object in the ``qutip.qip`` module.

- `Krotov`_ is a package implementing Krotov method for optimal control interfacing with QuTiP for noisy density-matrix quantum evolution.

- `PyGSTi`_ allows to characterize quantum circuits by implementing techniques such as gate set tomography (GST) and randomized benchmarking.

This is just a selection of open-source projects related to quantum error
mitigation. A more comprehensinve collection of software on quantum computing
can be found `here`_ and on `Unitary Fund`_'s list of supported projects.

.. _QuTiP: http://qutip.org

.. _Qiskit: https://qiskit.org

.. _Cirq: http://cirq.readthedocs.io/

.. _PyQuiL: https://github.com/rigetti/pyquil

.. _Krotov: http://krotov.readthedocs.io/

.. _PyGSTi: https://www.pygsti.info/

.. _here: https://github.com/qosf/awesome-quantum-software

.. _Unitary Fund: https://unitary.fund#grants-made
