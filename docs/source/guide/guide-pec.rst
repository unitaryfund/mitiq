.. pec:

*********************************************
Probabilistic Error Cancellation
*********************************************

Probabilistic error cancellation (PEC) capabilities are currently included in Mitiq. The :py:mod:`~mitiq.pec.pec` sub-package contains the relative modules. A minimal example of the use of PEC in Mitiq can be found in the :ref:`Getting Started section <pec_getting_started>` of the guide, while more details on the theory and limitation of this technique can be found in
the :ref:`About Error Mitigation <guide_qem_pec>` section.


.. figure:: ../img/pec_workflow2_steps.png
  :width: 400
  :alt: The PEC workflow in Mitiq is divided in two steps: Generating circuits and then performing the inference to obtain a noise mitigated expectation value.
  :name: figpec

  The diagram illustrates the typical workflow for probabilistic error cancellation (PEC) in Mitiq. The PEC workflow in Mitiq is divided in two steps: Generating circuits and then performing the inference to obtain a noise mitigated expectation value.


As visible in :ref:`the PEC workflow Figure<figpec>`, the application of PEC in Mitiq is divided in two main steps, similarly to ZNE: The first one involves generating circuits, while the second one involves performing the inference to obtain a noise mitigated expectation value. Through the application of :py:mod:`~mitiq.pec.pec`'s :py:func:`~mitiq.pec.pec.execute_with_pec`, the user thus effectively launches on the quantum hardware (or simulator) a batch of probabilistically sampled circuits. The noisy results from this execution are then used to infer an unbiased estimate of the mitigated expectation value. Differently from the :ref:`ZNE <guide_zne>` techniques, PEC on one hand does not require noise scaling, on the other hand requires knowledge of the noise model (or the tomography of the gates), for the correct representation on the noisy basis.

A tutorial on PEC, divided in four easy tasks to tackle, is available :ref:`here<label-pec-example>`.
