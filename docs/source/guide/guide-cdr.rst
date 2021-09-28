.. cdr:

************************
Clifford Data Regression
************************

Clifford Data Regression (CDR) is a learning-based quantum error mitigation technique. It is based on the


A figure of the typical workflow for CDR in Mitiq is shown in the figure below.

.. figure:: ../img/cdr_workflow2_steps.png
  :width: 400
  :alt: The CDR workflow in Mitiq is divided in two steps: Generating circuits, both for a classical simulator and on the intended backend, and then performing the inference from measurements to obtain a noise mitigated expectation value.
  :name: figcdr


The :ref:`CDR workflow Figure<figcdr>` above shows a schema of the implementation of CDR in Mitiq. Similarly to ZNE and PEC, also CDR in Mitiq is divided in two main stages: The first one of circuit generation and the second for inference of the mitigated value. However, in CDR, the generation of quantum circuits is divided in two steps: Clifford approximations of the actual circuit are first simulated on a classical simulator (Clifford circuits can be efficiently simulated classically). These results are then used as training data for the near-Clifford circuits that are run on the noisy backend (quantum hardware or another noisy simulator).