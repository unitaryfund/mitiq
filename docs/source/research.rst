.. mitiq documentation file

.. _research:

========
Research
========

Mitiq is designed to aid researchers in quantum computing and quantum error mitigation. Given the quick growth of the techniques for quantum error mitigation (you can read an overview in the `Users Guide <https://mitiq.readthedocs.io/en/stable/guide/guide-error-mitigation.html>`_ ), it is natural to update the toolchain with new techniques and features.

If you'd like to contribute new features to Mitiq, discuss it in an issue and once ready to upload the code, review the `contributing <contributing.html>`_ guidelines for the steps to take. If you have an example of using Mitiq with other software packages, or on a specific problem, we'd be glad to add it to the `Examples <examples/examples.html>`_ section of the documentation: please review the `contributing to the documentation <contributing_docs.html>`_ instructions.


.. _citing:

------------
Citing Mitiq
------------

If you are using Mitiq for your research, please cite the related `white paper <https://arxiv.org/abs/2009.04417>`_:


.. code-block::

	@misc{larose2020mitiq,
	      title={Mitiq: A software package for error mitigation on noisy quantum computers},
	      author={Ryan LaRose and Andrea Mari and Peter J. Karalekas and Nathan Shammah and William J. Zeng},
	      year={2020},
	      eprint={2009.04417},
	      archivePrefix={arXiv},
	      primaryClass={quant-ph}}


You can download the :download:`bibtex file <mitiq.bib>`.


If you have developed new features for error mitigation, or found bugs in Mitiq, please consider `contributing <contributing.html>`_ your code.


.. _code_data_mitiq_paper:

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Data and code supporting the Mitiq paper
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``mitiq-paper`` folder contains the data used in the ZNE plots of the Mitiq `white paper <https://arxiv.org/abs/2009.04417>`_.

You can find the `code snippets <examples/mitiq-paper/mitiq-paper-codeblocks.html>`_ for the codeblocks present in the Mitiq paper in the `Examples` section of the documentation relative to the Mitiq paper section of the documentation.


You can find the raw data supporting the plots in the Mitiq paper in the Github repository, in the `data <https://github.com/unitaryfund/mitiq/tree/master/docs/source/examples>`_ folder. The `raw` folder contains runs from the IBM Q London processor, while the
`qcs-aspen-8-32-33-rb.csv` file contains data from the Rigetti Aspen-8 processor.


.. _cited_by:

----------------------------
Papers citing or using Mitiq
----------------------------

The following papers use or cite Mitiq:

- *"Digital zero noise extrapolation for quantum error mitigation"*, T. Giurgica-Tiron, Y. Hindy, R. LaRose, A. Mari, W. J. Zeng, Proc. IEEE Intl. Conf. Q. Comp. and Eng. (2020), `DOI: 10.1109/QCE49297.2020.00045 <https://ieeexplore.ieee.org/xpl/conhome/9259908/proceeding>`_ `arXiv:2005.10921 <https://arxiv.org/abs/2005.10921>`_

- *"Gutzwiller Hybrid Quantum-Classical Computing Approach for Correlated Materials"*, Y. Yao, F. Zhang, C.-Z. Wang, K.-M. Ho, P. P. Orth, `arXiv:2003.04211 <https://arxiv.org/abs/2003.04211>`_

- *"Extending C++ for Heterogeneous Quantum-Classical Computing"*, T. Nguyen, A. Santana, T. Kharazi, D. Claudino, H. Finkel, A. McCaskey, `arXiv:2010.03935 <https://arxiv.org/abs/2010.03935>`_

- *"QFold: Quantum Walks and Deep Learning to Solve Protein Folding"*, P. A. M. Casares, R. Campos, M. A. Martin-Delgado, `arXiv:2101.10279 <https://arxiv.org/abs/2101.10279>`_

- *"Noisy intermediate-scale quantum (NISQ) algorithms"*, K. Bharti, A. Cervera-Lierta, T. H. Kyaw, *et al.*, `arXiv:2101.08448 <https://arxiv.org/abs/2101.08448>`_

An up-to-date list of papers citing the Mitiq paper can be found on `Semantic Scholar <https://www.semanticscholar.org/paper/Mitiq%3A-A-software-package-for-error-mitigation-on-LaRose-Mari/dc55b366d5b2212c6df8cd5c0bf05bab13104bd7#citing-papers>`_
and on `Google Scholar <https://scholar.google.com/scholar?cites=12810395086731011605>`_.

