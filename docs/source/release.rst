.. mitiq documentation file

.. _release:


================================================
Core Developers' Reference: Making a New Release
================================================

These instructions are aimed at the mantainers of the ``mitiq`` library.
When the time is ready for a new release, follow the checklist and instructions of this document to go through all the steps:


^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Work in a siloed environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Create a conda environment

.. code-block:: bash
	conda create -n mitiqenv
	conda activate mitiqenv


^^^^^^^^^^^^^^^^^^^
Create a new branch
^^^^^^^^^^^^^^^^^^^
- Create a branch in `git` for the documentation with the release number up to
minor (e.g., v.0.0.2--->v00X)

.. code-block:: bash
	(mitiqenv) git checkout -b v00X

You will then open a pull request which will be merged into the ``master`` branch.

------------------------------------------------
How to Make a New Release of the Code
------------------------------------------------

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Generate the html tree and the pdf file for the documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- To create the html structure

.. code-block:: bash
	make html


and for the pdf,

.. code-block:: bash
	make latexpdf

Since the `docs/build` folder is not kept track of, copy the pdf file
with the documentation from `docs/build/latex` to the `docs/pdf` folder, saving it as `mitiq.pdf`.
Add a copy by naming it according to the release version with major and minor, e.g., `mitiq-0.1.pdf` in the same folder.


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Release the new version on Github
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
You need to be a mantainer of the Github repository. Make a new release on Github from the website, at https://github.com/unitaryfund/mitiq/releases. Choose a tag and  add information on the release using the same style used for previous releases.


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Release the new versionon TestPyPI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before uploading the package on PyPI, since that action cannot be undone, it is good practice to upload it on the test channel TestPyPI.

You need to be a registered user on TestPyPI and a mantainer of the project in order to be able to upload the package.

- Upload the package. In order to upload it, you need to have ``twine``, which can be installed with ``pip install twine``. Go to the ``mitiq`` directory, after having created the source distribution version ``sdist``, and simply type

.. code-block:: bash
	twine


- You can verify the upload on https://test.pypi.org/project/mitiq

- In order to check that the distribution run correctly, set up a new (conda) environment and try to install the library, for example for version 0.1a1 this is done with:

.. code-block:: bash
	pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.python.org/simple/ mitiq==0.1a1

The ``--extra-index-url`` is necessary since otherwise ``TestPyPI``  would be looking for the required dependencies therein, but we want it to install them from the real PyPI channel.


^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Release on PyPI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you already created the source distribution and wheels and tested it on TestPyPI, then you need to just type from bash, in your local ``mitiq`` root directory

.. code-block:: bash
	twine upload dist/*

You will be prompted to insert your login credentials (username and password). You can then verify the upload on https://pypi.org/project/mitiq/.


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Release the new documentation on Read the Docs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once the relative pull request is merged, the `latest` documentation will be updated on the Read the Docs website, https://mitiq.readthedocs.io/. Ensure that the branch for the new release, as well as the branch relative to the previous release, are tagged in the project overview, activating the relative version
https://readthedocs.org/projects/mitiq/versions/.


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Bump the next version in the master
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Open a pull request bumping up the version number to development mode, e.g., if you just released 0.1.0, update the ``version.txt`` file to 0.1.1dev.

