.. mitiq documentation file

.. _release:


================================================
Core Developers' Reference: Making a New Release
================================================

.. note::
	These instructions are aimed at the mantainers of the ``mitiq`` library.

When the time is ready for a new release, follow the checklist and instructions of this document to go through all the steps below:

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Make sure that the changelog is updated
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Make sure that the changelog is updated at ``mitiq/CHANGELOG.md`` and if not, add the latest merged pull requests (PRs), including author and PR number (@username, gh-xxx).

^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Work in a siloed environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Create a conda environment

.. code-block:: shell-session

	$ conda create -n mitiqenv
	$ conda activate mitiqenv


^^^^^^^^^^^^^^^^^^^
Create a new branch
^^^^^^^^^^^^^^^^^^^
- Create a branch in `git` for the documentation with the release number up to
minor (e.g., v.0.0.2--->v00X)

.. code-block:: shell-session

	$(mitiqenv) git checkout -b v00X

You will then open a pull request which will be merged into the ``master`` branch.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Generate the html tree and the pdf file for the documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- To create the html structure

.. code-block:: shell-session

	$ make html


and for the pdf,

.. code-block:: shell-session

	$ make latexpdf

Since the `docs/build` folder is not kept track of, copy the pdf file
with the documentation from `docs/build/latex` to the `docs/pdf` folder, saving it as `mitiq.pdf`, thus replacing the previous version.
Add a copy of the pdf file by naming it according to the release version with major and minor, e.g., `mitiq-0.1.pdf` in the same folder.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Create a distribution locally
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Go to your local directory from the Terminal, e.g., ``github/mitiq/`` and there run

.. code-block:: shell-session

	$ python setup.py sdist bdist_wheel

This will create a source distribution and a built distribution with wheel. This should create a ``build/`` and ``sdist/`` folder.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Release the new version on Github
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. note::
	You need to be a write access of the ``mitiq``'s' Github repository to make a new release.

Make a new release on Github from the website, at https://github.com/unitaryfund/mitiq/releases.

	- Choose a tag and add information on the release using the same style used for previous releases.

	- Make sure that a branch with the version number has been created and is referenced for the given release.

	- Github will create compressed files with the repository. Upload the ``mitiq.pdf`` file and add the locally generated distribution and the Python wheels.

	- Add a brief text with a description of the version release.


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Release the new versionon TestPyPI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before uploading the package on PyPI, since that action cannot be undone, it is good practice to upload it on the test channel TestPyPI.

.. note::
	You need to be a registered user on TestPyPI and a mantainer of the ``mitiq`` project in order to be able to upload the package.

- Upload the package. In order to upload it, you need to have ``twine``, which can be installed with ``pip install twine``. Go to the ``mitiq`` directory, after having created the source distribution version ``sdist``, and simply type

.. code-block:: shell-session

	$ twine upload --repository testpypi dist/*

You can then check at https://test.pypi.org/project/mitiq that the library is correctly uploaded.


- In order to check that the distribution runs correctly, set up a new (conda) environment and try to install the library, for example for version 0.1a1 this is done with:

.. code-block:: shell-session

	$ pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.python.org/simple/ mitiq==0.1a1

The ``--extra-index-url`` is necessary since otherwise ``TestPyPI``  would be looking for the required dependencies therein, but we want it to install them from the real PyPI channel.


^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Release on PyPI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::
	You need to be a registered user on PyPI and a mantainer of the ``mitiq`` project in order to be able to upload the package.

If you already created the source distribution and wheels and tested it on TestPyPI, then you need to just type from bash, in your local ``mitiq`` root directory

.. code-block:: shell-session

	$ twine upload dist/*

You will be prompted to insert your login credentials (username and password). You can then verify the upload on https://pypi.org/project/mitiq/.


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Release the new documentation on Read the Docs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::
	You need to be a registered user at Read the Docs and a mantainer for the ``mitiq`` project on the Read the Docs website in order to be able to update the documentation. You can ensure that you have the permissions for ``mitiq`` at https://readthedocs.org/dashboard/.

Once the relative pull request is merged, the `latest` documentation will be updated on the Read the Docs website, https://mitiq.readthedocs.io/. Ensure that the branch for the new release, as well as the branch relative to the previous release, are tagged in the project overview, activating the relative version
https://readthedocs.org/projects/mitiq/versions/.


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Bump the next version in the master
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Open a pull request bumping up the version number to development mode, e.g., if you just released 0.1.0, update the ``version.txt`` file to 0.1.1dev.

