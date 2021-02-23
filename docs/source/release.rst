.. mitiq documentation file

.. _release:

================================
Releasing a new version of Mitiq
================================

.. note::
    These instructions are for Mitiq maintainers.

When the time is ready for a new release, follow the checklist and
instructions of this document to go through all the steps below:

.. contents::
   :local:
   :depth: 3

^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Work in a siloed environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is recommended that the release is performed in a new, clean virtual
environment, which makes it easier to verify that everything is working
as intended.

.. code-block:: shell-session

    conda create -n mitiqenv
    conda activate mitiqenv

^^^^^^^^^^^^^^^^^^^^
Update the changelog
^^^^^^^^^^^^^^^^^^^^

This task has two parts. One, make sure that ``CHANGELOG.md`` has an entry
for each pull request (PR) since the last release (PRs). These entries should
contain a short description of the PR, as well as the author username and PR
number in the form (@username, gh-xxx). Two, the release author should add
a "Summary" section with a couple sentences describing the latest release,
and then update the title of the release section to include the release
date and remove the "In Development" designation.

^^^^^^^^^^^^^^^^^^^^^^^^^^^
Bump version in VERSION.txt
^^^^^^^^^^^^^^^^^^^^^^^^^^^

When releasing a new version, one must of course update the ``VERSION.txt``
file which is the single source of truth for version information. We try to
follow SemVer, so typically a release will involve changing the version
``vX.Y.Z`` to ``vX.(Y+1).Z``, constituting a MINOR version increase.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Generate the HTML and PDF file for the docs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To create the HTML documentation, run the following from the top-level
directory of the repository:

.. code-block:: shell-session

    make docs

To create the PDF documentation, do the following:

.. code-block:: shell-session

    make pdf

Finally, Since the ``docs/build`` folder is not version controlled, copy the
newly created PDF file from ``docs/build/latex`` to ``docs/pdf`` folder as
``mitiq.pdf`` to overwrite the previous version.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Create a PR with the above changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After the required changes to ``VERSION.txt`` and ``CHANGELOG.md`` have been
made, and the PDF documentation has been generated and moved to the correct
location, it is recommended that the release author make a PR to master with
these changes (rather than pushing directly to master) just in case. After
this PR has been merged, the release author can go to the next step.

^^^^^^^^^^^^^^^^
Create a new tag
^^^^^^^^^^^^^^^^

Tag the new commit to master (using ``git tag``) with a tag that matches the
number ``VERSION.txt`` (with a preceding "v", so ``0.1.0`` is ``v0.1.0``) and
push this tag to the Github repository.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Create a source & built distribution locally
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

From the top-level directory of the repository, run:

.. code-block:: shell-session

    python setup.py sdist bdist_wheel

This will create a "source" distribution and a "built" distribution using
``wheel``. This should create a ``build/`` and ``sdist/`` folder.

**NOTE**: You will need to have installed ``wheel`` to create the "built"
distribution.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Release and test the new version on TestPyPI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before uploading the package on PyPI, since that action cannot be undone, it
is good practice to upload it on the test channel TestPyPI.

.. note::
    You need to be a registered user on TestPyPI and a maintainer of the
    Mitiq project in order to be able to upload the package.

Upload the package. In order to upload it, you need to have ``twine``,
which can be installed with ``pip install twine``. Go to the Mitiq
directory, after having created the source distribution version ``sdist``,
and simply run:

.. code-block:: shell-session

    twine upload --repository testpypi dist/*

You can then check at `here <https://test.pypi.org/project/mitiq>`_ that
the library has been correctly uploaded.

In order to check that the distribution runs correctly, set up a new virtual
environment and try to install the library. For example, for version ``x.y.z``
this is done via:

.. code-block:: shell-session

    pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.python.org/simple/ mitiq==x.y.z

The ``--extra-index-url`` is necessary since otherwise ``TestPyPI``  would be
looking for the required dependencies therein, but we want it to install them
from the real PyPI channel.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Release the new version on PyPI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::
    You need to be a registered user on PyPI and a maintainer of the Mitiq
    project in order to be able to upload the package.

If you already created the source distribution and wheels and tested it on
TestPyPI, then you need to just run the following from the top-level directory
of the Mitiq repository:

.. code-block:: shell-session

    twine upload dist/*

You will be prompted to insert your login credentials (username and password).
You can then verify the upload `here <https://pypi.org/project/mitiq/>`__.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Release the new version on Github
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::
    You need to have write access to the Mitiq Github repository to make
    a new release.

Make a new release on Github
`here <https://github.com/unitaryfund/mitiq/releases>`__.

    - Choose the tag you recently created, and add information on the release
      by pulling from ``CHANGELOG.md`` as in previous releases.
    - Github will create compressed files with the repository. Upload the
      ``mitiq.pdf`` file and add the locally generated distribution tarball and
      wheel.

.. note::
    If all the above steps have been successfully completed,
    ReadTheDocs (RTD) will automatically build new ``latest`` and ``stable`` versions
    of the documentation. So, no additional steps are needed for updating RTD. You can
    verify changes have been updating by viewing `<https://mitiq.readthedocs.io/>`__.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Update the changelog for new development version
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Add a new section to the ``CHANGELOG.md`` to track changes in the following
release, meaning that if ``vX.Y.Z`` was just released, then there should be
a section for ``vX.(Y+1).Z`` that is marked "In Development".
