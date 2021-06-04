.. mitiq documentation file

.. _release:

==============
Mitiq Git flow
==============

The basic development workflow for Mitiq is done in units of milestones.
These are tracked in the GitHub milestone feature and all issues that are
planned to be addressed in the current milestone should be tagged with the
proper milestone.

All releases for Mitiq are recorded on the ``release`` branch with tags for
the version number of the release.
Development work is done on separate branches and forks that get merged into
``master`` when they are ready to be included in the next release.

The main steps of our git flow are as follows:
- Feature work and bug fixes are done on branches (external contributors should fork and then work on branches)
- Once work is ready for review and inclusion in a release, make a PR from the branch/fork to master on the Mitiq repo.
- PRs are then reviewed by the team and the community and then merged into master as appropriate. This means that this feature/fix will be included in the next release.
- When it is time to make a release, a PR is made from the master branch to the release branch and final automatic testing and manual review is done to make sure it is good to be released.
- Once the code is ready to be released, the PR from Master to release is approved and a tag is created on release for the appropriate semantic version number.

================================
Releasing a new version of Mitiq
================================

.. note::
    These instructions are for Mitiq maintainers. Nigtly builds of the Mitiq
    package are uploaded to TestPyPI automatically.

When the time is ready for a new release, follow the checklist and
instructions of this document to go through all the steps below:

.. contents::
   :local:
   :depth: 3

--------------------------------
Make a PR from Master to Release
--------------------------------

The start of any release is drafting a PR from the master branch to the
release branch. This will trigger a complete round of tests to make sure the
code for the release passes them
If you have to update the changelog and the version number, do so as a
part of this PR.

^^^^^^^^^^^^^^^^^^^^
Verify the changelog
^^^^^^^^^^^^^^^^^^^^

This task has two parts:
1. Make sure that ``CHANGELOG.md`` has an entry for each pull request (PR)
since the last release (PRs). These entries should contain a short description
of the PR, as well as the author username and PR number in the form
(@username, gh-xxx).
2. The release author should add a "Summary" section with a couple sentences
describing the latest release, and then update the title of the release
section to include the release date and remove the "In Development"
designation.

^^^^^^^^^^^^^^^^^^^^^^^^^^^
Bump version in VERSION.txt
^^^^^^^^^^^^^^^^^^^^^^^^^^^

When releasing a new version, one must of course update the ``VERSION.txt``
file which is the single source of truth for version information. We try to
follow SemVer, so typically a release will involve changing the version from
``vX.Y.Zdev`` (development) to ``vX.Y.Z`` (released). This will be reflected as
a change in stable release versions from ``vX.(Y-1).Z`` to ``vX.Y.Z``,
in the case of a MINOR version increase.

----------------
Create a new tag
----------------

Once the PR to ``release`` is approved, tag the new commit on release
(using ``git tag``) with a tag that matches the number ``VERSION.txt``
(with a preceding "v", so ``0.1.0`` is ``v0.1.0``) and push this tag to the
Github repository.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Release the new version on Github
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::
    You need to have write access to the Mitiq Github repository to make
    a new release.

There should be a new draft release created by the tag you made in the previous step
`here <https://github.com/unitaryfund/mitiq/releases>`__. You will need to
review it and publish the release.

    - Github will create compressed files with the repository.

.. note::
    If all the above steps have been successfully completed,
    ReadTheDocs (RTD) will automatically build new ``latest`` and ``stable`` versions
    of the documentation. So, no additional steps are needed for updating RTD. You can
    verify changes have been updating by viewing `<https://mitiq.readthedocs.io/>`__.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Release the new version on Github
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::
    You need to have write access to the Mitiq Github repository to make
    a new release.

------------------------------------------------
Update the new development version
------------------------------------------------

Add a new section to the ``CHANGELOG.md`` to track changes in the following
release, meaning that if ``vX.Y.Z`` was just released, then there should be
a section for ``vX.(Y+1).Z`` that is marked "In Development". Also, change the
version in the ``VERSION.txt`` file from ``vX.Y.Z`` to ``vX.(Y+1).Zdev``.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Release the new version on PyPI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::
    There is a GitHub action in the ``.github`` folder to automatically upload Mitiq
    to PyPI. You can check the release status [here](https://pypi.org/project/mitiq/#history).

In case the action for the automatic release on PyPI fails, the commands to release Mitiq are

```
        python -m pip install --upgrade pip
        make install requirements
        pip install setuptools wheel twine
        python setup.py sdist bdist_wheel
        twine upload dist/*
```


.. note::
    You need to be a registered maintainer of Mitiq project on PyPI to upload
    a new release on PyPI from your local machine.

=========================
Releasing a version patch
=========================

The steps for the patch should be basically identical to a release other than cherry-picking from master which commits to make part of the PR from master to release, and the version number selected.
