.. mitiq documentation file

.. _release:

==============
Mitiq Git flow
==============

The basic development workflow for Mitiq is done in units of milestones.
These are tracked in the GitHub milestone feature and all issues that are
planned to be addressed in the current milestone should be tagged with the
proper milestone.

All releases for Mitiq are tagged on the ``master`` branch with tags for the 
version number of the release.
Development work is done on separate branches and forks that get merged into
``master`` when they are ready to be included in the next release.

The main steps of our git flow are as follows:
- Feature work and bug fixes are done on branches (external contributors should fork and then work on branches on their fork)
- Once work is ready for review and inclusion in a release, make a PR from the branch/fork to master on the Mitiq repo. Squash merges are preferred.
- PRs are then reviewed by the team and the community and then merged into master as appropriate. This means that this feature/fix will be included in the next release.
- When it is time to make a release, a PR is made from the master branch to the release branch and final automatic testing and manual review is done to make sure it is good to be released.
- Once the code is ready to be released, the PR from Master to release is approved and a tag is created on release for the appropriate semantic version number.

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

-------------------------
Prepare the master branch
-------------------------

The start of any release is drafting the changelog and bumping the version
number.

^^^^^^^^^^^^^^^^^^^^
Update the changelog
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

When releasing a new version, one must update the ``VERSION.txt``
file which is the single source of truth for version information. We follow 
SemVer, so typically a release will involve changing the version from
``vX.Y.Zdev`` (development) to ``vX.Y.Z`` (released).

--------------
Do the release
--------------

^^^^^^^^^^^^^^^^
Create a new tag
^^^^^^^^^^^^^^^^

Once the above changes (new changelog and new version) are merged into the master branch, checkout and pull the
latest on the master branch from your local machine. Then once you are up to date, tag the most recent
commit on master (using ``git tag``) with a tag that matches the number ``VERSION.txt``
(with a preceding "v", so ``0.1.0`` is ``v0.1.0``) and push this tag to the
Github repository.

.. code-block:: bash

    git tag v0.1.0
    git push origin v0.1.0


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Release the new version on Github
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::
    You need to have write access to the Mitiq Github repository to make
    a new release.

There should be a new draft release on GitHub created by the `gh-release` 
action, triggered by the tag you made in the previous step
`here <https://github.com/unitaryfund/mitiq/releases>`__. You will need to
review it and publish the release.

    - GitHub will create compressed files with the repository.
    - GitHub adds the full changelog in the draft release. Please keep the content related to the new release and remove the content related to previous releases.

.. note::
    If all the above steps have been successfully completed,
    ReadTheDocs (RTD) will automatically build new ``latest`` and ``stable`` versions
    of the documentation. So, no additional steps are needed for updating RTD. You can
    verify changes have been updating by viewing `<https://mitiq.readthedocs.io/>`__.
    Note that this may require a significant amount of time. You can check the
    build status `here <https://readthedocs.org/projects/mitiq/builds/>`__ 

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Release the new version on PyPI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once the GitHub release is published, the release is also published on PyPI 
by the `publish-pypi` action. This may require a few minutes. 
If it seems like it didn't push a new version to PyPI, you can trigger it manually.
Go to `<https://github.com/unitaryfund/mitiq/actions/workflows/publish-pypi.yml>`__ and use
the "Run Workflow" button to publish the new version on PyPI.

In case the action for releasing on PyPI fails, the Python commands to release Mitiq are:

.. code-block:: bash

    python -m pip install --upgrade pip
    make install requirements
    pip install setuptools wheel twine
    python setup.py sdist bdist_wheel
    twine upload dist/*


.. note::
    You need to be a registered maintainer of Mitiq project on PyPI to upload
    a new release on PyPI from your local machine.

------------------------------------------------
Update the new development version
------------------------------------------------

Add a new section to the ``CHANGELOG.md`` to track changes in the following
release, meaning that if ``vX.Y.Z`` was just released, then there should be
a section for ``vX.(Y+1).Z`` that is marked "In Development". Also, change the
version in the ``VERSION.txt`` file from ``vX.Y.Z`` to ``vX.(Y+1).0dev``.

=========================
Releasing a version patch
=========================

The steps for the patch should be basically identical to a release, however,
the commits for the patch should be pushed/cherry-picked onto a branch that
starts from the tag of the version it is patching. So if you had just made the
3.14.0 release (which would have a tag on ``master``) then you would want to
make a branch from that tag called v3.14.0 and then cherry-pick the commits
you need for the patch to that branch. Once the state of that branch reflects
the changes you need including updating the change log and version number, tag
the branch with the appropriate version tag and then review the auto-generated
GitHub release.

Now, there is history that is on this patch branch that is not on ``master``, 
so it is up to the maintainers to make sure that history is merged back into
``master``. This could be done by simply merging the branch back into ``master``,
and then resolving any conflicts. Maybe the changes are only relevant for that
version and are superseded by the next version, so only merging the changes in
the change log are all that are needed to be merged back.
