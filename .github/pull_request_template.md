Description
-----------

Please explain the changes you made here.

Checklist
---------

Check off the following once complete (or if not applicable) after opening the PR. The PR will be reviewed once this checklist is complete and all tests are passing.

- [ ] I added unit tests for new code.
- [ ] I used [type hints](https://www.python.org/dev/peps/pep-0484/) in function signatures.
- [ ] I used [Google-style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) docstrings for functions.
- [ ] I [updated the documentation](../blob/master/docs/CONTRIBUTING_DOCS.md) where relevant.
- [ ] Added myself / the copyright holder to the AUTHORS file

If some items remain, you can mark this a [draft pull request](https://github.blog/2019-02-14-introducing-draft-pull-requests/).

License
-------

- [ ] I license this contribution under the terms of the GNU GPL, version 3 and grant Unitary Fund the right to provide additional permissions as described in section 7 of the GNU GPL, version 3.


Tips
----

- If the validation check fails:

    1. Run `make check-types` (from the root directory of the repository) and fix any [mypy](https://mypy.readthedocs.io/en/stable/) errors.

    2. Run `make check-style` and fix any [flake8](http://flake8.pycqa.org) errors.

    3. Run `make format` to format your code with the [black](https://black.readthedocs.io/en/stable/index.html) autoformatter.

  For more information, check the [Mitiq style guidelines](https://mitiq.readthedocs.io/en/stable/contributing.html#style-guidelines).

- Write "Fixes #XYZ" in the description if this PR fixes Issue #XYZ.
