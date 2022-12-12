# Contributing to Mitiq

All contributions to this project are welcome, and they are greatly appreciated; every little bit helps.
The two most common ways to contribute here are
1. opening an [issue](https://github.com/unitaryfund/mitiq/issues/new) to report a bug or propose a new feature, or ask a question, and
2. opening a [pull request](https://github.com/unitaryfund/mitiq/pulls) to fix a bug, or implement a desired feature.

That said, feel free to open an issue to ask a question, provide general feedback, etc.

The rest of this document describes the technical details of getting set up to develop, and make your first contribution to Mitiq.


## Development environment

1. Ensure you have python 3.8 or greater installed. If not, you can find the downloads [here](https://www.python.org/downloads/). 
2. Set up a virtual environment to isolate dependencies. This can be done with many different tools including [Virtualenv](https://virtualenv.pypa.io/en/latest/), [Pipenv](https://pypi.org/project/pipenv/), [Poetry](https://python-poetry.org/), and [Anaconda](https://www.anaconda.com/products/distribution). In what follows we will use Anaconda, but if you're familiar with other tools feel free to use those.
3. Set up a local version of the [Mitiq repository](https://github.com/unitaryfund/mitiq). To do this you will need to use `git` which is a version control system. If you're unfamiliar, check out the [docs](https://git-scm.com/), and learn about what the typical [`git` workflow](https://www.asmeurer.com/git-workflow/) looks like.
4. Inside the Mitiq repository (`cd mitiq`), activate a virtual environment. With conda this is done using the following command.
```
conda create --name myenv python=3
conda activate myenv
```
5. Install the dependencies. First, to get an updated version of [`pip`](https://pypi.org/project/pip/) inside the virtual environment run `conda install pip` followed by
```
pip install -e .
pip install -r requirements.txt
```
6. You should now have a development environment set up to work on Mitiq! 🎉 To go forward with making the desired changes, please consult the ["Making changes" section](https://www.asmeurer.com/git-workflow/#making-changes) of the `git` workflow article. If you've encountered any problems thus far, please let us know by opening an issue! More information about workflow can be found below in the [lifecycle](#lifecycle) section.

What follows are recommendations/requirements to keep in mind while contributing.

## Making changes

### Adding tests
When modifying and/or adding new code it is important to ensure the changes are covered by tests.
Test thoroughly, but not excessively.
Mitiq uses a nested structure for packaging tests in directories named `tests` at the same level of each module.
The only except to this is that any tests requiring a QVM should be placed in the `mitiq_pyquil/tests` folder.

### Running tests

After making changes, please ensure your changes pass all the existing tests (and any new tests you've added).
You can run the tests using the `pytest` CLI, but the [Makefile][makefile] contains many of the common commands you'll need to ensure your code is aligned with our standards.

For example, to run all the tests that do not require a pyQuil QVM, run
```bash
make test
```
This is typically suitable for most development tasks and is the easiest, and most common way to test.

To run the tests for the pyQuil plugins, run
```bash
make test-pyquil
```

To run all tests, run
```bash
make test-all
```

**Note**: For the pyQuil tests to run, you will need to have QVM & quilc servers
running in the background. The easiest way to do this is with [Docker](https://www.docker.com/) via

```bash
docker run --rm -idt -p 5000:5000 rigetti/qvm -S
docker run --rm -idt -p 5555:5555 rigetti/quilc -R
```

If you've modified any docstrings/added new functions, run `make doctest` to ensure they are formatted correctly.
You may need to run `make docs` before you are able to run `make doctest`. 

### Updating the documentation
Follow these [instructions for contributing to the documentation](https://mitiq.readthedocs.io/en/latest/contributing_docs.html) which include guidelines about updating the API-doc list of modules and writing examples in the users guide.

### Style guidelines

Mitiq code is developed according the best practices of Python development.
* Please get familiar with [PEP 8](https://www.python.org/dev/peps/pep-0008/) (code) and [PEP 257](https://www.python.org/dev/peps/pep-0257/) (docstrings) guidelines.
* Use annotations for type hints in the objects' signature.
* Write [google-style docstrings](https://google.github.io/styleguide/pyguide.html#383-functions-and-methods).

We use [Black](https://black.readthedocs.io/en/stable/index.html) and `flake8` to automatically lint the code and enforce style requirements as part of the CI pipeline.
You can run these style tests yourself locally in the top-level directory of the repository.

You can check for violations of the `flake8` rules with
```bash
make check-style
```
In order to check if `black` would reformat the code, use
```bash
make check-format
```
If above format check fails then you will be presented with a [diff](https://black.readthedocs.io/en/stable/usage_and_configuration/the_basics.html#diffs) which can be resolved by running
```bash
make format
```

If you aren't presented with any errors, then that means your code is ready to commit!

## Proposing a new feature to Mitiq
If you are interested in adding a substantial new feature or functionality to Mitiq, please make a copy of our Request For Comments (RFC) [template](https://docs.google.com/document/d/1adomheXpbqp4YIBFQ49IsAJzuJKWyr75GRO1NeWg0Fo/) and fill out the details of your enhancement proposal.
Take a look at [previous RFCs](#list-of-accepted-rfcs) for examples on how to fill out your proposal.
Once you have completed your proposal, [create a feature request issue](https://github.com/unitaryfund/mitiq/issues/new?assignees=&labels=feature-request&template=feature_request.md&title=) and add a link to your proposal document (make sure to enable commenting on the RFC!).
For any part of the template that you weren't able to complete please mention that in the issue description.

### List of accepted RFCs
This is a list of accepted request-for-comments (RFC) documents by date of creation (reverse chronological order):

- [Implementation RFC for Mitiq calibration](https://docs.google.com/document/d/1EZUJyEEUQUH33UOgSIzCCvXyxP0WLOQn11W0x4Ox4nY/edit) by Andrea Mari (@andreamari) Nov 2, 2022
- [Calibration tools for error mitigation RFC (abstract general solutions)](https://docs.google.com/document/d/1otUHnTlyNS-0rxGAxltHLF1iD5C9qT9oEZ3jn8VHWgw/edit) by Andrea Mari (@andreamari) Oct 6, 2022
- [Identity insersion scaling RFC](https://docs.google.com/document/d/1hbd9frjYiSy0WujA0iCccc-oMO4Q-kZc2G4b3lkJHdk/edit) by Purva Thakre (@purva-thakre) Jun 29, 2022
- [Readout Confusion Inversion RFC](https://docs.google.com/document/d/1buO5PrO5sS02VXjcaYf37RuR0rF6xpyr4J9H1tI4vN4/edit) by Amir Ebrahimi (@amirebrahimi) Jun 16, 2022
- [Documentation reorganization RFC](https://docs.google.com/document/d/13un5TZPknSOhmOBkrL2rsofjGfdp2jDnd-DywLpGFPc/edit) by Ryan LaRose (@rmlarose) Dec 1, 2021
- [Learning-based PEC RFC](https://docs.google.com/document/d/1VItesy6R5SlUa_YXW1km7IjFZ8kzyFeHUepHak1fEh4/edit) by Misty Wahl (@Misty-W) Oct 25, 2021
- [Digital dynamical decoupling RFC](https://docs.google.com/document/d/1cRwFCTn6kUjI1P0kNydtevxIYtE4r8Omd_iWK0Pe8qo/edit) by Aaron Robertson (@Aaron-Robertson) Jan 28, 2021


## Code of conduct
Mitiq development abides to the [Contributors' Covenant](https://mitiq.readthedocs.io/en/latest/code_of_conduct.html).

## Lifecycle
The basic development workflow for Mitiq is done in units of milestones which are usually one month periods where we focus our efforts on thrusts decided by the development team, alongside community members.
Milestones are tracked using the [GitHub milestone feature](https://github.com/unitaryfund/mitiq/milestones) and all issues that are planned to be addressed should be tagged accordingly.

All releases for Mitiq are tagged on the `master` branch with tags for the version number of the release.
Find all the previous releases [here](https://github.com/unitaryfund/mitiq/releases).

---
### Special Note for Windows Users Using Python 3.8:
To prevent errors when running `make docs` and `make doctest`, Windows developers using Python 3.8 will also need to edit `__init__.py` in their environment's asyncio directory.
This is due to Python changing `asyncio`'s [default event loop in Windows beginning in Python 3.8](https://docs.python.org/3/library/asyncio-policy.html#asyncio.DefaultEventLoopPolicy).
The new default event loop will not support Unix-style APIs used by some dependencies.
1. Locate your environment directory (likely `C:\Users\{username}\anaconda3\envs\{your_env}`), and open `{env_dir}/Lib/asyncio/__init__.py`.
2. Add `import asyncio` to the file's import statements.
3. Find the block of code below and replace it with the provided replacement.
    * Original Code

          if sys.platform == 'win32':  # pragma: no cover
              from .windows_events import *
              __all__ += windows_events.__all__
          else:
              from .unix_events import *  # pragma: no cover
              __all__ += unix_events.__all__

    * Replacement Code

          if sys.platform == 'win32':  # pragma: no cover
              from .windows_events import *
              asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
              __all__ += windows_events.__all__
          else:
              from .unix_events import *  # pragma: no cover
              __all__ += unix_events.__all__

## Code of conduct
Mitiq development abides to the [Contributors' Covenant](https://mitiq.readthedocs.io/en/latest/code_of_conduct.html).

[makefile]: https://github.com/unitaryfund/mitiq/blob/master/Makefile
