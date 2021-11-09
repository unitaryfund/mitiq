# Contributing to Mitiq

Contributions are welcome, and they are greatly appreciated, every little bit helps.

## Opening an issue
You can begin contributing to `mitiq` code by raising an
[issue](https://github.com/unitaryfund/mitiq/issues/new), reporting a bug or
proposing a new feature request, using the labels to organize it.
Please use `mitiq.about()` to document your dependencies and working environment.

## Opening a pull request
You can open a [pull request](https://github.com/unitaryfund/mitiq/pulls) by pushing changes from a local branch, explaining the bug fix or new feature.

### Version control with git
git is a language that helps keeping track of the changes made. Have a look at these guidelines for getting started with [git workflow](https://www.asmeurer.com/git-workflow/).
Use short and explanatory comments to document the changes with frequent commits.

### Forking the repository
You can fork mitiq from the github repository, so that your changes are applied with respect to the current master branch. Use the Fork button, and then use git from the command line to clone your fork of the repository locally on your machine.
```bash
(base) git clone https://github.com/your_github_username/mitiq.git
```
You can also use SSH instead of a HTTPS protocol.

### Working in a virtual environment
It is best to set up a clean environment with anaconda, to keep track of all installed applications.
```bash
(base) conda create -n myenv python=3
```
accept the configuration ([y]) and switch to the environment
```bash
(base) conda activate myenv
(myenv) conda install pip
```
Once you will finish the modifications, you can deactivate the environment with
```bash
(myenv) conda deactivate myenv
```

### Development install
In order to install all the libraries useful for contributing to the
development of the library, from your local clone of the fork, run

```bash
(myenv) pip install -e .
(myenv) pip install -r dev_requirements.txt
```

#### Special Note for Windows Users Using Python 3.8:
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
  

### Adding tests
If you add new features to a function or class, it is required to add tests for such object. Mitiq uses a nested structure for packaging tests in directories named `tests` at the same level of each module.
The only except to this is that any tests requiring a QVM should be placed in the mitiq_pyquil/tests folder.

### Updating the documentation
Follow these [instructions for contributing to the documentation](https://mitiq.readthedocs.io/en/latest/contributing_docs.html) which include guidelines about updating the API-doc list of modules and writing examples in the users guide.

### Running local tests

After making changes, please ensure your changes still pass all the existing tests.
You can check that tests run with `pytest`. The [Makefile][makefile] contains
some commands for running different collections of tests for the repository.

To only run tests that do not require a pyQuil QVM running, run

```bash
(myenv) make test
```

To run the tests for the pyQuil plugins, run

```bash
(myenv) make test-pyquil
```

To run all tests, run

```bash
(myenv) make test-all
```

*NOTE*: For the pyQuil tests to run, you will need to have QVM & quilc servers
running in the background. The easiest way to do this is with Docker via

```bash
docker run --rm -idt -p 5000:5000 rigetti/qvm -S
docker run --rm -idt -p 5555:5555 rigetti/quilc -R
```

Please also remember to check that all tests run also in the documentation examples and
docstrings with

```bash
(myenv) make doctest
```
You may need to run `make docs` before you are able to run `make doctest`. 

### Style Guidelines

Mitiq code is developed according the best practices of Python development.
* Please get familiar with [PEP 8](https://www.python.org/dev/peps/pep-0008/) (code)
  and [PEP 257](https://www.python.org/dev/peps/pep-0257/) (docstrings) guidelines.
* Use annotations for type hints in the objects' signature.
* Write [google-style docstrings](https://google.github.io/styleguide/pyguide.html#doc-function-args).

We use [Black](https://black.readthedocs.io/en/stable/index.html) and `flake8` to automatically
lint the code and enforce style requirements as part of the CI pipeline. You can run these style
tests yourself locally in the top-level directory of
the repository.

You can check for violations of the `flake8` rules with
```bash
(myenv) make check-style
```
In order to check if `black` would reformat the code, use
```bash
(myenv) make check-format
```
If above format check fails then you will be presented with a [diff](https://black.readthedocs.io/en/stable/usage_and_configuration/the_basics.html#diffs) which can be resolved by running
```bash
(myenv) make format
```

 If you aren't presented with any errors, then that means your code is good enough
for the linter (`flake8`) and formatter (`black`). Black is very opinionated, but
saves a lot of time by removing the need for style nitpicks in PR review. We only deviate from its
default behavior in one category: we choose to use a line length of 79 rather than the Black
default of 88 (this is configured in the [`pyproject.toml`](https://github.com/unitaryfund/mitiq/blob/master/pyproject.toml) file).

## Proposing a new feature to Mitiq

If you are interested in adding a larger new feature or functionality to Mitiq, please check out our
Mitq enhancement proposal (MEP) template [`docs/mep/0000-feature-proposal-TEMPLATE.md`](https://github.com/unitaryfund/mitiq/blob/master/docs/mep/0000-feature-proposal-TEMPLATE.md). To help facilitate
discussion about the feature you would like to add, make a copy of the template and increment the proposal
number and change `feature-proposal-TEMPLATE` to a short description of what you are proposing.
Please fill out any relevant sections of that template as best you can and we can discuss in
both PR threads as well as on the [discord](http://discord.unitary.fund).

## Code of conduct
Mitiq development abides to the [Contributors' Covenant](https://mitiq.readthedocs.io/en/latest/code_of_conduct.html).

[makefile]: https://github.com/unitaryfund/mitiq/blob/master/Makefile
