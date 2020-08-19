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
(myenv) pip install -r requirements.txt
```

### Adding tests
If you add new features to a function or class, it is strongly encouraged to add tests for such object. Mitiq uses a nested structure for packaging tests in directories named `tests` at the same level of each module.

### Updating the documentation
Follow the guidelines in the Contributing to docs [instructions](https://mitiq.readthedocs.io/en/latest/read_README-docs.html) (look here on [GitHub](https://github.com/unitaryfund/mitiq/blob/master/docs/README-docs.md)), which include guidelines about updating the API-doc list of modules and writing examples in the users guide.

### Checking local tests

You can check that tests run with `pytest`. The [`Makefile`](https://github.com/unitaryfund/mitiq/blob/master/Makefile) contains
some commands for running different collections of tests for the repository.

To run just the tests contained in `mitiq/tests` and `mitiq/benchmarks/tests` run

```bash
(myenv) make test
```

To run the tests for the pyQuil and Qiskit plugins (which of course require for
pyQuil and Qiskit to be installed) run

```bash
(myenv) make test
```

*NOTE*: For the pyQuil tests to run, you will need to have QVM & quilc servers
running in the background. The easiest way to do this is with Docker via

```bash
docker run --rm -idt -p 5000:5000 rigetti/qvm -S
docker run --rm -idt -p 5555:5555 rigetti/quilc -R
```

You can also check that all tests run also in the documentation examples and
docstrings with

```bash
(myenv) make docs
```

If you add new `/tests` directories, you will need to update the `Makefile`
so that they will be included as part of continuous integration.

### Style Guidelines

Mitiq code is developed according the best practices of Python development.
* Please get familiar with [PEP 8](https://www.python.org/dev/peps/pep-0008/) (code)
  and [PEP 257](https://www.python.org/dev/peps/pep-0257/) (docstrings) guidelines.
* Use annotations for type hints in the objects' signature.
* Write [google-style docstrings](https://google.github.io/styleguide/pyguide.html#doc-function-args).

We use [Black](https://black.readthedocs.io/en/stable/index.html) and `flake8` to automatically
lint the code and enforce style requirements as part of the CI pipeline. You can run these style
tests yourself locally by running `make check-style` (to check for violations of the `flake8` rules)
and `make check-format` (to see if `black` would reformat the code) in the top-level directory of
the repository. If you aren't presented with any errors, then that means your code is good enough
for the linter (`flake8`) and formatter (`black`). If `make check-format` fails, it will present
you with a diff, which you can resolve by running `make format`. Black is very opinionated, but
saves a lot of time by removing the need for style nitpicks in PR review. We only deviate from its
default behavior in one category: we choose to use a line length of 79 rather than the Black
default of 88 (this is configured in the [`pyproject.toml`](pyproject.toml) file).

### Code of conduct
Mitiq development abides to the [Contributors' Covenant](CODE_OF_CONDUCT.md).
