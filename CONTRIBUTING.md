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
(myenv) pip install -r dev_requirements.txt
```

### Adding tests
If you add new features to a function or class, it is required to add tests for such object. Mitiq uses a nested structure for packaging tests in directories named `tests` at the same level of each module.

### Updating the documentation
Follow these [instructions for contributing to the documentation](https://mitiq.readthedocs.io/en/latest/contributing_docs.html) which include guidelines about updating the API-doc list of modules and writing examples in the users guide.

### Checking local tests

You can check that tests run with `pytest`. The [Makefile][makefile] contains
some commands for running different collections of tests for the repository.

To run just the tests contained in `mitiq/tests` and `mitiq/benchmarks/tests` run

```bash
(myenv) make test
```

To run the tests for the pyQuil and Qiskit plugins (which of course require for
pyQuil and Qiskit to be installed) run

```bash
(myenv) make test-pyquil
(myenv) make test-qiskit
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

A code block [can be created](https://docs.github.com/en/github/writing-on-github/creating-and-highlighting-code-blocks) using triple backticks ([or indenting with either four spaces or a tab](https://www.markdownguide.org/basic-syntax#code-blocks))  before and after the code with a language identifier for code syntax highlighting.

We use [Black](https://black.readthedocs.io/en/stable/index.html) and `flake8` to automatically
lint the code and enforce style requirements as part of the CI pipeline. You can run these style
tests yourself locally by running `make check-style` (to check for violations of the `flake8` rules)
and `make check-format` (to see if `Black` would reformat the code) in the top-level directory of
the repository. If you aren't presented with any errors, then that means your code is good enough
for the linter (`flake8`) and formatter (`Black`). If `make check-format` fails, it will present
you with a diff, which you can resolve by running `make format`. Black is very opinionated, but
saves a lot of time by removing the need for style nitpicks in PR review. We only deviate from its
default behavior in one category: we choose to use a line length of 79 rather than the Black
default of 88 (this is configured in the [`pyproject.toml`](https://github.com/unitaryfund/mitiq/blob/master/pyproject.toml) file).

#### Some possible issues with `Black`
Below is a summarized list of some issues leading to errors. A more detailed discussion can be found [in the documentation](https://black.readthedocs.io/en/stable/the_black_code_style.html#the-black-code-style).
- In general, `Black` strives for one expression or statement per line with uniform vertical and horizontal whitespace. This is described as `Black` doing everything to keep "[`pycodestyle` happy](https://black.readthedocs.io/en/stable/the_black_code_style.html#how-black-wraps-lines)". A list of errors due to `pycodestyle` can be found [here](https://pycodestyle.pycqa.org/en/latest/intro.html#error-codes).
- When a statement or expression is moved to its own line, a [trailing comma](https://black.readthedocs.io/en/stable/the_black_code_style.html#trailing-commas) will be added.
- Because uniform vertical whitespace is enforced, empty lines are only allowed in a [handful of scenarios](https://black.readthedocs.io/en/stable/the_black_code_style.html#empty-lines) like inside, before and after function definitions etc.
- [Default line length](https://black.readthedocs.io/en/stable/the_black_code_style.html#line-length) for `Black` is 88 characters per line. This number can be adjusted to have fewer or more characters as needed. As stated above, default length for `Mitiq` is 79 characters.
- If an expression or a statement can be fit in one line, trailing comma introduced before a line break will be removed along with the [parentheses](https://black.readthedocs.io/en/stable/the_black_code_style.html#parentheses).
- If backslashes are used for line breaks, these will be replaced with parentheses in keeping with [PEP 8 style guide](https://www.python.org/dev/peps/pep-0008/#maximum-line-length).
- To obey [PEP 8 style guide for improving readability](https://www.python.org/dev/peps/pep-0008/#should-a-line-break-before-or-after-a-binary-operator), a line break will be introduced before a binary operator.
- Use double quotes for [strings](https://black.readthedocs.io/en/stable/the_black_code_style.html#strings).


## Proposing a new feature to Mitiq

If you are interested in adding a larger new feature or functionality to Mitiq, please check out our
Mitq enhancement proposal (MEP) template [`docs/mep/0000-feature-proposal-TEMPLATE.md`](https://github.com/unitaryfund/mitiq/blob/master/docs/mep/0000-feature-proposal-TEMPLATE.md). To help facilitate
discussion about the feature you would like to add, make a copy of the template and increment the proposal
number and change `feature-proposal-TEMPLATE` to a short description of what you are proposing.
Please fill out any relevant sections of that template as best you can and we can discuss in
both PR threads as well as on the [discord](https://discord.unitary.fund).

## Code of conduct
Mitiq development abides to the [Contributors' Covenant](https://mitiq.readthedocs.io/en/latest/code_of_conduct.html).

[makefile]: https://github.com/unitaryfund/mitiq/blob/master/Makefile
