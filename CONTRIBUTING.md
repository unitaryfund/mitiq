# Contributing to mitiq

## Opening an issue
You can begin contributing to `mitiq` code by raising an
[issue](https://github.com/unitaryfund/mitiq/issues/new), reporting a bug or
proposing a new feature request, using the labels to organize it.
You can use `mitiq.about()` to document your dependencies and work environment.

## Opening a pull request
You can open a [pull request](https://github.com/unitaryfund/mitiq/pulls) by pushing changes from a local branch, explaining the bug fix or new feature.

### Version control with git
Use git to keep track of the changes made to the repository. Use short and explanatory comments to document the changes with frequent commits.

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
In order to install all the libraries useful for contributing to the development of the library, from your local clone of the fork, run
```bash
(myenv) pip install -e .[development]
```
This command will use `pip` to read the requirements contained in `requirements.txt` and `development_requirements.txt`

### Adding tests
If you add new features to a function or class, it is strongly encouraged to add tests for such function. Mitiq uses a nested structure for packaging tests in directories named `tests` at the same level of each module.

### Updating the documentation

### Checking local tests

### Code style
Mitiq code is developed according the best practices of Python development.
* Please get familiar with PEP8 and PEP257 guidelines.
* Set one blank link of space between different functions or classes.
* You can use `black` code style to check the code style.
* For the docstrings, use google-style formatting.
* Use annotations for type hints.
