![Python Build](https://github.com/unitaryfund/mitiq/workflows/Python%20Build/badge.svg?branch=master)
[![Unitary Fund](https://img.shields.io/badge/Supported%20By-UNITARY%20FUND-brightgreen.svg?style=for-the-badge)](http://unitary.fund)

# Mitiq
A Python toolkit for implementing error mitigation on quantum computers.

## Features
Mitiq allows to perform error mitigation protocols on quantum circuits such as:

- Zero noise extrapolation (ZNE)

- Unitary folding, i.e., identity insertions

## Contents
mitiq/mitiq/
    | about
    | factories
    | folding
    | matrices
    | pyquil   (package)
    	|- pyquil_utils
    	|- tests   (package)
       		|- test_zne
    | qiskit   (package)
    	|- conversions
    	|- qiskit_utils
       	|- tests   (package)
       		|- test_conversions
       		|- test_zne
    | tests    (package)
    	|- test_factories
    	|- test_folding
    	|- test_matrices
    	|- test_utils
    | utils
    | zne

## Installation
To install locally use:
```bash
pip install -e .
```

To install for development use:
```bash
pip install -e .[development]
```
Note that this will install our testing environment that depends
on `qiskit` and `pyquil`.

## Use
A [Getting Started]() tutorial can be found in the Documentation.

## Documentation
`Mitiq` documentation is found under `mitiq/docs`. A pdf with the documentation
updated to the latest release can be found
[here](docs/pdf/Mitiq-latest-release.pdf).

## Development and Testing
Ensure that you have installed the development environment. Then
you can run tests with `pytest`.

## Contributing
You can contribute to `mitiq` code by raising an
[issue](https://github.com/unitaryfund/mitiq/issues/new) reporting a bug or
proposing new feature, using the labels to organize it. You can open a
[pull request](https://github.com/unitaryfund/mitiq/pulls) by pushing changes
from a local branch, explaining the bug fix or new feature.
You can use `mitiq.about()` to document your dependencies and work environment.

To contribute to the documentation, read the
[instructions](docs/README-docs.md) in the `mitiq/docs` folder.


## Authors
Ryan LaRose, Andrea Mari, Nathan Shammah, and Will Zeng.
An up-to-date list of authors can be found
[here](https://github.com/unitaryfund/mitiq/graphs/contributors)

## License
[GNU GPL v.3.0.](LICENSE)