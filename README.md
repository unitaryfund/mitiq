![Python Build](https://github.com/unitaryfund/mitiq/workflows/Python%20Build/badge.svg?branch=master)

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
A Getting Started tutorial can be found in the Documentation.

## Development and Testing
Ensure that you have installed the development environment. Then
you can run tests with `pytest`.

## Contributing
You can contribute to `mitiq` code by raising an issue about a bug or
new feature, using the labels to organize it. You can open a pull request
and explain the bug fix or new feature. You can use `mitiq.about()` to document
the dependencies installed in your environment.

To contribute to the documentation, read the
[instructions](docs/README-docs.md) in the `mitiq/docs` folder.


## Authors
Ryan LaRose, Andrea Mari, Nathan Shammah, and Will Zeng.
An up-to-date list of authors can be found
[here](https://github.com/unitaryfund/mitiq/graphs/contributors)

## Licence
GNU GPL v.3.0.