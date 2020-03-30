![Python Build](https://github.com/unitaryfund/mitiq/workflows/Python%20Build/badge.svg?branch=master)

# Mitiq
A Python toolkit for implementing error mitigation on quantum computers.

## Features
Mitiq allows to perform error mitigation protocols on quantum circuits such as:

- Zero noise extrapolation (ZNE)

- Unitary folding, i.e., identity insertions

## Contents

    about
    factories
    folding
    matrices
    pyquil (package)
    qiskit (package)
    tests (package)
    utils
    zne

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

## Usage
A Getting Started tutorial can be found in the Documentation.


## Development and Testing
Ensure that you have installed the development enviornment. Then
you can run tests with `pytest`.


## Authors
Ryan LaRose, Andrea Mari, Nathan Shammah, and Will Zeng.


## Licence
GNU GPL v.3.0.