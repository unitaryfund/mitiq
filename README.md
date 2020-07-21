![Python Build](https://github.com/unitaryfund/mitiq/workflows/Python%20Build/badge.svg?branch=master)
[![PyPI version](https://badge.fury.io/py/mitiq.svg)](https://badge.fury.io/py/mitiq)
[![Unitary Fund](https://img.shields.io/badge/Supported%20By-UNITARY%20FUND-brightgreen.svg?style=for-the-badge)](http://unitary.fund)


# Mitiq
A Python toolkit for implementing error mitigation on quantum computers.

## Features
Mitiq performs error mitigation protocols on quantum circuits using zero-noise extrapolation.


## Contents
```
mitiq/mitiq/
    | benchmarks    (package)
        |- maxcut
        |- tests    (package)
            |- test_maxcut
            |- test_random_circ
        |- random_circ
        |- utils
    | factories
    | folding
    | mitiq_pyquil   (package)
    	|- pyquil_utils
    	|- tests   (package)
       		|- test_zne
    | mitiq_qiskit   (package)
    	|- conversions
    	|- qiskit_utils
       	|- tests   (package)
       		|- test_conversions
       		|- test_zne
    | tests    (package)
    	|- test_factories
    	|- test_folding
    	|- test_utils
        |- test_zne
    | utils
    | zne
```
## Installation

To install locally use:

```bash
pip install -e .
```

To install the requirements for development use:

```bash
pip install -r requirements.tx
```

Note that this will install our testing environment that depends
on `qiskit` and `pyquil`.

## Use
A [Getting Started](docs/source/guide/)
tutorial can be found in the Documentation.

## Documentation
`Mitiq` documentation is found under `mitiq/docs`. A pdf with the documentation
updated to the latest release can be found
[here](docs/pdf/Mitiq-latest-release.pdf).

## Development and Testing

Ensure that you have installed the development environment. Then you can run
the tests using `make test` and build the docs using `make docs`. For more
information, see the contributor's guide (linked below).

## Contributing
You can find information on contributing to `mitiq` code in the [contributing guidelines](CONTRIBUTING.md).

To contribute to the documentation, read the
[instructions](docs/README-docs.md) in the `mitiq/docs` folder.


## Authors
Ryan LaRose, Andrea Mari, Nathan Shammah, and Will Zeng.
An up-to-date list of authors can be found
[here](https://github.com/unitaryfund/mitiq/graphs/contributors)

## License
[GNU GPL v.3.0.](LICENSE)
