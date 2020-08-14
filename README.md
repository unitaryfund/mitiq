[![build](https://github.com/unitaryfund/mitiq/workflows/build/badge.svg)](https://github.com/unitaryfund/mitiq/actions)
[![codecov](https://codecov.io/gh/unitaryfund/mitiq/branch/master/graph/badge.svg)](https://codecov.io/gh/unitaryfund/mitiq)
[![PyPI version](https://badge.fury.io/py/mitiq.svg)](https://badge.fury.io/py/mitiq)
[![Unitary Fund](https://img.shields.io/badge/Supported%20By-UNITARY%20FUND-brightgreen.svg?style=for-the-badge)](http://unitary.fund)


# Mitiq
A Python toolkit for implementing error mitigation on quantum computers.

## Documentation
The full documentation is available at [mitiq.readthedocs.io](mitiq.readthedocs.io).

A pdf with the documentation updated to the latest release can be found
[here](docs/pdf/).

## Features
Mitiq  is  an  open-source Python library that interfaces with multiple front-end quantum programming languages to implement 
[error-mitigation](https://mitiq.readthedocs.io/en/latest/guide/guide_06-error-mitigation.html) techniques
on various real and simulated quantum  processors.

Mitiq is compatible with [Cirq](https://github.com/quantumlib/Cirq), [Qiskit](https://github.com/Qiskit), and [PyQuil](https://github.com/rigetti/pyquil).  

Mitiq currently implements [zero-noise extrapolation](https://mitiq.readthedocs.io/en/latest/guide/guide_06-error-mitigation.html#zero-noise-extrapolation) and is designed to be modular to support [additional techniques](https://github.com/unitaryfund/mitiq/wiki).

## Contents
```
mitiq/mitiq/
    | benchmarks
        |- maxcut
        |- random_circuits
        |- randomized_benchmarking
        |- utils
    | mitiq_pyquil
        |- conversions
    	|- pyquil_utils
        |- quil
    | mitiq_qiskit
    	|- conversions
    	|- qiskit_utils
    | zne
        |- zne
        |- inference
        |- scaling
```
## Installation

To install locally use:

```bash
pip install -e .
```

To install the requirements for development use:

```bash
pip install -r requirements.txt
```

Note that this will install our testing environment that depends
on `qiskit` and `pyquil`.

## Use
A [Getting Started](https://mitiq.readthedocs.io/en/latest/guide/guide_02-getting-started.html)
tutorial can be found in the [documentation](https://mitiq.readthedocs.io).


## Development and Testing

Ensure that you have installed the development environment. Then you can run
the tests using `make test` and build the docs using `make docs`. For more
information, see the contributor's guide (linked below).

## Contributing
You can find information on contributing to `mitiq` code in the [contributing guidelines](CONTRIBUTING.md).

To contribute to the documentation, read the
[instructions](docs/README-docs.md) in the `mitiq/docs` folder.


## Authors
An up-to-date list of authors can be found
[here](https://github.com/unitaryfund/mitiq/graphs/contributors)

## License
[GNU GPL v.3.0.](LICENSE)
