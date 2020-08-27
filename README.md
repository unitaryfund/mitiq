[![build](https://github.com/unitaryfund/mitiq/workflows/build/badge.svg)](https://github.com/unitaryfund/mitiq/actions)
[![codecov](https://codecov.io/gh/unitaryfund/mitiq/branch/master/graph/badge.svg)](https://codecov.io/gh/unitaryfund/mitiq)
[![Documentation Status](https://readthedocs.org/projects/mitiq/badge/?version=latest)](https://mitiq.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/mitiq.svg)](https://badge.fury.io/py/mitiq)
[![Unitary Fund](https://img.shields.io/badge/Supported%20By-UNITARY%20FUND-brightgreen.svg?style=for-the-badge)](http://unitary.fund)


# Mitiq

Mitiq is a Python toolkit for implementing error mitigation techniques on quantum computers.

## Installation

Mitiq can be installed from PyPi via

```bash
pip install mitiq
```

To test installation, run

```python
import mitiq
mitiq.about()
```

This prints out version information about core requirements and optional quantum software packages which Mitiq can
interface with. 

### Supported quantum programming languages

Mitiq can currently interface with

* [Cirq](https://github.com/quantumlib/Cirq) >= 0.9.0, 
* [Qiskit](https://qiskit.org/) >= 0.19.0, and 
* [pyQuil](https://github.com/rigetti/pyquil) >= 2.18.0. 

Cirq is a core requirement of Mitiq and is automatically installed. To use Mitiq with other quantum programming
languages, install the optional package(s) following the instructions linked above.

### Supported quantum processors

Mitiq can be used on any quantum processor which can be accessed by supported quantum programming languages and is 
available to the user.

## Getting started

See this [getting started](https://mitiq.readthedocs.io/en/latest/guide/guide-getting-started.html) guide in 
[Mitiq's documentation](https://mitiq.readthedocs.io).

## Error mitigation techniques

Mitiq currently implements [zero-noise extrapolation](https://mitiq.readthedocs.io/en/latest/guide/guide-zne.html) and 
is designed to support [additional techniques](https://github.com/unitaryfund/mitiq/wiki).

## Documentation

Mitiq's documentation is hosted at [mitiq.readthedocs.io](https://mitiq.readthedocs.io). A PDF version of the latest 
release can be found [here](https://mitiq.readthedocs.io/_/downloads/en/latest/pdf/).

## Developer information

We welcome contributions to Mitiq including bug fixes, feature requests, etc. Please see the 
[contribution guidelines](https://github.com/unitaryfund/mitiq/blob/master/CONTRIBUTING.md) for more details. To contribute to the documentation, please see these
[documentation guidelines](https://github.com/unitaryfund/mitiq/blob/master/docs/CONTRIBUTING_DOCS.md).


## Authors

An up-to-date list of authors can be found
[here](https://github.com/unitaryfund/mitiq/graphs/contributors).

## License

[GNU GPL v.3.0.](LICENSE)
