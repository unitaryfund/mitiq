"""
Command line output of information on Mitiq and dependencies.
"""
__all__ = ["about"]

import inspect
import platform
import os
import sys

from cirq import __version__ as cirq_version
from numpy import __version__ as numpy_version
from scipy import __version__ as scipy_version

import mitiq

MITIQ_INSTALL_PATH = os.path.dirname(inspect.getsourcefile(mitiq))
PYTHON_VERSION = sys.version_info[0:3]


def about():
    """
    About box for Mitiq. Gives version numbers for
    Mitiq, NumPy, SciPy, Cirq, PyQuil, Qiskit.
    """
    try:
        from pyquil import __version__ as pyquil_version
    except ImportError:
        pyquil_version = "Not installed"
    try:
        from qiskit import __version__ as qiskit_version
    except ImportError:
        qiskit_version = "Not installed"

    about_str = f"""
Mitiq: A Python toolkit for implementing error mitigation on quantum computers
==============================================================================
Authored by: Mitiq team, 2020 & later (https://github.com/unitaryfund/mitiq)

Mitiq Version:\t{mitiq.__version__}

Cirq Version:\t{cirq_version}
NumPy Version:\t{numpy_version}
SciPy Version:\t{scipy_version}
PyQuil Version:\t{pyquil_version}
Qiskit Version:\t{qiskit_version}

Python Version:\t{PYTHON_VERSION[0]}.{PYTHON_VERSION[1]}.{PYTHON_VERSION[2]}
Platform Info:\t{platform.system()} ({platform.machine()})
Install Path:\t{MITIQ_INSTALL_PATH}"""
    print(about_str)


if __name__ == "__main__":
    about()
