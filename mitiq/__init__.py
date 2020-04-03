# Licensed under ...
"""
This is the top level module from which functions and classes of
Mitiq can be directly imported.
"""
from importlib import import_module
import os
from typing import Union

from cirq import Circuit


# This is used to optionally import what program types should be allowed
# by mitiq based on what packages are installed in the environment
SUPPORTED_PROGRAM_TYPES = {
    "qiskit": "QuantumCircuit",
    "pyquil": "Program"
}
AVAILABLE_PROGRAM_TYPES = {Circuit}

for (module, program_type) in SUPPORTED_PROGRAM_TYPES.items():
    try:
        AVAILABLE_PROGRAM_TYPES.add(
            exec(f"from {module} import {program_type}")
        )
    except ImportError:
        pass

QPROGRAM = Union[tuple(AVAILABLE_PROGRAM_TYPES)]

# this must be after QPROGRAM as the zne.py module imports QPROGRAM
from mitiq.zne import execute_with_zne, mitigate_executor

_directory_of_this_file = os.path.dirname(os.path.abspath(__file__))

with open(str(_directory_of_this_file) + "/../VERSION.txt", "r") as f:
    __version__ = f.read().strip()


def version():
    """Returns the Mitiq version number."""
    return __version__
