# Licensed under ...
"""
This is the top level module from which functions and classes of
Mitiq can be directly imported.
"""
from mitiq.version import __version__

import os
from typing import Union

from cirq import Circuit

from mitiq.about import about


# This is used to optionally import what program types should be allowed
# by mitiq based on what packages are installed in the environment
SUPPORTED_PROGRAM_TYPES = {
    "cirq": "Circuit",
    "qiskit": "QuantumCircuit"
}
AVAILABLE_PROGRAM_TYPES = {}

for (module, program_type) in SUPPORTED_PROGRAM_TYPES.items():
    try:
        exec(f"from {module} import {program_type}")
        AVAILABLE_PROGRAM_TYPES.update({module: program_type})
    except ImportError:
        pass

QPROGRAM = Union[
    tuple(f"{package}.{circuit}"
     for package, circuit in AVAILABLE_PROGRAM_TYPES.items())
]

# this must be after QPROGRAM as the zne.py module imports QPROGRAM
from mitiq.zne import execute_with_zne, mitigate_executor


def version():
    """Returns the Mitiq version number."""
    return __version__
