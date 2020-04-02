# Licensed under ...
"""
This is the top level module from which functions and classes of
Mitiq can be directly imported.
"""
import os
from typing import Union

from cirq import Circuit


# This is used to optionally import what program types should be allowed
# by mitiq based on what packages are installed in the environment
PROGRAM_TYPES = [Circuit]
try:
    from pyquil import Program
    PROGRAM_TYPES.append(Program)
except ImportError:
    pass

try:
    from qiskit import QuantumCircuit
    PROGRAM_TYPES.append(QuantumCircuit)
except ImportError:
    pass

QPROGRAM = Union[tuple(PROGRAM_TYPES)]

# this must be after QPROGRAM as the zne.py module imports QPROGRAM
from mitiq.zne import execute_with_zne, mitigate_executor

directory_of_this_file = os.path.dirname(os.path.abspath(__file__))

with open(str(directory_of_this_file) + "/../VERSION.txt", "r") as f:
    __version__ = f.read().strip()


def version():
    """Returns the Mitiq version number."""
    return __version__
