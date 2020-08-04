"""
This file is used to optionally import what program types should be allowed
by mitiq based on what packages are installed in the environment.
"""

from typing import Union

SUPPORTED_PROGRAM_TYPES = {
    "cirq": "Circuit",
    "pyquil": "Program",
    "qiskit": "QuantumCircuit",
}

AVAILABLE_PROGRAM_TYPES = {}

for (module, program_type) in SUPPORTED_PROGRAM_TYPES.items():
    try:
        exec(f"from {module} import {program_type}")
        AVAILABLE_PROGRAM_TYPES.update({module: program_type})
    except ImportError:
        pass

QPROGRAM = Union[
    tuple(
        f"{package}.{circuit}" for package, circuit in AVAILABLE_PROGRAM_TYPES.items()
    )
]
