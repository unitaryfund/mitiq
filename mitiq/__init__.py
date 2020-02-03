# Licensed under ...
"""
This is the top level module from which functions and classes of
Mitiq can be directly imported.
"""

from .zne import Mitigator, class_mitigator, run_mitigation, fun_mitigator

with open("../VERSION.txt", "r") as f:
    __version__ = f.read().strip()


def version():
    """Returns the Mitiq version number."""
    return __version__
