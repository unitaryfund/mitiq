# Licensed under ...
"""
This is the top level module from which functions and classes of
Mitiq can be directly imported.
"""

from .zne import Mitigator, class_mitigator, run_mitigation, fun_mitigator


def version():
    """Returns the Mitiq version number."""
    return __version__