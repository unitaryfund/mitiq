# Licensed under ...
"""
This is the top level module from which functions and classes of
Mitiq can be directly imported.
"""

import numpy as np

from .zne import Mitigator, mitigate


def version():
    """Returns the Mitiq version number."""
    return __version__