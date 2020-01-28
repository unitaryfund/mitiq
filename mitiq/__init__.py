# Licensed under ...
"""
This is the top level module from which all basic functions and classes of
Mitiq can be directly imported.
"""

import numpy as np

import utils
import zne


def version():
    """Returns the Mitiq version number."""
    return __version__