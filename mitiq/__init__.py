# Licensed under GNI GPL v.3.0
"""
This is the top level module from which functions and classes of
Mitiq can be directly imported.
"""

from mitiq._about import about
from mitiq._typing import QPROGRAM
from mitiq._version import __version__
from mitiq.zne.zne import execute_with_zne, mitigate_executor

#import mitiq.zne.inference as mitiq.inference
#import mitiq.zne.scaling as mitiq.scaling
#import mitiq.zne.zne as mitiq.zne