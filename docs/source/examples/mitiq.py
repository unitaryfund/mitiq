# This is a virtual mitiq package.
# When a notebook is executed in this folder, the line "import mitiq"
# can be executed without errors even if mitiq is not installed via pip.

# See: https://mg.readthedocs.io/importing-local-python-modules-from-
# jupyter-notebooks/sys-path-in-helper-module/path-helper.html

import importlib
import os
import sys

# Add mitiq folder to sys.path
sys.path.insert(0, os.path.abspath("../../../"))

# Temporarily hijack __file__ to avoid adding names at module scope;
# __file__ will be overwritten again during the reload() call.
__file__ = {'sys': sys, 'importlib': importlib}

del importlib
del os
del sys

__file__['importlib'].reload(__file__['sys'].modules[__name__])
