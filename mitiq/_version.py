# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Reads in version information.
Note: This file will be overwritten by the packaging process."""

import os

directory_of_this_file = os.path.dirname(os.path.abspath(__file__))

with open(f"{directory_of_this_file}/../VERSION.txt", "r") as f:
    __version__ = f.read().strip()
