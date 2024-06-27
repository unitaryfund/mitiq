# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Tests for mitiq.about()."""

import mitiq


def test_stdout():
    """Tests function prints a str."""
    mitiq.about()
