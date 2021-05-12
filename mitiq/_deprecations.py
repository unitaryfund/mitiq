# Copyright (C) 2021 Unitary Fund
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Catches deprecated import statements."""


def raise_deprecation_warning(version: str, method: str, module: str):
    """Raises an import error for attempts to import modules that could be called
    directly from Mitiq but must now be imported from one of Mitiq's modules.
    """
    raise ImportError(
        "As of version "
        + version
        + ", "
        + method
        + " must be imported from "
        + module
        + " instead of directly from mitiq. Please update your code "
        + "accordingly."
    )


def execute_with_zne():
    raise_deprecation_warning("0.9", "execute_with_zne", "mitiq.zne")


def mitigate_executor():
    raise_deprecation_warning("0.9", "mitigate_executor", "mitiq.zne")


def zne_decorator():
    raise_deprecation_warning("0.9", "zne_decorator", "mitiq.zne")
