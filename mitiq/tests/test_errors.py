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

"""Tests for raising appropriate errors"""
import pytest

from mitiq import execute_with_zne, mitigate_executor, zne_decorator


def test_deprecate_execute_with_zne():
    with pytest.raises(ImportError):
        execute_with_zne()


def test_deprecate_mitigate_executor():
    with pytest.raises(ImportError):
        mitigate_executor()


def test_deprecate_zne_decorator():
    with pytest.raises(ImportError):
        zne_decorator()
