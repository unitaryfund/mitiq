# Copyright (C) 2020 Unitary Fund
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

"""Information about Mitiq and dependencies."""
import platform
from pkg_resources import parse_requirements

from cirq import __version__ as cirq_version
from numpy import __version__ as numpy_version
from scipy import __version__ as scipy_version

import mitiq

OPTIONAL_REQUIREMENTS = [
    "pyquil",
    "qiskit",
    "amazon-braket-sdk",
    "pennylane",
    "pennylane-qiskit",
]


def latest_supported():
    """Returns the versions of Mitiq's optional packages that are supported by
    the current version of Mitiq. Requires that the dependency has a pinned
    version in the dev_requirements.txt file.
    """
    return {
        req.project_name: req.specs[0][1]
        for req in parse_requirements(open("dev_requirements.txt"))
        if req.project_name in OPTIONAL_REQUIREMENTS
    }


def about() -> None:
    """Displays information about Mitiq, core/optional packages, and Python
    version/platform information.
    """
    try:
        from pyquil import __version__ as pyquil_version
    except ImportError:
        pyquil_version = "Not installed"
    try:
        from qiskit import __qiskit_version__  # pragma: no cover

        qiskit_version = __qiskit_version__["qiskit"]  # pragma: no cover
    except ImportError:
        qiskit_version = "Not installed"
    try:
        from braket._sdk import __version__ as braket_version
    except ImportError:
        braket_version = "Not installed"
    try:
        from pennylane import __version__ as pennylane_version
    except ImportError:
        pennylane_version = "Not installed"
    try:
        from pennylane_qiskit import __version__ as pennylane_qiskit_version
    except ImportError:
        pennylane_qiskit_version = "Not installed"

    optional_reqs = latest_supported()

    about_str = f"""
Mitiq: A Python toolkit for implementing error mitigation on quantum computers
==============================================================================
Authored by: Mitiq team, 2020 & later (https://github.com/unitaryfund/mitiq)

Mitiq Version:\t{mitiq.__version__}

Core Dependencies
-----------------
Cirq Version:\t{cirq_version}
NumPy Version:\t{numpy_version}
SciPy Version:\t{scipy_version}

Optional Dependencies
---------------------
PyQuil Version:           {pyquil_version}
    Latest Supported:     {optional_reqs["pyquil"]}
Qiskit Version:           {qiskit_version}
    Latest Supported:     {optional_reqs["qiskit"]}
Braket Version:           {braket_version}
    Latest Supported:     {optional_reqs["amazon-braket-sdk"]}
PennyLane Version:        {pennylane_version} 
    Latest Supported:     {optional_reqs["pennylane"]}
PennyLane-Qiskit Version: {pennylane_qiskit_version}
    Latest Supported:     {optional_reqs["pennylane-qiskit"]}

Python Version:\t{platform.python_version()}
Platform Info:\t{platform.system()} ({platform.machine()})"""
    print(about_str)


if __name__ == "__main__":
    about()
