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
import warnings
from pkg_resources import parse_requirements

LATEST_SUPPORTED_PKGS = {req.project_name: req.specs[0][1]
    for req in parse_requirements(open("requirements.txt"))}
INSTALLED_PKGS = {}

# Checking versions of Mitiq's core dependencies (must be installed)

from cirq import __version__ as cirq_version
INSTALLED_PKGS["cirq"] = cirq_version
from numpy import __version__ as numpy_version
INSTALLED_PKGS["numpy"] = numpy_version 
from scipy import __version__ as scipy_version
INSTALLED_PKGS["scipy"] = scipy_version 

import mitiq

# Checking versions of Mitiq's optional dependencies (may be installed)
OPTIONAL_PKGS = [
    "pyquil",
    "qiskit",
    "amazon-braket-sdk",
    "pennylane",
    "pennylane-qiskit",
]

try:
    from pyquil import __version__ as pyquil_version
    INSTALLED_PKGS["pyquil"] = pyquil_version 
except ImportError:
    INSTALLED_PKGS["pyquil"] = "Not installed"
try:
    from qiskit import __qiskit_version__  # pragma: no cover
    INSTALLED_PKGS["qiskit"] = __qiskit_version__["qiskit"]  # pragma: no cover
except ImportError:
    INSTALLED_PKGS["qiskit"] = "Not installed"
try:
    from braket._sdk import __version__ as braket_version
    INSTALLED_PKGS["amazon-braket-sdk"] = braket_version
except ImportError:
    INSTALLED_PKGS["amazon-braket-sdk"] = "Not installed"
try:
    from pennylane import __version__ as pennylane_version
    INSTALLED_PKGS["pennylane"] = pennylane_version
except ImportError:
    INSTALLED_PKGS["pennylane"] = "Not installed"
try:
    from pennylane_qiskit import __version__ as pennylane_qiskit_version
    INSTALLED_PKGS["pennylane-qiskit"] = pennylane_qiskit_version
except ImportError:
    INSTALLED_PKGS["pennylane-qiskit"] = "Not installed"


def latest_supported():
    """Returns the versions of Mitiq's optional packages that are supported by
    the current version of Mitiq. Requires that the dependency has a pinned
    version in the dev_requirements.txt file.
    """
    return {
        req.project_name: req.specs[0][1]
        for req in parse_requirements(open("dev_requirements.txt"))
        if req.project_name in (OPTIONAL_PKGS)
    }

LATEST_SUPPORTED_PKGS.update(latest_supported())

def check_versions() -> None:
    """Checks that the installed versions of Mitiq's dependencies are
    supported by the current version of Mitiq.
    """

    for name, version in INSTALLED_PKGS.items():
        if LATEST_SUPPORTED_PKGS[name] != version:
            warnings.warn(
                f"The currently installed version of {name} ({version}) is not"
                f"the currently supported version ({LATEST_SUPPORTED_PKGS[name]})."
                " If you are having trouble with this version of Mitiq, please "
                "consider updating to the latest supported dependency versions.",
                UserWarning,
            )
check_versions()

def about() -> None:
    """Displays information about Mitiq, core/optional packages, and Python
    version/platform information.
    """
    check_versions()
    about = f"""
    Mitiq: A Python toolkit for implementing error mitigation on quantum computers
    ==============================================================================
    Authored by: Mitiq team, 2020 & later (https://github.com/unitaryfund/mitiq)

    Mitiq Version: {mitiq.__version__}

    Core Dependencies (must be installed):
    --------------------------------------
    Cirq:
        - installed:        {INSTALLED_PKGS["cirq"]}
        - latest supported: {LATEST_SUPPORTED_PKGS["cirq"]}
    NumPy:
        - installed:        {INSTALLED_PKGS["numpy"]}
        - latest supported: {LATEST_SUPPORTED_PKGS["numpy"]}
    SciPy:
        - installed:        {INSTALLED_PKGS["scipy"]}
        - latest supported: {LATEST_SUPPORTED_PKGS["scipy"]}

    Optional Dependencies
    ---------------------
    Braket:
        - installed:        {INSTALLED_PKGS["amazon-braket-sdk"]}
        - latest supported: {LATEST_SUPPORTED_PKGS["amazon-braket-sdk"]}
    PennyLane:
        - installed:        {INSTALLED_PKGS["pennylane"]}
        - latest supported: {LATEST_SUPPORTED_PKGS["pennylane"]}
    PennyLane-Qiskit:
        - installed:        {INSTALLED_PKGS["pennylane-qiskit"]}
        - latest supported: {LATEST_SUPPORTED_PKGS["pennylane-qiskit"]}
    PyQuil:
        - installed:        {INSTALLED_PKGS["pyquil"]}
        - latest supported: {LATEST_SUPPORTED_PKGS["pyquil"]}
    Qiskit:
        - installed:        {INSTALLED_PKGS["qiskit"]}
        - latest supported: {LATEST_SUPPORTED_PKGS["qiskit"]}

    Python Version:\t{platform.python_version()}
    Platform Info:\t{platform.system()} ({platform.machine()})"""
    print(about)


if __name__ == "__main__":
    about()
