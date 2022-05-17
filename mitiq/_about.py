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
from typing import Dict
import platform
import warnings
from pkg_resources import parse_requirements
import os

from cirq import __version__ as cirq_version
from numpy import __version__ as numpy_version
from scipy import __version__ as scipy_version

from mitiq import __version__ as mitiq_version


def installed_packages() -> Dict[str, str]:
    """Returns the versions of (core and optional) packages
    that are installed in the local environment.
    """
    # Logging versions of Mitiq's core dependencies (must be installed)
    installed_pkgs = {}
    installed_pkgs["cirq"] = cirq_version
    installed_pkgs["numpy"] = numpy_version
    installed_pkgs["scipy"] = scipy_version

    # Logging versions of Mitiq's optional dependencies
    try:
        from pyquil import __version__ as pyquil_version

        installed_pkgs["pyquil"] = pyquil_version
    except ImportError:
        installed_pkgs["pyquil"] = "Not found"
    try:
        from qiskit import __qiskit_version__

        installed_pkgs["qiskit"] = __qiskit_version__["qiskit"]
    except ImportError:
        installed_pkgs["qiskit"] = "Not found"
    try:
        from braket._sdk import __version__ as braket_version

        installed_pkgs["amazon-braket-sdk"] = braket_version
    except ImportError:
        installed_pkgs["amazon-braket-sdk"] = "Not found"
    try:
        from pennylane import __version__ as pennylane_version

        installed_pkgs["pennylane"] = pennylane_version
    except ImportError:
        installed_pkgs["pennylane"] = "Not found"
    try:
        from pennylane_qiskit import __version__ as pennylane_qiskit_version

        installed_pkgs["pennylane-qiskit"] = pennylane_qiskit_version
    except ImportError:
        installed_pkgs["pennylane-qiskit"] = "Not found"

    return installed_pkgs


def latest_supported_packages() -> Dict[str, str]:
    """Returns the latest versions of (core and optional) packages
    that are supported by the current version of Mitiq.
    """
    # Mitiq optional packages whose versions are checked
    optional_pkg = [
        "pyquil",
        "qiskit",
        "amazon-braket-sdk",
        "pennylane",
        "pennylane-qiskit",
    ]

    _dir_of_this_file = os.path.dirname(os.path.abspath(__file__))
    with open(f"{_dir_of_this_file}/../requirements.txt", "r") as f:
        _requirements = f.read().strip()
    with open(f"{_dir_of_this_file}/../dev_requirements.txt", "r") as f:
        _dev_requirements = f.read().strip()

    latest_core = {
        req.project_name: req.specs[0][1]
        for req in parse_requirements(_requirements)
    }
    latest_dev = {
        req.project_name: req.specs[0][1]
        for req in parse_requirements(_dev_requirements)
        if req.project_name in optional_pkg
    }
    return dict(**latest_core, **latest_dev)


def check_versions() -> None:
    """Checks that the installed versions of Mitiq's dependencies are
    supported by the current version of Mitiq.
    """
    latest_supported_pkgs = latest_supported_packages()
    installed_pkgs = installed_packages()
    for name, version in installed_pkgs.items():
        if latest_supported_pkgs[name] != version:
            warnings.warn(
                f"The currently installed version of {name} ({version}) is "
                "not the currently supported version "
                f"({latest_supported_pkgs[name]}). If you are having trouble"
                f", please consider downgrading {name} to the latest supported"
                " dependency version.",
                UserWarning,
            )


def about() -> None:
    """Displays information about Mitiq, core/optional packages, and Python
    version/platform information.
    """
    latest_supported_pkgs = latest_supported_packages()
    installed_pkgs = installed_packages()
    about = f"""
Mitiq: A Python toolkit for implementing error mitigation on quantum computers
==============================================================================
Authored by: Mitiq team, 2020 & later (https://github.com/unitaryfund/mitiq)

Mitiq Version: {mitiq_version}

Core Dependencies (must be installed):
--------------------------------------
Cirq:
    - installed:        {installed_pkgs["cirq"]}
    - latest supported: {latest_supported_pkgs["cirq"]}
NumPy:
    - installed:        {installed_pkgs["numpy"]}
    - latest supported: {latest_supported_pkgs["numpy"]}
SciPy:
    - installed:        {installed_pkgs["scipy"]}
    - latest supported: {latest_supported_pkgs["scipy"]}

Optional Dependencies
---------------------
Braket:
    - installed:        {installed_pkgs["amazon-braket-sdk"]}
    - latest supported: {latest_supported_pkgs["amazon-braket-sdk"]}
PennyLane:
    - installed:        {installed_pkgs["pennylane"]}
    - latest supported: {latest_supported_pkgs["pennylane"]}
PennyLane-Qiskit:
    - installed:        {installed_pkgs["pennylane-qiskit"]}
    - latest supported: {latest_supported_pkgs["pennylane-qiskit"]}
PyQuil:
    - installed:        {installed_pkgs["pyquil"]}
    - latest supported: {latest_supported_pkgs["pyquil"]}
Qiskit:
    - installed:        {installed_pkgs["qiskit"]}
    - latest supported: {latest_supported_pkgs["qiskit"]}

Python Version:\t{platform.python_version()}
Platform Info:\t{platform.system()} ({platform.machine()})"""
    print(about)


if __name__ == "__main__":
    about()
