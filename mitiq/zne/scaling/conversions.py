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

"""Functions for converting to/from Mitiq's internal circuit representation."""
from functools import wraps
from typing import Callable, Tuple

from cirq import Circuit

from mitiq._typing import SUPPORTED_PROGRAM_TYPES, QPROGRAM


class UnsupportedCircuitError(Exception):
    pass


class CircuitConversionError(Exception):
    pass


def convert_to_mitiq(circuit: QPROGRAM) -> Tuple[Circuit, str]:
    """Converts any valid input circuit to a mitiq circuit.

    Args:
        circuit: Any quantum circuit object supported by mitiq.
                 See mitiq.SUPPORTED_PROGRAM_TYPES.

    Raises:
        UnsupportedCircuitError: If the input circuit is not supported.

    Returns:
        circuit: Mitiq circuit equivalent to input circuit.
        input_circuit_type: Type of input circuit represented by a string.
    """
    conversion_function: Callable[[QPROGRAM], Circuit]
    if "qiskit" in circuit.__module__:
        from mitiq.mitiq_qiskit.conversions import from_qiskit

        input_circuit_type = "qiskit"
        conversion_function = from_qiskit
    elif "pyquil" in circuit.__module__:
        from mitiq.mitiq_pyquil.conversions import from_pyquil

        input_circuit_type = "pyquil"
        conversion_function = from_pyquil
    elif isinstance(circuit, Circuit):
        input_circuit_type = "cirq"

        def conversion_function(circ):
            return circ

    else:
        raise UnsupportedCircuitError(
            f"Circuit from module {circuit.__module__} is not supported.\n\n"
            f"Circuit types supported by Mitiq are \n{SUPPORTED_PROGRAM_TYPES}"
        )

    try:
        mitiq_circuit = conversion_function(circuit)
    except Exception:
        raise CircuitConversionError(
            "Circuit could not be converted to an internal Mitiq circuit. "
            "This may be because the circuit contains custom gates or Pragmas "
            "(pyQuil). If you think this is a bug, you can open an issue at "
            "https://github.com/unitaryfund/mitiq."
        )

    return mitiq_circuit, input_circuit_type


def convert_from_mitiq(circuit: Circuit, conversion_type: str) -> QPROGRAM:
    """Converts a mitiq circuit to a type specificed by the conversion type.

    Args:
        circuit: Mitiq circuit to convert.
        conversion_type: String specifier for the converted circuit type.
    """
    if conversion_type == "qiskit":
        from mitiq.mitiq_qiskit.conversions import to_qiskit

        conversion_function = to_qiskit
    elif conversion_type == "pyquil":
        from mitiq.mitiq_pyquil.conversions import to_pyquil

        conversion_function = to_pyquil
    elif isinstance(circuit, Circuit):

        def conversion_function(circ):
            return circ

    else:
        raise UnsupportedCircuitError(
            f"Conversion to circuit of type {conversion_type} is unsupported."
            f"\nCircuit types supported by mitiq = {SUPPORTED_PROGRAM_TYPES}"
        )

    try:
        converted_circuit = conversion_function(circuit)
    except Exception:
        raise CircuitConversionError(
            f"Circuit could not be converted from an internal Mitiq type to a "
            f"circuit of type {conversion_type}."
        )

    return converted_circuit


def converter(fold_method: Callable) -> Callable:
    """Decorator for handling conversions."""

    @wraps(fold_method)
    def new_fold_method(circuit: QPROGRAM, *args, **kwargs) -> QPROGRAM:
        mitiq_circuit, input_circuit_type = convert_to_mitiq(circuit)
        if kwargs.get("return_mitiq") is True:
            return fold_method(mitiq_circuit, *args, **kwargs)
        return convert_from_mitiq(
            fold_method(mitiq_circuit, *args, **kwargs), input_circuit_type
        )

    return new_fold_method
