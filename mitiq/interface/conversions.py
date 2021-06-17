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
from typing import Any, Iterable, Callable, Tuple

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

    try:
        package = circuit.__module__
    except AttributeError:
        raise UnsupportedCircuitError(
            "Could not determine the package of the input circuit."
        )

    if "qiskit" in package:
        from mitiq.interface.mitiq_qiskit.conversions import from_qiskit

        input_circuit_type = "qiskit"
        conversion_function = from_qiskit
    elif "pyquil" in package:
        from mitiq.interface.mitiq_pyquil.conversions import from_pyquil

        input_circuit_type = "pyquil"
        conversion_function = from_pyquil
    elif "braket" in package:
        from mitiq.interface.mitiq_braket.conversions import from_braket

        input_circuit_type = "braket"
        conversion_function = from_braket
    elif isinstance(circuit, Circuit):
        input_circuit_type = "cirq"

        def conversion_function(circ: Circuit) -> Circuit:
            return circ

    else:
        raise UnsupportedCircuitError(
            f"Circuit from module {package} is not supported.\n\n"
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
    """Converts a Mitiq circuit to a type specified by the conversion type.

    Args:
        circuit: Mitiq circuit to convert.
        conversion_type: String specifier for the converted circuit type.
    """
    conversion_function: Callable[[Circuit], QPROGRAM]
    if conversion_type == "qiskit":
        from mitiq.interface.mitiq_qiskit.conversions import to_qiskit

        conversion_function = to_qiskit
    elif conversion_type == "pyquil":
        from mitiq.interface.mitiq_pyquil.conversions import to_pyquil

        conversion_function = to_pyquil
    elif conversion_type == "braket":
        from mitiq.interface.mitiq_braket.conversions import to_braket

        conversion_function = to_braket
    elif conversion_type == "cirq":

        def conversion_function(circ: Circuit) -> Circuit:
            return circ

    else:
        raise UnsupportedCircuitError(
            f"Conversion to circuit of type {conversion_type} is unsupported."
            f"\nCircuit types supported by Mitiq = {SUPPORTED_PROGRAM_TYPES}"
        )

    try:
        converted_circuit = conversion_function(circuit)
    except Exception:
        raise CircuitConversionError(
            f"Circuit could not be converted from an internal Mitiq type to a "
            f"circuit of type {conversion_type}."
        )

    return converted_circuit


def accept_any_qprogram_as_input(
    accept_cirq_circuit_function: Callable[[Circuit], Any]
) -> Callable[[QPROGRAM], Any]:
    @wraps(accept_cirq_circuit_function)
    def accept_any_qprogram_function(
        circuit: QPROGRAM, *args: Any, **kwargs: Any
    ) -> Any:
        cirq_circuit, _ = convert_to_mitiq(circuit)
        return accept_cirq_circuit_function(cirq_circuit, *args, **kwargs)

    return accept_any_qprogram_function


def atomic_converter(
    cirq_circuit_modifier: Callable[..., Any]
) -> Callable[..., Any]:
    """Decorator which allows for a function which inputs and returns a Cirq
    circuit to input and return any QPROGRAM.

    Args:
        cirq_circuit_modifier: Function which inputs a Cirq circuit and returns
            a (potentially modified) Cirq circuit.
    """

    @wraps(cirq_circuit_modifier)
    def qprogram_modifier(
        circuit: QPROGRAM, *args: Any, **kwargs: Any
    ) -> QPROGRAM:
        # Convert to Mitiq representation.
        mitiq_circuit, input_circuit_type = convert_to_mitiq(circuit)

        # Modify the Cirq circuit.
        scaled_circuit = cirq_circuit_modifier(mitiq_circuit, *args, **kwargs)

        if kwargs.get("return_mitiq") is True:
            return scaled_circuit

        # Base conversion back to input type.
        scaled_circuit = convert_from_mitiq(scaled_circuit, input_circuit_type)

        return scaled_circuit

    return qprogram_modifier


def atomic_one_to_many_converter(
    cirq_circuit_modifier: Callable[..., Iterable[Circuit]]
) -> Callable[..., Iterable[QPROGRAM]]:
    @wraps(cirq_circuit_modifier)
    def qprogram_modifier(
        circuit: QPROGRAM, *args: Any, **kwargs: Any
    ) -> Iterable[QPROGRAM]:
        mitiq_circuit, input_circuit_type = convert_to_mitiq(circuit)

        modified_circuits: Iterable[Circuit] = cirq_circuit_modifier(
            mitiq_circuit, *args, **kwargs
        )

        if kwargs.get("return_mitiq") is True:
            return modified_circuits

        return [
            convert_from_mitiq(modified_circuit, input_circuit_type)
            for modified_circuit in modified_circuits
        ]

    return qprogram_modifier


def noise_scaling_converter(
    noise_scaling_function: Callable[..., Any]
) -> Callable[..., Any]:
    """Decorator for handling conversions with noise scaling functions.

    Args:
        noise_scaling_function: Function which inputs a cirq.Circuit, modifies
            it to scale noise, then returns a cirq.Circuit.
    """

    @wraps(noise_scaling_function)
    def new_scaling_function(
        circuit: QPROGRAM, *args: Any, **kwargs: Any
    ) -> QPROGRAM:
        scaled_circuit = atomic_converter(noise_scaling_function)(
            circuit, *args, **kwargs
        )

        # Keep the same register structure and measurement order with Qiskit.
        if "qiskit" in scaled_circuit.__module__:
            from mitiq.interface.mitiq_qiskit.conversions import (
                _transform_registers,
                _measurement_order,
            )

            scaled_circuit.remove_final_measurements()
            _transform_registers(
                scaled_circuit, new_qregs=circuit.qregs,
            )
            if circuit.cregs and not scaled_circuit.cregs:
                scaled_circuit.add_register(*circuit.cregs)

            for q, c in _measurement_order(circuit):
                scaled_circuit.measure(q, c)

        return scaled_circuit

    return new_scaling_function
