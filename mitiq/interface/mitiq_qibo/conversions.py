# Copyright (C) 2022 Unitary Fund

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


from cirq import Circuit
from qibo import models
from qibo.models import Circuit as qiboCircuit
from mitiq.interface.mitiq_qiskit.conversions import from_qasm, to_qasm


def from_qibo(circuit: qiboCircuit) -> Circuit:
    """Returns a Cirq circuit equivalent to the input qibo circuit.

    Args:
        circuit: qibo Circuit to convert to a Cirq circuit.
    """
    c_qasm = models.Circuit.to_qasm(circuit)
    c_mitiq = from_qasm(c_qasm)
    return c_mitiq


def to_qibo(circuit: Circuit) -> qiboCircuit:
    """Returns a qibo circuit equivalent to the input Cirq circuit.

    Args:
        circuit: Cirq circuit to convert to a qibo circuit.
    """
    c_qasm = to_qasm(circuit)
    c_qibo = models.Circuit.from_qasm(c_qasm)
    return c_qibo
