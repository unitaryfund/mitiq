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

"""Unit tests for readout confusion inversion."""
import cirq

from numpy import generic
import pytest

from mitiq.rem.rci import execute_with_rci
from mitiq.interface.mitiq_cirq import sample_bitstrings

# Default qubit register and circuit for unit tests
qreg = [cirq.LineQubit(i) for i in range(2)]
circ = cirq.Circuit(cirq.ops.H.on_each(*qreg), cirq.measure_each(*qreg))


def generic_executor(circuit, noise_level: float = 0.1) -> float:
    """Executor that simulates a circuit of any type and returns
    the measurement result."""
    noisy_circuit = circuit.with_noise(cirq.depolarize(p=noise_level))
    result = cirq.DensityMatrixSimulator().run(noisy_circuit)
    return result


def test_rci_identity():
    execute_with_rci(circ, sample_bitstrings)


if __name__ == "__main__":
    test_rci_identity()

