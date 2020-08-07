"""Utility functions."""

from copy import deepcopy

from cirq import Circuit, CircuitDag, Gate, Moment, X, Y, Z, H, CNOT
from cirq.ops.measurement_gate import MeasurementGate


def _simplify_gate(gate: Gate) -> Gate:
    """If possible, returns a simpler but equivalent gate.
    Otherwise, the input gate is returned.
    The input does not mutate.

    Args:
        gate: The input gate to simplify.

    Returns: The simplified gate.
    """
    SELF_INVERSE_GATES = [X, Y, Z, H, CNOT]
    for self_inv_gate in SELF_INVERSE_GATES:
        if gate == self_inv_gate:
            return self_inv_gate
    return gate


def _simplify_circuit(circuit: Circuit) -> None:
    """If possible, mutates each gate of the
    input circuit with a simpler but equivalent gate.

    Args:
        gate: The input circuit to simplify.
    """
    # iterate over moments
    for moment_idx, moment in enumerate(circuit):
        simplified_operations = []
        # iterate over operations in moment
        for op in moment:
            simplified_gate = _simplify_gate(op.gate)
            simplified_operation = op.with_gate(simplified_gate)
            simplified_operations.append(simplified_operation)
        # mutate input circuit
        circuit[moment_idx] = Moment(simplified_operations)


def _equal(
    circuit_one: Circuit,
    circuit_two: Circuit,
    require_qubit_equality: bool = False,
    require_measurement_equality: bool = False,
) -> bool:
    """Returns True if the circuits are equal, else False.

    Args:
        circuit_one: Input circuit to compare to circuit_two.
        circuit_two: Input circuit to compare to circuit_one.
        require_qubit_equality: Requires that the qubits be equal
            in the two circuits.
        require_measurement_equality: Requires that measurements are equal on
            the two circuits, meaning that measurement keys are equal.

    Note:
        If set(circuit_one.all_qubits()) = {LineQubit(0)},
        then set(circuit_two_all_qubits()) must be {LineQubit(0)},
        else the two are not equal.
        If True, the qubits of both circuits must have a well-defined ordering.
    """
    if circuit_one is circuit_two:
        return True

    circuit_one = deepcopy(circuit_one)
    circuit_two = deepcopy(circuit_two)

    if not require_qubit_equality:
        # Transform the qubits of circuit one to those of circuit two
        qubit_map = dict(
            zip(
                sorted(circuit_one.all_qubits()),
                sorted(circuit_two.all_qubits()),
            )
        )
        circuit_one = circuit_one.transform_qubits(lambda q: qubit_map[q])

    if not require_measurement_equality:
        for circ in (circuit_one, circuit_two):
            measurements = [
                (moment, op)
                for moment, op, _ in circ.findall_operations_with_gate_type(
                    MeasurementGate
                )
            ]
            circ.batch_remove(measurements)

            for i in range(len(measurements)):
                measurements[i][1].gate.key = ""

            circ.batch_insert(measurements)

    return CircuitDag.from_circuit(circuit_one) == CircuitDag.from_circuit(
        circuit_two
    )
