"""Unitary folding methods written for Cirq circuits."""

from copy import deepcopy

import cirq


class FoldingError(Exception):
    pass


class UnitaryError(Exception):
    pass


def global_fold(circuit: cirq.Circuit) -> cirq.Circuit:
    """Returns a new circuit obtained by mapping

    circuit --> circuit * circuit^dagger * circuit

    Args:
        circuit: Circuit to fold.
    """
    new = deepcopy(circuit)
    try:
        inv = cirq.inverse(new)
    except TypeError:
        raise FoldingError("Circuit contains gates with undefined inverses. Cannot be folded")
    return new + inv + new


def local_fold_index(circuit: cirq.Circuit, moment: int, index: int) -> cirq.Circuit:
    """Returns a new circuit where the gate G in `moment, index` is replaced by G G^dagger G.

    Args:
        circuit: Circuit to fold.
        moment: Moment in which the gate sits in the circuit.
        index: Index of the gate within the specified moment.
    """
    new = deepcopy(circuit)
    op = new[moment].operations[index]
    inv = cirq.inverse(op)
    new.insert(moment, [inv, op])
    return new


def local_fold_qubit(circuit: cirq.Circuit, qubit: cirq.Qid) -> cirq.Circuit:
    """Folds all operations locally on a qubit.

    Args:
        circuit: Circuit to fold.
        qubit: Qubit to fold all operations on.
    """
    new = deepcopy(circuit)
    for (i, moment) in enumerate(circuit):
        op = circuit.operation_at(qubit, i)
        try:
            inv = cirq.inverse(op)
        except TypeError:
            raise UnitaryError(f"Operation {op} cannot be inverted.")
        new.insert(2 * i + 1, [inv, op])
    return new


def local_fold(circuit: cirq.Circuit, gate: cirq.ops.Gate) -> cirq.Circuit:
    """Folds all gates matching the input gate type."""
    pass


def test_global_fold_hgates():
    """Tests global folding a circuit of all Hadamard gates."""
    qreg = cirq.LineQubit.range(10)
    circuit = cirq.Circuit(
        [cirq.ops.H.on_each(*qreg)]
    )
    folded = global_fold(circuit)
    assert len(folded) == 3
    for moment in circuit:
        for op in moment:
            assert op.gate == cirq.ops.H


def test_local_fold_qubit_one_qubit():
    """Tests local folding all ops on a qubit on a single qubit circuit."""
    qbit = cirq.LineQubit(0)
    circ = cirq.Circuit(
        [cirq.ops.H.on(qbit), cirq.ops.X.on(qbit)]
    )
    folded = local_fold_qubit(circ, qbit)
    assert len(folded) == 6
    assert folded == cirq.Circuit(
        [cirq.ops.H.on(qbit)] * 3 + [cirq.ops.X.on(qbit)] * 3
    )


def test_local_fold_qubit_two_qubits():
    """Tests local folding all ops on a qubit on a two qubit circuit."""
    qreg = cirq.LineQubit.range(2)
    circ = cirq.Circuit(
        [cirq.ops.H.on(qreg[0]), cirq.ops.CNOT.on(*qreg)]
    )
    folded = local_fold_qubit(circ, qreg[0])
    assert len(folded) == 6
    assert folded == cirq.Circuit(
        [cirq.ops.H.on(qreg[0])] * 3 + [cirq.ops.CNOT.on(*qreg)] * 3
    )


def test_local_fold_index_one_qubit():
    """Tests local folding with a moment, index for a one qubit circuit."""
    qbit = cirq.LineQubit(0)
    circ = cirq.Circuit(
        [cirq.ops.H.on(qbit), cirq.ops.X.on(qbit), cirq.ops.Z.on(qbit)]
    )
    # Fold the zeroth operation in the zeroth moment
    folded = local_fold_index(circ, moment=0, index=0)
    assert folded == cirq.Circuit(
        [cirq.ops.H.on(qbit)] * 3 + [cirq.ops.X.on(qbit)] + [cirq.ops.Z.on(qbit)]
    )
    # Fold the zeroth operation in the first moment
    folded = local_fold_index(circ, moment=1, index=0)
    assert folded == cirq.Circuit(
        [cirq.ops.H.on(qbit)] + [cirq.ops.X.on(qbit)] * 3 + [cirq.ops.Z.on(qbit)]
    )
    # Fold the zeroth operation in the second moment
    folded = local_fold_index(circ, moment=2, index=0)
    assert folded == cirq.Circuit(
        [cirq.ops.H.on(qbit)] + [cirq.ops.X.on(qbit)] + [cirq.ops.Z.on(qbit)] * 3
    )
    # Make sure the original circuit wasn't modified
    assert circ == cirq.Circuit(
        [cirq.ops.H.on(qbit), cirq.ops.X.on(qbit), cirq.ops.Z.on(qbit)]
    )
