import cirq
from mitiq import PauliString, Observable


def convert_from_cirq_PauliString_to_Mitiq_PauliString(cirq_PauliString):
    map = {cirq.X: "X", cirq.Y: "Y", cirq.Z: "Z", cirq.I: "I"}
    N = 5
    new_value = ["I"] * N
    for v in cirq_PauliString.items():
        new_value[v[0].x] = map[v[1]]
    mitiq_pauli_string = PauliString(
        "".join(new_value), cirq_PauliString.coefficient, range(N)
    )
    return mitiq_pauli_string


def convert_from_cirq_PauliSum_to_Mitiq_Observable(cirq_PauliSum):
    mitiq_pauli_strings = []
    pauli_sum_list = list(cirq_PauliSum)  # convert to list so we can index
    for i in range(len(pauli_sum_list)):
        mitiq_pauli_string = (
            convert_from_cirq_PauliString_to_Mitiq_PauliString(
                pauli_sum_list[i]
            )
        )
        mitiq_pauli_strings.append(mitiq_pauli_string)
    return Observable(*mitiq_pauli_strings)


def convert_from_Mitiq_Observable_to_cirq_PauliSum(mitiq_Observable):
    observable_group_elements = [g.elements for g in mitiq_Observable.groups]
    flattened_list = [
        item for sublist in observable_group_elements for item in sublist
    ]
    return sum([pauli_string._pauli for pauli_string in flattened_list])
