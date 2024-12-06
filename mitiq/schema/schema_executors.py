import json
import jsonschema
import random
import itertools
import numpy as np
from jsonschema import validate
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
from qiskit.circuit.random import random_circuit
from mitiq import zne, ddd
from mitiq.zne.scaling.folding import fold_global, fold_gates_at_random, fold_all
from mitiq.zne.scaling.layer_scaling import get_layer_folding
from mitiq.zne.scaling.identity_insertion import insert_id_layers
from mitiq.benchmarks.ghz_circuits import generate_ghz_circuit
from qiskit.circuit import Gate
from collections import defaultdict
import functools
import tomlkit
import toml

class OverheadExecutors():

    def __init__(self):
        self.count_circs = []
        self.count_1qbit_gates = []
        self.count_2qbit_gates = []
        self.depths = []

    # count_circs = []
    # count_1qbit_gates = []
    # count_2qbit_gates = []
    # depths = []

    def execute_w_metadata(self, circuit, shots, backend):
    
        global count_circs, count_1qbit_gates, count_2qbit_gates, depths

        qc = circuit.copy()
        qc.measure_all()
        num_qubits = qc.num_qubits

        depth = qc.depth()

        gate_counts = {1: 0, 2: 0}
        for inst in circuit.data:
            if isinstance(inst.operation, Gate):
                qubit_count = len(inst.qubits)
                gate_counts[qubit_count] = gate_counts.get(qubit_count, 0) + 1

        self.count_circs.append(1)
        self.count_1qbit_gates.append(gate_counts[1])
        self.count_2qbit_gates.append(gate_counts[2])
        self.depths.append(depth)


        transpiled_qc = transpile(qc, backend=backend, optimization_level=0)
        result = backend.run(transpiled_qc, shots=shots, optimization_level=0).result()
        counts = result.get_counts(transpiled_qc)

        total_shots = sum(counts.values())
        probabilities = {state: count / total_shots for state, count in counts.items()}

        expectation_value = probabilities['0'*num_qubits]+probabilities['1'*num_qubits]

        return expectation_value
    


    def write_metadata(self, circuit, shots, backend):
    # def write_metadata(self, circuit, shots, backend, metadata_path):

        #doc = tomlkit.document()

        metadata_dict = {}

        metadata_dict = {x: set() for x in ['mit_circs', 'mit_1qbit_gates', 'mit_2qbit_gates', 'mit_depth', 'mit_exp_val']}

        new_exec = functools.partial(self.execute_w_metadata, shots=shots, backend=backend)

        mit_exp_val = zne.execute_with_zne(circuit, new_exec)

        metadata_dict["mit_circs"] = sum(self.count_circs)
        metadata_dict["mit_1qbit_gates"] = sum(self.count_1qbit_gates)
        metadata_dict["mit_2qbit_gates"] = sum(self.count_2qbit_gates)
        metadata_dict["mit_depth"] = sum(self.depths)
        metadata_dict["mit_exp_val"] = mit_exp_val

        #doc.update(metadata_dict)
        #doc.add(tomlkit.nl())

        #with open(metadata_path, 'w') as f:
        #    f.write(tomlkit.dumps(doc))

        self.count_circs = []
        self.count_1qbit_gates = []
        self.count_2qbit_gates = []
        self.depths = []

        return metadata_dict


        

