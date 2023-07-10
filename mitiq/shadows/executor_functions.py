from typing import List, Any
import cirq
import numpy as np
import qiskit
from qiskit_aer import Aer
from tqdm.auto import tqdm

from mitiq.interface.mitiq_qiskit.conversions import to_qiskit
from mitiq.typing import MeasurementResult


# cirq simulator
def cirq_simulator_shadow_executor_fn(
    circuits: List[cirq.Circuit],
) -> List[MeasurementResult]:
    outcomes = []
    simulator = cirq.Simulator()
    for circuit in tqdm(circuits, desc="Cirq Measurement"):
        result = simulator.run(circuit, repetitions=1)
        measurements = {}
        for key, value in result.measurements.items():
            # "q(0)" --> 0, "q(19)" --> 19, etc.
            measurements[int(key[2:-1])] = value[0][0]
        measurements = dict(sorted(measurements.items()))
        bitstrings = [list(measurements.values())]
        qubit_indices = tuple(measurements.keys())
        outcomes.append(MeasurementResult(bitstrings, qubit_indices))
    return outcomes


# qiskit simulator
def qiskit_simulator_shadow_executor_fn(
    cirq_circuits: List[cirq.Circuit], backend: Any = "aer_simulator"
) -> List[MeasurementResult]:
    outcomes = []
    if isinstance(backend, str):
        backend = Aer.get_backend(backend)
    noise_model = None
    for cirq_circuit in tqdm(cirq_circuits, desc="Qiskit Measurement"):
        circuit = to_qiskit(cirq_circuit)
        print(circuit)
        job = qiskit.execute(
            experiments=circuit,
            backend=backend,
            optimization_level=1
            if noise_model is None
            else 0,  # Important to preserve folded gates.
            shots=1,
            memory=True,
        )
        result = job.result()
        # print all functions of result
        # print(dir(result))
        # print(result.get_memory())
        counts = result.get_counts()

        assert len(counts) == 1
        bitstrings = list(counts.keys())[0].split(sep=" ")

        bitstrings = np.flip(bitstrings, axis=0)

        qubit_indices = tuple(range(len(bitstrings)))
        bitstrings = [[int(bitstring) for bitstring in bitstrings]]
        outcomes.append(MeasurementResult(bitstrings, qubit_indices))
    return outcomes
