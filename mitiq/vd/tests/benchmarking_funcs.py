
# general imports 
import sys
sys.path.insert(0, '..')
import numpy as np
import matplotlib.pyplot as plt
import cirq, cirq_google
import mitiq
import time
import pickle
from vd import execute_with_vd

# Generally usable progress bar :DDD

class progressBar():
    def __init__(self, maxTicks:int, size=40):
        self.maxTicks = maxTicks
        self.currentTicks = 0
        self.T0 = time.time()
        self.T1 = 0.0
        self.size = size
        progress_float = 0.0
        progress_str = " " * self.size
        print(f"Running: |{progress_str}|{progress_float:6.1%}", end='\r')

    def addTicks(self, NTicks):
        self.currentTicks += NTicks
        progress_float = self.currentTicks/self.maxTicks
        progress_str = "#" * round(self.size * progress_float) + " " * (self.size - round(self.size * progress_float))
        self.T1 = time.time()
        est_time = int((1-progress_float)/progress_float * (self.T1-self.T0))
        elapsed_time = int(self.T1-self.T0)
        print(f"Running: |{progress_str}|{progress_float:6.1%} \t T ~ { est_time//60:3d}m {est_time%60:2d}s left \t (tot: {(elapsed_time+est_time)//60:3d}m {(elapsed_time+est_time)%60:2d}s){' ':100}", end='\r')

    def finished(self):
        self.currentTicks = self.maxTicks
        progress_float = 1.0
        self.T1 = time.time()
        elapsed_time = int(self.T1-self.T0)
        print(f"Finished:|{'-'*((self.size-4)//2)}DONE{'-'*((self.size-3)//2)}|{progress_float:6.1%} \t T elapsed: { elapsed_time//60:3d}m {elapsed_time%60:2d}s {' ':100}")



def create_randomised_benchmarking_circuit(N_qubits, depth ,entanglement=True, seed=None):
    single_qubit_gates = np.array([cirq.XPowGate, cirq.YPowGate, cirq.ZPowGate])

    qubits = [cirq.LineQubit(i) for i in range(N_qubits)]
    
    if seed == None:
        RNG = np.random.default_rng()
    else:
        RNG = np.random.default_rng(seed)
    
    operations = []
    entanglement_gate_count = 0
    entanglement_offset = 0

    for l in range(depth + 1):
        layer_gate_numbers = RNG.choice(3, N_qubits) # N_qubit numbers of 0 to 2, identifying one of the s.q. gates from the list for each qubit
        layer_gate_exponents = RNG.choice([0.5, 1.0], N_qubits) # determine the exponent for each gate, so we apply eithet X or X^1/2, resp. for Y and Z
        for i in range(N_qubits):
            gate = single_qubit_gates[layer_gate_numbers[i]]
            operations.append(gate(exponent=layer_gate_exponents[i]).on(qubits[i], ))

        # add syncamore gates, alternating between left and right neighbours
        if entanglement and l < depth:
            for i in range(entanglement_offset, N_qubits, 2):
                if i+1 < len(qubits): 
                    operations.append(cirq_google.ops.SycamoreGate().on(qubits[i], qubits[i+1]))
            entanglement_offset = 1 - entanglement_offset # alternate between 0 and 1


    pure_rand_qc = cirq.Circuit(operations)


    return pure_rand_qc, len(operations), entanglement_gate_count

def expectation_Zi(circuit):
    """Returns Tr[ρ Z] where ρ is the state prepared by the circuit
    with depolarizing noise."""
    
    # density matrix
    dm = cirq.DensityMatrixSimulator().simulate(circuit).final_density_matrix 
    
    n_qubits = cirq.num_qubits(circuit)
    Z = np.array([[1, 0], [0, -1]])

    Zi_operators = np.array([ np.kron(np.kron(np.eye(2**i), Z), np.eye(2**(n_qubits-i-1))) for i in range(n_qubits)])

    # print(Zi_operators)

    Zi = np.trace(Zi_operators@ dm, axis1=1, axis2=2)
    return Zi

def vector_norm_distance(V, W):
    return np.linalg.norm((V - W))


def run_benchmarking(run_name, vd_iterations:list[int], N_datapoints=10,  N_qubits=6, N_layers=20, entangled=True):
    # Running and saving a benchmarking iteration

    import pickle


    rho, gate_count, entangle_gate_count = create_randomised_benchmarking_circuit(N_qubits, N_layers, entanglement=entangled)
    # (re)create a random circuit until at least one expectation value is nonzero
    true_Zi = expectation_Zi(rho) # Fault tolerant quantum computer
    while np.all(true_Zi == 0.+0.j):
        print("nope")
        rho, gate_count, entangle_gate_count = create_randomised_benchmarking_circuit(N_qubits, N_layers, entanglement=entangled)
        true_Zi = expectation_Zi(rho) # Fault tolerant quantum computer

    print(rho)
    dist_true_Zi = 0.0

    BMrun = {
        "benchmark_name": run_name,
        "observable": "Z",
        "N_qubits":N_qubits,
        "N_layers":N_layers,
        "rho":rho,
        "gate_count":gate_count,
        "entangle_gate_count":entangle_gate_count
    }

    assert input("Is the name correctly set? [N/y] ") == 'y', "Aborted, please change the name."



    pBar = progressBar(1 + 1 + N_datapoints * (1 + sum(vd_iterations)))
    pBar.addTicks(1)

    datapoints = []
    datapoint_labels = []

    for i, N_exp_Err in enumerate(np.logspace(-2, 0, base=10, num=N_datapoints)):

        noise_level = N_exp_Err / gate_count
        noisy_rho = rho.copy().with_noise(cirq.depolarize(p=noise_level))
        # print(noisy_rho)

        noisy_Zi = expectation_Zi(noisy_rho) # Noisy quantum computer
        dist_noisy_Zi = vector_norm_distance(true_Zi, noisy_Zi)

        # print(N_exp_Err, true_Zi, noisy_Zi)
        pBar.addTicks(1)

        measurement_list = [N_exp_Err, dist_noisy_Zi]
        datapoint_labels = ["Noisy"]

        for K in vd_iterations:
            vd_Zi = execute_with_vd(noisy_rho, 2, K) # Noisy quantum computer + virtual distillation
            dist_vd_Zi = vector_norm_distance(true_Zi, vd_Zi)
            measurement_list.append(dist_vd_Zi)
            datapoint_labels.append(f"vd: {K=}")
            pBar.addTicks(K)

        datapoints.append( tuple(measurement_list) )
        
    BMrun["true_Zi"] = true_Zi
    BMrun["datapoints"] = datapoints
    BMrun["datapoint_labels"] = datapoint_labels

    filename = BMrun["benchmark_name"].replace(" ", "_") + '.pkl'
    with open( filename, 'wb') as f:
        pickle.dump(BMrun, f)

    pBar.finished()

    print(filename)

    return BMrun

def plot_BM_run(file_name):

    with open(file_name, 'rb') as f:
        BM_dict = pickle.load(f)
    
    dist_random_state = vector_norm_distance(BM_dict["true_Zi"], np.zeros(BM_dict["N_qubits"]))

    BM_fig = present_in_plot(BM_dict["datapoints"], dist_random_state, BM_dict["observable"], BM_dict["datapoint_labels"], title=BM_dict["benchmark_name"])  

    return BM_fig


def present_in_plot(measurement_list, reference_value, observable_name:str ="Z", Y_vals_info=[], rel_err_on_Y=True , title="n qubits randomised identity circuit benchmark"):
    

    X, *Ys = list(zip(*measurement_list))
    xmin, xmax = 0.91 * np.min(X), 1.1 * np.max(X)

    assert len(Ys) == len(Y_vals_info), "mismath Ys and Y-infos"

    fig, ax = plt.subplots()
    fig.set_size_inches(8,6)
    hline_label = "random guessing error" if rel_err_on_Y else "true value"
    ax.axhline(reference_value, xmin, xmax, color='blue', lw="1", ls="-.", label=hline_label)

    for i, Y in enumerate(Ys):
        ax.plot(X, Y, label=Y_vals_info[i], lw=1, ms=4, marker='^')


    ax.set_xlabel("Expected # of errors")
    ax.set_xscale('log')
    if rel_err_on_Y:
        ax.set_yscale('log')
        ax.set_ylabel(f"square root distance of $\\langle {observable_name}_i \\rangle $ vector to true state")
    else:
        ax.set_ylabel(f"$\\langle {observable_name}_1 \\rangle $ of the first qubit")
        ax.set_ylim(-1.1, 1.1)
    ax.set_xlim(xmin, xmax)
    ax.legend()
    ax.set_title(title)

    return fig
