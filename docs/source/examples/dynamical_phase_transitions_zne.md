---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Introduction

This tutorial replicates some of the results from Y. Javanmard et al., ["Quantum simulation of dynamical phase transitions in noisy quantum devices"](https://arxiv.org/abs/2211.08318). We build a circuit that simulates the the time-evolution of a transverse-field Ising model (Eq. 1) after a quench at time $t = 0$, then run ideal, noisy, and noise-mitigated simulations of the circuit.

Assuming the system is in state $\psi_0 = H^{\otimes N} \ket { 0^{\otimes N} }$ at $t = 0$, we want to compute the probability of returning to the initial state at time $t$, $\Lambda(t) = \left|\bra{\psi_0}U(t)\ket{\psi_0}\right|^2$. To simulate the behavior over the interval $[0, t_{\textrm{max}}]$, we divide it into $M$ steps. Letting $\delta t = t_{\textrm{max}}/M$, we construct circuits corresponding to $U(t) = [U(\delta t)]^k$ for $k = 0, \ldots, m$.

+++

# Circuit definition

The following function returns the circuit used to compute $\Lambda(k \delta t)$, shown in Fig. 1 of the paper. The parameters used for the $R_{ZZ}$, $R_{XX}$, and $R_X$ gates depend on two parameters from the transverse-field Ising model, the spin-spin couplings $J_z$ and $J_x$ and the transverse field strength $h_x$. For simplicity, here we set $J_z = 1$, and follow Javanmard et al. in setting $J_x = h_x = 0.1J_z$.

```{code-cell} ipython3
from qiskit import QuantumCircuit

def create_circuit(delta_t : float,
                   k : int,
                   n_qubits: int = 6,
                   measure : bool = True) -> QuantumCircuit:
    theta = -delta_t
    phi = gamma = -0.1 * delta_t

    circuit = QuantumCircuit(n_qubits)
    
    circuit.h(circuit.qubits)
    
    for _ in range(k):
        for ii in range(0, n_qubits-1, 2):
            circuit.rzz(theta, ii, ii+1)
        for ii in range(1, n_qubits-1, 2):
            circuit.rzz(theta, ii, ii+1)
            
        for ii in range(0, n_qubits-1, 2):
            circuit.rxx(phi, ii, ii+1)
        for ii in range(1, n_qubits-1, 2):
            circuit.rxx(phi, ii, ii+1)
            
        circuit.rx(gamma, circuit.qubits)

    circuit.h(circuit.qubits)
    
    if measure:
        circuit.measure_all()

    return circuit
```

```{code-cell} ipython3
print(create_circuit(0.01, 1))
```

# Ideal simulation

We first run a series of noiseless state-vector simulations.

```{code-cell} ipython3
import numpy as np
import dataclasses

@dataclasses.dataclass
class SimulationParams:
    '''Simulation parameters. Note that we use much coarser
    time steps than in the paper, so that this demo runs in
    a reasonably short time.'''
    t_max : float = 8.5
    M : int = 25
    n_qubits : int = 6
    dt : float = dataclasses.field(init=False)
    t_vals : np.ndarray[float] = dataclasses.field(init=False)

    # Only used in noisy simulations
    n_shots : int = 2048

    def __post_init__(self):
        self.dt = self.t_max / self.M
        self.t_vals = np.linspace(0., self.t_max, self.M + 1, endpoint=True)
```

To get the ideal result for $\Lambda(t)$, we omit the measurements from the circuit, and take the probability amplitude $\bra{0^{\otimes N}} H^{\otimes N} U(t) H^{\otimes N} \ket{0^{\otimes N}}$ from the final state vector.

```{code-cell} ipython3
from qiskit_aer import QasmSimulator

def simulate_ideal(circuit: QuantumCircuit) -> float:
    simulator = QasmSimulator(method="statevector", noise_model=None)

    circuit.save_statevector()
    job = simulator.run(circuit, shots=1)
    
    psi = job.result().data()["statevector"]
    
    # Get the probability of returning to $\ket{0^{\otimes N}}$
    amp_0 = np.asarray(psi)[0]
    return np.abs(amp_0.real**2 + amp_0.imag**2)

def run_ideal(params : SimulationParams = SimulationParams()) -> tuple[np.ndarray[float]]:
    echo = np.array([
        simulate_ideal(
            create_circuit(params.dt, k, n_qubits=params.n_qubits, measure=False)
        )
        for k in range(0, params.M + 1)
    ])    

    return params.t_vals, echo
```

```{code-cell} ipython3
from matplotlib import pyplot as plt

def setup_plot(title : str = None):
    plt.figure(figsize=(6.0, 4.0))
    plt.xlabel("$t$")
    plt.ylabel("$\Lambda(t)$")
    if title is not None:
        plt.title(title)

def add_to_plot(x : np.ndarray[float],
                y : np.ndarray[float],
                label : str,
                legend : list[str]):
    if label == "ideal":
        plt.plot(x, y, color="black")
    elif label == "mitigated":
        plt.plot(x, y, marker='s', markersize=5)
    else:
        plt.plot(x, y, marker='.', markersize=10)
    legend.append(label)
```

```{code-cell} ipython3
# Run the state-vector simulation
result_ideal = run_ideal(SimulationParams(t_max=25.0, M=200))
```

```{code-cell} ipython3
setup_plot("State-vector simulation")
legend = []
add_to_plot(*result_ideal, "ideal", legend)
plt.legend(legend)
plt.show()
```

The result shows a series of Loschmidt echo peaks. Intuitively, in a noisy system we expect to see $\Lambda(t)$ rapidly approach $1/2^N$, where $N$ is the number of qubits. This means that as the noise level increases, the peaks will be suppressed, starting at larger values of $t$. At higher levels of noise, we may not be able to detect the Loschmidt echo at all.

In the rest of this notebook we will run simulations with two different noise models, and try to reconstruct the peak at $t \approx 6.5$ with zero-noise extrapolation.

```{code-cell} ipython3
# Re-run the simulation with the default value of t_max
result_ideal = run_ideal(SimulationParams(M=100))
setup_plot("State-vector simulation")
legend = []
add_to_plot(*result_ideal, "ideal", legend)
plt.legend(legend)
plt.show()
```

# Simulation with depolarizing noise

The next few cells run a simulation with depolarizing noise. Following the paper, we transpile the circuit with the basis gate set $\{u1, u2, u3, cx\}$, and optionally use gate folding to scale the noise.

```{code-cell} ipython3
from qiskit import transpile
from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise.errors.standard_errors import depolarizing_error
from mitiq.zne.scaling.folding import fold_gates_at_random

def simulate_noisy(circuit: QuantumCircuit,
                   noise_model: NoiseModel,
                   n_shots : int,
                   scale_factor: float = None) -> float:
    # Transpile the circuit
    backend = QasmSimulator(noise_model=noise_model)
    exec_circuit = transpile(
        circuit,
        backend=backend,
        basis_gates=["u1", "u2", "u3", "cx"],
        optimization_level=0
    )

    # Apply gate folding
    folded_circuit = exec_circuit if scale_factor is None \
                     else fold_gates_at_random(exec_circuit, scale_factor)
    
    job = backend.run(folded_circuit, shots=n_shots)

    # Get the probability of returning to $\ket{0^{\otimes N}}$
    counts = job.result().get_counts()
    ket = "0" * circuit.num_qubits
    if ket in counts:
        return counts[ket]/n_shots
    return 0.0

def run_depolarizing_noise(params : SimulationParams = SimulationParams(),
                           noise_level : float = 0.001,
                           scale_factor : float = None) -> tuple[np.ndarray[float]]:
    basis_gates = ["u1", "u2", "u3", "cx"]
    noise_model = NoiseModel(basis_gates)
    # Add depolarizing noise to the 1-qubit gates in the basis set
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(noise_level, 1), basis_gates[:3]
    )
    
    echo = np.array([
        simulate_noisy(
            create_circuit(params.dt, k, n_qubits=params.n_qubits, measure=True),
            noise_model,
            params.n_shots,
            scale_factor
        )
        for k in range(0, params.M + 1)
    ])    

    return params.t_vals, echo
```

```{code-cell} ipython3
result_depolarizing = run_depolarizing_noise(noise_level=0.0005)
```

```{code-cell} ipython3
setup_plot("Ideal and noisy simulations")
legend = []
add_to_plot(*result_ideal, "ideal", legend)
add_to_plot(*result_depolarizing, "depolarizing noise, $p = 0.0005$", legend)
plt.legend(legend)
plt.show()
```

As expected, we get a lower peak. Applying gate folding suppresses the peak further. This is qualitatively consistent with the result in Fig. 2(e) of Javanmard et al. For efficiency, this notebook uses a much coarser time step than in the paper, so we do not obtain the same results for a given value of the depolarizing noise strength.

```{code-cell} ipython3
scale_factors = [1, 2, 3]
result_depolarizing_scaled = [
    run_depolarizing_noise(noise_level=0.0005, scale_factor=alpha) for alpha in scale_factors
]
```

```{code-cell} ipython3
setup_plot("Scaled noisy simulations")
legend = []
add_to_plot(*result_ideal, "ideal", legend)
for alpha, result in zip(scale_factors, result_depolarizing_scaled):
    add_to_plot(*result, "$\\alpha = {}$".format(alpha), legend)
plt.legend(legend)
plt.show()
```

# Error mitigation with zero-noise extrapolation
At this level of noise, we can use ZNE to mostly recover the ideal result. Running the noisy simulation is expensive (especially as the scale factor $\alpha$ increases), so rather than using the high-level functions for ZNE, we apply the static method `RichardsonFactory.extrapolate` to the results from the previous cell.

```{code-cell} ipython3
from mitiq.zne.inference import RichardsonFactory

result_zne = RichardsonFactory.extrapolate(scale_factors,
                                           [r[1] for r in result_depolarizing_scaled])
```

```{code-cell} ipython3
setup_plot("ZNE for depolarizing noise model, $p = 0.0005$")
legend = []
add_to_plot(*result_ideal, "ideal", legend)
for alpha, result in zip(scale_factors, result_depolarizing_scaled):
    add_to_plot(*result, "$\\alpha = {}$".format(alpha), legend)
add_to_plot(result_depolarizing_scaled[0][0], result_zne, "mitigated", legend)
plt.legend(legend)
plt.show()
```

Increasing the baseline noise level makes it much harder to reconstruct the peak with ZNE.

```{code-cell} ipython3
scale_factors = [1, 2, 3]
result_depolarizing_scaled = [
    run_depolarizing_noise(noise_level=0.005, scale_factor=alpha) for alpha in scale_factors
]
```

```{code-cell} ipython3
result_zne = RichardsonFactory.extrapolate(scale_factors,
                                           [r[1] for r in result_depolarizing_scaled])
```

```{code-cell} ipython3
setup_plot("ZNE for depolarizing noise model, $p = 0.005$")
legend = []
add_to_plot(*result_ideal, "ideal", legend)
for alpha, result in zip(scale_factors, result_depolarizing_scaled):
    add_to_plot(*result, "$\\alpha = {}$".format(alpha), legend)
add_to_plot(result_depolarizing_scaled[0][0], result_zne, "mitigated", legend)
plt.legend(legend)
plt.show()
```

# Simulation with realistic device noise

Next, we run simulations using the noise model of the IBM Nairobi device. Again, the relatively high noise level makes it difficult to recover something close to the ideal signal.

```{code-cell} ipython3
from qiskit_ibm_runtime.fake_provider.backends import FakeNairobiV2

def run_ibm_nairobi_noise(params : SimulationParams = SimulationParams(),
                          scale_factor : float = None) -> tuple[np.ndarray[float]]:
    noise_model = NoiseModel.from_backend(FakeNairobiV2())
    
    echo = np.array([
        simulate_noisy(
            create_circuit(params.dt, k, n_qubits=params.n_qubits, measure=True),
            noise_model,
            params.n_shots,
            scale_factor
        )
        for k in range(0, params.M + 1)
    ])    

    return params.t_vals, echo
```

```{code-cell} ipython3
scale_factors = [1, 2, 3]
result_ibm_nairobi_scaled = [
    run_ibm_nairobi_noise(scale_factor=alpha) for alpha in scale_factors
]
```

```{code-cell} ipython3
result_zne = RichardsonFactory.extrapolate(scale_factors,
                                           [r[1] for r in result_ibm_nairobi_scaled])
```

```{code-cell} ipython3
setup_plot("ZNE for IBM Nairobi noise model")
legend = []
add_to_plot(*result_ideal, "ideal", legend)
for alpha, result in zip(scale_factors, result_ibm_nairobi_scaled):
    add_to_plot(*result, "$\\alpha = {}$".format(alpha), legend)
add_to_plot(result_depolarizing_scaled[0][0], result_zne, "mitigated", legend)
plt.legend(legend)
plt.show()
```
