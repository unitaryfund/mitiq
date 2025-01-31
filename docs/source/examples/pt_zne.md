---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

```{tags} cirq, zne, pt, intermediate
```
# Zero-Noise Extrapolation with Pauli Twirling

This tutorial explores how noise tailoring can improve the effectiveness of quantum error mitigation techniques.
Specifically, we analyze how converting coherent noise into incoherent noise through [Pauli Twirling (PT)](../guide/pt.md) 
 impacts the performance of [Zero-Noise Extrapolation](../guide/zne.md) (ZNE).

In this tutorial, we will:

1. Define and compare coherent and incoherent noise
2. Apply Pauli Twirling (PT) to transform coherent noise into incoherent noise
3. Compare the performance of ZNE
    1. on its own, and
    2. in combination with Pauli Twirling

By the end of the example, you will understand when and how noise tailoring can enhance ZNE.

## Coherent noise vs. Incoherent noise

Noise on quantum devices can be broadly categorized into two types: _coherent_ and _incoherent_. Each has different properties that can be unfavorable toward a quantum circuit in different ways.

**Coherent noise** is a reversible process as long as the noisy unitary transformation is known beforehand which is not always the case. These types of noise maintain the purity of the state. But in a quantum circuit subjected to coherent noise, the errors are easily carried across the circuit. This can be discerned through the **average gate infidelity** $r(\mathcal{E})$ . When coherent errors contribute to a portion of the total error-rate, the worst case infidelity can scale as $\sqrt{r(\mathcal{E})}$ {cite}`Wallman_2014` which in turn reduces the performance of a quantum device by orders of magnitude. Thus, dealing with coherent noise requires a large resource overhead to acquire inferred knowledge of the noise unitaries which can then be used to reverse the effects.

```{note}
If $\mathcal{F}$ is the average noisy gate fidelity defining the success of preparing an arbitrary pure state $\rho$, then
$1-\mathcal{F}$ is the **average gate infidelity**. 

$\mathcal{F}$ is associated with evolving a pure state through a noisy channel and then returning it to the original state {cite}`Nielsen_2002`.
```

**Incoherent noise** is a process that results in the quantum system entangling with its environment i.e. this type of noise is irreversible. The system and the environment end up in a mixed state. This type of noise scales linearly in the small error limit. The noise channel can be described using Pauli operators which makes it easy to analyze and simulate. Worst case error rate is directly proportional to the **average gate infidelity**. 

For example, a depolarizing noise channel is a stochastic noise channel where a noiseless process is probabilistically mixed with orthogonal errors. Pauli twirling strives to convert some noise channel into a Pauli noise channel. As shown in Eq.{math:numref}`depolarizing_noise` and {math:numref}`depolarizing_noise_paulis`, a local depolarizing noise channel can be described using Paulis i.e. it can be said that Pauli twirling tailors a noise channel into a local depolarizing noise channel {cite}`Garc_a_Mart_n_2024`.  

If $\rho$ is a single qubit state, $p$ is the probabilistic error rate and $\mathcal{E}(\rho)$ is the noise channel:

$$
\mathcal{E}(\rho) = (1-p) \rho + p \frac{I}{2}
$$ (depolarizing_noise)

$\frac{I}{2}$ is the maximally mixed state which can be described using Paulis.

$$
\frac{I}{2} = \frac{1}{4} (\rho + X \rho X + Y \rho Y + Z \rho Z)
$$

Thus, the depolarizing channel can be redescribed using Paulis as shown below. 

$$
\mathcal{E}(\rho) = (1-\frac{3p}{4}) \rho + \frac{p}{4} (X \rho X + Y \rho Y + Z \rho Z)

$$(depolarizing_noise_paulis)

### Pauli Transfer Matrix (PTM)

Let $\Lambda(\rho)$ be a $n$-qubit noise channel with $K_i$ being the corresponding Kraus operators. For a Pauli channel,
we can use $P_i$ and $P_j$ to be the Kraus operators in Eq.{math:numref}`CPTP_map`.

$$
\Lambda(\rho) = \sum_{i=1} K_i \rho {K_i}^\dagger 
$$(CPTP_map)

If $P_i$ and $P_j$ are lexicographically ordered $n$-qubit Paulis $\forall P_i, P_j \in \{I, X, Y, Z \}^{\otimes n}$, Eq. {math:numref}`PTM_expression` defines the entries of a Pauli Transfer Matrix (PTM). Here, 
$i$ defines the rows while $j$ defines the columns of the PTM.

$$
(R_{\Lambda})_{ij} = \frac{1}{2^n} \text{Tr} \{ P_i \Lambda(P_j)\}
$$(PTM_expression)

All entries of the PTM are real and in the interval $[-1, 1]$. A PTM allows us to distinguish between the two types of noise. The off-diagonal terms of the PTM are due to the effect of coherent noise while the diagonal terms are due to incoherent noise. To find the PTM of an entire circuit, we only need to take the product of the PTM of each layer in the circuit. Due to this, it is straightforward to see how coherent noise carries across different layers in the circuit and how incoherent errors are easier to deal with in the small error limit. The latter is due to only focusing on the diagonal terms of the PTM for incoherent noise such that the product of two or more diagonal matrices is also a diagonal matrix.

The known fault tolerant thresholds for stochastic noise are higher than coherent noise which makes the former a 'preferable' type of noise compared to the latter. To avoid dealing with coherent noise, Pauli twirling can be used to tailor coherent noise to incoherent noise. Same as Eq {math:numref}`depolarizing_noise_paulis`, when a coherent noise channel is Pauli twirled, the noise channel can be described using Paulis after averaging over multiple Pauli twirled circuits. Refer to the [Pauli Twirling user guide](../guide/pt.md) for additional information. 

It is worth noting that the number of Pauli twirled circuits required to transform coherent noise to incoherent noise differs
across the circuit used, noise stength, etc. The higher the number of generated twirled circuits, the better the result. Similarly, we get better results from Pauli twirling when the coherent noise strength is closer to the small error limit.

## Using Pauli Twirling in Mitiq

```{code-cell} ipython3

import cirq
import numpy as np
import numpy.typing as npt
from cirq.circuits import Circuit
from itertools import product
from functools import reduce

from mitiq.pec.channels import _circuit_to_choi, choi_to_super
from mitiq.utils import matrix_to_vector, vector_to_matrix

pauli_unitary_list = [
    cirq.unitary((cirq.I)),
    cirq.unitary((cirq.X)),
    cirq.unitary((cirq.Y)),
    cirq.unitary((cirq.Z)),
]


def n_qubit_paulis(num_qubits: int) -> list[npt.NDArray[np.complex64]]:
    """Get a list of n-qubit Pauli unitaries."""
    if num_qubits < 1:
        raise ValueError("Invalid number of qubits provided.")

    # get the n-qubit paulis from the Pauli group
    # disregard the n-qubit paulis with complex phase

    n_qubit_paulis = [reduce(lambda a, b: np.kron(a, b), combination)
        for combination in product(pauli_unitary_list, repeat=num_qubits)]
    return n_qubit_paulis


def pauli_vectorized_list(num_qubits: int) -> list[npt.NDArray[np.complex64]]:
    """Define a function to create a list of vectorized matrices.

    If the density matrix of interest has more than n>1 qubits, the
    Pauli group is used to generate n-fold tensor products before
    vectorizing the unitaries.
    """
    n_qubit_paulis1 = n_qubit_paulis(num_qubits)
    output_pauli_vec_list = []
    for i in n_qubit_paulis1:
        # the matrix_to_vector function stacks rows in vec form
        # transpose is used here to instead stack the columns
        matrix_trans = np.transpose(i)
        output_pauli_vec_list.append(matrix_to_vector(matrix_trans))
    return output_pauli_vec_list


def ptm_matrix(circuit: Circuit, num_qubits: int) -> npt.NDArray[np.complex64]:
    """Find the Pauli Transfer Matrix (PTM) of a circuit."""

    superop = choi_to_super(_circuit_to_choi(circuit))

    vec_pauli = pauli_vectorized_list(num_qubits)
    n_qubit_paulis1 = n_qubit_paulis(num_qubits)
    ptm_matrix = np.zeros([4**num_qubits, 4**num_qubits], dtype=complex)

    for i in range(len(vec_pauli)):
        superop_on_pauli_vec = np.matmul(superop, vec_pauli[i])
        superop_on_pauli_matrix_transpose = vector_to_matrix(
            superop_on_pauli_vec
        )
        superop_on_pauli_matrix = np.transpose(
            superop_on_pauli_matrix_transpose
        )

        for j in range(len(n_qubit_paulis1)):
            pauli_superop_pauli = np.matmul(
                n_qubit_paulis1[j], superop_on_pauli_matrix
            )
            ptm_matrix[j, i] = (0.5**num_qubits) * np.trace(
                pauli_superop_pauli
            )

    return ptm_matrix
```

Let us consider a simple circuit of a CNOT gate. We are going to subject this circuit to two types of noise and compare
their respective PTMs.

```{code-cell} ipython3
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

from cirq import LineQubit, Circuit, CNOT, Ry, depolarize, X, Y, Z

q0 = LineQubit(0)
q1 = LineQubit(1)

circuit = Circuit(CNOT(q0, q1))
print(circuit)
```
```{code-cell} ipython3
ptmcnot = ptm_matrix(circuit, 2)
ax = sns.heatmap(ptmcnot.real, linewidth=0.5, vmin=-1, vmax=1, cmap="PiYG")
ax.set_title('Ideal CNOT PTM')
plt.show()
```
```{code-cell} ipython3
# PTM of a noisy CNOT gate: depolarizing noise
noisy_circuit_incoherent = circuit.with_noise(depolarize(p=0.3))
print(noisy_circuit_incoherent)

ptmcnot = ptm_matrix(noisy_circuit_incoherent, 2)
ax = sns.heatmap(ptmcnot.real, linewidth=0.5, vmin=-1, vmax=1, cmap="PiYG")
ax.set_title('PTM of noisy CNOT (incoherent)')
plt.show()
```
```{code-cell} ipython3
# PTM of a noisy CNOT gate: Rz
noisy_circuit_coherent = circuit.with_noise(Ry(rads=np.pi/12))
print(noisy_circuit_coherent)

ptmcnot = ptm_matrix(noisy_circuit_coherent, 2)
ax = sns.heatmap(ptmcnot.real, linewidth=0.5, vmin=-1, vmax=1, cmap="PiYG")
ax.set_title('PTM of noisy CNOT (coherent)')
plt.show()
```
If we compare the PTM of the ideal CNOT gate to those when the gate was subjected to incoherent noise and coherent noise, 
there are additional sources of errors to deal with when coherent noise is acting on the CNOT gate. These can be reduced or tailored to be close to how the incoherent noise PTM appears through Pauli Twirling.

```{code-cell} ipython3
from mitiq.pt import generate_pauli_twirl_variants

# Generate twirled circuits
NUM_TWIRLED_VARIANTS = 3
twirled_circuits = generate_pauli_twirl_variants(
    circuit, num_circuits=NUM_TWIRLED_VARIANTS)
print("Example ideal twirled circuit", twirled_circuits[-1], sep="\n")
```
Now, lets add coherent noise to the CNOT gate in each twirled circuit.
```{code-cell} ipython3

noisy_twirled_circuits = []

for circ in twirled_circuits:
    split_circuit = Circuit(circ[0], circ[1], Ry(rads=np.pi/12)(q0), Ry(rads=np.pi/12)(q1), circ[-1])
    noisy_twirled_circuits.append(split_circuit)

print("Example noisy twirled circuit", noisy_twirled_circuits[-1], sep="\n")
```

The twirled PTM is averaged over each noisy twirled circuit such that the new PTM is close to that of the PTM of incoherent noise. We skip the step in this section as we require a very large number of twirled circuits to demonstrate the desired effect of averaging over multiple numpy arrays. The variations in pauli twirled PTMs are shown below when averaged over a different number of pauli twirled circuits.

```{figure} ../img/pt_zne_3_circuits.png
```
```{figure} ../img/pt_zne_5_circuits.png
```
```{figure} ../img/pt_zne_30_circuits.png
```


## Noisy ZNE

Lets define a larger circuit of CNOT and H gates. 

```{code-cell} ipython3

from mitiq.benchmarks import generate_ghz_circuit

circuit = generate_ghz_circuit(n_qubits=7)

print(circuit)

```

We are going to add coherent noise to this circuit and then get the error-mitigated expectation value. For a detailed discussion on this, refer to the [ZNE user guide](../guide/zne-1-intro.md). 

As we are using a simulator, we have to make sure the noise model adds coherent noise to CZ/CNOT gates in our circuit. For this, `get_noise_model` is used to add noise to CZ/CNOT gates. See [PT user guide](../guide/pt-1-intro.md) for more. 

```{code-cell} ipython3
from numpy import pi
from cirq import CircuitOperation, CXPowGate, CZPowGate, DensityMatrixSimulator
from cirq.devices.noise_model import GateSubstitutionNoiseModel

def get_noise_model(noise_level: float) -> GateSubstitutionNoiseModel:
    """Substitute each CZ and CNOT gate in the circuit
    with the gate itself followed by an Ry rotation on the output qubits.
    """
    rads = pi / 2 * noise_level
    def noisy_c_gate(op):
        if isinstance(op.gate, (CZPowGate, CXPowGate)):
            return CircuitOperation(
                Circuit(
                    op.gate.on(*op.qubits), 
                    Ry(rads=rads).on_each(op.qubits),
                ).freeze())
        return op

    return GateSubstitutionNoiseModel(noisy_c_gate)

def execute(circuit: Circuit, noise_level: float):
    """Returns Tr[ρ |0⟩⟨0|] where ρ is the state prepared by the circuit."""
    return (
        DensityMatrixSimulator(noise=get_noise_model(noise_level=noise_level))
        .simulate(circuit)
        .final_density_matrix[0, 0]
        .real
    )


# Set the intensity of the noise
NOISE_LEVEL = 0.2


# Compute the expectation value of the |0><0| observable
# in both the noiseless and the noisy setup
ideal_value = execute(circuit, noise_level=0.0)
noisy_value = execute(circuit, noise_level=NOISE_LEVEL)

NUM_TWIRLED_VARIANTS = 300
twirled_circuits = generate_pauli_twirl_variants(circuit, num_circuits=NUM_TWIRLED_VARIANTS)

# Average results executed over twirled circuits
from functools import partial
from mitiq import Executor
pt_vals = Executor(partial(execute, noise_level=NOISE_LEVEL)).evaluate(twirled_circuits)
twirled_result = np.average(pt_vals)


print(f"Error without twirling: {abs(ideal_value - noisy_value) :.3}")
print(f"Error with twirling: {abs(ideal_value - twirled_result) :.3}")
```

```{caution}
It is worth noting that Pauli twirling's goal is to tailor the noise from coherent to incoherent. Depending on the noise strength, type of coherent noise etc. this transformation might not give better results after the Pauli twirled circuit is executed. See the plot in the next section for more information.
```

## Combining Pauli Twirling with ZNE

```{code-cell} ipython3
from mitiq.zne import execute_with_zne

executor=partial(execute, noise_level=NOISE_LEVEL)
zne_pt_vals = []

for i in twirled_circuits:
    zne_pt_vals.append(execute_with_zne(i, executor))

mitigated_result = np.average(zne_pt_vals)

print(f"Error without twirling: {abs(ideal_value - noisy_value) :.3}")
print(f"Error with twirling: {abs(ideal_value - twirled_result) :.3}")
print(f"Error with ZNE + PT: {abs(ideal_value - mitigated_result) :.3}")

```
Accordingly, depending on the noise strength, a combination of PT and ZNE do not work that well compared to just PT or ZNE. Thus, it is important to understand when combining a noisy tailoring technique with an error mitigation technique provides a significant advantage. 

```{code-cell} ipython3

import matplotlib.pyplot as plt

# Plot error vs noise strength
noise_strength = np.linspace(0.0, 1.0, 50)
error_no_zne_no_twirling = []
error_with_twirling = []
error_with_twirling_and_zne = []
zne_vals = []
ideal_values = []
NUM_TWIRLED_VARIANTS = 30

for strength in noise_strength:
    ideal_value = execute(circuit, noise_level=0.0)
    ideal_values.append(ideal_value)
    noisy_value = execute(circuit, noise_level=strength)
    id_noisy_diff = abs(ideal_value-noisy_value)
    error_no_zne_no_twirling.append(id_noisy_diff)

    twirled_circuits = generate_pauli_twirl_variants(circuit, num_circuits=NUM_TWIRLED_VARIANTS)
    pt_vals = Executor(partial(execute, noise_level=strength)).evaluate(twirled_circuits)
    twirled_result = np.average(pt_vals)
    twirled_noisy_diff = abs(ideal_value-twirled_result)
    error_with_twirling.append(twirled_noisy_diff)

    executor=partial(execute, noise_level=strength)
    zne_vals.append(abs(ideal_value - execute_with_zne(circuit, executor)))
    
    zne_pt_vals = []
    for i in twirled_circuits:
        zne_pt_vals.append(execute_with_zne(i, executor))
    mitigated_twirled_result = np.average(zne_pt_vals)
    error_with_twirling_and_zne.append(abs(ideal_value - mitigated_twirled_result))


plt.plot(noise_strength, error_no_zne_no_twirling,"", label=r"|Ideal - Noisy|", color="#1f77b4")
plt.plot(noise_strength, zne_vals, "", label=r"|Ideal - ZNE|", color="#bcbd22")
plt.plot(noise_strength, error_with_twirling,"", label=r"|Ideal - Twirling|", color="#ff7f0e")
plt.plot(noise_strength, error_with_twirling_and_zne, "", label=r"|Ideal - (ZNE + Twirling)|", color="#2ca02c")

plt.xlabel(r"Noise strength, Coherent noise:$R_y(\frac{\pi}{2} \times \text{noise_strength})$")
plt.ylabel("Absolute Error")
plt.title("Comparison of expectation values with ideal as a function of noise strength")
plt.legend()
plt.show()
```

As we are plotting the difference between the ideal expectation value and the noisy, error-mitigated and/or noise-tailored
expectation values, the closer the curve is to `0.0` on the Y-axis, the technique provides an advantage.


```{warning}
You can get better results if you control the number of samples in `noise_strength` in addition to using a higher number for
`NUM_TWIRLED_VARIANTS`. We have chosen to not do so to reduce execution time for this tutorial.
```

## Conclusion

In this tutorial, we've shown how to use a noise tailoring method with Zero-Noise Extrapolation.
If you're interested in finding out more about these techniques, check out their respective sections of the users guide: [ZNE](../guide/zne.md), [Pauli Twilring](../guide/pt.md).
