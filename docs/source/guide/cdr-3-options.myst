---
jupytext:
  text_representation:
    extension: .myst
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# What additional options are available in CDR?

In addition to the four necessary ingredients shown in [How do I use CDR?](cdr-1-intro.myst), there are additional parameters in CDR.

One option is how many circuits are in the training set (default is 10). This can be changed as follows.

```{code-cell} ipython3
import warnings
warnings.filterwarnings("ignore")

import numpy as np

import cirq
from mitiq import cdr, Observable, PauliString
from mitiq.interface.mitiq_cirq import compute_density_matrix

a, b = cirq.LineQubit.range(2)
circuit = cirq.Circuit(
    cirq.H.on(a), # Clifford
    cirq.H.on(b), # Clifford
    cirq.rz(1.75).on(a),
    cirq.rz(2.31).on(b),
    cirq.CNOT.on(a, b),  # Clifford
    cirq.rz(-1.17).on(b),
    cirq.rz(3.23).on(a),
    cirq.rx(np.pi / 2).on(a),  # Clifford
    cirq.rx(np.pi / 2).on(b),  # Clifford
)
circuit = 5 * circuit

obs = Observable(PauliString("ZZ"), PauliString("X", coeff=-1.75))

def simulate(circuit: cirq.Circuit) -> np.ndarray:
    return compute_density_matrix(circuit, noise_level=(0.0,))

cdr.execute_with_cdr(
    circuit,
    compute_density_matrix,
    observable=obs,
    simulator=simulate,
    seed=0,
    num_training_circuits=20,
).real
```

+++

## Fit function

Another option is which fit function to use for regression (default is {func}`cdr.linear_fit_function`).
```{code-cell} ipython3
cdr.execute_with_cdr(
    circuit,
    compute_density_matrix,
    observable=obs,
    simulator=simulate,
    seed=0,
    fit_function=cdr.linear_fit_function_no_intercept,
).real
```

Beyond the built-in {func}`cdr.linear_fit_function` and {func}`cdr.linear_fit_function_no_intercept`,
the user could also define other custom functions.

## Variable noise CDR

+++

The `circuit` and the associated training circuits can also be run at different noise scale factors to implement [variable noise Clifford data regression](https://arxiv.org/abs/2011.01157) {cite}`Lowe_2021_PRR`.

```{code-cell} ipython3
from mitiq.zne import scaling

cdr.execute_with_cdr(
    circuit,
    compute_density_matrix,
    observable=obs,
    simulator=simulate,
    seed=0,
    scale_factors=(1.0, 3.0),
).real
```
