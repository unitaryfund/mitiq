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

# What additional options are available when using DDD?

Most additional functionalities when using DDD with Mitiq are related to the choice of dynamical decoupling sequences. 
Since the idle windows of a quantum circuit can have different sizes, one cannot directly define a unique fixed DDD sequence.
One can instead define a DDD _rule_, i.e., a Python function that generates a gate sequence from an input slack length. 

We'll discuss several ways of defining DDD rules:

1. Selecting from the built in rules;
2. Defining complex rules with the {func}`.general_rule()` and {func}`.repeated_rule()` functions;
3. Defining custom rules from scratch;
3. Nesting rules to fill long slack windows first, followed by shorter slack windows.

## Built-in DDD rules
Mitiq provides basic built in rules to approximate dynamical decoupling sequences that are most used and discussed
in the literature (for more details see [What is the theory behind DDD?](ddd-5-theory.myst)).
For example, the _XX_, _YY_, and _XYXY_ rules generate the corresponding gate sequences spaced evenly over the input slack window.
For each of these rules, the user may specify a different spacing between the gates in the sequence and pass the desired option as shown
in the next code cell.

```{code-cell} ipython3
import numpy as np

from mitiq import ddd
from cirq import LineQubit, Circuit, rx, rz, CNOT, X, Y, H, Z, SWAP, DensityMatrixSimulator, amplitude_damp

a, b = LineQubit.range(2)
circuit_one = Circuit(
    rx(0.1).on(a),
    rx(0.1).on(a),
    rz(0.4).on(a),
    rx(-0.72).on(a),
    rz(0.2).on(a),
    rx(0.1).on(a),
    rx(0.1).on(a),
    rz(0.4).on(a),
    rx(-0.72).on(a),
    rz(0.2).on(a),
    rx(-0.8).on(b),
    CNOT.on(a, b),
)

def execute(circuit, noise_level=0.1):
    """Returns Tr[ρ |0⟩⟨0|] where ρ is the state prepared by the circuit
    executed with amplitude damping noise.
    """
    # Replace with code based on your frontend and backend.
    noisy_circuit = circuit.with_noise(amplitude_damp(gamma=noise_level))
    rho = DensityMatrixSimulator().simulate(noisy_circuit).final_density_matrix
    return rho[0, 0].real

  
rule = ddd.rules.xx
mitigated_result = ddd.execute_with_ddd(
    circuit=circuit_one, 
    executor=execute, 
    rule=rule,
    rule_args={"spacing": 0},
)
```

```{note}
The default value of the `spacing` option is `-1`, which generates sequences with the maximum spacing that can fit the size of a slack window.
```

## More general sequences

If the user wishes to experiment with creating other gate sequences, a {func}`.general_rule()` is provided, which takes as input a list of gates and 
their spacing.
As an example, let's define a rule function that will generate an _XXYY_ sequence:

```{code-cell} ipython3
def xxyy(slack_length, spacing = -1):
    xxyy_sequence = ddd.rules.general_rule(
        slack_length=slack_length,
        spacing=spacing,
        gates=[X, X, Y, Y],
    )
    return xxyy_sequence

# Test
xxyy(slack_length=9)
```

A rule defined in this manner can similarly be used with {func}`.execute_with_ddd()` to error-mitigate expectation values as shown
at the top of this notebook.


To create a rule that generates repeated DDD sequences, we can use the {func}`.repeated_rule()` abstraction.
The function {func}`.repeated_rule()` fills slack windows with as many repetitions as possible of some input elementary sequence. 

```{code-cell} ipython3
def repeated_xxyy(slack_length):
    return ddd.rules.repeated_rule(slack_length=slack_length, gates=[X, X, Y, Y])

# Test
repeated_xxyy(8)
```

## Custom DD rules

Since in Mitiq a DDD rule is just a Python function, the user can define a custom rule from scratch. For example,
the following rule returns sequences of Hadamard gates only if `slack_length` is 2 or 4. 

```{code-cell} ipython3
import numpy as np 

def custom_rule(slack_length: int) -> Circuit:
    q = LineQubit(0)
    if slack_length == 2:
        sequence = Circuit([H(q), H(q)])
    elif slack_length == 4:
        sequence = Circuit([H(q), H(q), H(q), H(q)])
    else:
        sequence = Circuit()
    return sequence

# Test
print(custom_rule(2))
print(custom_rule(4))
```

## Nested rules

Suppose a user wants to mix sequences where, for example, _XYXY_ is applied first to long slack windows and then _XX_ is applied
to all the shorter windows that are left over.

As demonstrated in detail in the [next user guide section](ddd-4-low-level.myst), the function {func}`.insert_ddd_sequences()`
is all one needs to apply DDD.
So, to apply two nested rules, one only needs to call {func}`.insert_ddd_sequences()` twice as shown in the following example.

```{code-cell} ipython3
qreg = LineQubit.range(8)
x_layer = Circuit(X.on_each(qreg))
cnots_layer = Circuit(SWAP.on(q, q + 1) for q in qreg[:-1])
input_circuit = x_layer + cnots_layer + x_layer
input_circuit
```

```{code-cell} ipython3
long_rule = ddd.rules.xyxy
short_rule = ddd.rules.xx
circuit_with_xyxy = ddd.insert_ddd_sequences(input_circuit, rule=long_rule)
circuit_with_xyxy_and_xx = ddd.insert_ddd_sequences(circuit_with_xyxy, rule=short_rule)
circuit_with_xyxy_and_xx
```

As visible from the printed circuits, _XYXY_ sequences have been added in long windows, while _XX_ sequences have been added in short windows.

The associated unmitigated and mitigated expectation values are:

```{code-cell} ipython3
# Unmitigated expectation value
execute(input_circuit)
```

```{code-cell} ipython3
# Expectation value mitigated with nested DDD sequences
execute(circuit_with_xyxy_and_xx)
```
