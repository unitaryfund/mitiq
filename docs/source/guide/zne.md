# Zero Noise Extrapolation

Zero-noise extrapolation (ZNE) is an error mitigation technique in which an expectation
value is computed at different noise levels and, as a second step, the ideal
expectation value is inferred by extrapolating the measured results to the zero-noise
limit (see the section [What is the theory behind ZNE?](zne-5-theory.md)).

```{figure} ../img/zne_workflow2_steps.png
---
width: 700px
name: figzne-overview
---
The diagram shows the workflow of the zero noise extrapolation (ZNE) in Mitiq.
```

You can get started with ZNE in Mitiq with the following sections of the user guide:

```{toctree}
---
maxdepth: 1
---
zne-1-intro.md
zne-2-use-case.md
zne-3-options.md
zne-4-low-level.md
zne-5-theory.md
```
Here are some examples on how to use ZNE in Mitiq:
- [Zero-noise extrapolation with Qiskit on IBMQ backends](../examples/ibmq-backends.md)
- [Zero-noise extrapolation with Pennylane on IBMQ backends](../examples/pennylane-ibmq-backends.md)
- [Zero-noise extrapolation with Braket on the IonQ backend](../examples/zne-braket-ionq.md)
- [Zero-noise extrapolation of the energy landscape of a variational circuit with Cirq on a simulator](../examples/simple-landscape-cirq.md)

You can find many more in the **[Examples](../examples/examples.md)** section of the documentation.