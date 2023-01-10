# Zero Noise Extrapolation

Zero-noise extrapolation (ZNE) is an error mitigation technique in which an expectation
value is computed at different noise levels and, as a second step, the ideal
expectation value is inferred by extrapolating the measured results to the zero-noise
limit (see the section [What is the theory behind ZNE?](zne-5-theory.myst.md)).

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
zne-1-intro.myst.md
zne-2-use-case.myst.md
zne-3-options.myst.md
zne-4-low-level.myst.md
zne-5-theory.myst.md
```
Here are some examples on how to use ZNE in Mitiq:
- [Zero-noise extrapolation with Qiskit on IBMQ backends](../examples/ibmq-backends.myst.md)
- [Zero-noise extrapolation with Pennylane on IBMQ backends](../examples/pennylane-ibmq-backends.myst.md)
- [Zero-noise extrapolation with Braket on the IonQ backend](../examples/zne-braket-ionq.myst.md)
- [Zero-noise extrapolation of the energy landscape of a variational circuit with Cirq on a simulator](../examples/simple-landscape-cirq.myst.md)

You can find many more in the **[Examples](../examples/examples.myst.md)** section of the documentation.