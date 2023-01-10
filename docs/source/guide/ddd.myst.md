# Digital Dynamical Decoupling

Digital Dynamical Decoupling (DDD) is an error mitigation technique in which
sequences of gates are applied to slack windows, i.e. single-qubit idle
windows, in a quantum circuit. Such sequences of gates can reduce the coupling
between the qubits and the environment, mitigating the effects of noise.
For more discussion of the theory of DDD, see the section [What is the theory
behind DDD?](ddd-5-theory.myst.md).


```{figure} ../img/ddd_workflow.png
---
width: 700px
name: ddd-workflow-overview
---
Workflow of the DDD technique in Mitiq, detailed in the [What happens when I use DDD?](ddd-4-low-level.myst.md) section.
```

Below you can find sections of the documentation that address the following questions:


```{toctree}
---
maxdepth: 1
---
ddd-1-intro.myst.md
ddd-2-use-case.myst.md
ddd-3-options.myst.md
ddd-4-low-level.myst.md
ddd-5-theory.myst.md
```

Here is a tutorial on how to use DDD in Mitiq:

[DDD with Cirq: Mirror circuits](../examples/ddd_tutorial.myst.md)

You can find many more examples on a variety of error mitigation techniques in the **[Examples](../examples/examples.myst.md)** section of
the documentation.
