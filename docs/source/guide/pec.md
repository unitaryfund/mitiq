(guide/pec/pec)=
# Probabilistic error cancellation

Probabilistic error cancellation (PEC) is an error mitigation technique in which ideal operations are represented as linear combinations of noisy operations. In PEC, unbiased estimates of expectation values are obtained by Monte Carlo averaging over different noisy circuits (see [What is the theory behind PEC?](pec-5-theory.md) for more information on the theory).

```{figure} ../img/pec_workflow2_steps.png
---
width: 700px
name: pec-workflow-overview
---
Workflow of the implementation of PEC in Mitiq, further detailed in the [What happens when I use PEC?](pec-4-low-level.md) section.
```

You can read more about PEC in Mitiq in the following sections:

```{toctree}
---
maxdepth: 1
---
pec-1-intro.md
pec-2-use-case.md
pec-3-options.md
pec-4-low-level.md
pec-5-theory.md
```
Here are some tutorials on how to use PEC in Mitiq:
- [Learning quasiprobability representations with a depolarizing noise model](../examples/learning-depolarizing-noise.md)
- [Probabilistic error cancellation (PEC) with Mirror Circuits](../examples/pec_tutorial.md)

You can find many more examples on a variety of error mitigation techniques in the **[Examples](../examples/examples.md)** section of the documentation.