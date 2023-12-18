---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# What happens when I use CDR?

A figure of the typical workflow of CDR as implemented in Mitiq is shown in the figure below.

```{figure} ../img/cdr_workflow2_steps.png
---
width: 500px
name: cdr-workflow
---
The CDR workflow in Mitiq is divided in two steps: Generating circuits, both for a classical simulator and on the intended backend, and then performing the inference from measurements to obtain a noise mitigated expectation value.
```

Similarly to ZNE and PEC, CDR is divided in two main stages: first, one of circuit generation and a second for inference of the mitigated value.
In CDR, the generation of quantum circuits is different, as it involves the generation of training circuits.

```{warning}
In {cite}`Czarnik_2021_Quantum`, the authors lay out two different methods for generating the training circuits.

1. Randomly replacing gates in the target circuit with nearby Clifford gates.
2. Construct new circuits with the use of a Markov Chain Monte Carlo (MCMC) which produce classicaly simulable states.

The authors of {cite}`Czarnik_2021_Quantum` derive results with the use of the MCMC method, whereas Mitiq uses simpler approach presented in point 1.
```

The division of CDR into training, learning and prediction stages is shown more generally in the figure below.

```{figure} ../img/cdr_diagram2.png
---
width: 300px
name: cdr-process
---
Near-Clifford approximations of the actual circuit are simulated, without noise, on a classical simulator (circuits can be efficiently simulated classically) and executed on the noisy quantum computer (or a noisy simulator). These results are used as training data to infer the zero-noise expectation value of the error miitigated original circuit, that is finally run on the quantum computer (or noisy simulator).
```
