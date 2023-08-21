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

# Classical Shadows

Classical shadows protocol {cite}`huang2020predicting` aims to create an approximate classical representation
of a quantum state using minimal measurements. The protocol is based on the idea of shadow tomography,
which is a technique for reconstructing a quantum state from a small number of measurements.
This approach not only characterizes and mitigates noise effectively but also retains
sample efficiency and demonstrates noise resilience {cite}`chen2021robust`. For more details, see the section
([Classical Shadow Protocol and its Robust Estimation](shadows-5-theory.md)).

 
```{figure} ../img/shadows_workflow.png
---
width: 700px
name: shadows-workflow-overview
---
```

```{figure} ../img/rshadows_workflow.png
---
width: 700px
name: rshadows-workflow-overview
---
Workflow of the robust shadow estimation (RSE) in Mitiq.
```

You can get started with shadows in Mitiq with the following sections of the user guide:

```{toctree}
---
maxdepth: 1
---
shadows-1-intro.md
shadows-5-theory.md
```

Here are some examples on how to use shadows in Mitiq:

- [Classical Shadows Protocal with Cirq](../examples/shadows_tutorial.md)
- [Robust Shadows Estimation with Cirq](../examples/rshadows_tutorial.md)

