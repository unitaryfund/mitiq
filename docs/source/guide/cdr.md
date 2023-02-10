(guide/cdr/cdr)=
# Clifford Data Regression

Clifford data regression (CDR) is a learning-based quantum error mitigation technique in which an error mitigation model is trained with quantum circuits that _resemble_ the circuit of interest, but which are easier to classically simulate {cite}`Czarnik_2021_Quantum, Lowe_2021_PRR`.


```{figure} ../img/cdr_workflow2_steps.png
---
width: 700px
name: cdr-workflow-overview
---
The CDR workflow in Mitiq is fully explained in the [What happens when I use CDR?](cdr-4-low-level.md) section.
```

Below you can find sections of the documentation that address the following questions:


```{toctree}
---
maxdepth: 1
---
cdr-1-intro.md
cdr-2-use-case.md
cdr-3-options.md
cdr-4-low-level.md
cdr-5-theory.md
```

A simple tutorial on CDR can be found in the code blocks in the [Problem setup](cdr-1-intro.md) section of the first section of the user guide.
