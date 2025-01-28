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

# What additional options are available when using VD?

The function used for virtual distillation is `execute_with_vd()`. The standard mitiq error mitigation options are available. 

```{code-cell} ipython3
from mitiq.vd import execute_with_vd

vd_value = execute_with_vd(
  circuit,
  executor,
  observable,
  K_iters,
  num_copies # TODO
)
```

Next to the standard options, another argument has to be given, `K_iters`, which decides for how many iterations the VD algorithm should run. The `num_copies` argument decides how many copies are used in the VD algorithm. Currently only `num_copies=2` is supported. 