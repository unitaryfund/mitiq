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

# How do I use shadows?
Shadows is a technique that can be used to {ref}`mitigate readout errors <guide/rem-1-intro>` in noisy quantum devices. It is compatible with any frontend supported by Mitiq:

```{code-cell} ipython3
import mitiq
