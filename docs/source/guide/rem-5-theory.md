---
jupytext:
  text_representation:
    extension: .myst
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# What is the theory behind REM?

Readout error mitigation (REM), is one of the most general and earliest studied error mitigation techniques, which can refer to a variety of specific approaches.

A simple version of readout error mitigation is postselection of bitstrings. For
example, if one knows that the measured bitstrings should preserve some
symmetry, bitstrings that do not preserve it can be discarded. Such capability 
is indeed available in {mod}`mitiq.rem.post_select`.

With regards to the more elaborate technique of confusion matrix inversion,
also [supported](rem-1-intro) in Mitiq, some relevant references are Refs. {cite}`Maciejewski_2020,Bravyi_2021,Garion_2021,Geller_2021`.