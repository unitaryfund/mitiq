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
also [supported](rem-1-intro) in Mitiq, the most relevant references are Refs. {cite}`Maciejewski_2020,Bravyi_2021`.
In Ref. {cite}`Maciejewski_2020`, a REM technique for NISQ devices has been
introduced in conjuction with the reconstruction of a Positive-Operator Valued
Measure (POVM) describing the measurement. The technique was there tested on a
Rigetti quantum processor and a 5-qubit IBM quantum processor.

In Ref. {cite}`Bravyi_2021` two REM schemes are introduced, based on tensor
product noise and correlated Markovian noise models. The tecniques have been
tested on the 20-qubit superconducting circuit processor ibmq-johannesburg.


Other related work on REM is performed in Ref. {cite}`Garion_2021` (in
conjuction with a custom randomized benchmarking technique) and in
Ref. {cite}`Geller_2021` (to correct SPAM errors, and tested on an IBM Q
processor with 4 and 8 superconducting qubits).