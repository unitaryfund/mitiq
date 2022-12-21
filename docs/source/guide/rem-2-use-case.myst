---
jupytext:
  text_representation:
    extension: .myst
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# When should I use REM?

## Advantages

Readout error mitigation is a technique that deals with errors that many other
techniques do not handle. For that reason it can often be combined with
existing workflows to produce better results without much modification. The
technique can also accept as much information as the user has about the
measurement statistics in order to generate more accurate predictions.

## Disadvantages

Readout-error mitigation requires the preliminary characterization of the 
measurement errors associated to a specific backend. Measurement errors can
be expressed in the form of a confusion matrix and its estimation involves
numerous state preparations and measurements in the computational basis.

The confusion matrix must then be inverted and provided as an input to the
technique. The complete characterization of a full confusion matrix scales
exponentially in the number of qubits. For a better scaling when generating
these matrices, readout-error characterization can be simplified under the
assumption that measurement errors are local with respect to individual qubits
or group of qubits.
