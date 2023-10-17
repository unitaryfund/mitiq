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


# When should I use QSE?

## Advantages 

The main advantage of QSE is that it enables the approximation of excited states and reduction of errors without the need for a formal error-correcting procedure, in a way that is effective against many types of errors.
QSE eliminates the need for ancilla measurements, also reducing overhead as compared to full quantum error correction and other symmetry-based techniques.
QSE can also be used in conjunction with other quantum error mitigation techniques, such as zero noise extrapolation.

## Disadvantages

In QSE, the tradeoff for eliminating the need for ancilla measurements is the exponential classical computation. Specifically, depending on the error model, we may need exponentially many operators in order to construct the projector to the error-free subspace. And, in turn, solving the generalized eigenvalue problem is an exponential classical computation. 
