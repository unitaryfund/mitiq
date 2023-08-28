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


# When should I use QSE?

## Advantages 

The advantage of QSE is that it allows determining the value of excited electronic states without the need for a formal error-correcting procedure. Furthermore, the method is error-agnostic. Computationally, the advantage is eliminating the need for ancilla measurements.  

## Disadvantages

In QSE, the tradeoff for eliminating the need for ancilla measurements is the exponential classical computation. Specifically, depending on the error model, we may need exponentially many operators in order to construct the projector to the error-free subspace. And, in turn, solving the generalized eigenvalue problem is an exponential classical computation. 
