---
jupytext:
  text_representation:
    extension: .myst
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.12.0
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Hands-on lab on error mitigation with Mitiq.
+++

This is a hands-on notebook created for the [`SQMS/GGI 2022 Summer School on Quantum Simulation of Field Theories`](https://www.ggi.infn.it/showevent.pl?id=436). 

It is a guided tutorial on error mitigation with Mitiq and is focused on the zero-noise extrapolation (ZNE) technique. As this is
intended to be a hands-on exercise, the solutions to the examples are linked at the end of the notebook. 

Useful links :

- [`Mitiq repository`](https://github.com/unitaryfund/mitiq)
- [`Mitiq documentation`](https://mitiq.readthedocs.io/en/stable/)
- [`Mitiq docs on ZNE`](https://mitiq.readthedocs.io/en/stable/guide/zne.html)
- [`Mitiq white paper`](https://arxiv.org/abs/2009.04417)
- [`Unitary Fund`](https://unitary.fund)

```{figure} ../img/zne_workflow2_steps.png
---
width: 400
name: figzne
---
The diagram shows the workflow of the zero noise extrapolation (ZNE) technique in Mitiq.
```

The lab is split into the following sections :

- [](#checking-python-packages-are-installed-correctly)
- [](#computing-a-quantum-expectation-value-without-error-mitigation)
- [](#apply-zero-noise-extrapolation-with-mitiq)
- [](#explicitly-selecting-the-noise-scaling-method-and-the-extrapolation-method)
- [](#what-happens-behind-the-scenes-a-low-level-application-of-ZNE)

+++

## Checking Python packages are installed correctly

This notebook was tested with **Mitiq v0.17.0** and **qiskit v0.36.2**. It probably works with other versions too. Moreover, with minor changes, it can be adapted to quantum libraries that are different from Qiskit: Cirq, Braket, PyQuil, etc..

If you need to install Mitiq and/or Qiskit, you can uncomment and run the following cells.

```{code-cell} ipython3
# !pip install mitiq==0.17.0
```

```{code-cell} ipython3
# !pip install qiskit==0.36.2
```

If you encounter problems when installing Mitiq on your local machine,
you can try creating a new notebook in the online Binder einvironment at [`this link`](https://mybinder.org/v2/gh/unitaryfund/mitiq/0da4965f3d80b9ee7ed9e93527c7e7c09d4b2f7e
).

You can check your locally installed version of Mitiq and of the associated frontend libraries by running the next cell.

```{code-cell} ipython3
from mitiq import about

about()
```
+++

## Computing a quantum expectation value without error mitigation
+++
### Define the circuit of interest
+++
### Run the circuit with a noiseless backend and with a noisy backend
+++
## Apply zero-noise extrapolation with Mitiq
+++
## Explicitly selecting the noise-scaling method and the extrapolation method
+++
## What happens behind the scenes? A low-level application of ZNE
+++
### STEP 1: Noise-scaled expectation values are evaluated via gate-level "unitary folding" transformations
+++
### STEP 2: Inference of the ideal result via zero-noise extrapolation
+++
## References
+++
