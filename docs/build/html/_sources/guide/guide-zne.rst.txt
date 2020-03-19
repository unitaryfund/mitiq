.. mitiq documentation file

*********************************************
Zero Noise Extrapolation
*********************************************

Introduction
============

Zero noise extrapolation (ZNE) was introduced concurrently in Ref. [1] and [2].
With `mitiq.zne` module it is possible to extrapolate what the expected value would be without noise. This is done by first setting up one of the key objects in `mitiq`, which is a :class:`mitiq.Factory` object.

Importing Quantum Circuits
==========================

`mitiq` allows one to flexibly import and export quantum circuits from other libraries. Here is an example:

.. code-block:: python

   >>> from mitiq import Factory
