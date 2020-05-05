.. _guide_zne_lindblad:

*********************************************
Zero-Noise Extrapolation
*********************************************

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Zero-Noise Extrapolation: Lindblad dynamics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In an ideal device, the time evolution is unitary, and as is modeled in
the intermediate representation of a quantum circuit,

.. math::

   \frac{d}{dt}|\psi\rangle=-\frac{i}{\hbar}H(t)|\psi\rangle,

where :math:`|\psi\rangle` is the initial state of the system (e.g., the qubits
involved in the operation) and :math:`U(t)` the unitary
time evolution set by a time-dependent Hamiltonian, H(t).


In the simplest scenario for the system-environment interaction, it is still
possible to describe the time evolution in terms of operators acting on the
system only, at the cost of losing the unitarity of the evolution.


The first required condition to develop such framework, is that the system
interacts more weakly with the environment than within its own
sub-constituents. This allows to proceed with a perturbative approach to solve
the problem, with a coupling constant :math:`\lambda` quantifying the
magnitude of the first-order expansion terms.

In this case, it is possible to write the time evolution of the density matrix
associated to the state, :math:`\hat{\rho}=|\psi\rangle\langle \psi|`, as

.. math::

   \frac{\partial }{ \partial t}\hat{\rho}=
   \frac{i}{\hbar}\lbrack H(t), \hat{\rho}\rbrack+\lambda \mathcal{L}
   \lbrack\hat{\rho}\rbrack,

where :math:`mathcal{L}` is a super-operator acting on the Hilbert space.

The subsequent most straightforward set of sensible approximations includes
assuming that at time zero the system and environment are not entangled, that
the environment is memoryless, and that there is a dominant scale of times set
by the interactions, wich allows to cut off high-frequency perturbations.

These approximations -- called the Born, Markov, and Rotating-Wave approximations, respectively --
lead to a so-called Lindblad form of the *dissipation*, i.e. to a special
structure of the system-environment interaction that can be represented with
a linear superoperator that always admits the Lindblad form

.. math::

   \mathcal{L}\lbrack\hat{\rho}\rbrack=\mathcal{L}\hat{\rho}
   =\sum_{i=1}^{N^2-1} \gamma_i \left( A_i\hat{\rho} A_i^\dagger
   - \frac{1}{2}( A_i^\dagger A_i\hat{\rho}+ \hat{\rho}A_i^\dagger A_i )\right)
   ,

where :math:`\gamma_i` are constants that set the strenghts of the dissipation
mechanisms defined by the jump operators, :math:`A_i`.
