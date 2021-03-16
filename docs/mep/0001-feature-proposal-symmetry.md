---
author: Nathan Shammah, nathan@unitary.fund
status: <**Draft** | Active | Accepted | Deferred | Rejected | Withdrawn | Final | Superseded>
type: <**Standards Track** | Process>
created: created on 2021-03-15
resolution: <url> (required for Accepted | Rejected | Withdrawn)
---

# Mitiq Enhancement Proposal 0001 — Symmetry-based quantum error mitigation methods

## Abstract
---

Add a subpackpage with symmetry-based quantum error mitigation methods, which have appeared in the recent literature, enabling also new techniques to be devised with Mitiq.


## Motivation and Scope
---

Symmetry-based techniques seem very powerful for error mitigation. At a minimum, benchmarking them against other error mitigation techniques, such as ZNE or PEC, may be useful to users, such as researchers.

## Usage and Impact
---

Symmetry-based techniques will be an independent and additional set of quantum error mitigation techniques that Mitiq users can use.

## Backward compatibility
---

This is a new feature, so no backward compatibility assessment seems crucial.

## Detailed description
---

Create a subpackage, such as `mitiq/sbm` (symmetry-based methods, to use the three-letter standard implemented so far), which would contain other modules, which may include common files, such as a generic `symmetry.py` module, and more technique-specific moudules, such as
generic `verification.py` for symmetry verification, `distillation.py` for virtual distillation, and so forth.


All of these techniques seem to be based on the use of superoperators, as noise models are placed in superoperator space. All techniques seem also to rely on a stabilizer group or symmetry group, whose objects could be defined by the user, although pre-sets of pre-defined objects could be provided to simplify their use, e.g., $\\mathbb{{Z}}_2$ for $N$ qubits.

There are generally a sequence of common steps in the use of symmetry-based techniques, which here we exemplify for the specific case of subspace expansion:

- 1. Find a Hamiltonian problem on which to map your variational algorithm, so that it possesses a given symmetry.

- 2. Exploit the symmetry to deduce properties of the observables whose expectation variables are calculated at the end of the quantum circuit.

- 3. Define projectors based on the symmetry group.

- 4. Apply the projectors either at the end of the circuit (or during the circuit run) and thus obtain statistics for mitigated / unscathed / orthogonal projection results.

- 5. Return the mitigated variable, either by sequential (deterministic) mitigation process or through a stochastic process whcih implements point 4 above.


The implementation could possess some differences whether the study is performed on a simulator or on real(istic) quantum processor in the back-end. In the simulator case, the full density matrix is simulable, whereas in the realistic processor case, the projection can be an effective one, implemented through gates, without information on the unscathed counts.


### Examples

Give concrete examples for syntax and behavior to illustrate the implementation suggestions above.

#### Example 1:

Insert title and description of the example

```python
# Insert code example that illustrates what is described above.
# Comment your code to further elaborate and clarify the example.
```

## Related Work
---

Some relevant code may be present in Cirq (Python), as well as in other Mathematica-based plugins (Oxford group).

## Implementation
---

The major step in the symmetry implementation seems to be defining how to describe the symmetry operators and noise processes. One natural option could be to use [Cirq](https://github.com/quantumlib/Cirq/), which is Mitiq's main dependecy. Another option could be to use [QuTiP](http://qutip.org/), which seems to have more expressability in terms of superoperator space and superoperator properties and functions.

There are currently no pull requests or development branches relative to symmetry methods yet. Any pull requests or development branches containing work on this MEP should
be linked to from here.  (A MEP does not need to be implemented in a single
pull request if it makes sense to implement it in discrete phases).


## Alternatives
---

Alternatives could be found in making simple APIs in Mitiq, producing only example notebooks for the documentation, e.g., using QuTiP.

## Discussion
---

This section may just be a bullet list including links to any discussions
regarding the MEP:

- This includes links to discord threads or relevant GitHub issues.


## References and Footnotes
---

- _Each MEP must either be explicitly labeled as placed in the public domain (see
   this MEP as an example) or licensed under the `Open Publication License`_. [^1]

- _Open Publication License: https://www.opencontent.org/openpub/_

- Pull Request #598 [(gh-598)](https://github.com/unitaryfund/mitiq/pull/598)

### Related Literature References

- A section on symmetry-based quantum error mitigation techniques can be found in Mitiq's documentation, e.g., on [subspace expansion](https://mitiq.readthedocs.io/en/stable/guide/guide-error-mitigation.html#research-articles).

#### Copyright

[^1]: This document has been placed in the public domain.