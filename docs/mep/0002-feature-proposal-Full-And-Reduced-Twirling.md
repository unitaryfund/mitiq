---
Author: Purva Thakre
Status: Draft
Created: 2021-03-16
Resolution: <url>
---

# Mitiq Enhancement Proposal 0002 — Full Twirling and Reduced Twirling

## Abstract
---
#### Checklist before submission
- [ ] Short description of what is proposed

Pauli Twirling or Clifford Twirling is used to convert a known error channel into Pauli error channel. Full twirling uses entire set of error operators whereas reduced twirling is based in finding generators of error operator set.

## Motivation and Scope
---
#### Checklist before submission
- [ ] Describe the need for proposed change - Gottesman-Knill Theorem
- [ ] Proposed change describes the existing problem
- [ ] What does the change affect
- [ ] What is the change trying to solve
- [ ] Address scope and key requirements for
the proposed change

#### Full Twirling

#### Reduced Twirling

## Usage and Impact
---
#### Checklist before submission
- [ ] How Mitiq users will use proposed features
- [ ] Code examples that would not be possible without these proposed changes.
- [ ] What impact will proposed changes have on the Mitiq ecosystem
- [ ] Write users perspective
  - [ ] How do users benefit from this new addition ?
- [ ] Include ample details to improve functionality


#### Full Twirling

#### Reduced Twirling

## Backward compatibility
---
#### Checklist before submission
- [ ] Describe how this breaks backward compatibility usage and impact

#### Full Twirling

#### Reduced Twirling


## Detailed description
---
#### Checklist before submission
- [ ] Detailed description of the proposed change
- [ ] Examples of how new functionality can be used, intended use-cases and pseudo-code illustrating its use

:heavy_exclamation_mark: **Proposed functions from previous rough sketch** :heavy_exclamation_mark:

1. A group G defining n-qubit Pauli operators
2. Fidelities of the elements in group G
3. Commutator function defined over group G based on commutator and anticommutator relationships of the group elements.
  - A defined inverse of the commutator function
  - A commutator table defined as a numpy array ? -- ASCII table as print output
  - A generator table defined as a numpy array ? (Table II in the paper.) -- ASCII table as print output
4. _is_measurement, _pop_measurement and _append_measurement can be used to move measurements in the circuit towards the end.
5. Convert noise operator’s basis into Pauli basis using group G.
  - Since the noise operator has to be known, a function to check if the decomposition is  a valid error channel with unitary noise operators i.e. check if the twirling process could be done or not.
6. Full Pauli Twirling Function
  - Check if the noise operator’s Pauli basis (V) list has any repeated elements -then remove one of these elements.
  - Act on the state with V and conjugated V’s - check if process failed or not
7. Reduced Pauli Twirling Function
  - Check if the noise operator’s Pauli basis (V) list has any repeated elements -then remove one of these elements.
  - Reduced twirling set - depends on identifying a composition relation in V. Not sure how to identify a composition relation except for:
    - Commutator function of the generating set element and an element of V should be equal to 0.
    - Non-generating set elements could be discarded by iterating over elements of V and calculating the commutator function value.
    - Function to find size (N) of new generating set based on whole set V and generating set size.
    - New generator table (numpy array) based on generating set and function N. -- ASCII table as print output
      - Generating set elements are mapped to the new twirling set - check composition relation.
      - Elements not in the generating set or not calculated from composition relation are also mapped to the new twirling set.
    - Check and selectively discard repeated generator elements. This will give a new twirling set.
8. Define functions for exact twirling (iterate over the whole twirling set) and random twirling (choose random twirling set elements). Check if the error channel is a Pauli error channel or not.






#### Full Twirling

#### Reduced Twirling


### Examples

#### Checklist before submission
- [ ] Give concrete examples for syntax and behavior to illustrate the implementation suggestions above.

#### Example 1:

Insert title and description of the example

```python
# Insert code example that illustrates what is described above.
# Comment your code to further elaborate and clarify the example.
```

## Related Work
---

This section should list relevant and/or similar technologies, possibly in other
libraries. It does not need to be comprehensive, just list the major examples of
prior and relevant art.

## Implementation
---
#### Checklist before submission
- [ ] List the major steps required to implement the MEP
- [ ] Note where one step is dependent on another
- [ ] Note which steps could be omitted


Any pull requests or development branches containing work on this MEP should
be linked to from here.  (A MEP does not need to be implemented in a single
pull request if it makes sense to implement it in discrete phases).
#### Full Twirling

#### Reduced Twirling

## Alternatives
---

If there were any alternative solutions to solving the same problem, they should
be discussed here, along with a justification for the chosen approach.
#### Full Twirling

#### Reduced Twirling

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

[1] Gottesman, D. The heisenberg representation of quantum computers. arXiv preprint quant-ph/9807006 (1998) https://arxiv.org/abs/quant-ph/9807006

[2] Aaronson, S. & Gottesman, D. Improved simulation of stabilizer circuits. Phys. Rev. A 70, 052328 (2004) https://doi.org/10.1103/PhysRevA.70.052328

[3] Cai, Z., Benjamin, S. Constructing Smaller Pauli Twirling Sets for Arbitrary Error Channels. Sci Rep 9, 11281 (2019). https://doi.org/10.1038/s41598-019-46722-7

#### Copyright

[^1]: This document has been placed in the public domain.
