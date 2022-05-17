---
author: Amir Ebrahimi <github@aebrahimi.com>
status: Draft
type: <Standards Track | Process>
created: 2022-05-16
resolution: <url> (required for Accepted | Rejected | Withdrawn)
---

# Mitiq Enhancement Proposal 0001 — Readout Confusion Inversion (RCI)

## Abstract
---

Implement a readout error mitigation technique by using inverted confusion matrices. The technique works by using the inverted confusion matrix to improve measurement results on circuits that run for the experiment.

## Motivation and Scope
---

Mitiq needs more error mitigation techniques. Currently, it has post-selection, but no readout error mitigation.

The scope of this improvement is to add a new `execute_with_rci` method, a `mitigate_executor`, and an `rci_decorator`.

_Out of scope_:
   - Providing utilities to generate confusion matrices or their inverses.

## Usage and Impact
---

A user would be able to use this feature by:

```python
from mitiq.rem.rci import execute_with_rci
execute_with_rci(circuit, executor)

# Or

from mitiq.rem.rci import mitigate_executor
rci_executor = mitigate_executor(executor, observable inverse_confusion_matrix=inverse_confusion_matrix)

# Or

@rci_decorator(observable=observable, inverse_confusion_matrix=inverse_confusion_matrix)
def noisy_readout_decorated_executor(qp: QPROGRAM) -> MeasurementResult:
    return noisy_readout_executor(qp)
```

## Backward compatibility
---

RCI adds new functionality, but it does require an `Executor` that returns MeasurementResults and not simply expectation values. It is necessary to get the raw measurement results in order to mitigate them. However, `execute_with_rci` returns an expectation value, so it can potentially have other error mitigation techniques stacked on top.

## Detailed description
---

Briefly, readout error mitigation has the following assumptions/steps:
1. Assume readout errors are uncorrelated.
1. Measure |0> and |1> on each of the n selected qubits (total of 2n experiments).
1. Form the confusion matrix on each qubit:
   - ![](https://latex.codecogs.com/png.image?%5Cinline%20%5Cdpi%7B110%7D%5Cbg%7Bwhite%7DP%20=%20%5Cbegin%7Bpmatrix%7D1%20-%20e_0%20&%20e_1%20%5C%5Ce_0%20&%201%20-%20e_1%20%5C%5C%5Cend%7Bpmatrix%7D)
   - e0 is the probability to get |1> when |0> is prepared
   - e1 is the probability to get |0> when |1> is prepared
1. Invert the confusion matrix (d = 1 - e0 - e1):
   - ![](https://latex.codecogs.com/png.image?%5Cinline%20%5Cdpi%7B110%7D%5Cbg%7Bwhite%7DP%5E%7B-1%7D%20=%20%5Cfrac%7B1%7D%7Bd%7D%5Cbegin%7Bpmatrix%7D1%20-%20e_1%20&%20-e_1%20%5C%5C-e_0%20&%201%20-%20e_0%20%5C%5C%5Cend%7Bpmatrix%7D)
1. Apply P inverse to the measured results to get the corrected results.

### Examples

The following are some concrete examples.

#### Example 1:

This example uses a Z0 + Z1 observable with a circuit that flips each qubit to the |1> state. However, this is an incredibly noisy device where it actually flips all results. So, we mitigate these error results with RCI.

```python
# Default qubit register and circuit for unit tests
qreg = [cirq.LineQubit(i) for i in range(2)]
circ = cirq.Circuit(cirq.ops.X.on_each(*qreg), cirq.measure_each(*qreg))
observable = Observable(PauliString("ZI"), PauliString("IZ"))

def test_rci_with_matrix():
    # test with an executor that completely flips results
    p0 = 1
    p1 = 1
    noisy_executor = partial(noisy_readout_executor, p0=p0, p1=p1)
    unmitigated = raw_execute(circ, noisy_executor, observable)
    assert np.isclose(unmitigated, 2.0)

    inverse_confusion_matrix = [
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
    ]

    mitigated = execute_with_rci(
        circ,
        noisy_executor,
        observable,
        inverse_confusion_matrix=inverse_confusion_matrix,
    )
    assert np.isclose(mitigated, -2.0)
```

#### Example 2:

This is the same example as the above, but without passing an inverse_confusion_matrix.

```python
# Default qubit register and circuit for unit tests
qreg = [cirq.LineQubit(i) for i in range(2)]
circ = cirq.Circuit(cirq.ops.X.on_each(*qreg), cirq.measure_each(*qreg))
observable = Observable(PauliString("ZI"), PauliString("IZ"))

def test_rci_without_matrix():
    # test with an executor that completely flips results
    p0 = 1
    p1 = 1
    noisy_executor = partial(noisy_readout_executor, p0=p0, p1=p1)
    unmitigated = raw_execute(circ, noisy_executor, observable)
    assert np.isclose(unmitigated, 2.0)

    mitigated = execute_with_rci(
        circ, noisy_executor, observable, p0=p0, p1=p1
    )
    assert np.isclose(mitigated, -2.0)
```

## Related Work
---

There is an implementation of this technique in Qiskit Ignis. In literature, Appendix H of https://arxiv.org/abs/2009.13485 provides a succinct description.

Cirq added utilites for measurement error mitigation in quantumlib/Cirq#4800

## Implementation
---

The recent utilities added to Cirq (quantumlib/Cirq#4800) have made it much easier to implement this error mitigation technique. Confusion matrices and their correction matrices (i.e. inverted confusion matrices) can be generated with the `measure_confusion_matrix` method. From there it is only necessary to transform the raw measurement results with the inverted matrices and then to retrieve the observable expectation value from the measurements.

Currently, this has been implemented on the following branch:
- [amirebrahimi/mitiq@rci](https://github.com/amirebrahimi/mitiq/tree/rci/)
- Main files:
   - [rci.py](https://github.com/amirebrahimi/mitiq/blob/rci/mitiq/rem/rci.py)
   - [test_rci.py](https://github.com/amirebrahimi/mitiq/blob/rci/mitiq/rem/tests/test_rci.py)


## Alternatives
---

No alternative solutions are currently being explored.

## Discussion
---

Relevant discussion / work / issues:

- unitaryfund/mitiq#815
- quantumlib/Cirq#4800
- [Previous RFC](https://docs.google.com/document/d/1Mb-OoojXBm0k8VTapNUkQQi4YQR5vsUvNJqmVgFpIns/edit?usp=sharing)
- [QCourse 570-1 project](https://gitlab.com/qworld/qeducation/qcourse570-1/-/issues/7)
- [Initial discord discussion](https://discord.com/channels/764231928676089909/773957956659052605/928775883886579762)



## References and Footnotes
---

- This document has been placed in the public domain.
- [arXiv:2009.13485](https://arxiv.org/abs/2009.13485)
- [Measurement Error Mitigation Qiskit tutorial](https://learn.qiskit.org/course/quantum-hardware/measurement-error-mitigation)