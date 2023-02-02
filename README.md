# <a href="https://github.com/unitaryfund/mitiq"><img src="https://github.com/unitaryfund/mitiq/blob/master/docs/source/img/mitiq-logo.png?raw=true" alt="Mitiq logo" width="350"/></a>

<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-34-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->
[![build](https://github.com/unitaryfund/mitiq/actions/workflows/build.yml/badge.svg?branch=master)](https://github.com/unitaryfund/mitiq/actions)
[![Documentation Status](https://readthedocs.org/projects/mitiq/badge/?version=stable)](https://mitiq.readthedocs.io/en/stable/)
[![codecov](https://codecov.io/gh/unitaryfund/mitiq/branch/master/graph/badge.svg)](https://codecov.io/gh/unitaryfund/mitiq)
[![PyPI version](https://badge.fury.io/py/mitiq.svg)](https://badge.fury.io/py/mitiq)
[![arXiv](https://img.shields.io/badge/arXiv-2009.04417-<COLOR>.svg)](https://arxiv.org/abs/2009.04417)
[![Downloads](https://static.pepy.tech/personalized-badge/mitiq?period=total&units=international_system&left_color=black&right_color=green&left_text=Downloads)](https://pepy.tech/project/mitiq)
[![Repository](https://img.shields.io/badge/GitHub-5C5C5C.svg?logo=github)](https://github.com/unitaryfund/mitiq)
[![Unitary Fund](https://img.shields.io/badge/Supported%20By-Unitary%20Fund-FFFF00.svg)](https://unitary.fund)
[![Discord Chat](https://img.shields.io/badge/dynamic/json?color=blue&label=Discord&query=approximate_presence_count&suffix=%20online.&url=https%3A%2F%2Fdiscord.com%2Fapi%2Finvites%2FJqVGmpkP96%3Fwith_counts%3Dtrue)](http://discord.unitary.fund)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/unitaryfund/mitiq/master?urlpath=%2Ftree%2Fdocs%2Fsource%2Fexamples)

Mitiq is a Python toolkit for implementing error mitigation techniques on
quantum computers.

Current quantum computers are noisy due to interactions with the environment,
imperfect gate applications, state preparation and measurement errors, etc.
Error mitigation seeks to reduce these effects at the software level by
compiling quantum programs in clever ways.

Want to know more? Check out our
[documentation](https://mitiq.readthedocs.io/en/stable/guide/guide.html) and chat with us on [Discord](http://discord.unitary.fund).

> Do you use near-term quantum hardware? Have you tried Mitiq? Either way, take our survey and help make Mitiq better! [bit.ly/mitiq-survey](https://bit.ly/mitiq-survey)

## Quickstart

### Installation

```bash
pip install mitiq
```

### Example

Define a function which inputs a circuit and returns an expectation value you want to compute, then use Mitiq to mitigate errors.

```python
import cirq
from mitiq import zne, benchmarks


def execute(circuit: cirq.Circuit, noise_level: float = 0.001) -> float:
    """Returns Tr[ρ |0⟩⟨0|] where ρ is the state prepared by the circuit with depolarizing noise."""
    noisy_circuit = circuit.with_noise(cirq.depolarize(p=noise_level))
    return cirq.DensityMatrixSimulator().simulate(noisy_circuit).final_density_matrix[0, 0].real


circuit: cirq.Circuit = benchmarks.generate_rb_circuits(n_qubits=1, num_cliffords=50)[0]

true_value = execute(circuit, noise_level=0.0)       # Ideal quantum computer.
noisy_value = execute(circuit)                       # Noisy quantum computer.
zne_value = zne.execute_with_zne(circuit, execute)   # Noisy quantum computer + Mitiq.

print(f"Error (w/o  Mitiq): %0.4f" %abs((true_value - noisy_value) / true_value))
print(f"Error (with Mitiq): %0.4f" %abs((true_value - zne_value) / true_value))
```
Sample output:
```
Error (w/o  Mitiq): 0.0688
Error (with Mitiq): 0.0002
```

See our [guides](https://mitiq.readthedocs.io/en/stable/guide/guide.html) and [examples](https://mitiq.readthedocs.io) for more explanation, techniques, and benchmarks.
The examples and other notebooks can be run interactively on the cloud with [mybinder.org](https://mybinder.org/v2/gh/unitaryfund/mitiq/master?filepath=%2Fdocs%2Fsource%2Fexamples).

## Quick Tour

### Error mitigation techniques

| Technique                                 | Documentation                                                | Mitiq module                                                            | Paper Reference(s)                                                                                                                                 |
| ----------------------------------------- | ------------------------------------------------------------ | ----------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| Zero-noise extrapolation                  | [ZNE](https://mitiq.readthedocs.io/en/latest/guide/zne.html) | [mitiq.zne](https://github.com/unitaryfund/mitiq/tree/master/mitiq/zne) | [1611.09301](https://arxiv.org/abs/1611.09301)<br>[1612.02058](https://arxiv.org/abs/1612.02058)<br>[1805.04492](https://arxiv.org/abs/1805.04492) |
| Probabilistic error cancellation          | [PEC](https://mitiq.readthedocs.io/en/latest/guide/pec.html) | [mitiq.pec](https://github.com/unitaryfund/mitiq/tree/master/mitiq/pec) | [1612.02058](https://arxiv.org/abs/1612.02058)<br>[1712.09271](https://arxiv.org/abs/1712.09271)<br>[1905.10135](https://arxiv.org/abs/1905.10135) |
| (Variable-noise) Clifford data regression | [CDR](https://mitiq.readthedocs.io/en/latest/guide/cdr.html) | [mitiq.cdr](https://github.com/unitaryfund/mitiq/tree/master/mitiq/cdr) | [2005.10189](https://arxiv.org/abs/2005.10189)<br>[2011.01157](https://arxiv.org/abs/2011.01157)                                                   |
| Digital dynamical decoupling                      | [DDD](https://mitiq.readthedocs.io/en/latest/guide/ddd.html) | [mitiq.cdd](https://github.com/unitaryfund/mitiq/tree/master/mitiq/ddd) | [9803057](https://arxiv.org/abs/quant-ph/9803057)<br>[1807.08768](https://arxiv.org/abs/1807.08768)                                                |
| Readout-error mitigation                  | [REM](https://mitiq.readthedocs.io/en/latest/guide/rem.html) | [mitiq.rem](https://github.com/unitaryfund/mitiq/tree/master/mitiq/rem) | [1907.08518](https://arxiv.org/abs/1907.08518) <br>[2006.14044](https://arxiv.org/abs/2006.14044)                        |

See our [roadmap](https://github.com/unitaryfund/mitiq/wiki) for additional candidate techniques to implement. If there is a technique you are looking for, please file a [feature request](https://github.com/unitaryfund/mitiq/issues/new?assignees=&labels=feature-request&template=feature_request.md&title=).

### Interface

We refer to any programming language you can write quantum circuits in as a *frontend*, and any quantum computer / simulator you can simulate circuits in as a *backend*.

#### Supported frontends

| [Cirq](https://quantumai.google/cirq)                                                                                                                                         | [Qiskit](https://qiskit.org/)                                                                                         | [pyQuil](https://github.com/rigetti/pyquil)                                                                                                             | [Braket](https://github.com/aws/amazon-braket-sdk-python)                                                                                                                         | [PennyLane](https://pennylane.ai/)                                                                                                   |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| <a href="https://quantumai.google/cirq"><img src="https://raw.githubusercontent.com/quantumlib/Cirq/master/docs/images/Cirq_logo_color.png" alt="Cirq logo" width="130"/></a> | <a href="https://qiskit.org/"><img src="https://qiskit.org/images/qiskit-logo.png" alt="Qiskit logo" width="80"/></a> | <a href="https://github.com/rigetti/pyquil"><img src="https://www.rigetti.com/uploads/Logos/logo-rigetti-gray.jpg" alt="Rigetti logo" width="150"/></a> | <a href="https://github.com/aws/amazon-braket-sdk-python"><img src="https://a0.awsstatic.com/libra-css/images/logos/aws_logo_smile_1200x630.png" alt="AWS logo" width="150"/></a> | &nbsp;&nbsp;  <a href="https://pennylane.ai/"><img src="https://pennylane.ai/img/xanadu_x.png" alt="PennyLane logo" width="60"/></a> |

Note: Cirq is a core requirement of Mitiq and is installed when you `pip install mitiq`.

#### Supported backends

You can use Mitiq with any backend you have access to that can interface with supported frontends.

### Benchmarks

Mitiq uses [`asv`](https://asv.readthedocs.io/en/stable/) to benchmark the core functionalities of the project.
They are found in the [`benchmarks/`](https://github.com/unitaryfund/mitiq/tree/master/benchmarks) directory and their changes can be seen overtime at https://benchmarks.mitiq.dev/.

### Citing Mitiq

If you use Mitiq in your research, please reference the [Mitiq whitepaper](https://quantum-journal.org/papers/q-2022-08-11-774/) using the bibtex entry found in [`CITATION.bib`](https://github.com/unitaryfund/mitiq/blob/master/CITATION.bib).

A list of papers citing Mitiq can be found on [Google Scholar](https://scholar.google.com/scholar?cites=12810395086731011605) / [Semantic Scholar](https://www.semanticscholar.org/paper/Mitiq%3A-A-software-package-for-error-mitigation-on-LaRose-Mari/dc55b366d5b2212c6df8cd5c0bf05bab13104bd7#citing-papers).

## License

[GNU GPL v.3.0.](https://github.com/unitaryfund/mitiq/blob/master/LICENSE)

## Contributing

We welcome contributions to Mitiq including bug fixes, feature requests, etc. To get started, check out our [contribution
guidelines](https://mitiq.readthedocs.io/en/stable/toc_contributing.html) and/or [documentation guidelines](https://mitiq.readthedocs.io/en/stable/contributing_docs.html).
An up-to-date list of contributors can be found [here](https://github.com/unitaryfund/mitiq/graphs/contributors) and below.

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center"><a href="https://github.com/Yash-10"><img src="https://avatars.githubusercontent.com/u/68844397?v=4?s=100" width="100px;" alt="Yash-10"/><br /><sub><b>Yash-10</b></sub></a><br /><a href="https://github.com/unitaryfund/mitiq/commits?author=Yash-10" title="Tests">⚠️</a> <a href="https://github.com/unitaryfund/mitiq/commits?author=Yash-10" title="Code">💻</a></td>
      <td align="center"><a href="https://github.com/LaurentAjdnik"><img src="https://avatars.githubusercontent.com/u/83899250?v=4?s=100" width="100px;" alt="Laurent AJDNIK"/><br /><sub><b>Laurent AJDNIK</b></sub></a><br /><a href="https://github.com/unitaryfund/mitiq/commits?author=LaurentAjdnik" title="Documentation">📖</a></td>
      <td align="center"><a href="https://github.com/HaoTy"><img src="https://avatars.githubusercontent.com/u/36152061?v=4?s=100" width="100px;" alt="Tianyi Hao"/><br /><sub><b>Tianyi Hao</b></sub></a><br /><a href="https://github.com/unitaryfund/mitiq/commits?author=HaoTy" title="Code">💻</a> <a href="https://github.com/unitaryfund/mitiq/issues?q=author%3AHaoTy" title="Bug reports">🐛</a></td>
      <td align="center"><a href="https://github.com/briancylui"><img src="https://avatars.githubusercontent.com/u/18178086?v=4?s=100" width="100px;" alt="Brian Lui"/><br /><sub><b>Brian Lui</b></sub></a><br /><a href="https://github.com/unitaryfund/mitiq/commits?author=briancylui" title="Code">💻</a> <a href="https://github.com/unitaryfund/mitiq/issues?q=author%3Abriancylui" title="Bug reports">🐛</a></td>
      <td align="center"><a href="https://github.com/ckissane"><img src="https://avatars.githubusercontent.com/u/9607290?v=4?s=100" width="100px;" alt="Cole Kissane"/><br /><sub><b>Cole Kissane</b></sub></a><br /><a href="https://github.com/unitaryfund/mitiq/commits?author=ckissane" title="Code">💻</a> <a href="https://github.com/unitaryfund/mitiq/issues?q=author%3Ackissane" title="Bug reports">🐛</a></td>
      <td align="center"><a href="http://www.mustythoughts.com"><img src="https://avatars.githubusercontent.com/u/7314136?v=4?s=100" width="100px;" alt="Michał Stęchły"/><br /><sub><b>Michał Stęchły</b></sub></a><br /><a href="https://github.com/unitaryfund/mitiq/commits?author=mstechly" title="Code">💻</a></td>
      <td align="center"><a href="http://kunalmarwaha.com"><img src="https://avatars.githubusercontent.com/u/2541209?v=4?s=100" width="100px;" alt="Kunal Marwaha"/><br /><sub><b>Kunal Marwaha</b></sub></a><br /><a href="https://github.com/unitaryfund/mitiq/commits?author=marwahaha" title="Documentation">📖</a></td>
    </tr>
    <tr>
      <td align="center"><a href="https://github.com/k-m-schultz"><img src="https://avatars.githubusercontent.com/u/15523976?v=4?s=100" width="100px;" alt="k-m-schultz"/><br /><sub><b>k-m-schultz</b></sub></a><br /><a href="#example-k-m-schultz" title="Examples">💡</a></td>
      <td align="center"><a href="http://www.linkedin.com/in/bobin-mathew"><img src="https://avatars.githubusercontent.com/u/32351527?v=4?s=100" width="100px;" alt="Bobin Mathew"/><br /><sub><b>Bobin Mathew</b></sub></a><br /><a href="https://github.com/unitaryfund/mitiq/commits?author=BobinMathew" title="Documentation">📖</a></td>
      <td align="center"><a href="https://github.com/LogMoss"><img src="https://avatars.githubusercontent.com/u/61593765?v=4?s=100" width="100px;" alt="LogMoss"/><br /><sub><b>LogMoss</b></sub></a><br /><a href="https://github.com/unitaryfund/mitiq/commits?author=LogMoss" title="Documentation">📖</a> <a href="https://github.com/unitaryfund/mitiq/issues?q=author%3ALogMoss" title="Bug reports">🐛</a></td>
      <td align="center"><a href="https://github.com/DSamuel1"><img src="https://avatars.githubusercontent.com/u/40476737?v=4?s=100" width="100px;" alt="DSamuel1"/><br /><sub><b>DSamuel1</b></sub></a><br /><a href="#example-DSamuel1" title="Examples">💡</a> <a href="https://github.com/unitaryfund/mitiq/commits?author=DSamuel1" title="Code">💻</a></td>
      <td align="center"><a href="https://github.com/sid1993"><img src="https://avatars.githubusercontent.com/u/4842078?v=4?s=100" width="100px;" alt="sid1993"/><br /><sub><b>sid1993</b></sub></a><br /><a href="https://github.com/unitaryfund/mitiq/commits?author=sid1993" title="Code">💻</a> <a href="https://github.com/unitaryfund/mitiq/issues?q=author%3Asid1993" title="Bug reports">🐛</a></td>
      <td align="center"><a href="https://github.com/yhindy"><img src="https://avatars.githubusercontent.com/u/11757328?v=4?s=100" width="100px;" alt="Yousef Hindy"/><br /><sub><b>Yousef Hindy</b></sub></a><br /><a href="https://github.com/unitaryfund/mitiq/commits?author=yhindy" title="Code">💻</a> <a href="https://github.com/unitaryfund/mitiq/commits?author=yhindy" title="Tests">⚠️</a> <a href="https://github.com/unitaryfund/mitiq/commits?author=yhindy" title="Documentation">📖</a></td>
      <td align="center"><a href="https://github.com/elmandouh"><img src="https://avatars.githubusercontent.com/u/73552047?v=4?s=100" width="100px;" alt="Mohamed El Mandouh"/><br /><sub><b>Mohamed El Mandouh</b></sub></a><br /><a href="https://github.com/unitaryfund/mitiq/commits?author=elmandouh" title="Code">💻</a> <a href="https://github.com/unitaryfund/mitiq/commits?author=elmandouh" title="Tests">⚠️</a> <a href="https://github.com/unitaryfund/mitiq/commits?author=elmandouh" title="Documentation">📖</a></td>
    </tr>
    <tr>
      <td align="center"><a href="https://github.com/Aaron-Robertson"><img src="https://avatars.githubusercontent.com/u/58564008?v=4?s=100" width="100px;" alt="Aaron Robertson"/><br /><sub><b>Aaron Robertson</b></sub></a><br /><a href="#example-Aaron-Robertson" title="Examples">💡</a> <a href="https://github.com/unitaryfund/mitiq/commits?author=Aaron-Robertson" title="Tests">⚠️</a> <a href="https://github.com/unitaryfund/mitiq/issues?q=author%3AAaron-Robertson" title="Bug reports">🐛</a> <a href="#ideas-Aaron-Robertson" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/unitaryfund/mitiq/commits?author=Aaron-Robertson" title="Code">💻</a> <a href="https://github.com/unitaryfund/mitiq/commits?author=Aaron-Robertson" title="Documentation">📖</a></td>
      <td align="center"><a href="https://ashishpanigrahi.me"><img src="https://avatars.githubusercontent.com/u/59497618?v=4?s=100" width="100px;" alt="Ashish Panigrahi"/><br /><sub><b>Ashish Panigrahi</b></sub></a><br /><a href="https://github.com/unitaryfund/mitiq/commits?author=paniash" title="Documentation">📖</a></td>
      <td align="center"><a href="https://github.com/maxtremblay"><img src="https://avatars.githubusercontent.com/u/52462375?v=4?s=100" width="100px;" alt="Maxime Tremblay"/><br /><sub><b>Maxime Tremblay</b></sub></a><br /><a href="https://github.com/unitaryfund/mitiq/commits?author=maxtremblay" title="Code">💻</a> <a href="https://github.com/unitaryfund/mitiq/commits?author=maxtremblay" title="Documentation">📖</a> <a href="#ideas-maxtremblay" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center"><a href="https://github.com/andre-a-alves"><img src="https://avatars.githubusercontent.com/u/20098360?v=4?s=100" width="100px;" alt="Andre"/><br /><sub><b>Andre</b></sub></a><br /><a href="https://github.com/unitaryfund/mitiq/commits?author=andre-a-alves" title="Documentation">📖</a> <a href="https://github.com/unitaryfund/mitiq/commits?author=andre-a-alves" title="Tests">⚠️</a></td>
      <td align="center"><a href="https://github.com/purva-thakre"><img src="https://avatars.githubusercontent.com/u/66048318?v=4?s=100" width="100px;" alt="Purva Thakre"/><br /><sub><b>Purva Thakre</b></sub></a><br /><a href="https://github.com/unitaryfund/mitiq/commits?author=purva-thakre" title="Documentation">📖</a> <a href="#infra-purva-thakre" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="https://github.com/unitaryfund/mitiq/commits?author=purva-thakre" title="Code">💻</a> <a href="#ideas-purva-thakre" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center"><a href="http://karalekas.com"><img src="https://avatars.githubusercontent.com/u/3578739?v=4?s=100" width="100px;" alt="Peter Karalekas"/><br /><sub><b>Peter Karalekas</b></sub></a><br /><a href="#maintenance-karalekas" title="Maintenance">🚧</a> <a href="https://github.com/unitaryfund/mitiq/commits?author=karalekas" title="Code">💻</a> <a href="https://github.com/unitaryfund/mitiq/commits?author=karalekas" title="Documentation">📖</a> <a href="#infra-karalekas" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#ideas-karalekas" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center"><a href="https://www.sckaiser.com"><img src="https://avatars.githubusercontent.com/u/6486256?v=4?s=100" width="100px;" alt="Sarah Kaiser"/><br /><sub><b>Sarah Kaiser</b></sub></a><br /><a href="#maintenance-crazy4pi314" title="Maintenance">🚧</a> <a href="https://github.com/unitaryfund/mitiq/commits?author=crazy4pi314" title="Code">💻</a> <a href="https://github.com/unitaryfund/mitiq/commits?author=crazy4pi314" title="Documentation">📖</a> <a href="#infra-crazy4pi314" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#ideas-crazy4pi314" title="Ideas, Planning, & Feedback">🤔</a></td>
    </tr>
    <tr>
      <td align="center"><a href="https://sites.google.com/site/andreamari84/home"><img src="https://avatars.githubusercontent.com/u/46054446?v=4?s=100" width="100px;" alt="Andrea Mari"/><br /><sub><b>Andrea Mari</b></sub></a><br /><a href="#maintenance-andreamari" title="Maintenance">🚧</a> <a href="https://github.com/unitaryfund/mitiq/commits?author=andreamari" title="Code">💻</a> <a href="https://github.com/unitaryfund/mitiq/commits?author=andreamari" title="Documentation">📖</a> <a href="#infra-andreamari" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#ideas-andreamari" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center"><a href="http://willzeng.com"><img src="https://avatars.githubusercontent.com/u/5214594?v=4?s=100" width="100px;" alt="Will Zeng"/><br /><sub><b>Will Zeng</b></sub></a><br /><a href="#maintenance-willzeng" title="Maintenance">🚧</a> <a href="https://github.com/unitaryfund/mitiq/commits?author=willzeng" title="Code">💻</a> <a href="https://github.com/unitaryfund/mitiq/commits?author=willzeng" title="Documentation">📖</a> <a href="#infra-willzeng" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#ideas-willzeng" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center"><a href="https://www.nathanshammah.com"><img src="https://avatars.githubusercontent.com/u/14573436?v=4?s=100" width="100px;" alt="Nathan Shammah"/><br /><sub><b>Nathan Shammah</b></sub></a><br /><a href="#maintenance-nathanshammah" title="Maintenance">🚧</a> <a href="https://github.com/unitaryfund/mitiq/commits?author=nathanshammah" title="Code">💻</a> <a href="https://github.com/unitaryfund/mitiq/commits?author=nathanshammah" title="Documentation">📖</a> <a href="#infra-nathanshammah" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#ideas-nathanshammah" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center"><a href="http://ryanlarose.com"><img src="https://avatars.githubusercontent.com/u/32416820?v=4?s=100" width="100px;" alt="Ryan LaRose"/><br /><sub><b>Ryan LaRose</b></sub></a><br /><a href="#maintenance-rmlarose" title="Maintenance">🚧</a> <a href="https://github.com/unitaryfund/mitiq/commits?author=rmlarose" title="Code">💻</a> <a href="https://github.com/unitaryfund/mitiq/commits?author=rmlarose" title="Documentation">📖</a> <a href="#infra-rmlarose" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#ideas-rmlarose" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center"><a href="https://github.com/francespoblete"><img src="https://avatars.githubusercontent.com/u/65167390?v=4?s=100" width="100px;" alt="francespoblete"/><br /><sub><b>francespoblete</b></sub></a><br /><a href="#design-francespoblete" title="Design">🎨</a></td>
      <td align="center"><a href="https://github.com/Misty-W"><img src="https://avatars.githubusercontent.com/u/82074193?v=4?s=100" width="100px;" alt="Misty-W"/><br /><sub><b>Misty-W</b></sub></a><br /><a href="https://github.com/unitaryfund/mitiq/commits?author=Misty-W" title="Code">💻</a> <a href="#example-Misty-W" title="Examples">💡</a> <a href="#ideas-Misty-W" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/unitaryfund/mitiq/commits?author=Misty-W" title="Tests">⚠️</a> <a href="#maintenance-Misty-W" title="Maintenance">🚧</a> <a href="https://github.com/unitaryfund/mitiq/commits?author=Misty-W" title="Documentation">📖</a></td>
      <td align="center"><a href="https://github.com/AkashNarayanan"><img src="https://avatars.githubusercontent.com/u/83135130?v=4?s=100" width="100px;" alt="AkashNarayanan B"/><br /><sub><b>AkashNarayanan B</b></sub></a><br /><a href="#infra-AkashNarayanan" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
    </tr>
    <tr>
      <td align="center"><a href="https://github.com/L-P-B"><img src="https://avatars.githubusercontent.com/u/32333736?v=4?s=100" width="100px;" alt="L-P-B"/><br /><sub><b>L-P-B</b></sub></a><br /><a href="https://github.com/unitaryfund/mitiq/commits?author=L-P-B" title="Tests">⚠️</a> <a href="https://github.com/unitaryfund/mitiq/commits?author=L-P-B" title="Code">💻</a></td>
      <td align="center"><a href="https://github.com/Rahul-Mistri"><img src="https://avatars.githubusercontent.com/u/52910775?v=4?s=100" width="100px;" alt="Rahul Mistri"/><br /><sub><b>Rahul Mistri</b></sub></a><br /><a href="https://github.com/unitaryfund/mitiq/commits?author=Rahul-Mistri" title="Tests">⚠️</a> <a href="https://github.com/unitaryfund/mitiq/commits?author=Rahul-Mistri" title="Code">💻</a></td>
      <td align="center"><a href="http://vtomole.com"><img src="https://avatars.githubusercontent.com/u/8405160?v=4?s=100" width="100px;" alt="Victory Omole"/><br /><sub><b>Victory Omole</b></sub></a><br /><a href="https://github.com/unitaryfund/mitiq/commits?author=vtomole" title="Tests">⚠️</a> <a href="https://github.com/unitaryfund/mitiq/commits?author=vtomole" title="Code">💻</a></td>
      <td align="center"><a href="http://natestemen.xyz"><img src="https://avatars.githubusercontent.com/u/12703123?v=4?s=100" width="100px;" alt="nate stemen"/><br /><sub><b>nate stemen</b></sub></a><br /><a href="#infra-natestemen" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="https://github.com/unitaryfund/mitiq/commits?author=natestemen" title="Documentation">📖</a> <a href="https://github.com/unitaryfund/mitiq/commits?author=natestemen" title="Tests">⚠️</a> <a href="https://github.com/unitaryfund/mitiq/commits?author=natestemen" title="Code">💻</a></td>
      <td align="center"><a href="https://github.com/1ucian0"><img src="https://avatars.githubusercontent.com/u/766693?v=4?s=100" width="100px;" alt="Luciano Bello"/><br /><sub><b>Luciano Bello</b></sub></a><br /><a href="#infra-1ucian0" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="https://github.com/unitaryfund/mitiq/commits?author=1ucian0" title="Code">💻</a></td>
      <td align="center"><a href="http://amirebrahimi.com/"><img src="https://avatars.githubusercontent.com/u/904110?v=4?s=100" width="100px;" alt="Amir Ebrahimi"/><br /><sub><b>Amir Ebrahimi</b></sub></a><br /><a href="https://github.com/unitaryfund/mitiq/commits?author=amirebrahimi" title="Code">💻</a> <a href="https://github.com/unitaryfund/mitiq/commits?author=amirebrahimi" title="Tests">⚠️</a> <a href="https://github.com/unitaryfund/mitiq/commits?author=amirebrahimi" title="Documentation">📖</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind are welcome!
