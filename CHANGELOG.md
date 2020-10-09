# Changelog

[//]: # " ## Development 0.X.Ydev (Month DDth, YYYY)"
[//]: # " ## (Future) Version 0.1.1 (Date)"
[//]: # " ### Changes"
[//]: # " - **MAJOR FEATURE**: New integration."
[//]: # " - Improve something."
[//]: # " - [Bug Fix]"
[//]: # " - Fix the bug."

## Version 0.3.0 (In Development)

### All Changes

- Fix broken links on the website (@erkska, gh-400).
- Use cirq v0.9.0 instead of cirq-unstable (@karalekas, gh-402).
- Update mitiq.about() (@rmlarose, gh-399).

## Version 0.2.0 (October 4th, 2020)

### Announcements

The preprint for Mitiq is live on the arXiv [here][arxiv]!

### Summary

This release centered on source code reorganization and documentation, as well
as wrapping up some holdovers from the initial public release. In addition, the
team began investigating probabilistic error cancellation (PEC), which will be the
main focus of the following release.

### All Changes

- Re-organize scaling code into its own module (@rmlarose, gh-338).
- Add BibTeX snippet for [arXiv citation][arxiv] (@karalekas, gh-351).
- Fix broken links in PR template (@rmlarose, gh-359).
- Add limitations of ZNE section to docs (@rmlarose, gh-361).
- Add static extrapolate method to all factories (@andreamari, gh-352).
- Removes barriers in conversions from a Qiskit circuit (@rmlarose, gh-362).
- Add arXiv badge to readme header (@nathanshammah, gh-376).
- Add note on shot list in factory docs (@rmlarose, gh-375).
- Spruce up the README a bit (@karalekas, gh-383).
- Make mypy checking stricter (@karalekas, gh-380).
- Add pyQuil executor examples and benchmarking circuits (@karalekas, gh-339).

[arxiv]: https://arxiv.org/abs/2009.04417

## Version 0.1.0 (September 2nd, 2020)

### Summary

This marks the first public release of Mitiq on a stable version.

### All Changes

- Add static extrapolate method to all factories (@andreamari, gh-352).
- Add the ``angle`` module for parameter noise scaling (@yhindy, gh-288).
- Add to the documentation instructions for maintainers to make a new release (@nathanshammah, gh-332).
- Add basic compilation facilities, don't relabel qubits (@karalekas, gh-324).
- Update readme (@rmlarose, gh-330).
- Add mypy type checking to CI, resolve existing issues (@karalekas, gh-326).
- Add readthedocs badge to readme (@nathanshammah, gh-329).
- Add change log as markdown file (@nathanshammah, gh-328).
- Add documentation on mitigating the energy landscape for QAOA MaxCut on two qubits (@rmlarose, gh-241).
- Simplify inverse gates before conversion to QASM (@andreamari, gh-283).
- Restructure library with ``zne/`` subpackage, modules renaming (@nathanshammah, gh-298).
- [Bug Fix] Fix minor problems in executors documentation (@andreamari, gh-292).
- Add better link to docs and more detailed features (@andreamari, gh-306).
- [Bug Fix] Fix links and author list in README (@willzeng, gh-302).
- Add new sections and more explanatory titles to the documentation's guide (@nathanshammah, gh-285).
- Store optimal parameters after calling reduce in factories (@rmlarose, gh-318).
- Run CI on all commits in PRs to master and close #316 (@karalekas, gh-325).
- Add Unitary Fund logo to the documentation html and close #297 (@nathanshammah, gh-323).
- Add circuit conversion error + tests (@rmlarose, gh-321).
- Make test file names unique (@rmlarose, gh-319).
- Update package version from v. 0.1a2, released, to 0.10dev (@nathanshammah, gh-314).


## Version 0.1a2 (August 17th, 2020)

- **Initial public release**: on [Github][Github] and [PyPI][PyPI].

## Version 0.1a1 (June 5th, 2020)

- **Initial release (internal).**

[Github]: https://github.com/unitaryfund/mitiq
[PyPI]: https://pypi.org/project/mitiq/0.1a2/
