# Changelog

% # " ## Development 0.X.Ydev (Month DDth, YYYY)"
% # " ## (Future) Version 0.1.1 (Date)"
% # " ### Changes"
% # " - **MAJOR FEATURE**: New integration."
% # " - Improve something."
% # " - [Bug Fix]"
% # " - Fix the bug."

## Version 0.8.0 (May 5th, 2021)

### Summary

This release has the following major components:

- Re-implements local folding functions (`zne.scaling.fold_gates_from_left`, `zne.scaling.fold_gates_from_right` and `zne.scaling.fold_gates_at_random`) to make them more uniform at large scale factors and match how they were defined in [Digital zero-noise extrapolation for quantum error mitigation](https://arxiv.org/abs/2005.10921).
- Adds a new noise scaling method, `zne.scaling.fold_all`, which locally folds all gates. This can be used, e.g., to "square CNOTs," a common literature technique, by calling `fold_all` and excluding single-qubit gates.
- Adds functionality for the training portion of [Clifford data regression](https://arxiv.org/abs/2005.10189), specifically to map an input (pre-compiled) circuit to a set of (near) Clifford circuits which are used as training data for the method. The full CDR technique is still in development and will be complete with the addition of regression methods.
- Improves the (sampling) performance of PEC (by a lot!) via fewer circuit conversions.
- Adds `PauliString` object, the first change of several in the generalization of executors. This object is not yet used in any error mitigation pipelines but can be used as a stand-alone.

Additionally, this release 

- Fixes some CI components including uploading coverage from master and suppressing nightly Test PyPI uploads on forks.
- Adds links to GitHub on README and RTD.

Special thanks to all contributors - @purva-thakre, @Aaron-Robertson, @andre-a-alves, @mstechly, @ckissane, @HaoTy, @briancylui, and @L-P-B - for your work on this release! 

### All Changes

- Redefine and re-implement local folding functions (@andreamari, gh-649).
- Move Cirq executors from docs to utilities (@purva-thakre, gh-603).
- Add functionality for the training portion of Clifford Data Regression (@L-P-B, gh-601).
- Fix custom factory example in docs (@andreamari, gh-601).
- Improves PEC sampling speed via fewer conversions (@ckissane, @HaoTy, @briancylui, gh-647).
- Minor typing fixes (@mstechly, gh-652).
- Add new `fold_all` scaling method (@rmlarose, gh-648).
- Add `PauliString` (@rmlarose, gh-633).
- Update information about formatting in RTD (@purva-thakre, gh-622).
- Fix typo in getting started guide (@andre-a-alves, gh-640).
- Suppress Test PyPI nightly upload on forks (@purva-thakre, gh-597).
- Add link to GitHub in README and add octocat link to GitHub on RTD (@andre-a-alves, gh-637).
- Move all benchmark circuit generation to `mitiq.benchmarks`, adding option for converions (@Aaron-Robertson, gh-632).
- Use `execute_with_shots_and_noise` in Qiskit utils test (@Aaron-Robertson, gh-621).
- Install only the Qiskit packages we need (@purva-thakre, gh-614).
- Update PR template (@rmlarose, gh-634).
- Add blurb about unitaryHACK (@nathanshammah).


## Version 0.7.0 (April 7th, 2021)

### Summary

This release focuses on contributions from the community.
Many thanks @yhindi for adding a method to parametrically scale noise in circuit, to @aaron-robertson and @pchung39 for adding Qiskit executors with depolarizing noise to the utils,
to @purva-thakre for several bug fixes and improvements. Thanks also to @BobinMathew and @marwahaha for typo corrections.


### All Changes

- Pin `docutils` version to solve CI issues (@purva-thakre, gh-626)
- Add qiskit executor example for exact density matrix simulation with depolarizing noise (@aaron-robertson, gh-574)
- Replace Qiskit utils with Qiskit executors from docs (@pchung39, @aaron-robertson, gh-584)
- Change custom depolarizing channel in PEC (@purva-thakre, gh-615)
- Document codecov fetch depth change and increase codecov action version (@grmlarose, gh-618)
- Fix spelling typo (@marwahaha, gh-616)
- Add link for MEP (@purva-thakre, gh-610)
- Correct a typo in the readme (@BobinMathew, gh-609)
- Add ``represent_operations_in_circuit_...`` functions for PEC (@andreamari, gh-515)
- Add qiskit and cirq executor examples gifs to readme (@nathanshammah, gh-587)
- Move Cirq executors in docs to cirq_utils (@purva-thakre, gh-603)
- Add parametrized scaling (@yhindyYousef, gh-411)
- Fix mitiq.about() qiskit version (@aaron-robertson, gh-598)
- Fix typo in ZNE documentation (@purva-thakre, gh-602)
- Add minimal section on PEC to documentation (@nathanshammah @andreamari, gh-564)
- Various improvements to CI and testing (@rmlarose, gh-583)
- Corrects typo in docs of error mitigation guide (@purva-thakre, gh-585)
- Remove factory equality and fix bug in `PolyExpFactory.extrapolate` (@rmlarose, gh-580)
- [Bug Fix] Examples (notebooks) in the online documentation now have output code cells (@Aaron-Robertson, gh-576)
- Update version to dev (@andreamari, gh-572)
- Change custom depolarizing channel in PEC test (@purva-thakre, gh-615)

## Version 0.6.0 (March 1st, 2021)

### Summary

The automated workflows for builds and releases are improved and PyPI releases are now automated.
We have more documentation on PEC and have a new tutorial on QAOA with Mitiq, as well as some miscellaneous bug fixes.

### All Changes
- Add minimal section on PEC to documentation (@nathanshammah, gh-564)

- Improve CI and release/patch workflow and documentation (@crazy4pi314 gh-566).
- Adding a Mitiq Enhancement Proposal template and process (@crazy4pi314 gh-563).
- Add to the documentation papers citing Mitiq, close gh-424 (@nathanshammah gh-560).
- Added a new tutorial where MaxCut is solved with a mitigated QAOA (@andreamari, gh-562).
- Bumping the version of Qiskit version in `dev_requirements` (@andreamari gh-554)
- Retain measurement order when folding Qiskit circuits (@rmlarose gh-557)
- Standardize and touch up docs (@rmlarose gh-553)
- Adding make doc-clean command (@crazy4pi314 gh-549)
- Fix and refactor RB circuits function (@rmlarose gh-539)
- Improve GateOperation simplification in exponents (@rmlarose gh-541)

## Version 0.5.0 (February 8th, 2021)

### Summary

The implementation of Probabilistic Error Cancellation is now multi-platform and
documented in the docs (for the moment only in the *Getting Started* section).
A new infrastructure based on [MyST](https://myst-parser.readthedocs.io/en/stable/)
can now be used for writing the documentation and, in particular, for adding new examples
in the *Mitiq Examples* section.

### All Changes

- Adding documentation section for examples, support for Jyupyter notebooks (@crazy4pi314, gh-509).
- Add minimal documentation for PEC in the Getting Started (@andreamari, gh-532).
- Optionally return a dictionary with all pec data in execute_with_pec (@andreamari, gh-518).
- Added local_depolarizing_representations and Choi-based tests (@andreamari, gh-502).
- Add multi-platform support for PEC (@rmlarose, gh-500).
- Added new `plot_data` and `plot_fit` methods for factories(@crazy4pi314 @elmandouh, gh-333).
- Fixes random failure of PEC sampling test(@rmlarose, gh-481).
- Exact copying of shell commands, update make target for pdf and update development version(@rmlarose, gh-469).
- Add a new FakeNodesFactory class based on an alternative interpolation method (@elmandouh, gh-444).
- Add parameter calibration method to find base noise for parameter noise (@yhindy, gh-411).
- Remove duplication of the reduce method in every (non-adaptive) factory (@elmandouh, gh-470).

## Version 0.4.1 (January 12th, 2021)

### Summary

This release fixes a bug in the docs.

### All Changes

- [Bug Fix] Ensure code is tested in IBMQ guide and adds a doctest target in Makefile(@rmlarose, gh-488)

## Version 0.4.0 (December 6th, 2020)

### Summary

This release adds new getter methods for fit errors, extrapolation curves, etc. in ZNE factory objects as well as
custom types for noisy operations, noisy bases, and decompositions in PEC. It also includes small updates and fixes
to the documentation, seeding options for PEC sampling functions, and bug fixes for a few non-deterministic test failures.

### All Changes

- Add reference to review paper in docs (@willzeng, gh-423).
- Add unitary folding API to RTD (@rmlarose, gh-429).
- Add theory subsection on PEC in docs (@elmandouh, gh-428).
- Fix small typo in documentation function name (@nathanshammah, gh-435).
- Seed Qiskit simulator to fix non-deterministic test failure (@rmlarose, gh-425).
- Fix formatting typo and include hyperlinks to documentation objects (@nathanshammah, gh-438).
- Remove error in docs testing without tensorflow (@nathanshammah, gh-439).
- Add seed to PEC functions (@rmlarose, gh-432).
- Consolidate functions to generate randomized benchmarking circuits in different platforms, and clean up pyquil utils (@rmlarose, gh-426).
- Add new get methods (for fit errors, extrapolation curve, etc.) to Factory objects (@crazy4pi314, @andreamari, gh-403).
- Update notebook version in requirements to resolve vulnerability found by security bot.(@nathanshammah, gh-445).
- Add brief description of noise and error mitigtation to readme (@rmlarose, gh-422).
- Fix broken links in documentation (@purva-thakre, gh-448).
- Link to stable RTD instead of latest RTD in readme (@rmlarose, gh-449).
- Add option to automatically deduce the number of samples in PEC (@andreamari, gh-451).
- Fix PEC sampling bug (@rmlarose, gh-453).
- Add types for PEC (@rmlarose, gh-408).
- Add warning for large samples in PEC (@sid1993, gh-459).
- Seed a PEC test to avoid non-deterministic failure (@andreamari, gh-460).
- Update contributing docs (@purva-thakre, gh-465).

## Version 0.3.0 (October 30th, 2020)

### Summary

Factories now support "batched" executors, meaning that when a backend allows
for the batch execution of a collection of quantum circuits, factories can now
leverage that functionality. In addition, the main focus of this release was
implementing probabilistic error cancellation (PEC), which was introduced in
[Temme2017][temme2017] as a method for quantum error mitigation. We completed
a first draft of the major components in the PEC workflow, and in the next
release plan to demonstrate the full end-to-end operation of the new technique.

[temme2017]: https://arxiv.org/abs/1612.02058

### All Changes

- Fix broken links on the website (@erkska, gh-400).
- Use cirq v0.9.0 instead of cirq-unstable (@karalekas, gh-402).
- Update mitiq.about() (@rmlarose, gh-399).
- Refresh the release process documentation (@karalekas, gh-392).
- Redesign factories, batch runs in BatchedFactory, fix Qiskit utils tests (@rmlarose, @andreamari, gh-381).
- Add note on batched executors to docs (@rmlarose, gh-405).
- Added Tensorflow Quantum executor to docs (@k-m-schultz, gh-348).
- Fix a collection of small build & docs issues (@karalekas, gh-410).
- Add optimal QPR decomposition for depolarizing noise (@karalekas, gh-371).
- Add PEC basic implementation assuming a decomposition dictionary is given (@andreamari, gh-373).
- Make tensorflow requirements optional for docs (@karalekas, gh-417).

Thanks to @erkska and @k-m-schultz for their contributions to this release!


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
