# Changelog

## Version 0.11.1  (November 29th, 2021)

### Summary

This patch release fixes two bugs:

- Bug: PEC could only be used with `cirq.Circuit`s, not `mitiq.QPROGRAM`, due to a missing conversion.
    - Fix: PEC can now be used with any `mitiq.QPROGRAM` (gh-1018).
- Bug: CDR classically simulated the wrong circuits when doing regression. 
    - Fix: The correct circuits are now classically simulated (gh-1026).

Also fixes a smaller bug where some tools in `mitiq.interface.mitiq_qiskit` modified `qiskit.QuantumCircuit`s when they shouldn't.

### All Changes

- Update scipy requirement from ~=1.7.1 to ~=1.7.2 (@dependabot, gh-1017)
- CDR: Run the training circuits on the simulator (@rmlarose and @andreamari, gh-1026).
- Update scipy requirement from ~=1.7.1 to ~=1.7.2 (@dependabot, gh-1017)
- Update pydata-sphinx-theme requirement from ~=0.7.1 to ~=0.7.2 (@dependabot, gh-1024)
- Update qiskit requirement from ~=0.31.0 to ~=0.32.0 (@dependabot, gh-1025)
- Update pydata-sphinx-theme requirement from ~=0.7.1 to ~=0.7.2 (@dependabot, gh-1024)
- [Bug fix] Avoid circuit mutation in qiskit executors (@andreamari, gh-1019)
- [Bug fix] Add back-conversions in execute_with_pec (@andreamari, gh-1018) 
- Increase shots in zne tests with shot_list (@andreamari, gh-1020)
- Update pennylane requirement from ~=0.18.0 to ~=0.19.0 (@nathanshammah, gh-1022)
- Add workflow figures and technique descriptions (@nathanshammah, gh-953)
- Prepare to release 0.11.0 (@rmlarose, gh-1010)

## Version 0.11.0  (November 3rd, 2021)

### Summary

This release introduces `Observable`s as a major new feature to the Mitiq workflow and a few breaking changes. Support
for Pennylane has been added by adding `pennylane.QuantumTape`s to `mitiq.QPROGRAM`.

**New features**

- Specify and use a `mitiq.Observable` in any error-mitigation technique.
  - This means the `executor` function does not have to return the expectation value as a `float` anymore, but rather 
    can return a `mitiq.QuantumResult` - i.e., an object from which the expectation value can be computed provided 
    an observable.
  - The `executor` function can still return a `float`, in which case the `Observable` does not need to be specified
    (and should not be specified).

- All error mitigation techniques can now use batching with the same interface.

- PEC can be run with only a subset of representations of the gates in a circuit. In other words, if the circuit has 
  two gates, `H` and `CNOT`, you can run `execute_with_pec` by only providing an `OperationRepresentation` for, e.g.,
  the `CNOT`.
  - Before, you had to provide all representations or an error would be raised.
  - Performance of PEC may be better by providing all `OperationRepresentation`s. This change is only with respect to
    usability and not performance.

- Circuits written in `Pennylane` (as `QuantumTape`s) are now recognized and suppported by Mitiq.

**Breaking changes**

- Signatures of `execute_with_xxx` error-mitigation techniques have changed to include `Observable`s. The default value
is `None`, meaning that the `executor` should return a `float` as in the old usage, but the additional argument and
change to keyword-only arguments (see below) may require you to make updates.

- You must now use provide keywords for technique-specific arguments in error mitigation techniques. Example:

```python
# Example new usage of providing keyword arguments. Do this.
execute_with_pec(circuit, executor, observable, representations=representations)
```

instead of

```python
# Old usage. Don't do this. Technique-specific arguments like `representations` are now keyword-only.
execute_with_pec(circuit, executor, observable, representations)
```

The latter will raise `# TypeError: execute_with_pec() missing 1 required keyword-only argument: 'representations'`.

- The first argument of `execute_with_zne` is now `circuit` instead of `qp` to match signatures of other 
`execute_with_xxx` functions.

### All Changes

- Increase shots in test_zne.py (@andreamari, gh-1011)
- Refactor `qp` to `circuit` in `execute_with_zne` (@rmlarose, gh-1009)
- Add `Observable` documentation and `Observable.from_pauli_string_collections` method (@rmlarose, gh-1007)
- Executor docs (@rmlarose, gh-1008)
- Bump qiskit to version 0.31 and pin it explicitly in dev requirements (@andreamari, gh-993)
- Use `Executor.evaluate` for batching in ZNE (@rmlarose, gh-1005)
- Add `Executor.evaluate` and use for batched execution (@rmlarose, gh-1001)
- Bump PyQuil (@rmlarose, gh-992)
- CDR with Observables (@rmlarose, gh-985)
- Fix bug in OperationRepresentation printing (@andreamari, gh-975)
- Ignore patch releases in dependabot (@rmlarose, gh-981)
- Update pydata-sphinx-theme requirement from ~=0.6.3 to ~=0.7.1 (@dependabot, gh-962)
- Update flake8 requirement from ~=3.9.2 to ~=4.0.1 (@dependabot, gh-982)
- Add PennyLane to frontend table in the readme (@andreamari, gh-973)
- Update amazon-braket-sdk requirement from ~=1.9.1 to ~=1.9.5 (@dependabot, gh-970)
- Update pytest-cov requirement from ~=2.12.1 to ~=3.0.0 (@dependabot, gh-963)
- Fix pip package resolving problems (@andreamari, gh-976)
- Add support for pennylane circuits (everybody and their grandmother, gh-836)
- Keyword only arguments in execute_with_technique functions (@rmlarose, gh-971)
- Foldability check includes a check for inverse (@purva-thakre, gh-939) 
- Bump actions/github-script from 3 to 5 (@dependabot, gh-969) 
- PEC with Observables & skip operations without known representations (@rmlarose, gh-954)
- Update qiskit-terra requirement from ~=0.18.2 to ~=0.18.3 (@dependabot, gh-955) 
- Add observable to zne_decorator (@rmlarose, gh-967)
- Update qiskit-ibmq-provider requirement from ~=0.16.0 to ~=0.17.0 (@dependabot, gh-965) 
- Binder badge workflow (@AkashNarayanan, gh-964)
- Fix links and typos in vqe-pyquil-demo.myst and pyquil_demo.myst (@Misty-W, gh-959)
- ZNE with Observables (@rmlarose, gh-948)
- Add PauliString multiplication (@rmlarose, gh-949)
- Update amazon-braket-sdk requirement from ~=1.9.0 to ~=1.9.1 (@dependabot, gh-950)
- Fixing changelog and improving release documentation (@crazy4pi314, gh-946)
- Update version to 0.11.0dev (@rmlarose, gh-947)
- Update pytest-xdist[psutil] requirement from ~=2.3.0 to ~=2.4.0 (@dependabot, gh-938)
- Fix AWS example by fixing a bug in mirror circuits (@andreamari, gh-940)
- Bump codecov/codecov-action from 2.0.3 to 2.1.0 (@dependabot, gh-922)
- Improve OperationRepresentation printing (@andreamari, gh-901)
- Updating release process (@crazy4pi314, gh-936)

Huge thanks to all contributors on this release! @Misty-W, @AkashNarayanan, @purva-thakre, @trbromley, @nathanshammah,
@crazy4pi314, @andreamari, and @rmlarose.

## Version 0.10.0  (September 17, 2021)

### Summary

This release adds a ton of new stuff, both error mitigation tools as well as infrastructure upgrades.
Some highlights:

- New integration with [AWS Braket](https://aws.amazon.com/braket/)
- A new pyQuil example by @Misty-W and lots of pyQuil debugging.
- Support for mirror circuits by @DSamuel1.
- Dependabot is now helping us keep our dependencies up to date.
- Lots of documentation fixes and features like a gallery view of the examples and mybinder.org support.
- New `Observable` and `MeasurementResult` dataclass.
  
Thanks to @Misty-W and @DSamuel1 for their great contributions this release! 🎉

### All Changes

- Reduce noise in braket example (@andreamari, gh-933)
- Add ZNE example on AWS Braket (@rmlarose, gh-929)
- A few changes / fixes to mirror circuits (@rmlarose, gh-928)
- Change Binder link from jupyterlab to classic view (@andreamari, gh-925)
- Dependabot settings patch: one dependancy per line (@rmlarose, gh-921)
- Dependabot settings (@rmlarose, gh-920)
- Update sphinxcontrib-bibtex requirement from ~=2.3.0 to ~=2.4.1 (@dependabot, gh-919)
- Rename Mitiq Examples -> Examples (@nathanshammah, gh-916)
- Update qiskit-terra requirement from ~=0.18.1 to ~=0.18.2 (@dependabot, gh-912)
- Add `Observable.expectation` with support for measurement results and density matrices; Refactor `Collector` -> `Executor` (@rmlarose, gh-904)
- Install mitiq in readthedocs to avoid mitiq.py files (@andreamari, gh-903)
- Crazy4pi314/example gallery (@crazy4pi314, gh-902)
- Import mirror circuits (@rmlarose, gh-900)
- Mirror Circuits Update: Resolves #890 and #891 (@DSamuel1, gh-895)
- Fix html rendering problems of the PEC example (@andreamari, gh-894)
- Add `qubit_indices` to `MeasurementResult` (@rmlarose, gh-892)
- Update mirror circuits docstrings (@rmlarose, gh-889)
- Update flake8 requirement from ~=3.7.9 to ~=3.9.2 (@dependabot, gh-887)
- Update pytest-cov requirement from ~=2.11.1 to ~=2.12.1 (@dependabot, gh-885)
- Update sphinx-copybutton requirement from ~=0.3.0 to ~=0.4.0 (@dependabot, gh-884)
- Update mypy requirement from ~=0.812 to ~=0.910 (@dependabot, gh-883)
- Update pytest-xdist[psutil] requirement from ~=2.2.1 to ~=2.3.0 (@dependabot, gh-880)
- Update sphinxcontrib-bibtex requirement from ~=2.2.0 to ~=2.3.0 (@dependabot, gh-879)
- Update amazon-braket-sdk requirement from ~=1.5.10 to ~=1.8.0 (@dependabot, gh-878)
- Bump codecov/codecov-action from 1.3.1 to 2.0.3 (@dependabot, gh-875)
- Bump actions/stale from 3.0.19 to 4 (@dependabot, gh-874)
- Updating pinned Scipy version (@crazy4pi314, gh-871)
- Add some missing package metadata causing problems (@crazy4pi314, gh-870)
- Fixing syntax for dependabot to make it work correctly (@crazy4pi314, gh-866)
- Make `MeasurementResult` a dataclass; Add expectation from measurements (@rmlarose, gh-860)
- Mirror circuits (@DSamuel1, gh-859)
- Fix link in README (@andreamari, gh-856)
- Add step-by-sep tutorial on PEC (@andreamari, gh-854)
- Adding mitiq survey to readme (@crazy4pi314, gh-853)
- Add `Observable` (@rmlarose, gh-852)
- Manual PyPI deployment trigger (@crazy4pi314, gh-851)
- Make sure stale marking of issues happens (@crazy4pi314, gh-849)
- Fix invisible output in mitiq codeblocks (@andreamari, gh-847)
- update references in README (@andreamari, gh-846)
- Update to latest Qiskit (@rmlarose, gh-845)
- Set measurement result type and add post-selection (@rmlarose, gh-844)
- pyQuil parametric compilation example (@Misty-W, gh-843)
- Add My Binder (@nathanshammah, gh-841)
- Show output of Mitiq paper code blocks in example on RTD (@nathanshammah, gh-840)
- Add code blocks from the paper (@nathanshammah, gh-838)
- Make ZNE (more) usable with PyQuil (@rmlarose, gh-835)
- Improve PEC efficiency with batched sampling (@andreamari, gh-833)
- Fix warnings in docs log output during build (@andreamari, gh-832)
- Update readme and remove research.rst (@rmlarose, gh-831)
- Add linkcheck to docs build and fix broken links (@rmlarose, gh-827)
- Initial support for executors which return measurement results: part 1/2 (@rmlarose, gh-826)
- Adding table of techniques to readme (@crazy4pi314, gh-825)
- Move example/template notebook to contributing TOC (@rmlarose, gh-814)
- Fix multiplication order when adding NoisyOperations (@andreamari, gh-811)
- Better error message for `CircuitConversionError`s (@rmlarose, gh-809)
- Fix some documentation not being tested & remove global imports in docs config (@rmlarose, gh-804)

## Version 0.9.3  (July 7th, 2021)

### Summary

This primary reason for this patch release is to fix a bug interfacing with Qiskit circuits (gh-802).

### All Changes

- [Docs] Add CDR to README and braket to Overview (@rmlarose, gh-778).
- Rename parameter calibration function and make it visible (@rmlarose, gh-780).
- Allow adding qubits when transforming registers in a Qiskit circuit (@rmlarose, gh-803).

## Version 0.9.2  (June 30th, 2021)

### Summary

This patch release fixes a Braket integration bug (gh-767).
It also adds an example about Clifford data regression in the documentation.

### All Changes

- Ensure short circuit warning is multi-platform (@andreamari gh-769).
- Add CDR example to docs + small change to `cdr.calculate_observable` (@rmlarose, gh-750).

## Version 0.9.1  (June 24th, 2021)

### Summary

This is a patch release to fix two bugs (gh-736, gh-737) related to the integration with optional packages.
It also fixes other minor problems (see the list of changes below).

### All Changes

- Patch 0.9.0 (@rmlarose, gh-739).
- Make readthedocs succeed in building the pdf with pdflatex (@andreamari, gh-743).
- Update energy landscape example in docs (@andreamari, gh-742).
- Remove old deprecation warnings (@rmlarose, gh-744).

## Version 0.9.0 (June 17th, 2021)

### Summary

The main addition introduced in this release is the implementation of a new error mitigation technique: (variable-noise) Clifford data regression ([arXiv:2005.10189](https://arxiv.org/abs/2005.10189), [arXiv:2011.01157](https://arxiv.org/abs/2011.01157)). This is structured as
a Mitiq module called `mitiq.cdr`.

Another important change is the integration with Amazon Braket, such that Mitiq is now compatible with circuits of type `braket.circuits.Circuit`. Moreover all the existing Mitiq integrations are now organized into unique module called `mitiq.interface`.

In the context of probabilistic error cancellation, the sub-module `mitiq.pec.representations` has been significantly improved. Now one can easily compute the optimal quasi-probabiliy representation of an ideal gate as a linear combination of `NoisyOperation` objects.

Thanks to all contributors (@L-P-B, @Aaron-Robertson, @francespoblete, @LaurentAjdnik, @maxtremblay, @andre-a-alves, @paniash, @purva-thakre) and in particular to the participants of [unitaryHACK](https://unitaryfund.github.io/unitaryhack/)!

### All Changes

- New notation NoisyOperation(circuit, channel_matrix) (@andreamari, gh-725).
- Update docs for transforming Qiskit registers (@rmlarose, gh-724).
- Remove classical register transformations in Qiskit conversions (@Aaron-Robertson, gh-672).
- Change return format for `execute_with_cdr` (@rmlarose, gh-722).
- Organize supported packages in `mitiq.interface` (@rmlarose, gh-706).
- Change requirements to Cirq 0.10 (@andreamari, gh-717).
- Make CDR work with any `QPROGRAM` (@rmlarose, gh-718).
- Clearer return type for extrapolate and remove deprecated method (@rmlarose, gh-714).
- Update feature request template (@rmlarose, gh-713).
- CDR part two - Clifford data regression functions to fit training data (@L-P-B gh-677).
- Consolidate images, update README. (@rmlarose, gh-709).
- New logo for readme (@crazy4pi314, @francespoblete,  gh-709).
- Update Guidelines for Release (@nathanshammah, gh-707).
- Add action to close stale issues/pr #698 (@crazy4pi314, gh-699).
- Optimal quasi-probability representations (@andreamari, gh-701).
- Update links to Cirq documentation (@LaurentAjdnik, gh-704).
- Warning for short programs [unitaryHACK] (@Yash-10, gh-700).
- Preparing for optimal representations: Helper functions for channel manipulation. (@andreamari, gh-694).
- [UnitaryHACK] Improve conversion from braket to cirq (@maxtremblay, gh-688).
- [unitaryHack] Add instructions to solve make docs error for Windows/3.8 users (@andre-a-alves, gh-691).
- Add link to docs for source installation in README (@paniash, gh-682).
- Mention Braket in the README (@andreamari, gh-687).
- [UnitaryHACK] ZNE-PEC uniformity (@andre-a-alves, gh-656).
- Minor update of parameter noise scaling (@andreamari, gh-684).
- Add braket support via rudimentary translator (@rmlarose, gh-590).
- Specify qiskit elements in output of about.py (@purva-thakre, gh-674).
- Depend only on cirq-core (@rmlarose, gh-673). This was reverted in gh-717.
- Fix docs build (@rmlarose, gh-671).
- Fix is_clifford logic for Clifford Data Regression (@L-P-B gh-669).


## Version 0.8.0 (May 6th, 2021)

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

- Remove docs/pdf in favor of RTD (@rmlarose, gh-662).
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
