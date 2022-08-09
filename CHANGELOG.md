# Changelog

## Version 0.17.1

### Summary

This patch release includes support for the latest versions of Qiskit (0.37.1), Cirq (1.0.0), and Pyquil (3.2.1), along with other minor dependency upgrades and bug fixes.

### All Changes

- Update pyquil requirement from ~=3.0.0 to ~=3.2.1 (#1425) [@dependabot]
- Update cirq requirement from ~=0.15.0 to ~=1.0.0 (#1402) [@dependabot]
- Add fold methods as parameters to benchmarks (#1374) [@natestemen]
- Updates tests and docstrings for cirq utils (#1371) [@purva-thakre]
- Update amazon-braket-sdk requirement from ~=1.27.1 to ~=1.28.1 (#1430) [@dependabot]
- Bump openfermion from 1.5.0 to 1.5.1 (#1429) [@dependabot]
- Update flake8 requirement from ~=5.0.3 to ~=5.0.4 (#1423) [@dependabot]
- Update flake8 requirement from ~=4.0.1 to ~=5.0.3 (#1419) [@dependabot]
- Support for Qiskit 0.37 (#1421) [@1ucian0]
- Update amazon-braket-sdk requirement from ~=1.26.1 to ~=1.27.1 (#1417) [@dependabot]
- Bump openfermion from 1.3.0 to 1.5.0 (#1412) [@dependabot]
- Update amazon-braket-sdk requirement from ~=1.26.0 to ~=1.26.1 (#1405) [@dependabot]
- Update mypy requirement from ~=0.961 to ~=0.971 (#1406) [@dependabot]
- Update release docs (#1404) [@natestemen]
- Update amazon-braket-sdk requirement from ~=1.25.2 to ~=1.26.0 (#1403) [@dependabot]
- update version.txt to dev (#1398) [@natestemen]
- Prevent idle qubits with qiskit barriers from being lost after conversions (#1369) [@andreamari]
- Add custom RTD favicon (#1390) [@natestemen]
- Update python-rapidjson requirement from <=1.6 to <1.8 (#1389) [@dependabot]

## Version 0.17.0

### Summary

This release includes contributions from UnitaryHACK 2022! 🎉
We had 3 merged pull requests, and a fourth is looking to be merged in the next milestone.
Along with the great contributions from hackers, this release focused on expanding our ZNE examples to other frontends (Cirq, Braket, and Pennylane), building out learning-based PEC, and improving our benchmarking infrastructure.


### All Changes

- Support the latest Cirq version (0.15.0) (@andreamari, gh-1379)
- add ZNE example using pennylane (@natestemen, gh-1384)
- Add ZNE example with Braket and IonQ (@andreamari, gh-1363)
- docs: add natestemen as a contributor for infra, doc, test, code (allcontributors, gh-1383)
- remove comma (@natestemen, gh-1381)
- add myself to authors (@natestemen, gh-1372)
- Deploy asv benchmark data to web (@natestemen, gh-1354)
- Batched mitigate (@nylewong, gh-1286)
- Update black requirement from ~=22.3 to ~=22.6 (@dependabot, gh-1370)
- Add Error Mitigation example with Cirq (@misty-w, gh-1351)
- Update amazon-braket-sdk requirement from ~=1.25.1 to ~=1.25.2 (@dependabot, gh-1367)
- Update pennylane-qiskit requirement from ~=0.23.0 to ~=0.24.0 (@dependabot, gh-1361)
- Update Mitiq paper code blocks based on paper review (@andreamari, gh-1357)
- Update pennylane requirement from ~=0.23.1 to ~=0.24.0 (@dependabot, gh-1362)
- Update amazon-braket-sdk requirement from ~=1.25.0 to ~=1.25.1 (@dependabot, gh-1359)
- Improving the H2 example (@obliviateandsurrender, gh-1337)
- Test on windows and mac on each PR and add openfermion to dev requirements (@andreamari, gh-1348)
- Loss function for learning biased noise (@misty-w, gh-1340)
- Modify docstrings, documentation to use citations from refs.bib [unitaryhack] (@wingcode, gh-1325)
- [unitaryhack] v2 Updated molecular hydrogen tutorial OpenFermion (@lockwo, gh-1349)
- Update amazon-braket-sdk requirement from ~=1.24.0 to ~=1.25.0 (@dependabot, gh-1345)
- Bump actions/setup-python from 3 to 4 (@dependabot, gh-1344)
- [unitaryhack] Add thumbnails for examples (@obliviateandsurrender, gh-1338)
- Revert "[unitaryhack] Updated molecular hydrogen tutorial" (@andreamari, gh-1347)
- [unitaryhack] Updated molecular hydrogen tutorial (@lockwo, gh-1296)
- Update mypy requirement from ~=0.960 to ~=0.961 (@dependabot, gh-1342)
- Merge pull request #1320 from unitaryfund/nts-contrib-docs (@natestemen)


## Version 0.16.0 (June 3rd, 2022)
### Summary

This release officially adds support for the digital dynamical decoupling (DDD) technique which is now [fully documented](https://mitiq.readthedocs.io/en/latest/guide/ddd.html) and so ready to be applied by Mitiq users. This is still very new technique and so, if you notice any bugs, please let us know by
opening [issues](https://github.com/unitaryfund/mitiq/issues) on GihHub.
A further notable addition is the function [generate_quantum_volume_circuit()](https://mitiq.readthedocs.io/en/latest/apidoc.html#mitiq.benchmarks.quantum_volume_circuits.generate_quantum_volume_circuit) by @nickdgardner, extending the Mitiq benchmarking module with quantum volume
circuits.

Congratulations to the new member of the Mitiq team @natestemen and special thanks to the external contributors @Aaron-Robertson, @nickdgardner, @ZhaoyiLi-HekJukZaaiZyuJan and @amirebrahimi!

### All Changes

- Add section 2 (use case) on DDD to user guide (@nathanshammah, gh-1304)
- Update DDD options documentation (third section) (@andreamari, gh-1327)
- Add additional options subsection for DDD (third section) (@Aaron-Robertson, gh-1326)
- Add theory subsection for DDD (fifth section) (@andreamari, @natestemen, gh-1311)
- Update amazon-braket-sdk requirement from ~=1.19.0 to ~=1.24.0 (@dependabot gh-1324, gh-1310, gh-1306, gh-1303, gh-1297, gh-1278, gh-1271)
- Ensure lists in documentation are displayed properly (@natestemen, gh-1322)
- Add quantum volume circuits to benchmarking module (@nickdgardner, gh-1281)
- Update readme table with DDD (@andreamari, gh-1318)
- Test PR for unitaryhack test bounty (@nathanshammah, gh-1317)
- Fixes unexpected error from repeated rule (@Aaron-Robertson, gh-1316)
- Update mypy requirement from ~=0.950 to ~=0.960 (@dependabot, gh-1314)
- Use single URL when referring to google style (@natestemen, gh-1309)
- Update close-stale.yml (@andreamari, gh-1295)
- Update contributing docs (@amirebrahimi, gh-1305)  
- Add fourth section of DDD docs (@andreamari, gh-1276) 
- Fix citation file (@natestemen, gh-1299) 
- Write the first sub-section of the DDD guide: How do I use DDD? (@Misty-W, gh-1277) 
- Update qiskit requirement from ~=0.36.1 to ~=0.36.2 (@dependabot, gh-1302) 
- Rename test_randomized_benchmaking.py to test_randomized_benchmarking.py (@nickdgardner, gh-1284)
- Fix broken links in docs (@andreamari, gh-1282) 
- Create file structure (@andreamari, gh-1275)
- Update pennylane requirement from ~=0.23.0 to ~=0.23.1 (@dependabot, gh-1274) 
- Fixed typo in documentation formula (@ZhaoyiLi-HekJukZaaiZyuJan, gh-1246) 
- Update version to 0.16dev (@nathanshammah, gh-1268)


## Version 0.15.0 (April 29th, 2022)

### Summary
This milestone focused on updating dependencies and making progress on two new features, dynamical decoupling and learning based PEC. For dynamical decoupling, high-level functions and rules were added. For learning-based PEC, a function calculating representations with a biased (combination of depolarizing and dephasing) noise model was added. Several high priority bugs and issues were also fixed.

Special thanks to new contributors @RubidgeCarrie and @nickdgardner for their contributions to this release!

### All Changes

- Add digital dynamical decoupling high-level functions (@andreamari, gh-1251)
- Fix qubit naming in OperationRepresentations for Qiskit circuits (@Misty-W, gh-1238)
- Fix broken link in contributing guide (@Misty-W, gh-1257)
- Insert sequences for dynamical decoupling (@Aaron-Robertson, gh-1221)
- Short sequences in DDD API docs (@andreamari, gh-1237)
- Update pennylane-qiskit requirement from ~=0.22.0 to ~=0.23.0 (@dependabot, gh-1249)
- Update qiskit requirement from ~=0.36.0 to ~=0.36.1 (@dependabot, gh-1242)
- Update pennylane requirement from ~=0.22.2 to ~=0.23.0 (@dependabot, gh-1248)
- Move the gh pages deploy to a manually triggered action (@crazy4pi314, gh-1243)
- Update AUTHORS (@nickdgardner, gh-1239)
- fix typo (@nickdgardner, gh-1240)
- Add DDD to Api-Doc, fixes #1209 (@nathanshammah, gh-1226)
- Add rules for dynamical decoupling (@Aaron-Robertson, gh-1202)
- Fix compilation to native gateset in Braket notebook (@andreamari, gh-1225)
- Update amazon-braket-sdk requirement from ~=1.17.0 to ~=1.19.0 (@ dependabot, 1235)
- Function to represent OperationRepresentations for biased noise (@Misty-W, gh-1233)
- Bump codecov/codecov-action from 2.1.0 to 3.0.0 (@dependabot, gh-1216)
- Bump actions/stale from 4 to 5 (@dependabot, gh-1217)
- Update sphinxcontrib-bibtex requirement from ~=2.4.1 to ~=2.4.2 (@dependabot, gh-1218)
- Update cirq requirement from ~=0.14.0 to ~=0.14.1 @dependabot, gh-1219)
- Added citiation file for the repo (RubidgeCarrie, gh-1200)
- Add mitigate_executor and cdr_decorator to mitiq.cdr (@Aaron-Robertson, gh-1204)
- Update qiskit requirement from ~=0.35.0 to ~=0.36.0 (@dependabot, gh-1213)

## Version 0.14.0  (April 6th, 2022)

### Summary

This milestone focused on updating dependencies and making progress on two new features, dynamical decoupling and learning based error mitigation techniques. A number of high priority bugs and issues also were fixed.

### All Changes

- Update pennylane requirement from ~=0.22.1 to ~=0.22.2 (@dependabot, gh-1203)
- Update mypy requirement from ~=0.931 to ~=0.942 (@dependabot, gh-1183)
- Update pennylane-qiskit requirement from ~=0.20.0 to ~=0.22.0 (@dependabot, gh-1170)
- Update pennylane requirement from ~=0.21.0 to ~=0.22.1 (@dependabot, gh-1172)
- Update qiskit requirement from ~=0.34.2 to ~=0.35.0 (@dependabot, gh-1199)
- Update black requirement from ~=22.1 to ~=22.3 (@dependabot @andreamari @crazy4pi314, gh-1191)
- Avoid creation of empty moments when using fold_global (@andreamari, gh-1196)
- Merge pull request #1198 from unitaryfund/1140-raise-error-if-executo (@Misty-W)
- Port changes from old branch - workaround (@Misty-W)
- Merge pull request #1195 from unitaryfund/update-qiskit (@Misty-W)
- Update docs/source/guide/pec-3-options.myst @crazy4pi314 (@Misty-W)
- Merge branch 'master' into update-qiskit (@crazy4pi314)
- Update Cirq to 0.14  (@andreamari, gh-1193)
- fix pec-3-options notebook problem (@andreamari)
- Remove option to squash moments in global folding  (@purva-thakre @andreamari, gh-1113)
- update qiskit and fix conversions (@andreamari)
- Idle qubits in conversion  (@Aaron-Robertson, gh-1185)
- Add get_circuit_mask() for dynamical decoupling  (@Aaron-Robertson, gh-1178)
- Update pydata-sphinx-theme requirement from ~=0.8.0 to ~=0.8.1  (@dependabot, gh-1188)
- Add get_slack_matrix_from_circuit_matrix() function for dynamical decoupling (@andreamari @Aaron-Robertson,gh-1178)
- Fix sphinx dependency problem  (@andreamari , gh-1181)
- Create file structure for digital dynamical decoupling  (@andreamari, gh-1175)
- adding pages publish step (@crazy4pi314,gh-1135)
- cast gate exponents to float (@andreamari,gh-1174)
- Update release.rst (@andreamari,gh-1145)
- Bump actions/setup-python from 2 to 3  (@dependabot, gh-1149)
- Bump actions/checkout from 2 to 3  (@dependabot, gh-1151)
- Update amazon-braket-sdk requirement from ~=1.15.0 to ~=1.17.0  (@dependabot, gh-1153)
- fixing install permissions, container user is root (@crazy4pi314 , gh-1147)

## Version 0.13.0  (February 25th, 2022)

### Summary

Mitiq is now compatible with the latest version (0.13.1) of Cirq! This update was blocked for a long time because of some technical difficulties. So, many thanks to @vtomole for finding a solution to this issue!
This should solve several dependency conflicts or warnings that you may have got when running `pip install mitiq` or `pip install -U mitiq`.

The HTML rendering of all PyQuil examples in our documentation is now fixed. Thanks @astrojuanlu for useful suggestions about readthedocs!



We also thank @Rahul-Mistri for adding GHZ circuits to our benchmarking module and for making Clifford circuits compatible with the Mitiq CDR technique (instead of raising an error as it happened before this release).

We discussed and approved the design documents (RFC) for two new error-mitigation techniques: _learning-based PEC_ and _digital dynamical decoupling_. You can find them at [this link](https://github.com/unitaryfund/mitiq/projects/7). Special thanks go to @Misty-W and @Aaron-Robertson!

### All Changes

- Add pre-executed pyquil notebooks (@andreamari, gh-1142)
- Fix optimal representation tests and unskip one of them (@andreamari gh-1141)
- Update amazon-braket-sdk requirement from ~=1.11.1 to ~=1.15.0 (@dependabot, gh-1137, gh-1116, gh-1108, gh-1105) 
- Update black requirement from ~=19.10b0 to ~=22.1 (@dependabot, @crazy4pi314, gh-1110) 
- Bump actions/github-script from 5 to 6 (@dependabot, gh-1129) 
- Update mypy requirement from ~=0.930 to ~=0.931 (@dependabot, gh-1078) 
- docs: add vtomole as a contributor for test, code (@allcontributors, @andreamari, gh-1132) 
- docs: add Rahul-Mistri as a contributor for test, code (@allcontributors, @andreamari, gh-1130) 
- docs: add L-P-B as a contributor for test, code (@allcontributors, gh-1131) 
- Update PR template (@nathanshammah, gh-1117) 
- Remove unused functions from `cirq_utils` and fix non-deterministic tests. (@andreamari gh-1123)
- Update pennylane requirement from ~=0.20.0 to ~=0.21.0 (@dependabot, gh-1122) 
- Bump cirq version from 0.10.0 to 0.13.0 (@vtomole, gh-988)
- Can use Clifford Circuits with `execute_with_cdr` (@Rahul-Mistri, gh-1104) 
- Docstring for GHZ-circuits reformatted (@Rahul-Mistri, gh-1101)




## Version 0.12.0  (January 21st, 2022)

### Summary

This release contains a considerable overhaul of the documentation organization and content:

- The guide is now divided into Core concepts and a new presentation of the quantum errror mitigation techniques (ZNE, PEC and CDR). Each technique contains subsections that explain with code snippets how to use them in Mitiq (gh-1021, gh-1004 gh-1031, gh-1099). Also the API doc has been extended and improved. Many thanks to @purvathakre @Misty-W for their help on rewriting the documentation and reviewing the pull requests.
- An example on how to use ZNE to improve the calculations of the energy potential landscape of molecular Hydrogen using VQE was added by @andreamari.

**New features**
- GHZ circuits were added to the benchmark subpackage by @Rahul-Mistri.
- Airspeed-velocity (asv) has been added to the CI by @rmlarose.

### All Changes

- Add core concepts guide page (@crazy4pi314, @nathanshammah, @andreamari, gh-1053)
- Add cdr-2-use-case.myst (gh-1099) (@andreamari, @nathanshammah)
- CDR documentation reorg (@nathanshammah, @andreamari, @crazy4pi314, gh-1031)
- Add molecular Hydrogen example (@andreamari, gh-1087)
- Update pydata-sphinx-theme requirement from ~=0.7.2 to ~=0.8.0 (@dependabot, gh-1091)
- Use raw.execute in asv benchmarks and remove GHZ circuits from asv (@rmlarose)
- Added GHZ circuits to benchmark (@Rahul-Mistri, gh-1089)
- Update scipy requirement from ~=1.7.2 to ~=1.7.3 (@dependabot
, gh-1084)
- PR to trigger CI and fix a broken link (@andreamari, gh-1082)
- Fix readthedocs by changing a LaTex equation (@andreamari, @nathanshammah, gh-1077)
- Add asv benchmarking framework (@rmlarose, gh-1047)
- Update zne-3-options.myst (@andreamari, gh-1075)
- Update API doc with REM, executors and observables (@nathanshammah, gh-1050)
- ZNE Guide Reorg (@crazy4pi314, @Misty-W, @purva-thakre, @nathanshammah, @andreamari, gh-1021)
- Fix broken link on master. (@andreamari, gh-1069)
- Update mypy requirement from ~=0.910 to ~=0.930 (@dependabot, gh-1065)
- Docs: add AkashNarayanan as a contributor for infra (@allcontributors, gh-1059)
- Fixes #1034 release docs update (@crazy4pi314, @nathanshammah, gh-1054)
- Docs: add DSamuel1 as a contributor for code (@allcontributors, @Misty-W, gh-1056)
- Docs: add Misty-W as a contributor for code, example (@allcontributors, gh-1055)
- Docs: update README.md (@allcontributors)
- Update PEC docs (@andreamari, @nathanshammah, @rmlarose, gh-1004)
- Revert "Update mypy requirement from ~=0.910 to ~=0.920 (@nathanshammah, gh-1052)
- Update mypy requirement from ~=0.910 to ~=0.920 (@dependabot, gh-1051)
- Update pytest-xdist[psutil] requirement from ~=2.4.0 to ~=2.5.0 (@dependabot, gh-1044)
- Update amazon-braket-sdk requirement from ~=1.11.0 to ~=1.11.1 (@dependabot, gh-1042)
- Update pennylane requirement from ~=0.19.0 to ~=0.19.1 (@dependabot, gh-1029)
- Update amazon-braket-sdk requirement from ~=1.9.5 to ~=1.11.0 (@dependabot, gh-1038)


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
