![Python Build](https://github.com/unitaryfund/mitiq/workflows/Python%20Build/badge.svg?branch=master)

# Mitiq Documentation
This is the documentation of Mitiq, a Python toolkit for
implementing error mitigation on quantum computers.

## Requirements
The documentation is generated with
[Sphinx](https://www.sphinx-doc.org/en/master/usage/installation.html).
```bash
pip install -U sphinx
```
To check that Sphinx is installed you can run
```bash
pip install -U sphinx
```

## How to Build the Documentation
Follow this recipe:
- Create a conda environment for the documentation
```bash
conda create -n mitiqenv
conda activate mitiqenv
```
- Create a branch in `git` for the documentation with the release number up to
minor (e.g., 0.0.2--->00X)
```bash
(mitiqenv) git checkout -b mitiq00X
```
- Since the documentation is already created, you need not to generate it
from scratch. If you had to generate it from scratch, the first step would
involve creating the `conf.py` file. This can be generated with a wizard
```bash
(mitiqenv) sphinx-quickstart
```

- To build the documentation, from `bash`, move to the `docs` folder and run
```bash
sphinx-build -b html source build
```
this generates the `docs/build` folder. This folder is not kept track of in the
 github repository, as `docs/build` is present in the `.gitignore` file.


The `html` and `latex`  and `pdf` files will be automatically created in the
`docs/build` folder.


- To create the html

make html
make latexpdf


- To add new classes and functions to the API doc, make sure that autodoc
extension is enabled in the `conf.py` file and that these are listed
in the appropriate `.rst` file (such as `index.rst`), e.g.,

```
Factories
---------
.. automodule:: mitiq.factories
   :members:
```

## Additional information
[Here](https://github.com/nathanshammah/scikit-project/blob/master/5-docs.md)
are some notes on how to build docs.
