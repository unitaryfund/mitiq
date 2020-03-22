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
### Check your installation
To check that Sphinx is installed you can run
```bash
sphinx-build --version
```

## How to Update the Documentation

### Work in an environment
- Create a conda environment for the documentation
```bash
conda create -n mitiqenv
conda activate mitiqenv
```

### Create a new branch
- Create a branch in `git` for the documentation with the release number up to
minor (e.g., 0.0.2--->00X)
```bash
(mitiqenv) git checkout -b mitiq00X
```

### Create a new branch
- Since the documentation is already created, you need not to generate it
from scratch. If you had to generate it from scratch, the first step would
involve creating the `conf.py` file. This can be generated with a wizard
```bash
(mitiqenv) sphinx-quickstart
```

### Build the documentation locally
- To build the documentation, from `bash`, move to the `docs` folder and run
```bash
sphinx-build -b html source build
```
this generates the `docs/build` folder. This folder is not kept track of in the
 github repository, as `docs/build` is present in the `.gitignore` file.


The `html` and `latex`  and `pdf` files will be automatically created in the
`docs/build` folder.


### Create the html
- To create the html structure,

```bash
make html
```

### Create the pdf
- To create the latex files and output a pdf,

```bash
make latexpdf
```

### Add information in the guide

The documentation is divided into a guide, whose content needs to be written
from scratch, and an API doc part, which can be partly automatically generated.

- To add information in the guide, it is possible to include new information
as a restructured text (`.rst`) or markdown (`.md`) file.

The main file is `index.rst`. It includes a `guide.rst` and an `apidoc.rst`
file, as well as other files. Like in LaTeX, each file can include other files.


### Automatically add information to the API doc

- To add new classes and functions to the API doc, make sure that autodoc
extension is enabled in the `conf.py` file and that these are listed
in the appropriate `.rst` file (such as `index.rst`), e.g.,

```
Factories
---------
.. automodule:: mitiq.factories
   :members:
```

### Add modules with API documentation
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
