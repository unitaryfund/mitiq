![Python Build](https://github.com/unitaryfund/mitiq/workflows/Python%20Build/badge.svg?branch=master)

# Mitiq Documentation Contributors Guide
This is the Ccntributors guide for the documentation of Mitiq,
a Python toolkit for implementing error mitigation on quantum computers.

## Requirements
The documentation is generated with
[Sphinx](https://www.sphinx-doc.org/en/master/usage/installation.html).
```bash
pip install -U sphinx recommonmark
```
`recommonmark` allows to include `.md` files, besides `.rst`.

### Check your Sphinx installation
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

### The configuration file
- Since the documentation is already created, you need not to generate a
configuration file from scratch (this is done with `sphinx-quickstart`).
Meta-data, extentions and other custom specifications are accounted for
in the `conf.py` file.

### Build the documentation locally
- To build the documentation, from `bash`, move to the `docs` folder and run
```bash
sphinx-build -b html source build
```
this generates the `docs/build` folder. This folder is not kept track of in the
 github repository, as `docs/build` is present in the `.gitignore` file.
 You need not to modify the `docs/build` folder, as it is automatically
 generated. You will modify only the `docs/source` files.


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

### Add features in the conf.py file

- To add specific feature to the documentation, extensions can be include.
For example to add classes and functions to the API doc, make sure that autodoc
extension is enabled in the `conf.py` file,

```
extensions = ['sphinx.ext.autodoc']
```

### Update the guide with a tree of restructured text files

The documentation is divided into a guide, whose content needs to be written
from scratch, and an API doc part, which can be partly automatically generated.

- To add information in the guide, it is possible to include new information
as a restructured text (`.rst`) or markdown (`.md`) file.

The main file is `index.rst`. It includes a `guide.rst` and an `apidoc.rst`
file, as well as other files. Like in LaTeX, each file can include other files.
Make sure they are included in the table of contents

```
.. toctree::
   :maxdepth: 2
   :caption: Contents:

   changelog.rst
```
### If you want to include in the guide markdown files

- Information to the guide can also be added from markdown (`.md`) files.
This requires `recommonmark` (`pip install --upgrade recommonmark`) and
we added to the `conf.py` file
```
extensions = ['recommonmark']
```

### Automatically add information to the API doc

- New modules, classes and functions can be added by listing them
in the appropriate `.rst` file (such as `autodoc.rst` or a child), e.g.,

```
Factories
---------
.. automodule:: mitiq.factories
   :members:
```
will add all elements of the `mitiq.factories` module. One can hand-pick
classes and functions to add, to comment them, as well as exclude them.

### Save the pdf file in the `docs/pdf` folder

Since the `docs/build` folder is not kept track of, copy the pdf file
with the documentation from `docs/build/latex` to the `docs/pdf` folder,
naming it according to the release version with major and minor.
Make a copy named `Mitiq-latest-release.pdf` in the same folder.


## Additional information
[Here](https://github.com/nathanshammah/scikit-project/blob/master/5-docs.md)
are some notes on how to build docs.

[Here](https://thomas-cokelaer.info/tutorials/sphinx/rest_syntax.html) is a 
cheat sheet for restructed text formatting, e.g. syntax for links etc.
