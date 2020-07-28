# Contributing to the Documentation
This is the Contributors guide for the documentation of Mitiq,
the Python toolkit for implementing error mitigation on quantum computers.

![Python Build](https://github.com/unitaryfund/mitiq/workflows/Python%20Build/badge.svg?branch=master)
[![PyPI version](https://badge.fury.io/py/mitiq.svg)](https://badge.fury.io/py/mitiq)
[![Unitary Fund](https://img.shields.io/badge/Supported%20By-UNITARY%20FUND-brightgreen.svg?style=for-the-badge)](http://unitary.fund)

## Requirements
The documentation is generated with
[Sphinx](https://www.sphinx-doc.org/en/master/usage/installation.html).
The necessary packages can be installed, from the root `mitiq` directory
```bash
pip install -e .
pip install -r requirements.txt
```
as they are present in the `requirements.txt` file. Otherwise, with

```bash
pip install -U sphinx m2r sphinxcontrib-bibtex pybtex sphinx-copybutton sphinx-autodoc-typehints
```

`m2r` allows to include `.md` files, besides `.rst`, in the documentation.
`sphinxcontrib-bibtex` allows to include citations in a `.bib` file and
`pybtex` allows to customize how they are rendered, e.g., APS-style.
`sphinx-copybutton` allows to easily copy-paste code snippets from examples.
`sphinx-autodoc-typehints` allows to control how annotations are displayed in the API-doc part of the documentation, integrating with `sphinx-autodoc` and `sphinx-napoleon`.


You can check that Sphinx is installed with `sphinx-build --version`.

## How to Update the Documentation

### The configuration file
- Since the documentation is already created, you need not to generate a
configuration file from scratch (this is done with `sphinx-quickstart`).
Meta-data, extentions and other custom specifications are accounted for
in the `conf.py` file.

### Add features in the conf.py file

- To add specific feature to the documentation, extensions can be include.
For example to add classes and functions to the API doc, make sure that autodoc
extension is enabled in the `conf.py` file, and for tests the `doctest` one,

```
extensions = ['sphinx.ext.autodoc','sphinx.ext.doctest']
```

### Update the guide with a tree of restructured text files

You need not to modify the `docs/build` folder, as it is automatically
 generated. You will modify only the `docs/source` files.

The documentation is divided into a **guide**, whose content needs to be
written from scratch, and an **API-doc** part, which can be partly
automatically generated.

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
### You can include markdown files in the guide

- Information to the guide can also be added from markdown (`.md`) files, since
 `m2r` (`pip install --upgrade m2r`) is installed and
added to the `conf.py` file (`extensions = ['m2r']`). Just add the `.md` file
to the toctree.

To include `.md` files outside of the documentation `source` directory, you can
 add in `source` an `.rst` file to the toctree that contains inside it the
`.. mdinclude:: ../file.md` command, where `file.md` is the one to be added.


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

## How to Test the Documentation Examples

There are several ways to check that the documentation examples work.
Currently, `mitiq` is testing them with the `doctest`
[extension](https://www.sphinx-doc.org/en/master/usage/extensions/doctest.html)
of `sphinx`. This is set in the `conf.py` file and is executed with

```bash
make doctest
```
from the `mitiq/docs` directory. From the root directory `mitiq`, simply run
```bash
make docs
```
to obtain the same result.

These equivalent commands test the code examples in the guide and ".rst" files, as well as testing the docstrings, since these are imported with the `autodoc` extension.

When writing a new example, you can use different directives in the rst file to
include code blocks. One of them is

```
.. code-block:: python

   1+1        # simple example

```

In order to make sure that the block is parsed with `make doctest`, use the
`testcode` directive. This can be used in pair with `testoutput`, if something
is printed, and, eventually `testsetup`, to import modules or set up variables
in an invisible block. An example is:

```
.. testcode:: python

   1+1        # simple example

```
with no output and

```
.. testcode:: python

   print(1+1)        # explicitly print

.. testoutput:: python

   2        # match the print message


```

The use of `testsetup` allows blocks that do not render:

```
.. testsetup:: python

   import numpy as np  # this block is not rendered in the html or pdf

.. testcode:: python

   np.array(2)

.. testoutput:: python

   array(2)

```

There is also the `doctest` directive, which allows to include interactive
Python blocks. These need to be given this way:

```
.. doctest:: python

   >>> import numpy as np
   >>> print(np.array(2))
   array(2)

Notice that no space is left between the last input and the output.
```
 A way to test docstrings without installing sphinx is with [`pytest` +
 `doctest`](http://doc.pytest.org/en/latest/doctest.html):

```bash
pytest --doctest-glob='*.rst'
```
or alternatively

```bash
pytest --doctest-modules
```

However, this only checks `doctest` blocks, and does not recognize `testcode`
blocks. Moreover, it does not parse the `conf.py` file nor uses sphinx.
A way to include testing of `testcode` and `testoutput` blocks is with the
[`pytest-sphinx`](https://github.com/thisch/pytest-sphinx) plugin. Once
installed,
```bash
pip install pytest-sphinx
```
it will show up as a plugin, just like `pytest-coverage` and others, simply
calling
```bash
pytest --doctest-glob='*.rst'
```
The `pytest-sphinx` plugin does not support `testsetup` directives.

In order to skip a test, if this is problematic, one can use the `SKIP` and
`IGNORE` keywords, adding them as comments next to the relevant line or block:

```
>>> something_that_raises()  # doctest: +IGNORE
```
One can also use various `doctest` [features](http://doc.pytest.org/en/latest/doctest.html#using-doctest-options) by configuring them in the
`docs/pytest.ini` file.


## How to Make a New Release of the Documentation

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

### Create the html and pdf file and save it in the `docs/pdf` folder
- To create the html structure
```bash
make html
```
 and for the pdf,
```bash
make latexpdf
```
Since the `docs/build` folder is not kept track of, copy the pdf file
with the documentation from `docs/build/latex` to the `docs/pdf` folder,
naming it according to the release version with major and minor.
Make a copy named `Mitiq-latest-release.pdf` in the same folder.


## Additional information
[Here](https://github.com/nathanshammah/scikit-project/blob/master/5-docs.md)
are some notes on how to build docs.

[Here](https://thomas-cokelaer.info/tutorials/sphinx/rest_syntax.html) is a
cheat sheet for restructed text formatting, e.g. syntax for links etc.
