# Contributing to the Documentation
This is the contributors guide for the documentation of Mitiq,
the Python toolkit for implementing error mitigation on quantum computers.

![Python Build](https://github.com/unitaryfund/mitiq/workflows/Python%20Build/badge.svg?branch=master)
[![PyPI version](https://badge.fury.io/py/mitiq.svg)](https://badge.fury.io/py/mitiq)
[![Unitary Fund](https://img.shields.io/badge/Supported%20By-UNITARY%20FUND-brightgreen.svg?style=for-the-badge)](http://unitary.fund)

## Requirements
Our documentation is generated with
[Sphinx](https://www.sphinx-doc.org/en/master/usage/installation.html).
The necessary packages can be installed, from the root `mitiq` directory
```bash
pip install -e .
pip install -r requirements-dev.txt
```
as they are included in the `requirements-dev.txt` file.
Alternately, you can use the docker image provided in the repo and all requirements for working with the docs are already installed there.

### Sphinx extensions used to build the docs
- [`myst-nb`](https://myst-nb.readthedocs.io/en/latest/) and [`myst-parser`](https://myst-parser.readthedocs.io/en/latest/) allow both markdown and jupyter notebooks to be included and run by the Sphinx build. Also adds support for [MyST markdown](https://myst-parser.readthedocs.io/en/latest/using/syntax.html) spec.
- [`sphinxcontrib-bibtex`](https://sphinxcontrib-bibtex.readthedocs.io/en/latest/) allows to include citations in a `.bib` file.
- [`pybtex`](https://pybtex.org/) allows to customize how citations are rendered, e.g., APS-style.
- [`sphinx-copybutton`](https://sphinx-copybutton.readthedocs.io/en/latest/) allows to easily copy-paste code snippets from examples.
- [`sphinx-autodoc-typehints`](https://pypi.org/project/sphinx-autodoc-typehints/) allows to control how annotations are displayed in the API-doc part of the documentation, integrating with  [`sphinx-autodoc`](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html) and [`sphinx-napoleon`](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/) which handle specific styling requirements for the API documentation comments.

You can check that Sphinx is installed with `sphinx-build --version`.

In addition, there are two requirements, `tensorflow` and `tensorflow-quantum`,
which are used solely in `guide/guide-executors.rst`. They can be installed via:

```bash
pip install -r docs/requirements.txt
```

If they are not installed, the test that uses them will be skipped. We do this because
`tensorflow-quantum` has incompatibility issues -- version `0.4.0` works on `py38` but
not Windows, and version `0.3.1` works on Windows but not `py38`. Therefore, these two
requirements cannot be installed on Windows. See [gh-419](https://github.com/unitaryfund/mitiq/issues/419) for more information.

### The configuration file
Since the documentation is already created, you need not to generate a
configuration file from scratch (this is done with `sphinx-quickstart`).
Meta-data, extensions and other custom specifications are accounted for
in the `conf.py` file.

### Add/change Sphinx features in the conf.py file

To add specific feature to the documentation, Sphinx extensions can be added to the build.
As and example, to add classes and functions to the API doc, make sure that autodoc
extension is enabled in the `conf.py` file, and for tests the `doctest` one,

```python
extensions = ['sphinx.ext.autodoc','sphinx.ext.doctest']
```

## Updating the Documentation

### Updating the guide by adding files and updating the table of contents

You need not to modify the `docs/build` folder, as it is automatically generated. You should only modify the `docs/source` files.

The documentation is divided into:
-  a **guide**, whose content needs to be
written from scratch, 
- **examples** which can be either jupyter notebooks or MyST formatted notebooks, and
- an **API-doc** part, which is (mostly)
automatically generated.

To add information in the guide, it is recommended to add markdown (`.md`) or MyST markdown files (`.myst`) to the `docs/guide/` directory.
Currently, `.rst` is still supported, but the migration plan is to move everything to MyST serialization.
If you want a good intro to MyST and how it compares to `.rst` see [this guide](https://myst-parser.readthedocs.io/en/latest/using/intro.html#intro-writing).

```{admonition} Note
Remember to add any files you add to the `docs/guide/` directory to the guide TOC file {doc}`docs/source/guide/guide.myst` 
```

The main file is `index.myst`. It includes a `guide.myst` and an `apidoc.myst`
file, as well as other files. Like in LaTeX, each file can include other files.
Make sure they are included in the table of contents
````
```{toctree}
---
maxdepth: 2 
caption: Contents
---
readme.myst
```
````

### Including markdown files in the guide 

Information to the guide can be added as markdown (`.md`) files, since
the `myst-parser` extension supports both basic markdown syntax as well as
the extended MyST syntax. 
Just add the `.md` file to repo and the toctree.

To include `.md` files outside of the documentation `source` directory, you can add a stub `*.myst` file to the toctree inside the `docs\source` directory that contains:

````
```{include} ../../file.md
:relative-docs: docs/
:relative-images:
```
````

where `file.md` is the one to be added. For more info on including files external to the docs, see the [MyST docs](https://myst-parser.readthedocs.io/en/latest/using/howto.html#include-a-file-from-outside-the-docs-folder-like-readme-md).

### Automatically add information from the API docs

New modules, classes and functions can be added by listing them
in the appropriate `.md or `*.myst` file (such as `apidoc.myst` or a child), e.g.,

```
## Factories
```{automodule} mitiq.factories
   :members:
```
will add all elements of the `mitiq.factories` module. You can hand-pick
classes and functions to add, to comment them, as well as exclude them.

```{tip}
If you are adding new features to Mitiq, make sure to add API docs in the
source code, and to the API page `apidoc.rst`.
```

### Build the documentation locally
The easiest way to build the docs is to just run `make docs` from the project 
root directory in bash, which by default builds the html docs output.
You can also use from root `make pdf` to generate the PDF version.

If you want to call sphinx directly, you can from bash move to the `docs` 
folder and run

```bash
sphinx-build -b html source build

```
this generates the `docs/build` folder. This folder is not kept track of in the
github repository, as `docs/build` is present in the `.gitignore` file.

```{note}
The `html` and `latex` and `pdf` files will be automatically created in the
`docs/build` folder.
```

## Testing the Documentation 

When writing a new code example in the docs, you can use different directives 
to include code blocks. 

### Just the code, don't evaluate
If you want to include a code snippet that doesn't get run (but has syntax
highlighting), use the `code-block` directive:

````
```{code-block} python

   1+1        # simple example
```
````
### Run the code with doctest
In order to make sure that the block is parsed with `make doctest`, use the
`testcode` directive. This can be used in pair with `testoutput`, if something
is printed, and, eventually `testsetup`, to import modules or set up variables
in an invisible block. An example is:

````
```{testcode} python

   1+1        # simple example
```
````
with no output and

````
```{testcode} python
   print(1+1)        # explicitly print
```

```{testoutput} python
   2        # match the print message
```
````

If you have code blocks you want to run, but not be displayed, use the 
`testsetup` directive:

````
```{testsetup} python
   import numpy as np  # this block is not rendered in the html or pdf
```

```{testcode} python
   np.array(2)
```

```{testoutput} python
   array(2)
```
````
### IPython code blocks
There is also the `doctest` directive, which allows to include interactive
Python blocks. These need to be given this way:

````md
```{doctest} python
   >>> import numpy as np
   >>> print(np.array(2))
   array(2)
```
````
{note}`Notice that no space is left between the last input and the output.`

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

### Running the tests

Mitiq uses the `doctest` [extension](https://www.sphinx-doc.org/en/master/usage/extensions/doctest.html) to run and test code in the docs, which is configured in the `conf.py` file. To execute the tests in bash, run:

```bash
make doctest
```
from the root directory. 

This command tests the code examples in the documentation files, as well as testing the docstrings, since these are imported with the `autodoc` extension.


## Additional information
[Here](https://github.com/nathanshammah/scikit-project/blob/master/5-docs.md)
are some notes on how to build docs.

[Here](https://thomas-cokelaer.info/tutorials/sphinx/rest_syntax.html) is a
cheat sheet for restructed text formatting, e.g. syntax for links etc.
