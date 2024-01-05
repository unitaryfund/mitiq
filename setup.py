# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup, find_packages

with open("VERSION.txt", "r") as f:
    __version__ = f.read().strip()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open("dev_requirements.txt") as f:
    dev_requirements = f.read().splitlines()

# save the source code in _version.py
with open("mitiq/_version.py", "r") as f:
    version_file_source = f.read()

# overwrite _version.py in the source distribution
with open("mitiq/_version.py", "w") as f:
    f.write(f"__version__ = '{__version__}'\n")

setup(
    name="mitiq",
    version=__version__,
    install_requires=requirements,
    extras_require={
        "development": set(dev_requirements),
    },
    packages=find_packages(),
    include_package_data=True,
    description="Mitiq is an open source toolkit for implementing error "
    "mitigation techniques on most current intermediate-scale quantum "
    "computers.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Unitary Fund",
    author_email="info@unitary.fund",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Compilers",
        "Typing :: Typed",
    ],
    license="GPL v3.0",
    url="https://unitary.fund",
    project_urls={
        "Bug Tracker": "https://github.com/unitaryfund/mitiq/issues/",
        "Documentation": "https://mitiq.readthedocs.io/en/stable/",
        "Source": "https://github.com/unitaryfund/mitiq/",
    },
    python_requires=">=3.9,<3.12",
)

# restore _version.py to its previous state
with open("mitiq/_version.py", "w") as f:
    f.write(version_file_source)
