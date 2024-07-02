# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

from setuptools import find_packages, setup


def load_requirements(filename):
    with open(f"requirements/{filename}", "r") as file:
        return file.read().splitlines()


with open("INTEGRATIONS.txt", "r") as f:
    integrations = f.read().splitlines()

with open("VERSION.txt", "r") as f:
    __version__ = f.read().strip()

# save the source code in _version.py
with open("mitiq/_version.py", "r") as f:
    version_file_source = f.read()

# overwrite _version.py in the source distribution
with open("mitiq/_version.py", "w") as f:
    f.write(f"__version__ = '{__version__}'\n")

third_party_integration_requirements = {
    k: load_requirements(f"requirements-{k}.txt") for k in integrations
}

setup(
    name="mitiq",
    version=__version__,
    install_requires=load_requirements("requirements.txt"),
    extras_require={
        "development": set().union(
            *third_party_integration_requirements.values(),
            load_requirements("requirements-dev.txt"),
        )
    }
    | third_party_integration_requirements,
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
    python_requires=">=3.10,<3.13",
)

# restore _version.py to its previous state
with open("mitiq/_version.py", "w") as f:
    f.write(version_file_source)
