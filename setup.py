# Copyright (C) 2020 Unitary Fund
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
    extras_require={"development": set(dev_requirements), },
    packages=find_packages(),
    include_package_data=True,
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Unitary Fund",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: MacOS",
        "Operating System :: Unix",
        "Operating System :: Microsoft :: Windows",
        "Topic :: Scientific/Engineering",
    ],
    license="GPL v3.0",
    url="https://unitary.fund",
    python_requires='>=3.7',
)

# restore _version.py to its previous state
with open("mitiq/_version.py", "w") as f:
    f.write(version_file_source)
name='numpy',
        maintainer="NumPy Developers",
        maintainer_email="numpy-discussion@python.org",
        description=DOCLINES[0],
        long_description="\n".join(DOCLINES[2:]),
        url="https://www.numpy.org",
        author="Travis E. Oliphant et al.",
        download_url="https://pypi.python.org/pypi/numpy",
        project_urls={
            "Bug Tracker": "https://github.com/numpy/numpy/issues",
            "Documentation": get_docs_url(),
            "Source Code": "https://github.com/numpy/numpy",
        },
        license='BSD',
        classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
        platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
        test_suite='pytest',
        version=versioneer.get_version(),
        cmdclass=cmdclass,
        python_requires='>=3.7',
        zip_safe=False,
        entry_points={
            'console_scripts': f2py_cmds