from setuptools import setup, find_packages

with open("VERSION.txt", "r") as f:
    __version__ = f.read().strip()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# save the source code in _version.py
with open("mitiq/_version.py", "r") as f:
    version_file_source = f.read()

# overwrite _version.py in the source distribution
with open("mitiq/_version.py", "w") as f:
    f.write(f"__version__ = '{__version__}'\n")

setup(
    name='mitiq',
    version=__version__,
    install_requires=[
        # The minimum spec for a working mitiq install
        # note: this should be a subset of requirements.txt
        "numpy~=1.18.1",
        "scipy~=1.4.1",
        # this is the version that has quil_output.py
        # TODO gh-271: later versions break folding
        "cirq-unstable==0.9.0.dev20200508234715",
    ],
    extras_require={
        'development': set(requirements),
        'test': requirements,
    },
    packages=find_packages(),
    include_package_data=True,
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
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
)

# restore _version.py to its previous state
with open("mitiq/_version.py", "w") as f:
    f.write(version_file_source)
