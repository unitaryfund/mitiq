from setuptools import setup, find_packages

with open("VERSION.txt", "r") as f:
    __version__ = f.read().strip()

INSTALL_REQUIRES = [
                    "numpy~=1.18.1",
                    "scipy~=1.4.1",
                    "cirq~=0.7.0"
                    ]
DEV_REQUIREMENTS ={
                    "pyquil~=2.18.0",
                    "qiskit~=0.16.2",
                    "pytest~=5.4.1"
                    }
TEST_REQUIRES = [
                    "pytest~=5.4.1"
                    ]
NAME = "mitiq"
AUTHOR = "Unitary Fund"
URL = "https://github.com/unitaryfund"
LICENSE = "GPL v3.0"
setup(
    name=NAME,
    version=__version__,
    packages = find_packages(),
    include_package_data=True,
    install_requires=INSTALL_REQUIRES,
    dev_requirements=DEV_REQUIREMENTS,
    tests_require=TEST_REQUIRES,
    author=AUTHOR,
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url = URL,
   classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: MacOS",
        "Operating System :: Unix",
        "Operating System :: Microsoft :: Windows",
        "Topic :: Scientific/Engineering",
        ],
    license = LICENSE
)
