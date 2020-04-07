from setuptools import setup, find_packages

with open("VERSION.txt", "r") as f:
    __version__ = f.read().strip()

with open('requirements.txt') as f:
    INSTALL_REQUIRES = f.read().splitlines()

with open("development_requirements.txt", "r") as f:
    TEST_REQUIREMENTS = f.read().splitlines()
    DEV_REQUIREMENTS = set(TEST_REQUIREMENTS)

NAME = "mitiq"
AUTHOR = "Unitary Fund"
URL = "https://unitary.fund"
LICENSE = "GPL v3.0"
setup(
    name=NAME,
    version=__version__,
    packages = find_packages(),
    include_package_data=True,
    install_requires=INSTALL_REQUIRES,
    dev_requirements=DEV_REQUIREMENTS,
    tests_require=TEST_REQUIREMENTS,
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
